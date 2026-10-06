"""Rewrite an existing result.jsonl without rerunning the model.

Two independent evidence-side operations, applied in a fixed order:

- ``--rerank`` re-selects each issue's evidence sentences from the model's
  own sentences plus ``--pool-size`` neighbours on each side, scoring bge
  cosine between the issue name and each candidate sentence and keeping the
  model's evidence count. Same encoder family as the judge, so selection
  aligns with the metric. Falls back loudly to bigrams when bge is
  unavailable — do not submit a hash-reranked file.
- ``--trim-evidence`` cuts each whole-sentence evidence down to the clause
  window near the gold evidence length (~55 chars), staying an exact source
  substring (see `compile.trim_evidence_text`).

Rerank runs first: it recovers sentence ids by exact text match, which a
prior trim would break. A pre-flight check requires nearly every evidence
string to be a substring of the selected split's documents — sample ids are
renumbered from SAMPLE_0001 in every split, so scoring a test result with
``--split val`` would silently no-op the rerank and mis-apply the trim.

    python3 -m scorer.postprocess result_test_2024.jsonl --split test \
        --out result_pp.jsonl --rerank --trim-evidence
    python3 -m scorer.check_submit result_pp.jsonl --split test

Score the variants on val first (repro/val.sh writes them), then spend a
submission slot only on the winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scorer.compile import trim_chain
from scorer.datautil import load_split
from scorer.segment import segment_sample


def recover_ids(chain: list[str], segmented) -> list[str]:
    """Sentence ids behind an argument_chain, by exact text then containment."""
    ids: list[str] = []
    for text in chain:
        hit = None
        for sent in segmented.sentences:
            if sent.text == text:
                hit = sent.sid
                break
        if hit is None:
            for sent in segmented.sentences:
                if text and text in sent.text:
                    hit = sent.sid
                    break
        if hit and hit not in ids:
            ids.append(hit)
    return ids


def _neighbourhood(ids: list[str], segmented, radius: int) -> list[str]:
    pool: list[str] = []
    for sid in ids:
        num = int(sid[1:])
        for cand in range(num - radius, num + radius + 1):
            if cand < 1:
                continue
            neighbour = f"S{cand:02d}"
            if neighbour in segmented.by_id and neighbour not in pool:
                pool.append(neighbour)
    return pool


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mmr_select(name_sims: list[float], pair_sims: list[list[float]], keep: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance over the candidate pool.

    Gold chains cover DIFFERENT aspects of the issue, so picking the top-k
    sentences by name cosine alone tends to return k near-duplicates, which
    dilutes the argument cosine in both evidence modes. MMR trades a bit of
    relevance for redundancy: score = sim(name, s) - lam * max sim(s, picked).
    lam=0 degrades to plain top-k. Ties resolve to the lowest index.
    """
    n = len(name_sims)
    if lam <= 0 or keep <= 1:
        return sorted(range(n), key=lambda k: (-name_sims[k], k))[:keep]
    picked: list[int] = []
    remaining = set(range(n))
    while len(picked) < min(keep, n):
        best_k = None
        best = None
        for k in sorted(remaining):
            redundancy = max((pair_sims[k][j] for j in picked), default=0.0)
            score = name_sims[k] - lam * redundancy
            if best is None or score > best:
                best_k, best = k, score
        picked.append(best_k)
        remaining.remove(best_k)
    return picked


def rerank_issue(name: str, ids: list[str], segmented, encoder, keep: int, radius: int = 1, mmr: float = 0.0) -> list[str]:
    """Select `keep` candidate sentences by cosine to the issue name, doc order.

    radius=1 scores the model's sentences plus one neighbour each;
    radius=0 widens the pool to every sentence in the sample. mmr>0 enables
    redundancy-aware selection (see `_mmr_select`).
    """
    if radius <= 0:
        pool = [sent.sid for sent in segmented.sentences]
    else:
        pool = _neighbourhood(ids, segmented, radius)
    if not pool:
        return []
    vectors = encoder.encode([name] + [segmented.by_id[sid].text for sid in pool])
    name_vec = vectors[0]
    name_sims = [_dot(name_vec, v) for v in vectors[1:]]
    pair_sims = [[_dot(a, b) for b in vectors[1:]] for a in vectors[1:]]
    picked = _mmr_select(name_sims, pair_sims, keep, mmr)
    chosen = sorted((pool[k] for k in picked), key=lambda sid: int(sid[1:]))
    return [segmented.by_id[sid].text for sid in chosen]


def semantic_trim_chain(name: str, chain: list[str], encoder, target: int = 55, max_len: int = 64) -> list[str]:
    """Encoder-guided trimming: pick each evidence's clause window by the
    judge's own encoder's cosine to the issue name, instead of the lexical
    bigram heuristic in `compile.trim_evidence_text`. Near-duplicate windows
    still collapse so the joined chain does not repeat itself."""
    from scorer.compile import _clause_spans, _near_dup

    candidate_sets: list[list[str]] = []
    for evidence in chain:
        evidence = evidence.strip()
        if len(evidence) <= max_len:
            candidate_sets.append([evidence])
            continue
        spans = _clause_spans(evidence)
        if len(spans) <= 1:
            candidate_sets.append([evidence])
            continue
        windows: list[str] = []
        for a in range(len(spans)):
            for b in range(a, len(spans)):
                start, end = spans[a][0], spans[b][1]
                if end - start > max_len:
                    break
                piece = evidence[start:end].strip(" ，。；、,;:： ")
                if piece:
                    windows.append(piece)
        candidate_sets.append(windows or [evidence])

    flat = [text for group in candidate_sets for text in group]
    vectors = encoder.encode([name] + flat)
    name_vec = vectors[0]
    trimmed: list[str] = []
    cursor = 0
    for group in candidate_sets:
        k = len(group)
        sims = [_dot(name_vec, vectors[1 + cursor + j]) for j in range(k)]
        best = group[max(range(k), key=lambda j: (sims[j], -j))]
        if best and not any(_near_dup(best, kept) for kept in trimmed):
            trimmed.append(best)
        cursor += k
    return trimmed or list(chain)


def load_result(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid json ({exc.msg})") from exc
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def substring_ratio(rows: list[dict], docs_by_id: dict[str, list[str]]) -> tuple[float, int, int]:
    """Fraction of evidence strings that are substrings of the split's docs."""
    hits = total = 0
    for row in rows:
        docs = docs_by_id.get(row["sample_id"])
        if not docs:
            continue
        for issue in row.get("issue_list") or []:
            for evidence in issue.get("argument_chain") or []:
                total += 1
                hits += bool(any(evidence in doc for doc in docs))
    return (hits / total if total else 1.0), hits, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="a result.jsonl written by scorer.infer")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trim-evidence", action="store_true")
    parser.add_argument(
        "--semantic-trim",
        action="store_true",
        help="trim with the bge encoder picking the clause window (overrides "
        "--trim-evidence's lexical heuristic; requires a real encoder, falls "
        "back loudly on the hash smoke encoder)",
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="rerank candidate neighbourhood radius around the model's sentences; 0 = the whole sample",
    )
    parser.add_argument(
        "--mmr",
        type=float,
        default=0.0,
        help="MMR redundancy penalty for evidence selection (0 = plain top-k; try 0.5)",
    )
    parser.add_argument("--encoder", choices=["auto", "bge", "hash"], default="auto")
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if not args.trim_evidence and not args.rerank and not args.semantic_trim:
        raise SystemExit("nothing to do: pass --trim-evidence / --semantic-trim and/or --rerank")
    if args.rerank and args.encoder == "hash":
        print("[warn] hash rerank is a smoke test only, never submit it.")

    rows = load_result(Path(args.path))
    wanted = {row["sample_id"] for row in rows}
    split_rows = [s for s in load_split(args.split) if s["sample_id"] in wanted]
    docs_by_id = {
        s["sample_id"]: [d.get("full_text") or "" for d in s.get("docs") or []] for s in split_rows
    }
    segments = {s["sample_id"]: segment_sample(s) for s in split_rows}
    unknown = wanted - set(segments)
    if unknown:
        print(f"[warn] {len(unknown)} sample_ids not in split {args.split}: {sorted(unknown)[:5]}")

    pre_ratio, pre_hits, pre_total = substring_ratio(rows, docs_by_id)
    if pre_total and pre_ratio < 0.9:
        raise SystemExit(
            f"only {pre_hits}/{pre_total} evidences are substrings of --split {args.split} "
            f"({pre_ratio:.0%}). Sample ids are renumbered per split — this result file "
            "very likely belongs to a different split. Refusing to rewrite it."
        )
    if pre_ratio < 1.0:
        print(f"[warn] {pre_total - pre_hits} evidences are not substrings of split {args.split} docs.")

    encoder = None
    use_semantic = False
    if args.rerank or args.semantic_trim:
        from scorer.evaluate import pick_encoder

        encoder, enc_label = pick_encoder(args.encoder, args.device or None)
    if args.rerank:
        print(f"[rerank] encoder={enc_label} pool_size={args.pool_size} mmr={args.mmr}")
    if args.semantic_trim:
        # semantic window selection is meaningless on the bigram smoke encoder
        use_semantic = "bge" in enc_label
        if use_semantic:
            print(f"[trim] semantic window selection with {enc_label}")
        else:
            print("[warn] semantic trim needs a real encoder; falling back to the lexical heuristic.")

    trimmed_count = reranked_count = missing = 0
    for row in rows:
        sample_id = row["sample_id"]
        segmented = segments.get(sample_id)
        if segmented is None:
            missing += 1
            continue
        if args.rerank:
            for issue in row.get("issue_list") or []:
                chain = list(issue.get("argument_chain") or [])
                ids = recover_ids(chain, segmented)
                if not ids or len(ids) != len(chain):
                    continue
                reranked = rerank_issue(
                    str(issue.get("issue_name") or ""),
                    ids,
                    segmented,
                    encoder,
                    len(ids),
                    args.pool_size,
                    args.mmr,
                )
                if reranked and reranked != chain:
                    reranked_count += 1
                    issue["argument_chain"] = reranked
        if use_semantic:
            for issue in row.get("issue_list") or []:
                chain = list(issue.get("argument_chain") or [])
                new_chain = semantic_trim_chain(str(issue.get("issue_name") or ""), chain, encoder)
                if new_chain != chain:
                    trimmed_count += sum(1 for a, b in zip(chain, new_chain) if a != b)
                    issue["argument_chain"] = new_chain
        elif args.trim_evidence:
            for issue in row.get("issue_list") or []:
                chain = list(issue.get("argument_chain") or [])
                new_chain = trim_chain(str(issue.get("issue_name") or ""), chain)
                if new_chain != chain:
                    trimmed_count += sum(1 for a, b in zip(chain, new_chain) if a != b)
                    issue["argument_chain"] = new_chain

    post_ratio, post_hits, post_total = substring_ratio(rows, docs_by_id)
    if post_total and post_ratio < pre_ratio:
        print(
            f"[error] substring ratio dropped {pre_ratio:.4f} -> {post_ratio:.4f}; "
            "an operation produced out-of-source evidence. Do not submit this file."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(rows)} rows to {out_path} reranked_issues={reranked_count} "
        f"trimmed_evidences={trimmed_count} unknown_ids={missing} "
        f"substring_ratio={post_hits}/{post_total}"
    )
    print("next: python3 -m scorer.check_submit", out_path, "--split", args.split)


if __name__ == "__main__":
    main()
