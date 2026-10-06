"""Rewrite an existing result.jsonl without rerunning the model.

Two independent evidence-side operations, applied in a fixed order:

- ``--rerank`` re-selects each issue's evidence sentences from the model's
  own sentences plus their immediate neighbours, scoring bge cosine between
  the issue name and each candidate sentence, keeping the model's evidence
  count. Same encoder family as the judge, so selection aligns with the
  metric. Falls back loudly to bigrams when bge is unavailable — do not
  submit a hash-reranked file.
- ``--trim-evidence`` cuts each whole-sentence evidence down to the clause
  window near the gold evidence length (~55 chars), staying an exact source
  substring (see `compile.trim_evidence_text`).

Both operations only narrow or reorder evidence inside what the segmenter
already guarantees is source text, so `check_submit` still passes. Rerank
runs first because it recovers sentence ids by exact text match, which a
prior trim would break.

    python3 -m scorer.postprocess result_test_2024.jsonl --split test \
        --out result_pp.jsonl --rerank --trim-evidence
    python3 -m scorer.check_submit result_pp.jsonl --split test

Score the variants on val first: `repro/val.sh` writes the result, then run
this tool on it and `repro/eval.sh` both files before touching a slot.
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


def _neighbourhood(ids: list[str], segmented) -> list[str]:
    pool: list[str] = []
    for sid in ids:
        num = int(sid[1:])
        for cand in (num - 1, num, num + 1):
            neighbour = f"S{cand:02d}"
            if neighbour in segmented.by_id and neighbour not in pool:
                pool.append(neighbour)
    return pool


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rerank_issue(name: str, ids: list[str], segmented, encoder, keep: int) -> list[str]:
    """Top-`keep` candidate sentences by cosine to the issue name, doc order."""
    pool = _neighbourhood(ids, segmented)
    if not pool:
        return []
    vectors = encoder.encode([name] + [segmented.by_id[sid].text for sid in pool])
    name_vec = vectors[0]
    scored = sorted(range(len(pool)), key=lambda k: (-_dot(name_vec, vectors[k + 1]), pool[k]))
    chosen = sorted((pool[k] for k in scored[:keep]), key=lambda sid: int(sid[1:]))
    return [segmented.by_id[sid].text for sid in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="a result.jsonl written by scorer.infer")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trim-evidence", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--encoder", choices=["auto", "bge", "hash"], default="auto")
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if not args.trim_evidence and not args.rerank:
        raise SystemExit("nothing to do: pass --trim-evidence and/or --rerank")
    if args.rerank and args.encoder == "hash":
        print("[warn] hash rerank is a smoke test only, never submit it.")

    encoder = None
    if args.rerank:
        from scorer.evaluate import pick_encoder

        encoder, enc_label = pick_encoder(args.encoder, args.device or None)
        print(f"[rerank] encoder={enc_label}")

    segments = {sample["sample_id"]: segment_sample(sample) for sample in load_split(args.split)}
    rows: list[dict] = []
    with Path(args.path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.path}:{line_no}: invalid json ({exc.msg})") from exc

    trimmed_count = reranked_count = missing = 0
    for row in rows:
        sample_id = row["sample_id"]
        segmented = segments.get(sample_id)
        if segmented is None:
            missing += 1
            continue
        futures = list(row.get("future_argument") or [])
        if args.rerank:
            for issue in row.get("issue_list") or []:
                chain = list(issue.get("argument_chain") or [])
                ids = recover_ids(chain, segmented)
                if not ids or len(ids) != len(chain):
                    continue
                reranked = rerank_issue(str(issue.get("issue_name") or ""), ids, segmented, encoder, len(ids))
                if reranked and reranked != chain:
                    reranked_count += 1
                    issue["argument_chain"] = reranked
        if args.trim_evidence:
            for issue in row.get("issue_list") or []:
                chain = list(issue.get("argument_chain") or [])
                new_chain = trim_chain(str(issue.get("issue_name") or ""), chain)
                if new_chain != chain:
                    trimmed_count += sum(1 for a, b in zip(chain, new_chain) if a != b)
                    issue["argument_chain"] = new_chain
        row["future_argument"] = futures[: len(row.get("issue_list") or [])]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(rows)} rows to {out_path} reranked_issues={reranked_count} "
        f"trimmed_evidences={trimmed_count} unknown_ids={missing}"
    )
    print("next: python3 -m scorer.check_submit", out_path, "--split", args.split)


if __name__ == "__main__":
    main()
