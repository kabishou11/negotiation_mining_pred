"""Fuse several team-A result files produced from the same base model.

The rules explicitly allow fusing different checkpoints / inference results
of the one allowed base. Three such files already exist (checkpoint-1800,
-2000, -2024), so this is the only new-score lever that costs zero GPU:
fuse, check, and let one submission slot measure it.

    Per sample, issues are clustered across files by issue-name bigram
    similarity (>= 0.6, one issue per file per cluster). The PRIORITY file
    (pass it first — the best checkpoint) is ANCHORED: every issue it
    asserts is kept, so the fused file is never weaker in coverage than the
    best single file (two checkpoints can name the same gold issue so
    differently that no cluster reaches the vote threshold — measured on
    real runs). A cluster asserted only by lower-priority files needs
    ``--min-votes`` of them, which drops single-file hallucinations while
    still adding issues the priority file missed. Stances vote, ties resolve
    to the highest-priority file; evidence chains are unioned by frequency
    with near-duplicates collapsed; the future text comes from the
    highest-priority file in the cluster.

    python3 -m scorer.merge_results result_fused.jsonl r_2024.jsonl r_2000.jsonl r_1800.jsonl
    python3 -m scorer.check_submit result_fused.jsonl --split test

    Validate on val first: infer val with each checkpoint, fuse, and
    `repro/eval.sh` the fused file against the singles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scorer.compare_results import _bigrams, _set_cosine
from scorer.compile import _fallback_future

_NAME_SIM = 0.6


def _majority(values: list[str]) -> str:
    """Most frequent value; ties go to the earliest (highest-priority) file.

    Deliberately not max(set(...)): set iteration order follows string
    hashing, which would make ties depend on PYTHONHASHSEED.
    """
    best = values[0]
    best_count = values.count(best)
    for value in values[1:]:
        count = values.count(value)
        if count > best_count:
            best, best_count = value, count
    return best


def _cluster(files_issues: list[list[dict]]) -> list[list[tuple[int, dict]]]:
    """Greedy name clustering; a file contributes at most one issue each."""
    clusters: list[list[tuple[int, dict]]] = []
    for file_idx, issues in enumerate(files_issues):
        for issue in issues:
            grams = _bigrams(str(issue.get("issue_name") or ""))
            best: list[tuple[int, dict]] | None = None
            best_sim = 0.0
            for cluster in clusters:
                if any(f == file_idx for f, _ in cluster):
                    continue
                sim = _set_cosine(grams, _bigrams(str(cluster[0][1].get("issue_name") or "")))
                if sim > best_sim:
                    best, best_sim = cluster, sim
            if best is not None and best_sim >= _NAME_SIM:
                best.append((file_idx, issue))
            else:
                clusters.append([(file_idx, issue)])
    return clusters


def fuse_sample(files_rows: list[dict], min_votes: int, max_issues: int, max_evidence: int) -> dict:
    """Fuse one sample's rows from files given in priority order."""
    files_issues = [list(row.get("issue_list") or []) for row in files_rows]
    futures_by_file = [list(row.get("future_argument") or []) for row in files_rows]
    clusters = _cluster(files_issues)
    issues: list[dict] = []
    futures: list[str] = []
    for cluster in clusters:
        votes = len({f for f, _ in cluster})
        # Priority anchoring: an issue the best file asserts is always kept,
        # so fusion never loses coverage against the priority file alone.
        # Only non-priority single-vote clusters are filtered.
        anchored = any(f == 0 for f, _ in cluster)
        if votes < min_votes and not anchored:
            continue
        first_idx, first_issue = cluster[0]
        name = _majority([str(i.get("issue_name") or "") for _f, i in cluster])
        stance = _majority([str(i.get("stance") or "") for _f, i in cluster])
        # Union the chains, collapsing near-duplicates (the same sentence
        # with a different trailing punctuation recurs across checkpoints)
        # so redundant variants do not burn evidence slots.
        kept: list[list] = []  # [text, votes, first_order]
        seen_order = 0
        for _f, issue in cluster:
            for ev in issue.get("argument_chain") or []:
                seen_order += 1
                grams = _bigrams(ev)
                hit = next(
                    (entry for entry in kept if _set_cosine(grams, _bigrams(entry[0])) >= 0.8),
                    None,
                )
                if hit is None:
                    kept.append([ev, 1, seen_order])
                else:
                    hit[1] += 1
        chain = [
            entry[0]
            for entry in sorted(kept, key=lambda entry: (-entry[1], entry[2]))
        ][:max_evidence]
        if not name or not chain:
            continue
        # the cluster's first member is the highest-priority file's issue
        idx_in_file = None
        for pos, issue in enumerate(files_issues[first_idx]):
            if issue is first_issue:
                idx_in_file = pos
                break
        first_future = ""
        if idx_in_file is not None and idx_in_file < len(futures_by_file[first_idx]):
            first_future = str(futures_by_file[first_idx][idx_in_file])
        issues.append({"issue_name": name, "stance": stance, "argument_chain": chain})
        futures.append(first_future or _fallback_future({"argument_chain": [chain[0]]}))
        if len(issues) >= max_issues:
            break
    return {"issue_list": issues, "future_argument": futures}


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid json ({exc.msg})") from exc
            rows[row["sample_id"]] = row
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("inputs", nargs="+", help="result files, best checkpoint FIRST")
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--max-issues", type=int, default=6)
    parser.add_argument("--max-evidence", type=int, default=3)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("fusion needs at least two input files")
    files_rows = [load_rows(Path(p)) for p in args.inputs]
    reference = files_rows[0]
    out_rows: list[dict] = []
    for sample_id, row in reference.items():
        present = [files[sample_id] for files in files_rows if sample_id in files]
        # A sample missing from a lower-priority file must not be dropped:
        # clamp the vote requirement to what is actually present, or the
        # fused submission would fail check_submit on missing ids.
        required = min(args.min_votes, len(present))
        fused = fuse_sample(present, required, args.max_issues, args.max_evidence)
        if not fused["issue_list"]:
            continue
        out_rows.append({"sample_id": sample_id, **fused})
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"fused {len(out_rows)} rows from {len(files_rows)} files "
        f"(min_votes={args.min_votes}) into {out_path}"
    )
    print("next: check_submit, and on val also repro/eval.sh against the singles")


if __name__ == "__main__":
    main()
