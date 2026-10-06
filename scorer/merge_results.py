"""Fuse several team-A result files produced from the same base model.

The rules explicitly allow fusing different checkpoints / inference results
of the one allowed base. Three such files already exist (checkpoint-1800,
-2000, -2024), so this is the only new-score lever that costs zero GPU:
fuse, check, and let one submission slot measure it.

Per sample, issues are clustered across files by issue-name bigram
similarity (>= 0.6, one issue per file per cluster). Clusters need
``--min-votes`` files to survive, which drops single-file hallucinations;
stances vote, ties resolve to the highest-priority file; evidence chains
are unioned by frequency (ties by first appearance) and capped; the future
text comes from the highest-priority file in the cluster — so pass the
best checkpoint first.

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
        if votes < min_votes:
            continue
        first_idx, first_issue = cluster[0]
        name = _majority([str(i.get("issue_name") or "") for _f, i in cluster])
        stance = _majority([str(i.get("stance") or "") for _f, i in cluster])
        evidence_count: dict[str, int] = {}
        evidence_order: dict[str, int] = {}
        for _f, issue in cluster:
            for ev in issue.get("argument_chain") or []:
                evidence_count[ev] = evidence_count.get(ev, 0) + 1
                evidence_order.setdefault(ev, len(evidence_order))
        chain = [
            ev
            for ev, _ in sorted(evidence_count.items(), key=lambda kv: (-kv[1], evidence_order[kv[0]]))
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
        fused = fuse_sample(present, args.min_votes, args.max_issues, args.max_evidence)
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
