"""Consensus filter over greedy + sampled extractions (no anchoring).

Unlike scorer.merge_results (priority file anchored, additions only), this
variant lets the vote REMOVE issues: a cluster must gather >= min_votes files
to survive. That trades a little recall for precision — the val attribution
shows pred_extra (~380) is the biggest loss bucket, so pruning unstable
issues is the point. Stances/evidence/futures still prefer the earliest
(highest-priority) file inside each winning cluster.

    python3 -m scorer.consensus_filter out.jsonl greedy.jsonl s1.jsonl s2.jsonl --min-votes 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scorer.compare_results import _bigrams, _set_cosine

_NAME_SIM = 0.6


def _load(path: str) -> dict:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            rows[obj["sample_id"]] = obj
    return rows


def _similar(a: str, b: str) -> bool:
    return _set_cosine(_bigrams(a), _bigrams(b)) >= _NAME_SIM


def fuse_sample(files: list[dict]) -> dict:
    """files: per-file row dicts for ONE sample, priority order."""
    clusters: list[list[tuple[int, dict]]] = []
    for fi, row in enumerate(files):
        for issue in row.get("issue_list") or []:
            placed = False
            for cl in clusters:
                members = [ij for f, ij in cl]
                if any(_similar(str(issue.get("issue_name")), str(m.get("issue_name"))) for m in members):
                    if not any(f == fi for f, _ in cl):
                        cl.append((fi, issue))
                        placed = True
                    else:
                        # same file already in cluster: keep the first (priority)
                        pass
                    break
            if not placed and not any(
                _similar(str(issue.get("issue_name")), str(m.get("issue_name")))
                for cl in clusters for _, m in cl
            ):
                clusters.append([(fi, issue)])

    issues, futures = [], []
    for cl in clusters:
        votes = len({f for f, _ in cl})
        if votes < ARGS.min_votes:
            continue
        head = cl[0][1]
        stance_votes: list[str] = [ij.get("stance") for _, ij in cl if ij.get("stance")]
        stance = stance_votes[0]
        best = stance_votes.count(stance)
        for v in stance_votes[1:]:
            if stance_votes.count(v) > best:
                stance, best = v, stance_votes.count(v)
        chain: list[str] = []
        for _, ij in cl:
            for ev in ij.get("argument_chain") or []:
                if not any(_similar(ev, kept) or ev == kept for kept in chain):
                    chain.append(ev)
        issues.append({
            "issue_name": head["issue_name"],
            "stance": stance,
            "argument_chain": chain[:3],
        })
        frow = files[cl[0][0]]
        idx = (frow.get("issue_list") or []).index(head)
        futs = frow.get("future_argument") or []
        futures.append(futs[idx] if idx < len(futs) else "")
    return {"sample_id": files[0]["sample_id"], "issue_list": issues, "future_argument": futures}


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--max-evidence", type=int, default=3)
    ARGS = p.parse_args()

    loaded = [_load(f) for f in ARGS.inputs]
    ids = list(loaded[0].keys())
    n_empty = 0
    with Path(ARGS.out).open("w", encoding="utf-8", newline="\n") as fh:
        for sid in ids:
            files = [rows[sid] for rows in loaded if sid in rows]
            if not any(f.get("issue_list") for f in files):
                fused = files[0]
                n_empty += 1
            else:
                fused = fuse_sample(files)
            fh.write(json.dumps(fused, ensure_ascii=False) + "\n")
    print(f"consensus-fused {len(ids)} rows from {len(loaded)} files (min_votes={ARGS.min_votes}), empty={n_empty}")
    print(f"next: python3 -m scorer.check_submit {ARGS.out} --split val")


if __name__ == "__main__":
    main()
