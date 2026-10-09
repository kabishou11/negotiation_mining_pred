"""MBR (minimum-Bayes-risk) variant selection, per sample, no gold needed.

Voting-that-filters (consensus_filter) broke recall by pruning issues.
MBR instead picks, per sample, the WHOLE variant whose issue set is most
similar to every other variant's set (the medoid). It keeps each variant's
internal consistency, changes nothing when variants agree, and breaks ties
toward the priority (first) file.

    python3 -m scorer.mbr_select out.jsonl greedy.jsonl s1.jsonl s2.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scorer.compare_results import _bigrams, _set_cosine


def _load(path: str) -> dict:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            rows[obj["sample_id"]] = obj
    return rows


def _pair_score(a: dict, b: dict) -> float:
    """Issue-set agreement between two result rows: matched name similarity
    + stance agreement bonus; normalised by the larger set size."""
    ia, ib = a.get("issue_list") or [], b.get("issue_list") or []
    if not ia or not ib:
        return 0.0
    total, matched = 0.0, 0
    used = set()
    for i in ia:
        ga = _bigrams(str(i.get("issue_name") or ""))
        best, bj = 0.0, -1
        for j, o in enumerate(ib):
            if j in used:
                continue
            sim = _set_cosine(ga, _bigrams(str(o.get("issue_name") or "")))
            if sim > best:
                best, bj = sim, j
        if bj >= 0 and best >= 0.6:
            used.add(bj)
            matched += 1
            total += best + (0.25 if i.get("stance") == ib[bj].get("stance") else 0.0)
    return total / max(len(ia), len(ib))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("inputs", nargs="+", help="variants, priority file first")
    args = p.parse_args()

    loaded = [_load(f) for f in args.inputs]
    ids = [i for i in loaded[0] if all(i in rows for rows in loaded)]
    picks = [0] * len(loaded)
    with Path(args.out).open("w", encoding="utf-8", newline="\n") as fh:
        for sid in ids:
            rows = [r[sid] for r in loaded]
            scores = [sum(_pair_score(rows[k], rows[j]) for j in range(len(rows)) if j != k)
                      for k in range(len(rows))]
            best = max(range(len(rows)), key=lambda k: (scores[k], -k))
            picks[best] += 1
            fh.write(json.dumps(rows[best], ensure_ascii=False) + "\n")
    print(f"MBR picked per-sample medoid over {len(ids)} samples: {picks} (priority-first tie-break)")
    print(f"next: python3 -m scorer.evaluate {args.out} --split val")


if __name__ == "__main__":
    main()
