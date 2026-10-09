"""RAFT winner mining: best-of-N extractions scored by the local judge.

For every train document, score each sampled extraction (from
scorer.infer_vllm --raft-n) against gold with the same evaluator stack the
leaderboard replica uses (bge-small-zh triple matching + bert-base-chinese
alpha), and keep the highest-scoring variant per document as the new SFT
extract target. Training on winners the model can actually reach (rather
than gold it fails to fit) is the classic RAFT remedy for exposure bias,
and the selection metric is the judge itself.

  python3 -m scorer.raft_winners raft_samples.jsonl --out raft_targets.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scorer.datautil import load_split
from scorer.encoders import BgeEncoder, BertScore
from scorer.evaluate import _memoize_pair
from scorer.score import score_sample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("samples", help="raft_samples.jsonl: one row per (sample, variant)")
    p.add_argument("--out", required=True, help="winner rows per doc")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    enc = BgeEncoder(device=args.device)
    alpha_fn = _memoize_pair(BertScore(device=args.device) if "device" in BertScore.__init__.__code__.co_varnames else BertScore())
    gold = {s["sample_id"]: s for s in load_split("train")}

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for line in Path(args.samples).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_doc[row["sample_id"]].append(row)

    kept, improved, ties = 0, 0, 0
    with Path(args.out).open("w", encoding="utf-8", newline="\n") as fh:
        for sid, rows in by_doc.items():
            g = gold.get(sid)
            if g is None:
                continue
            best, best_s, greedy_s = None, -1.0, None
            for r in rows:
                try:
                    sc = score_sample(
                        {"sample_id": sid,
                         "issue_list": r["issue_list"],
                         "future_argument": r.get("future_argument") or []},
                        g,
                        enc,
                        alpha_fn,
                    )
                except Exception:
                    continue
                val = getattr(sc, "score", None) or (sc.get("score") if isinstance(sc, dict) else None)
                if val is None:
                    continue
                if r.get("greedy"):
                    greedy_s = val
                if val > best_s:
                    best_s, best = val, r
            if best is None:
                continue
            kept += 1
            if greedy_s is not None and best_s > greedy_s + 1e-9:
                improved += 1
            fh.write(json.dumps({
                "sample_id": sid,
                "score": best_s,
                "greedy_score": greedy_s,
                "issue_list": best["issue_list"],
                "future_argument": best.get("future_argument") or [],
            }, ensure_ascii=False) + "\n")
    print(f"winners kept {kept}/{len(by_doc)} docs; improved-over-greedy {improved}")
    print(f"next: rebuild SFT with extract targets from {args.out}")


if __name__ == "__main__":
    main()
