"""Leaderboard-style local evaluation of a result.jsonl.

This closes the two offline approximations named in `score.py`: the bigram
stand-in α and the ROUGE-L future stand-in. `score.py` keeps its offline
defaults so `selfcheck` still runs on numpy+scipy alone; the real encoders
are wired in here.

On the server (torch + transformers + bert-score installed, and
`BAAI/bge-small-zh-v1.5` / `bert-base-chinese` cached for the offline run):

    python3 -m scorer.evaluate result_val.jsonl --split val
    python3 -m scorer.evaluate result_train.jsonl --split train --ids-file data/dev40_ids.txt

Without torch or bert-score it falls back to the offline encoders and says so
loudly: those numbers are for smoke tests, not leaderboard estimates. Both
argument-chain modes (newline-joined vs per-evidence mean) are printed from
one run, so the pending newline-vs-mean decision needs a single command.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scorer.attribute import attribute_sample
from scorer.datautil import load_split
from scorer.encoders import (
    BgeEncoder,
    BertScore,
    BertScoreText,
    HashingEncoder,
    OrthogonalEncoder,
    get_bert_scorer,
)
from scorer.score import score_dataset, semantic_alpha


def load_result(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid json ({exc.msg})") from exc
            if not isinstance(row.get("sample_id"), str):
                raise SystemExit(f"{path}:{line_no}: no string sample_id")
            rows.append(row)
    return rows


def pick_encoder(name: str, device: str | None):
    if name in ("auto", "bge"):
        try:
            return BgeEncoder(device=device), "BAAI/bge-small-zh-v1.5"
        except Exception as exc:
            if name == "bge":
                raise
            print(f"[warn] bge encoder unavailable ({type(exc).__name__}: {exc}).")
            print("[warn] falling back to character bigrams: numbers are NOT leaderboard-like.")
    if name == "orthogonal":
        return OrthogonalEncoder(), "orthogonal exact-match"
    return HashingEncoder(), "char-bigram hash (smoke test only)"


def _memoize_pair(fn):
    """Cache a per-pair scorer on the pair's text.

    The same (pred, gold) pair recurs across samples and across the two
    arg-mode passes, and every hit skips a bert-base-chinese forward pass.
    """
    memo: dict = {}

    def wrap(left, right):
        if isinstance(left, str):
            key = (left, right)
        else:
            key = (
                json.dumps(left, ensure_ascii=False, sort_keys=True),
                json.dumps(right, ensure_ascii=False, sort_keys=True),
            )
        if key not in memo:
            memo[key] = fn(left, right)
        return memo[key]

    return wrap


def pick_semantic(name: str):
    """(alpha_fn, future_sim_fn, label), real bert-score when it is installed."""
    if name in ("auto", "bertscore"):
        try:
            get_bert_scorer()  # fail fast, before the scoring loop
            return _memoize_pair(BertScore()), _memoize_pair(BertScoreText()), "bert-base-chinese"
        except Exception as exc:
            if name == "bertscore":
                raise
            print(f"[warn] bert-score unavailable ({type(exc).__name__}: {exc}).")
            print("[warn] using the offline stub: S_semantic will equal F1_pred.")
    return semantic_alpha, None, "offline stub (rouge-l)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="result.jsonl written by scorer.infer")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--ids-file", default="", help="score only these sample_ids, e.g. data/dev40_ids.txt")
    parser.add_argument("--limit", type=int, default=0, help="0 means every selected sample")
    parser.add_argument("--encoder", choices=["auto", "bge", "hash", "orthogonal"], default="auto")
    parser.add_argument("--semantic", choices=["auto", "bertscore", "stub"], default="auto")
    parser.add_argument("--device", default="", help="cuda/cpu for the encoders (default: auto)")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("test has no labels. Use --split val, or --split train --ids-file data/dev40_ids.txt.")

    gold_rows = load_split(args.split)
    wanted = None
    if args.ids_file:
        wanted = {
            line.strip()
            for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    gold = [row for row in gold_rows if wanted is None or row["sample_id"] in wanted]
    if args.limit:
        gold = gold[: args.limit]
    if not gold:
        raise SystemExit("no gold samples selected")

    result_rows = load_result(Path(args.path))
    counts = Counter(row["sample_id"] for row in result_rows)
    dupes = sorted(sid for sid, n in counts.items() if n > 1)
    if dupes:
        print(f"[warn] duplicate sample_id in result, last row wins: {dupes[:5]}")
    by_id = {row["sample_id"]: row for row in result_rows}
    missing = [row["sample_id"] for row in gold if row["sample_id"] not in by_id]
    extra = sorted(set(by_id) - {row["sample_id"] for row in gold})
    pairs = [(by_id[row["sample_id"]], row) for row in gold if row["sample_id"] in by_id]
    if not pairs:
        raise SystemExit(f"no overlapping sample_ids ({len(gold)} gold, {len(result_rows)} result rows)")
    if missing:
        print(f"[warn] {len(missing)} gold ids missing from result: {missing[:5]}")
    if extra:
        print(f"[warn] {len(extra)} result ids outside the gold set: {extra[:5]}")

    encoder, enc_label = pick_encoder(args.encoder, args.device or None)
    alpha_fn, future_fn, sem_label = pick_semantic(args.semantic)
    print(f"n={len(pairs)} encoder={enc_label} semantic={sem_label}")

    pred_chars = [len(str(f)) for pred, _gold in pairs for f in (pred.get("future_argument") or [])]
    gold_chars = [len(str(f)) for _pred, gold in pairs for f in (gold.get("future_argument") or [])]
    if pred_chars and gold_chars:
        print(
            f"future chars: pred mean {sum(pred_chars) / len(pred_chars):.0f} "
            f"vs gold mean {sum(gold_chars) / len(gold_chars):.0f}"
        )

    for arg_mode in ("newline", "mean"):
        agg = score_dataset(pairs, encoder, alpha_fn, future_sim_fn=future_fn, arg_mode=arg_mode)
        print(
            f"[{arg_mode}] score={agg['score']:.4f} s_ext={agg['s_ext']:.4f} "
            f"f1_ext={agg['f1_ext']:.4f} alpha={agg['alpha']:.4f} "
            f"s_pred={agg['s_pred']:.4f} f1_pred={agg['f1_pred']:.4f} "
            f"s_semantic={agg['s_semantic']:.4f}"
        )

    totals: Counter = Counter()
    for pred, gold in pairs:
        totals.update(
            attribute_sample(pred, gold, encoder, alpha_fn, future_sim_fn=future_fn, arg_mode="newline")
        )
    keys = ("matched", "stance_blocked", "low_sim", "pred_extra", "gold_missed")
    print("attribution[newline]: " + " ".join(f"{key}={totals[key]}" for key in keys))


if __name__ == "__main__":
    main()
