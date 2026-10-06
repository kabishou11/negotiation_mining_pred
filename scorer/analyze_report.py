"""Turn an evaluate --out report into a ranked, quantified action list.

The decision tree in SUBMISSIONS.md says "fix the branch with the biggest
headroom". This computes that headroom from the per-sample attribution in
the report instead of eyeballing counts: for each miss class, the score if
every convertible miss of that class became a match. The bound is never
reached in practice — read the numbers as a relative ranking of where the
points are, not as promises.

Classes and their conversion model:
- low_sim      → evidence trim/rerank raises cosine; the issue then matches
                 a free gold issue. Denominators unchanged.
- stance_blocked → the stance flip makes the already-above-threshold edge
                 legal. Denominators unchanged.
- gold_missed  → --min-issues finds NEW issues for gold misses that no
                 current prediction could take; n_pred grows accordingly.
                 Only gold misses not already claimed by low_sim or
                 stance_blocked conversions count here.

    python3 -m scorer.analyze_report result_val_*_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMANDS = {
    "low_sim": "python3 -m scorer.postprocess <val_result> --split val --out pp.jsonl --rerank --trim-evidence",
    "stance_blocked": "./repro/val.sh <model> <adapter> val --stance-check",
    "gold_missed": "./repro/val.sh <model> <adapter> val --min-issues 4",
}


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def analyze_mode(samples: list[dict]) -> dict:
    n = len(samples)
    out: dict = {"n": n}
    for field in ("score", "s_ext", "f1_ext", "alpha", "s_pred"):
        out[field] = sum(s[field] for s in samples) / n if n else 0.0
    counts: dict[str, int] = {}
    for key in ("matched", "stance_blocked", "low_sim", "pred_extra", "gold_missed"):
        counts[key] = sum(s["detail"][key] for s in samples)
    out["attribution"] = counts

    def converted_score(sample: dict, gained: int, extra_pred: int) -> float:
        mp = sample["n_matched"] + gained
        p = mp / (sample["n_pred"] + extra_pred) if sample["n_pred"] + extra_pred else 0.0
        r = mp / sample["n_gold"] if sample["n_gold"] else 0.0
        return 0.8 * _f1(p, r) * (0.7 + 0.3 * sample["alpha"])

    def headroom(kind: str) -> float:
        deltas = []
        for s in samples:
            d = s["detail"]
            if kind == "gold_missed":
                room = max(0, d["gold_missed"] - d["low_sim"] - d["stance_blocked"])
                deltas.append(converted_score(s, room, room) - 0.8 * s["s_ext"])
            else:
                gained = min(d[kind], d["gold_missed"])
                deltas.append(converted_score(s, gained, 0) - 0.8 * s["s_ext"])
        return sum(deltas) / n if n else 0.0

    out["headroom"] = {
        "low_sim": headroom("low_sim"),
        "stance_blocked": headroom("stance_blocked"),
        "gold_missed": headroom("gold_missed"),
    }
    # Over-prediction signal: padding more issues is counterproductive when
    # extras already outnumber the convertible misses.
    out["pad_risk"] = sum(
        1
        for s in samples
        if s["detail"]["pred_extra"] > 0 and s["n_pred"] >= s["n_gold"]
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="the _report.json written by scorer.evaluate --out")
    parser.add_argument("--mode", choices=["newline", "mean"], default="newline")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    samples = [s for s in report.get("samples", []) if s.get("mode") == args.mode]
    if not samples:
        raise SystemExit(f"no {args.mode} samples in {args.report}")
    stats = analyze_mode(samples)

    print(
        f"n={stats['n']} mode={args.mode} score={stats['score']:.4f} "
        f"s_ext={stats['s_ext']:.4f} f1_ext={stats['f1_ext']:.4f} alpha={stats['alpha']:.4f} "
        f"s_pred={stats['s_pred']:.4f}"
    )
    print("attribution: " + " ".join(f"{k}={v}" for k, v in stats["attribution"].items()))
    modes = report.get("modes", {})
    if {"newline", "mean"} <= set(modes):
        delta = modes["newline"]["score"] - modes["mean"]["score"]
        if abs(delta) < 1e-9:
            print(f"evidence mode: newline-mean delta={delta:+.4f} → identical locally (either works)")
        else:
            print(f"evidence mode: newline-mean delta={delta:+.4f} → local pick: {'newline' if delta > 0 else 'mean'}")
    print("headroom (upper bound of a full fix, relative ranking only):")
    ranked = sorted(stats["headroom"].items(), key=lambda kv: -kv[1])
    for kind, value in ranked:
        print(f"  {kind:>14}: +{value:.4f}   {COMMANDS[kind]}")
    if stats["pad_risk"] * 2 > stats["n"]:
        print("[warn] over half the samples over-predict; prefer --min-issues 4 over 5, or skip padding.")
    best = ranked[0]
    # the report stores 6-decimal rounded scores, so recompute noise is
    # ~1e-6; real headrooms of interest are >=1e-3
    if best[1] <= 1e-5:
        print("[ok] no conversion headroom left in extraction — the misses are not of these classes.")
    else:
        print(f"[next] {best[0]} first: +{best[1]:.4f} upper bound.")


if __name__ == "__main__":
    main()
