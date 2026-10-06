"""Sweep the segmenter constants against gold-span containment.

`scorer/segment.py` calls its constants load-bearing: they were set once
and never measured against alternatives. This runs `containment_stats`
over a train subsample for each parameter combination and prints the
sentence-count/length trade-off beside it, so a constant change becomes a
measurement instead of a guess. Confirm the winner on the full train
before touching `scorer/segment.py` — and remember the constants affect
the NEXT training run only: the current adapter was trained on the
current segmentation.

    python3 scripts/sweep_segment.py --docs 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer import segment
from scorer.datautil import load_split
from scorer.segment import containment_stats

DEFAULTS = {"_MIN_KEEP": 32, "_MAX_MERGE": 180, "_PRIMARY_LIMIT": 160, "_HARD_CAP": 220}

CONFIGS = {
    "baseline": {},
    "mk24": {"_MIN_KEEP": 24},
    "mk40": {"_MIN_KEEP": 40},
    "mk48": {"_MIN_KEEP": 48},
    "mk56": {"_MIN_KEEP": 56},
    "mm200": {"_MAX_MERGE": 200},
    "mm160": {"_MAX_MERGE": 160},
    "pl180": {"_PRIMARY_LIMIT": 180},
    "pl140": {"_PRIMARY_LIMIT": 140},
    "hc256": {"_HARD_CAP": 256},
    "combo": {"_MIN_KEEP": 40, "_MAX_MERGE": 200, "_PRIMARY_LIMIT": 180},
    "combo48": {"_MIN_KEEP": 48, "_MAX_MERGE": 200, "_PRIMARY_LIMIT": 180},
}


def length_stats(samples: list[dict]) -> dict:
    lens: list[int] = []
    for sample in samples:
        seg = segment.segment_sample(sample)
        lens.extend(len(sent.text) for sent in seg.sentences)
    lens.sort()
    n = len(lens)
    return {"sentences": n, "mean": sum(lens) / n, "p95": lens[int(n * 0.95)], "max": lens[-1]}


def trimmed_coverage(samples: list[dict]) -> dict:
    """End-metric: how much of each gold evidence survives the real chain
    (align to 1-3 sentences -> trim each -> concatenate). in_one alone
    hides the tension that longer sentences get trimmed more aggressively,
    so the window may cut the gold span's edges."""
    from difflib import SequenceMatcher

    from scorer.compile import trim_evidence_text
    from scorer.segment import align_chain

    total = fully_kept = 0
    recall_sum = 0.0
    for sample in samples:
        seg = segment.segment_sample(sample)
        for issue in sample.get("issue_list") or []:
            name = str(issue.get("issue_name") or "")
            for ev in issue.get("argument_chain") or []:
                ids = align_chain(sample, seg, [str(ev)])
                if not ids:
                    continue
                total += 1
                cited = "".join(trim_evidence_text(name, seg.by_id[sid].text) for sid in ids)
                matcher = SequenceMatcher(None, str(ev), cited, autojunk=False)
                recall = sum(block.size for block in matcher.get_matching_blocks()) / len(str(ev))
                recall_sum += recall
                fully_kept += recall >= 0.9
    if not total:
        return {"n": 0, "mean_recall": 0.0, "fully_kept": 0.0}
    return {"n": total, "mean_recall": recall_sum / total, "fully_kept": fully_kept / total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=800, help="train subsample for screening")
    parser.add_argument(
        "--coverage-configs",
        default="",
        help="comma-separated configs to also measure post-trim gold recall on --coverage-docs docs",
    )
    parser.add_argument("--coverage-docs", type=int, default=300)
    args = parser.parse_args()
    train = load_split("train")[: args.docs]
    length_probe = train[:200]
    for name, overrides in CONFIGS.items():
        for attr, value in DEFAULTS.items():
            setattr(segment, attr, value)
        for attr, value in overrides.items():
            setattr(segment, attr, value)
        stats = containment_stats(train)
        lens = length_stats(length_probe)
        print(
            f"{name:>9}: in_one={stats['in_one']:.4f} in_one_or_two={stats['in_one_or_two']:.4f} "
            f"sentences={lens['sentences']:>5} mean_len={lens['mean']:.0f} p95={lens['p95']}"
        )
        if name in args.coverage_configs.split(","):
            coverage = trimmed_coverage(train[: args.coverage_docs])
            print(
                f"{name:>9}   post-trim gold recall={coverage['mean_recall']:.4f} "
                f"fully_kept={coverage['fully_kept']:.4f} (n={coverage['n']})"
            )
    for attr, value in DEFAULTS.items():
        setattr(segment, attr, value)


if __name__ == "__main__":
    main()
