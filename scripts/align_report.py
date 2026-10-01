"""Print gold-span containment for train and val."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer.datautil import load_split
from scorer.segment import containment_stats


def main() -> None:
    for name in ("train", "val"):
        stats = containment_stats(load_split(name))
        print(
            f"{name}: n={stats['n']} in_one={stats['in_one']:.4f} "
            f"in_one_or_two={stats['in_one_or_two']:.4f}"
        )


if __name__ == "__main__":
    main()
