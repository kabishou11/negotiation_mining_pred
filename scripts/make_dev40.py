"""Write data/dev40_ids.txt and the complementary train fit ids."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer.devsplit import write_dev_split


def main() -> None:
    info = write_dev_split()
    print(info)


if __name__ == "__main__":
    main()
