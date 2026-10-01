"""Locate the preliminary jsonl files and load them.

The zip stays in the original materials directory. The first call extracts it
under `negotiation_mining_pred/data/prelim/`.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "prelim"
ZIP_PATH = (
    ROOT.parent
    / "赛题三谈判对手长文本观点挖掘与预测"
    / "赛题3-初赛.zip"
)


def ensure_extracted() -> Path:
    train = next(DATA_DIR.glob("**/train.jsonl"), None)
    if train is not None:
        return train.parent
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"preliminary zip not found: {ZIP_PATH}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    train = next(DATA_DIR.glob("**/train.jsonl"))
    return train.parent


def split_path(name: str) -> Path:
    if name not in {"train", "val", "test"}:
        raise ValueError(name)
    return ensure_extracted() / f"{name}.jsonl"


def load_split(name: str) -> list[dict]:
    rows = []
    with split_path(name).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
