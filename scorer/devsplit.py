"""Hold out a stratified 40-document dev slice from train.

The slice is the only set used for daytime decisions. It is removed from the
fit ids so a later training run cannot tune on it. At least 8 documents
contain an `oppose` issue, because that class is 3% of issues and is a hard
gate in the judge.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from scorer.datautil import ROOT, load_split

QUOTA = {
    "国际新闻报道": 17,
    "联合声明": 15,
    "政策文件": 5,
    "记者会": 2,
    "评论文章": 1,
    "官方记者会文字实录": 0,
}
DEV_PATH = ROOT / "data" / "dev40_ids.txt"
FIT_PATH = ROOT / "data" / "train_fit_ids.txt"
META_PATH = ROOT / "data" / "dev40_meta.json"


def _has_oppose(sample: dict) -> bool:
    return any(issue.get("stance") == "oppose" for issue in sample.get("issue_list") or [])


def choose_dev40(samples: list[dict], n_oppose: int = 8, seed: int = 20261002) -> list[str]:
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    oppose: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        doc_type = sample["docs"][0]["doc_type"]
        by_type[doc_type].append(sample["sample_id"])
        if _has_oppose(sample):
            oppose[doc_type].append(sample["sample_id"])
    for bucket in list(by_type.values()) + list(oppose.values()):
        rng.shuffle(bucket)

    picked: list[tuple[str, str]] = []
    queues = {doc_type: list(ids) for doc_type, ids in oppose.items()}
    order = sorted(queues)
    while len(picked) < n_oppose and any(queues.values()):
        for doc_type in order:
            if not queues[doc_type]:
                continue
            picked.append((doc_type, queues[doc_type].pop()))
            if len(picked) == n_oppose:
                break
    taken = Counter(doc_type for doc_type, _sid in picked)
    selected = {sid for _doc_type, sid in picked}

    remaining = {doc_type: max(0, QUOTA.get(doc_type, 0) - taken[doc_type]) for doc_type in by_type}
    # Types the quota table does not know about still need a chance if the
    # oppose pass did not already fill the 40.
    shortfall = 40 - (len(selected) + sum(remaining.values()))
    if shortfall > 0:
        spare = sorted(by_type, key=lambda doc_type: len(by_type[doc_type]), reverse=True)
        for doc_type in spare:
            if shortfall == 0:
                break
            room = len(by_type[doc_type]) - taken[doc_type] - remaining.get(doc_type, 0)
            give = min(shortfall, max(0, room))
            remaining[doc_type] = remaining.get(doc_type, 0) + give
            shortfall -= give
    if shortfall < 0:
        # Oppose picks overshot a small quota. Drop fill slots from the
        # largest types until the total is 40.
        overflow = -shortfall
        for doc_type, _count in Counter(remaining).most_common():
            if overflow == 0:
                break
            cut = min(overflow, remaining[doc_type])
            remaining[doc_type] -= cut
            overflow -= cut

    for doc_type, need in remaining.items():
        for sid in by_type[doc_type]:
            if need == 0:
                break
            if sid in selected:
                continue
            selected.add(sid)
            need -= 1
    if len(selected) != 40:
        raise RuntimeError(f"dev split produced {len(selected)} ids, expected 40")
    return sorted(selected)


def write_dev_split(samples: list[dict] | None = None) -> dict:
    samples = samples if samples is not None else load_split("train")
    dev_ids = choose_dev40(samples)
    dev_set = set(dev_ids)
    fit_ids = sorted(sample["sample_id"] for sample in samples if sample["sample_id"] not in dev_set)
    by_id = {sample["sample_id"]: sample for sample in samples}
    meta = []
    for sid in dev_ids:
        sample = by_id[sid]
        meta.append(
            {
                "sample_id": sid,
                "doc_type": sample["docs"][0]["doc_type"],
                "n_issues": len(sample["issue_list"]),
                "has_oppose": _has_oppose(sample),
            }
        )
    DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEV_PATH.write_text("\n".join(dev_ids) + "\n", encoding="utf-8")
    FIT_PATH.write_text("\n".join(fit_ids) + "\n", encoding="utf-8")
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dev": len(dev_ids),
        "fit": len(fit_ids),
        "oppose_docs": sum(1 for row in meta if row["has_oppose"]),
        "by_type": dict(Counter(row["doc_type"] for row in meta)),
        "dev_path": str(DEV_PATH),
        "fit_path": str(FIT_PATH),
    }
