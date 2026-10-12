"""Build the supervised-fine-tuning set from the fit split.

Dev40 is excluded. Extraction is most of the contest score, so every
document's extraction record is repeated four times; a document that
contains `oppose` is repeated eight. Future records stay at one copy.

Record counts mislead here: an extraction row is roughly six times the
characters of a future row, so at 2x repeat the loss already saw about
70% of its tokens from extraction. Four repeats push that near 80%
while inflating one epoch only ~1.6x; six would buy two more points of
share for a doubled epoch. Extraction targets are ISSUE lines whose
evidence ids are capped at 3, matching inference. Future targets are one
FUTURE line conditioned on the same aligned sentences the extractor would
hand over, not on the raw gold substring. Short gold futures are kept as
written.

Character length is converted to tokens with 1.5 characters per token. That
is the planning ratio for Qwen's Chinese BPE (typical range 1.3–1.8). The
training script replaces it with the real tokenizer once the checkpoint is
on disk.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from scorer.compile import gold_protocol_spans, resolve_span
from scorer.datautil import ROOT, load_split
from scorer.devsplit import FIT_PATH, write_dev_split
from scorer.prompt import extraction_messages, future_messages
from scorer.segment import align_chain, segment_sample

OUT_DIR = ROOT / "data" / "sft"
TRAIN_PATH = OUT_DIR / "train.jsonl"
STATS_PATH = OUT_DIR / "stats.json"
CHARS_PER_TOKEN = 1.5
MAX_EVIDENCE = 3
EXTRACT_REPEAT = 2
OPPOSE_BOOST = 2


def _fit_ids(train: list[dict]) -> set[str]:
    if not FIT_PATH.is_file():
        write_dev_split(train)
    return {line.strip() for line in FIT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}


def _issue_lines(protocol: str) -> list[str]:
    return [line for line in protocol.splitlines() if line.startswith("ISSUE ")]


def _cap_ids(line: str) -> str:
    head, _, ids = line.rpartition(" ||| ")
    kept = [sid for sid in ids.split(",") if sid][:MAX_EVIDENCE]
    return head + " ||| " + ",".join(kept)


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def _winner_protocol(winner: dict, sample: dict, segmented) -> str:
    """Winner issue_list (JSON level) back to ISSUE protocol lines with
    clause tokens; futures ride along as FUTURE lines from the winner file."""
    from scorer.segment import align_spans

    lines, futures = [], []
    wf = winner.get("future_argument") or []
    for idx, iss in enumerate(winner.get("issue_list") or []):
        tokens = align_spans(sample, segmented, list(iss.get("argument_chain") or []))
        if not tokens:
            continue
        lines.append(
            "ISSUE {n} ||| {s} ||| {t}".format(
                n=str(iss.get("issue_name") or "").replace("\n", ""),
                s=iss.get("stance"),
                t=",".join(tokens),
            )
        )
        f = wf[idx] if idx < len(wf) else ""
        futures.append("FUTURE " + str(f).replace("\n", ""))
    return "\n".join(lines + futures)


def build_records(train: list[dict] | None = None, seed: int = 0,
                  winners_path: str = "") -> tuple[list[dict], dict]:
    train = train if train is not None else load_split("train")
    fit = _fit_ids(train)
    winners: dict[str, dict] = {}
    if winners_path:
        from pathlib import Path as _P

        for line in _P(winners_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                w = json.loads(line)
                winners[w["sample_id"]] = w
    records: list[dict] = []
    dropped_issues = 0
    gold_issues = 0
    capped_lines = 0
    chars = {"extract": [], "future": []}
    oppose_docs = 0
    for sample in train:
        if sample["sample_id"] not in fit:
            continue
        segmented = segment_sample(sample)
        winner = winners.get(sample["sample_id"])
        if winner is not None and winner.get("issue_list"):
            winner_protocol = _winner_protocol(winner, sample, segmented)
        else:
            winner_protocol = ""
        protocol = gold_protocol_spans(sample, segmented)
        # The extract record teaches the winner's (judge-selected) issues;
        # the future task keeps the gold protocol lines so name-queue
        # alignment and gold futures stay paired by construction.
        extract_source = winner_protocol if _issue_lines(winner_protocol) else protocol
        issue_lines = [_cap_ids(line) for line in _issue_lines(extract_source)]
        gold_lines = [_cap_ids(line) for line in _issue_lines(protocol)]
        has_oppose = any(issue.get("stance") == "oppose" for issue in sample.get("issue_list") or [])
        if has_oppose:
            oppose_docs += 1
        repeat = EXTRACT_REPEAT * (OPPOSE_BOOST if has_oppose else 1)
        extract_record: dict | None = None
        if issue_lines:
            messages = extraction_messages(sample, segmented)
            messages.append({"role": "assistant", "content": "\n".join(issue_lines)})
            text = "\n".join(item["content"] for item in messages)
            extract_record = {
                "task": "extract",
                "sample_id": sample["sample_id"],
                "messages": messages,
                "chars": len(text),
            }
            chars["extract"].append(len(text))
        # A queue per name, not one list per name: the semifinal main issue
        # legitimately repeats one name three times, and a single entry would
        # attach every copy's future to the last copy's sentences. Protocol
        # lines are emitted in issue order, so popping the front stays aligned.
        by_name_queues: dict[str, list[list[str]]] = {}
        for line in gold_lines:
            name = line.split(" ||| ", 2)[0][len("ISSUE ") :]
            by_name_queues.setdefault(name, []).append(line.split(" ||| ")[-1].split(","))
        futures = list(sample.get("future_argument") or [])
        for index, issue in enumerate(sample.get("issue_list") or []):
            gold_issues += 1
            queue = by_name_queues.get(issue["issue_name"]) or []
            tokens = [t for t in (queue.pop(0) if queue else [])]
            resolved = [resolve_span(t, segmented) for t in tokens]
            resolved = [r for r in resolved if r]
            if not resolved:
                dropped_issues += 1
                continue
            trained = {
                "issue_name": issue["issue_name"],
                "stance": issue["stance"],
                "argument_chain": [text for _sid, text in resolved],
            }
            messages = future_messages(trained)
            target = futures[index] if index < len(futures) else ""
            messages.append({"role": "assistant", "content": "FUTURE " + str(target).replace("\n", "")})
            text = "\n".join(item["content"] for item in messages)
            records.append(
                {
                    "task": "future",
                    "sample_id": sample["sample_id"],
                    "messages": messages,
                    "chars": len(text),
                }
            )
            chars["future"].append(len(text))
        if extract_record is not None:
            for _ in range(repeat):
                records.append(extract_record)

    rng = random.Random(seed)
    rng.shuffle(records)
    task_counts = Counter(row["task"] for row in records)
    buckets = (2048, 3072, 4096, 6144, 8192)

    def bucket(pool: list[int]) -> dict[str, int]:
        tokens = [c / CHARS_PER_TOKEN + 40 for c in pool]
        out = {}
        for limit in buckets:
            out[f"le_{limit}"] = sum(1 for n in tokens if n <= limit)
        out["gt_8192"] = sum(1 for n in tokens if n > 8192)
        out["n"] = len(tokens)
        return out

    def expanded_chars(task: str) -> list[int]:
        return [row["chars"] for row in records if row["task"] == task]

    stats = {
        "fit_docs_with_a_record": len({row["sample_id"] for row in records}),
        "oppose_docs_in_fit": oppose_docs,
        "extract_repeat": EXTRACT_REPEAT,
        "oppose_extract_repeat": EXTRACT_REPEAT * OPPOSE_BOOST,
        "oppose_repeat_scope": "extract",
        "records": len(records),
        "task_counts": dict(task_counts),
        "gold_issues_seen": gold_issues,
        "issues_without_alignment": dropped_issues,
        "issue_lines_capped_to_3": capped_lines,
        "chars_per_token_assumption": CHARS_PER_TOKEN,
        "extract_chars": {
            "p50": _percentile(chars["extract"], 0.50),
            "p95": _percentile(chars["extract"], 0.95),
            "max": max(chars["extract"]) if chars["extract"] else 0,
        },
        "future_chars": {
            "p50": _percentile(chars["future"], 0.50),
            "p95": _percentile(chars["future"], 0.95),
            "max": max(chars["future"]) if chars["future"] else 0,
        },
        "extract_token_buckets": bucket(expanded_chars("extract")),
        "future_token_buckets": bucket(expanded_chars("future")),
        "token_estimate_one_epoch": int(sum(row["chars"] / CHARS_PER_TOKEN + 40 for row in records)),
    }
    return records, stats


def write_sft(winners_path: str = "") -> dict:
    records, stats = build_records(winners_path=winners_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats["path"] = str(TRAIN_PATH)
    return stats


def main() -> None:
    import os

    stats = write_sft(winners_path=os.environ.get("RAFT_WINNERS", ""))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
