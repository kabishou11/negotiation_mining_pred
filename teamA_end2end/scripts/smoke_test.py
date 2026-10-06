"""Offline gate for the second-team pipeline. No torch, no scorer imports.

Run from teamA_end2end/:

    python3 scripts/smoke_test.py

Exercises the prompt render, JSON extraction (fences, trailing commas,
truncation repair), evidence substring forcing, normalisation, and the
fallback result against a real validation document.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import infer_e2e as e2e  # noqa: E402


def _fail(message: str) -> None:
    raise AssertionError(message)


def check_prompt(doc: dict, cfg: dict) -> None:
    user = e2e.user_prompt(doc, cfg)
    if "__FULL_TEXT__" in user or "__DOC_TYPE__" in user or "__EXAMPLES__" in user:
        _fail("prompt placeholders were not filled")
    if doc["docs"][0]["full_text"] not in user:
        _fail("prompt lost the document body")
    messages = e2e.build_messages(doc, cfg)
    if messages[0]["role"] != "system" or "JSON" not in messages[0]["content"]:
        _fail("system turn missing")


def check_few_shot(cfg: dict) -> None:
    k = int(cfg.get("few_shot", 0))
    if k <= 0:
        return
    block = e2e.few_shot_block(k)
    if not block or "<example>" not in block or "<output>" not in block:
        _fail("few-shot block missing")
    import re as _re

    # Every example must be internally consistent: each shown evidence string
    # is a substring of the shown document text.
    for example in block.split("<example>")[1:]:
        doc_open = example.index("<document")
        shown = example[example.index(">", doc_open) + 1 : example.index("</document>")]
        payload = example.split("<output>")[1].split("</output>")[0]
        obj = json.loads(payload)
        if not obj["issue_list"] or len(obj["future_argument"]) != len(obj["issue_list"]):
            _fail("few-shot output malformed")
        for issue in obj["issue_list"]:
            for evidence in issue["argument_chain"]:
                if evidence not in shown:
                    _fail("few-shot example shows evidence outside the shown text")
        if _re.search(r"\bissue_name\b", shown):
            _fail("few-shot block leaked JSON keys into the document")


def check_extract_json() -> None:
    ok = e2e.extract_json('```json\n{"issue_list": [], "future_argument": []}\n```')
    if ok != {"issue_list": [], "future_argument": []}:
        _fail(f"fenced json failed: {ok}")
    ok = e2e.extract_json('noise before {"issue_list": [{"issue_name": "甲", "stance": "support", "argument_chain": ["x"]},], "future_argument": ["f"],} noise after')
    if ok is None or ok["issue_list"][0]["issue_name"] != "甲":
        _fail(f"trailing-comma repair failed: {ok}")
    # Truncated inside a string value: an inner issue object must NOT
    # surface as the top level; the repair keeps the complete issues.
    ok = e2e.extract_json(
        '{"issue_list": [{"issue_name": "关税", "stance": "support"},'
        ' {"issue_name": "能源", "stance": "neutral", "argument_chain": ["双方同意'
    )
    if ok is None or not ok.get("issue_list"):
        _fail(f"truncation repair lost the complete issues: {ok}")
    if ok["issue_list"][0]["issue_name"] != "关税":
        _fail(f"truncation repair returned a fragment: {ok}")
    # A bare inner issue object is never mistaken for the whole result.
    bare = e2e.extract_json('{"issue_list": [{"issue_name": "关税", "stance": "support"}, {"issue_name": "能源", "stance"')
    if bare is None or "issue_list" not in bare:
        _fail(f"inner object surfaced as top level: {bare}")
    # Truncated right after a key: the dangling key is dropped.
    ok = e2e.extract_json('{"issue_list": [{"issue_name": "关税", "stance": "support", "argument_chain"')
    if ok is None or ok["issue_list"][0].get("argument_chain"):
        _fail(f"dangling key not stripped: {ok}")
    if e2e.extract_json("no json here") is not None:
        _fail("garbage produced an object")
    if e2e._close_json("{}") is not None:
        _fail("a complete object was 'repaired'")


def check_normalize(doc: dict, cfg: dict) -> None:
    full = "\n".join(str(d.get("full_text") or "") for d in doc["docs"])
    issue = doc["issue_list"][0]
    evidence = issue["argument_chain"][0]
    junk = evidence[: len(evidence) // 2] + "（改写部分）"
    obj = {
        "issue_list": [
            {"issue_name": issue["issue_name"], "stance": "支持", "argument_chain": [evidence, junk, "完全无关的编造证据内容无效"]},
            {"issue_name": "坏项", "stance": "拒绝", "argument_chain": [evidence]},
        ],
        "future_argument": [],
    }
    norm = e2e.normalize(obj, doc["docs"], cfg)
    chain = norm["issue_list"][0]["argument_chain"]
    if chain[0] != evidence or not all(c in full for c in chain):
        _fail(f"evidence forcing failed: {chain}")
    if len(norm["future_argument"]) != len(norm["issue_list"]):
        _fail("futures length drifted from issues")
    if norm["issue_list"][0]["stance"] != "support":
        _fail("stance map failed")
    if any(i["stance"] not in ("support", "oppose", "neutral") for i in norm["issue_list"]):
        _fail("a bad stance survived")
    short = e2e.force_substring("太短", full, cfg)
    if short:
        _fail("a too-short evidence passed")


def check_fallback(doc: dict, cfg: dict) -> None:
    full = "\n".join(str(d.get("full_text") or "") for d in doc["docs"])
    fb = e2e.fallback_result(doc, cfg)
    evidence = fb["issue_list"][0]["argument_chain"][0]
    if evidence not in full:
        _fail("fallback evidence left the source")
    if fb["issue_list"][0]["stance"] not in ("support", "oppose", "neutral"):
        _fail("fallback stance invalid")
    if len(fb["future_argument"]) != len(fb["issue_list"]):
        _fail("fallback length mismatch")


def check_merge() -> None:
    candidates = [
        {
            "issue_list": [
                {"issue_name": "关税问题", "stance": "support", "argument_chain": ["证据甲", "证据乙"]},
                {"issue_name": "能源合作", "stance": "neutral", "argument_chain": ["证据丙"]},
            ],
            "future_argument": ["后续一", "后续二"],
        },
        {
            "issue_list": [
                {"issue_name": "关税问题", "stance": "oppose", "argument_chain": ["证据甲"]},
                {"issue_name": "能源合作", "stance": "neutral", "argument_chain": ["证据丙", "证据丁"]},
                {"issue_name": "渔业谈判", "stance": "support", "argument_chain": ["证据戊"]},
            ],
            "future_argument": ["后续三", "后续四", "后续五"],
        },
    ]
    cfg = {"max_issues": 6, "max_evidence": 3}
    merged = e2e.merge_candidates(candidates, cfg)
    names = [issue["issue_name"] for issue in merged["issue_list"]]
    if names != ["关税问题", "能源合作", "渔业谈判"]:
        _fail(f"merge clusters wrong: {names}")
    tariffs = merged["issue_list"][0]
    if tariffs["stance"] != "support":
        _fail(f"merge majority stance wrong: {tariffs}")
    if tariffs["argument_chain"] != ["证据甲", "证据乙"]:
        _fail(f"merge evidence frequency order wrong: {tariffs}")
    if merged["future_argument"] != ["后续一", "后续二", "后续五"]:
        _fail(f"merge futures misaligned: {merged['future_argument']}")
    if e2e.merge_candidates([], cfg) is not None:
        _fail("empty candidate list produced a merge")
    capped = e2e.merge_candidates(candidates, {"max_issues": 1, "max_evidence": 3})
    if len(capped["issue_list"]) != 1 or len(capped["future_argument"]) != 1:
        _fail("max_issues cap broke the futures alignment")


def main() -> int:
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "decode.json").read_text(encoding="utf-8"))
    docs = e2e.load_docs("val")
    doc = docs[0]
    check_prompt(doc, cfg)
    check_few_shot(cfg)
    check_extract_json()
    check_normalize(doc, cfg)
    check_fallback(doc, cfg)
    check_merge()
    print(f"smoke_test ok on {doc['sample_id']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
