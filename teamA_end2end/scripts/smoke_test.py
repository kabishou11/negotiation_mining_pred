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


def check_prompt(doc: dict) -> None:
    user = e2e.user_prompt(doc)
    if "__FULL_TEXT__" in user or "__DOC_TYPE__" in user:
        _fail("prompt placeholders were not filled")
    if doc["docs"][0]["full_text"] not in user:
        _fail("prompt lost the document body")
    messages = e2e.build_messages(doc)
    if messages[0]["role"] != "system" or "JSON" not in messages[0]["content"]:
        _fail("system turn missing")


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
    norm = e2e.normalize(obj, full, cfg)
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


def main() -> int:
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "decode.json").read_text(encoding="utf-8"))
    docs = e2e.load_docs("val")
    doc = docs[0]
    check_prompt(doc)
    check_extract_json()
    check_normalize(doc, cfg)
    check_fallback(doc, cfg)
    print(f"smoke_test ok on {doc['sample_id']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
