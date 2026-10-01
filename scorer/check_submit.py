"""Validate a `result.jsonl` buffer against the submission rules.

Official files are UTF-8 JSON Lines without a BOM and without an outer array.
Each object has exactly `sample_id`, `issue_list`, `future_argument`. The id
set must equal the test set. `future_argument` has the same length as
`issue_list`. Stances are the three published strings. When source documents
are provided, every evidence string must be an exact substring of that sample.
"""

from __future__ import annotations

import json

_TOP = {"sample_id", "issue_list", "future_argument"}
_ISSUE_KEYS = {"issue_name", "stance", "argument_chain"}
_STANCES = {"support", "oppose", "neutral"}


def check_bytes(payload: bytes, expected_ids: list[str] | None = None, docs_by_id: dict[str, list[str]] | None = None) -> list[str]:
    errors: list[str] = []
    if payload.startswith(b"\xef\xbb\xbf"):
        errors.append("UTF-8 BOM is present")
        payload = payload[3:]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return errors + [f"not utf-8: {exc}"]
    if not text.strip():
        return errors + ["file is empty"]
    stripped = text.strip()
    if stripped.startswith("["):
        errors.append("outer JSON array is not JSON Lines")

    seen: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            errors.append(f"line {line_no}: blank")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid json ({exc.msg})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {line_no}: not an object")
            continue
        extra = set(obj) - _TOP
        missing = _TOP - set(obj)
        if extra:
            errors.append(f"line {line_no}: extra keys {sorted(extra)}")
        if missing:
            errors.append(f"line {line_no}: missing keys {sorted(missing)}")
            continue
        sample_id = obj["sample_id"]
        if not isinstance(sample_id, str):
            errors.append(f"line {line_no}: sample_id is not a string")
            continue
        seen.append(sample_id)
        issues = obj["issue_list"]
        futures = obj["future_argument"]
        if not isinstance(issues, list) or not isinstance(futures, list):
            errors.append(f"{sample_id}: issue_list and future_argument must be arrays")
            continue
        if len(issues) != len(futures):
            errors.append(f"{sample_id}: future_argument length {len(futures)} != issue_list {len(issues)}")
        if len(issues) == 0:
            errors.append(f"{sample_id}: empty issue_list")
        doc_texts = (docs_by_id or {}).get(sample_id)
        for k, issue in enumerate(issues):
            if not isinstance(issue, dict):
                errors.append(f"{sample_id} issue {k}: not an object")
                continue
            if set(issue) != _ISSUE_KEYS:
                errors.append(f"{sample_id} issue {k}: keys {sorted(issue)} != {sorted(_ISSUE_KEYS)}")
            stance = issue.get("stance")
            if stance not in _STANCES:
                errors.append(f"{sample_id} issue {k}: bad stance {stance!r}")
            name = issue.get("issue_name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{sample_id} issue {k}: empty issue_name")
            chain = issue.get("argument_chain")
            if not isinstance(chain, list) or not chain:
                errors.append(f"{sample_id} issue {k}: empty argument_chain")
                continue
            for t, evidence in enumerate(chain):
                if not isinstance(evidence, str) or not evidence.strip():
                    errors.append(f"{sample_id} issue {k} evidence {t}: empty")
                    continue
                if doc_texts is not None and not any(evidence in doc for doc in doc_texts):
                    errors.append(f"{sample_id} issue {k} evidence {t}: not a source substring")
        for k, future in enumerate(futures):
            if not isinstance(future, str) or not future.strip():
                errors.append(f"{sample_id} future {k}: empty")

    dupes = sorted({sid for sid in seen if seen.count(sid) > 1})
    if dupes:
        errors.append(f"duplicate sample_id: {dupes}")
    if expected_ids is not None:
        got = set(seen)
        exp = set(expected_ids)
        if got - exp:
            errors.append(f"unexpected sample_id: {sorted(got - exp)[:5]}")
        if exp - got:
            errors.append(f"missing sample_id: {sorted(exp - got)[:5]}")
    return errors
