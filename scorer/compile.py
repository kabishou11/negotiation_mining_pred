"""Turn the line protocol into the official result object.

The model never emits raw JSON. Sentence ids are looked up and replaced with
the exact sentence text stored on the segmented document, so every evidence
string is a substring of the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scorer.segment import Segmented, align_chain

_ISSUE = re.compile(
    r"^ISSUE\s+(.+?)\s*\|\|\|\s*(support|oppose|neutral)\s*\|\|\|\s*(.*)$",
    re.IGNORECASE,
)
_FUTURE = re.compile(r"^FUTURE\s*(.*)$")
_SID = re.compile(r"S\d+")
_STANCES = {"support", "oppose", "neutral"}


@dataclass
class Compiled:
    issue_list: list[dict]
    future_argument: list[str]
    warnings: list[str] = field(default_factory=list)

    def public(self, sample_id: str) -> dict:
        issues = []
        for issue in self.issue_list:
            issues.append(
                {
                    "issue_name": issue["issue_name"],
                    "stance": issue["stance"],
                    "argument_chain": list(issue["argument_chain"]),
                }
            )
        return {
            "sample_id": sample_id,
            "issue_list": issues,
            "future_argument": list(self.future_argument),
        }


def _name_grams_of(name: str) -> set[str]:
    flat = "".join(str(name).split())
    if len(flat) < 2:
        return {flat} if flat else set()
    return {flat[i : i + 2] for i in range(len(flat) - 1)}


def mentions_issue(name: str, text: str) -> bool:
    """Whether `text` already touches the issue name by any 2-gram.

    Measured on the train split: 97.9% of gold futures mention their issue
    name this way, so a generated future that misses it entirely is
    off-distribution and worth correcting (see postprocess
    --future-name-check)."""
    grams = _name_grams_of(name)
    return bool(grams) and any(gram in text for gram in grams)


def _fallback_future(issue: dict) -> str:
    evidence = issue["argument_chain"][0].strip()
    snippet = evidence[:40]
    return f"后续将延续「{snippet}」所体现的现有安排。"


_TRIM_DELIMS = set("，。！？；：、,,;:")
_TRIM_MAX_WHOLE = 64
_TRIM_TARGET = 55
_TRIM_MIN_CLAUSE = 1


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _TRIM_DELIMS:
            if i + 1 - start >= _TRIM_MIN_CLAUSE and text[start : i + 1].strip():
                spans.append((start, i + 1))
            start = i + 1
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _name_bigrams(name: str) -> set[str]:
    flat = "".join(str(name).split())
    if len(flat) < 2:
        return {flat} if flat else set()
    return {flat[i : i + 2] for i in range(len(flat) - 1)}


def trim_evidence_text(name: str, text: str) -> str:
    """Cut a long evidence sentence down to the clause window that keeps the
    issue name's bigrams and stays near the gold evidence length (~55 chars).

    Gold evidence averages 53 characters; whole emitted sentences run much
    longer, which dilutes the bge cosine against the >0.7 threshold. Any
    returned string is a substring of `text`, and `text` is already an exact
    source slice, so the submission constraint holds. Sentences at or under
    64 characters are kept whole.
    """
    text = text.strip()
    if len(text) <= _TRIM_MAX_WHOLE:
        return text
    spans = _clause_spans(text)
    if len(spans) <= 1:
        return text
    grams = _name_bigrams(name)
    best_key: tuple[int, int] | None = None
    best = ""
    for a in range(len(spans)):
        for b in range(a, len(spans)):
            start, end = spans[a][0], spans[b][1]
            if end - start > _TRIM_MAX_WHOLE:
                break
            piece = text[start:end]
            score = sum(1 for gram in grams if gram in piece)
            key = (score, -abs(len(piece) - _TRIM_TARGET))
            if best_key is None or key > best_key:
                best_key = key
                best = piece
    trimmed = best.strip(" ,，;；、:： ")
    return trimmed if trimmed else text


def _near_dup(a: str, b: str, threshold: float = 0.85) -> bool:
    """Character-bigram cosine over two evidence strings.

    Private to the trim path: two DIFFERENT sentences can trim down to
    windows sharing most of their text, and a joined chain that repeats
    itself reads as redundant to the bge cosine against the gold chain.
    """
    grams_a = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    grams_b = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    if not grams_a or not grams_b:
        return False
    return len(grams_a & grams_b) / ((len(grams_a) * len(grams_b)) ** 0.5) >= threshold


def trim_chain(issue_name: str, chain: list[str]) -> list[str]:
    out: list[str] = []
    for evidence in chain:
        trimmed = trim_evidence_text(issue_name, evidence)
        if trimmed and trimmed not in out and not any(_near_dup(trimmed, kept) for kept in out):
            out.append(trimmed)
    return out or list(chain)


_SUPPORT_DOC_TYPES = {"联合声明", "政策文件", "记者会"}


def _ordered_docs(sample: dict) -> list[dict]:
    docs = list(enumerate(sample.get("docs") or []))
    docs.sort(key=lambda item: (item[1].get("publish_date") or "", item[0]))
    return [doc for _index, doc in docs]


def rule_fallback(sample: dict, segmented: Segmented) -> Compiled | None:
    """One submittable issue when extraction emits no ISSUE line.

    Evidence is the first sentence, an exact source substring. Stance follows
    the majority prior: support for joint statements, policy texts, and press
    conferences; neutral otherwise. The fallback never emits oppose. Returns
    None when there is no sentence to cite.
    """
    if not segmented.sentences:
        return None
    sent = segmented.sentences[0]
    ordered = _ordered_docs(sample)
    dtype = ""
    if ordered and sent.doc_index < len(ordered):
        dtype = ordered[sent.doc_index].get("doc_type") or ""
    stance = "support" if dtype in _SUPPORT_DOC_TYPES else "neutral"
    issue = {
        "issue_name": "文本主议题",
        "stance": stance,
        "argument_chain": [sent.text],
        "sent_ids": [sent.sid],
    }
    return Compiled(
        issue_list=[issue],
        future_argument=[_fallback_future(issue)],
        warnings=["rule fallback: empty extraction"],
    )


def expand_semifinal(compiled: Compiled, segmented: Segmented) -> Compiled:
    """Repeat the main issue once per speaking party, up to three.

    A party is one source document that produced at least one sentence, in
    publish_date order. Each copy keeps the same issue name; copies are not
    merged. A sub-issue is kept once, and only when its evidence is a real
    sentence from a document that spoke. Speaker ids stay internal; `public`
    drops them. Preliminary samples have one document, so the prelim path
    must not call this.
    """
    if not compiled.issue_list or not segmented.sentences:
        return compiled
    by_doc: dict[int, list] = {}
    for sent in segmented.sentences:
        by_doc.setdefault(sent.doc_index, []).append(sent)
    party_docs = sorted(by_doc)[:3]
    main = compiled.issue_list[0]
    main_future = compiled.future_argument[0] if compiled.future_argument else _fallback_future(main)
    issues: list[dict] = []
    futures: list[str] = []
    main_ids = list(main.get("sent_ids") or [])
    for doc_index in party_docs:
        chain_ids = [
            sid
            for sid in main_ids
            if sid in segmented.by_id and segmented.by_id[sid].doc_index == doc_index
        ]
        if chain_ids:
            chain = [segmented.by_id[sid].text for sid in chain_ids]
        else:
            first = by_doc[doc_index][0]
            chain_ids = [first.sid]
            chain = [first.text]
        issues.append(
            {
                "issue_name": main["issue_name"],
                "stance": main["stance"],
                "argument_chain": chain,
                "sent_ids": chain_ids,
                "speaker": f"P{doc_index + 1}",
            }
        )
        futures.append(main_future)
    for offset, issue in enumerate(compiled.issue_list[1:], start=1):
        chain = list(issue.get("argument_chain") or [])
        if not chain:
            continue
        ids = list(issue.get("sent_ids") or [])
        host = None
        if ids and ids[0] in segmented.by_id:
            host = segmented.by_id[ids[0]].doc_index
        else:
            for sent in segmented.sentences:
                if chain[0] == sent.text or chain[0] in sent.text:
                    host = sent.doc_index
                    break
        if host is None or host not in by_doc:
            continue
        future = compiled.future_argument[offset] if offset < len(compiled.future_argument) else ""
        if not str(future).strip():
            future = _fallback_future(issue)
        issues.append(
            {
                "issue_name": issue["issue_name"],
                "stance": issue["stance"],
                "argument_chain": chain,
                "sent_ids": ids,
                "speaker": f"P{host + 1}",
            }
        )
        futures.append(future)
    warnings = list(compiled.warnings)
    warnings.append(f"semifinal expanded main issue across {len(party_docs)} parties")
    return Compiled(issue_list=issues, future_argument=futures, warnings=warnings)


def compile_protocol(
    text: str,
    segmented: Segmented,
    *,
    max_issues: int | None = 6,
    max_evidence: int = 3,
    fill_missing_future: bool = False,
) -> Compiled:
    warnings: list[str] = []
    issues: list[dict] = []
    futures: list[str] = []
    # Exact duplicates only: same name, stance, and ids. Names repeat
    # legitimately in the semifinal (main issue x3), so name alone must not
    # be the key; an exact duplicate would just burn a prediction slot.
    seen_issues: set[tuple[str, str, tuple[str, ...]]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        issue_match = _ISSUE.match(line)
        if issue_match:
            name = issue_match.group(1).strip()
            stance = issue_match.group(2).lower()
            if stance not in _STANCES:
                warnings.append(f"bad stance dropped: {line}")
                continue
            ids: list[str] = []
            for sid in _SID.findall(issue_match.group(3)):
                if sid not in segmented.by_id:
                    warnings.append(f"unknown sentence id {sid}")
                    continue
                if sid not in ids:
                    ids.append(sid)
            if max_evidence is not None and len(ids) > max_evidence:
                warnings.append(f"evidence capped at {max_evidence} for {name}")
                ids = ids[:max_evidence]
            chain = [segmented.by_id[sid].text for sid in ids]
            if not name or not chain:
                warnings.append(f"issue dropped (empty name or evidence): {name}")
                continue
            key = (name, stance, tuple(ids))
            if key in seen_issues:
                warnings.append(f"duplicate issue line dropped: {name}")
                continue
            seen_issues.add(key)
            issues.append({"issue_name": name, "stance": stance, "argument_chain": chain, "sent_ids": ids})
            continue
        future_match = _FUTURE.match(line)
        if future_match:
            futures.append(future_match.group(1).strip())
            continue
        warnings.append(f"unparsed line: {line[:80]}")

    if max_issues is not None and len(issues) > max_issues:
        warnings.append(f"issues capped at {max_issues}")
        issues = issues[:max_issues]
    futures = futures[: len(issues)]
    if fill_missing_future:
        while len(futures) < len(issues):
            futures.append(_fallback_future(issues[len(futures)]))
            warnings.append("filled a missing future from evidence")
    elif len(futures) < len(issues):
        warnings.append(f"future rows {len(futures)} < issues {len(issues)}")
        while len(futures) < len(issues):
            futures.append("")
    return Compiled(issue_list=issues, future_argument=futures, warnings=warnings)


def gold_protocol(sample: dict, segmented: Segmented) -> str:
    """Training-target lines for one labelled sample.

    Ids are not capped here. The compiler cap is an inference decision and is
    measured separately.
    """
    lines: list[str] = []
    futures: list[str] = []
    gold_futures = list(sample.get("future_argument") or [])
    for index, issue in enumerate(sample.get("issue_list") or []):
        ids = align_chain(sample, segmented, list(issue.get("argument_chain") or []))
        if not ids:
            continue
        lines.append(
            "ISSUE {name} ||| {stance} ||| {ids}".format(
                name=str(issue.get("issue_name") or "").replace("\n", ""),
                stance=issue.get("stance"),
                ids=",".join(ids),
            )
        )
        future = gold_futures[index] if index < len(gold_futures) else ""
        futures.append("FUTURE " + str(future).replace("\n", ""))
    return "\n".join(lines + futures)
