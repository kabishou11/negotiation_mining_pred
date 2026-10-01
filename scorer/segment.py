"""Quote-aware sentence segmentation and gold-span alignment.

Operating point, measured on the preliminary train split (27,938 evidence
spans): a gold span is contained in one emitted sentence 90.1% of the time,
and in one sentence or two consecutive sentences 98.7% of the time. Those
two rates are the gate for training. Do not simplify the splitter without
re-running `scripts/align_report.py`.

Constants are load-bearing:
- primary delimiters are `。！？` and newline, not semicolons (semicolons cut
  through gold spans and lower containment);
- Chinese quotes `“”「」『』` suppress splits, which raises containment;
- spans longer than 160 characters are broken on `；;`, then short pieces
  shorter than 32 characters merge back if the merge stays within 180;
- anything still longer than 220 characters is cut on commas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_OPEN = {"“": "”", "「": "」", "『": "』"}
_CLOSE = {v: k for k, v in _OPEN.items()}
_PRIMARY = set("。！？\n")
_MIN_KEEP = 32
_MAX_MERGE = 180
_PRIMARY_LIMIT = 160
_HARD_CAP = 220


@dataclass(frozen=True)
class Sentence:
    sid: str
    text: str
    start: int
    end: int
    doc_index: int

    def prompt_line(self) -> str:
        return f"[{self.sid}] {self.text.strip()}"


@dataclass
class Segmented:
    sentences: list[Sentence]
    by_id: dict[str, Sentence] = field(init=False)
    prompt_body: str = ""

    def __post_init__(self) -> None:
        self.by_id = {s.sid: s for s in self.sentences}


def _base_spans(doc: str, delims: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    expected: list[str] = []
    for i, ch in enumerate(doc):
        if ch in _OPEN and not expected:
            expected.append(_OPEN[ch])
        elif expected and ch == expected[-1]:
            expected.pop()
        if ch in delims and not expected:
            if i + 1 > start and doc[start : i + 1].strip():
                spans.append((start, i + 1))
            start = i + 1
    if start < len(doc) and doc[start:].strip():
        spans.append((start, len(doc)))
    return spans


def _split_long(spans: list[tuple[int, int]], doc: str, limit: int, secondary: set[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if b - a <= limit:
            out.append((a, b))
            continue
        sub = _base_spans(doc[a:b], secondary)
        if len(sub) <= 1:
            i = a
            while i < b:
                j = min(b, i + limit)
                if j < b:
                    window_start = max(i, j - 40)
                    window = doc[window_start:j]
                    rel = max(window.rfind("，"), window.rfind(","), window.rfind("；"), window.rfind(" "))
                    if rel >= 8:
                        j = window_start + rel + 1
                if j <= i:
                    j = min(b, i + limit)
                if doc[i:j].strip():
                    out.append((i, j))
                i = j
        else:
            out.extend((a + sa, a + sb) for sa, sb in sub)
    return out


def _pack(spans: list[tuple[int, int]], doc: str) -> list[tuple[int, int]]:
    if not spans:
        return []
    out: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for a, b in spans[1:]:
        prev = out[-1]
        prev_len = len(doc[prev[0] : prev[1]].strip())
        cur_len = len(doc[a:b].strip())
        if (prev_len < _MIN_KEEP or cur_len < _MIN_KEEP) and (b - prev[0]) <= _MAX_MERGE:
            prev[1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def sentence_spans(doc: str) -> list[tuple[int, int]]:
    spans = _base_spans(doc, _PRIMARY)
    spans = _split_long(spans, doc, _PRIMARY_LIMIT, set("；;"))
    spans = _pack(spans, doc)
    spans = _split_long(spans, doc, _HARD_CAP, set("，,"))
    return spans


def segment_sample(sample: dict) -> Segmented:
    """Number sentences across every doc, in publish_date order."""
    docs = list(enumerate(sample.get("docs") or []))
    docs.sort(key=lambda item: (item[1].get("publish_date") or "", item[0]))
    sentences: list[Sentence] = []
    blocks: list[str] = []
    running = 0
    for doc_index, (_orig, doc) in enumerate(docs):
        text = doc.get("full_text") or ""
        header = "【文档{idx}｜{dtype}｜{date}】".format(
            idx=doc_index + 1,
            dtype=doc.get("doc_type") or "",
            date=doc.get("publish_date") or "",
        )
        lines = [header]
        for start, end in sentence_spans(text):
            running += 1
            raw = text[start:end]
            stripped = raw.strip()
            if not stripped:
                running -= 1
                continue
            lead = raw.find(stripped)
            sent = Sentence(
                sid=f"S{running:02d}",
                text=stripped,
                start=start + lead,
                end=start + lead + len(stripped),
                doc_index=doc_index,
            )
            sentences.append(sent)
            lines.append(sent.prompt_line())
        blocks.append("\n".join(lines))
    return Segmented(sentences=sentences, prompt_body="\n".join(blocks))


def _cover(sentences: list[Sentence], evidence: str, doc_text: str) -> list[str]:
    """Smallest run of 1, then 2, then 3 consecutive sentences that contains `evidence`."""
    if not evidence:
        return []
    singles = [s for s in sentences if evidence in s.text]
    if singles:
        best = min(singles, key=lambda s: (len(s.text), s.sid))
        return [best.sid]
    n = len(sentences)
    for width in (2, 3):
        best_ids: list[str] | None = None
        best_len: int | None = None
        for i in range(n - width + 1):
            group = sentences[i : i + width]
            if any(s.doc_index != group[0].doc_index for s in group):
                continue
            blob = doc_text[group[0].start : group[-1].end]
            if evidence in blob or evidence in "".join(s.text for s in group):
                length = group[-1].end - group[0].start
                if best_len is None or length < best_len:
                    best_len = length
                    best_ids = [s.sid for s in group]
        if best_ids:
            return best_ids
    return []


def align_chain(sample: dict, segmented: Segmented, chain: list[str]) -> list[str]:
    """Map a gold evidence list onto sentence ids, preserving evidence order."""
    docs = list(enumerate(sample.get("docs") or []))
    docs.sort(key=lambda item: (item[1].get("publish_date") or "", item[0]))
    ordered_docs = [doc for _i, doc in docs]
    ids: list[str] = []
    seen: set[str] = set()
    for evidence in chain:
        host = None
        host_text = ""
        for doc_index, doc in enumerate(ordered_docs):
            full = doc.get("full_text") or ""
            if evidence in full:
                host = doc_index
                host_text = full
                break
        pool = [s for s in segmented.sentences if host is None or s.doc_index == host]
        for sid in _cover(pool, evidence, host_text):
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def containment_stats(samples: list[dict]) -> dict[str, float]:
    """Fraction of gold evidences covered by 1 sentence, or by 1–2 consecutive."""
    one = two = total = 0
    for sample in samples:
        segmented = segment_sample(sample)
        docs = list(enumerate(sample.get("docs") or []))
        docs.sort(key=lambda item: (item[1].get("publish_date") or "", item[0]))
        ordered = [doc for _i, doc in docs]
        per_doc = {i: [s for s in segmented.sentences if s.doc_index == i] for i in range(len(ordered))}
        for issue in sample.get("issue_list") or []:
            for evidence in issue.get("argument_chain") or []:
                total += 1
                placed = False
                for doc_index, doc in enumerate(ordered):
                    full = doc.get("full_text") or ""
                    if evidence not in full and evidence not in "".join(s.text for s in per_doc[doc_index]):
                        continue
                    sents = per_doc[doc_index]
                    if any(evidence in s.text for s in sents):
                        one += 1
                        two += 1
                        placed = True
                        break
                    hit = False
                    for i in range(len(sents) - 1):
                        blob = full[sents[i].start : sents[i + 1].end]
                        joined = sents[i].text + sents[i + 1].text
                        if evidence in blob or evidence in joined:
                            hit = True
                            break
                    if hit:
                        two += 1
                        placed = True
                        break
                if not placed and any(evidence in s.text for s in segmented.sentences):
                    one += 1
                    two += 1
    if total == 0:
        return {"n": 0, "in_one": 0.0, "in_one_or_two": 0.0}
    return {"n": total, "in_one": one / total, "in_one_or_two": two / total}
