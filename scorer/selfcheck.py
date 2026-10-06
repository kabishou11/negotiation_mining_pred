"""Gate for the local judge, the segmenter, and the submission checker.

Run from `negotiation_mining_pred`:

    python3 -m scorer.selfcheck
"""

from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
from pathlib import Path

from scorer.attribute import attribute_sample
from scorer.check_submit import check_bytes
from scorer.compile import compile_protocol, expand_semifinal, gold_protocol, rule_fallback
from scorer.datautil import load_split
from scorer.devsplit import write_dev_split
from scorer.encoders import OrthogonalEncoder
from scorer.matching import max_weight_assignment
from scorer.prompt import extraction_messages
from scorer.score import rouge_l_f1, score_sample, semantic_alpha
from scorer.segment import align_chain, containment_stats, segment_sample


def _fail(message: str) -> None:
    raise AssertionError(message)


def _brute_best(weights: list[list[float | None]]) -> float:
    n = len(weights)
    m = len(weights[0]) if n else 0
    best = 0.0

    def walk(row: int, used: int, total: float) -> None:
        nonlocal best
        if row == n:
            if total > best:
                best = total
            return
        walk(row + 1, used, total)
        for col in range(m):
            if used & (1 << col):
                continue
            weight = weights[row][col]
            if weight is None:
                continue
            walk(row + 1, used | (1 << col), total + weight)

    walk(0, 0, 0.0)
    return best


def check_matching() -> None:
    rng = random.Random(0)
    for _ in range(40):
        n = rng.randint(1, 4)
        m = rng.randint(1, 5)
        weights: list[list[float | None]] = []
        for _i in range(n):
            row = []
            for _j in range(m):
                if rng.random() < 0.35:
                    row.append(None)
                else:
                    row.append(round(rng.random(), 3))
            weights.append(row)
        chosen = max_weight_assignment(weights)
        got = sum(weights[i][j] for i, j in chosen)
        cols = [j for _i, j in chosen]
        if len(cols) != len(set(cols)):
            _fail("matching reused a column")
        if abs(got - _brute_best(weights)) > 1e-6:
            _fail(f"matching not optimal: {got} vs {_brute_best(weights)} on {weights}")
    forced = max_weight_assignment([[None, None], [None, None]])
    if forced:
        _fail("forbidden edges were selected")
    prefer = max_weight_assignment([[0.2, 0.9], [0.4, None]])
    if set(prefer) != {(0, 1), (1, 0)}:
        _fail(f"expected the heavier edges, got {prefer}")


def check_threshold() -> None:
    """0.7 exactly is not a match. The rule says 超过 0.7."""

    class Fixed:
        def __init__(self, vectors: dict[str, list[float]]) -> None:
            self.vectors = vectors

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [self.vectors[text] for text in texts]

    def case(arg_cos: float) -> int:
        # name cosine is 1. weighted = 0.4 + 0.6 * arg_cos.
        orth = (1 - arg_cos * arg_cos) ** 0.5
        encoder = Fixed(
            {
                "议题": [1.0, 0.0],
                "金标句": [1.0, 0.0],
                "预测句": [arg_cos, orth],
            }
        )
        pred = {
            "issue_list": [{"issue_name": "议题", "stance": "support", "argument_chain": ["预测句"]}],
            "future_argument": ["后文"],
        }
        gold = {
            "issue_list": [{"issue_name": "议题", "stance": "support", "argument_chain": ["金标句"]}],
            "future_argument": ["后文"],
        }
        return score_sample(pred, gold, encoder, semantic_alpha).n_matched

    if case(0.5) != 0:
        _fail("cosine exactly 0.7 was accepted")
    if case(0.51) != 1:
        _fail("cosine just above 0.7 was rejected")


def _as_result(sample: dict) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "issue_list": [
            {
                "issue_name": issue["issue_name"],
                "stance": issue["stance"],
                "argument_chain": list(issue["argument_chain"]),
            }
            for issue in sample["issue_list"]
        ],
        "future_argument": list(sample["future_argument"]),
    }


def check_identity(samples: list[dict]) -> None:
    encoder = OrthogonalEncoder()
    for sample in samples:
        result = _as_result(sample)
        scored = score_sample(result, result, encoder, semantic_alpha)
        if abs(scored.score - 1.0) > 1e-9 or scored.n_matched != len(result["issue_list"]):
            _fail(f"{sample['sample_id']} self-score {scored.score} matched {scored.n_matched}")


def check_perturbations(sample: dict) -> None:
    encoder = OrthogonalEncoder()
    base = _as_result(sample)
    original = score_sample(base, base, encoder, semantic_alpha)

    flipped = copy.deepcopy(base)
    flipped["issue_list"][0]["stance"] = "oppose" if base["issue_list"][0]["stance"] != "oppose" else "support"
    flip_score = score_sample(flipped, base, encoder, semantic_alpha)
    if flip_score.n_matched != original.n_matched - 1:
        _fail(f"stance flip matched {flip_score.n_matched}, expected {original.n_matched - 1}")
    if not (flip_score.s_ext < original.s_ext):
        _fail("stance flip did not lower S_ext")
    attrs = attribute_sample(flipped, base, encoder, semantic_alpha)
    if attrs["stance_blocked"] < 1:
        _fail(f"stance flip was not attributed, got {attrs}")

    order = list(range(len(base["issue_list"])))
    order.reverse()
    moved = {
        "sample_id": base["sample_id"],
        "issue_list": [base["issue_list"][i] for i in order],
        "future_argument": [base["future_argument"][i] for i in order],
    }
    moved_score = score_sample(moved, base, encoder, semantic_alpha)
    if abs(moved_score.score - original.score) > 1e-9:
        _fail("joint permutation changed the score")

    futures_only = copy.deepcopy(base)
    futures_only["future_argument"] = list(reversed(base["future_argument"]))
    fut_score = score_sample(futures_only, base, encoder, semantic_alpha)
    if abs(fut_score.s_ext - original.s_ext) > 1e-9:
        _fail("future-only reversal changed S_ext")
    if not (fut_score.s_pred < original.s_pred - 1e-6):
        _fail("future-only reversal did not lower S_pred")

    extra = copy.deepcopy(base)
    extra["issue_list"].append(
        {"issue_name": "完全无关的虚构议题名", "stance": "neutral", "argument_chain": ["这段文字不会出现在金标里。"]}
    )
    extra["future_argument"].append("这条预测不对应任何金标议题。")
    extra_score = score_sample(extra, base, encoder, semantic_alpha)
    if not (extra_score.precision < original.precision and extra_score.score < original.score):
        _fail("an extra issue did not lower precision and score")


def check_rouge() -> None:
    if rouge_l_f1("甲乙丙", "甲乙丙") != 1:
        _fail("identical rouge")
    if abs(rouge_l_f1("甲乙", "甲丙") - 0.5) > 1e-9:
        _fail("partial rouge")
    if rouge_l_f1("", "甲") != 0:
        _fail("empty rouge")


def check_segments(train: list[dict]) -> None:
    stats = containment_stats(train)
    print(
        f"alignment train n={stats['n']} in_one={stats['in_one']:.4f} "
        f"in_one_or_two={stats['in_one_or_two']:.4f}"
    )
    if stats["in_one"] < 0.90:
        _fail("single-sentence containment fell below 0.90")
    if stats["in_one_or_two"] < 0.98:
        _fail("two-sentence coverage fell below 0.98")

    contained = total = 0
    issues_kept = issues_gold = 0
    for sample in train:
        segmented = segment_sample(sample)
        docs = sorted(enumerate(sample["docs"]), key=lambda item: (item[1].get("publish_date") or "", item[0]))
        ordered = [doc for _i, doc in docs]
        for sent in segmented.sentences:
            full = ordered[sent.doc_index]["full_text"]
            if full[sent.start : sent.end] != sent.text or sent.text not in full:
                _fail(f"{sample['sample_id']} {sent.sid} is not an exact slice")
        protocol = gold_protocol(sample, segmented)
        compiled = compile_protocol(protocol, segmented, max_issues=None, max_evidence=8)
        if len(compiled.future_argument) != len(compiled.issue_list):
            _fail("compiled gold protocol broke length equality")
        for issue in compiled.issue_list:
            for evidence in issue["argument_chain"]:
                if not any(evidence in doc["full_text"] for doc in sample["docs"]):
                    _fail("compiled evidence left the source text")
        for issue in sample["issue_list"]:
            issues_gold += 1
            ids = align_chain(sample, segmented, list(issue["argument_chain"]))
            if ids:
                issues_kept += 1
            for evidence in issue["argument_chain"]:
                total += 1
                ev_ids = align_chain(sample, segmented, [evidence])
                if not ev_ids:
                    continue
                sents = [segmented.by_id[sid] for sid in ev_ids]
                full = ordered[sents[0].doc_index]["full_text"]
                blob = full[sents[0].start : sents[-1].end]
                if evidence in blob or evidence in "".join(s.text for s in sents):
                    contained += 1
        body = extraction_messages(sample, segmented)[1]["content"]
        if sample["docs"][0]["doc_type"] not in body:
            _fail("prompt dropped doc_type")
        if segmented.sentences and segmented.sentences[0].sid not in body:
            _fail("prompt dropped a sentence id")
    cover = contained / total if total else 0
    print(f"gold evidence inside aligned sentence run {contained}/{total}={cover:.4f}")
    print(f"issues with at least one aligned sentence {issues_kept}/{issues_gold}")
    if cover < 0.98:
        _fail("aligned sentence runs do not cover gold evidences")


def check_submit(sample: dict) -> None:
    result = _as_result(sample)
    line = json.dumps(result, ensure_ascii=False).encode("utf-8")
    docs = {sample["sample_id"]: [doc["full_text"] for doc in sample["docs"]]}
    errors = check_bytes(line + b"\n", expected_ids=[sample["sample_id"]], docs_by_id=docs)
    if errors:
        _fail(f"valid sample rejected: {errors}")
    if not any("BOM" in err for err in check_bytes(b"\xef\xbb\xbf" + line)):
        _fail("BOM was accepted")
    bad = dict(result)
    bad["issue_list"] = [dict(bad["issue_list"][0], stance="yes")]
    bad_line = json.dumps(bad, ensure_ascii=False).encode("utf-8")
    if not check_bytes(bad_line):
        _fail("bad stance was accepted")
    short = dict(result)
    short["future_argument"] = result["future_argument"][:-1] or []
    if not check_bytes(json.dumps(short, ensure_ascii=False).encode("utf-8")):
        _fail("length mismatch was accepted")
    doubled = line + b"\n" + line + b"\n"
    if not any("duplicate" in err for err in check_bytes(doubled)):
        _fail("duplicate id was accepted")
    if not any("array" in err for err in check_bytes(b"[\n" + line + b"\n]")):
        _fail("outer array was accepted")
    empty = dict(result)
    empty["issue_list"] = []
    empty["future_argument"] = []
    if not any("empty issue_list" in err for err in check_bytes(json.dumps(empty, ensure_ascii=False).encode("utf-8"))):
        _fail("empty issue_list was accepted")
    segmented = segment_sample(sample)
    fallback = rule_fallback(sample, segmented)
    if fallback is None or len(fallback.issue_list) != 1:
        _fail("rule fallback did not emit one issue")
    public = fallback.public(sample["sample_id"])
    evidence = public["issue_list"][0]["argument_chain"][0]
    if evidence not in sample["docs"][0]["full_text"] and not any(evidence in doc["full_text"] for doc in sample["docs"]):
        _fail("rule fallback evidence left the source")
    fb_errors = check_bytes(
        (json.dumps(public, ensure_ascii=False) + "\n").encode("utf-8"),
        expected_ids=[sample["sample_id"]],
        docs_by_id=docs,
    )
    if fb_errors:
        _fail(f"rule fallback rejected: {fb_errors}")


def check_semifinal() -> None:
    sample = {
        "sample_id": "SEMI",
        "docs": [
            {"doc_type": "记者会", "publish_date": "2024-01-01", "full_text": "甲方支持开放合作。甲方反对加征关税。"},
            {"doc_type": "记者会", "publish_date": "2024-01-02", "full_text": "乙方主张维持现有税率。"},
            {"doc_type": "记者会", "publish_date": "2024-01-03", "full_text": "丙方建议明年评估效果。"},
        ],
    }
    segmented = segment_sample(sample)
    protocol = "\n".join(
        [
            "ISSUE 开放合作安排 ||| support ||| S01",
            "ISSUE 关税政策走向 ||| oppose ||| S02",
            "FUTURE 后续将延续开放合作。",
            "FUTURE 后续将讨论关税。",
        ]
    )
    compiled = compile_protocol(protocol, segmented, max_issues=6, max_evidence=3, fill_missing_future=False)
    if len(compiled.issue_list) != 2:
        _fail("prelim protocol should stay at two issues before expansion")
    expanded = expand_semifinal(compiled, segmented)
    public = expanded.public(sample["sample_id"])
    names = [issue["issue_name"] for issue in public["issue_list"]]
    if names.count("开放合作安排") != 3 or names.count("关税政策走向") != 1:
        _fail(f"semifinal expansion shape is wrong: {names}")
    if len(public["future_argument"]) != len(public["issue_list"]):
        _fail("semifinal futures drifted from issues")
    for issue in public["issue_list"]:
        if set(issue) != {"issue_name", "stance", "argument_chain"}:
            _fail("semifinal public object leaked an internal field")
        for evidence in issue["argument_chain"]:
            if not any(evidence in doc["full_text"] for doc in sample["docs"]):
                _fail("semifinal evidence left the source")
    single = {
        "sample_id": "ONE",
        "docs": [{"doc_type": "政策文件", "publish_date": "2024-01-01", "full_text": "政策要求规范督查。"}],
    }
    one_seg = segment_sample(single)
    one = rule_fallback(single, one_seg)
    if one is None or one.issue_list[0]["stance"] != "support":
        _fail("policy-document fallback should be support")
    press = {
        "sample_id": "PRESS",
        "docs": [{"doc_type": "记者会", "publish_date": "2024-01-01", "full_text": "发言人说明现有安排。"}],
    }
    news = {
        "sample_id": "NEWS",
        "docs": [{"doc_type": "国际新闻报道", "publish_date": "2024-01-01", "full_text": "报道称双方仍在磋商。"}],
    }
    if rule_fallback(press, segment_sample(press)).issue_list[0]["stance"] != "support":
        _fail("press-conference fallback should stay support")
    if rule_fallback(news, segment_sample(news)).issue_list[0]["stance"] != "neutral":
        _fail("news fallback should stay neutral")
    if len(expand_semifinal(one, one_seg).issue_list) != 1:
        _fail("a single speaking party must not be padded to three issues")


def check_loss_mask() -> None:
    from scorer.train import encode_example, resolve_max_length

    picked = (
        resolve_max_length(0, 80),
        resolve_max_length(0, 45),
        resolve_max_length(0, 40),
        resolve_max_length(0, 32),
    )
    if picked != (6144, 6144, 4096, 3072):
        _fail(f"gpu length table changed: {picked}")
    if resolve_max_length(3072, 80) != 3072:
        _fail("an explicit max-length was overridden")

    class Batch:
        def __init__(self, ids: list[int]) -> None:
            self.input_ids = ids

    class Tok:
        eos_token = "<|im_end|>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
                if enable_thinking is False:
                    text += "<think>\n\n</think>\n\n"
            return text

        def __call__(self, text, add_special_tokens=False):
            return Batch([ord(ch) for ch in text])

    class OldTok(Tok):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return text

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "UNIQUE_USER_MARK\n[S01] 甲支持。"},
        {"role": "assistant", "content": "ISSUE 开放合作 ||| support ||| S01"},
    ]
    item = encode_example(Tok(), messages, 4096)
    if item is None:
        _fail("aligned example was dropped")
    supervised = [token for token in item["labels"] if token != -100]
    text = "".join(chr(token) for token in supervised)
    if not text.startswith(messages[-1]["content"]) or "UNIQUE_USER_MARK" in text or "<think>" in text:
        _fail(f"loss mask leaked the prompt: {text[:60]!r}")
    if item["labels"][0] != -100:
        _fail("prompt token was trained")
    if encode_example(Tok(), messages, 8) is not None:
        _fail("an overlong example was kept")
    old = encode_example(OldTok(), messages, 4096)
    if old is None:
        _fail("legacy template dropped the example")
    old_text = "".join(chr(token) for token in old["labels"] if token != -100)
    if not old_text.startswith("ISSUE ") or "<think>" in old_text or "UNIQUE_USER_MARK" in old_text:
        _fail(f"legacy template trained the wrong span: {old_text[:60]!r}")
    from scorer.infer import _strip_think

    if _strip_think("<think>\n推理\n</think>\nISSUE 甲 ||| support ||| S01") != "ISSUE 甲 ||| support ||| S01":
        _fail("a think span was left in the decoded answer")


def check_resume() -> None:
    from scorer.infer import _prepare_resume

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.jsonl"
        path.write_text(
            '{"sample_id": "A", "issue_list": [], "future_argument": []}\n{"sample_id": "B"',
            encoding="utf-8",
        )
        done = _prepare_resume(path, True)
        text = path.read_text(encoding="utf-8")
        if done != {"A"} or "B" in text or not text.endswith("\n"):
            _fail(f"resume did not drop a truncated line: {done} {text!r}")


def check_trim() -> None:
    from scorer.compile import trim_chain, trim_evidence_text

    name = "农产品准入"
    long_sentence = (
        "双方代表团围绕关税减免问题进行了长达十二个小时的闭门磋商，"
        "在农产品准入问题上达成了初步共识，"
        "但服务贸易的开放节奏仍然存在明显分歧，双方同意下周继续会谈。"
    )
    trimmed = trim_evidence_text(name, long_sentence)
    if trimmed not in long_sentence:
        _fail("trim left the source sentence")
    if len(trimmed) > 64:
        _fail(f"trim did not shorten: {len(trimmed)}")
    if "农产品准入" not in trimmed:
        _fail("trim dropped the clause matching the issue name")
    if trim_evidence_text(name, "短句。") != "短句。":
        _fail("short evidence was modified")
    no_hit = trim_evidence_text("量子通信", long_sentence)
    if not no_hit or no_hit not in long_sentence:
        _fail("no-overlap trim broke the substring property")
    chain = trim_chain(name, [long_sentence, "在农产品准入问题上达成了初步共识。", long_sentence])
    if chain != [trimmed, "在农产品准入问题上达成了初步共识。"]:
        _fail(f"trim_chain dedupe/order wrong: {chain}")


def check_attribution_consistency(samples: list[dict]) -> None:
    from scorer.attribute import attribute_detail, attribute_sample

    encoder = OrthogonalEncoder()
    for sample in samples:
        gold = _as_result(sample)
        flipped = copy.deepcopy(gold)
        flipped["issue_list"][0]["stance"] = "oppose" if gold["issue_list"][0]["stance"] != "oppose" else "support"
        scored = score_sample(flipped, gold, encoder, semantic_alpha)
        counts = attribute_sample(flipped, gold, encoder, semantic_alpha)
        detail = attribute_detail(flipped, gold, scored)
        for key in ("matched", "stance_blocked", "low_sim", "pred_extra", "gold_missed"):
            if counts[key] != detail[key]:
                _fail(f"attribute_detail diverged on {key}: {counts} vs {detail}")


def check_postprocess_ops(sample: dict) -> None:
    from scorer.postprocess import recover_ids, rerank_issue

    segmented = segment_sample(sample)
    issue = sample["issue_list"][0]
    ids = align_chain(sample, segmented, list(issue["argument_chain"]))
    if not ids:
        return
    chain = [segmented.by_id[sid].text for sid in ids]
    if recover_ids(chain, segmented) != ids:
        _fail("recover_ids round trip failed")
    reranked = rerank_issue(issue["issue_name"], ids, segmented, OrthogonalEncoder(), len(ids))
    if len(reranked) != len(ids):
        _fail("rerank changed the evidence count")
    for text in reranked:
        if not any(text in doc["full_text"] for doc in sample["docs"]):
            _fail("rerank left the source text")


def check_postprocess_preflight(sample: dict) -> None:
    from scorer.postprocess import _neighbourhood, substring_ratio

    segmented = segment_sample(sample)
    docs_by_id = {sample["sample_id"]: [doc["full_text"] for doc in sample["docs"]]}
    ratio, hits, total = substring_ratio([_as_result(sample)], docs_by_id)
    if total == 0 or ratio != 1.0:
        _fail(f"preflight rejected a valid result: {hits}/{total}")
    foreign = [
        {
            "sample_id": sample["sample_id"],
            "issue_list": [{"issue_name": "x", "stance": "support", "argument_chain": ["这句证据不属于该文档，用于错配检测。"]}],
        }
    ]
    ratio, _hits, _total = substring_ratio(foreign, docs_by_id)
    if ratio != 0.0:
        _fail("preflight accepted foreign evidence")

    ids = align_chain(sample, segmented, list(sample["issue_list"][0]["argument_chain"]))
    if not ids:
        return
    if _neighbourhood(ids, segmented, 0) != ids:
        _fail("radius 0 changed the model's own sentences")
    pool = _neighbourhood(ids, segmented, 1)
    nums = {int(sid[1:]) for sid in ids}
    if not set(ids) <= set(pool):
        _fail("neighbourhood dropped a model sentence")
    for sid in pool:
        if min(abs(int(sid[1:]) - num) for num in nums) > 1:
            _fail(f"neighbourhood leaked {sid}")


def main() -> int:
    check_loss_mask()
    check_rouge()
    check_matching()
    check_threshold()
    train = load_split("train")
    val = load_split("val")
    check_identity(val)
    # A 5-issue sample whose futures actually differ, so reversing them moves score.
    host = next(
        sample
        for sample in val
        if len(sample["issue_list"]) >= 4 and len(set(sample["future_argument"])) == len(sample["future_argument"])
    )
    check_perturbations(host)
    check_submit(host)
    check_semifinal()
    check_resume()
    check_trim()
    check_attribution_consistency(val[:5])
    check_postprocess_ops(host)
    check_postprocess_preflight(host)
    check_segments(train)
    split = write_dev_split(train)
    print(
        f"dev40={split['dev']} fit={split['fit']} oppose_docs={split['oppose_docs']} "
        f"types={split['by_type']}"
    )
    if split["oppose_docs"] < 8 or split["dev"] + split["fit"] != len(train):
        _fail(f"bad dev split {split}")
    print("selfcheck ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"SELFCHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
