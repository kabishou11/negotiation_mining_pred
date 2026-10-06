"""Second-team end-to-end inference: one base Qwen3-32B call per document.

No training, no adapter, no retrieval, and no code shared with the first
team's pipeline: the raw document goes in inside an English XML wrapper and
the model returns the whole JSON object at once. Decoding is temperature
0.2 / top_p 0.9 / seed 42 (per-sample seeded so resume stays stable), a
repair loop fixes the JSON, and every evidence string is forced back onto an
exact substring of the source document (difflib longest-block fallback), so
the output always satisfies the submission checker.

Run from teamA_end2end/:

    python3 src/infer_e2e.py --split test --model /path/to/Qwen3-32B \
        --output result.jsonl
    python3 src/infer_e2e.py --split test --limit 2 --dry-run

Local scoring of a val output uses the shared judge for measurement only:

    python3 -m scorer.evaluate result_val.jsonl --split val
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TEAM_ROOT = SRC_DIR.parent
DATA_DIR = TEAM_ROOT.parent / "data" / "prelim"
PROMPT_PATH = TEAM_ROOT / "prompts" / "extract_en.xml"
DEFAULT_CONFIG = TEAM_ROOT / "configs" / "decode.json"

SYSTEM_TEXT = "You are an extraction engine. Output valid JSON only."
_STANCE_MAP = {
    "支持": "support",
    "反对": "oppose",
    "中立": "neutral",
    "中性": "neutral",
}
_STANCES = {"support", "oppose", "neutral"}
_SENT_SPLIT = re.compile(r"[^。！？!?]+[。！？!?]?")
_SUPPORT_TYPES = {"联合声明", "政策文件", "记者会"}


def load_docs(split: str) -> list[dict]:
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    path = next(DATA_DIR.glob(f"**/{split}.jsonl"), None)
    if path is None:
        raise SystemExit(f"dataset not found under {DATA_DIR}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


_TRAIN_CACHE: list[dict] | None = None
_EXAMPLE_CACHE: list[str] | None = None


def _train_rows() -> list[dict]:
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = load_docs("train")
    return _TRAIN_CACHE


def few_shot_block(k: int, prefix_chars: int = 500, scan_budget: int = 60) -> str:
    """k worked examples drawn deterministically from the official train split.

    Legal and disclosable: train data with its gold annotations, no other
    model involved. The document is truncated at a sentence boundary and the
    output keeps only the gold issues whose every evidence string survives
    the truncation, so the example never shows evidence outside the shown
    text — copying from thin air is exactly the behaviour to avoid teaching.

    Among the first `scan_budget` qualifying rows, the picker prefers the
    most stance-diverse example. The scan happens to surface all-support
    rows first, and an all-support example would teach exactly the
    support-bias the leaderboard punishes.
    """
    global _EXAMPLE_CACHE
    if _EXAMPLE_CACHE is None:
        qualifying: list[tuple[int, int, str]] = []
        for order, row in enumerate(_train_rows()):
            docs = row.get("docs") or []
            if not docs:
                continue
            full = str(docs[0].get("full_text") or "")
            issues = row.get("issue_list") or []
            futures = row.get("future_argument") or []
            if not 3 <= len(issues) <= 5 or len(futures) != len(issues) or len(full) <= prefix_chars:
                continue
            cut = full.rfind("。", 0, prefix_chars)
            if cut == -1:
                continue
            prefix = full[: cut + 1]
            kept_issues, kept_futures = [], []
            for issue, future in zip(issues, futures):
                chain = [str(e) for e in issue.get("argument_chain") or []]
                if chain and all(e in prefix for e in chain):
                    kept_issues.append(
                        {"issue_name": issue["issue_name"], "stance": issue["stance"], "argument_chain": chain}
                    )
                    kept_futures.append(str(future))
            if len(kept_issues) < 3:
                continue
            diversity = len({issue["stance"] for issue in kept_issues})
            payload = json.dumps(
                {"issue_list": kept_issues, "future_argument": kept_futures}, ensure_ascii=False
            )
            block = (
                "<example>\n<document type=\"{t}\" date=\"{d}\">\n{p}\n</document>\n<output>\n{o}\n</output>\n</example>".format(
                    t=docs[0].get("doc_type") or "unknown",
                    d=docs[0].get("publish_date") or "unknown",
                    p=prefix,
                    o=payload,
                )
            )
            qualifying.append((diversity, order, block))
            if len(qualifying) >= scan_budget:
                break
        qualifying.sort(key=lambda item: (-item[0], item[1]))
        _EXAMPLE_CACHE = [block for _d, _o, block in qualifying[:k]]
    return "\n\n".join(_EXAMPLE_CACHE[:k])


def user_prompt(sample: dict, cfg: dict) -> str:
    """Build the XML user turn from a full sample row.

    The dataset row carries the document inside ``docs``; the prelim split
    always has exactly one. Passing a bare doc here would silently render an
    empty <document>, which the smoke test exists to catch.
    """
    docs = sample.get("docs") or [{}]
    examples = few_shot_block(int(cfg.get("few_shot", 0)))
    if examples:
        examples += "\n\nThe example above is from the training split. Treat it as a style reference only; its content has nothing to do with the document below.\n"
    template = PROMPT_PATH.read_text(encoding="utf-8")
    text = "\n".join(str(d.get("full_text") or "") for d in docs)
    return (
        template.replace("__EXAMPLES__", examples)
        .replace("__DOC_TYPE__", docs[0].get("doc_type") or "unknown")
        .replace("__PUBLISH_DATE__", docs[0].get("publish_date") or "unknown")
        .replace("__FULL_TEXT__", text)
    )


def build_messages(sample: dict, cfg: dict, suffix: str = "") -> list[dict[str, str]]:
    user = user_prompt(sample, cfg)
    if suffix:
        user += "\n" + suffix
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user},
    ]


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _balanced_object(text: str) -> str | None:
    """Span of the JSON object starting at the first ``{``.

    Only the first brace counts. When the model's top-level object is
    truncated, a later inner issue object is still balanced; accepting it
    would surface a bare issue as the whole result. Truncated output must
    fall through to `_close_json` instead.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _close_json(fragment: str) -> str | None:
    """Close a JSON object truncated mid-generation (max_new_tokens hit).

    Pass one finds the last complete element (a closing quote or bracket).
    Pass two recomputes the bracket stack over the head alone — pushes that
    happened after the cut point must not leak into the closers. A dangling
    key whose value never arrived is dropped, keyed on whether the stack top
    at the cut is an object (the string was a key) or an array (it was an
    element). A truncated tail then costs at most the last issue instead of
    poisoning the whole object.
    """
    start = fragment.find("{")
    if start == -1:
        return None
    frag = fragment[start:]
    in_string = False
    escape = False
    last_complete = -1
    for i, ch in enumerate(frag):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_complete = i
            continue
        if ch == '"':
            in_string = True
        elif ch in "}]":
            last_complete = i
    if last_complete < 0:
        return None
    head = frag[: last_complete + 1]
    rest = frag[last_complete + 1 :].lstrip()

    stack: list[str] = []
    in_string = False
    escape = False
    for ch in head:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
    if not stack:
        return None

    drop_key = rest.startswith(":")
    if not rest and frag[last_complete] == '"' and stack[-1] == "}":
        opening = head.rfind('"', 0, len(head) - 1)
        before = head[:opening].rstrip() if opening != -1 else ""
        drop_key = not before.endswith(":")
    if drop_key:
        head = re.sub(r',?\s*"[^"]*"\s*$', "", head)
    return head + "".join(reversed(stack))


def extract_json(raw: str) -> dict | None:
    text = _strip_think(raw)
    blob = _balanced_object(text)
    candidates: list[str] = []
    if blob is not None:
        candidates.append(blob)
        candidates.append(re.sub(r",\s*([}\]])", r"\1", blob))
    else:
        closed = _close_json(text)
        if closed is not None:
            candidates.append(closed)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def force_substring(evidence: str, full_text: str, cfg: dict) -> str:
    evidence = evidence.strip()
    if len(evidence) < cfg["min_evidence_chars"]:
        return ""
    if len(evidence) > cfg["max_evidence_chars"]:
        evidence = evidence[: cfg["max_evidence_chars"]]
    if evidence in full_text:
        return evidence
    matcher = difflib.SequenceMatcher(None, evidence, full_text, autojunk=False)
    _a, b, size = matcher.find_longest_match(0, len(evidence), 0, len(full_text))
    if size >= max(10, len(evidence) // 2):
        return full_text[b : b + size]
    return ""


def fallback_future(issue: dict) -> str:
    snippet = str(issue["argument_chain"][0])[:24]
    return f"接下来双方将围绕「{snippet}」进一步落实相关安排。"


def trim_to_focus(text: str, name: str, target: int = 53, max_len: int = 80) -> str:
    """Cut a long recovered evidence down toward the gold evidence length.

    Gold chains average ~53 characters; a 150-character verbatim block
    dilutes the argument cosine exactly like the first team's whole-sentence
    problem did. Own algorithm on purpose: each clause window is scored by
    the issue-name bigrams it keeps plus a length prior toward `target`,
    which is a different formulation from the other team's window search.
    """
    text = text.strip()
    if len(text) <= max_len:
        return text
    clauses: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in "，。；！？、":
            if text[start : i + 1].strip():
                clauses.append((start, i + 1))
            start = i + 1
    if start < len(text):
        clauses.append((start, len(text)))
    if len(clauses) <= 1:
        return text
    grams = _name_grams(name)
    best: str | None = None
    best_key: tuple[int, int] | None = None
    for a in range(len(clauses)):
        for b in range(a, len(clauses)):
            piece = text[clauses[a][0] : clauses[b][1]]
            if len(piece) > max_len:
                break
            hits = sum(1 for gram in grams if gram in piece)
            key = (hits, -abs(len(piece) - target))
            if best_key is None or key > best_key:
                best_key, best = key, piece
    trimmed = (best or text).strip(" ，。；、,; ")
    return trimmed or text


def normalize(obj: dict, docs: list[dict], cfg: dict) -> dict:
    """Validate one parsed JSON object against the source documents.

    Evidence forcing runs per document: a difflib block recovered against
    the joined text could cross a document boundary and fail the per-doc
    substring check in the submission validator.
    """
    texts = [str(d.get("full_text") or "") for d in docs]
    issues: list[dict] = []
    futures_in = obj.get("future_argument")
    for item in (obj.get("issue_list") or [])[: cfg["max_issues"]]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("issue_name") or "").strip()
        stance = _STANCE_MAP.get(str(item.get("stance") or "").strip().lower())
        if stance is None:
            stance = str(item.get("stance") or "").strip().lower()
        if stance not in _STANCES or not name:
            continue
        chain: list[str] = []
        for raw in (item.get("argument_chain") or [])[: cfg["max_evidence"]]:
            for text in texts:
                fixed = force_substring(str(raw), text, cfg)
                if fixed:
                    break
            if fixed and len(fixed) > int(cfg.get("focus_len", 80)):
                fixed = trim_to_focus(fixed, name)
            if fixed and fixed not in chain:
                chain.append(fixed)
        if not chain:
            continue
        issues.append({"issue_name": name, "stance": stance, "argument_chain": chain})
    futures: list[str] = []
    for index, issue in enumerate(issues):
        future = str(futures_in[index]).strip() if isinstance(futures_in, list) and index < len(futures_in) else ""
        future = future or fallback_future(issue)
        # Gold futures touch their issue name 97.9% of the time (measured on
        # train); a name-less future is off-distribution, so point it.
        if cfg.get("future_name_check") and not _touches_name(issue["issue_name"], future):
            future = f"关于{issue['issue_name']}，{future}"
        futures.append(future)
    return {"issue_list": issues, "future_argument": futures}


def _touches_name(name: str, text: str) -> bool:
    grams = _name_grams(name)
    return bool(grams) and any(gram in text for gram in grams)


def fallback_result(doc: dict, cfg: dict) -> dict:
    """Keep the sample submittable when no JSON survives the repair loop."""
    full_text = str((doc.get("docs") or [{}])[0].get("full_text") or "")
    sentences = [s.strip() for s in _SENT_SPLIT.findall(full_text) if len(s.strip()) >= 12]
    evidence = max(sentences, key=len)[: cfg["max_evidence_chars"]] if sentences else full_text[:60]
    doc_type = (doc.get("docs") or [{}])[0].get("doc_type")
    stance = "support" if doc_type in _SUPPORT_TYPES else "neutral"
    issue = {"issue_name": "核心议题", "stance": stance, "argument_chain": [evidence]}
    return {"issue_list": [issue], "future_argument": [fallback_future(issue)]}


def generate(model, tokenizer, messages: list[dict[str, str]], cfg: dict) -> str:
    import torch
    from transformers import GenerationConfig

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    config = GenerationConfig(
        do_sample=True,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_new_tokens=cfg["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=config)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def load_model(model_path: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("Qwen3-32B inference expects a CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        device_map={"": 0} if device_map == "single" else "auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _name_grams(name: str) -> set[str]:
    flat = "".join(str(name).split())
    if len(flat) < 2:
        return {flat} if flat else set()
    return {flat[i : i + 2] for i in range(len(flat) - 1)}


def _gram_cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def merge_candidates(candidates: list[dict], cfg: dict) -> dict | None:
    """Fuse several normalized predictions of the same sample into one.

    The rules allow fusing inference results of the one allowed base, and a
    sampled base model profits from voting more than a greedy adapter does.
    Issues cluster greedily by name similarity; the cluster keeps the name
    surface that repeats most (ties: first seen), the majority stance, the
    evidence union ordered by how often each string was drawn, and the
    future of its earliest member. Deterministic for a given candidate list.
    """
    if not candidates:
        return None
    clusters: list[list[tuple[int, dict, str]]] = []
    for cand_idx, cand in enumerate(candidates):
        cand_futures = cand.get("future_argument") or []
        for issue_idx, issue in enumerate(cand["issue_list"]):
            future = str(cand_futures[issue_idx]) if issue_idx < len(cand_futures) else ""
            grams = _name_grams(issue["issue_name"])
            best: list[tuple[int, dict, str]] | None = None
            best_sim = 0.0
            for cluster in clusters:
                if any(c == cand_idx for c, _i, _f in cluster):
                    continue
                sim = _gram_cosine(grams, _name_grams(cluster[0][1]["issue_name"]))
                if sim > best_sim:
                    best, best_sim = cluster, sim
            if best is not None and best_sim >= 0.6:
                best.append((cand_idx, issue, future))
            else:
                clusters.append([(cand_idx, issue, future)])

    issues: list[dict] = []
    futures: list[str] = []
    for cluster in clusters:
        # Counter.most_common is stable on insertion order, so ties resolve
        # to the earliest candidate deterministically.
        name = Counter(issue["issue_name"] for _c, issue, _f in cluster).most_common(1)[0][0]
        stance = Counter(issue["stance"] for _c, issue, _f in cluster).most_common(1)[0][0]
        evidence_count: dict[str, int] = {}
        evidence_order: dict[str, int] = {}
        for _c, issue, _f in cluster:
            for ev in issue["argument_chain"]:
                evidence_count[ev] = evidence_count.get(ev, 0) + 1
                evidence_order.setdefault(ev, len(evidence_order))
        chain = [
            ev
            for ev, _ in sorted(evidence_count.items(), key=lambda kv: (-kv[1], evidence_order[kv[0]]))
        ][: int(cfg["max_evidence"])]
        if not chain:
            continue
        future = cluster[0][2]
        issues.append({"issue_name": name, "stance": stance, "argument_chain": chain})
        futures.append(future or fallback_future({"argument_chain": chain}))
        if len(issues) >= int(cfg["max_issues"]):
            break
    if not issues:
        return None
    return {"issue_list": issues, "future_argument": futures}


def predict(doc: dict, model, tokenizer, cfg: dict, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    consistency = int(cfg.get("consistency", 0))
    if consistency > 1:
        # Voting replaces the JSON-retry loop: each draw is seeded apart, and
        # unparseable draws simply do not vote. min_issues does not apply —
        # the merge already maximises coverage across draws.
        candidates: list[dict] = []
        for draw in range(consistency):
            torch.manual_seed(seed + 1000 * draw)
            obj = extract_json(generate(model, tokenizer, build_messages(doc, cfg), cfg))
            if obj is None or not isinstance(obj.get("issue_list"), list):
                continue
            cand = normalize(obj, doc.get("docs") or [], cfg)
            if cand["issue_list"]:
                candidates.append(cand)
        result = merge_candidates(candidates, cfg) if len(candidates) >= 2 else (candidates[0] if candidates else None)
        if result is None or not result["issue_list"]:
            return fallback_result(doc, cfg)
        return result

    obj = None
    for attempt in range(cfg["retries"] + 1):
        suffix = "" if attempt == 0 else (
            "Your previous reply was not valid JSON. Output ONLY the JSON object, nothing else."
        )
        obj = extract_json(generate(model, tokenizer, build_messages(doc, cfg, suffix), cfg))
        if obj is not None and isinstance(obj.get("issue_list"), list):
            break
    result = normalize(obj, doc.get("docs") or [], cfg) if obj is not None else None
    # A valid JSON with too few issues gets one more generation. A padded
    # issue that matches nothing costs precision, so the retry is adopted
    # only when it actually finds more.
    min_issues = int(cfg.get("min_issues", 0))
    if result is not None and min_issues and len(result["issue_list"]) < min_issues:
        suffix = (
            f"Your previous reply contained only {len(result['issue_list'])} issue(s). "
            "Cover every distinct aspect of the document: 4 to 5 issues."
        )
        retry = extract_json(generate(model, tokenizer, build_messages(doc, cfg, suffix), cfg))
        if retry is not None:
            retried = normalize(retry, doc.get("docs") or [], cfg)
            if len(retried["issue_list"]) > len(result["issue_list"]):
                result = retried
    if result is None or not result["issue_list"]:
        return fallback_result(doc, cfg)
    return result


def dry_run(docs: list[dict], limit: int, cfg: dict) -> None:
    for doc in docs[:limit]:
        print(f"===== {doc['sample_id']} =====")
        print(build_messages(doc, cfg)[1]["content"][:1500])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model", default="", help="path to the local Qwen3 32B base weights")
    parser.add_argument("--output", default="result.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip sample_ids already present in --output",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    docs = load_docs(args.split)
    if args.limit:
        docs = docs[: args.limit]
    if args.dry_run:
        dry_run(docs, limit=max(1, args.limit or 1), cfg=cfg)
        return
    if not args.model:
        raise SystemExit("pass --model, or --dry-run to preview the prompt")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    model, tokenizer = load_model(args.model, args.device_map)
    from tqdm import tqdm

    written = 0
    with output.open("a" if done else "w", encoding="utf-8", newline="\n") as handle:
        for doc in tqdm(docs, desc="e2e", dynamic_ncols=True, mininterval=1.0):
            sample_id = doc["sample_id"]
            if sample_id in done:
                continue
            seed = cfg["seed"] + int(re.sub(r"\D", "", sample_id) or 0)
            try:
                result = predict(doc, model, tokenizer, cfg, seed)
            except Exception as exc:  # keep the run alive; log and fall back
                print(f"{sample_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                result = fallback_result(doc, cfg)
            handle.write(json.dumps({"sample_id": sample_id, **result}, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    print(f"{written} sample predictions appended/created in {output}")


if __name__ == "__main__":
    main()
