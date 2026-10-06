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


def user_prompt(sample: dict) -> str:
    """Build the XML user turn from a full sample row.

    The dataset row carries the document inside ``docs``; the prelim split
    always has exactly one. Passing a bare doc here would silently render an
    empty <document>, which the smoke test exists to catch.
    """
    docs = sample.get("docs") or [{}]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    text = "\n".join(str(d.get("full_text") or "") for d in docs)
    return (
        template.replace("__DOC_TYPE__", docs[0].get("doc_type") or "unknown")
        .replace("__PUBLISH_DATE__", docs[0].get("publish_date") or "unknown")
        .replace("__FULL_TEXT__", text)
    )


def build_messages(doc: dict, suffix: str = "") -> list[dict[str, str]]:
    user = user_prompt(doc)
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
            if fixed and fixed not in chain:
                chain.append(fixed)
        if not chain:
            continue
        issues.append({"issue_name": name, "stance": stance, "argument_chain": chain})
    futures: list[str] = []
    for index, issue in enumerate(issues):
        future = str(futures_in[index]).strip() if isinstance(futures_in, list) and index < len(futures_in) else ""
        futures.append(future or fallback_future(issue))
    return {"issue_list": issues, "future_argument": futures}


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


def predict(doc: dict, model, tokenizer, cfg: dict, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    messages = build_messages(doc)
    obj = None
    for attempt in range(cfg["retries"] + 1):
        suffix = "" if attempt == 0 else (
            "Your previous reply was not valid JSON. Output ONLY the JSON object, nothing else."
        )
        obj = extract_json(generate(model, tokenizer, build_messages(doc, suffix), cfg))
        if obj is not None and isinstance(obj.get("issue_list"), list):
            break
    if obj is None:
        return fallback_result(doc, cfg)
    result = normalize(obj, doc.get("docs") or [], cfg)
    if not result["issue_list"]:
        return fallback_result(doc, cfg)
    return result


def dry_run(docs: list[dict], limit: int) -> None:
    for doc in docs[:limit]:
        print(f"===== {doc['sample_id']} =====")
        print(build_messages(doc)[1]["content"][:1200])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model", default="", help="local Qwen3-32B directory")
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
        dry_run(docs, limit=max(1, args.limit or 1))
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
    print(f"wrote {written} new lines to {output}")


if __name__ == "__main__":
    main()
