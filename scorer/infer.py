"""Greedy Qwen3-32B inference into `result.jsonl`.

Generation is deterministic: `do_sample=False`, thinking disabled. This module
is the submission path. It does not download weights and it does not call any
model other than the checkpoint you pass.

Optional second passes, each flag-gated so the plain path stays identical:

- `--min-issues N`: when extraction settles below N issues, one retry with a
  prompt that says the count is low. Padding with a wrong issue costs more
  than the N=max miss, so the retry is kept only when it finds more.
- `--stance-check`: one focused stance-verification call per issue. A strict
  parse keeps the original stance on anything but a clean STANCE line.
- `--ids-file`: restrict inference to those sample_ids (dev40 checkpoint
  picking).

Evidence-side polish (rerank/trim) lives in `scorer.postprocess`, which can
also transform files that were already generated.

Dry-run (`--dry-run`) prints the extraction prompt and the per-issue future
prompt for the first samples without loading weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scorer.compile import _fallback_future, compile_protocol, expand_semifinal, rule_fallback
from scorer.datautil import load_split
from scorer.prompt import extraction_messages, future_messages, render_prompt, stance_check_messages
from scorer.segment import segment_sample


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    return render_prompt(tokenizer, messages)


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def generate_text(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    import torch
    from transformers import GenerationConfig

    prompt = render_chat(tokenizer, messages)
    # add_special_tokens=False matches the training-side encode in
    # scorer.train._ids. Qwen3 adds nothing either way, but a tokenizer that
    # prepends a BOS here would silently break train/infer prefix equality.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = _input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # A fresh config, not the checkpoint's. Qwen3 ships do_sample=True and
    # some transformers builds put that back when temperature is also set.
    # temperature>0 enables seeded sampling for the fusion experiments: the
    # sampled outputs are merged with the greedy run by scorer.merge_results,
    # which the rules explicitly allow for the one permitted base model.
    decode = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        config = GenerationConfig(do_sample=True, temperature=temperature, top_p=top_p, **decode)
    else:
        config = GenerationConfig(do_sample=False, **decode)
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=config)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return _strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True))


def load_model(model_path: str, adapter_path: str = "", device_map: str = "single"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("Qwen3-32B inference expects a CUDA GPU. Refusing to load it on CPU.")
    placement = {"": 0} if device_map == "single" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        device_map=placement,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _compile_extraction(
    model,
    tokenizer,
    sample,
    segmented,
    max_new_tokens: int,
    extra: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    messages = extraction_messages(sample, segmented)
    if extra:
        messages = [
            messages[0],
            {"role": "user", "content": messages[1]["content"] + extra},
        ]
    raw = generate_text(model, tokenizer, messages, max_new_tokens, temperature, top_p)
    return compile_protocol(raw, segmented, max_issues=6, max_evidence=3, fill_missing_future=False)


def _fill_futures(model, tokenizer, compiled, temperature: float = 0.0, top_p: float = 1.0) -> None:
    futures: list[str] = []
    for issue in compiled.issue_list:
        raw = generate_text(model, tokenizer, future_messages(issue), 160, temperature, top_p)
        line = ""
        for candidate in raw.splitlines():
            candidate = candidate.strip()
            if candidate.startswith("FUTURE"):
                line = candidate[len("FUTURE") :].strip()
                break
        if not line:
            line = _fallback_future(issue)
        futures.append(line)
    compiled.future_argument = futures


def _calibrate_stances(model, tokenizer, compiled, sample, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """Second-pass stance verification. Strict parse; no parse, no change."""
    docs = sorted(sample.get("docs") or [], key=lambda doc: doc.get("publish_date") or "")
    doc_type = docs[0].get("doc_type") or "" if docs else ""
    changed = 0
    for issue in compiled.issue_list:
        try:
            raw = generate_text(model, tokenizer, stance_check_messages(issue, doc_type), 12, temperature, top_p)
        except Exception:
            continue
        match = re.search(r"STANCE\s+(support|oppose|neutral)", raw, re.IGNORECASE)
        if match and match.group(1).lower() != issue["stance"]:
            issue["stance"] = match.group(1).lower()
            changed += 1
    return changed


def predict_sample(
    model,
    tokenizer,
    sample: dict,
    max_new_tokens: int,
    mode: str = "prelim",
    min_issues: int = 0,
    stance_check: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> dict:
    segmented = segment_sample(sample)
    if temperature and temperature > 0:
        # per-sample seeding keeps a sampled run reproducible under resume
        import torch

        torch.manual_seed(seed)
    compiled = _compile_extraction(
        model, tokenizer, sample, segmented, max_new_tokens, "", temperature, top_p
    )
    # Greedy decoding repeats itself, so a retry only helps if the prompt changes.
    if not compiled.issue_list:
        compiled = _compile_extraction(
            model,
            tokenizer,
            sample,
            segmented,
            max_new_tokens,
            "\n上一次没有输出 ISSUE 行。请至少给出主议题和它的句子编号。",
            temperature,
            top_p,
        )
    # A padded issue that matches nothing costs precision, so a low-count
    # retry is adopted only when it actually finds more sides.
    if min_issues and 0 < len(compiled.issue_list) < min_issues:
        retry = _compile_extraction(
            model,
            tokenizer,
            sample,
            segmented,
            max_new_tokens,
            f"\n上一次只给出 {len(compiled.issue_list)} 个议题，偏少。请重新通读全文，把明显不同的侧面补全，输出 4 到 5 个 ISSUE 行。",
            temperature,
            top_p,
        )
        if len(retry.issue_list) > len(compiled.issue_list):
            compiled = retry
    if stance_check and compiled.issue_list:
        _calibrate_stances(model, tokenizer, compiled, sample, temperature, top_p)
    if not compiled.issue_list:
        fallback = rule_fallback(sample, segmented)
        if fallback is None:
            return {
                "sample_id": sample["sample_id"],
                "issue_list": [],
                "future_argument": [],
            }
        print(f"rule fallback {sample.get('sample_id')}", file=sys.stderr)
        compiled = fallback
    else:
        _fill_futures(model, tokenizer, compiled, temperature, top_p)
    if mode == "semifinal":
        compiled = expand_semifinal(compiled, segmented)
    return compiled.public(sample["sample_id"])


def dry_run(samples: list[dict], limit: int) -> None:
    for sample in samples[:limit]:
        segmented = segment_sample(sample)
        messages = extraction_messages(sample, segmented)
        print(f"===== {sample['sample_id']} sentences={len(segmented.sentences)} =====")
        print(messages[1]["content"][:1500])
        print("----- future prompt (format only, gold issue 0) -----")
        if sample.get("issue_list"):
            preview = dict(sample["issue_list"][0])
            # Show the real prompt shape with source sentences, not gold substrings.
            if segmented.sentences:
                preview = {
                    "issue_name": preview["issue_name"],
                    "stance": preview["stance"],
                    "argument_chain": [segmented.sentences[0].text],
                }
            print(future_messages(preview)[1]["content"][:800])


def _log_failure(path: Path, sample_id: str, attempt: int, error: str, recovered: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"sample_id": sample_id, "attempt": attempt, "error": error, "recovered": recovered}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_resume(output: Path, resume: bool) -> set[str]:
    """Ids already written. A truncated last line is dropped so append stays valid JSON Lines."""
    if not resume or not output.is_file():
        return set()
    raw = output.read_bytes()
    if raw and not raw.endswith(b"\n"):
        head, sep, tail = raw.rpartition(b"\n")
        try:
            json.loads(tail.decode("utf-8"))
            output.write_bytes(raw + b"\n")
        except (UnicodeDecodeError, json.JSONDecodeError):
            output.write_bytes(head + sep)
    done: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["sample_id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def write_predictions(
    samples: list[dict],
    model,
    tokenizer,
    output: Path,
    max_new_tokens: int,
    *,
    resume: bool,
    mode: str,
    min_issues: int = 0,
    stance_check: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _prepare_resume(output, resume)
    failure_log = Path(str(output) + ".failures.jsonl")
    file_mode = "a" if resume and output.is_file() else "w"
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    from tqdm import tqdm

    written = 0
    with output.open(file_mode, encoding="utf-8", newline="\n") as handle:
        for sample in tqdm(pending, desc="infer", dynamic_ncols=True, mininterval=1.0):
            sample_id = sample["sample_id"]
            result = None
            last_error = ""
            for attempt in (1, 2):
                try:
                    result = predict_sample(
                        model,
                        tokenizer,
                        sample,
                        max_new_tokens,
                        mode,
                        min_issues=min_issues,
                        stance_check=stance_check,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed + int(re.sub(r"\D", "", sample_id) or 0),
                    )
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    _log_failure(failure_log, sample_id, attempt, last_error, recovered=False)
            if result is None:
                segmented = segment_sample(sample)
                fallback = rule_fallback(sample, segmented)
                if fallback is not None and mode == "semifinal":
                    fallback = expand_semifinal(fallback, segmented)
                if fallback is None:
                    print(f"dropped {sample_id}: {last_error}", file=sys.stderr)
                    continue
                result = fallback.public(sample_id)
                _log_failure(failure_log, sample_id, 2, last_error, recovered=True)
                print(f"rule fallback after error {sample_id}", file=sys.stderr)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    print(f"wrote {written} new lines to {output} skipped {len(done)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split")
    parser.add_argument("--model", default="", help="local Qwen3-32B directory")
    parser.add_argument("--adapter", default="", help="LoRA directory written by scorer.train")
    parser.add_argument("--output", default="", help="result.jsonl path")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--mode", choices=["prelim", "semifinal"], default="prelim")
    parser.add_argument("--ids-file", default="", help="infer only these sample_ids, e.g. data/dev40_ids.txt")
    parser.add_argument(
        "--min-issues",
        type=int,
        default=0,
        help="retry extraction when fewer than this many issues come back; 0 disables the retry",
    )
    parser.add_argument(
        "--stance-check",
        action="store_true",
        help="one focused stance-verification call per issue after extraction",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (submission default); >0 enables seeded sampling so the "
        "run can be fused with the greedy one via scorer.merge_results",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus cutoff when sampling")
    parser.add_argument("--seed", type=int, default=0, help="base seed; per-sample seed adds the sample number")
    parser.add_argument(
        "--device-map",
        choices=["single", "auto"],
        default="single",
        help="single pins the 4-bit model to cuda:0; auto shards when one card cannot hold weights plus KV",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip sample_ids already present in --output (default). --no-resume truncates the file",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    samples = load_split(args.split)
    if args.ids_file:
        wanted = {
            line.strip()
            for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        samples = [sample for sample in samples if sample["sample_id"] in wanted]
        if not samples:
            raise SystemExit(f"no samples matched {args.ids_file}")
    if args.limit:
        samples = samples[: args.limit]
    if args.dry_run:
        dry_run(samples, limit=max(1, args.limit or 1))
        return
    if not args.model:
        raise SystemExit("pass --model, or --dry-run to preview prompts without weights")
    model, tokenizer = load_model(args.model, args.adapter, args.device_map)
    output = Path(args.output or f"result_{args.split}.jsonl")
    write_predictions(
        samples,
        model,
        tokenizer,
        output,
        args.max_new_tokens,
        resume=args.resume,
        mode=args.mode,
        min_issues=args.min_issues,
        stance_check=args.stance_check,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
