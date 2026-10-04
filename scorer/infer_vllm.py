"""Batched vllm inference: same protocol as scorer.infer, much faster decode.

Requires: vllm installed in env, plus a LoRA adapter directory from scorer.train.
Quantizes the base with bitsandbytes to fit one 49GB card, same 4-bit/nf4 feel
as the training path. Greedy, thinking disabled, thinking template identical to
scorer.train/scorer.infer via the same render_prompt call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scorer.compile import _fallback_future, compile_protocol, expand_semifinal, rule_fallback
from scorer.datautil import load_split
from scorer.infer import _prepare_resume, _strip_think  # reuse resume trimming
from scorer.prompt import extraction_messages, future_messages, render_prompt
from scorer.segment import segment_sample


def _render(tokenizer, messages):
    return render_prompt(tokenizer, messages)


def run(split: str, model: str, adapter: str, output: str, limit: int,
        max_new_tokens: int, resume: bool, mode: str) -> None:
    import os
    os.environ["VLLM_USE_V1"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    samples = load_split(split)
    if limit:
        samples = samples[:limit]
    done = _prepare_resume(Path(output), resume)
    todo = [s for s in samples if s["sample_id"] not in done]
    seg_map = {s["sample_id"]: segment_sample(s) for s in todo}
    print(f"[vllm] todo={len(todo)} done={len(done)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        quantization="bitsandbytes",
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=6144,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
        disable_log_stats=True,
    )
    lora = LoRARequest("ck", 1, adapter) if adapter else None
    extract_sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    future_sp = SamplingParams(temperature=0.0, max_tokens=160)

    def gen(prompts, sp):
        return llm.generate(prompts, sp, lora_request=lora)

    results = {}
    prompts = [_render(tokenizer, extraction_messages(s, seg_map[s["sample_id"]])) for s in todo]
    outs = gen(prompts, extract_sp)
    retry = []
    for s, o in zip(todo, outs):
        raw = _strip_think(o.outputs[0].text.strip())
        compiled = compile_protocol(raw, seg_map[s["sample_id"]], max_issues=6, max_evidence=3, fill_missing_future=False)
        if compiled.issue_list:
            results[s["sample_id"]] = compiled
        else:
            retry.append((s, raw))
    if retry:
        print(f"[vllm] extract retry {len(retry)}", flush=True)
        retry_prompts = []
        for s, _ in retry:
            msgs = extraction_messages(s, seg_map[s["sample_id"]])
            msgs = [msgs[0], {"role": "user", "content": msgs[1]["content"] + "\n上一次没有输出 ISSUE 行。请至少给出主议题和它的句子编号。"}]
            retry_prompts.append(_render(tokenizer, msgs))
        outs2 = gen(retry_prompts, extract_sp)
        for (s, _), o in zip(retry, outs2):
            raw = _strip_think(o.outputs[0].text.strip())
            compiled = compile_protocol(raw, seg_map[s["sample_id"]], max_issues=6, max_evidence=3, fill_missing_future=False)
            if compiled.issue_list:
                results[s["sample_id"]] = compiled
            else:
                fb = rule_fallback(s, seg_map[s["sample_id"]])
                if fb is None:
                    results[s["sample_id"]] = None
                else:
                    print(f"rule fallback {s.get('sample_id')}", file=sys.stderr)
                    results[s["sample_id"]] = fb

    # future phase, batched across all issues of all samples
    flat = []  # (sample_id, issue_index, prompt)
    for s in todo:
        compiled = results.get(s["sample_id"])
        if compiled is None:
            continue
        for i, iss in enumerate(compiled.issue_list):
            flat.append((s["sample_id"], i, _render(tokenizer, future_messages(iss))))
    print(f"[vllm] future prompts={len(flat)}", flush=True)
    fouts = gen([p for _, _, p in flat], future_sp)
    futures_by = {}
    for (sid, idx, _), o in zip(flat, fouts):
        raw = _strip_think(o.outputs[0].text.strip())
        line = ""
        for cand in raw.splitlines():
            c = cand.strip()
            if c.startswith("FUTURE"):
                line = c[len("FUTURE"):].strip()
                break
        futures_by.setdefault(sid, {})[idx] = line

    for s in todo:
        compiled = results.get(s["sample_id"])
        if compiled is None:
            continue
        new_futures = []
        for i, iss in enumerate(compiled.issue_list):
            line = futures_by.get(s["sample_id"], {}).get(i, "")
            if not line:
                line = _fallback_future(iss)
            new_futures.append(line)
        compiled.future_argument = new_futures
        if mode == "semifinal":
            compiled = expand_semifinal(compiled, seg_map[s["sample_id"]])
        with Path(output).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(compiled.public(s["sample_id"]), ensure_ascii=False) + "\n")
    print("[vllm] done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--mode", choices=["prelim", "semifinal"], default="prelim")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    run(a.split, a.model, a.adapter, a.output, a.limit, a.max_new_tokens, not a.no_resume, a.mode)


if __name__ == "__main__":
    main()
