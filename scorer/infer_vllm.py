"""Batched vllm inference: same protocol as scorer.infer, much faster decode.

Requires: vllm installed in env, plus a LoRA adapter directory from scorer.train.
Quantizes the base with bitsandbytes to fit one 49GB card. Thinking template
identical to scorer.train/scorer.infer via the same render_prompt call.

Extensions over the plain greedy path (all batched, same semantics as
scorer.infer): --stance-check, --min-issues, --temperature/--top-p/--seed
for sampled runs (vllm per-request seed keeps the run reproducible).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scorer.compile import _fallback_future, compile_protocol, expand_semifinal, rule_fallback
from scorer.datautil import load_split
from scorer.infer import _prepare_resume, _strip_think
from scorer.prompt import extraction_messages, future_messages, render_prompt, stance_check_messages
from scorer.segment import segment_sample


def run(split, model, adapter, output, limit, max_new_tokens, resume, mode,
        min_issues=0, stance_check=False, temperature=0.0, top_p=1.0, seed=0) -> None:
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
        enforce_eager=True,
        disable_log_stats=True,
    )
    lora = LoRARequest("ck", 1, adapter) if adapter else None
    sp = SamplingParams(temperature=temperature, top_p=top_p, seed=seed if temperature else None,
                        max_tokens=max_new_tokens)
    sp_fut = SamplingParams(temperature=temperature, top_p=top_p, seed=seed if temperature else None,
                            max_tokens=160)
    sp_stance = SamplingParams(temperature=temperature, top_p=top_p, seed=seed if temperature else None,
                               max_tokens=12)

    def render(messages):
        return render_prompt(tokenizer, messages)

    def gen(prompts, params):
        return llm.generate(prompts, params, lora_request=lora)

    def compile_of(sample, text):
        return compile_protocol(text, seg_map[sample["sample_id"]], max_issues=6,
                                max_evidence=3, fill_missing_future=False)

    # ---- phase 1: extraction, batched over all samples
    results, failures = {}, {}
    prompts = [render(extraction_messages(s, seg_map[s["sample_id"]])) for s in todo]
    for s, o in zip(todo, gen(prompts, sp)):
        raw = _strip_think(o.outputs[0].text.strip())
        compiled = compile_of(s, raw)
        if compiled.issue_list:
            results[s["sample_id"]] = compiled
        else:
            failures[s["sample_id"]] = s
    if failures:
        print(f"[vllm] extract retry {len(failures)}", flush=True)
        retry_prompts = []
        for s in failures.values():
            msgs = extraction_messages(s, seg_map[s["sample_id"]])
            msgs = [msgs[0], {"role": "user", "content": msgs[1]["content"] + "\n上一次没有输出 ISSUE 行。请至少给出主议题和它的句子编号。"}]
            retry_prompts.append(render(msgs))
        for s, o in zip(list(failures.values()), gen(retry_prompts, sp)):
            raw = _strip_think(o.outputs[0].text.strip())
            compiled = compile_of(s, raw)
            if compiled.issue_list:
                results[s["sample_id"]] = compiled

    # ---- phase 2: min-issues top-up (adopted only when it finds more sides)
    if min_issues:
        low = [s for s in todo
               if s["sample_id"] in results and 0 < len(results[s["sample_id"]].issue_list) < min_issues]
        if low:
            print(f"[vllm] min-issues retry {len(low)}", flush=True)
            retry_prompts = []
            for s in low:
                n = len(results[s["sample_id"]].issue_list)
                msgs = extraction_messages(s, seg_map[s["sample_id"]])
                msgs = [msgs[0], {"role": "user", "content": msgs[1]["content"] + f"\n上一次只给出 {n} 个议题，偏少。请重新通读全文，把明显不同的侧面补全，输出 4 到 5 个 ISSUE 行。"}]
                retry_prompts.append(render(msgs))
            for s, o in zip(low, gen(retry_prompts, sp)):
                raw = _strip_think(o.outputs[0].text.strip())
                retry = compile_of(s, raw)
                if len(retry.issue_list) > len(results[s["sample_id"]].issue_list):
                    results[s["sample_id"]] = retry

    # ---- phase 3: rule fallback for total failures
    for s in todo:
        if s["sample_id"] not in results:
            fb = rule_fallback(s, seg_map[s["sample_id"]])
            if fb is None:
                results[s["sample_id"]] = None
            else:
                print(f"rule fallback {s.get('sample_id')}", file=sys.stderr)
                results[s["sample_id"]] = fb

    # ---- phase 4: stance verification, one short call per issue, batched
    if stance_check:
        flat = []
        for s in todo:
            compiled = results.get(s["sample_id"])
            if not compiled or not compiled.issue_list:
                continue
            docs = sorted(s.get("docs") or [], key=lambda d: d.get("publish_date") or "")
            doc_type = docs[0].get("doc_type") or "" if docs else ""
            for i, iss in enumerate(compiled.issue_list):
                flat.append((s["sample_id"], i, render(stance_check_messages(iss, doc_type))))
        if flat:
            print(f"[vllm] stance checks {len(flat)}", flush=True)
            outs = gen([p for _, _, p in flat], sp_stance)
            changed = 0
            for (sid, idx, _), o in zip(flat, outs):
                raw = _strip_think(o.outputs[0].text.strip())
                m = re.search(r"STANCE\s+(support|oppose|neutral)", raw, re.IGNORECASE)
                if m and m.group(1).lower() != results[sid].issue_list[idx]["stance"]:
                    results[sid].issue_list[idx]["stance"] = m.group(1).lower()
                    changed += 1
            print(f"[vllm] stance changed {changed}", flush=True)

    # ---- phase 5: futures, batched across all issues
    flat = []
    for s in todo:
        compiled = results.get(s["sample_id"])
        if compiled is None:
            continue
        for i, iss in enumerate(compiled.issue_list):
            flat.append((s["sample_id"], i, render(future_messages(iss))))
    print(f"[vllm] future prompts={len(flat)}", flush=True)
    fouts = gen([p for _, _, p in flat], sp_fut)
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

    out_path = Path(output)
    for s in todo:
        compiled = results.get(s["sample_id"])
        if compiled is None:
            public = {"sample_id": s["sample_id"], "issue_list": [], "future_argument": []}
        else:
            new_futures = []
            for i, iss in enumerate(compiled.issue_list):
                line = futures_by.get(s["sample_id"], {}).get(i, "")
                if not line:
                    line = _fallback_future(iss)
                new_futures.append(line)
            compiled.future_argument = new_futures
            if mode == "semifinal":
                compiled = expand_semifinal(compiled, seg_map[s["sample_id"]])
            public = compiled.public(s["sample_id"])
        with out_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(public, ensure_ascii=False) + "\n")
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
    p.add_argument("--min-issues", type=int, default=0)
    p.add_argument("--stance-check", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    run(a.split, a.model, a.adapter, a.output, a.limit, a.max_new_tokens,
        not a.no_resume, a.mode, a.min_issues, a.stance_check, a.temperature, a.top_p, a.seed)


if __name__ == "__main__":
    main()
