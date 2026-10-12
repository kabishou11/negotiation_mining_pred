"""Regenerate ONLY the futures of an existing result file with an altered
future instruction (e.g. anti-copy abstraction), reusing vllm batch decode.

    python3 -m scorer.refuture IN.jsonl OUT.jsonl --model ... [--adapter ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scorer.prompt as P
from scorer.compile import _fallback_future
from scorer.infer import _prepare_resume, _strip_think

ANTI_COPY = """你在根据已经确定的议题、立场和原文证据，写一条后续观点。只输出一行 FUTURE 开头的话。
关键要求：用自己的话重新组织表达，像分析师转述一样改写，禁止照抄证据原句；可以保留专有名词、数字和机构名。
多数约 80 字；沿现有机制往下推，可写“将”“应”“未来”。不要新造文中没有的条约、日期或金额。"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    import os
    os.environ["VLLM_USE_V1"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, quantization="bitsandbytes", enable_lora=True,
              max_lora_rank=64, max_model_len=7168, gpu_memory_utilization=0.92,
              enforce_eager=True, disable_log_stats=True)
    lora = LoRARequest("ck", 1, args.adapter) if args.adapter else None
    sp = SamplingParams(temperature=0.0, max_tokens=160)

    old = P.FUTURE_INSTRUCTION
    P.FUTURE_INSTRUCTION = ANTI_COPY
    flat = []
    for r in rows:
        for i, iss in enumerate(r["issue_list"]):
            flat.append((r["sample_id"], i, P.render_prompt(tokenizer, P.future_messages(iss))))
    P.FUTURE_INSTRUCTION = old
    print(f"[refuture] {len(flat)} futures", flush=True)
    outs = llm.generate([p for _, _, p in flat], sp, lora_request=lora)
    fut = {}
    for (sid, idx, _), o in zip(flat, outs):
        raw = _strip_think(o.outputs[0].text.strip())
        line = ""
        for cand in raw.splitlines():
            c = cand.strip()
            if c.startswith("FUTURE"):
                line = c[len("FUTURE"):].strip()
                break
        fut.setdefault(sid, {})[idx] = line

    with Path(args.outfile).open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            nf = []
            for i, iss in enumerate(r["issue_list"]):
                line = fut.get(r["sample_id"], {}).get(i, "")
                if not line:
                    line = (r.get("future_argument") or [""] * len(r["issue_list"]))[i] or _fallback_future(iss)
                nf.append(line)
            fh.write(json.dumps({**r, "future_argument": nf}, ensure_ascii=False) + "\n")
    print("[refuture] done", flush=True)


if __name__ == "__main__":
    main()
