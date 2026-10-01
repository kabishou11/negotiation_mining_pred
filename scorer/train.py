"""QLoRA SFT on one Qwen3-32B. One GPU. Greedy-compatible adapter.

The fit set is whatever `build_sft.py` wrote. Default `--max-length` is
6144: under the 1.5-character planning ratio every current extract example
fits, and an A100-80G is the machine that holds that context. A V100-32G
does not; pass `--max-length 3072` there. Samples that still overflow are
skipped rather than truncated, because cutting the tail of a document
deletes gold evidence. After step 20 the script prints a measured ETA from
this machine; the numbers in the plan are only a prior.

V100 (capability 7.0) trains in fp16. Ampere and later train in bf16.
The process uses cuda:0 only. A second card does not speed this script up.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scorer.build_sft import TRAIN_PATH, write_sft


def _require_stack():
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "training dependencies are missing. On the server:\n"
            "  pip install torch transformers peft bitsandbytes accelerate\n"
            f"original error: {exc}"
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def _apply_template(tokenizer, messages, add_generation_prompt: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": None,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    full_ids = _apply_template(tokenizer, messages, add_generation_prompt=False)
    prompt_ids = _apply_template(tokenizer, messages[:-1], add_generation_prompt=True)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Template did not keep the prompt as a prefix. Mask by locating the
        # assistant string; drop the example if that fails.
        target = messages[-1]["content"]
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        start = None
        width = len(target_ids)
        for i in range(0, len(full_ids) - width + 1):
            if full_ids[i : i + width] == list(target_ids):
                start = i
        if start is None:
            return None
        labels = [-100] * start + full_ids[start:]
    else:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(full_ids) > max_length:
        return None
    if all(token == -100 for token in labels):
        return None
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _dataset_cls(torch):
    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            return self.rows[index]

    return EncodedDataset


def _collate(pad_id: int):
    def collate(features: list[dict]) -> dict:
        import torch

        width = max(len(row["input_ids"]) for row in features)
        input_ids, labels, mask = [], [], []
        for row in features:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            mask.append(row["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local Qwen3-32B directory")
    parser.add_argument("--data", default=str(TRAIN_PATH))
    parser.add_argument("--output", default="runs/qlora-r16")
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    stack = _require_stack()
    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA of Qwen3-32B needs a CUDA GPU.")
    data_path = Path(args.data)
    if args.rebuild or not data_path.is_file():
        write_sft()
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    tokenizer = stack["AutoTokenizer"].from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = []
    skipped = 0
    bad_template = 0
    token_total = 0
    for row in rows:
        item = encode_example(tokenizer, row["messages"], args.max_length)
        if item is None:
            # Distinguish overflow from a template mismatch by length alone.
            rough = row.get("chars", 0) / 1.5 + 40
            if rough > args.max_length:
                skipped += 1
            else:
                bad_template += 1
                skipped += 1
            continue
        encoded.append(item)
        token_total += len(item["input_ids"])
    if not encoded:
        raise SystemExit("every training example was skipped. Raise --max-length.")
    print(
        f"examples {len(encoded)}/{len(rows)} skipped {skipped} "
        f"(template_or_other {bad_template}) tokens {token_total} "
        f"max_length {args.max_length}"
    )

    capability = torch.cuda.get_device_capability(0)
    use_bf16 = capability[0] >= 8
    compute = torch.bfloat16 if use_bf16 else torch.float16
    quant = stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute,
    )
    load_kwargs = {
        "quantization_config": quant,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    try:
        model = stack["AutoModelForCausalLM"].from_pretrained(
            args.model, attn_implementation="sdpa", **load_kwargs
        )
    except (TypeError, ValueError):
        model = stack["AutoModelForCausalLM"].from_pretrained(args.model, **load_kwargs)
    model = stack["prepare"](model)
    model = stack["get_peft_model"](
        model,
        stack["LoraConfig"](
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    class EtaCallback(stack["TrainerCallback"]):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step not in (10, 20) and step % 100 != 0:
                return
            elapsed = time.time() - self.started
            per_step = elapsed / max(step, 1)
            left = (state.max_steps - step) * per_step
            print(
                f"[eta] step {step}/{state.max_steps}  {per_step:.2f}s/optim_step  "
                f"remaining {left / 3600:.2f}h",
                flush=True,
            )

    training = stack["TrainingArguments"](
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=0,
        data_seed=0,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = stack["Trainer"](
        model=model,
        args=training,
        train_dataset=_dataset_cls(torch)(encoded),
        data_collator=_collate(tokenizer.pad_token_id),
        callbacks=[EtaCallback()],
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"adapter saved to {args.output}")


if __name__ == "__main__":
    main()
