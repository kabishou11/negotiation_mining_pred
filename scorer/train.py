"""QLoRA SFT on one Qwen3-32B. One GPU. Greedy-compatible adapter.

`--max-length 0` (the default) picks the context from the GPU that is
actually visible:

- 70GiB and above: 6144. On the official Qwen3-32B tokenizer every current
  extract example fits (measured maximum 4977 tokens) and every future
  example is under 600. 8192 trains no additional row.
- 40GiB up to 70GiB: 4096.
- below 40GiB, including a V100-32G: 3072.

Rows longer than the chosen length are skipped, not truncated, because
cutting the tail deletes gold evidence. Pass an explicit `--max-length` to
override the table. After step 20 the script prints a measured ETA.

The loss is applied only to the assistant answer appended after the same
closed-thinking prefix inference uses. If that prefix is not a token prefix
of the training sequence, the run stops before the 32B weights are loaded.

V100 (capability 7.0) trains in fp16. Ampere and later train in bf16.
The process uses cuda:0 only. A second card does not speed this script up.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from collections import Counter
from pathlib import Path

from scorer.build_sft import TRAIN_PATH, write_sft
from scorer.prompt import render_prompt


class MaskError(Exception):
    """The supervised span is not the assistant answer."""


def resolve_max_length(requested: int, gib: float) -> int:
    """`requested == 0` selects a length the visible GPU can hold."""
    if requested:
        return requested
    if gib >= 70:
        return 6144
    if gib >= 40:
        return 4096
    return 3072


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


def _ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    """Mask the inference prefix. Supervise the answer and the end-of-turn token.

    The sequence is the inference prompt plus the assistant text plus the
    Qwen end token. It is not rendered by a second template pass, so an
    empty `<think>` block cannot be both prefilled and trained.
    """
    answer = messages[-1]["content"]
    if not str(answer).strip():
        return None
    prompt_text = render_prompt(tokenizer, messages[:-1])
    eos = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
    full_text = prompt_text + answer + eos + "\n"
    prompt_ids = _ids(tokenizer, prompt_text)
    full_ids = _ids(tokenizer, full_text)
    answer_ids = _ids(tokenizer, answer)
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise MaskError("prompt tokens are not a prefix of the training sequence")
    supervised = full_ids[len(prompt_ids) :]
    if supervised[: len(answer_ids)] != answer_ids:
        raise MaskError("supervised tokens do not start with the assistant answer")
    if len(full_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + supervised
    if all(token == -100 for token in labels):
        raise MaskError("every label is masked")
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
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="0 picks 6144 / 4096 / 3072 from GPU memory. An explicit value is used as given",
    )
    parser.add_argument("--resume-from", default="", help="checkpoint directory to resume")
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

    gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    max_length = resolve_max_length(args.max_length, gib)
    encoded = []
    skipped_by_task: Counter = Counter()
    token_total = 0
    longest = 0
    for row in rows:
        try:
            item = encode_example(tokenizer, row["messages"], max_length)
        except MaskError as exc:
            raise SystemExit(
                "loss mask is not aligned with the inference prompt. "
                "Refusing to load Qwen3-32B.\n"
                f"{exc}"
            ) from exc
        if item is None:
            skipped_by_task[row.get("task", "?")] += 1
            continue
        encoded.append(item)
        token_total += len(item["input_ids"])
        longest = max(longest, len(item["input_ids"]))
    skipped = sum(skipped_by_task.values())
    if not encoded:
        raise SystemExit("every training example was skipped. Raise --max-length.")
    print(
        f"[lock] gpu={gib:.1f}GiB max_length={max_length} epochs={args.epochs} "
        f"r={args.r} alpha={args.alpha} lr={args.lr} records={len(rows)} "
        f"kept={len(encoded)} skipped={skipped} by_task={dict(skipped_by_task)} "
        f"longest={longest} tokens={token_total}",
        flush=True,
    )
    if gib >= 70 and skipped and not args.max_length:
        raise SystemExit(
            f"{skipped} examples exceed {max_length} on a {gib:.0f}GiB GPU. "
            "Refusing to start. Rerun with --max-length 8192."
        )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_lock.json").write_text(
        json.dumps(
            {
                "max_length": max_length,
                "epochs": args.epochs,
                "lr": args.lr,
                "r": args.r,
                "alpha": args.alpha,
                "gpu_gib": round(gib, 2),
                "records": len(rows),
                "kept": len(encoded),
                "skipped": skipped,
                "skipped_by_task": dict(skipped_by_task),
                "longest_tokens": longest,
                "resume_from": args.resume_from,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
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

    training_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
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
    if "gradient_checkpointing_kwargs" in inspect.signature(stack["TrainingArguments"]).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training = stack["TrainingArguments"](**training_kwargs)
    trainer = stack["Trainer"](
        model=model,
        args=training,
        train_dataset=_dataset_cls(torch)(encoded),
        data_collator=_collate(tokenizer.pad_token_id),
        callbacks=[EtaCallback()],
    )
    trainer.train(resume_from_checkpoint=args.resume_from or None)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"adapter saved to {args.output}")


if __name__ == "__main__":
    main()
