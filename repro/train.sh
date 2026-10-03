#!/bin/sh
# Build the fit-split SFT file, then QLoRA-train Qwen3-32B on cuda:0.
#   ./repro/train.sh /path/to/Qwen3-32B
#   ./repro/train.sh /path/to/Qwen3-32B runs/qlora-r16
#   ./repro/train.sh /path/to/Qwen3-32B runs/qlora-r16 3072
# With no third argument the length is chosen from GPU memory:
# 6144 at 43GiB+ (a 48GB card), 4096 at 40GiB+, 3072 on a V100-32G.
# A checkpoint-* directory already in the output path is resumed.
# Overflowing samples are skipped, not truncated.
# After training:
#   ./repro/run.sh /path/to/Qwen3-32B result_val.jsonl runs/qlora-r16 val
#   python3 -m scorer.check_submit result_val.jsonl --split val
set -eu
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${1:?pass the local Qwen3-32B directory}"
OUT="${2:-runs/qlora-r16}"
MAXLEN="${3:-}"
python3 -m scorer.build_sft
set -- --model "$MODEL" --output "$OUT"
if [ -n "$MAXLEN" ]; then
  set -- "$@" --max-length "$MAXLEN"
fi
if [ -d "$OUT" ]; then
  CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | while read -r dir; do
    if [ -f "$dir/trainer_state.json" ]; then
      printf '%s\n' "$dir"
    fi
  done | awk -F- '{print $NF, $0}' | sort -n | awk '{print $2}' | tail -n 1 || true)
  if [ -n "$CKPT" ]; then
    echo "resuming $CKPT"
    set -- "$@" --resume-from "$CKPT"
  fi
fi
python3 -m scorer.train "$@"
