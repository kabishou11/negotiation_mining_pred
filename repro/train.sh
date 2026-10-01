#!/bin/sh
# Build the fit-split SFT file, then QLoRA-train Qwen3-32B on cuda:0.
#   ./repro/train.sh /path/to/Qwen3-32B
#   ./repro/train.sh /path/to/Qwen3-32B runs/qlora-r16
#   ./repro/train.sh /path/to/Qwen3-32B runs/qlora-r16 3072
# With no third argument the length is chosen from GPU memory:
# 6144 on 70GiB+, 4096 on 40GiB+, 3072 on a V100-32G.
# A checkpoint-* directory already in the output path is resumed.
# Overflowing samples are skipped, not truncated.
set -eu
cd "$(dirname "$0")/.."
MODEL="${1:?pass the local Qwen3-32B directory}"
OUT="${2:-runs/qlora-r16}"
MAXLEN="${3:-}"
python3 -m scorer.build_sft
set -- --model "$MODEL" --output "$OUT"
if [ -n "$MAXLEN" ]; then
  set -- "$@" --max-length "$MAXLEN"
fi
if [ -d "$OUT" ]; then
  CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | awk -F- '{print $NF, $0}' | sort -n | awk '{print $2}' | tail -n 1 || true)
  if [ -n "$CKPT" ]; then
    echo "resuming $CKPT"
    set -- "$@" --resume-from "$CKPT"
  fi
fi
python3 -m scorer.train "$@"
