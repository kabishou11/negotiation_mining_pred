#!/bin/sh
# Build the fit-split SFT file, then QLoRA-train Qwen3-32B on cuda:0.
#   ./repro/train.sh /path/to/Qwen3-32B
#   ./repro/train.sh /path/to/Qwen3-32B runs/qlora-r16
#   ./repro/train.sh /path/to/Qwen3-32B runs/qlora-r16 3072
# The third argument overrides --max-length. Default 6144 covers the fit
# set under the 1.5-character token estimate (A100). One V100-32G should
# pass 3072. Overflowing samples are skipped, not truncated.
set -eu
cd "$(dirname "$0")/.."
MODEL="${1:?pass the local Qwen3-32B directory}"
OUT="${2:-runs/qlora-r16}"
MAXLEN="${3:-}"
python3 -m scorer.build_sft
if [ -n "$MAXLEN" ]; then
  python3 -m scorer.train --model "$MODEL" --output "$OUT" --max-length "$MAXLEN"
else
  python3 -m scorer.train --model "$MODEL" --output "$OUT"
fi
