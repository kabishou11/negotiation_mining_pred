#!/bin/sh
# Validation-split diagnosis loop: infer then score, report to disk.
#   ./repro/val.sh /path/to/Qwen3-32B runs/qlora-r16
#   ./repro/val.sh /path/to/Qwen3-32B runs/qlora-r16/checkpoint-1800
#   ./repro/val.sh /path/to/Qwen3-32B "" val                     # base model
#   ./repro/val.sh MODEL ADAPTER val --min-issues 4 --stance-check
# Extra arguments after the split are forwarded to scorer.infer verbatim.
# Writes result_val_<name>.jsonl plus a _report.json with per-sample scores,
# matches and miss attribution for both evidence modes. This is the loop that
# decides where the leaderboard points are lost; run it before every
# submission-side experiment. For scoring an existing file only, use
# repro/eval.sh; for model-free evidence polish use scorer/postprocess.
set -eu
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODEL="${1:?pass the local Qwen3-32B directory}"
ADAPTER="${2:-}"
SPLIT="${3:-val}"
shift 3 2>/dev/null || true
EXTRA="$*"
if [ -n "$ADAPTER" ] && [ -f "$ADAPTER/adapter_config.json" ]; then
  NAME=$(basename "$(dirname "$ADAPTER")")-$(basename "$ADAPTER")
else
  NAME=$(basename "${ADAPTER:-base}")
fi
OUT="result_${SPLIT}_${NAME}.jsonl"
# shellcheck disable=SC2086
python3 -m scorer.infer --split "$SPLIT" --model "$MODEL" --adapter "$ADAPTER" --output "$OUT" $EXTRA
python3 -m scorer.evaluate "$OUT" --split "$SPLIT" --out "${OUT%.jsonl}_report.json"
python3 -m scorer.check_submit "$OUT" --split "$SPLIT"
