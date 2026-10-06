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
# Forward only arguments beyond the first three. A bare `shift 3 || true`
# leaves MODEL/ADAPTER in "$*" when fewer than three args were given and
# they would reach infer as stray positionals.
EXTRA=""
if [ "$#" -gt 3 ]; then
  shift 3
  EXTRA="$*"
fi
if [ -n "$ADAPTER" ] && [ -f "$ADAPTER/adapter_config.json" ]; then
  NAME=$(basename "$(dirname "$ADAPTER")")-$(basename "$ADAPTER")
else
  NAME=$(basename "${ADAPTER:-base}")
fi
# Different experiment flags must write different files: a sampled rerun
# into the greedy run's output would resume-skip every sample silently.
OUT="result_${SPLIT}_${NAME}.jsonl"
if [ -n "$EXTRA" ]; then
  TAG=$(printf '%s' "$EXTRA" | cksum | cut -d' ' -f1 | cut -c1-6)
  OUT="result_${SPLIT}_${NAME}_${TAG}.jsonl"
fi
# shellcheck disable=SC2086
python3 -m scorer.infer --split "$SPLIT" --model "$MODEL" --adapter "$ADAPTER" --output "$OUT" $EXTRA
python3 -m scorer.evaluate "$OUT" --split "$SPLIT" --out "${OUT%.jsonl}_report.json"
python3 -m scorer.check_submit "$OUT" --split "$SPLIT"
