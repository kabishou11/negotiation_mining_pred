#!/bin/sh
# Minutes-long environment sanity check before any long run.
#   ./repro/preflight.sh /path/to/Qwen3-32B [adapter]
# Loads the 4-bit model, runs 3 val samples end-to-end (extraction plus
# per-issue futures), evaluates them and prints the scores. Catches
# bitsandbytes/transformers version problems, a wrong --adapter path,
# tokenizer drift, or an OOM at batch level before they burn an evening.
# For a full validation pass use repro/val.sh.
set -eu
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODEL="${1:?pass the local Qwen3-32B directory}"
ADAPTER="${2:-}"
OUT=result_preflight.jsonl
rm -f "$OUT" "$OUT.failures.jsonl"
set -- --split val --limit 3 --model "$MODEL" --output "$OUT"
if [ -n "$ADAPTER" ]; then
  set -- "$@" --adapter "$ADAPTER"
fi
python3 -m scorer.infer "$@"
python3 -m scorer.evaluate "$OUT" --split val --limit 3 --out result_preflight_report.json
echo "preflight ok: 3 val samples inferred and scored. If the scores look sane, run repro/val.sh next."
