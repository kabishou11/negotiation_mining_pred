#!/bin/sh
# Greedy, thinking-disabled inference.
#   ./repro/run.sh                         # preview one val prompt, no weights
#   ./repro/run.sh /path/to/Qwen3-32B
#   ./repro/run.sh /path/to/Qwen3-32B result.jsonl runs/qlora-r16
#   ./repro/run.sh /path/to/Qwen3-32B result_val.jsonl runs/qlora-r16 val
# A rerun skips sample_ids already in the output file. Pass --no-resume
# on the python command to start over. --device-map auto is the two-card
# escape hatch; the default stays on cuda:0. Decoding stays greedy, one
# sample at a time.
set -eu
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  python3 -m scorer.infer --split val --limit 1 --dry-run
  exit 0
fi
OUT="${2:-result.jsonl}"
ADAPTER="${3:-}"
SPLIT="${4:-test}"
python3 -m scorer.infer --split "$SPLIT" --model "$MODEL" --output "$OUT" --adapter "$ADAPTER"
