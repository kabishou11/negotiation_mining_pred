#!/bin/sh
# Second-team zero-training inference. Run from anywhere; paths are fixed.
#   ./repro/run.sh                                   # preview one prompt, no weights
#   ./repro/run.sh /path/to/Qwen3-32B                # test split -> result.jsonl
#   ./repro/run.sh /path/to/Qwen3-32B result_val.jsonl val
# Decoding is temperature 0.2 / top_p 0.9 / seed 42 (configs/decode.json).
# A rerun skips sample_ids already in the output file. After the run:
#   python3 -m scorer.check_submit result.jsonl --split test
set -eu
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  python3 src/infer_e2e.py --split val --limit 1 --dry-run
  exit 0
fi
OUT="${2:-result.jsonl}"
SPLIT="${3:-test}"
python3 src/infer_e2e.py --split "$SPLIT" --model "$MODEL" --output "$OUT"
