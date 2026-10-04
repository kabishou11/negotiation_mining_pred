#!/bin/sh
# Leaderboard-style local scoring of a result file (see scorer/evaluate.py).
#   ./repro/eval.sh result_val.jsonl val
#   ./repro/eval.sh result_train.jsonl train data/dev40_ids.txt
# Uses BAAI/bge-small-zh-v1.5 + bert-base-chinese when the server stack
# (torch, transformers, bert-score) is present; without them it still runs,
# printing smoke-test numbers with a loud warning. Cache both models before
# the offline run. After scoring, gate the submission:
#   python3 -m scorer.check_submit result_val.jsonl --split val
set -eu
cd "$(dirname "$0")/.."
RESULT="${1:?pass the result.jsonl path}"
SPLIT="${2:-val}"
IDS="${3:-}"
if [ -n "$IDS" ]; then
  python3 -m scorer.evaluate "$RESULT" --split "$SPLIT" --ids-file "$IDS"
else
  python3 -m scorer.evaluate "$RESULT" --split "$SPLIT"
fi
