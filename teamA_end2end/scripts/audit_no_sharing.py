"""String-level zero-sharing audit across the two team codebases.

The contract (../../DIVERGENCE_CONTRACT.md) bans shared code between
scorer/ (team 1, adapter pipeline) and teamA_end2end/ (team 2, base
end-to-end). Comments and docs differ by construction; what can silently
converge is string literals — fallback templates, prompt phrases, error
messages. This walks both ASTs and reports every literal both sides share
at 6+ characters that is not necessarily-shared vocabulary.

Run from the repository root (exit 1 = findings):

    python3 teamA_end2end/scripts/audit_no_sharing.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A_FILES = sorted((ROOT / "scorer").glob("*.py"))
B_FILES = sorted((ROOT / "teamA_end2end" / "src").glob("*.py"))

# Necessarily identical — grouped by why neither side can differ:
#   submission schema:      the output contract of the competition
#   stance vocabulary:      the three published strings
#   dataset schema:         the released jsonl field names
#   split/file naming:      dictated by the rules (result.jsonl) and data
#   framework surface:      argparse flags, action names, HF tensor keys,
#                           chat message roles, device-map choices — the
#                           interface both pipelines must speak
WHITELIST = {
    # submission schema
    "sample_id",
    "issue_list",
    "future_argument",
    "issue_name",
    "stance",
    "argument_chain",
    # stance vocabulary
    "support",
    "oppose",
    "neutral",
    # dataset schema
    "full_text",
    "doc_type",
    "publish_date",
    "docs",
    # split / file naming
    "train",
    "val",
    "test",
    "utf-8",
    "\n",
    "result.jsonl",
    ".jsonl",
    "prelim",
    # framework surface (argparse / HF / chat template)
    "--model",
    "--split",
    "--output",
    "--limit",
    "--device-map",
    "--dry-run",
    "--resume",
    "store_true",
    "__main__",
    "input_ids",
    "content",
    "system",
    "single",
}

THRESHOLD = 6


def collect(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value:
                out.add(value)
    return out


def main() -> int:
    a: set[str] = set()
    for path in A_FILES:
        a |= collect(path)
    b: set[str] = set()
    for path in B_FILES:
        b |= collect(path)
    shared = sorted((a & b) - WHITELIST, key=lambda s: (-len(s), s))
    flagged = [s for s in shared if len(s) >= THRESHOLD]
    if flagged:
        print(f"{len(flagged)} shared literals of {THRESHOLD}+ chars — fix one side or whitelist with a reason:")
        for text in flagged:
            print(f"  {text!r}")
        return 1
    print(f"audit ok: no shared {THRESHOLD}+ char literals between {len(A_FILES)} A files and {len(B_FILES)} B files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
