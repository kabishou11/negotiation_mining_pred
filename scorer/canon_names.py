"""Canonicalize predicted issue names against the train-gold name lexicon.

The annotator vocabulary is finite and stylistic (4-16 char noun phrases).
Predicted names that drift from it land in the 0.6-0.7 near-miss zone and
fail the 0.7 weighted threshold. For every predicted name, if the nearest
train-gold name (bge cosine) is >= --threshold, the name is replaced by
that canonical form. Names already close to the lexicon stay untouched
(their nearest neighbour is effectively themselves).

    python3 -m scorer.canon_names out.jsonl in.jsonl [--threshold 0.70] [--min-sim 0.60]

--min-sim: only rewrite when the predicted name is at least this similar to
the lexicon entry (avoids hijacking genuinely different issues).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scorer.datautil import load_split
from scorer.encoders import BgeEncoder


def _load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--threshold", type=float, default=0.70, help="rewrite when lexicon sim >= this")
    p.add_argument("--min-sim", type=float, default=0.60, help="never rewrite below this similarity")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    lex: list[str] = []
    seen = set()
    for s in load_split("train"):
        for i in s.get("issue_list") or []:
            n = str(i.get("issue_name") or "").strip()
            if n and n not in seen:
                seen.add(n)
                lex.append(n)
    print(f"lexicon: {len(lex)} unique train-gold names")

    enc = BgeEncoder(device=args.device)
    def emb(texts):
        v = np.array(enc.encode(texts), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v

    L = emb(lex)

    for src in args.inputs:
        rows = _load_rows(src)
        names = [str(i["issue_name"]) for r in rows for i in r["issue_list"]]
        if not names:
            continue
        P = emb(names)
        S = P @ L.T
        best_idx = S.argmax(axis=1)
        best_sim = S[np.arange(len(names)), best_idx]
        rewrite = (best_sim >= args.min_sim) & (best_sim >= 0)  # candidates
        k = 0
        changed = 0
        with Path(args.out).open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                new_issues = []
                for i in r["issue_list"]:
                    sim = float(best_sim[k]); idx = int(best_idx[k]); k += 1
                    name = i["issue_name"]
                    if sim >= args.threshold and sim <= 0.985:
                        name = lex[idx]
                        changed += 1
                    elif sim >= 0.985:
                        pass  # identical already
                    new_issues.append({**i, "issue_name": name})
                fh.write(json.dumps({**r, "issue_list": new_issues}, ensure_ascii=False) + "\n")
        print(f"{src}: rewrote {changed}/{len(names)} names -> {args.out}")


if __name__ == "__main__":
    main()
