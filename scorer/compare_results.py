"""Measure surface similarity between the two teams' result files.

The pipelines share no code, prompts, decoding, or structure (the
anti-similarity contract); both, however, approximate the same gold
answers, so their outputs are expected to converge in content. What must
NOT converge is surface behaviour that looks like copying: identical
fallback strings, near-identical futures, same evidence length profile.
This tool quantifies exactly those, per field, so a divergence decision
is made on numbers instead of vibes.

    python3 -m scorer.compare_results result_a.jsonl result_b.jsonl

Guidance printed with the numbers: high name similarity is expected and
fine (both teams name the same gold issues differently in code but land
near the same names); high FUTURE similarity is the real risk because
futures are free-form generation — if it climbs past the threshold,
change team B's future wording, not team A's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scorer.score import rouge_l_f1


def _bigrams(text: str) -> set[str]:
    flat = "".join(str(text).split())
    if len(flat) < 2:
        return {flat} if flat else set()
    return {flat[i : i + 2] for i in range(len(flat) - 1)}


def _set_cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / ((len(a) * len(b)) ** 0.5)


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid json ({exc.msg})") from exc
            rows[row["sample_id"]] = row
    return rows


def _greedy_name_pairs(a_names: list[str], b_names: list[str]) -> list[tuple[int, int, float]]:
    scored: list[tuple[float, int, int]] = []
    for i, a_name in enumerate(a_names):
        a_grams = _bigrams(a_name)
        for j, b_name in enumerate(b_names):
            scored.append((_set_cosine(a_grams, _bigrams(b_name)), i, j))
    scored.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for sim, i, j in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, sim))
    return pairs


def compare(rows_a: dict[str, dict], rows_b: dict[str, dict]) -> dict:
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        raise SystemExit("no overlapping sample_ids between the two files")
    name_sims: list[float] = []
    same_stance: list[int] = []
    name_pairs_count = 0
    evidence_exact = 0
    evidence_total = 0
    evidence_lens = {"a": [], "b": []}
    future_sims: list[float] = []
    count_delta: list[int] = []
    for sid in shared:
        issues_a = rows_a[sid].get("issue_list") or []
        issues_b = rows_b[sid].get("issue_list") or []
        count_delta.append(len(issues_b) - len(issues_a))
        pairs = _greedy_name_pairs(
            [str(i.get("issue_name") or "") for i in issues_a],
            [str(i.get("issue_name") or "") for i in issues_b],
        )
        name_pairs_count += len(pairs)
        for i, j, sim in pairs:
            name_sims.append(sim)
            same_stance.append(int(issues_a[i].get("stance") == issues_b[j].get("stance")))
            chain_a = [str(e) for e in issues_a[i].get("argument_chain") or []]
            chain_b = [str(e) for e in issues_b[j].get("argument_chain") or []]
            evidence_total += 1
            evidence_exact += int(bool(set(chain_a) & set(chain_b)))
            evidence_lens["a"].extend(len(e) for e in chain_a)
            evidence_lens["b"].extend(len(e) for e in chain_b)
        for i, j, _sim in pairs:
            fut_a = str((rows_a[sid].get("future_argument") or ["" ] * len(issues_a))[i])
            fut_b = str((rows_b[sid].get("future_argument") or ["" ] * len(issues_b))[j])
            future_sims.append(rouge_l_f1(fut_a, fut_b))

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    mean_len_a = avg([float(v) for v in evidence_lens["a"]])
    mean_len_b = avg([float(v) for v in evidence_lens["b"]])
    return {
        "n_samples": len(shared),
        "name_pairs": name_pairs_count,
        "name_similarity": avg(name_sims),
        "stance_agreement": avg([float(v) for v in same_stance]),
        "evidence_verbatim_share": evidence_exact / evidence_total if evidence_total else 0.0,
        "evidence_mean_len_a": mean_len_a,
        "evidence_mean_len_b": mean_len_b,
        "future_rouge_l": avg(future_sims),
        "issue_count_delta": avg([float(v) for v in count_delta]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", help="team A result.jsonl (adapter pipeline)")
    parser.add_argument("file_b", help="team B result.jsonl (base end-to-end)")
    args = parser.parse_args()
    stats = compare(load(Path(args.file_a)), load(Path(args.file_b)))
    print(
        f"n={stats['n_samples']} name_pairs={stats['name_pairs']} "
        f"name_similarity={stats['name_similarity']:.3f} "
        f"stance_agreement={stats['stance_agreement']:.3f}"
    )
    print(
        f"evidence verbatim={stats['evidence_verbatim_share']:.3f} "
        f"mean_len a={stats['evidence_mean_len_a']:.0f} b={stats['evidence_mean_len_b']:.0f} "
        f"(length gap keeps the surfaces apart)"
    )
    print(
        f"future_rouge_l={stats['future_rouge_l']:.3f} "
        f"issue_count_delta={stats['issue_count_delta']:+.2f}"
    )
    if stats["future_rouge_l"] > 0.55:
        print("[action] futures too alike: reword team B's future instruction before submitting.")
    else:
        print("[ok] future surface similarity within tolerance.")
    if (
        stats["evidence_verbatim_share"] > 0.5
        and abs(stats["evidence_mean_len_a"] - stats["evidence_mean_len_b"]) < 15
    ):
        print("[action] evidence surfaces converged: push team B to shorter substrings.")
    else:
        print("[ok] evidence surfaces distinguishable.")


if __name__ == "__main__":
    main()
