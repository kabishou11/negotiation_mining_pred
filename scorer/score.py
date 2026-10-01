"""Sample-level reproduction of the published judge.

Assumptions, because the official write-up does not pin them down:

- Argument-chain vectors are the embedding of ``"\\n".join(argument_chain)``.
  `arg_mode="mean"` averages one vector per evidence string instead. The
  10/05 submission slot exists to choose between them.
- The weighted cosine must be **strictly greater than 0.7** ("超过").
- Matching maximises the sum of edge weights. Missing edges stay unmatched.
- BERTScore-F1 is supplied by the caller. The published α is
  ``bert-base-chinese`` with baseline rescaling off, so α stays in ``[0, 1]``.
  See `encoders.BertScore`.
- ROUGE-L is character-level LCS, matching the formula as written (no
  tokenizer is specified).
- Future sentence ``i`` of the prediction is compared with future sentence
  ``j`` of the reference when issue ``i`` was matched to issue ``j``.
- A dataset score is the unweighted mean of sample scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scorer.matching import max_weight_assignment


def join_chain(chain: list[str]) -> str:
    return "\n".join(chain)


def lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    prev = [0] * (len(right) + 1)
    for ch in left:
        cur = [0]
        for j, other in enumerate(right, start=1):
            if ch == other:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(prev[j] if prev[j] >= cur[-1] else cur[-1])
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    if pred == gold:
        return 1.0
    common = lcs_length(pred, gold)
    precision = common / len(pred)
    recall = common / len(gold)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def _l2(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_matrix(encoder, left: list[str], right: list[str]) -> list[list[float]]:
    if not left or not right:
        return [[0.0] * len(right) for _ in left]
    vecs = encoder.encode(list(left) + list(right))
    if len(vecs) != len(left) + len(right):
        raise ValueError("encoder returned the wrong number of vectors")
    a = [_l2(list(map(float, v))) for v in vecs[: len(left)]]
    b = [_l2(list(map(float, v))) for v in vecs[len(left) :]]
    return [[_dot(x, y) for y in b] for x in a]


def _mean_vectors(encoder, chains: list[list[str]]) -> list[list[float]]:
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for chain in chains:
        start = len(flat)
        flat.extend(chain if chain else [""])
        spans.append((start, len(flat)))
    vecs = [_l2(list(map(float, v))) for v in encoder.encode(flat)]
    dim = len(vecs[0]) if vecs else 0
    out: list[list[float]] = []
    for start, end in spans:
        acc = [0.0] * dim
        count = max(1, end - start)
        for vec in vecs[start:end]:
            for k, value in enumerate(vec):
                acc[k] += value
        out.append(_l2([v / count for v in acc]))
    return out


def argument_cosine(encoder, pred_chains: list[list[str]], gold_chains: list[list[str]], mode: str) -> list[list[float]]:
    if mode == "newline":
        return cosine_matrix(encoder, [join_chain(c) for c in pred_chains], [join_chain(c) for c in gold_chains])
    if mode == "mean":
        left = _mean_vectors(encoder, pred_chains)
        right = _mean_vectors(encoder, gold_chains)
        return [[_dot(x, y) for y in right] for x in left]
    raise ValueError(mode)


@dataclass
class SampleScore:
    score: float
    s_ext: float
    s_pred: float
    f1_ext: float
    precision: float
    recall: float
    alpha: float
    f1_pred: float
    s_semantic: float
    n_matched: int
    n_pred: int
    n_gold: int
    matches: list[tuple[int, int]] = field(default_factory=list)
    weighted: list[list[float]] = field(default_factory=list)


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def _default_future_sim(pred_text: str, gold_text: str) -> float:
    """Offline stand-in for future-argument BERTScore. Identical to character ROUGE-L."""
    return rouge_l_f1(pred_text, gold_text)


def score_sample(
    pred: dict,
    gold: dict,
    encoder,
    alpha_fn,
    *,
    future_sim_fn=None,
    threshold: float = 0.7,
    arg_mode: str = "newline",
) -> SampleScore:
    """Score one sample.

    `alpha_fn(pred_issue, gold_issue)` is the triple BERTScore-F1.
    `future_sim_fn(pred_future, gold_future)` is the future-argument BERTScore-F1.
    The default future stand-in is character ROUGE-L, so offline `S_semantic`
    equals `F1_pred`. Pass a real BERTScore callable for a leaderboard estimate.
    """
    if future_sim_fn is None:
        future_sim_fn = _default_future_sim
    pred_issues = list(pred.get("issue_list") or [])
    gold_issues = list(gold.get("issue_list") or [])
    pred_futs = list(pred.get("future_argument") or [])
    gold_futs = list(gold.get("future_argument") or [])
    n_pred = len(pred_issues)
    n_gold = len(gold_issues)

    name_cos = cosine_matrix(
        encoder,
        [str(it.get("issue_name") or "") for it in pred_issues],
        [str(it.get("issue_name") or "") for it in gold_issues],
    )
    arg_cos = argument_cosine(
        encoder,
        [list(it.get("argument_chain") or []) for it in pred_issues],
        [list(it.get("argument_chain") or []) for it in gold_issues],
        arg_mode,
    )
    weighted = [[0.4 * name_cos[i][j] + 0.6 * arg_cos[i][j] for j in range(n_gold)] for i in range(n_pred)]
    allowed: list[list[float | None]] = []
    for i, pred_issue in enumerate(pred_issues):
        row: list[float | None] = []
        for j, gold_issue in enumerate(gold_issues):
            same_stance = pred_issue.get("stance") == gold_issue.get("stance")
            score = weighted[i][j]
            row.append(score if same_stance and score > threshold else None)
        allowed.append(row)
    matches = max_weight_assignment(allowed) if n_pred and n_gold else []

    nc = len(matches)
    precision = nc / n_pred if n_pred else 0.0
    recall = nc / n_gold if n_gold else 0.0
    f1_ext = _f1(precision, recall)
    if matches:
        alpha = sum(float(alpha_fn(pred_issues[i], gold_issues[j])) for i, j in matches) / nc
    else:
        alpha = 0.0
    s_ext = f1_ext * (0.7 + 0.3 * alpha)

    denom = max(n_pred, n_gold)
    if denom == 0:
        f1_pred = 0.0
        s_semantic = 0.0
    else:
        rouge_sum = 0.0
        sem_sum = 0.0
        for i, j in matches:
            pred_text = str(pred_futs[i]) if i < len(pred_futs) else ""
            gold_text = str(gold_futs[j]) if j < len(gold_futs) else ""
            rouge_sum += rouge_l_f1(pred_text, gold_text)
            sem_sum += float(future_sim_fn(pred_text, gold_text))
        f1_pred = rouge_sum / denom
        s_semantic = sem_sum / denom
    s_pred = 0.6 * f1_pred + 0.4 * s_semantic
    return SampleScore(
        score=0.8 * s_ext + 0.2 * s_pred,
        s_ext=s_ext,
        s_pred=s_pred,
        f1_ext=f1_ext,
        precision=precision,
        recall=recall,
        alpha=alpha,
        f1_pred=f1_pred,
        s_semantic=s_semantic,
        n_matched=nc,
        n_pred=n_pred,
        n_gold=n_gold,
        matches=matches,
        weighted=weighted,
    )


def semantic_alpha(pred_issue: dict, gold_issue: dict) -> float:
    """Stand-in α: 1 when the triple text is identical, else character bigram F1.

    This is **not** bert-base-chinese. It lets the judge's arithmetic be tested
    offline. Swap in `encoders.BertScore` for a leaderboard estimate.
    """
    def text_of(issue: dict) -> str:
        if issue.get("issue_name", "") == "" and issue.get("argument_chain"):
            return "\n".join(issue["argument_chain"])
        return str(issue.get("issue_name") or "") + "\n" + join_chain(list(issue.get("argument_chain") or []))

    left = text_of(pred_issue)
    right = text_of(gold_issue)
    if left == right:
        return 1.0
    return rouge_l_f1(left, right)


def score_dataset(pairs: list[tuple[dict, dict]], encoder, alpha_fn=semantic_alpha, **kwargs) -> dict[str, float]:
    if not pairs:
        return {"n": 0, "score": 0.0, "s_ext": 0.0, "s_pred": 0.0}
    rows = [score_sample(pred, gold, encoder, alpha_fn, **kwargs) for pred, gold in pairs]
    n = len(rows)
    def avg(name: str) -> float:
        return sum(getattr(row, name) for row in rows) / n
    return {
        "n": n,
        "score": avg("score"),
        "s_ext": avg("s_ext"),
        "s_pred": avg("s_pred"),
        "f1_ext": avg("f1_ext"),
        "alpha": avg("alpha"),
        "f1_pred": avg("f1_pred"),
        "s_semantic": avg("s_semantic"),
    }
