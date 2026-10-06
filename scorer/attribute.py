"""Split extraction misses into stance errors and similarity errors.

A prediction that would have cleared 0.7 against some gold issue, but never
shares that issue's stance, is a stance miss. A prediction with no gold issue
above 0.7 even after ignoring stance is a similarity miss. Gold issues that
remain unmatched are counted as misses; leftover predictions as extras.
"""

from __future__ import annotations

from scorer.score import score_sample


def classify_unmatched(i, issue, gold_issues, weighted, threshold: float) -> str:
    """One of stance_blocked / low_sim / extra for an unmatched prediction.

    Mirrors the counting rules of `attribute_sample`: a prediction that clears
    the threshold against some gold issue but never with a shared stance is a
    stance miss; one that clears nothing even ignoring stance is a similarity
    miss; the rest cleared with the right stance somewhere and still lost the
    bipartite matching, which makes them duplicates.
    """
    cleared_any = False
    cleared_same = False
    cleared_diff = False
    for j, gold_issue in enumerate(gold_issues):
        if j >= len(weighted):
            break
        if weighted[i][j] > threshold:
            cleared_any = True
            if issue.get("stance") == gold_issue.get("stance"):
                cleared_same = True
            else:
                cleared_diff = True
    if not cleared_any:
        return "low_sim"
    if cleared_diff and not cleared_same:
        return "stance_blocked"
    return "extra"


def attribute_detail(pred: dict, gold: dict, scored, threshold: float = 0.7) -> dict:
    """Per-issue attribution for the evaluation report.

    `scored` is the SampleScore of this pair, so the weighted matrix is not
    recomputed. The count fields keep the `attribute_sample` names; the lists
    carry the names and stances so a report can be read without the data.
    """
    matched_pred = {i for i, _j in scored.matches}
    matched_gold = {j for _i, j in scored.matches}
    pred_issues = list(pred.get("issue_list") or [])
    gold_issues = list(gold.get("issue_list") or [])
    unmatched_pred: list[dict] = []
    stance_blocked = low_sim = extra = 0
    for i, issue in enumerate(pred_issues):
        if i in matched_pred:
            continue
        kind = classify_unmatched(i, issue, gold_issues, scored.weighted, threshold)
        if kind == "stance_blocked":
            stance_blocked += 1
        elif kind == "low_sim":
            low_sim += 1
        else:
            extra += 1
        unmatched_pred.append(
            {"name": str(issue.get("issue_name") or ""), "stance": issue.get("stance"), "class": kind}
        )
    return {
        "matched": len(scored.matches),
        "stance_blocked": stance_blocked,
        "low_sim": low_sim,
        "pred_extra": stance_blocked + low_sim + extra,
        "gold_missed": len(gold_issues) - len(matched_gold),
        "matches": [
            {
                "pred": str(pred_issues[i].get("issue_name") or ""),
                "gold": str(gold_issues[j].get("issue_name") or ""),
                "stance": gold_issues[j].get("stance"),
                "weight": round(float(scored.weighted[i][j]), 4),
            }
            for i, j in scored.matches
        ],
        "unmatched_pred": unmatched_pred,
        "unmatched_gold": [
            str(gold_issues[j].get("issue_name") or "")
            for j in range(len(gold_issues))
            if j not in matched_gold
        ],
    }


def attribute_sample(pred: dict, gold: dict, encoder, alpha_fn, **kwargs) -> dict[str, int]:
    scored = score_sample(pred, gold, encoder, alpha_fn, **kwargs)
    matched_pred = {i for i, _j in scored.matches}
    matched_gold = {j for _i, j in scored.matches}
    pred_issues = list(pred.get("issue_list") or [])
    gold_issues = list(gold.get("issue_list") or [])
    stance_blocked = 0
    low_sim = 0
    for i, issue in enumerate(pred_issues):
        if i in matched_pred:
            continue
        cleared = False
        blocked = False
        for j, gold_issue in enumerate(gold_issues):
            if j >= len(scored.weighted[i]):
                break
            if scored.weighted[i][j] > kwargs.get("threshold", 0.7):
                cleared = True
                if issue.get("stance") != gold_issue.get("stance"):
                    blocked = True
        if blocked and not any(
            issue.get("stance") == gold_issues[j].get("stance") and scored.weighted[i][j] > kwargs.get("threshold", 0.7)
            for j in range(len(gold_issues))
        ):
            stance_blocked += 1
        elif not cleared:
            low_sim += 1
    return {
        "matched": len(scored.matches),
        "stance_blocked": stance_blocked,
        "low_sim": low_sim,
        "pred_extra": len(pred_issues) - len(matched_pred),
        "gold_missed": len(gold_issues) - len(matched_gold),
    }
