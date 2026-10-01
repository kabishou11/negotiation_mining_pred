"""Split extraction misses into stance errors and similarity errors.

A prediction that would have cleared 0.7 against some gold issue, but never
shares that issue's stance, is a stance miss. A prediction with no gold issue
above 0.7 even after ignoring stance is a similarity miss. Gold issues that
remain unmatched are counted as misses; leftover predictions as extras.
"""

from __future__ import annotations

from scorer.score import score_sample


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
