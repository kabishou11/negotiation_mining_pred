"""Maximum-weight bipartite matching for the extraction judge.

Edges that fail the stance gate or the 0.7 threshold are absent. A missing
edge must not be forced into the assignment: extra predictions and missed
gold issues stay unmatched, which is what `N = max(Ng, Np)` later penalises.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

_LARGE = 1e6


def max_weight_assignment(weights: list[list[float | None]]) -> list[tuple[int, int]]:
    """Return pairs `(row, col)` maximising the sum of present weights.

    `None` is a forbidden edge. Rows and columns may remain unmatched. A
    negative weight is never selected, because leaving the node unmatched
    scores 0.
    """
    n = len(weights)
    m = len(weights[0]) if n else 0
    if n == 0 or m == 0:
        return []

    # Assignment on a padded square: real-real allowed edges carry -weight,
    # forbidden real-real edges are expensive, and dummy nodes cost 0 so a
    # node can stay unmatched.
    size = n + m
    cost = np.zeros((size, size), dtype=np.float64)
    for i in range(n):
        row = weights[i]
        if len(row) != m:
            raise ValueError("ragged weight matrix")
        for j in range(m):
            weight = row[j]
            cost[i, j] = _LARGE if weight is None else -float(weight)

    rows, cols = linear_sum_assignment(cost)
    chosen: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < n and j < m and weights[i][j] is not None:
            chosen.append((i, j))
    return chosen


def max_cardinality_assignment(weights: list[list[float | None]]) -> list[tuple[int, int]]:
    """Return pairs maximising the NUMBER of present edges.

    Hedge for the other reading of 一对一最优匹配. With threshold-gated
    weights in (0.7, 1.0] the two objectives can disagree only from four
    vertices up: one 0.95 edge may crowd out two 0.71 edges unless a longer
    augmenting path pays for itself. The official write-up says 最优匹配
    (weight-based), so this stays a diagnostic, not the default.
    """
    n = len(weights)
    m = len(weights[0]) if n else 0
    if n == 0 or m == 0:
        return []
    size = n + m
    cost = np.zeros((size, size), dtype=np.float64)
    for i in range(n):
        row = weights[i]
        if len(row) != m:
            raise ValueError("ragged weight matrix")
        for j in range(m):
            cost[i, j] = _LARGE if row[j] is None else -1.0
    rows, cols = linear_sum_assignment(cost)
    chosen: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < n and j < m and weights[i][j] is not None:
            chosen.append((i, j))
    return chosen
