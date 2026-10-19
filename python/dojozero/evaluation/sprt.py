"""Sequential probability ratio test on paired game scores (EVALUATION.md §5).

Pair scores are the mean of the two games of a pair (0, .25, .5, .75, 1). The log-likelihood ratio
uses the normal (generalized SPRT) approximation used by Fishtest:
    LLR = (s1 - s0) * (2 * sum(x) - n * (s0 + s1)) / (2 * var)
with s_k the expected score for logistic Elo elo_k and var the sample variance of pair scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def expected_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


@dataclass
class SprtResult:
    llr: float
    lower: float
    upper: float
    decision: str  # "H1" (accept: candidate stronger), "H0" (reject), "continue"
    pairs: int
    score: float


def sprt(pair_scores: list[float], elo0: float = 0.0, elo1: float = 15.0, alpha: float = 0.05,
         beta: float = 0.05, min_var: float = 1e-3) -> SprtResult:
    n = len(pair_scores)
    lower, upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    if n < 2:
        return SprtResult(0.0, lower, upper, "continue", n, float(sum(pair_scores) / max(n, 1)))
    mean = sum(pair_scores) / n
    var = max(min_var, sum((x - mean) ** 2 for x in pair_scores) / (n - 1))
    s0, s1 = expected_score(elo0), expected_score(elo1)
    llr = (s1 - s0) * (2 * sum(pair_scores) - n * (s0 + s1)) / (2 * var)
    decision = "H1" if llr >= upper else "H0" if llr <= lower else "continue"
    return SprtResult(llr, lower, upper, decision, n, mean)
