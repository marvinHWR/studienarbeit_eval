"""Predefined statistical plan: one test per hypothesis, Bonferroni-corrected.

Same question answered by every arm -> paired design. The plan was decided BEFORE running, per
the concept plan, and deliberately runs exactly three confirmatory tests:

  T1 (H1)  label accuracy, RAG vs. No-Context        -> exact McNemar
  T2 (H2)  faithfulness,   RAG vs. No-Context        -> paired t-test
  T3 (H3)  label accuracy, Full-Context vs. RAG      -> exact McNemar

The test is chosen by the scale of the measure, not by taste: label accuracy is binary per
question (majority vote of the N samples vs. the gold label), which is exactly what McNemar is
for — a t-test on 0/1 data would rest on a normality assumption it cannot support. Faithfulness
is continuous on [0, 1], so the paired t-test applies. Everything else (answer relevancy, the
retrieval diagnostics, tokens and latency) is reported descriptively, without a p-value: those
differences are constructional or orders-of-magnitude, and a significance test would add nothing.

Three tests -> Bonferroni (p x 3), the strictest of the common corrections and explainable in one
sentence. No Wilcoxon, no Holm, no bootstrap CIs, no rank-biserial effect sizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, ttest_rel


@dataclass
class McNemarResult:
    metric: str
    arm_a: str
    arm_b: str
    n_pairs: int                # questions compared (both arms present)
    b: int                      # only arm A correct
    c: int                      # only arm B correct
    n_discordant: int           # b + c — the only questions the test looks at
    p_value: float


@dataclass
class PairedTResult:
    metric: str
    arm_a: str
    arm_b: str
    n: int                      # pairs after dropping NaN
    mean_a: float
    mean_b: float
    statistic: float            # t
    df: int                     # n - 1
    p_value: float


def mcnemar_exact(
    a_correct: np.ndarray, b_correct: np.ndarray, metric: str, arm_a: str, arm_b: str
) -> McNemarResult:
    """Exact (binomial) McNemar test on two paired boolean correctness vectors.

    Only the discordant questions carry information: b = "A right, B wrong", c = the reverse.
    Under H0 each discordant question is a fair coin, so the two-sided exact p-value is
    2 * P(X <= min(b, c)) with X ~ Binomial(b + c, 0.5), capped at 1. The exact form (rather than
    the chi-square approximation) is used because it stays valid for small discordant counts —
    T3 has only 42 of them. b + c == 0 means the arms never disagree -> p = 1.0.
    """
    a = np.asarray(a_correct, bool)
    b_arr = np.asarray(b_correct, bool)
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))
    n_disc = b + c
    p = 1.0 if n_disc == 0 else min(1.0, 2.0 * float(binom.cdf(min(b, c), n_disc, 0.5)))
    return McNemarResult(
        metric=metric, arm_a=arm_a, arm_b=arm_b, n_pairs=int(a.size),
        b=b, c=c, n_discordant=n_disc, p_value=p,
    )


def paired_t(a: np.ndarray, b: np.ndarray, metric: str, arm_a: str, arm_b: str) -> PairedTResult:
    """Paired t-test on two per-question metric vectors (same questions, same order).

    Pairs where either arm is NaN are dropped first — a degenerate answer yields a NaN RAGAS
    score, and ttest_rel would otherwise propagate it into the statistic. `n`, the means and the
    degrees of freedom all reflect the post-drop pairs (faithfulness: 197 of 200).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 2:
        return PairedTResult(
            metric=metric, arm_a=arm_a, arm_b=arm_b, n=int(a.size),
            mean_a=float("nan"), mean_b=float("nan"),
            statistic=float("nan"), df=max(int(a.size) - 1, 0), p_value=1.0,
        )
    t, p = ttest_rel(a, b)
    return PairedTResult(
        metric=metric, arm_a=arm_a, arm_b=arm_b, n=int(a.size),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        statistic=float(t), df=int(a.size) - 1, p_value=float(p),
    )


def bonferroni_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Bonferroni-adjusted p-values: multiply by the family size, cap at 1.0.

    Controls the family-wise error rate over the confirmatory family (here: three tests).
    """
    p = np.asarray(pvalues, float)
    if p.size == 0:
        return p
    return np.minimum(p * p.size, 1.0)
