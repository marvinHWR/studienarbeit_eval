"""Predefined statistical plan: paired tests across arms on the same questions.

Same question answered by every arm -> paired design. Wilcoxon signed-rank (non-parametric,
no normality assumption), rank-biserial effect size, and bootstrap CIs on the paired mean
difference. Decided BEFORE running, per the concept plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class PairedResult:
    metric: str
    arm_a: str
    arm_b: str
    n: int
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    effect_size: float          # rank-biserial correlation
    ci_low: float               # bootstrap CI on mean(a - b)
    ci_high: float


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 0.0, 1.0
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for Wilcoxon signed-rank."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total)


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare_arms(a: np.ndarray, b: np.ndarray, metric: str, arm_a: str, arm_b: str) -> PairedResult:
    a, b = np.asarray(a, float), np.asarray(b, float)

    # Drop pairs where either arm is NaN (a degenerate answer yields a NaN RAGAS score). Without
    # this, wilcoxon's default nan_policy='propagate' silently returns a NaN statistic/p-value.
    # Filtering here means n, medians, effect size, and the CI all reflect the post-drop pairs.
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return PairedResult(
            metric=metric, arm_a=arm_a, arm_b=arm_b, n=0,
            median_a=float("nan"), median_b=float("nan"),
            statistic=0.0, p_value=1.0, effect_size=0.0,
            ci_low=float("nan"), ci_high=float("nan"),
        )

    stat, p = paired_wilcoxon(a, b)
    lo, hi = bootstrap_ci(a, b)
    return PairedResult(
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        n=a.size,
        median_a=float(np.median(a)),
        median_b=float(np.median(b)),
        statistic=stat,
        p_value=p,
        effect_size=rank_biserial(a, b),
        ci_low=lo,
        ci_high=hi,
    )


# --- KB-size hypotheses (H3/H4) -----------------------------------------------------------
# The KB-size sweep measures each question at several corpus sizes, so it is a repeated-measures
# design along one ordered axis. We summarize each question's response as a single OLS slope of
# the metric over kb_size, then test those per-question slopes with the same paired, non-parametric
# machinery used everywhere else:
#   H3 (within-arm trend)      -> one-sample Wilcoxon on the slopes vs. 0        (trend_test)
#   H4 (arm x kb-size interact) -> paired Wilcoxon on slope differences (compare_arms on slopes)


@dataclass
class TrendResult:
    metric: str
    arm: str
    n: int
    median_slope: float         # median per-question slope of metric over kb_size
    statistic: float
    p_value: float
    effect_size: float          # rank-biserial of slopes vs 0
    ci_low: float               # bootstrap CI on the mean slope
    ci_high: float


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary-least-squares slope of y on x: metric change per unit of KB size.

    beta = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2). Returns 0.0 when x is constant
    (a single distinct KB size carries no trend information).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / denom)


def trend_test(slopes: np.ndarray, metric: str, arm: str) -> TrendResult:
    """H3: do the per-question slopes over KB size differ systematically from zero?

    One-sample Wilcoxon signed-rank on the slopes (paired against a zero vector), reusing the
    shared Wilcoxon / rank-biserial / bootstrap helpers. NaN slopes are dropped and `n` reports
    the post-drop count.
    """
    a = np.asarray(slopes, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return TrendResult(
            metric=metric, arm=arm, n=0, median_slope=float("nan"),
            statistic=0.0, p_value=1.0, effect_size=0.0,
            ci_low=float("nan"), ci_high=float("nan"),
        )
    zero = np.zeros_like(a)
    stat, p = paired_wilcoxon(a, zero)
    lo, hi = bootstrap_ci(a, zero)
    return TrendResult(
        metric=metric, arm=arm, n=a.size, median_slope=float(np.median(a)),
        statistic=stat, p_value=p, effect_size=rank_biserial(a, zero),
        ci_low=lo, ci_high=hi,
    )


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (controls the family-wise error rate).

    Sort p ascending; the k-th smallest is scaled by (m - k) and the running maximum is carried
    forward so adjusted values stay monotone. Adjusted p is capped at 1.0 and returned in the
    original order. Use when a table of tests forms one confirmatory family.
    """
    p = np.asarray(pvalues, float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p[idx]))
        adj[idx] = min(running, 1.0)
    return adj
