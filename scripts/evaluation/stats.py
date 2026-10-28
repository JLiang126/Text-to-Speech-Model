"""Uncertainty for the metrics: bootstrap confidence intervals and paired tests.

Every system is scored on the same test sentences, so experiments are compared
sentence by sentence (paired). That removes the variation caused by some sentences
simply being harder than others.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

N_BOOT = 10_000
CI_LEVEL = 0.95


def _clean(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x[~np.isnan(x)]


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT, ci: float = CI_LEVEL,
                 seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lower, upper) with a percentile bootstrap over utterances. NaNs are ignored."""
    x = _clean(values)
    if x.size == 0:
        return (float("nan"),) * 3
    if x.size == 1:
        return (float(x[0]),) * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def bootstrap_ratio_ci(numerators: Sequence[float], denominators: Sequence[float],
                       n_boot: int = N_BOOT, ci: float = CI_LEVEL, seed: int = 0) -> Tuple[float, float, float]:
    """CI for sum(num)/sum(den), e.g. corpus WER = total word errors / total reference words."""
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    keep = ~(np.isnan(num) | np.isnan(den))
    num, den = num[keep], den[keep]
    if num.size == 0 or den.sum() == 0:
        return (float("nan"),) * 3
    point = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    ratios = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1e-12)
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(ratios, alpha)), float(np.quantile(ratios, 1.0 - alpha))


def paired_comparison(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT,
                      ci: float = CI_LEVEL, seed: int = 0) -> Dict[str, float]:
    """Compares system B against system A on the same utterances (difference = B - A).

    Returns:
    - the mean difference with a bootstrap CI,
    - a two-sided Wilcoxon signed-rank p-value,
    - Cohen's d_z (mean difference / SD of the differences): about 0.2 small, 0.5 medium, 0.8 large.

    If the CI excludes 0, the difference is unlikely to be sentence-sampling noise. It
    says nothing about training-run noise; that needs a repeated training seed.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    diff = b - a
    mean_diff, lo, hi = bootstrap_ci(diff, n_boot=n_boot, ci=ci, seed=seed)
    try:
        p_value = float(wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p_value = float("nan")
    sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
    dz = mean_diff / sd if sd and sd > 0 else (0.0 if diff.size and np.all(diff == 0) else float("nan"))
    return {"n": int(a.size), "mean_a": float(a.mean()) if a.size else float("nan"),
            "mean_b": float(b.mean()) if b.size else float("nan"),
            "mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "wilcoxon_p": p_value, "cohens_dz": dz}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (controls false positives when testing several metrics).

    NaNs stay NaN and are not counted in the number of tests.
    """
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.size
    if m == 0:
        return adjusted.tolist()
    order = valid[np.argsort(p[valid], kind="stable")]
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running_max
    return adjusted.tolist()
