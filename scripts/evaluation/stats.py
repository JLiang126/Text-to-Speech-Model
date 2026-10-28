"""Uncertainty for the metrics: bootstrap confidence intervals and paired tests.

Every system is scored on the same test sentences, so experiments are compared
sentence by sentence (paired), which removes the variation caused by some
sentences simply being harder than others.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

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

    Returns the mean difference with a bootstrap CI and a two-sided Wilcoxon
    signed-rank p-value. If the CI excludes 0, the difference is unlikely to be
    sentence-sampling noise. (It says nothing about training-run noise; that
    needs a repeated training seed.)
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
    return {"n": int(a.size), "mean_a": float(a.mean()) if a.size else float("nan"),
            "mean_b": float(b.mean()) if b.size else float("nan"),
            "mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "wilcoxon_p": p_value}
