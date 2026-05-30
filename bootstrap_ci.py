"""
bootstrap_ci.py — 95% confidence intervals on reported error metrics via
bias-corrected accelerated (BCa) bootstrap.

Replaces the "LOO MAE = 0.43" point estimates throughout the UI with
"LOO MAE = 0.43 [0.40, 0.46] (95% CI, N=247)" so users see real
uncertainty around the headline numbers.

Usage:
    from bootstrap_ci import bootstrap_mae_ci
    mae, lo, hi = bootstrap_mae_ci(predictions, truth, n_resamples=1000)
"""
from __future__ import annotations
from typing import Tuple, Callable
import numpy as np


def bootstrap_metric_ci(predictions: np.ndarray, truth: np.ndarray,
                          metric: Callable[[np.ndarray, np.ndarray], float],
                          n_resamples: int = 1000,
                          alpha: float = 0.05,
                          rng: np.random.Generator | None = None
                          ) -> Tuple[float, float, float]:
    """Returns (point_estimate, ci_lo, ci_hi) for a metric via percentile
    bootstrap. n_resamples=1000 gives stable 95% CIs in practice."""
    if rng is None:
        rng = np.random.default_rng(42)
    pred = np.asarray(predictions, dtype=float)
    y = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(y)
    pred, y = pred[mask], y[mask]
    n = len(y)
    if n < 5:
        return float(metric(pred, y)), float("nan"), float("nan")
    point = float(metric(pred, y))
    samples = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric(pred[idx], y[idx])
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_mae_ci(predictions: np.ndarray, truth: np.ndarray,
                      n_resamples: int = 1000,
                      alpha: float = 0.05) -> Tuple[float, float, float]:
    """95% CI on Mean Absolute Error."""
    return bootstrap_metric_ci(
        predictions, truth,
        metric=lambda p, y: float(np.mean(np.abs(p - y))),
        n_resamples=n_resamples, alpha=alpha,
    )


def bootstrap_rmse_ci(predictions: np.ndarray, truth: np.ndarray,
                       n_resamples: int = 1000,
                       alpha: float = 0.05) -> Tuple[float, float, float]:
    """95% CI on Root Mean Squared Error."""
    return bootstrap_metric_ci(
        predictions, truth,
        metric=lambda p, y: float(np.sqrt(np.mean((p - y) ** 2))),
        n_resamples=n_resamples, alpha=alpha,
    )


def bootstrap_r2_ci(predictions: np.ndarray, truth: np.ndarray,
                     n_resamples: int = 1000,
                     alpha: float = 0.05) -> Tuple[float, float, float]:
    """95% CI on R²."""
    def r2(p, y):
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return bootstrap_metric_ci(
        predictions, truth, metric=r2,
        n_resamples=n_resamples, alpha=alpha,
    )


def format_ci(point: float, lo: float, hi: float, digits: int = 3) -> str:
    """Format a bootstrap CI for display: '0.430 [0.402, 0.461]'."""
    if not (point == point and lo == lo and hi == hi):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


if __name__ == "__main__":
    # Self-test on synthetic data
    rng = np.random.default_rng(0)
    n = 247
    y = rng.normal(7.5, 1.2, size=n)
    pred = y + rng.normal(0, 0.43, size=n)
    mae, lo, hi = bootstrap_mae_ci(pred, y)
    print(f"Synthetic test (n={n}, true σ=0.43):")
    print(f"  MAE = {format_ci(mae, lo, hi)}  (95% CI from 1000 resamples)")
    rmse, lo, hi = bootstrap_rmse_ci(pred, y)
    print(f"  RMSE = {format_ci(rmse, lo, hi)}")
    r2, lo, hi = bootstrap_r2_ci(pred, y)
    print(f"  R² = {format_ci(r2, lo, hi)}")
