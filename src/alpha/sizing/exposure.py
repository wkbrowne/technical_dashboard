"""Exposure computation and volatility adjustment utilities."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_gross_exposure(weights: pd.Series) -> float:
    """Compute gross exposure (sum of absolute weights).

    Args:
        weights: Position weights.

    Returns:
        Gross exposure as fraction.
    """
    return weights.abs().sum()


def compute_net_exposure(weights: pd.Series) -> float:
    """Compute net exposure (sum of weights, considering signs).

    Args:
        weights: Position weights (positive = long, negative = short).

    Returns:
        Net exposure as fraction.
    """
    return weights.sum()


def compute_volatility_adjustment(
    weights: pd.Series,
    volatilities: pd.Series,
    target_vol: float = 0.15,
    annualization: float = 252.0,
) -> Tuple[float, pd.Series]:
    """Compute volatility adjustment factor.

    Calculates the scale factor needed to hit target volatility.

    Args:
        weights: Position weights.
        volatilities: Per-position volatilities (as decimals).
        target_vol: Target annualized volatility.
        annualization: Trading days per year.

    Returns:
        Tuple of (adjustment_factor, adjusted_weights).
    """
    # Align weights and volatilities
    common_idx = weights.index.intersection(volatilities.index)
    w = weights.loc[common_idx].values
    v = volatilities.loc[common_idx].values

    # Ensure volatilities are positive
    v = np.maximum(v, 0.001)

    # Portfolio variance (assuming zero correlation for simplicity)
    # More sophisticated: use correlation matrix
    portfolio_var = np.sum((w * v) ** 2)
    portfolio_vol = np.sqrt(portfolio_var)

    # Annualize (volatilities may already be annualized or daily)
    # Assume input is daily, annualize
    portfolio_vol_annual = portfolio_vol * np.sqrt(annualization)

    # Adjustment factor
    if portfolio_vol_annual > 0:
        adjustment = target_vol / portfolio_vol_annual
    else:
        adjustment = 1.0

    # Adjusted weights
    adjusted_weights = weights * adjustment

    return adjustment, adjusted_weights


def normalize_weights(
    weights: pd.Series,
    target_exposure: float = 1.0,
    max_individual: Optional[float] = None,
) -> pd.Series:
    """Normalize weights to target exposure.

    Args:
        weights: Position weights.
        target_exposure: Target gross exposure.
        max_individual: Optional cap per position.

    Returns:
        Normalized weights.
    """
    gross = weights.abs().sum()

    if gross > 0:
        normalized = weights * (target_exposure / gross)
    else:
        normalized = weights.copy()

    # Apply individual cap if specified
    if max_individual is not None:
        normalized = normalized.clip(-max_individual, max_individual)

    return normalized


def compute_concentration(weights: pd.Series) -> dict:
    """Compute concentration metrics.

    Args:
        weights: Position weights.

    Returns:
        Dict with concentration metrics.
    """
    w = weights[weights > 0].values

    if len(w) == 0:
        return {
            "n_positions": 0,
            "hhi": 0.0,
            "effective_n": 0.0,
            "top1_pct": 0.0,
            "top5_pct": 0.0,
        }

    # Normalize to sum to 1
    w_norm = w / w.sum()

    # Herfindahl-Hirschman Index
    hhi = np.sum(w_norm ** 2)

    # Effective number of positions (1/HHI)
    effective_n = 1 / hhi if hhi > 0 else 0

    # Top N concentration
    w_sorted = np.sort(w_norm)[::-1]
    top1_pct = w_sorted[0] * 100 if len(w_sorted) > 0 else 0
    top5_pct = w_sorted[:5].sum() * 100 if len(w_sorted) >= 5 else w_norm.sum() * 100

    return {
        "n_positions": len(w),
        "hhi": hhi,
        "effective_n": effective_n,
        "top1_pct": top1_pct,
        "top5_pct": top5_pct,
    }


def estimate_portfolio_volatility(
    weights: pd.Series,
    volatilities: pd.Series,
    correlation: Optional[np.ndarray] = None,
) -> float:
    """Estimate portfolio volatility.

    Args:
        weights: Position weights.
        volatilities: Per-position volatilities.
        correlation: Optional correlation matrix (assumes identity if None).

    Returns:
        Portfolio volatility estimate.
    """
    common_idx = weights.index.intersection(volatilities.index)
    w = weights.loc[common_idx].values
    v = volatilities.loc[common_idx].values
    n = len(w)

    if n == 0:
        return 0.0

    v = np.maximum(v, 0.001)

    if correlation is not None:
        # Full covariance calculation
        cov = np.outer(v, v) * correlation[:n, :n]
        var = w @ cov @ w
    else:
        # Assume zero correlation (conservative for long-only)
        var = np.sum((w * v) ** 2)

    return np.sqrt(max(var, 0))


def compute_tracking_error(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    volatilities: pd.Series,
) -> float:
    """Compute tracking error vs benchmark.

    Args:
        weights: Portfolio weights.
        benchmark_weights: Benchmark weights.
        volatilities: Per-position volatilities.

    Returns:
        Tracking error estimate.
    """
    # Active weights
    all_symbols = weights.index.union(benchmark_weights.index)
    active_weights = pd.Series(0.0, index=all_symbols)

    for sym in all_symbols:
        port_w = weights.get(sym, 0.0)
        bench_w = benchmark_weights.get(sym, 0.0)
        active_weights[sym] = port_w - bench_w

    # Tracking error is volatility of active portfolio
    return estimate_portfolio_volatility(active_weights, volatilities)
