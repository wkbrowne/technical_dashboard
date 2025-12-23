"""Portfolio constraints for position sizing.

This module provides constraint enforcement for:
    - Maximum weight per name
    - Maximum gross exposure (NAV fraction)
    - Cash buffer
    - Sector caps
    - Turnover penalties
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class PortfolioConstraints:
    """Portfolio constraints configuration.

    Attributes:
        max_gross_exposure: Maximum gross exposure as NAV fraction.
        max_name_weight: Maximum weight per position.
        min_name_weight: Minimum weight to include position.
        cash_buffer: Minimum cash reserve fraction.
        sector_caps: Dict of sector -> max aggregate weight.
        max_positions: Maximum number of positions.
    """
    max_gross_exposure: float = 1.0
    max_name_weight: float = 0.10
    min_name_weight: float = 0.01
    cash_buffer: float = 0.0
    sector_caps: Optional[Dict[str, float]] = None
    max_positions: Optional[int] = None


def apply_constraints(
    weights: pd.Series,
    signals: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> pd.Series:
    """Apply portfolio constraints to raw weights.

    Applies constraints in order:
        1. Clip individual weights to max_name_weight
        2. Apply sector caps
        3. Filter by min_name_weight
        4. Apply max_positions
        5. Scale to max_gross_exposure (respecting cash_buffer)
        6. Re-clip individual weights after scaling

    Args:
        weights: Raw position weights.
        signals: Signals DataFrame with sector info.
        constraints: PortfolioConstraints object.

    Returns:
        Constrained weights.
    """
    weights = weights.copy()

    # 1. Clip individual weights
    weights = weights.clip(0, constraints.max_name_weight)

    # 2. Apply sector caps
    if constraints.sector_caps and "sector" in signals.columns:
        weights = _apply_sector_caps(weights, signals, constraints.sector_caps)

    # 3. Filter by minimum weight
    weights[weights < constraints.min_name_weight] = 0

    # 4. Apply max positions
    if constraints.max_positions is not None:
        n_positions = (weights > 0).sum()
        if n_positions > constraints.max_positions:
            # Keep only top positions by weight
            threshold = weights.nlargest(constraints.max_positions).min()
            weights[weights < threshold] = 0

    # 5. Scale to max exposure (with cash buffer)
    max_investable = constraints.max_gross_exposure * (1 - constraints.cash_buffer)
    gross_exposure = weights.sum()

    if gross_exposure > max_investable:
        weights = weights * (max_investable / gross_exposure)

    # 6. Re-clip after scaling (may have been pushed up)
    weights = weights.clip(0, constraints.max_name_weight)

    # Re-scale if clipping reduced exposure significantly
    gross_exposure = weights.sum()
    if gross_exposure < max_investable * 0.95:  # Allow 5% slack
        # Don't re-scale (clipping was binding)
        pass

    return weights


def _apply_sector_caps(
    weights: pd.Series,
    signals: pd.DataFrame,
    sector_caps: Dict[str, float],
) -> pd.Series:
    """Apply sector-level weight caps.

    Args:
        weights: Position weights.
        signals: Signals with sector column.
        sector_caps: Dict of sector -> max aggregate weight.

    Returns:
        Sector-constrained weights.
    """
    weights = weights.copy()

    for sector, cap in sector_caps.items():
        sector_mask = signals["sector"] == sector

        if not sector_mask.any():
            continue

        sector_weight = weights[sector_mask].sum()

        if sector_weight > cap:
            # Scale down sector positions proportionally
            scale = cap / sector_weight
            weights[sector_mask] *= scale

    return weights


def apply_turnover_penalty(
    new_weights: pd.Series,
    old_weights: pd.Series,
    penalty: float,
    signals: pd.DataFrame,
) -> pd.Series:
    """Apply turnover penalty to weight changes.

    Penalizes deviating from current positions to reduce turnover.

    Args:
        new_weights: Proposed new weights (indexed by signal index).
        old_weights: Current weights (indexed by symbol).
        penalty: Penalty per unit turnover (0 to 1).
        signals: Signals DataFrame with symbol column.

    Returns:
        Adjusted weights with turnover penalty.
    """
    if penalty <= 0:
        return new_weights

    adjusted = new_weights.copy()

    # Map signals to symbols
    for idx in new_weights.index:
        if idx in signals.index:
            symbol = signals.loc[idx, "symbol"]
            new_w = new_weights.loc[idx]

            if symbol in old_weights.index:
                old_w = old_weights[symbol]
                # Blend toward old weight
                adjusted.loc[idx] = (1 - penalty) * new_w + penalty * old_w
            # New position: no adjustment

    return adjusted


def compute_turnover(
    new_weights: pd.Series,
    old_weights: pd.Series,
    signals: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """Compute portfolio turnover.

    Args:
        new_weights: New weights (indexed by signal index).
        old_weights: Current weights (indexed by symbol).
        signals: Signals DataFrame.

    Returns:
        Tuple of (total_turnover, per_symbol_turnover).
    """
    per_symbol = {}

    # Map new weights to symbols
    new_by_symbol = {}
    for idx in new_weights.index:
        if idx in signals.index:
            symbol = signals.loc[idx, "symbol"]
            new_by_symbol[symbol] = new_weights.loc[idx]

    # All symbols in either portfolio
    all_symbols = set(new_by_symbol.keys()) | set(old_weights.index)

    total_turnover = 0.0
    for symbol in all_symbols:
        old_w = old_weights.get(symbol, 0.0)
        new_w = new_by_symbol.get(symbol, 0.0)
        change = abs(new_w - old_w)
        per_symbol[symbol] = change
        total_turnover += change

    # Turnover is half of total changes (buys = sells in rebalance)
    return total_turnover / 2, per_symbol


def normalize_weights(
    weights: pd.Series,
    target_sum: float = 1.0,
) -> pd.Series:
    """Normalize weights to sum to target.

    Args:
        weights: Position weights.
        target_sum: Target sum of weights.

    Returns:
        Normalized weights.
    """
    current_sum = weights.sum()
    if current_sum > 0:
        return weights * (target_sum / current_sum)
    return weights


def check_constraints_satisfied(
    weights: pd.Series,
    signals: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> Dict[str, bool]:
    """Check if constraints are satisfied.

    Args:
        weights: Position weights.
        signals: Signals DataFrame.
        constraints: PortfolioConstraints object.

    Returns:
        Dict of constraint -> is_satisfied.
    """
    results = {}

    # Max name weight
    max_weight = weights.max()
    results["max_name_weight"] = max_weight <= constraints.max_name_weight + 1e-6

    # Min name weight (for non-zero positions)
    non_zero = weights[weights > 0]
    if len(non_zero) > 0:
        min_weight = non_zero.min()
        results["min_name_weight"] = min_weight >= constraints.min_name_weight - 1e-6
    else:
        results["min_name_weight"] = True

    # Gross exposure
    gross = weights.sum()
    max_investable = constraints.max_gross_exposure * (1 - constraints.cash_buffer)
    results["max_gross_exposure"] = gross <= max_investable + 1e-6

    # Sector caps
    if constraints.sector_caps and "sector" in signals.columns:
        results["sector_caps"] = True
        for sector, cap in constraints.sector_caps.items():
            sector_mask = signals["sector"] == sector
            sector_weight = weights[sector_mask].sum()
            if sector_weight > cap + 1e-6:
                results["sector_caps"] = False
                break

    # Max positions
    if constraints.max_positions is not None:
        n_positions = (weights > 0).sum()
        results["max_positions"] = n_positions <= constraints.max_positions

    return results
