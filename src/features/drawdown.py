"""
Drawdown and recovery features.

This module computes features related to drawdowns from rolling highs,
time spent in drawdown, and recovery dynamics. These capture behavioral
aspects of price action that are relatively robust to survivorship bias.

Features:
- Drawdown depth: How far below rolling high
- Drawdown duration: How long since rolling high
- Recovery metrics: Bounce from rolling lows
- Drawdown regime: Categorical drawdown state
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-12  # small epsilon for safe divisions


def _safe_div(num: pd.Series, den: pd.Series, eps: float = EPS) -> pd.Series:
    """Elementwise safe division; returns NaN where |den| <= eps."""
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    mask = den.abs() > eps
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    out[mask] = num[mask] / den[mask]
    return out


def _rolling_max(s: pd.Series, w: int) -> pd.Series:
    """Rolling maximum with reasonable min_periods."""
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=max(2, w // 3)).max()


def _rolling_min(s: pd.Series, w: int) -> pd.Series:
    """Rolling minimum with reasonable min_periods."""
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=max(2, w // 3)).min()


def _days_since_high(price: pd.Series, window: int) -> pd.Series:
    """
    Compute days since rolling high within window.

    Returns integer days since the rolling max was achieved.
    """
    rolling_high = _rolling_max(price, window)

    # Create mask where price equals rolling high (at the high)
    at_high = (price >= rolling_high - EPS).astype(float)

    # Count days since last high using cumsum trick
    # Reset counter when at_high == 1
    groups = at_high.cumsum()
    days_since = groups.groupby(groups).cumcount()

    return days_since.astype('float32')


def _expanding_max(s: pd.Series) -> pd.Series:
    """Expanding (cumulative) maximum."""
    return pd.to_numeric(s, errors="coerce").expanding(min_periods=1).max()


def add_drawdown_features(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (20, 60, 120),
    z_window: int = 252,
) -> pd.DataFrame:
    """
    Add drawdown and recovery features.

    Features computed for each window:
    - drawdown_{w}d: Percentage drawdown from rolling high (negative values)
    - drawdown_{w}d_z: Z-scored drawdown vs historical distribution
    - days_since_high_{w}d: Trading days since rolling high
    - days_since_high_{w}d_norm: Normalized (0-1) time in drawdown
    - recovery_{w}d: Percentage above rolling low (positive values)
    - recovery_{w}d_z: Z-scored recovery vs historical distribution

    Additional features:
    - drawdown_expanding: Drawdown from all-time high
    - drawdown_regime: Categorical regime (0=shallow, 1=moderate, 2=deep)
    - drawdown_velocity_{w}d: Rate of drawdown change (improving/worsening)

    Args:
        df: DataFrame with price data (needs 'close' or 'adjclose')
        windows: Tuple of lookback windows for rolling high/low
        z_window: Window for computing z-scores of drawdown

    Returns:
        DataFrame with drawdown features added (mutates in-place)
    """
    # Get price series
    if "close" in df.columns:
        price = pd.to_numeric(df["close"], errors="coerce")
    elif "adjclose" in df.columns:
        price = pd.to_numeric(df["adjclose"], errors="coerce")
    else:
        logger.warning("No price column found; skipping drawdown features")
        return df

    # Collect new columns to add at once (avoid fragmentation)
    new_cols = {}

    for w in windows:
        # Rolling high and low
        rolling_high = _rolling_max(price, w)
        rolling_low = _rolling_min(price, w)

        # --- Drawdown from rolling high ---
        # Drawdown = (price - high) / high, typically negative or zero
        drawdown = _safe_div(price - rolling_high, rolling_high)
        new_cols[f"drawdown_{w}d"] = drawdown.astype("float32")

        # Z-score of drawdown (how unusual is this drawdown?)
        dd_mean = drawdown.rolling(z_window, min_periods=60).mean()
        dd_std = drawdown.rolling(z_window, min_periods=60).std(ddof=0)
        drawdown_z = _safe_div(drawdown - dd_mean, dd_std)
        new_cols[f"drawdown_{w}d_z"] = drawdown_z.clip(-5, 5).astype("float32")

        # --- Days since rolling high ---
        days_since = _days_since_high(price, w)
        new_cols[f"days_since_high_{w}d"] = days_since

        # Normalized: days_since / window (0 = at high, 1 = been w days)
        days_norm = (days_since / w).clip(0, 1)
        new_cols[f"days_since_high_{w}d_norm"] = days_norm.astype("float32")

        # --- Recovery from rolling low ---
        # Recovery = (price - low) / low, typically positive or zero
        recovery = _safe_div(price - rolling_low, rolling_low)
        new_cols[f"recovery_{w}d"] = recovery.astype("float32")

        # Z-score of recovery
        rec_mean = recovery.rolling(z_window, min_periods=60).mean()
        rec_std = recovery.rolling(z_window, min_periods=60).std(ddof=0)
        recovery_z = _safe_div(recovery - rec_mean, rec_std)
        new_cols[f"recovery_{w}d_z"] = recovery_z.clip(-5, 5).astype("float32")

        # --- Drawdown velocity (is drawdown getting better or worse?) ---
        # Positive = recovering, negative = drawdown deepening
        drawdown_velocity = drawdown.diff(5)  # 5-day change in drawdown
        new_cols[f"drawdown_velocity_{w}d"] = drawdown_velocity.astype("float32")

    # --- Expanding (all-time) drawdown ---
    expanding_high = _expanding_max(price)
    drawdown_expanding = _safe_div(price - expanding_high, expanding_high)
    new_cols["drawdown_expanding"] = drawdown_expanding.astype("float32")

    # --- Drawdown regime (categorical) ---
    # Based on 60d drawdown: 0=shallow (<5%), 1=moderate (5-15%), 2=deep (>15%)
    if "drawdown_60d" in new_cols:
        dd60 = new_cols["drawdown_60d"]
    elif 60 in windows:
        dd60 = new_cols["drawdown_60d"]
    else:
        # Use first available window
        dd60 = new_cols[f"drawdown_{windows[0]}d"]

    regime = pd.Series(0, index=df.index, dtype="float32")
    regime = regime.where(dd60 > -0.05, 1)  # moderate if dd < -5%
    regime = regime.where(dd60 > -0.15, 2)  # deep if dd < -15%
    new_cols["drawdown_regime"] = regime.astype("float32")

    # --- High-low range ratio (measures drawdown + recovery context) ---
    # How much of the range from low to high has price captured?
    if 60 in windows:
        hi60 = _rolling_max(price, 60)
        lo60 = _rolling_min(price, 60)
        range_60 = hi60 - lo60
        hl_position = _safe_div(price - lo60, range_60)
        new_cols["hl_range_position_60d"] = hl_position.astype("float32")

    # Add all columns at once to avoid fragmentation
    new_df = pd.DataFrame(new_cols, index=df.index)
    for col in new_df.columns:
        df[col] = new_df[col]

    logger.debug(f"Added {len(new_cols)} drawdown/recovery features")
    return df
