"""
Range and breakout features based on high-low price ranges and volatility.

This module computes features related to price ranges, breakouts, gaps,
and true range measurements across multiple timeframes.
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _handle_infinities(series: pd.Series) -> pd.Series:
    """Replace infinite values with 0 to prevent downstream issues."""
    return series.replace([np.inf, -np.inf], 0.0)


def add_range_breakout_features(
    df: pd.DataFrame,
    win_list: Tuple[int, ...] = (5, 10, 20)
) -> pd.DataFrame:
    """
    Add comprehensive range and breakout features.
    
    Features added:
    - hl_range: High-low range
    - hl_range_pct_close: HL range as percentage of close
    - true_range: True range (max of HL, H-prevC, prevC-L)
    - tr_pct_close: True range as percentage of close
    - atr_percent: 14-period ATR as percentage of close
    - gap_pct: Gap percentage from previous close
    - gap_atr_ratio: Gap size relative to ATR
    - {w}d_high/low/range: Rolling high/low/range over w days
    - {w}d_range_pct_close: Range as percentage of close
    - pos_in_{w}d_range: Position within the w-day range (0-1)
    - breakout_up/dn_{w}d: Breakout above/below previous w-day high/low
    - range_expansion_{w}d: Range expansion vs previous period
    - range_z_{w}d: Z-score of range vs 60-day history
    - range_x_rvol20: Range interaction with relative volume (if volume available)
    
    Args:
        df: Input DataFrame with OHLCV data
        win_list: Tuple of windows for range calculations
        
    Returns:
        DataFrame with added range/breakout features (mutates input DataFrame)
    """
    required_cols = {"high", "low"}
    if not required_cols.issubset(df.columns):
        logger.warning(f"Required columns {required_cols} not found, skipping range features")
        return df

    logger.debug(f"Computing range/breakout features for windows: {win_list}")
    
    # Use close if available, otherwise adjclose
    close = pd.to_numeric(df["close"] if "close" in df.columns else df["adjclose"], errors='coerce')
    high = pd.to_numeric(df["high"], errors='coerce')
    low = pd.to_numeric(df["low"], errors='coerce')

    prev_close = close.shift(1)
    hl_range = (high - low)
    
    # Basic range features
    df["hl_range"] = hl_range
    df["hl_range_pct_close"] = _handle_infinities(hl_range / close.replace(0, np.nan))

    # True Range calculation
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    df["true_range"] = tr
    df["tr_pct_close"] = _handle_infinities(tr / close.replace(0, np.nan))
    
    # Average True Range (ATR)
    atr = tr.rolling(14, min_periods=5).mean()
    df["atr_percent"] = _handle_infinities(atr / close.replace(0, np.nan)).astype('float32')

    # Gap features
    df["gap_pct"] = _handle_infinities(close / prev_close - 1.0)
    df["gap_atr_ratio"] = _handle_infinities(df["gap_pct"] / df["atr_percent"].replace(0, np.nan))

    # Multi-timeframe range features
    for w in win_list:
        hi = high.rolling(w, min_periods=max(2, w//3)).max()
        lo = low.rolling(w, min_periods=max(2, w//3)).min()
        rng = hi - lo
        
        df[f"{w}d_high"] = hi
        df[f"{w}d_low"] = lo
        df[f"{w}d_range"] = rng
        df[f"{w}d_range_pct_close"] = _handle_infinities(rng / close.replace(0, np.nan))
        
        # Position within range (0 = at low, 1 = at high)
        df[f"pos_in_{w}d_range"] = _handle_infinities((close - lo) / rng.replace(0, np.nan))
        
        # Breakout signals
        df[f"breakout_up_{w}d"] = (close > hi.shift(1)).astype('float32')
        df[f"breakout_dn_{w}d"] = (close < lo.shift(1)).astype('float32')
        
        # Range expansion (handle zero previous range to avoid infinities)
        prev_rng = rng.shift(1).replace(0, np.nan)
        df[f"range_expansion_{w}d"] = _handle_infinities(rng / prev_rng - 1.0)
        
        # Range z-score vs 60-day history
        mu = rng.rolling(60, min_periods=20).mean()
        sd = rng.rolling(60, min_periods=20).std(ddof=0)
        df[f"range_z_{w}d"] = _handle_infinities((rng - mu) / sd.replace(0, np.nan))

    # Volume-range interaction (if volume available)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors='coerce')
        vol_ma20 = vol.rolling(20, min_periods=5).mean()
        
        # Range normalized by relative volume (handle zero rvol to avoid infinities)
        range_pct = _handle_infinities(hl_range / close.replace(0, np.nan))
        rvol = _handle_infinities(vol / vol_ma20.replace(0, np.nan))
        df["range_x_rvol20"] = _handle_infinities(range_pct / rvol.replace(0, np.nan))
    
    logger.debug("Range/breakout features computation completed")
    return df