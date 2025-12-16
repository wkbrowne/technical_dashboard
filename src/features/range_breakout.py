"""
Range and breakout features based on high-low price ranges and volatility.

This module computes features related to price ranges, breakouts, gaps,
and true range measurements across multiple timeframes.
"""
import logging
from typing import Tuple, Iterable
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("pandas_ta is required for ATR calculation. Please: pip install pandas-ta")

logger = logging.getLogger(__name__)

EPS = 1e-12  # small epsilon for safe divisions

def _safe_div(num: pd.Series, den: pd.Series, eps: float = EPS) -> pd.Series:
    """Elementwise safe division; returns NaN where |den| <= eps."""
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    mask = den.abs() > eps
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    out[mask] = (num[mask] / den[mask])
    return out

def _rolling_max(s: pd.Series, w: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=max(2, w // 3)).max()

def _rolling_min(s: pd.Series, w: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=max(2, w // 3)).min()

def add_range_breakout_features(
    df: pd.DataFrame,
    win_list: Tuple[int, ...] = (5, 10, 20)
) -> pd.DataFrame:
    """
    Add comprehensive range and breakout features with safe math (no +/-inf).
    Returns df (mutates in-place).
    """
    required = {"high", "low"}
    if not required.issubset(df.columns):
        logger.warning("Required columns %s not found; skipping range features", required)
        return df

    # Use close if available, otherwise adjclose
    close = pd.to_numeric(df["close"] if "close" in df.columns else df["adjclose"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"],  errors="coerce")

    prev_close = close.shift(1)
    hl_range   = (high - low)

    # --- Basic ranges (safe ratios) ---
    df["hl_range"] = hl_range
    df["hl_range_pct_close"] = _safe_div(hl_range, close)

    # --- True range & ATR ---
    # Use np.maximum instead of pd.concat to avoid DataFrame fragmentation
    tr = np.maximum(
        np.maximum(high - low, (high - prev_close).abs()),
        (low - prev_close).abs()
    )
    df["true_range"] = tr
    df["tr_pct_close"] = _safe_div(tr, close)

    # Use pandas_ta for clean ATR calculation
    atr14 = ta.atr(high=high, low=low, close=close, length=14)

    # Handle case where ATR calculation returns None (insufficient data)
    if atr14 is None:
        logger.debug("ATR calculation returned None, creating NaN series")
        atr14 = pd.Series(np.nan, index=df.index, dtype="float32")
    else:
        atr14 = atr14.astype("float32")

    df["atr14"] = atr14  # Raw ATR value
    df["atr_percent"] = _safe_div(atr14, close).astype("float32")  # ATR as % of close

    # --- Gaps ---
    # gap = (close / prev_close - 1) but safe
    df["gap_pct"] = _safe_div(close, prev_close) - 1.0
    df["gap_atr_ratio"] = _safe_div(df["gap_pct"], df["atr_percent"])  # NaN if ATR% ~ 0

    # --- Multi-timeframe features ---
    for w in win_list:
        hi = _rolling_max(high, w)
        lo = _rolling_min(low,  w)
        rng = hi - lo

        df[f"{w}d_high"] = hi
        df[f"{w}d_low"]  = lo
        df[f"{w}d_range"] = rng
        df[f"{w}d_range_pct_close"] = _safe_div(rng, close)

        # pos in range ∈ [0,1] (NaN where rng ~ 0)
        df[f"pos_in_{w}d_range"] = _safe_div(close - lo, rng)

        # Breakouts vs prior window’s extreme
        df[f"breakout_up_{w}d"] = (close > hi.shift(1)).astype("float32")
        df[f"breakout_dn_{w}d"] = (close < lo.shift(1)).astype("float32")

        # Range expansion: (rng / rng[-1]) - 1, but safe
        prev_rng = rng.shift(1)
        df[f"range_expansion_{w}d"] = _safe_div(rng, prev_rng) - 1.0

        # Z-score of range over 60d history (safe std)
        mu = rng.rolling(60, min_periods=20).mean()
        sd = rng.rolling(60, min_periods=20).std(ddof=0)
        df[f"range_z_{w}d"] = _safe_div(rng - mu, sd)

    # --- Volume-range interaction ---
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        vol_ma20 = vol.rolling(20, min_periods=5).mean()

        # relative volume: vol / ma20 (NaN when ma20 ~ 0)
        rvol = _safe_div(vol, vol_ma20)

        # range% / rvol (NaN when rvol ~ 0)
        range_pct = df["hl_range_pct_close"]  # already safe
        df["range_x_rvol20"] = _safe_div(range_pct, rvol)

    # --- Optional: clip extreme z-scores to reduce downstream leakage/instability ---
    # (Keeps NaNs as NaNs; only caps finite outliers)
    for col in list(df.columns):
        if col.startswith("range_z_"):
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.clip(lower=-10, upper=10)

    # We intentionally DO NOT replace NaNs here; your training code imputes per-fold (leak‑safe).
    return df