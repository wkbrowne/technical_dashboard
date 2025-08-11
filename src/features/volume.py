"""
Volume-based features including volume patterns, shocks, and price-volume relationships.

This module computes various volume-related features including relative volume,
volume z-scores, dollar volume, on-balance volume, and volume shock indicators.
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive volume-based features.
    
    Features added:
    - vol_ma_20/50: Volume moving averages
    - vol_z_20/60: Volume z-scores over different windows
    - rvol_20/50: Relative volume vs moving averages
    - dollar_vol_ma_20: Dollar volume moving average
    - rdollar_vol_20: Relative dollar volume
    - obv: On-Balance Volume
    - obv_z_60: OBV z-score
    - vol_rolling_20d/60d: Volume rolling averages
    - vol_of_vol_20d: Volatility of log volume changes
    
    Args:
        df: Input DataFrame with volume data
        
    Returns:
        DataFrame with added volume features (mutates input DataFrame)
    """
    if "volume" not in df.columns:
        logger.warning("Volume column not found, skipping volume features")
        return df

    logger.debug("Computing volume features")
    
    vol = pd.to_numeric(df["volume"], errors='coerce')
    
    # Volume moving averages
    df["vol_ma_20"] = vol.rolling(20, min_periods=5).mean()
    df["vol_ma_50"] = vol.rolling(50, min_periods=10).mean()

    # Volume z-scores
    mu20 = vol.rolling(20, min_periods=5).mean()
    sd20 = vol.rolling(20, min_periods=5).std(ddof=0)
    df["vol_z_20"] = (vol - mu20) / sd20.replace(0, np.nan)

    mu60 = vol.rolling(60, min_periods=15).mean()
    sd60 = vol.rolling(60, min_periods=15).std(ddof=0)
    df["vol_z_60"] = (vol - mu60) / sd60.replace(0, np.nan)

    # Relative volume
    df["rvol_20"] = vol / df["vol_ma_20"].replace(0, np.nan)
    df["rvol_50"] = vol / df["vol_ma_50"].replace(0, np.nan)

    # Price-volume features (if price data available)
    px_col = "adjclose" if "adjclose" in df.columns else ("close" if "close" in df.columns else None)
    if px_col:
        px = pd.to_numeric(df[px_col], errors='coerce')
        
        # Dollar volume
        dvol = px * vol
        dvol_ma20 = dvol.rolling(20, min_periods=5).mean()
        df["dollar_vol_ma_20"] = dvol_ma20
        df["rdollar_vol_20"] = dvol / dvol_ma20.replace(0, np.nan)

        # On-Balance Volume (OBV)
        price_direction = np.sign(px.diff()).fillna(0.0)
        obv = (price_direction * vol).fillna(0.0).cumsum()
        df["obv"] = obv
        
        # OBV z-score
        obv_mu = obv.rolling(60, min_periods=20).mean()
        obv_sd = obv.rolling(60, min_periods=20).std(ddof=0)
        df["obv_z_60"] = (obv - obv_mu) / obv_sd.replace(0, np.nan)

    # Volume volatility features
    lv = np.log(vol.replace(0, np.nan))
    d_lv = lv.diff()
    
    # Additional rolling volume averages
    df["vol_rolling_20d"] = vol.rolling(20, min_periods=5).mean()
    df["vol_rolling_60d"] = vol.rolling(60, min_periods=15).mean()
    
    # Volatility of volume (volume of volume changes)
    df["vol_of_vol_20d"] = d_lv.rolling(20, min_periods=5).std(ddof=0)

    logger.debug("Volume features computation completed")
    return df


def add_volume_shock_features(
    df: pd.DataFrame,
    vol_col: str = "volume",
    price_col_primary: str = "close",
    price_col_fallback: str = "adjclose",
    lookback: int = 20,
    ema_span: int = 10,
    prefix: str = "volshock"
) -> pd.DataFrame:
    """
    Add volume shock and trend alignment features.
    
    Features added:
    - {prefix}_z: Z-score of volume vs rolling mean/std
    - {prefix}_dir: Directional shock (shock sign * price move sign)
    - {prefix}_ema: EMA-smoothed directional shock
    
    Args:
        df: Input DataFrame with volume and price data
        vol_col: Column name for volume data
        price_col_primary: Primary price column to use
        price_col_fallback: Fallback price column if primary not available
        lookback: Lookback window for volume statistics
        ema_span: EMA span for smoothing directional shock
        prefix: Prefix for output column names
        
    Returns:
        DataFrame with added volume shock features (mutates input DataFrame)
    """
    if vol_col not in df.columns:
        logger.warning(f"Volume column '{vol_col}' not found, skipping volume shock features")
        return df

    logger.debug(f"Computing volume shock features with lookback={lookback}")
    
    out = df.copy()

    # Choose price column (primary -> fallback)
    price_col = price_col_primary if price_col_primary in out.columns else (
        price_col_fallback if price_col_fallback in out.columns else None
    )
    if price_col is None:
        logger.warning("No suitable price column found for volume shock features")
        return out

    vol = pd.to_numeric(out[vol_col], errors="coerce")
    px = pd.to_numeric(out[price_col], errors="coerce")

    # Rolling volume statistics
    roll_mean = vol.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    roll_std = vol.rolling(lookback, min_periods=max(5, lookback // 3)).std(ddof=0)

    # Volume shock z-score
    z = (vol - roll_mean) / roll_std.replace(0, np.nan)
    out[f"{prefix}_z"] = z.astype("float32")

    # Price direction over shorter horizon to align shock with price impulse
    half = max(1, lookback // 2)
    px_dir = np.sign(px - px.shift(half)).astype("float32")

    # Directional shock: volume shock aligned with price direction
    out[f"{prefix}_dir"] = (z * px_dir).astype("float32")

    # EMA-smoothed directional shock
    out[f"{prefix}_ema"] = (
        out[f"{prefix}_dir"].ewm(span=ema_span, adjust=False, min_periods=1).mean().astype("float32")
    )

    logger.debug("Volume shock features computation completed")
    return out