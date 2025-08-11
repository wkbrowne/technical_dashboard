"""
Distance-to-moving-average features for measuring price deviation from trend lines.

This module computes features based on how far price deviates from various
moving averages, including normalized z-scores of these deviations.
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_ma(df: pd.DataFrame, src: str = 'adjclose', p: int = 20, minp: int = None) -> pd.Series:
    """
    Ensure a moving average column exists or compute it from source data.
    
    Args:
        df: Input DataFrame
        src: Source column name for price data
        p: Moving average period
        minp: Minimum periods required (defaults to max(5, p//2))
        
    Returns:
        Moving average Series
    """
    if minp is None:
        minp = max(5, p // 2)
    
    if f"ma_{p}" in df.columns:
        return pd.to_numeric(df[f"ma_{p}"], errors='coerce')
    
    if src in df.columns:
        return pd.to_numeric(df[src], errors='coerce').rolling(p, min_periods=minp).mean()
    
    return pd.Series(index=df.index, dtype='float64')


def add_distance_to_ma_features(
    df: pd.DataFrame,
    src_col: str = 'adjclose',
    ma_lengths: Tuple[int, ...] = (20, 50, 100, 200),
    z_window: int = 60
) -> pd.DataFrame:
    """
    Add distance-to-moving-average features.
    
    Features added:
    - pct_dist_ma_{L}: Percentage distance from MA of length L
    - pct_dist_ma_{L}_z: Z-score of percentage distance (if z_window > 0)
    - min_pct_dist_ma: Minimum absolute percentage distance across all MAs
    - relative_dist_20_50: Relative distance between MA20 and MA50
    - relative_dist_20_50_z: Z-score of relative distance (if z_window > 0)
    
    Args:
        df: Input DataFrame with price data
        src_col: Column name for source price data
        ma_lengths: Tuple of moving average lengths
        z_window: Window for computing z-scores (0 to disable)
        
    Returns:
        DataFrame with added distance features (mutates input DataFrame)
    """
    if src_col not in df.columns:
        logger.warning(f"Source column '{src_col}' not found, skipping distance features")
        return df

    logger.debug(f"Computing distance-to-MA features for lengths: {ma_lengths}")
    
    px = pd.to_numeric(df[src_col], errors='coerce')
    pct_cols = []
    
    for L in ma_lengths:
        ma = _ensure_ma(df, src=src_col, p=L)
        col = f"pct_dist_ma_{L}"
        
        # Percentage distance: (price - MA) / MA
        df[col] = (px - ma) / ma.replace(0, np.nan)
        pct_cols.append(col)
        
        # Z-score of percentage distance
        if z_window:
            m = df[col].rolling(z_window, min_periods=max(5, z_window//3)).mean()
            s = df[col].rolling(z_window, min_periods=max(5, z_window//3)).std(ddof=0)
            df[f"{col}_z"] = (df[col] - m) / s.replace(0, np.nan)
    
    # Aggregate distance features
    if pct_cols:
        # Minimum absolute distance to any MA
        df["min_pct_dist_ma"] = df[pct_cols].abs().min(axis=1)
        
        # Relative distance between short and medium MAs
        if "ma_20" in df.columns and "ma_50" in df.columns:
            df["relative_dist_20_50"] = (df["ma_20"] - df["ma_50"]) / df["ma_50"].replace(0, np.nan)
            
            if z_window:
                m = df["relative_dist_20_50"].rolling(z_window, min_periods=max(5, z_window//3)).mean()
                s = df["relative_dist_20_50"].rolling(z_window, min_periods=max(5, z_window//3)).std(ddof=0)
                df["relative_dist_20_50_z"] = (df["relative_dist_20_50"] - m) / s.replace(0, np.nan)
    
    logger.debug("Distance-to-MA features computation completed")
    return df