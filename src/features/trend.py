"""
Trend analysis features based on moving averages and their slopes.

This module computes trend-related features including moving average slopes,
trend alignment, and trend persistence metrics.
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


def add_trend_features(
    df: pd.DataFrame,
    src_col: str = 'adjclose',
    ma_periods: Tuple[int, ...] = (10, 20, 30, 50, 75, 100, 150, 200),
    slope_window: int = 20,
    eps: float = 1e-5
) -> pd.DataFrame:
    """
    Add comprehensive trend features based on moving averages.
    
    Features added:
    - ma_{p}: Moving averages for each period
    - pct_slope_ma_{p}: Percentage slope of each MA over slope_window
    - sign_ma_{p}: Sign of MA slope (1, 0, -1)
    - trend_score_granular: Average of non-zero MA slope signs
    - trend_score_sign: Sign of trend_score_granular
    - trend_score_slope: Change in trend_score_granular
    - trend_persist_ema: EMA-smoothed trend persistence
    - trend_alignment: Fraction of MAs with positive slopes
    
    Args:
        df: Input DataFrame with price data
        src_col: Column name for source price data
        ma_periods: Tuple of moving average periods to compute
        slope_window: Window for computing MA slopes
        eps: Epsilon threshold for slope sign classification
        
    Returns:
        DataFrame with added trend features (mutates input DataFrame)
    """
    logger.debug(f"Adding trend features for {len(ma_periods)} MA periods")
    
    sign_cols = []
    for p in ma_periods:
        ma = _ensure_ma(df, src=src_col, p=p)
        df[f"ma_{p}"] = ma
        
        # Compute percentage slope over slope_window
        slope = (ma / ma.shift(slope_window) - 1.0)
        df[f"pct_slope_ma_{p}"] = slope.astype('float32')
        
        # Classify slope sign
        sign = np.where(slope > eps, 1.0, np.where(slope < -eps, -1.0, 0.0))
        df[f"sign_ma_{p}"] = sign.astype('float32')
        sign_cols.append(f"sign_ma_{p}")

    if sign_cols:
        # Compute aggregate trend metrics
        sign_mat = df[sign_cols].to_numpy(dtype='float32')
        nz = (sign_mat != 0).sum(axis=1)  # Count of non-zero slopes
        sums = sign_mat.sum(axis=1)       # Sum of slope signs
        
        # Trend score: average of non-zero slope signs
        trend_score = np.divide(
            sums, np.where(nz == 0, np.nan, nz),
            out=np.zeros_like(sums, dtype='float32'), where=nz != 0
        )
        
        df["trend_score_granular"] = trend_score.astype('float32')
        df["trend_score_sign"] = np.sign(trend_score).astype('float32')
        df["trend_score_slope"] = pd.Series(trend_score, index=df.index).diff().astype('float32')
        
        # EMA-smoothed trend persistence
        df["trend_persist_ema"] = (
            df["trend_score_sign"].ewm(span=10, adjust=False, min_periods=1).mean().astype('float32')
        )
        
        # Trend alignment: fraction of MAs with positive slopes
        pos = (sign_mat > 0).sum(axis=1)
        neg = (sign_mat < 0).sum(axis=1)
        denom = (pos + neg).astype('float32')
        df["trend_alignment"] = (pos / np.where(denom == 0, np.nan, denom)).astype('float32')
    
    logger.debug("Trend features computation completed")
    return df