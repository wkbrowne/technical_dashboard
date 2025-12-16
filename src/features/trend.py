"""
Trend analysis features based on moving averages and their slopes.

This module computes trend-related features including moving average slopes,
trend alignment, and trend persistence metrics.
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta

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


def add_rsi_features(
    df: pd.DataFrame,
    src_col: str = 'adjclose',
    periods: Tuple[int, ...] = (14, 21, 30)
) -> pd.DataFrame:
    """
    Add RSI (Relative Strength Index) features using pandas-ta.
    
    Features added:
    - rsi_{period}: RSI values for each period (raw values for tree models)
    
    Args:
        df: Input DataFrame with price data
        src_col: Column name for source price data  
        periods: Tuple of RSI periods to compute
        
    Returns:
        DataFrame with added RSI features (mutates input DataFrame)
    """
    logger.debug(f"Adding RSI features for periods: {periods}")
    
    if src_col not in df.columns:
        logger.warning(f"Source column '{src_col}' not found for RSI calculation")
        return df
    
    price_series = pd.to_numeric(df[src_col], errors='coerce')
    
    for period in periods:
        try:
            # Use pandas-ta to calculate RSI
            rsi_values = ta.rsi(price_series, length=period)
            df[f"rsi_{period}"] = rsi_values.astype('float32')
            logger.debug(f"Added RSI_{period} feature")
        except Exception as e:
            logger.warning(f"Failed to compute RSI_{period}: {e}")
    
    logger.debug("RSI features computation completed")
    return df


def add_macd_features(
    df: pd.DataFrame,
    src_col: str = 'adjclose',
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    derivative_ema_span: int = 3
) -> pd.DataFrame:
    """
    Add MACD histogram and its exponential moving average derivative using pandas-ta.
    
    Features added:
    - macd_histogram: MACD histogram (MACD line - signal line)
    - macd_hist_deriv_ema3: 3-day EMA of MACD histogram derivative
    
    Args:
        df: Input DataFrame with price data
        src_col: Column name for source price data
        fast: Fast EMA period for MACD (default: 12)
        slow: Slow EMA period for MACD (default: 26) 
        signal: Signal line EMA period (default: 9)
        derivative_ema_span: EMA span for histogram derivative (default: 3)
        
    Returns:
        DataFrame with added MACD features (mutates input DataFrame)
    """
    logger.debug(f"Adding MACD features (fast={fast}, slow={slow}, signal={signal})")
    
    if src_col not in df.columns:
        logger.warning(f"Source column '{src_col}' not found for MACD calculation")
        return df

    price_series = pd.to_numeric(df[src_col], errors='coerce')

    # Check if we have enough valid data for MACD calculation
    # MACD needs at least slow + signal periods of valid data
    min_required = slow + signal
    valid_count = price_series.notna().sum()
    if valid_count < min_required:
        logger.debug(f"Insufficient data for MACD: {valid_count} valid points, need {min_required}")
        return df

    try:
        # Use pandas-ta to calculate MACD (returns DataFrame with MACD, signal, and histogram)
        macd_data = ta.macd(price_series, fast=fast, slow=slow, signal=signal)
        
        if macd_data is not None and not macd_data.empty:
            # Extract MACD histogram (already computed by pandas-ta)
            histogram_col = f'MACDh_{fast}_{slow}_{signal}'  # pandas-ta naming convention
            
            if histogram_col in macd_data.columns:
                macd_histogram = macd_data[histogram_col].astype('float32')
                df['macd_histogram'] = macd_histogram
                logger.debug("Added MACD histogram feature")
                
                # Calculate histogram derivative (change from previous period)
                histogram_derivative = macd_histogram.diff()
                
                # Apply 3-day EMA to the derivative
                macd_hist_deriv_ema = histogram_derivative.ewm(
                    span=derivative_ema_span, 
                    adjust=False, 
                    min_periods=1
                ).mean().astype('float32')
                
                df['macd_hist_deriv_ema3'] = macd_hist_deriv_ema
                logger.debug(f"Added MACD histogram derivative EMA{derivative_ema_span} feature")
            else:
                logger.warning(f"Expected MACD histogram column '{histogram_col}' not found in pandas-ta output")
                logger.debug(f"Available MACD columns: {list(macd_data.columns)}")
        else:
            logger.warning("MACD calculation returned empty result")
            
    except Exception as e:
        logger.warning(f"Failed to compute MACD features: {e}")
    
    logger.debug("MACD features computation completed")
    return df