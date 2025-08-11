"""
Hurst exponent features for measuring long-term memory in time series.

This module computes Hurst exponent features using the R/S method to identify
trending vs mean-reverting behavior in price series.
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd

try:
    import nolds
except ImportError:
    raise ImportError("nolds is required for Hurst features. Please: pip install nolds")

logger = logging.getLogger(__name__)


def _safe_hurst_rs(x: np.ndarray) -> float:
    """
    Safe computation of R/S Hurst exponent using nolds library.
    
    Args:
        x: Input array of values
        
    Returns:
        Hurst exponent (0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting)
        Returns NaN if computation fails or insufficient data
    """
    try:
        x_clean = np.asarray(x, dtype=float)
        x_clean = x_clean[~np.isnan(x_clean)]  # Remove NaNs
        
        if len(x_clean) < 10:  # Need minimum data for reliable calculation
            return np.nan
            
        result = float(nolds.hurst_rs(x_clean, fit="poly"))
        return result
    except Exception as e:
        logger.debug(f"Hurst calculation failed: {e}")
        return np.nan


def add_hurst_features(
    df: pd.DataFrame,
    ret_col: str = "ret",
    windows: Tuple[int, ...] = (128,),
    ema_halflife: int = 5,
    prefix: str = "hurst_ret"
) -> pd.DataFrame:
    """
    Add Hurst exponent features to measure long-term memory in returns.
    
    Features added:
    - {prefix}_{w}: Rolling Hurst exponent over window w
    - {prefix}_{w}_emaHL{ema_halflife}: EMA-smoothed Hurst exponent
    
    Args:
        df: Input DataFrame with return data
        ret_col: Column name for returns
        windows: Tuple of windows for Hurst calculation
        ema_halflife: Half-life for EMA smoothing
        prefix: Prefix for Hurst feature column names
        
    Returns:
        DataFrame with added Hurst features (mutates input DataFrame)
        
    Note:
        Hurst exponent interpretation:
        - H ≈ 0.5: Random walk (no memory)
        - H > 0.5: Trending behavior (positive autocorrelation)
        - H < 0.5: Mean-reverting behavior (negative autocorrelation)
    """
    if ret_col not in df.columns:
        logger.warning(f"Return column '{ret_col}' not found, skipping Hurst features")
        return df

    logger.debug(f"Computing Hurst features for windows: {windows}")
    
    s = pd.to_numeric(df[ret_col], errors="coerce")

    for w in windows:
        col = f"{prefix}_{w}"
        min_periods = max(50, w//2)  # Need substantial data for reliable Hurst
        
        df[col] = s.rolling(window=w, min_periods=min_periods).apply(_safe_hurst_rs, raw=False)

    # Add EMA smoothing if requested
    if ema_halflife and len(windows):
        base = f"{prefix}_{windows[0]}"
        if base in df.columns:
            df[f"{base}_emaHL{ema_halflife}"] = (
                df[base].ewm(halflife=ema_halflife, adjust=False, min_periods=1).mean()
            )
    
    logger.debug("Hurst features computation completed")
    return df