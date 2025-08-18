"""
OHLC adjustment module for aligning price data with adjusted close.

This module provides functionality to adjust Open, High, Low, Close prices
to match the adjusted close, ensuring consistency across all price-based
technical indicators and features.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def adjust_ohlc_to_adjclose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-adjust OHLC prices to show real recent prices while maintaining ratio consistency.
    
    Uses a simple approach: scale OHLC prices so that recent close ≈ recent adjclose,
    while preserving the relative price movements shown in adjclose throughout history.
    
    Args:
        df: DataFrame with OHLC and adjclose columns, sorted by date
        
    Returns:
        DataFrame with forward-adjusted OHLC prices
        
    Notes:
        - Requires 'adjclose' and 'close' columns
        - Scales all OHLC to match adjclose price levels and movements
        - Preserves volume (not adjusted)
        - Returns original DataFrame if required columns missing
    """
    # Check for required columns
    required_cols = ['adjclose', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.debug("Missing required columns for OHLC adjustment")
        return df
    
    # Check for OHLC columns to adjust
    ohlc_cols = ['open', 'high', 'low', 'close']
    available_ohlc = [col for col in ohlc_cols if col in df.columns]
    
    if not available_ohlc:
        logger.debug("No OHLC columns found to adjust")
        return df
    
    df_adjusted = df.copy()
    
    # Convert to numeric and handle invalid values
    close_values = pd.to_numeric(df['close'], errors='coerce')
    adjclose_values = pd.to_numeric(df['adjclose'], errors='coerce')
    
    # Find valid data points
    valid_mask = (close_values != 0) & pd.notna(close_values) & pd.notna(adjclose_values)
    if not valid_mask.any():
        logger.debug("No valid price data found for adjustment")
        return df
    
    # Simple approach: calculate adjustment factor for each row
    # This makes OHLC track the adjusted price movements exactly
    adjustment_factors = pd.Series(1.0, index=df.index)
    adjustment_factors[valid_mask] = adjclose_values[valid_mask] / close_values[valid_mask]
    
    # Apply adjustments to each OHLC column
    adjustment_applied = False
    for col in available_ohlc:
        if col in df.columns:
            original_values = pd.to_numeric(df[col], errors='coerce')
            df_adjusted[col] = original_values * adjustment_factors
            adjustment_applied = True
    
    # Log adjustment summary
    if adjustment_applied and adjustment_factors[valid_mask].std() > 0.001:
        avg_factor = adjustment_factors[valid_mask].mean()
        factor_range = (adjustment_factors[valid_mask].min(), adjustment_factors[valid_mask].max())
        logger.debug(f"Applied OHLC adjustment: avg factor = {avg_factor:.4f}, "
                    f"range = {factor_range[0]:.4f} to {factor_range[1]:.4f}")
    
    return df_adjusted