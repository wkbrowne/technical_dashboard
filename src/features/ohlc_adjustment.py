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
    Adjust OHLC prices to match adjusted close for splits and dividends.
    
    This ensures all price-based technical indicators use consistent,
    split/dividend-adjusted values for accurate calculations.
    
    Args:
        df: DataFrame with OHLC and adjclose columns
        
    Returns:
        DataFrame with adjusted OHLC prices
        
    Notes:
        - Requires 'adjclose' and 'close' columns
        - Adjusts 'open', 'high', 'low', 'close' to match 'adjclose'
        - Preserves volume (not adjusted)
        - Returns original DataFrame if required columns missing
    """
    # Check for required columns
    required_cols = ['adjclose', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.debug("Missing required columns for OHLC adjustment")
        return df
    
    # Check for OHLC columns to adjust
    ohlc_cols = ['open', 'high', 'low']
    available_ohlc = [col for col in ohlc_cols if col in df.columns]
    
    if not available_ohlc:
        logger.debug("No OHLC columns found to adjust")
        return df
    
    # Calculate adjustment factor
    # Handle division by zero and invalid values
    close_values = pd.to_numeric(df['close'], errors='coerce')
    adjclose_values = pd.to_numeric(df['adjclose'], errors='coerce')
    
    # Avoid division by zero
    valid_mask = (close_values != 0) & pd.notna(close_values) & pd.notna(adjclose_values)
    adjustment_factor = pd.Series(1.0, index=df.index)
    adjustment_factor[valid_mask] = adjclose_values[valid_mask] / close_values[valid_mask]
    
    # Apply adjustments
    df_adjusted = df.copy()
    
    for col in available_ohlc:
        if col in df.columns:
            original_values = pd.to_numeric(df[col], errors='coerce')
            df_adjusted[col] = original_values * adjustment_factor
    
    # Set close to adjusted close
    df_adjusted['close'] = df_adjusted['adjclose']
    
    # Log adjustment summary
    avg_factor = adjustment_factor[valid_mask].mean() if valid_mask.any() else 1.0
    if abs(avg_factor - 1.0) > 0.001:  # Only log if meaningful adjustment
        logger.debug(f"Applied OHLC adjustment: avg factor = {avg_factor:.4f}, "
                    f"adjusted {len(available_ohlc)} price columns")
    
    return df_adjusted