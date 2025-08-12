"""
Data postprocessing utilities for cleaning and preparing feature data.

This module contains functions for handling missing values, outliers, and
other data quality issues in the computed feature datasets.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def interpolate_internal_gaps(indicators_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Interpolate internal NaN gaps in numeric columns only.
    
    This function fills NaN values that are strictly between observed values,
    without filling leading or trailing NaNs. This preserves data integrity
    by only filling gaps in continuous time series data.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        
    Returns:
        Dictionary with same structure but interpolated DataFrames
        
    Notes:
        - Only fills NaNs that are bounded by non-NaN values (limit_area='inside')
        - Uses 'time' interpolation for datetime indexes, 'linear' for others
        - Replaces inf/-inf with NaN before interpolation
        - Works only on numeric columns, preserves other column types
    """
    logger.info("Interpolating internal NaN gaps (numeric cols only, no end fills)...")
    filled_summary = []
    
    for sym, df in indicators_by_symbol.items():
        if df is None or df.empty:
            continue

        # Ensure chronological order
        df.sort_index(inplace=True)

        # Work on numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            continue

        # Replace inf/-inf with NaN first
        before_nans = df[num_cols].isna().sum().sum()
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        # Interpolate only gaps strictly inside observed values
        method = 'time' if np.issubdtype(df.index.dtype, np.datetime64) else 'linear'
        df[num_cols] = df[num_cols].interpolate(
            method=method,
            limit_area='inside',         # <-- only fill NaNs bounded by numbers
            limit_direction='both'       # direction doesn't matter with 'inside', but harmless
        )

        after_nans = df[num_cols].isna().sum().sum()
        filled_summary.append((sym, int(before_nans - after_nans)))

    # Optional: brief log of how much was filled
    total_filled = sum(cnt for _, cnt in filled_summary)
    logger.info(f"Interpolation complete. Symbols touched: {len(filled_summary)} | "
                f"values filled: {total_filled:,}")
    
    return indicators_by_symbol


def clean_infinite_values(indicators_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Replace infinite values with NaN across all numeric columns.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        
    Returns:
        Dictionary with same structure but cleaned DataFrames
    """
    logger.info("Cleaning infinite values from numeric columns...")
    cleaned_count = 0
    
    for sym, df in indicators_by_symbol.items():
        if df is None or df.empty:
            continue
            
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            continue
            
        # Count infinities before cleaning
        inf_mask = np.isinf(df[num_cols]).any(axis=1)
        inf_count = inf_mask.sum()
        
        if inf_count > 0:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            cleaned_count += inf_count
    
    logger.info(f"Cleaned {cleaned_count:,} rows containing infinite values")
    return indicators_by_symbol


def get_data_quality_summary(indicators_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate a data quality summary report.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        
    Returns:
        DataFrame with quality metrics per symbol
    """
    quality_data = []
    
    for sym, df in indicators_by_symbol.items():
        if df is None or df.empty:
            quality_data.append({
                'symbol': sym,
                'total_rows': 0,
                'total_cols': 0,
                'numeric_cols': 0,
                'nan_count': 0,
                'inf_count': 0,
                'completeness': 0.0
            })
            continue
            
        num_cols = df.select_dtypes(include=[np.number]).columns
        total_values = len(df) * len(num_cols) if len(num_cols) > 0 else 0
        nan_count = df[num_cols].isna().sum().sum() if len(num_cols) > 0 else 0
        inf_count = np.isinf(df[num_cols]).sum().sum() if len(num_cols) > 0 else 0
        
        quality_data.append({
            'symbol': sym,
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'numeric_cols': len(num_cols),
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'completeness': (total_values - nan_count) / total_values if total_values > 0 else 0.0
        })
    
    return pd.DataFrame(quality_data)