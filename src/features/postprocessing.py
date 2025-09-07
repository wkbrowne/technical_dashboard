"""
Data postprocessing utilities for cleaning and preparing feature data.

This module contains functions for handling missing values, outliers, and
other data quality issues in the computed feature datasets.
"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def _interpolate_single_symbol(symbol: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, int]:
    """
    Interpolate internal NaN gaps for a single symbol's DataFrame.
    
    Args:
        symbol: Symbol identifier
        df: DataFrame with features to interpolate
        
    Returns:
        Tuple of (symbol, processed_dataframe, filled_count)
    """
    try:
        if df is None or df.empty:
            return symbol, df, 0

        # Create a copy to avoid modifying original
        df_processed = df.copy()

        # Ensure chronological order
        df_processed.sort_index(inplace=True)

        # Work on numeric columns only
        num_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return symbol, df_processed, 0

        # Replace inf/-inf with NaN first
        before_nans = df_processed[num_cols].isna().sum().sum()
        df_processed[num_cols] = df_processed[num_cols].replace([np.inf, -np.inf], np.nan)

        # Interpolate only gaps strictly inside observed values
        method = 'time' if np.issubdtype(df_processed.index.dtype, np.datetime64) else 'linear'
        df_processed[num_cols] = df_processed[num_cols].interpolate(
            method=method,
            limit_area='inside',         # <-- only fill NaNs bounded by numbers
            limit_direction='both'       # direction doesn't matter with 'inside', but harmless
        )

        after_nans = df_processed[num_cols].isna().sum().sum()
        filled_count = int(before_nans - after_nans)
        
        return symbol, df_processed, filled_count

    except Exception as e:
        logger.warning(f"Error interpolating {symbol}: {e}")
        return symbol, df, 0


def interpolate_internal_gaps(
    indicators_by_symbol: Dict[str, pd.DataFrame], 
    n_jobs: int = -1,
    batch_size: int = 32
) -> Dict[str, pd.DataFrame]:
    """
    Interpolate internal NaN gaps in numeric columns only using parallel processing.
    
    This function fills NaN values that are strictly between observed values,
    without filling leading or trailing NaNs. This preserves data integrity
    by only filling gaps in continuous time series data.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        batch_size: Number of symbols to process per batch
        
    Returns:
        Dictionary with same structure but interpolated DataFrames
        
    Notes:
        - Only fills NaNs that are bounded by non-NaN values (limit_area='inside')
        - Uses 'time' interpolation for datetime indexes, 'linear' for others
        - Replaces inf/-inf with NaN before interpolation
        - Works only on numeric columns, preserves other column types
        - Parallelized for improved performance with large symbol counts
    """
    if not indicators_by_symbol:
        logger.info("No symbols to interpolate")
        return indicators_by_symbol
    
    logger.info(f"Interpolating internal NaN gaps for {len(indicators_by_symbol)} symbols "
               f"(parallel processing: {n_jobs} jobs, batch size: {batch_size})...")
    
    # Filter out empty/None DataFrames before processing
    valid_symbols = [(sym, df) for sym, df in indicators_by_symbol.items() 
                    if df is not None and not df.empty]
    
    if not valid_symbols:
        logger.info("No valid symbols with data to interpolate")
        return indicators_by_symbol
    
    logger.debug(f"Processing {len(valid_symbols)} valid symbols out of {len(indicators_by_symbol)} total")
    
    # Use sequential processing if n_jobs=1 or small dataset
    if n_jobs == 1 or len(valid_symbols) < 10:
        logger.info("Using sequential processing for interpolation")
        results = []
        for symbol, df in valid_symbols:
            result = _interpolate_single_symbol(symbol, df)
            results.append(result)
    else:
        # Parallel processing
        logger.info(f"Using parallel processing with {n_jobs} jobs")
        try:
            results = Parallel(
                n_jobs=n_jobs,
                backend='loky',
                batch_size=batch_size,
                verbose=1,
                prefer="processes"
            )(
                delayed(_interpolate_single_symbol)(symbol, df)
                for symbol, df in valid_symbols
            )
        except Exception as e:
            logger.warning(f"Parallel processing failed ({e}), falling back to sequential")
            results = []
            for symbol, df in valid_symbols:
                result = _interpolate_single_symbol(symbol, df)
                results.append(result)
    
    # Reconstruct the dictionary with processed DataFrames
    processed_dict = indicators_by_symbol.copy()  # Start with original (includes empty/None)
    
    total_filled = 0
    successful_symbols = 0
    
    for symbol, processed_df, filled_count in results:
        processed_dict[symbol] = processed_df
        total_filled += filled_count
        if filled_count > 0:
            successful_symbols += 1
    
    logger.info(f"Interpolation complete. Symbols processed: {len(results)} | "
               f"Symbols with fills: {successful_symbols} | "
               f"Total values filled: {total_filled:,}")
    
    return processed_dict


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