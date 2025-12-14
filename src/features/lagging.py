"""
Feature lagging utilities for creating temporally lagged feature columns.

This module provides functions to apply configurable daily and weekly lags to features,
with proper temporal semantics ensuring weekly lags respect week boundaries.
"""
import logging
import re
import multiprocessing
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _apply_lags_batch(
    items_batch: List[Tuple[str, pd.DataFrame]],
    daily_lags: List[int],
    weekly_lags: List[int]
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Process a batch of symbols for lag application to reduce IPC overhead.

    Args:
        items_batch: List of (symbol, dataframe) tuples
        daily_lags: List of daily lag periods
        weekly_lags: List of weekly lag periods

    Returns:
        List of (symbol, processed_dataframe) tuples
    """
    results = []
    for symbol, df in items_batch:
        result = _apply_lags_single_symbol(symbol, df, daily_lags, weekly_lags)
        results.append(result)
    return results


def identify_daily_features(df: pd.DataFrame) -> List[str]:
    """
    Identify daily feature columns based on naming patterns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of daily feature column names
    """
    daily_patterns = [
        r'^(ret|adjclose|volume|high|low|open)$',
        r'^(rsi|macd|ema|sma|ma_slope)_',
        r'^(hurst|vol_regime|distance|breakout|volshock)_',
        r'^(trend_|alpha_|relstr_)',
        r'^(breadth_|xsec_)',
        r'^atr\d+$',
        r'^(cci|stoch|williams)_',
        r'_(zscore|ema|ratio|momentum)$'
    ]
    
    daily_columns = []
    for col in df.columns:
        if col in ['symbol', 'date']:
            continue
        if col.startswith('w_'):  # Skip weekly features
            continue
            
        # Check if column matches any daily pattern
        for pattern in daily_patterns:
            if re.search(pattern, col, re.IGNORECASE):
                daily_columns.append(col)
                break
    
    return daily_columns


def identify_weekly_features(df: pd.DataFrame) -> List[str]:
    """
    Identify weekly feature columns (those starting with 'w_').
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of weekly feature column names
    """
    return [col for col in df.columns if col.startswith('w_')]


def apply_daily_lags(df: pd.DataFrame, features: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Apply daily lags to features using simple row shifts.
    
    Args:
        df: DataFrame with features (must be sorted by date)
        features: List of feature columns to lag
        lags: List of lag periods (in trading days)
        
    Returns:
        DataFrame with additional lagged columns
    """
    if not lags or not features:
        return df
    
    result = df.copy()
    
    for lag in lags:
        if lag <= 0:
            logger.warning(f"Skipping invalid daily lag: {lag}")
            continue
            
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found for daily lagging")
                continue
                
            lag_col_name = f"{feature}_lag{lag}d"
            result[lag_col_name] = df[feature].shift(lag)
    
    return result


def apply_weekly_lags(df: pd.DataFrame, features: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Apply weekly lags to features using week-boundary aware logic.
    
    Weekly lag N means "the value from N complete weeks ago", held constant 
    for all days within each week.
    
    Args:
        df: DataFrame with features (must be sorted by date with DatetimeIndex)
        features: List of weekly feature columns to lag
        lags: List of lag periods (in weeks)
        
    Returns:
        DataFrame with additional weekly lagged columns
    """
    if not lags or not features:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Weekly lagging requires DatetimeIndex, skipping")
        return df
    
    result = df.copy()
    
    # Apply lags to weekly data
    for lag in lags:
        if lag <= 0:
            logger.warning(f"Skipping invalid weekly lag: {lag}")
            continue
            
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found for weekly lagging")
                continue
            
            lag_col_name = f"{feature}_lag{lag}w"
            
            # Create weekly grouping based on Friday end of week
            df_with_week = result.copy()
            
            # Add week grouping column (W-FRI frequency)
            week_periods = df_with_week.index.to_period('W-FRI')
            df_with_week['_week_group'] = week_periods
            
            # For each week, get the last value and use it for that week
            weekly_values = df_with_week.groupby('_week_group')[feature].last()
            
            # Apply lag to weekly values
            weekly_lagged = weekly_values.shift(lag)
            
            # Map back to daily index by matching week groups
            daily_lagged = df_with_week['_week_group'].map(weekly_lagged)
            
            result[lag_col_name] = daily_lagged
    
    return result


def _apply_lags_single_symbol(
    symbol: str, 
    df: pd.DataFrame, 
    daily_lags: List[int], 
    weekly_lags: List[int]
) -> Tuple[str, pd.DataFrame]:
    """
    Apply daily and weekly lags to a single symbol's DataFrame.
    
    Args:
        symbol: Symbol identifier
        df: DataFrame with features
        daily_lags: List of daily lag periods
        weekly_lags: List of weekly lag periods
        
    Returns:
        Tuple of (symbol, processed_dataframe)
    """
    try:
        if df.empty:
            return symbol, df
        
        # Ensure chronological order
        df_sorted = df.sort_index()
        
        # Identify feature types
        daily_features = identify_daily_features(df_sorted)
        weekly_features = identify_weekly_features(df_sorted)
        
        logger.debug(f"{symbol}: Found {len(daily_features)} daily features, {len(weekly_features)} weekly features")
        
        # Apply daily lags
        if daily_lags and daily_features:
            df_sorted = apply_daily_lags(df_sorted, daily_features, daily_lags)
        
        # Apply weekly lags  
        if weekly_lags and weekly_features:
            df_sorted = apply_weekly_lags(df_sorted, weekly_features, weekly_lags)
        
        return symbol, df_sorted
        
    except Exception as e:
        logger.error(f"Error applying lags to {symbol}: {e}")
        return symbol, df


def apply_configurable_lags(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    daily_lags: List[int] = None,
    weekly_lags: List[int] = None,
    n_jobs: int = -1
) -> Dict[str, pd.DataFrame]:
    """
    Apply configurable daily and weekly lags to all symbols in parallel.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        daily_lags: List of daily lag periods (in trading days)
        weekly_lags: List of weekly lag periods (in weeks)  
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        
    Returns:
        Dictionary with same structure but with additional lagged columns
        
    Notes:
        - Daily lags use simple row shifts (N trading days ago)
        - Weekly lags use week-boundary aware logic (N complete weeks ago)
        - Weekly lagged values are held constant within each week
        - Feature identification uses regex patterns to classify daily vs weekly
    """
    if not indicators_by_symbol:
        logger.info("No symbols for lag application")
        return indicators_by_symbol
    
    # Default to no lags if not specified
    daily_lags = daily_lags or []
    weekly_lags = weekly_lags or []
    
    if not daily_lags and not weekly_lags:
        logger.info("No lags specified, returning original data")
        return indicators_by_symbol
    
    logger.info(f"Applying lags to {len(indicators_by_symbol)} symbols: "
               f"daily_lags={daily_lags}, weekly_lags={weekly_lags}")
    
    # Filter valid symbols
    valid_symbols = [(sym, df) for sym, df in indicators_by_symbol.items() 
                    if df is not None and not df.empty]
    
    if not valid_symbols:
        logger.info("No valid symbols for lag application")
        return indicators_by_symbol
    
    # Calculate effective parallel processing parameters
    n_symbols = len(valid_symbols)
    if n_jobs == -1:
        effective_jobs = multiprocessing.cpu_count()
    else:
        effective_jobs = n_jobs

    # Calculate optimal chunk size: minimum 100 symbols per chunk to amortize IPC overhead
    min_chunk_size = 100
    chunk_size = max(min_chunk_size, n_symbols // effective_jobs)
    n_chunks = max(1, n_symbols // chunk_size)
    actual_workers = min(effective_jobs, n_chunks)

    logger.info(f"Processing lags for {n_symbols} symbols in {n_chunks} batches "
                f"(~{chunk_size} symbols/batch, {actual_workers} workers)")

    # Apply lags (sequential or parallel)
    if actual_workers == 1 or n_symbols < 10:
        logger.info("Using sequential processing for lag application")
        results = []
        for symbol, df in valid_symbols:
            result = _apply_lags_single_symbol(symbol, df, daily_lags, weekly_lags)
            results.append(result)
    else:
        # Create batches for parallel processing
        work_chunks = _chunk_list(valid_symbols, chunk_size)

        try:
            batch_results = Parallel(
                n_jobs=actual_workers,
                backend='loky',
                verbose=0,
                prefer='processes'
            )(
                delayed(_apply_lags_batch)(chunk, daily_lags, weekly_lags)
                for chunk in work_chunks
            )

            # Flatten batch results
            results = []
            for batch in batch_results:
                results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel processing failed ({e}), falling back to sequential")
            results = []
            for symbol, df in valid_symbols:
                result = _apply_lags_single_symbol(symbol, df, daily_lags, weekly_lags)
                results.append(result)
    
    # Reconstruct dictionary
    processed_dict = indicators_by_symbol.copy()
    
    total_daily_cols = 0
    total_weekly_cols = 0
    
    for symbol, processed_df in results:
        processed_dict[symbol] = processed_df
        
        # Count new lag columns
        new_daily_cols = len([col for col in processed_df.columns if re.search(r'_lag\d+d$', col)])
        new_weekly_cols = len([col for col in processed_df.columns if re.search(r'_lag\d+w$', col)])
        
        total_daily_cols += new_daily_cols
        total_weekly_cols += new_weekly_cols
    
    logger.info(f"Lag application complete. Added {total_daily_cols} daily lag columns, "
               f"{total_weekly_cols} weekly lag columns across {len(results)} symbols")
    
    return processed_dict


def parse_lag_specification(lag_spec: str) -> List[int]:
    """
    Parse command-line lag specification into list of integers.
    
    Supports formats:
    - Single value: "5"
    - Comma-separated: "1,2,5" 
    - Range notation: "1-5" (expands to [1,2,3,4,5])
    - Mixed: "1,3,5-7" (expands to [1,3,5,6,7])
    
    Args:
        lag_spec: String specification of lags
        
    Returns:
        List of positive integer lag values
    """
    if not lag_spec or lag_spec.strip() == "":
        return []
    
    lags = []
    
    for part in lag_spec.split(','):
        part = part.strip()
        if not part:
            continue
            
        if '-' in part and not part.startswith('-'):
            # Range notation: "1-5"
            try:
                start, end = part.split('-', 1)
                start_val = int(start.strip())
                end_val = int(end.strip())
                if start_val <= end_val and start_val > 0:
                    lags.extend(range(start_val, end_val + 1))
                else:
                    logger.warning(f"Invalid range specification: {part}")
            except (ValueError, TypeError):
                logger.warning(f"Could not parse range: {part}")
        else:
            # Single value
            try:
                lag_val = int(part)
                if lag_val > 0:
                    lags.append(lag_val)
                else:
                    logger.warning(f"Ignoring non-positive lag: {lag_val}")
            except (ValueError, TypeError):
                logger.warning(f"Could not parse lag value: {part}")
    
    # Remove duplicates and sort
    unique_lags = sorted(list(set(lags)))
    
    if unique_lags:
        logger.info(f"Parsed lag specification '{lag_spec}' -> {unique_lags}")
    
    return unique_lags