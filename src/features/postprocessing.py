"""
Data postprocessing utilities for cleaning and preparing feature data.

This module contains functions for handling missing values, outliers, and
other data quality issues in the computed feature datasets.
"""
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Import ParallelConfig - with fallback for different import contexts
try:
    from ..config.parallel import ParallelConfig
except ImportError:
    from src.config.parallel import ParallelConfig

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

        # Handle duplicate columns - keep first occurrence only
        # This can happen when features are merged multiple times
        if df_processed.columns.duplicated().any():
            dup_cols = df_processed.columns[df_processed.columns.duplicated()].unique().tolist()
            logger.debug(f"{symbol}: Removing {len(dup_cols)} duplicate columns: {dup_cols[:5]}")
            df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]

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

        # Use column-by-column interpolation to avoid assignment issues with column selection
        for col in num_cols:
            df_processed[col] = df_processed[col].interpolate(
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


def _interpolate_batch(symbol_data_list: List[Tuple[str, pd.DataFrame]]) -> List[Tuple[str, pd.DataFrame, int]]:
    """
    Process a batch of symbols for NaN interpolation.

    Args:
        symbol_data_list: List of (symbol, dataframe) tuples to process

    Returns:
        List of (symbol, processed_dataframe, filled_count) tuples
    """
    results = []
    for symbol, df in symbol_data_list:
        result = _interpolate_single_symbol(symbol, df)
        results.append(result)
    return results


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def interpolate_internal_gaps(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    n_jobs: int = -1,
    batch_size: int = 16,
    parallel_config: Optional[ParallelConfig] = None
) -> Dict[str, pd.DataFrame]:
    """
    Interpolate internal NaN gaps in numeric columns only using parallel processing.

    This function fills NaN values that are strictly between observed values,
    without filling leading or trailing NaNs. This preserves data integrity
    by only filling gaps in continuous time series data.

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
                Deprecated: Use parallel_config instead
        batch_size: Number of symbols to process per batch
                    Deprecated: Use parallel_config instead
        parallel_config: ParallelConfig for parallel processing settings

    Returns:
        Dictionary with same structure but interpolated DataFrames

    Notes:
        - Only fills NaNs that are bounded by non-NaN values (limit_area='inside')
        - Uses 'time' interpolation for datetime indexes, 'linear' for others
        - Replaces inf/-inf with NaN before interpolation
        - Works only on numeric columns, preserves other column types
        - Parallelized with batching for improved performance with large symbol counts
    """
    # Use ParallelConfig if provided, otherwise fall back to legacy params
    if parallel_config is not None:
        n_jobs = parallel_config.n_jobs
        backend = parallel_config.backend
        verbose = parallel_config.verbose
        prefer = parallel_config.prefer
    else:
        backend = 'loky'
        verbose = 0
        prefer = 'processes'

    if not indicators_by_symbol:
        logger.info("No symbols to interpolate")
        return indicators_by_symbol

    # Filter out empty/None DataFrames before processing
    valid_symbols = [(sym, df) for sym, df in indicators_by_symbol.items()
                    if df is not None and not df.empty]

    if not valid_symbols:
        logger.info("No valid symbols with data to interpolate")
        return indicators_by_symbol

    n_symbols = len(valid_symbols)

    # Calculate effective jobs for chunk sizing
    if n_jobs == -1:
        import multiprocessing
        effective_jobs = multiprocessing.cpu_count()
    else:
        effective_jobs = n_jobs

    # Calculate optimal chunk size: minimum 100 symbols per chunk to amortize IPC overhead
    # If we can't achieve that, reduce the number of effective workers
    min_chunk_size = 100
    chunk_size = max(min_chunk_size, n_symbols // effective_jobs)
    n_chunks = max(1, n_symbols // chunk_size)
    actual_workers = min(effective_jobs, n_chunks)

    logger.info(f"Interpolating internal NaN gaps for {n_symbols} symbols...")

    # Use sequential processing if n_jobs=1 or small dataset
    if n_jobs == 1 or n_symbols < 10:
        logger.info("Using sequential processing for interpolation")
        results = []
        for symbol, df in valid_symbols:
            result = _interpolate_single_symbol(symbol, df)
            results.append(result)
    else:
        # Create batches for parallel processing
        symbol_chunks = _chunk_list(valid_symbols, chunk_size)
        n_chunks = len(symbol_chunks)

        logger.info(f"Processing in {n_chunks} batches (~{chunk_size} symbols/batch, {actual_workers} workers)")

        try:
            batch_results = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                verbose=verbose,
                prefer=prefer
            )(
                delayed(_interpolate_batch)(chunk)
                for chunk in symbol_chunks
            )

            # Flatten batch results
            results = []
            for batch in batch_results:
                results.extend(batch)

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


def drop_rows_with_excessive_nans(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    min_valid_ratio: float = 0.5,
    exclude_cols: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Drop rows with too many NaN values across all symbols.

    This is the final cleanup step in the pipeline, after all features have been
    computed and internal gaps have been interpolated. Uses pandas built-in dropna()
    with the thresh parameter.

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        min_valid_ratio: Minimum ratio of non-NaN values required (0.0 to 1.0)
                        E.g., 0.5 means at least 50% of features must be non-NaN
        exclude_cols: Columns to exclude when counting NaNs (e.g., ['symbol', 'date'])
                     These columns won't be considered in the threshold calculation

    Returns:
        Dictionary with filtered DataFrames (same symbols, fewer rows)

    Example:
        # Keep rows with at least 50% non-NaN feature values
        clean_data = drop_rows_with_excessive_nans(
            indicators_by_symbol,
            min_valid_ratio=0.5,
            exclude_cols=['symbol', 'date']
        )
    """
    if exclude_cols is None:
        exclude_cols = []

    logger.info(f"Dropping rows with < {min_valid_ratio:.0%} valid feature values...")

    filtered_dict = {}
    total_rows_before = 0
    total_rows_after = 0

    for symbol, df in indicators_by_symbol.items():
        if df is None or df.empty:
            filtered_dict[symbol] = df
            continue

        total_rows_before += len(df)

        # Get feature columns (exclude specified columns)
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if len(feature_cols) == 0:
            filtered_dict[symbol] = df
            total_rows_after += len(df)
            continue

        # Calculate minimum number of non-NaN values required
        thresh = int(len(feature_cols) * min_valid_ratio)

        # Use pandas built-in dropna with thresh parameter
        # thresh: require that many non-NA values in the specified columns
        df_filtered = df.dropna(subset=feature_cols, thresh=thresh)

        filtered_dict[symbol] = df_filtered
        total_rows_after += len(df_filtered)

        rows_removed = len(df) - len(df_filtered)
        if rows_removed > 0:
            logger.debug(f"{symbol}: removed {rows_removed}/{len(df)} rows "
                        f"({rows_removed/len(df)*100:.1f}%)")

    rows_removed_total = total_rows_before - total_rows_after

    logger.info(f"Row filtering complete:")
    logger.info(f"  Symbols: {len(indicators_by_symbol)} (unchanged)")
    logger.info(f"  Rows: {total_rows_before:,} → {total_rows_after:,} "
               f"({rows_removed_total:,} removed, {rows_removed_total/total_rows_before*100:.1f}%)")

    return filtered_dict


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