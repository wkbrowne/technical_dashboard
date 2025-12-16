"""
Triple barrier target generation for financial time series.

This module generates triple barrier targets for stock trajectories using configurable
barriers based on ATR (Average True Range). Each target represents a potential trade
with upper/lower barriers and time horizon constraints.
"""
import logging
from typing import Dict, Optional, List, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gc

# Import ParallelConfig - with fallback for different import contexts
try:
    from ..config.parallel import ParallelConfig
except ImportError:
    from src.config.parallel import ParallelConfig

logger = logging.getLogger(__name__)


def generate_triple_barrier_targets(df: pd.DataFrame, config: Dict, 
                                   weight_min_clip: float = 0.01, 
                                   weight_max_clip: float = 10.0) -> pd.DataFrame:
    """
    Generate triple barrier targets from long-format price data.
    
    Creates targets by setting upper and lower barriers around entry prices
    and tracking which barrier (if any) is hit within the maximum horizon.
    
    Args:
        df: Long-format DataFrame with columns: symbol, date, close, high, low, atr
        config: Configuration dictionary with keys:
            - up_mult: Upper barrier multiplier (e.g., 3.0 = 3x ATR above entry)
            - dn_mult: Lower barrier multiplier (e.g., 3.0 = 3x ATR below entry)  
            - max_horizon: Maximum days to track each target
            - start_every: Minimum days between new targets (when not hit early)
        weight_min_clip: Minimum weight value (prevents zero weights)
        weight_max_clip: Maximum weight value (prevents extreme weights)
            
    Returns:
        DataFrame with columns: symbol, t0, t_hit, hit, entry_px, top, bot, 
                               h_used, price_hit, ret_from_entry, n_overlapping_trajs, 
                               weight_overlap, weight_class_balance, weight_final
        
    Notes:
        - hit: 1 = upper barrier hit, -1 = lower barrier hit, 0 = time expired
        - ret_from_entry: Log return from entry to exit price
        - n_overlapping_trajs: Count of overlapping trajectories at t0 for this symbol
        - weight_overlap: Inverse of overlap count (uniqueness weight)
        - weight_class_balance: Inverse frequency weight for class balancing
        - weight_final: Combined multiplicative weight (clipped & normalized)
        - Assumes df is sorted by symbol and date
    """
    required_cols = {'symbol', 'date', 'close', 'high', 'low', 'atr'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate config
    required_config = {'up_mult', 'dn_mult', 'max_horizon', 'start_every'}
    missing_config = required_config - set(config.keys())
    if missing_config:
        raise ValueError(f"Missing required config keys: {missing_config}")
    
    logger.debug(f"Generating targets with config: {config}")
    
    # Early data cleaning and validation
    logger.debug(f"Input DataFrame shape: {df.shape}")
    
    # Convert date column to datetime64[ns] early in pipeline
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.debug("Converting date column to datetime64[ns]")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with missing values in required columns
    initial_rows = len(df)
    df_clean = df.dropna(subset=list(required_cols))
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.debug(f"Dropped {dropped_rows} rows with missing values in required columns")
    
    # Additional validation for numeric columns
    numeric_cols = ['close', 'high', 'low', 'atr']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            logger.debug(f"Converting {col} to numeric")
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop rows where numeric conversion failed
    df_clean = df_clean.dropna(subset=numeric_cols)
    total_dropped = initial_rows - len(df_clean)
    if total_dropped > 0:
        logger.info(f"Cleaned data: dropped {total_dropped} rows with invalid data, {len(df_clean)} rows remaining")
    
    if df_clean.empty:
        logger.warning("No valid data remaining after cleaning")
        return pd.DataFrame()
    
    # Process each symbol independently
    results = []
    symbols = df_clean['symbol'].unique()
    logger.debug(f"Processing {len(symbols)} symbols after data cleaning")
    
    for symbol in symbols:
        symbol_df = df_clean[df_clean['symbol'] == symbol].copy()
        
        # Enhanced logging for symbol processing
        if len(symbol_df) < config['max_horizon'] + 2:
            logger.debug(f"Skipping {symbol}: insufficient data ({len(symbol_df)} rows, need {config['max_horizon'] + 2})")
            continue
        
        # Additional validation for date sorting
        if not symbol_df['date'].is_monotonic_increasing:
            logger.debug(f"Sorting data by date for {symbol}")
            symbol_df = symbol_df.sort_values('date').reset_index(drop=True)
        
        logger.debug(f"Processing {symbol} with {len(symbol_df)} rows from {symbol_df['date'].min()} to {symbol_df['date'].max()}")
        
        symbol_targets = _compute_triple_barriers_numpy(symbol, symbol_df, config)
        if not symbol_targets.empty:
            results.append(symbol_targets)
            logger.debug(f"Generated {len(symbol_targets)} targets for {symbol}")
        else:
            logger.debug(f"No targets generated for {symbol}")
    
    if not results:
        logger.warning("No targets generated for any symbols after processing")
        return pd.DataFrame()
    
    targets_df = pd.concat(results, ignore_index=True)
    
    # Add overlap counting
    targets_df = _add_overlap_counts(targets_df, config)
    
    # Calculate combined weights (overlap + class balance)
    targets_df = _add_combined_weights(targets_df, weight_min_clip, weight_max_clip)
    
    logger.info(f"Added combined weights: min={targets_df['weight_final'].min():.6f}, "
               f"max={targets_df['weight_final'].max():.6f}, sum={targets_df['weight_final'].sum():.2f}")

    # Validate and filter extreme target values
    targets_df = validate_and_filter_extreme_targets(
        targets_df,
        ret_zscore_threshold=5.0,
        ret_abs_threshold=1.0,
        report=True
    )

    # Top-level validation to ensure no NaNs in t0 column
    if 't0' in targets_df.columns:
        nan_t0_count = targets_df['t0'].isna().sum()
        if nan_t0_count > 0:
            logger.error(f"Found {nan_t0_count} NaN values in t0 column - this should not happen")
            # Remove rows with NaN t0 values
            targets_df = targets_df.dropna(subset=['t0'])
            logger.warning(f"Removed {nan_t0_count} rows with NaN t0 values, {len(targets_df)} rows remaining")
        
        if targets_df.empty:
            logger.error("All targets have invalid t0 values after validation")
            return pd.DataFrame()
    
    logger.info(f"Generated {len(targets_df)} targets across {len(results)} symbols")
    
    return targets_df


def _compute_triple_barriers_numpy(symbol: str, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compute triple barrier targets for a single symbol using NumPy for speed.
    
    Args:
        symbol: Symbol identifier
        df: DataFrame for single symbol, sorted by date
        config: Configuration dictionary
        
    Returns:
        DataFrame with target information for this symbol
    """
    # Sort and reset index to ensure consistent positioning
    df = df.sort_values('date').reset_index(drop=True)
    
    # Input validation for the DataFrame
    if df.empty:
        logger.debug(f"Empty DataFrame for {symbol}")
        return pd.DataFrame()
    
    # Validate that required columns exist and have valid data
    required_cols = ['date', 'close', 'high', 'low', 'atr']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing column {col} for {symbol}")
            return pd.DataFrame()
    
    # Convert to NumPy arrays for speed with robust error handling
    try:
        dates = df['date'].to_numpy('datetime64[ns]')
        close = df['close'].to_numpy(dtype=float)
        atr = df['atr'].to_numpy(dtype=float)
        high = df['high'].to_numpy(dtype=float)
        low = df['low'].to_numpy(dtype=float)
    except Exception as e:
        logger.warning(f"Failed to convert data to numpy arrays for {symbol}: {e}")
        return pd.DataFrame()
    
    # Validate array shapes match
    if not all(len(arr) == len(dates) for arr in [close, atr, high, low]):
        logger.warning(f"Mismatched array lengths for {symbol}")
        return pd.DataFrame()
    
    # Check for invalid dates (NaT values)
    if np.any(np.isnat(dates)):
        logger.warning(f"Found NaT (invalid) dates for {symbol}")
        return pd.DataFrame()
    
    n = len(df)
    pos = 0
    targets = []
    skipped_invalid = 0
    
    up_mult = config['up_mult']
    dn_mult = config['dn_mult']
    max_horizon = config['max_horizon']
    start_every = config['start_every']
    
    while pos < n - max_horizon - 1:
        c0, a0, date0 = close[pos], atr[pos], dates[pos]
        
        # Skip if entry price or ATR is invalid
        if not (np.isfinite(c0) and np.isfinite(a0) and a0 > 0 and not np.isnat(date0)):
            skipped_invalid += 1
            pos += 1
            continue
        
        # Define barriers
        top_barrier = c0 + up_mult * a0
        bot_barrier = c0 - dn_mult * a0
        
        # Define analysis window (next day through max horizon)
        start_idx = pos + 1
        end_idx = start_idx + max_horizon
        
        if end_idx > n:
            break  # Not enough data for full horizon
        
        # Slice arrays for the analysis window
        h_slice = high[start_idx:end_idx]
        l_slice = low[start_idx:end_idx]
        c_slice = close[start_idx:end_idx]
        
        # Find barrier hits
        hit_top_indices = np.where(h_slice >= top_barrier)[0]
        hit_bot_indices = np.where(l_slice <= bot_barrier)[0]
        
        # Determine which barrier was hit first (if any)
        hit_type = 0  # 0 = time expired
        hit_idx = max_horizon - 1  # Default to last day
        price_hit = c_slice[hit_idx] if len(c_slice) > hit_idx else np.nan
        
        if len(hit_top_indices) > 0 and (len(hit_bot_indices) == 0 or hit_top_indices[0] <= hit_bot_indices[0]):
            # Upper barrier hit first
            hit_type = 1
            hit_idx = hit_top_indices[0]
            price_hit = top_barrier
        elif len(hit_bot_indices) > 0:
            # Lower barrier hit first  
            hit_type = -1
            hit_idx = hit_bot_indices[0]
            price_hit = bot_barrier
        
        # Calculate metrics
        horizon_used = hit_idx + 1
        t_hit_idx = start_idx + hit_idx
        
        # Validate t_hit index and calculate return
        if t_hit_idx >= len(dates):
            pos += 1
            continue
            
        t_hit = dates[t_hit_idx]
        ret_from_entry = np.log(price_hit / c0) if price_hit > 0 and c0 > 0 else np.nan
        
        targets.append({
            'symbol': symbol,
            't0': date0,
            't_hit': t_hit,
            'hit': hit_type,
            'entry_px': c0,
            'top': top_barrier,
            'bot': bot_barrier,
            'h_used': horizon_used,
            'price_hit': price_hit,
            'ret_from_entry': ret_from_entry
        })
        
        # Advance position: if barrier hit early, use that horizon, otherwise use start_every
        if hit_type != 0 and horizon_used < start_every:
            pos += horizon_used
        else:
            pos += start_every
    
    # Log processing summary for this symbol
    if skipped_invalid > 0:
        logger.debug(f"Skipped {skipped_invalid} invalid rows for {symbol}")
    
    if targets:
        logger.debug(f"Successfully created {len(targets)} targets for {symbol}")
    else:
        logger.debug(f"No valid targets created for {symbol}")
    
    return pd.DataFrame(targets)


def get_target_summary(targets_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for triple barrier targets.
    
    Args:
        targets_df: Output from generate_triple_barrier_targets()
        
    Returns:
        Dictionary with summary statistics
    """
    if targets_df.empty:
        return {'total_targets': 0}
    
    hit_counts = targets_df['hit'].value_counts()
    
    return {
        'total_targets': len(targets_df),
        'symbols': targets_df['symbol'].nunique(),
        'hit_upper': hit_counts.get(1, 0),
        'hit_lower': hit_counts.get(-1, 0), 
        'hit_time': hit_counts.get(0, 0),
        'avg_horizon_used': targets_df['h_used'].mean(),
        'avg_return': targets_df['ret_from_entry'].mean(),
        'return_std': targets_df['ret_from_entry'].std(),
        'date_range': (targets_df['t0'].min(), targets_df['t0'].max())
    }


def _add_overlap_counts(targets_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Add overlap counting to targets DataFrame using vectorized sweep-line algorithm.

    Counts how many trajectories overlap at each date per symbol and adds
    this information back to the original targets DataFrame.

    Uses O(n log n) sweep-line algorithm instead of O(n²) naive approach.

    Args:
        targets_df: DataFrame with generated targets
        config: Configuration dictionary (unused but kept for API consistency)

    Returns:
        DataFrame with added overlap count column
    """
    if targets_df.empty:
        return targets_df

    overlap_col = "n_overlapping_trajs"

    # Work on a copy
    targets_df = targets_df.copy()
    targets_df[overlap_col] = 0

    # Process each symbol using vectorized counting
    for symbol in targets_df['symbol'].unique():
        symbol_mask = targets_df['symbol'] == symbol
        symbol_indices = targets_df.index[symbol_mask]

        if len(symbol_indices) == 0:
            continue

        # Get the subset for this symbol
        symbol_targets = targets_df.loc[symbol_indices, ['t0', 'h_used']].copy()

        # Use vectorized overlap counting
        overlap_counts = _count_overlaps_vectorized(symbol_targets)

        # Assign directly using index alignment
        targets_df.loc[symbol_indices, overlap_col] = overlap_counts

    return targets_df


def _count_overlaps_vectorized(symbol_targets: pd.DataFrame) -> np.ndarray:
    """
    Count overlaps for a single symbol using fully vectorized numpy operations.

    This is O(n) memory and O(n) time complexity using a sweep-line algorithm,
    compared to the O(n²) naive approach.

    Args:
        symbol_targets: DataFrame with t0 and h_used columns for one symbol

    Returns:
        Array of overlap counts for each trajectory
    """
    n = len(symbol_targets)
    if n == 0:
        return np.array([], dtype=np.int32)

    # Convert to numpy arrays (as int64 nanoseconds for datetime math)
    # Use .values to get numpy array and avoid pandas index issues
    t0_ns = symbol_targets['t0'].values.astype('datetime64[ns]').view('int64')
    h_used = symbol_targets['h_used'].values.astype('float64')

    # Handle NaN values
    valid_mask = ~(np.isnan(h_used) | pd.isna(symbol_targets['t0'].values))

    if not valid_mask.any():
        return np.zeros(n, dtype=np.int32)

    # Calculate end times (t0 + (h_used-1) days in nanoseconds)
    day_ns = 24 * 60 * 60 * 1e9  # nanoseconds per day
    t_end_ns = t0_ns + ((h_used - 1) * day_ns).astype('int64')

    # Use sweep-line algorithm: O(n log n) instead of O(n²)
    # Create events: (time, type, index) where type=0 is start, type=1 is end
    events = []
    for i in range(n):
        if valid_mask[i]:
            events.append((t0_ns[i], 0, i))      # start event
            events.append((t_end_ns[i], 1, i))   # end event

    # Sort events by time, then by type (starts before ends at same time)
    events.sort(key=lambda x: (x[0], x[1]))

    # Sweep through events tracking active trajectories
    overlap_counts = np.zeros(n, dtype=np.int32)
    active_set = set()

    for time_ns, event_type, idx in events:
        if event_type == 0:  # start
            # Count current active trajectories as overlapping with this one
            overlap_counts[idx] = len(active_set) + 1  # +1 for self
            # Also increment count for all currently active trajectories
            for active_idx in active_set:
                overlap_counts[active_idx] += 1
            active_set.add(idx)
        else:  # end
            active_set.discard(idx)

    return overlap_counts


def _count_overlaps_efficient(symbol_targets: pd.DataFrame) -> Dict:
    """
    Efficiently count overlaps for a single symbol.

    This is a wrapper that calls the vectorized version and returns
    the result in the legacy format for backwards compatibility.

    Args:
        symbol_targets: DataFrame with t0 and h_used columns for one symbol

    Returns:
        Dictionary mapping index -> (t0, overlap_count)
    """
    overlap_counts_array = _count_overlaps_vectorized(symbol_targets)
    t0_array = symbol_targets['t0'].values

    return {
        i: (t0_array[i], count)
        for i, count in enumerate(overlap_counts_array)
        if not pd.isna(t0_array[i])
    }


def _process_symbol_group(df: pd.DataFrame, config: Dict, symbols: List[str],
                         weight_min_clip: float = 0.01, weight_max_clip: float = 10.0) -> pd.DataFrame:
    """
    Process a group of symbols by filtering and running triple barrier target generation.
    
    Args:
        df: Long-format DataFrame with all symbol data
        config: Triple barrier configuration dictionary
        symbols: List of symbols to process in this chunk
        weight_min_clip: Minimum weight value (prevents zero weights)
        weight_max_clip: Maximum weight value (prevents extreme weights)
        
    Returns:
        DataFrame with triple barrier targets for the specified symbols
    """
    if not symbols:
        return pd.DataFrame()
    
    # Filter to only the symbols in this chunk
    chunk_df = df[df['symbol'].isin(symbols)]
    
    if chunk_df.empty:
        logger.debug(f"No data found for symbols: {symbols}")
        return pd.DataFrame()
    
    logger.debug(f"Processing {len(symbols)} symbols: {symbols}")
    
    # Generate targets for this chunk
    return generate_triple_barrier_targets(chunk_df, config, weight_min_clip, weight_max_clip)


def generate_targets_parallel(
    df: pd.DataFrame,
    config: Dict,
    n_jobs: int = -1,
    chunk_size: int = 16,
    weight_min_clip: float = 0.01,
    weight_max_clip: float = 10.0,
    parallel_config: Optional[ParallelConfig] = None
) -> pd.DataFrame:
    """
    Chunk long-format data by symbol and run target generation in parallel.

    Args:
        df: Long-format DataFrame with columns ['symbol', 'date', 'close', 'high', 'low', 'atr']
        config: Triple barrier config dictionary with keys:
                - up_mult: Upper barrier multiplier
                - dn_mult: Lower barrier multiplier
                - max_horizon: Maximum days to track each target
                - start_every: Minimum days between new targets
        n_jobs: Number of parallel workers (-1 = all cores)
                Deprecated: Use parallel_config instead
        chunk_size: Number of symbols per parallel chunk
                    Deprecated: Use parallel_config.batch_size instead
        weight_min_clip: Minimum weight value (prevents zero weights)
        weight_max_clip: Maximum weight value (prevents extreme weights)
        parallel_config: ParallelConfig for parallel processing settings

    Returns:
        Combined DataFrame with triple barrier targets for all symbols
    """
    # Use ParallelConfig if provided, otherwise fall back to legacy params
    if parallel_config is not None:
        n_jobs = parallel_config.n_jobs
        # batch_size can be 'auto' (str), so handle it
        if isinstance(parallel_config.batch_size, int):
            chunk_size = parallel_config.batch_size
        # else keep the default chunk_size
        backend = parallel_config.backend
        verbose = parallel_config.verbose
        prefer = parallel_config.prefer
    else:
        backend = 'loky'
        verbose = 0
        prefer = 'processes'

    if df.empty:
        logger.warning("Empty DataFrame provided to generate_targets_parallel")
        return pd.DataFrame()

    # Validate required columns
    required_cols = {'symbol', 'date', 'close', 'high', 'low', 'atr'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for parallel target generation: {missing_cols}")

    # Get unique symbols and create chunks
    unique_symbols = df['symbol'].dropna().unique()
    n_symbols = len(unique_symbols)
    if n_symbols == 0:
        logger.warning("No valid symbols found in DataFrame")
        return pd.DataFrame()

    # Calculate effective jobs for optimal chunk sizing
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

    # Create symbol chunks
    symbol_chunks = [
        unique_symbols[i:i + chunk_size].tolist()
        for i in range(0, n_symbols, chunk_size)
    ]

    logger.info(f"Processing {n_symbols} symbols in {len(symbol_chunks)} chunks "
                f"(~{chunk_size} symbols/chunk, {actual_workers} workers)")

    # Reset the loky executor pool to avoid stale state from previous parallel operations
    # This fixes the '_ReusablePoolExecutor' object has no attribute '_temp_folder_manager' error
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass  # Ignore if executor doesn't exist or can't be shut down
    gc.collect()

    # Process chunks in parallel
    try:
        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            prefer=prefer
        )(
            delayed(_process_symbol_group)(df, config, chunk, weight_min_clip, weight_max_clip)
            for chunk in symbol_chunks
        )

        # Filter out empty results and concatenate
        valid_results = [result for result in results if not result.empty]

        if not valid_results:
            logger.warning("No valid targets generated from any chunks")
            return pd.DataFrame()

        # Combine all results
        combined_targets = pd.concat(valid_results, ignore_index=True)

        logger.info(f"Generated {len(combined_targets)} total targets across {len(valid_results)} chunks")

        return combined_targets

    except Exception as e:
        logger.error(f"Error in parallel target generation: {e}")
        raise


def validate_and_filter_extreme_targets(
    targets_df: pd.DataFrame,
    ret_zscore_threshold: float = 5.0,
    ret_abs_threshold: float = 1.0,
    report: bool = True
) -> pd.DataFrame:
    """
    Validate targets and filter out rows with extreme/invalid values.

    This function detects and removes targets with implausible return values
    that could indicate data errors or extreme outliers that would harm model training.

    Args:
        targets_df: DataFrame with target columns including ret_from_entry
        ret_zscore_threshold: Z-score threshold for return outliers (default 5.0 = ~3 per 10000)
        ret_abs_threshold: Absolute return threshold (default 1.0 = 100% return, log scale)
        report: Whether to log detailed report of filtered values

    Returns:
        DataFrame with extreme values filtered out

    Notes:
        - Filters applied:
          1. NaN/Inf values in ret_from_entry
          2. Absolute returns > ret_abs_threshold (default 100%, which is extreme for daily/weekly)
          3. Z-score outliers > ret_zscore_threshold
        - A warning is logged if >1% of rows are filtered
        - A detailed report is logged if report=True and any rows are filtered
    """
    if targets_df.empty:
        return targets_df

    initial_count = len(targets_df)
    targets_df = targets_df.copy()

    # Track filtering reasons for report
    filter_reasons = {}

    # 1. Check for NaN/Inf in ret_from_entry
    if 'ret_from_entry' in targets_df.columns:
        nan_mask = targets_df['ret_from_entry'].isna()
        inf_mask = ~np.isfinite(targets_df['ret_from_entry'].fillna(0))
        nan_inf_mask = nan_mask | inf_mask
        nan_inf_count = nan_inf_mask.sum()

        if nan_inf_count > 0:
            filter_reasons['nan_inf'] = {
                'count': nan_inf_count,
                'pct': 100 * nan_inf_count / initial_count,
                'description': 'NaN or Inf values in ret_from_entry'
            }
            targets_df = targets_df[~nan_inf_mask]

    # 2. Check for extreme absolute returns
    if 'ret_from_entry' in targets_df.columns and len(targets_df) > 0:
        extreme_abs_mask = np.abs(targets_df['ret_from_entry']) > ret_abs_threshold
        extreme_abs_count = extreme_abs_mask.sum()

        if extreme_abs_count > 0:
            extreme_values = targets_df.loc[extreme_abs_mask, 'ret_from_entry']
            filter_reasons['extreme_abs'] = {
                'count': extreme_abs_count,
                'pct': 100 * extreme_abs_count / initial_count,
                'description': f'Absolute return > {ret_abs_threshold:.2f} (log scale)',
                'min_val': extreme_values.min(),
                'max_val': extreme_values.max(),
                'examples': extreme_values.head(5).tolist()
            }
            targets_df = targets_df[~extreme_abs_mask]

    # 3. Check for z-score outliers
    if 'ret_from_entry' in targets_df.columns and len(targets_df) > 10:
        ret_mean = targets_df['ret_from_entry'].mean()
        ret_std = targets_df['ret_from_entry'].std()

        if ret_std > 0:
            ret_zscore = (targets_df['ret_from_entry'] - ret_mean) / ret_std
            zscore_outlier_mask = np.abs(ret_zscore) > ret_zscore_threshold
            zscore_outlier_count = zscore_outlier_mask.sum()

            if zscore_outlier_count > 0:
                outlier_values = targets_df.loc[zscore_outlier_mask, 'ret_from_entry']
                filter_reasons['zscore_outlier'] = {
                    'count': zscore_outlier_count,
                    'pct': 100 * zscore_outlier_count / initial_count,
                    'description': f'Z-score > {ret_zscore_threshold:.1f}',
                    'min_val': outlier_values.min(),
                    'max_val': outlier_values.max(),
                    'examples': outlier_values.head(5).tolist()
                }
                targets_df = targets_df[~zscore_outlier_mask]

    # Calculate total filtered
    final_count = len(targets_df)
    total_filtered = initial_count - final_count
    filter_pct = 100 * total_filtered / initial_count if initial_count > 0 else 0

    # Generate report
    if total_filtered > 0:
        # Always warn if significant filtering occurred
        if filter_pct > 1.0:
            logger.warning(
                f"EXTREME TARGET VALUES: Filtered {total_filtered:,} rows ({filter_pct:.2f}%) "
                f"with extreme/invalid target values"
            )
        else:
            logger.info(
                f"Target validation: filtered {total_filtered:,} rows ({filter_pct:.2f}%) "
                f"with extreme/invalid values"
            )

        if report:
            logger.info("=" * 60)
            logger.info("TARGET VALIDATION REPORT")
            logger.info("=" * 60)
            logger.info(f"Initial rows: {initial_count:,}")
            logger.info(f"Final rows:   {final_count:,}")
            logger.info(f"Filtered:     {total_filtered:,} ({filter_pct:.2f}%)")
            logger.info("-" * 60)

            for reason_key, reason_data in filter_reasons.items():
                logger.info(f"  {reason_data['description']}:")
                logger.info(f"    Count: {reason_data['count']:,} ({reason_data['pct']:.2f}%)")
                if 'min_val' in reason_data:
                    logger.info(f"    Range: [{reason_data['min_val']:.4f}, {reason_data['max_val']:.4f}]")
                if 'examples' in reason_data:
                    logger.info(f"    Examples: {reason_data['examples']}")

            logger.info("=" * 60)
    else:
        logger.info("Target validation: no extreme values detected")

    # Final stats on cleaned data
    if 'ret_from_entry' in targets_df.columns and len(targets_df) > 0:
        ret_stats = targets_df['ret_from_entry'].describe()
        logger.info(
            f"Cleaned ret_from_entry stats: "
            f"mean={ret_stats['mean']:.4f}, std={ret_stats['std']:.4f}, "
            f"min={ret_stats['min']:.4f}, max={ret_stats['max']:.4f}"
        )

    return targets_df


def _add_combined_weights(targets_df: pd.DataFrame,
                         weight_min_clip: float = 0.01,
                         weight_max_clip: float = 10.0) -> pd.DataFrame:
    """
    Add combined weights to targets DataFrame using overlap and class balance components.
    
    Args:
        targets_df: DataFrame with targets and overlap counts
        weight_min_clip: Minimum weight value (prevents zero weights)
        weight_max_clip: Maximum weight value (prevents extreme weights)
        
    Returns:
        DataFrame with additional weight columns: weight_overlap, weight_class_balance, weight_final
    """
    if targets_df.empty:
        logger.warning("Empty targets DataFrame provided for weight calculation")
        return targets_df
    
    targets_df = targets_df.copy()
    
    # 1. Calculate overlap weights (inverse of overlap count with pseudocount)
    targets_df['weight_overlap'] = 1.0 / (targets_df['n_overlapping_trajs'] + 0.5)
    
    # 2. Calculate class balance weights (inverse frequency per class)
    if 'hit' not in targets_df.columns:
        logger.error("Missing 'hit' column for class balance weight calculation")
        # Fallback to overlap-only weights
        targets_df['weight_class_balance'] = 1.0
        targets_df['weight_final'] = targets_df['weight_overlap']
    else:
        # Count class frequencies
        class_counts = targets_df['hit'].value_counts()
        total_samples = len(targets_df)
        n_classes = len(class_counts)
        
        # Calculate class balance weights (sklearn-style: n_samples / (n_classes * bincount))
        class_weights = {}
        for class_val, count in class_counts.items():
            class_weights[class_val] = total_samples / (n_classes * count)
        
        # Map class weights to each sample
        targets_df['weight_class_balance'] = targets_df['hit'].map(class_weights)
        
        # 3. Combine multiplicatively
        targets_df['weight_final'] = targets_df['weight_overlap'] * targets_df['weight_class_balance']
    
    # 4. Apply clipping to raw combined weights
    targets_df['weight_final'] = np.clip(targets_df['weight_final'], weight_min_clip, weight_max_clip)
    
    # 5. Normalize to sum to number of samples (standard for ML frameworks)
    weight_sum = targets_df['weight_final'].sum()
    if weight_sum > 0:
        targets_df['weight_final'] = targets_df['weight_final'] * len(targets_df) / weight_sum
        # Re-apply clipping after normalization to ensure bounds are respected
        targets_df['weight_final'] = np.clip(targets_df['weight_final'], weight_min_clip, weight_max_clip)
    else:
        logger.error("All weights are zero after clipping - using uniform weights")
        targets_df['weight_final'] = 1.0
    
    # Log weight statistics
    logger.info(f"Weight statistics:")
    logger.info(f"  Overlap weights: min={targets_df['weight_overlap'].min():.4f}, "
               f"max={targets_df['weight_overlap'].max():.4f}")
    
    if 'weight_class_balance' in targets_df.columns and 'hit' in targets_df.columns:
        logger.info(f"  Class balance weights: min={targets_df['weight_class_balance'].min():.4f}, "
                   f"max={targets_df['weight_class_balance'].max():.4f}")
        logger.info(f"  Class distribution: {dict(targets_df['hit'].value_counts().sort_index())}")
    
    logger.info(f"  Final weights: min={targets_df['weight_final'].min():.4f}, "
               f"max={targets_df['weight_final'].max():.4f}, "
               f"sum={targets_df['weight_final'].sum():.2f}")
    
    return targets_df


