"""
Unified timeframe handler for daily, weekly, and monthly data.

This module provides a clean interface for:
- Resampling data to different timeframes (D/W/M)
- Computing features at each timeframe
- Merging higher-timeframe features back to daily (leakage-safe)
- Parallel processing of multiple symbols

Key design decisions:
- Single-pass data partitioning (O(N) not O(N²))
- Memory-efficient dtype optimization
- Leakage-safe merge using merge_asof with direction='backward'
- Unified ParallelConfig for consistent parallelization
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Import ParallelConfig - with fallback for different import contexts
try:
    from ..config.parallel import ParallelConfig
except ImportError:
    from src.config.parallel import ParallelConfig

logger = logging.getLogger(__name__)


class TimeframeType(str, Enum):
    """Supported timeframes."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"


# Keep TimeframeConfig for backward compatibility, but prefer ParallelConfig
@dataclass
class TimeframeConfig:
    """Configuration for timeframe processing (legacy, prefer ParallelConfig)."""
    n_jobs: int = -1
    backend: str = 'loky'
    batch_size: int = 1
    verbose: int = 0
    dtype_optimization: bool = True
    memory_limit_mb: int = 2048

    @classmethod
    def from_parallel_config(cls, pc: ParallelConfig) -> 'TimeframeConfig':
        """Create TimeframeConfig from ParallelConfig."""
        return cls(
            n_jobs=pc.n_jobs,
            backend=pc.backend,
            batch_size=pc.batch_size,
            verbose=pc.verbose
        )


def _get_parallel_params(config: Union[ParallelConfig, TimeframeConfig, None]) -> dict:
    """Extract joblib Parallel parameters from config."""
    if config is None:
        config = ParallelConfig.default()
    elif isinstance(config, TimeframeConfig):
        # Convert legacy config
        return {
            'n_jobs': config.n_jobs,
            'backend': config.backend,
            'verbose': config.verbose,
        }

    return {
        'n_jobs': config.n_jobs,
        'backend': config.backend,
        'verbose': config.verbose,
        'prefer': getattr(config, 'prefer', 'processes'),
    }


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _get_effective_batch_params(n_items: int, config: Union[ParallelConfig, TimeframeConfig, None]) -> Tuple[int, int, int]:
    """
    Calculate effective parallel processing parameters with min 100 items per chunk.

    Returns:
        Tuple of (chunk_size, n_chunks, actual_workers)
    """
    if config is None:
        config = ParallelConfig.default()

    n_jobs = config.n_jobs
    if n_jobs == -1:
        import multiprocessing
        effective_jobs = multiprocessing.cpu_count()
    else:
        effective_jobs = n_jobs

    # Calculate optimal chunk size: minimum 100 symbols per chunk to amortize IPC overhead
    min_chunk_size = 100
    chunk_size = max(min_chunk_size, n_items // effective_jobs)
    n_chunks = max(1, n_items // chunk_size)
    actual_workers = min(effective_jobs, n_chunks)

    return chunk_size, n_chunks, actual_workers


class TimeframeResampler:
    """Unified resampling for daily/weekly/monthly data.

    This class handles all timeframe-related operations:
    - Resampling from daily to higher timeframes
    - Merging higher timeframe features back to daily
    - Parallel processing across symbols
    """

    # Resampling rules for each timeframe
    RESAMPLE_RULES = {
        TimeframeType.DAILY: None,  # No resampling needed
        TimeframeType.WEEKLY: 'W-FRI',  # Friday week-end
        TimeframeType.MONTHLY: 'ME',  # Month-end
    }

    # Standard OHLCV aggregation rules
    OHLCV_AGG = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'adjclose': 'last',
        'volume': 'sum',
    }

    # Column prefixes for each timeframe
    PREFIXES = {
        TimeframeType.DAILY: '',
        TimeframeType.WEEKLY: 'w_',
        TimeframeType.MONTHLY: 'm_',
    }

    def __init__(self, config: Union[ParallelConfig, TimeframeConfig, None] = None):
        """Initialize resampler with configuration.

        Args:
            config: ParallelConfig or legacy TimeframeConfig for parallel processing
        """
        if config is None:
            self.parallel_config = ParallelConfig.default()
        elif isinstance(config, ParallelConfig):
            self.parallel_config = config
        else:
            # Legacy TimeframeConfig
            self.parallel_config = ParallelConfig(
                n_jobs=config.n_jobs,
                backend=config.backend,
                batch_size=config.batch_size,
                verbose=config.verbose
            )

        # Keep legacy attribute for backward compatibility
        self.config = TimeframeConfig.from_parallel_config(self.parallel_config)

    @classmethod
    def resample_single(
        cls,
        df: pd.DataFrame,
        timeframe: TimeframeType,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Resample a single symbol's DataFrame to target timeframe.

        Args:
            df: DataFrame with DatetimeIndex or 'date' column
            timeframe: Target timeframe (D, W, or M)
            symbol: Optional symbol for metadata

        Returns:
            Resampled DataFrame
        """
        if timeframe == TimeframeType.DAILY:
            return df.copy()

        rule = cls.RESAMPLE_RULES.get(timeframe)
        if not rule:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Prepare for resampling
        df_work = df.copy()
        if 'date' in df_work.columns:
            df_work = df_work.set_index('date')
        elif not isinstance(df_work.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex or 'date' column")

        # Build aggregation rules for all columns
        agg_rules = {}
        for col in df_work.columns:
            if col in cls.OHLCV_AGG:
                agg_rules[col] = cls.OHLCV_AGG[col]
            elif col in ['symbol']:
                agg_rules[col] = 'first'
            elif pd.api.types.is_numeric_dtype(df_work[col]):
                agg_rules[col] = 'last'

        # Perform resampling
        resampled = df_work.resample(rule).agg(agg_rules)
        resampled = resampled.dropna(subset=['close'])

        if resampled.empty:
            return pd.DataFrame()

        # Add metadata
        period_col = 'week_end' if timeframe == TimeframeType.WEEKLY else 'month_end'
        resampled[period_col] = resampled.index
        if symbol:
            resampled['symbol'] = symbol

        # Calculate returns at this timeframe
        ret_col = f'{cls.PREFIXES[timeframe]}ret'
        resampled[ret_col] = np.log(
            pd.to_numeric(resampled['close'], errors='coerce')
        ).diff()

        # Reset index
        result = resampled.reset_index(drop=True)

        # Optimize dtypes
        result = cls._optimize_dtypes(result)

        return result

    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency.

        Args:
            df: DataFrame to optimize

        Returns:
            DataFrame with optimized dtypes
        """
        result = df.copy()

        # float64 -> float32
        float_cols = result.select_dtypes(include=['float64']).columns
        for col in float_cols:
            result[col] = result[col].astype('float32')

        # int64 -> int32 where safe
        int_cols = result.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min, col_max = result[col].min(), result[col].max()
            if pd.notna(col_min) and pd.notna(col_max):
                if col_min >= -2**31 and col_max < 2**31:
                    result[col] = result[col].astype('int32')

        return result

    def resample_symbols(
        self,
        daily_data: Dict[str, pd.DataFrame],
        timeframe: TimeframeType
    ) -> Dict[str, pd.DataFrame]:
        """Resample multiple symbols in parallel with batching.

        Args:
            daily_data: Dict mapping symbol -> daily DataFrame
            timeframe: Target timeframe

        Returns:
            Dict mapping symbol -> resampled DataFrame
        """
        if timeframe == TimeframeType.DAILY:
            return {sym: df.copy() for sym, df in daily_data.items()}

        n_symbols = len(daily_data)
        logger.info(f"Resampling {n_symbols} symbols to {timeframe.value}")
        start_time = time.time()

        # Get batching parameters
        chunk_size, n_chunks, actual_workers = _get_effective_batch_params(
            n_symbols, self.parallel_config
        )

        logger.info(f"Processing in {n_chunks} batches (~{chunk_size} symbols/batch, {actual_workers} workers)")

        work_items = list(daily_data.items())

        # Use sequential processing for small datasets
        if n_symbols < 10 or actual_workers == 1:
            results = []
            for symbol, df in work_items:
                result = self._resample_symbol_safe(symbol, df, timeframe)
                results.append(result)
        else:
            # Batch processing
            work_chunks = _chunk_list(work_items, chunk_size)

            def _resample_batch(items_batch: List[Tuple[str, pd.DataFrame]]) -> List[Tuple[str, Optional[pd.DataFrame]]]:
                batch_results = []
                for symbol, df in items_batch:
                    result = self._resample_symbol_safe(symbol, df, timeframe)
                    batch_results.append(result)
                return batch_results

            params = _get_parallel_params(self.parallel_config)
            params['n_jobs'] = actual_workers

            try:
                batch_results = Parallel(**params)(
                    delayed(_resample_batch)(chunk)
                    for chunk in work_chunks
                )
                # Flatten results
                results = []
                for batch in batch_results:
                    results.extend(batch)
            except Exception as e:
                logger.warning(f"Parallel resampling failed ({e}), falling back to sequential")
                results = []
                for symbol, df in work_items:
                    result = self._resample_symbol_safe(symbol, df, timeframe)
                    results.append(result)

        # Collect results
        resampled_data = {}
        for symbol, resampled in results:
            if resampled is not None and not resampled.empty:
                resampled_data[symbol] = resampled

        elapsed = time.time() - start_time
        logger.info(f"Resampling completed: {len(resampled_data)}/{len(daily_data)} symbols in {elapsed:.2f}s")

        return resampled_data

    def _resample_symbol_safe(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: TimeframeType
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """Safely resample a single symbol with error handling.

        Args:
            symbol: Symbol identifier
            df: Daily DataFrame
            timeframe: Target timeframe

        Returns:
            Tuple of (symbol, resampled DataFrame or None)
        """
        try:
            resampled = self.resample_single(df, timeframe, symbol)
            return symbol, resampled
        except Exception as e:
            logger.warning(f"Resampling failed for {symbol}: {e}")
            return symbol, None

    @classmethod
    def merge_to_daily(
        cls,
        daily_df: pd.DataFrame,
        higher_tf_df: pd.DataFrame,
        timeframe: TimeframeType,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Merge higher timeframe features back to daily data (leakage-safe).

        Uses merge_asof with direction='backward' to ensure we only use
        data that would have been available at each daily point.

        Args:
            daily_df: Daily DataFrame with 'date' column
            higher_tf_df: Higher timeframe DataFrame
            timeframe: The higher timeframe (W or M)
            feature_cols: Specific columns to merge (optional)

        Returns:
            Daily DataFrame with higher timeframe features added
        """
        if higher_tf_df.empty:
            return daily_df

        # Determine the period end column
        period_col = 'week_end' if timeframe == TimeframeType.WEEKLY else 'month_end'

        if period_col not in higher_tf_df.columns:
            logger.warning(f"Missing {period_col} column in higher timeframe data")
            return daily_df

        # Get prefix for this timeframe
        prefix = cls.PREFIXES[timeframe]

        # Determine columns to merge
        if feature_cols is None:
            # Get all feature columns (exclude metadata)
            exclude_cols = {'symbol', 'date', period_col, 'open', 'high', 'low', 'close', 'volume', 'adjclose'}
            feature_cols = [col for col in higher_tf_df.columns if col not in exclude_cols]

        # Add prefix to feature columns if not already present
        cols_to_merge = [period_col]
        rename_map = {}
        for col in feature_cols:
            if not col.startswith(prefix):
                new_col = f"{prefix}{col}"
                rename_map[col] = new_col
            cols_to_merge.append(col)

        # Prepare higher timeframe data for merge
        merge_df = higher_tf_df[cols_to_merge].copy()
        if rename_map:
            merge_df = merge_df.rename(columns=rename_map)

        # Drop columns that already exist in daily_df to avoid duplicates (_x/_y suffixes)
        existing_cols = set(daily_df.columns)
        cols_to_drop = [c for c in merge_df.columns if c in existing_cols and c != period_col]
        if cols_to_drop:
            logger.debug(f"Skipping {len(cols_to_drop)} columns that already exist in daily_df: {cols_to_drop[:5]}...")
            merge_df = merge_df.drop(columns=cols_to_drop)

        # Ensure proper datetime types
        daily_sorted = daily_df.copy()
        daily_sorted['date'] = pd.to_datetime(daily_sorted['date'])
        merge_df[period_col] = pd.to_datetime(merge_df[period_col])

        daily_sorted = daily_sorted.sort_values('date')
        merge_df = merge_df.sort_values(period_col)

        # Leakage-safe merge
        result = pd.merge_asof(
            daily_sorted,
            merge_df,
            left_on='date',
            right_on=period_col,
            direction='backward'
        )

        # Clean up temporary column
        if period_col in result.columns:
            result = result.drop(columns=[period_col])

        return result

    def merge_symbols_to_daily(
        self,
        daily_data: Dict[str, pd.DataFrame],
        higher_tf_data: Dict[str, pd.DataFrame],
        timeframe: TimeframeType
    ) -> Dict[str, pd.DataFrame]:
        """Merge higher timeframe features to daily data for multiple symbols (parallel with batching).

        Args:
            daily_data: Dict mapping symbol -> daily DataFrame
            higher_tf_data: Dict mapping symbol -> higher timeframe DataFrame
            timeframe: The higher timeframe

        Returns:
            Dict mapping symbol -> daily DataFrame with higher TF features
        """
        logger.info(f"Merging {timeframe.value} features to daily for {len(daily_data)} symbols")
        start_time = time.time()

        # Prepare merge tasks
        merge_tasks = []
        symbols_without_higher_tf = []

        for symbol, daily_df in daily_data.items():
            if symbol in higher_tf_data:
                merge_tasks.append((symbol, daily_df, higher_tf_data[symbol]))
            else:
                symbols_without_higher_tf.append(symbol)

        # Get batching parameters
        n_tasks = len(merge_tasks)
        if n_tasks > 0:
            chunk_size, n_chunks, actual_workers = _get_effective_batch_params(
                n_tasks, self.parallel_config
            )
            logger.info(f"Merging in {n_chunks} batches (~{chunk_size} symbols/batch, {actual_workers} workers)")

        # Process merge tasks
        if merge_tasks:
            # Sequential for small datasets
            if n_tasks < 10 or actual_workers == 1:
                results = []
                for symbol, daily_df, htf_df in merge_tasks:
                    result = _merge_single_symbol(symbol, daily_df, htf_df, timeframe)
                    results.append(result)
            else:
                # Batch processing
                work_chunks = _chunk_list(merge_tasks, chunk_size)

                def _merge_batch(tasks_batch: List[Tuple[str, pd.DataFrame, pd.DataFrame]]) -> List[Tuple[str, pd.DataFrame]]:
                    batch_results = []
                    for symbol, daily_df, htf_df in tasks_batch:
                        result = _merge_single_symbol(symbol, daily_df, htf_df, timeframe)
                        batch_results.append(result)
                    return batch_results

                params = _get_parallel_params(self.parallel_config)
                params['n_jobs'] = actual_workers

                try:
                    batch_results = Parallel(**params)(
                        delayed(_merge_batch)(chunk)
                        for chunk in work_chunks
                    )
                    # Flatten results
                    results = []
                    for batch in batch_results:
                        results.extend(batch)
                except Exception as e:
                    logger.warning(f"Parallel merge failed ({e}), falling back to sequential")
                    results = []
                    for symbol, daily_df, htf_df in merge_tasks:
                        result = _merge_single_symbol(symbol, daily_df, htf_df, timeframe)
                        results.append(result)

            merged_data = dict(results)
        else:
            merged_data = {}

        # Add symbols without higher timeframe data
        for symbol in symbols_without_higher_tf:
            merged_data[symbol] = daily_data[symbol]

        elapsed = time.time() - start_time
        logger.info(f"Merge completed in {elapsed:.2f}s")

        return merged_data


def _merge_single_symbol(
    symbol: str,
    daily_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
    timeframe: TimeframeType
) -> Tuple[str, pd.DataFrame]:
    """Merge higher TF features for a single symbol (for parallel execution).

    Args:
        symbol: Symbol identifier
        daily_df: Daily DataFrame
        higher_tf_df: Higher timeframe DataFrame
        timeframe: The higher timeframe

    Returns:
        Tuple of (symbol, merged DataFrame)
    """
    try:
        merged = TimeframeResampler.merge_to_daily(daily_df, higher_tf_df, timeframe)
        return symbol, merged
    except Exception as e:
        logger.warning(f"Merge failed for {symbol}: {e}")
        return symbol, daily_df


def partition_by_symbol(
    df_long: pd.DataFrame,
    optimize_dtypes: bool = True
) -> Dict[str, pd.DataFrame]:
    """Partition long-format DataFrame by symbol (single-pass O(N)).

    This eliminates the O(N²) data duplication problem by creating
    each symbol's DataFrame only once.

    Args:
        df_long: Long-format DataFrame with 'symbol' column
        optimize_dtypes: Whether to optimize dtypes for memory

    Returns:
        Dict mapping symbol -> DataFrame
    """
    logger.info(f"Partitioning {len(df_long)} rows by symbol")
    start_time = time.time()

    symbol_data = {}
    for symbol, group_df in df_long.groupby('symbol', observed=True):
        symbol_df = group_df.reset_index(drop=True)
        if optimize_dtypes:
            symbol_df = TimeframeResampler._optimize_dtypes(symbol_df)
        symbol_data[symbol] = symbol_df

    elapsed = time.time() - start_time
    logger.info(f"Partitioned into {len(symbol_data)} symbols in {elapsed:.2f}s")

    return symbol_data


def combine_to_long(
    symbol_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Combine symbol DataFrames back to long format.

    Args:
        symbol_data: Dict mapping symbol -> DataFrame (date can be index or column)

    Returns:
        Combined long-format DataFrame with 'date' and 'symbol' columns
    """
    if not symbol_data:
        return pd.DataFrame()

    frames = []
    for symbol, df in symbol_data.items():
        frame = df.copy()

        # Handle date as index - convert to column
        if 'date' not in frame.columns and frame.index.name in ('date', None):
            frame = frame.reset_index()
            # If index was unnamed, try to identify the date column
            if 'index' in frame.columns:
                # Check if it looks like dates
                if pd.api.types.is_datetime64_any_dtype(frame['index']):
                    frame = frame.rename(columns={'index': 'date'})
                else:
                    frame = frame.drop(columns=['index'])

        # Remove duplicate columns (keep first occurrence)
        if frame.columns.duplicated().any():
            frame = frame.loc[:, ~frame.columns.duplicated()]

        if 'symbol' not in frame.columns:
            frame['symbol'] = symbol
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    # Sort if date column exists
    if 'date' in combined.columns:
        return combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    else:
        return combined.sort_values(['symbol']).reset_index(drop=True)


def compute_features_at_timeframe(
    data_dict: Dict[str, pd.DataFrame],
    feature_func: Callable[[pd.DataFrame], pd.DataFrame],
    timeframe: TimeframeType,
    config: Union[ParallelConfig, TimeframeConfig, None] = None,
    **feature_kwargs
) -> Dict[str, pd.DataFrame]:
    """Compute features for all symbols at a given timeframe (with batching).

    Args:
        data_dict: Dict mapping symbol -> DataFrame at target timeframe
        feature_func: Function that computes features on a single DataFrame
        timeframe: The timeframe of the data
        config: ParallelConfig or legacy TimeframeConfig
        **feature_kwargs: Additional arguments passed to feature_func

    Returns:
        Dict mapping symbol -> DataFrame with features added
    """
    if config is None:
        config = ParallelConfig.default()

    n_symbols = len(data_dict)
    logger.info(f"Computing features for {n_symbols} symbols at {timeframe.value}")
    start_time = time.time()

    def process_symbol(symbol: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        try:
            result = feature_func(df, **feature_kwargs)
            return symbol, result
        except Exception as e:
            logger.warning(f"Feature computation failed for {symbol}: {e}")
            return symbol, df

    # Get batching parameters
    chunk_size, n_chunks, actual_workers = _get_effective_batch_params(n_symbols, config)
    logger.info(f"Processing in {n_chunks} batches (~{chunk_size} symbols/batch, {actual_workers} workers)")

    work_items = list(data_dict.items())

    # Sequential for small datasets
    if n_symbols < 10 or actual_workers == 1:
        results = [process_symbol(symbol, df) for symbol, df in work_items]
    else:
        # Batch processing
        work_chunks = _chunk_list(work_items, chunk_size)

        def _process_batch(items_batch: List[Tuple[str, pd.DataFrame]]) -> List[Tuple[str, pd.DataFrame]]:
            batch_results = []
            for symbol, df in items_batch:
                result = process_symbol(symbol, df)
                batch_results.append(result)
            return batch_results

        params = _get_parallel_params(config)
        params['n_jobs'] = actual_workers

        try:
            batch_results = Parallel(**params)(
                delayed(_process_batch)(chunk)
                for chunk in work_chunks
            )
            # Flatten results
            results = []
            for batch in batch_results:
                results.extend(batch)
        except Exception as e:
            logger.warning(f"Parallel feature computation failed ({e}), falling back to sequential")
            results = [process_symbol(symbol, df) for symbol, df in work_items]

    # Collect results
    feature_data = dict(results)

    elapsed = time.time() - start_time
    logger.info(f"Feature computation completed in {elapsed:.2f}s")

    return feature_data


def add_prefix_to_features(
    df: pd.DataFrame,
    prefix: str,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Add prefix to feature columns.

    Args:
        df: DataFrame with features
        prefix: Prefix to add (e.g., 'w_' for weekly)
        exclude_cols: Columns to not prefix

    Returns:
        DataFrame with prefixed columns
    """
    exclude = set(exclude_cols or [])
    exclude.update({'symbol', 'date', 'week_end', 'month_end',
                    'open', 'high', 'low', 'close', 'adjclose', 'volume'})

    rename_map = {}
    for col in df.columns:
        if col not in exclude and not col.startswith(prefix):
            rename_map[col] = f"{prefix}{col}"

    return df.rename(columns=rename_map)
