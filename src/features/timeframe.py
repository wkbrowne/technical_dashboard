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
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class TimeframeType(str, Enum):
    """Supported timeframes."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"


@dataclass
class TimeframeConfig:
    """Configuration for timeframe processing."""
    n_jobs: int = -1
    backend: str = 'loky'
    batch_size: int = 1
    verbose: int = 0
    dtype_optimization: bool = True
    memory_limit_mb: int = 2048


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

    def __init__(self, config: Optional[TimeframeConfig] = None):
        """Initialize resampler with configuration.

        Args:
            config: Optional configuration for parallel processing
        """
        self.config = config or TimeframeConfig()

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
        """Resample multiple symbols in parallel.

        Args:
            daily_data: Dict mapping symbol -> daily DataFrame
            timeframe: Target timeframe

        Returns:
            Dict mapping symbol -> resampled DataFrame
        """
        if timeframe == TimeframeType.DAILY:
            return {sym: df.copy() for sym, df in daily_data.items()}

        logger.info(f"Resampling {len(daily_data)} symbols to {timeframe.value}")
        start_time = time.time()

        # Process in parallel
        results = Parallel(
            n_jobs=self.config.n_jobs,
            backend=self.config.backend,
            verbose=self.config.verbose
        )(
            delayed(self._resample_symbol_safe)(symbol, df, timeframe)
            for symbol, df in daily_data.items()
        )

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
        """Merge higher timeframe features to daily data for multiple symbols.

        Args:
            daily_data: Dict mapping symbol -> daily DataFrame
            higher_tf_data: Dict mapping symbol -> higher timeframe DataFrame
            timeframe: The higher timeframe

        Returns:
            Dict mapping symbol -> daily DataFrame with higher TF features
        """
        logger.info(f"Merging {timeframe.value} features to daily for {len(daily_data)} symbols")
        start_time = time.time()

        merged_data = {}
        for symbol, daily_df in daily_data.items():
            if symbol in higher_tf_data:
                merged = self.merge_to_daily(
                    daily_df,
                    higher_tf_data[symbol],
                    timeframe
                )
                merged_data[symbol] = merged
            else:
                merged_data[symbol] = daily_df

        elapsed = time.time() - start_time
        logger.info(f"Merge completed in {elapsed:.2f}s")

        return merged_data


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
        symbol_data: Dict mapping symbol -> DataFrame

    Returns:
        Combined long-format DataFrame
    """
    if not symbol_data:
        return pd.DataFrame()

    frames = []
    for symbol, df in symbol_data.items():
        frame = df.copy()
        if 'symbol' not in frame.columns:
            frame['symbol'] = symbol
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(['symbol', 'date']).reset_index(drop=True)


def compute_features_at_timeframe(
    data_dict: Dict[str, pd.DataFrame],
    feature_func: Callable[[pd.DataFrame], pd.DataFrame],
    timeframe: TimeframeType,
    config: Optional[TimeframeConfig] = None,
    **feature_kwargs
) -> Dict[str, pd.DataFrame]:
    """Compute features for all symbols at a given timeframe.

    Args:
        data_dict: Dict mapping symbol -> DataFrame at target timeframe
        feature_func: Function that computes features on a single DataFrame
        timeframe: The timeframe of the data
        config: Processing configuration
        **feature_kwargs: Additional arguments passed to feature_func

    Returns:
        Dict mapping symbol -> DataFrame with features added
    """
    config = config or TimeframeConfig()

    logger.info(f"Computing features for {len(data_dict)} symbols at {timeframe.value}")
    start_time = time.time()

    def process_symbol(symbol: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        try:
            result = feature_func(df, **feature_kwargs)
            return symbol, result
        except Exception as e:
            logger.warning(f"Feature computation failed for {symbol}: {e}")
            return symbol, df

    results = Parallel(
        n_jobs=config.n_jobs,
        backend=config.backend,
        verbose=config.verbose
    )(
        delayed(process_symbol)(symbol, df)
        for symbol, df in data_dict.items()
    )

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
