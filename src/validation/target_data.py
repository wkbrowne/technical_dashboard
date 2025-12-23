"""
Centralized data validation for triple barrier target generation.

This module consolidates all validation and outlier filtering logic used in:
- Target generation (src/features/target_generation.py)
- Barrier calibration (scripts/calibrate_barriers.py)
- Target recomputation (scripts/recompute_targets.py)
- Pipeline orchestration (src/pipelines/orchestrator.py)

All thresholds are defined here as constants for consistency.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for target data validation thresholds."""

    # Price range validation (per-symbol filtering)
    max_price: float = 50_000.0  # Max price per share
    min_price: float = 0.01      # Min price per share
    max_price_ratio: float = 1000.0  # Max price range ratio (max/min)

    # ATR validation
    min_atr: float = 0.0  # ATR must be > this value

    # MFE/MAE outlier filtering (for calibration)
    max_mfe_mae_atr: float = 50.0  # Max MFE/MAE in ATR units

    # Return outlier filtering (for generated targets)
    ret_zscore_threshold: float = 5.0  # Z-score threshold for returns
    ret_abs_threshold: float = 1.0     # Absolute log return threshold (~170% linear)

    # Minimum data requirements
    min_rows_per_symbol: int = 25  # Minimum rows for target generation

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_price <= self.min_price:
            raise ValueError("max_price must be > min_price")
        if self.max_price_ratio <= 1.0:
            raise ValueError("max_price_ratio must be > 1.0")
        if self.min_atr < 0:
            raise ValueError("min_atr must be >= 0")


# Global default configuration
DEFAULT_VALIDATION_CONFIG = ValidationConfig()


# =============================================================================
# SYMBOL-LEVEL FILTERING (Price Range)
# =============================================================================

def filter_symbols_by_price(
    df: pd.DataFrame,
    config: Optional[ValidationConfig] = None,
    price_col: str = 'close',
    symbol_col: str = 'symbol',
    report: bool = True
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Filter out symbols with suspicious price data.

    This catches:
    - Extremely high prices (data errors, unadjusted for splits)
    - Extremely low prices (penny stocks, delisting)
    - Extreme price ratios (reverse splits not adjusted)

    Args:
        df: DataFrame with price data (long format)
        config: Validation configuration (uses defaults if None)
        price_col: Column name for close price
        symbol_col: Column name for symbol identifier
        report: Whether to log filtered symbols

    Returns:
        Tuple of (filtered DataFrame, set of removed symbols)
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    if df.empty:
        return df, set()

    initial_symbols = set(df[symbol_col].unique())

    # Compute per-symbol price statistics
    symbol_stats = df.groupby(symbol_col)[price_col].agg(['min', 'max'])

    # Apply filters
    valid_mask = (
        (symbol_stats['max'] <= config.max_price) &
        (symbol_stats['min'] >= config.min_price) &
        (symbol_stats['min'] > 0) &  # Avoid division by zero
        (symbol_stats['max'] / symbol_stats['min'] <= config.max_price_ratio)
    )

    valid_symbols = set(symbol_stats[valid_mask].index)
    removed_symbols = initial_symbols - valid_symbols

    if removed_symbols and report:
        logger.info(f"Price validation: removed {len(removed_symbols)} symbols "
                   f"(max>{config.max_price:.0f}, min<{config.min_price}, "
                   f"ratio>{config.max_price_ratio:.0f})")
        if len(removed_symbols) <= 20:
            logger.debug(f"  Removed symbols: {sorted(removed_symbols)}")

    filtered_df = df[df[symbol_col].isin(valid_symbols)].copy()

    return filtered_df, removed_symbols


def filter_symbols_by_price_dict(
    data: Dict[str, pd.DataFrame],
    config: Optional[ValidationConfig] = None,
    price_col: str = 'adjclose',
    report: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Set[str]]:
    """
    Filter symbols from a dictionary of DataFrames (wide format).

    Used by orchestrator when processing indicators_by_symbol dict.

    Args:
        data: Dictionary mapping symbol -> DataFrame
        config: Validation configuration
        price_col: Column name for close price
        report: Whether to log filtered symbols

    Returns:
        Tuple of (filtered dict, set of removed symbols)
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    removed_symbols = set()
    filtered_data = {}

    for symbol, df in data.items():
        if df.empty or price_col not in df.columns:
            removed_symbols.add(symbol)
            continue

        prices = df[price_col].dropna()
        if len(prices) == 0:
            removed_symbols.add(symbol)
            continue

        max_price = prices.max()
        min_price = prices.min()

        # Apply validation
        if max_price > config.max_price:
            removed_symbols.add(symbol)
            continue
        if min_price < config.min_price:
            removed_symbols.add(symbol)
            continue
        if min_price > 0 and max_price / min_price > config.max_price_ratio:
            removed_symbols.add(symbol)
            continue

        filtered_data[symbol] = df

    if removed_symbols and report:
        logger.info(f"Price validation: removed {len(removed_symbols)} symbols from dict")

    return filtered_data, removed_symbols


# =============================================================================
# ROW-LEVEL FILTERING (Invalid Entries)
# =============================================================================

def filter_invalid_entries(
    close: np.ndarray,
    atr: np.ndarray,
    dates: Optional[np.ndarray] = None,
    config: Optional[ValidationConfig] = None
) -> np.ndarray:
    """
    Create mask for valid entry points in numpy arrays.

    This is used during target generation to skip invalid rows.

    Args:
        close: Array of close prices
        atr: Array of ATR values
        dates: Optional array of dates (datetime64)
        config: Validation configuration

    Returns:
        Boolean mask where True = valid entry point
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    # Basic validity checks
    valid = (
        np.isfinite(close) &
        np.isfinite(atr) &
        (close > 0) &
        (atr > config.min_atr)
    )

    # Date validity if provided
    if dates is not None:
        valid = valid & ~np.isnat(dates)

    return valid


# =============================================================================
# MFE/MAE FILTERING (Calibration)
# =============================================================================

def filter_extreme_mfe_mae(
    mfe: float,
    mae: float,
    config: Optional[ValidationConfig] = None
) -> bool:
    """
    Check if MFE/MAE values are within acceptable range.

    Used during barrier calibration to filter outliers.

    Args:
        mfe: Maximum Favorable Excursion in ATR units
        mae: Maximum Adverse Excursion in ATR units (can be negative)
        config: Validation configuration

    Returns:
        True if values are valid (not outliers), False otherwise
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    # Both MFE and MAE should be within reasonable bounds
    if abs(mfe) > config.max_mfe_mae_atr:
        return False
    if abs(mae) > config.max_mfe_mae_atr:
        return False

    return True


def compute_mfe_mae_with_filtering(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    pos: int,
    horizon: int,
    config: Optional[ValidationConfig] = None
) -> Optional[Dict[str, float]]:
    """
    Compute MFE/MAE for a single entry point with validation.

    Args:
        close: Full close price array
        high: Full high price array
        low: Full low price array
        atr: Full ATR array
        pos: Entry position index
        horizon: Forward horizon in days
        config: Validation configuration

    Returns:
        Dictionary with mfe_long, mae_long, mfe_short, mae_short if valid,
        None if entry is invalid or results are outliers
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    n = len(close)
    start_idx = pos + 1
    end_idx = start_idx + horizon

    if end_idx > n:
        return None

    c0 = close[pos]
    a0 = atr[pos]

    # Validate entry point
    if not (np.isfinite(c0) and np.isfinite(a0) and a0 > config.min_atr and c0 > 0):
        return None

    # Compute forward extremes
    h_max = np.max(high[start_idx:end_idx])
    l_min = np.min(low[start_idx:end_idx])

    # Compute MFE/MAE in ATR units
    mfe_long = (h_max - c0) / a0
    mae_long = (l_min - c0) / a0
    mfe_short = (c0 - l_min) / a0
    mae_short = (c0 - h_max) / a0

    # Filter outliers
    if not filter_extreme_mfe_mae(mfe_long, mae_long, config):
        return None
    if not filter_extreme_mfe_mae(mfe_short, mae_short, config):
        return None

    return {
        'mfe_long': mfe_long,
        'mae_long': mae_long,
        'mfe_short': mfe_short,
        'mae_short': mae_short,
    }


# =============================================================================
# RETURN FILTERING (Generated Targets)
# =============================================================================

def filter_extreme_returns(
    targets_df: pd.DataFrame,
    ret_col: str = 'ret_from_entry',
    config: Optional[ValidationConfig] = None,
    report: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter targets with extreme return values.

    This removes:
    - NaN/Inf returns
    - Absolute returns exceeding threshold
    - Z-score outliers

    Args:
        targets_df: DataFrame with target columns
        ret_col: Column name for returns
        config: Validation configuration
        report: Whether to log detailed report

    Returns:
        Tuple of (filtered DataFrame, dict of filter counts)
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    if targets_df.empty or ret_col not in targets_df.columns:
        return targets_df, {}

    initial_count = len(targets_df)
    filter_counts = {}
    targets_df = targets_df.copy()

    # 1. Filter NaN/Inf
    nan_mask = targets_df[ret_col].isna()
    inf_mask = ~np.isfinite(targets_df[ret_col].fillna(0))
    nan_inf_mask = nan_mask | inf_mask
    nan_inf_count = nan_inf_mask.sum()

    if nan_inf_count > 0:
        filter_counts['nan_inf'] = int(nan_inf_count)
        targets_df = targets_df[~nan_inf_mask]

    # 2. Filter extreme absolute returns
    if len(targets_df) > 0:
        extreme_abs_mask = np.abs(targets_df[ret_col]) > config.ret_abs_threshold
        extreme_abs_count = extreme_abs_mask.sum()

        if extreme_abs_count > 0:
            filter_counts['extreme_abs'] = int(extreme_abs_count)
            targets_df = targets_df[~extreme_abs_mask]

    # 3. Filter z-score outliers
    if len(targets_df) > 10:
        ret_mean = targets_df[ret_col].mean()
        ret_std = targets_df[ret_col].std()

        if ret_std > 0:
            ret_zscore = (targets_df[ret_col] - ret_mean) / ret_std
            zscore_mask = np.abs(ret_zscore) > config.ret_zscore_threshold
            zscore_count = zscore_mask.sum()

            if zscore_count > 0:
                filter_counts['zscore_outlier'] = int(zscore_count)
                targets_df = targets_df[~zscore_mask]

    # Report
    total_filtered = initial_count - len(targets_df)
    if total_filtered > 0 and report:
        filter_pct = 100 * total_filtered / initial_count
        if filter_pct > 1.0:
            logger.warning(f"Return validation: filtered {total_filtered:,} rows "
                          f"({filter_pct:.2f}%) - {filter_counts}")
        else:
            logger.info(f"Return validation: filtered {total_filtered:,} rows "
                       f"({filter_pct:.2f}%)")

    return targets_df, filter_counts


def filter_extreme_returns_multi(
    targets_df: pd.DataFrame,
    ret_cols: List[str],
    config: Optional[ValidationConfig] = None,
    report: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter targets with extreme returns across multiple return columns.

    Used for multi-model targets where each model has its own return column.

    Args:
        targets_df: DataFrame with multiple ret_* columns
        ret_cols: List of return column names
        config: Validation configuration
        report: Whether to log report

    Returns:
        Tuple of (filtered DataFrame, dict of filter counts)
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    if targets_df.empty:
        return targets_df, {}

    initial_count = len(targets_df)
    total_counts = {}
    targets_df = targets_df.copy()

    for ret_col in ret_cols:
        if ret_col not in targets_df.columns:
            continue

        targets_df, counts = filter_extreme_returns(
            targets_df, ret_col, config, report=False
        )
        for k, v in counts.items():
            total_counts[f"{ret_col}_{k}"] = v

    total_filtered = initial_count - len(targets_df)
    if total_filtered > 0 and report:
        filter_pct = 100 * total_filtered / initial_count
        logger.info(f"Multi-return validation: filtered {total_filtered:,} rows "
                   f"({filter_pct:.2f}%)")

    return targets_df, total_counts


# =============================================================================
# COMBINED VALIDATION
# =============================================================================

def validate_target_input_data(
    df: pd.DataFrame,
    config: Optional[ValidationConfig] = None,
    required_cols: Optional[List[str]] = None,
    report: bool = True
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Comprehensive validation of input data for target generation.

    Applies all relevant filters in sequence:
    1. Required column check
    2. Symbol-level price filtering
    3. Row-level NaN/invalid value filtering
    4. Minimum data per symbol check

    Args:
        df: Input DataFrame (long format with symbol, date, close, high, low, atr)
        config: Validation configuration
        required_cols: Required column names (defaults to target generation requirements)
        report: Whether to log validation results

    Returns:
        Tuple of (validated DataFrame, validation summary dict)
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG

    if required_cols is None:
        required_cols = ['symbol', 'date', 'close', 'high', 'low', 'atr']

    summary = {
        'initial_rows': len(df),
        'initial_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
        'removed_symbols': set(),
        'dropped_rows': 0,
    }

    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 1. Symbol-level price filtering
    df, removed_symbols = filter_symbols_by_price(
        df, config, price_col='close', report=report
    )
    summary['removed_symbols'] = removed_symbols

    # 2. Row-level filtering
    initial_rows = len(df)

    # Drop NaN in required columns
    df = df.dropna(subset=['close', 'high', 'low', 'atr'])

    # Filter invalid ATR/close
    df = df[(df['atr'] > config.min_atr) & (df['close'] > 0)]

    summary['dropped_rows'] = initial_rows - len(df)

    # 3. Minimum data per symbol
    symbol_counts = df.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= config.min_rows_per_symbol].index
    insufficient_symbols = set(symbol_counts.index) - set(valid_symbols)

    if insufficient_symbols:
        df = df[df['symbol'].isin(valid_symbols)]
        summary['insufficient_data_symbols'] = insufficient_symbols
        if report:
            logger.info(f"Removed {len(insufficient_symbols)} symbols with "
                       f"< {config.min_rows_per_symbol} rows")

    summary['final_rows'] = len(df)
    summary['final_symbols'] = df['symbol'].nunique()

    if report:
        logger.info(f"Validation complete: {summary['initial_rows']:,} -> {summary['final_rows']:,} rows, "
                   f"{summary['initial_symbols']} -> {summary['final_symbols']} symbols")

    return df, summary


# =============================================================================
# VALIDATOR CLASS (Stateful Interface)
# =============================================================================

class TargetDataValidator:
    """
    Stateful validator for target data processing.

    Provides a consistent interface for validation across the pipeline
    with configurable thresholds.

    Example:
        validator = TargetDataValidator()
        df, summary = validator.validate_input(df)
        targets, counts = validator.filter_returns(targets)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_VALIDATION_CONFIG

    def validate_input(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
        report: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Validate input data for target generation."""
        return validate_target_input_data(df, self.config, required_cols, report)

    def filter_symbols(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        report: bool = True
    ) -> Tuple[pd.DataFrame, Set[str]]:
        """Filter symbols by price validation."""
        return filter_symbols_by_price(df, self.config, price_col, report=report)

    def filter_symbols_dict(
        self,
        data: Dict[str, pd.DataFrame],
        price_col: str = 'adjclose',
        report: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Set[str]]:
        """Filter symbols from dict format."""
        return filter_symbols_by_price_dict(data, self.config, price_col, report)

    def filter_returns(
        self,
        targets_df: pd.DataFrame,
        ret_col: str = 'ret_from_entry',
        report: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Filter extreme returns."""
        return filter_extreme_returns(targets_df, ret_col, self.config, report)

    def filter_returns_multi(
        self,
        targets_df: pd.DataFrame,
        ret_cols: List[str],
        report: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Filter extreme returns across multiple columns."""
        return filter_extreme_returns_multi(targets_df, ret_cols, self.config, report)

    def is_valid_entry(
        self,
        close: float,
        atr: float,
        date: Optional[np.datetime64] = None
    ) -> bool:
        """Check if a single entry point is valid."""
        if not (np.isfinite(close) and np.isfinite(atr)):
            return False
        if close <= 0 or atr <= self.config.min_atr:
            return False
        if date is not None and np.isnat(date):
            return False
        return True

    def is_valid_mfe_mae(self, mfe: float, mae: float) -> bool:
        """Check if MFE/MAE values are within acceptable range."""
        return filter_extreme_mfe_mae(mfe, mae, self.config)

    def compute_mfe_mae(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        pos: int,
        horizon: int
    ) -> Optional[Dict[str, float]]:
        """Compute MFE/MAE with validation."""
        return compute_mfe_mae_with_filtering(
            close, high, low, atr, pos, horizon, self.config
        )
