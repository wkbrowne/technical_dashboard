"""
Cross-sectional momentum features comparing each symbol's performance to the universe.

This module computes cross-sectional z-scores of momentum for different lookback periods,
with optional sector-neutral adjustments to control for sector effects.

Optimized with numpy for better CPU utilization on multi-core systems.
"""
import logging
import warnings
from typing import Dict, Iterable, Optional, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress expected warnings for NaN slices during cross-sectional computations
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')


def _numpy_rolling_sum(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Compute rolling sum along axis 0 (dates) using pure numpy.

    Uses cumsum for O(n) complexity instead of O(n*window).
    """
    n_dates, n_symbols = arr.shape
    result = np.full_like(arr, np.nan)

    # Use cumsum trick for efficient rolling sum
    # Pad with zeros for cumsum, handle NaN by treating as 0
    arr_filled = np.nan_to_num(arr, nan=0.0)
    valid_mask = ~np.isnan(arr)

    # Cumsum of values
    cumsum = np.cumsum(arr_filled, axis=0)
    # Cumsum of valid counts
    valid_cumsum = np.cumsum(valid_mask.astype(np.int32), axis=0)

    # Rolling sum = cumsum[i] - cumsum[i-window]
    for i in range(window - 1, n_dates):
        if i >= window:
            rolling_sum = cumsum[i] - cumsum[i - window]
            rolling_count = valid_cumsum[i] - valid_cumsum[i - window]
        else:
            rolling_sum = cumsum[i]
            rolling_count = valid_cumsum[i]

        # Only set where we have enough valid values
        valid_rows = rolling_count >= min_periods
        result[i, valid_rows] = rolling_sum[valid_rows]

    return result


def _numpy_cross_sectional_zscore(arr: np.ndarray) -> np.ndarray:
    """
    Compute cross-sectional z-scores (across columns) for each row.
    Uses median and std for robustness. Pure numpy, no numba.
    """
    # nanmedian and nanstd along axis=1 (across symbols)
    with np.errstate(all='ignore'):
        median = np.nanmedian(arr, axis=1, keepdims=True)
        std = np.nanstd(arr, axis=1, keepdims=True)

    # Avoid division by zero
    std = np.where(std < 1e-10, np.nan, std)

    result = (arr - median) / std
    return result


def _numpy_cross_sectional_percentile(arr: np.ndarray) -> np.ndarray:
    """
    Compute cross-sectional percentile ranks (0-100) for each row.
    Uses numpy argsort for efficient ranking.
    """
    n_dates, n_symbols = arr.shape
    result = np.full_like(arr, np.nan)

    for i in range(n_dates):
        row = arr[i, :]
        valid_mask = ~np.isnan(row)
        valid_count = np.sum(valid_mask)

        if valid_count < 2:
            continue

        # Get indices of valid values sorted by value
        valid_indices = np.where(valid_mask)[0]
        valid_values = row[valid_indices]

        # argsort gives the ranking
        order = np.argsort(valid_values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))

        # Convert to percentile (0-100)
        percentiles = ranks / (valid_count - 1) * 100
        result[i, valid_indices] = percentiles

    return result


def _build_price_panel_fast(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    price_col: str
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str], Dict[str, np.ndarray]]:
    """
    Build numpy array panel from symbol DataFrames.
    Also returns index mapping for efficient write-back.

    Returns:
        Tuple of (price_array, date_index, symbol_list, symbol_to_panel_indices)
        symbol_to_panel_indices maps symbol -> array of panel row indices for each df row
    """
    syms = [s for s, df in indicators_by_symbol.items() if price_col in df.columns]
    if not syms:
        return np.array([]), pd.DatetimeIndex([]), [], {}

    # Get union of all dates
    all_dates = set()
    for s in syms:
        all_dates.update(indicators_by_symbol[s].index)
    date_index = pd.DatetimeIndex(sorted(all_dates))

    # Create fast date lookup
    date_to_idx = pd.Series(np.arange(len(date_index)), index=date_index)

    # Build numpy array using vectorized operations
    n_dates = len(date_index)
    n_symbols = len(syms)
    panel = np.full((n_dates, n_symbols), np.nan, dtype=np.float64)

    # Also build reverse mapping for write-back
    symbol_to_panel_indices = {}

    for col_idx, s in enumerate(syms):
        df = indicators_by_symbol[s]
        prices = pd.to_numeric(df[price_col], errors='coerce')

        # Get panel row indices for this symbol's dates
        common_dates = df.index.intersection(date_index)
        if len(common_dates) == 0:
            symbol_to_panel_indices[s] = np.array([], dtype=np.int64)
            continue

        panel_indices = date_to_idx.loc[common_dates].values.astype(np.int64)
        df_indices = np.arange(len(df))[df.index.isin(common_dates)]

        # Store mapping for write-back
        symbol_to_panel_indices[s] = panel_indices

        # Fill panel
        valid_prices = prices.iloc[df_indices].values
        valid_mask = ~np.isnan(valid_prices) & (valid_prices > 0)
        panel[panel_indices[valid_mask], col_idx] = valid_prices[valid_mask]

    return panel, date_index, syms, symbol_to_panel_indices


def _write_back_to_dfs(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    result_array: np.ndarray,
    syms: List[str],
    date_index: pd.DatetimeIndex,
    col_name: str,
    pending_writes: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Efficiently write results back to DataFrames using vectorized indexing.

    If pending_writes is provided, accumulates results instead of writing immediately.
    Call _flush_pending_writes() to write all accumulated results at once.
    """
    # Create result DataFrame for efficient reindex
    result_df = pd.DataFrame(result_array, index=date_index, columns=syms, dtype=np.float32)

    if pending_writes is not None:
        # Accumulate results for batch writing
        for s in syms:
            if s not in pending_writes:
                pending_writes[s] = {}
            df = indicators_by_symbol[s]
            pending_writes[s][col_name] = result_df[s].reindex(df.index).values
        return pending_writes
    else:
        # Immediate write (legacy behavior)
        for s in syms:
            df = indicators_by_symbol[s]
            df[col_name] = result_df[s].reindex(df.index).values
        return None


def _write_back_to_dfs_weekly(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    result_array: np.ndarray,
    syms: List[str],
    weekly_dates: pd.DatetimeIndex,
    col_name: str,
    pending_writes: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Efficiently write weekly results back to daily DataFrames with forward-fill.

    If pending_writes is provided, accumulates results instead of writing immediately.
    """
    # Create weekly result DataFrame
    weekly_df = pd.DataFrame(result_array, index=weekly_dates, columns=syms, dtype=np.float32)

    if pending_writes is not None:
        # Accumulate results for batch writing
        for s in syms:
            if s not in pending_writes:
                pending_writes[s] = {}
            df = indicators_by_symbol[s]
            pending_writes[s][col_name] = weekly_df[s].reindex(df.index, method='ffill').values
        return pending_writes
    else:
        # Immediate write (legacy behavior)
        for s in syms:
            df = indicators_by_symbol[s]
            df[col_name] = weekly_df[s].reindex(df.index, method='ffill').values
        return None


def _flush_pending_writes(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    pending_writes: Dict[str, Dict[str, np.ndarray]]
) -> None:
    """
    Write all accumulated pending writes to DataFrames using pd.concat to avoid fragmentation.
    """
    for sym, col_data in pending_writes.items():
        if not col_data:
            continue
        df = indicators_by_symbol[sym]
        # Create DataFrame with all new columns at once
        new_cols_df = pd.DataFrame(col_data, index=df.index)
        # Use pd.concat to add all columns at once (avoids fragmentation)
        indicators_by_symbol[sym] = pd.concat([df, new_cols_df], axis=1)


def add_xsec_momentum_panel(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (5, 20, 60),
    price_col: str = "adjclose",
    sector_map: Optional[Dict[str, str]] = None,
    col_prefix: str = "xsec_mom"
) -> None:
    """
    Add cross-sectional momentum z-scores for each lookback period.

    Optimized with numpy for better multi-core utilization.

    For each lookback L:
    1. Compute L-day log-return per symbol
    2. For each date, z-score across symbols: z = (ret - cross_median) / cross_std
    3. If sector_map provided: sector-neutral first (subtract sector median on each date),
       then z-score across all symbols (so values are directly comparable)

    Features added:
    - {col_prefix}_{L}d_z: Plain cross-sectional momentum z-score
    - {col_prefix}_{L}d_sect_neutral_z: Sector-neutral cross-sectional momentum z-score (if sector_map provided)
    """
    logger.info(f"Computing cross-sectional momentum for lookbacks: {list(lookbacks)}")
    lookbacks = list(lookbacks)

    # Build numpy panel with efficient index mapping
    panel, date_index, syms, _ = _build_price_panel_fast(indicators_by_symbol, price_col)
    if len(syms) == 0:
        logger.warning(f"No symbols have {price_col} column")
        return

    logger.debug(f"Built numpy panel: {panel.shape[0]} dates × {panel.shape[1]} symbols")

    # Compute log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log(panel)
    ret1 = np.diff(logp, axis=0, prepend=np.nan)

    # Pre-compute sector column indices if needed
    sect_col_indices = None
    if sector_map:
        sect_col_indices = {}
        for s_idx, s in enumerate(syms):
            sec = sector_map.get(s)
            if isinstance(sec, str):
                sect_col_indices.setdefault(sec, []).append(s_idx)

    # Accumulate writes to avoid DataFrame fragmentation
    pending_writes: Dict[str, Dict[str, np.ndarray]] = {}

    for L in lookbacks:
        logger.debug(f"Processing {L}-day momentum")

        # Rolling sum of log returns
        min_periods = max(3, L // 3)
        momL = _numpy_rolling_sum(ret1, L, min_periods)

        # Sector-neutral adjustment
        if sector_map and sect_col_indices:
            # Create sector-neutral array
            mom_sect_neutral = momL.copy()

            for sec, col_indices in sect_col_indices.items():
                if len(col_indices) < 2:
                    continue
                # Get sector subset
                sect_data = momL[:, col_indices]
                # Compute sector median per row
                with np.errstate(all='ignore'):
                    sect_median = np.nanmedian(sect_data, axis=1, keepdims=True)
                # Subtract sector median
                mom_sect_neutral[:, col_indices] = sect_data - sect_median

            # Z-score the sector-neutral values
            z_sect = _numpy_cross_sectional_zscore(mom_sect_neutral)
            sect_name = f"{col_prefix}_{L}d_sect_neutral_z"
            _write_back_to_dfs(indicators_by_symbol, z_sect, syms, date_index, sect_name, pending_writes)

        # Plain z-score
        z_plain = _numpy_cross_sectional_zscore(momL)
        plain_name = f"{col_prefix}_{L}d_z"
        _write_back_to_dfs(indicators_by_symbol, z_plain, syms, date_index, plain_name, pending_writes)

    # Flush all accumulated writes at once
    _flush_pending_writes(indicators_by_symbol, pending_writes)

    features_added = len(lookbacks) * (2 if sector_map else 1)
    logger.info(f"Added {features_added} cross-sectional momentum features to {len(syms)} symbols")


def add_weekly_xsec_momentum_panel(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (1, 4, 13),
    price_col: str = "adjclose",
    sector_map: Optional[Dict[str, str]] = None,
    col_prefix: str = "w_xsec_mom"
) -> None:
    """
    Add weekly cross-sectional momentum z-scores for each lookback period.

    Optimized with numpy for better multi-core utilization.
    """
    logger.info(f"Computing weekly cross-sectional momentum for lookbacks: {list(lookbacks)} weeks")
    lookbacks = list(lookbacks)

    # Build numpy panel
    panel, date_index, syms, _ = _build_price_panel_fast(indicators_by_symbol, price_col)
    if len(syms) == 0:
        logger.warning(f"No symbols have {price_col} column")
        return

    # Convert to pandas for resampling (only for date alignment)
    panel_df = pd.DataFrame(panel, index=date_index, columns=syms)
    weekly_panel_df = panel_df.resample('W-FRI').last()
    weekly_dates = weekly_panel_df.index
    weekly_panel = weekly_panel_df.values

    logger.debug(f"Resampled to {len(weekly_dates)} weeks")

    # Compute log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        logp_weekly = np.log(weekly_panel)
    ret1_weekly = np.diff(logp_weekly, axis=0, prepend=np.nan)

    # Pre-compute sector column indices
    sect_col_indices = None
    if sector_map:
        sect_col_indices = {}
        for s_idx, s in enumerate(syms):
            sec = sector_map.get(s)
            if isinstance(sec, str):
                sect_col_indices.setdefault(sec, []).append(s_idx)

    # Accumulate writes to avoid DataFrame fragmentation
    pending_writes: Dict[str, Dict[str, np.ndarray]] = {}

    for L in lookbacks:
        logger.debug(f"Processing {L}-week momentum")

        min_periods = max(1, L // 2)
        momL = _numpy_rolling_sum(ret1_weekly, L, min_periods)

        # Sector-neutral adjustment
        if sector_map and sect_col_indices:
            mom_sect_neutral = momL.copy()
            for sec, col_indices in sect_col_indices.items():
                if len(col_indices) < 2:
                    continue
                sect_data = momL[:, col_indices]
                with np.errstate(all='ignore'):
                    sect_median = np.nanmedian(sect_data, axis=1, keepdims=True)
                mom_sect_neutral[:, col_indices] = sect_data - sect_median
            z_sect = _numpy_cross_sectional_zscore(mom_sect_neutral)

            sect_name = f"{col_prefix}_{L}w_sect_neutral_z"
            _write_back_to_dfs_weekly(indicators_by_symbol, z_sect, syms, weekly_dates, sect_name, pending_writes)

        z_plain = _numpy_cross_sectional_zscore(momL)
        plain_name = f"{col_prefix}_{L}w_z"
        _write_back_to_dfs_weekly(indicators_by_symbol, z_plain, syms, weekly_dates, plain_name, pending_writes)

    # Flush all accumulated writes at once
    _flush_pending_writes(indicators_by_symbol, pending_writes)

    features_added = len(lookbacks) * (2 if sector_map else 1)
    logger.info(f"Added {features_added} weekly cross-sectional momentum features to {len(syms)} symbols")


def add_xsec_percentile_rank(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (5, 20, 60),
    price_col: str = "adjclose",
    sector_map: Optional[Dict[str, str]] = None,
    col_prefix: str = "xsec_pct"
) -> None:
    """
    Add cross-sectional percentile ranks for momentum at each lookback period.

    Optimized with numpy for better multi-core utilization.
    """
    logger.info(f"Computing cross-sectional percentile ranks for lookbacks: {list(lookbacks)}")
    lookbacks = list(lookbacks)

    # Build numpy panel
    panel, date_index, syms, _ = _build_price_panel_fast(indicators_by_symbol, price_col)
    if len(syms) == 0:
        logger.warning(f"No symbols have {price_col} column")
        return

    # Compute log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log(panel)
    ret1 = np.diff(logp, axis=0, prepend=np.nan)

    # Pre-compute sector column indices
    sect_col_indices = None
    if sector_map:
        sect_col_indices = {}
        for s_idx, s in enumerate(syms):
            sec = sector_map.get(s)
            if isinstance(sec, str):
                sect_col_indices.setdefault(sec, []).append(s_idx)

    # Accumulate writes to avoid DataFrame fragmentation
    pending_writes: Dict[str, Dict[str, np.ndarray]] = {}

    for L in lookbacks:
        logger.debug(f"Processing {L}-day percentile rank")

        min_periods = max(3, L // 3)
        momL = _numpy_rolling_sum(ret1, L, min_periods)

        # Cross-sectional percentile rank
        pct_rank = _numpy_cross_sectional_percentile(momL)
        col_name = f"{col_prefix}_{L}d"
        _write_back_to_dfs(indicators_by_symbol, pct_rank, syms, date_index, col_name, pending_writes)

        # Sector-relative percentile
        if sector_map and sect_col_indices:
            sect_pct = np.full_like(pct_rank, np.nan)
            for sec, col_indices in sect_col_indices.items():
                if len(col_indices) < 2:
                    sect_pct[:, col_indices] = pct_rank[:, col_indices]
                    continue
                sect_data = momL[:, col_indices]
                sect_pct[:, col_indices] = _numpy_cross_sectional_percentile(sect_data)

            # Handle symbols without sector - use overall percentile
            for s_idx, s in enumerate(syms):
                if s not in sector_map:
                    sect_pct[:, s_idx] = pct_rank[:, s_idx]

            sect_col_name = f"{col_prefix}_{L}d_sect"
            _write_back_to_dfs(indicators_by_symbol, sect_pct, syms, date_index, sect_col_name, pending_writes)

    # Flush all accumulated writes at once
    _flush_pending_writes(indicators_by_symbol, pending_writes)

    features_added = len(lookbacks) * (2 if sector_map else 1)
    logger.info(f"Added {features_added} cross-sectional percentile features to {len(syms)} symbols")


def add_weekly_xsec_percentile_rank(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (1, 4, 13),
    price_col: str = "adjclose",
    sector_map: Optional[Dict[str, str]] = None,
    col_prefix: str = "w_xsec_pct"
) -> None:
    """
    Add weekly cross-sectional percentile ranks for momentum.

    Optimized with numpy for better multi-core utilization.
    """
    logger.info(f"Computing weekly cross-sectional percentile ranks for lookbacks: {list(lookbacks)} weeks")
    lookbacks = list(lookbacks)

    # Build numpy panel
    panel, date_index, syms, _ = _build_price_panel_fast(indicators_by_symbol, price_col)
    if len(syms) == 0:
        logger.warning(f"No symbols have {price_col} column")
        return

    # Convert to pandas for resampling
    panel_df = pd.DataFrame(panel, index=date_index, columns=syms)
    weekly_panel_df = panel_df.resample('W-FRI').last()
    weekly_dates = weekly_panel_df.index
    weekly_panel = weekly_panel_df.values

    logger.debug(f"Resampled to {len(weekly_dates)} weeks")

    # Compute log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        logp_weekly = np.log(weekly_panel)
    ret1_weekly = np.diff(logp_weekly, axis=0, prepend=np.nan)

    # Pre-compute sector column indices
    sect_col_indices = None
    if sector_map:
        sect_col_indices = {}
        for s_idx, s in enumerate(syms):
            sec = sector_map.get(s)
            if isinstance(sec, str):
                sect_col_indices.setdefault(sec, []).append(s_idx)

    # Accumulate writes to avoid DataFrame fragmentation
    pending_writes: Dict[str, Dict[str, np.ndarray]] = {}

    for L in lookbacks:
        logger.debug(f"Processing {L}-week percentile rank")

        min_periods = max(1, L // 2)
        momL = _numpy_rolling_sum(ret1_weekly, L, min_periods)

        pct_rank = _numpy_cross_sectional_percentile(momL)
        col_name = f"{col_prefix}_{L}w"
        _write_back_to_dfs_weekly(indicators_by_symbol, pct_rank, syms, weekly_dates, col_name, pending_writes)

        # Sector-relative percentile
        if sector_map and sect_col_indices:
            sect_pct = np.full_like(pct_rank, np.nan)
            for sec, col_indices in sect_col_indices.items():
                if len(col_indices) < 2:
                    sect_pct[:, col_indices] = pct_rank[:, col_indices]
                    continue
                sect_data = momL[:, col_indices]
                sect_pct[:, col_indices] = _numpy_cross_sectional_percentile(sect_data)

            sect_col_name = f"{col_prefix}_{L}w_sect"
            _write_back_to_dfs_weekly(indicators_by_symbol, sect_pct, syms, weekly_dates, sect_col_name, pending_writes)

    # Flush all accumulated writes at once
    _flush_pending_writes(indicators_by_symbol, pending_writes)

    features_added = len(lookbacks) * (2 if sector_map else 1)
    logger.info(f"Added {features_added} weekly cross-sectional percentile features to {len(syms)} symbols")
