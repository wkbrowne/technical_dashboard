"""
Relative strength features comparing individual securities to market and sector benchmarks.

This module computes relative strength metrics by comparing each symbol's performance
to broader market indices (like SPY) and sector-specific ETFs.

Parallelization: Per-symbol RS computation is embarrassingly parallel since each
symbol only depends on its own price series and pre-computed benchmark series.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Import parallel config for stocks_per_worker based parallelism
try:
    from ..config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER
except ImportError:
    from src.config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER

logger = logging.getLogger(__name__)


# Try to import numba for accelerated slope computation
try:
    from numba import jit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @jit(nopython=True)
    def _numba_rolling_slope(y_arr, window, min_periods, alpha):
        """Numba-accelerated rolling linear slope computation."""
        n = len(y_arr)
        result = np.empty(n)
        result[:] = np.nan

        # Precompute x values for the window
        x = np.arange(window, dtype=np.float64)
        x_mean = np.mean(x)
        x_centered = x - x_mean
        var_x = np.sum(x_centered ** 2)

        for i in range(window - 1, n):
            y_window = y_arr[i - window + 1:i + 1]

            # Count valid (non-nan) values
            valid_count = 0
            y_sum = 0.0
            for j in range(window):
                if not np.isnan(y_window[j]):
                    valid_count += 1
                    y_sum += y_window[j]

            if valid_count < min_periods:
                continue

            y_mean = y_sum / valid_count

            # Compute covariance: Cov(x, y) = sum(x_centered * y_centered)
            cov_xy = 0.0
            for j in range(window):
                if not np.isnan(y_window[j]):
                    cov_xy += x_centered[j] * (y_window[j] - y_mean)

            # Ridge regression slope: slope = Cov(x,y) / (Var(x) + alpha)
            result[i] = cov_xy / (var_x + alpha)

        return result


def _bayesian_linear_slope(series: pd.Series, window: int, alpha: float = 0.01) -> pd.Series:
    """
    Calculate rolling Bayesian linear regression slope with L2 regularization.

    Uses ridge regression (L2 regularization) which is equivalent to Bayesian linear
    regression with a Gaussian prior on the slope parameter.

    This implementation uses numba JIT compilation when available (~3000x faster),
    with a numpy strided fallback (~500x faster than naive rolling.apply).

    Args:
        series: Input time series
        window: Rolling window size
        alpha: Regularization strength (smaller = less regularization)
               Default 0.01 provides slight regularization without over-smoothing

    Returns:
        Rolling slope series (per-period change)
    """
    min_periods = max(3, window // 2)
    y_arr = series.values.astype('float64')

    # Use numba if available (fastest)
    if _HAS_NUMBA:
        result = _numba_rolling_slope(y_arr, window, min_periods, alpha)
        return pd.Series(result, index=series.index, dtype='float32')

    # Fallback: Vectorized numpy with strided arrays
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(y_arr)

    # Pad with NaNs at start to handle edge cases
    padded = np.concatenate([np.full(window - 1, np.nan), y_arr])
    windows = sliding_window_view(padded, window)  # Shape: (n, window)

    # Precompute x values for the window
    x = np.arange(window, dtype='float64')
    x_mean = x.mean()
    x_centered = x - x_mean
    var_x = (x_centered ** 2).sum()

    # Vectorized: y_mean for each window
    y_means = np.nanmean(windows, axis=1, keepdims=True)
    y_centered = windows - y_means

    # Replace NaNs with 0 for dot product (they won't contribute)
    y_centered_clean = np.nan_to_num(y_centered, nan=0.0)

    # Cov(x, y) = sum(x_centered * y_centered) for each window
    cov_xy = (x_centered * y_centered_clean).sum(axis=1)

    # Ridge regression slope
    slopes = cov_xy / (var_x + alpha)

    # Mask out windows with insufficient valid data
    valid_counts = np.sum(~np.isnan(windows), axis=1)
    slopes[valid_counts < min_periods] = np.nan

    return pd.Series(slopes, index=series.index, dtype='float32')


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI on a series (typically relative strength ratio)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.astype('float32')


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD on a series (typically relative strength ratio).

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.astype('float32'), signal_line.astype('float32'), histogram.astype('float32')


def _compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling z-score of a series."""
    roll_mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    roll_std = series.rolling(window, min_periods=max(5, window // 3)).std()
    zscore = (series - roll_mean) / roll_std.replace(0, np.nan)
    return zscore.astype('float32')


def _compute_rs_batch(
    work_items_batch: List[Tuple[str, pd.Index, np.ndarray, Dict[str, Any]]],
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    spy_symbol: str,
    qqq_symbol: str = "QQQ"
) -> List[Tuple[str, Dict[str, pd.Series]]]:
    """
    Process a batch of symbols for RS computation to reduce IPC overhead.

    Args:
        work_items_batch: List of (symbol, px_index, px_values, mapping) tuples
        benchmarks: Pre-extracted benchmark price series
        spy_symbol: Symbol used as market benchmark
        qqq_symbol: Symbol used as QQQ benchmark (growth/liquidity factor)

    Returns:
        List of (symbol, rs_features_dict) tuples
    """
    results = []
    for sym, px_index, px_values, mapping in work_items_batch:
        rs_features = _compute_rs_for_symbol(
            sym, px_index, px_values, benchmarks, mapping, spy_symbol, qqq_symbol
        )
        results.append((sym, rs_features))
    return results


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _get_required_benchmarks_for_rs_batch(
    work_items_batch: List[Tuple[str, pd.Index, np.ndarray, Dict[str, Any]]],
    all_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    spy_symbol: str = "SPY",
    qqq_symbol: str = "QQQ"
) -> Dict[str, Tuple[pd.Index, np.ndarray]]:
    """
    Extract only the benchmarks needed by a specific batch of symbols for RS computation.

    This reduces memory usage by only sending relevant benchmark data to each worker
    instead of the entire benchmark dictionary.

    Args:
        work_items_batch: List of (symbol, px_index, px_values, mapping) tuples
        all_benchmarks: Complete benchmark dictionary
        spy_symbol: Market benchmark symbol (always included)
        qqq_symbol: Growth benchmark symbol (always included)

    Returns:
        Subset of benchmarks needed by this batch
    """
    # Core benchmarks always needed
    required = {'SPY', 'QQQ', 'RSP'}

    # Add ETFs referenced by symbols in this batch
    for item in work_items_batch:
        if len(item) >= 4 and isinstance(item[3], dict):
            mapping = item[3]
            if mapping.get('sector_etf'):
                required.add(mapping['sector_etf'])
            if mapping.get('equal_weight_etf'):
                required.add(mapping['equal_weight_etf'])
            if mapping.get('subsector_etf'):
                required.add(mapping['subsector_etf'])

    # Extract only required benchmarks
    subset = {}
    for key in required:
        if key in all_benchmarks:
            subset[key] = all_benchmarks[key]

    return subset


def _compute_relative_strength_block(
    price: pd.Series,
    bench: pd.Series,
    look: int = 60,
    slope_win: int = 10,
    slope_alpha: float = 0.01,
    compute_extended: bool = True
) -> Dict[str, pd.Series]:
    """
    Compute comprehensive relative strength metrics for a single price series vs benchmark.

    Args:
        price: Price series for the security
        bench: Benchmark price series
        look: Lookback window for RS normalization
        slope_win: Window for computing RS slope (in periods, default 10)
        slope_alpha: Regularization strength for Bayesian slope (default 0.01)
        compute_extended: If True, also compute RSI, MACD, z-score on RS

    Returns:
        Dict with keys:
        - 'rs': Raw relative strength ratio (price / benchmark)
        - 'rs_norm': Normalized RS (vs rolling mean, mean-centered)
        - 'rs_slope': Bayesian linear regression slope of log(RS)
        - 'rs_rsi': RSI(14) computed on the RS ratio
        - 'rs_macd': MACD line on RS ratio
        - 'rs_macd_signal': MACD signal line
        - 'rs_macd_hist': MACD histogram
        - 'rs_zscore': Rolling z-score of RS ratio
    """
    # Align benchmark to price index and handle missing values
    bench_aligned = pd.to_numeric(bench.reindex(price.index), errors='coerce').replace(0, np.nan)

    # Raw relative strength ratio
    rs = price / bench_aligned

    # Normalized RS: (current_rs / rolling_mean_rs) - 1
    roll = rs.rolling(look, min_periods=max(5, look//3)).mean()
    rs_norm = (rs / roll) - 1.0

    # RS slope: Bayesian linear regression on log(RS) with regularization
    log_rs = np.log(rs.replace(0, np.nan))
    rs_slope = _bayesian_linear_slope(log_rs, window=slope_win, alpha=slope_alpha)

    result = {
        'rs': rs.astype('float32'),
        'rs_norm': rs_norm.astype('float32'),
        'rs_slope': rs_slope.astype('float32'),
    }

    # Extended indicators on the RS series
    if compute_extended:
        # RSI on RS ratio - measures overbought/oversold in relative performance
        result['rs_rsi'] = _compute_rsi(rs, period=14)

        # MACD on RS ratio - measures momentum in relative performance
        macd_line, macd_signal, macd_hist = _compute_macd(rs, fast=12, slow=26, signal=9)
        result['rs_macd'] = macd_line
        result['rs_macd_signal'] = macd_signal
        result['rs_macd_hist'] = macd_hist

        # Z-score of RS ratio - measures how extreme current RS is
        result['rs_zscore'] = _compute_zscore(rs, window=20)

    return result


def _compute_rs_for_symbol(
    sym: str,
    px_index: pd.Index,
    px_values: np.ndarray,
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    symbol_mapping: Dict[str, Any],
    spy_symbol: str = "SPY",
    qqq_symbol: str = "QQQ"
) -> Dict[str, pd.Series]:
    """
    Compute all relative strength features for a single symbol.

    This is a pure function designed for parallel execution - it takes
    pre-extracted numpy arrays as input and returns computed features.

    Args:
        sym: Symbol ticker
        px_index: DatetimeIndex for the symbol's price data
        px_values: Numpy array of adjusted close prices
        benchmarks: Dict mapping benchmark name -> (index, values) tuple
        symbol_mapping: Dict with keys like 'sector_etf', 'subsector_etf', 'equal_weight_etf'
        spy_symbol: Symbol used as market benchmark
        qqq_symbol: Symbol used as QQQ benchmark (growth/liquidity factor)

    Returns:
        Dict mapping column_name -> pd.Series with computed RS features
    """
    result = {}
    px = pd.Series(px_values, index=px_index, dtype='float64')

    # Skip if this is a benchmark symbol
    if sym == spy_symbol or sym == qqq_symbol:
        return result

    # 1) RS vs SPY
    if 'SPY' in benchmarks:
        spy_idx, spy_vals = benchmarks['SPY']
        spy_px = pd.Series(spy_vals, index=spy_idx, dtype='float64')
        rs_metrics = _compute_relative_strength_block(px, spy_px, compute_extended=True)
        result["rel_strength_spy"] = rs_metrics['rs']
        result["rel_strength_spy_norm"] = rs_metrics['rs_norm']
        # NOTE: slope20 removed due to near-zero variance (not useful for ML)
        result["rel_strength_spy_rsi"] = rs_metrics['rs_rsi']
        result["rel_strength_spy_macd"] = rs_metrics['rs_macd']
        result["rel_strength_spy_macd_signal"] = rs_metrics['rs_macd_signal']
        result["rel_strength_spy_macd_hist"] = rs_metrics['rs_macd_hist']
        result["rel_strength_spy_zscore"] = rs_metrics['rs_zscore']

    # 2) RS vs QQQ (growth/liquidity factor)
    # Applied to ALL stocks regardless of sector to capture growth sensitivity
    if 'QQQ' in benchmarks:
        qqq_idx, qqq_vals = benchmarks['QQQ']
        qqq_px = pd.Series(qqq_vals, index=qqq_idx, dtype='float64')
        rs_metrics = _compute_relative_strength_block(px, qqq_px, compute_extended=True)
        result["rel_strength_qqq"] = rs_metrics['rs']
        result["rel_strength_qqq_norm"] = rs_metrics['rs_norm']
        result["rel_strength_qqq_rsi"] = rs_metrics['rs_rsi']
        result["rel_strength_qqq_macd"] = rs_metrics['rs_macd']
        result["rel_strength_qqq_macd_signal"] = rs_metrics['rs_macd_signal']
        result["rel_strength_qqq_macd_hist"] = rs_metrics['rs_macd_hist']
        result["rel_strength_qqq_zscore"] = rs_metrics['rs_zscore']

        # QQQ vs SPY spread (captures growth vs value regime)
        # Positive = stock outperforming relative to QQQ more than SPY
        if 'SPY' in benchmarks:
            spy_rs = result.get("rel_strength_spy")
            qqq_rs = rs_metrics['rs']
            if spy_rs is not None:
                # Ratio of QQQ-relative to SPY-relative RS
                rs_qqq_spy_spread = (qqq_rs / spy_rs.replace(0, np.nan)).astype('float32')
                result["rel_strength_qqq_spy_spread"] = rs_qqq_spy_spread
                # Normalized spread (mean-centered)
                roll = rs_qqq_spy_spread.rolling(60, min_periods=20).mean()
                result["rel_strength_qqq_spy_spread_norm"] = ((rs_qqq_spy_spread / roll) - 1.0).astype('float32')

    # 3) RS vs RSP (equal-weight market)
    if 'RSP' in benchmarks:
        rsp_idx, rsp_vals = benchmarks['RSP']
        rsp_px = pd.Series(rsp_vals, index=rsp_idx, dtype='float64')
        rs_metrics = _compute_relative_strength_block(px, rsp_px, compute_extended=True)
        result["rel_strength_rsp"] = rs_metrics['rs']
        result["rel_strength_rsp_norm"] = rs_metrics['rs_norm']
        result["rel_strength_rsp_rsi"] = rs_metrics['rs_rsi']
        result["rel_strength_rsp_macd"] = rs_metrics['rs_macd']
        result["rel_strength_rsp_macd_hist"] = rs_metrics['rs_macd_hist']
        result["rel_strength_rsp_zscore"] = rs_metrics['rs_zscore']

    # 4) RS vs Sector ETF
    sector_etf = symbol_mapping.get('sector_etf')
    if sector_etf and sector_etf in benchmarks:
        sec_idx, sec_vals = benchmarks[sector_etf]
        sec_px = pd.Series(sec_vals, index=sec_idx, dtype='float64')
        rs_metrics = _compute_relative_strength_block(px, sec_px, compute_extended=True)
        result["rel_strength_sector"] = rs_metrics['rs']
        result["rel_strength_sector_norm"] = rs_metrics['rs_norm']
        result["rel_strength_sector_rsi"] = rs_metrics['rs_rsi']
        result["rel_strength_sector_macd"] = rs_metrics['rs_macd']
        result["rel_strength_sector_macd_signal"] = rs_metrics['rs_macd_signal']
        result["rel_strength_sector_macd_hist"] = rs_metrics['rs_macd_hist']
        result["rel_strength_sector_zscore"] = rs_metrics['rs_zscore']

        # Sector vs market (shared across all stocks in sector)
        if 'SPY' in benchmarks:
            spy_idx, spy_vals = benchmarks['SPY']
            spy_px = pd.Series(spy_vals, index=spy_idx, dtype='float64')
            rs_sect_mkt = _compute_relative_strength_block(sec_px, spy_px, compute_extended=False)
            result["rel_strength_sector_vs_market"] = rs_sect_mkt['rs']
            result["rel_strength_sector_vs_market_norm"] = rs_sect_mkt['rs_norm']

    # 5) RS vs Equal-Weight Best-Match ETF (replaces broken equal-weight sector)
    # Uses R²-based best-match from equal-weight ETF candidates
    ew_etf = symbol_mapping.get('equal_weight_etf')
    if ew_etf and ew_etf in benchmarks:
        ew_idx, ew_vals = benchmarks[ew_etf]
        ew_px = pd.Series(ew_vals, index=ew_idx, dtype='float64')
        rs_metrics = _compute_relative_strength_block(px, ew_px, compute_extended=True)
        result["rel_strength_sector_ew"] = rs_metrics['rs']
        result["rel_strength_sector_ew_norm"] = rs_metrics['rs_norm']
        result["rel_strength_sector_ew_rsi"] = rs_metrics['rs_rsi']
        result["rel_strength_sector_ew_macd"] = rs_metrics['rs_macd']
        result["rel_strength_sector_ew_macd_hist"] = rs_metrics['rs_macd_hist']
        result["rel_strength_sector_ew_zscore"] = rs_metrics['rs_zscore']

    # NOTE: Subsector features (rel_strength_subsector_*) have been REMOVED.
    # They had 40-48% NaN rates due to unstable correlation-based subsector discovery.
    # The best-match ETF selection (using R² from factor_regression.py) now provides
    # stable sector/subsector references through sector_etf and equal_weight_etf.

    return result


def add_relative_strength(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    spy_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    enhanced_mappings: Optional[Dict[str, Dict]] = None,
    n_jobs: int = -1
) -> None:
    """
    Add comprehensive relative strength features with cap-weighted, equal-weighted, and subsector analysis.

    This function is parallelized across symbols using joblib. Each symbol's RS computation
    is independent and only depends on pre-computed benchmark price series.

    Standard features (always added if benchmarks available):
    - rel_strength_spy: Raw relative strength vs SPY
    - rel_strength_spy_norm: Normalized relative strength vs SPY
    - rel_strength_qqq: Raw relative strength vs QQQ (growth/liquidity factor)
    - rel_strength_qqq_norm: Normalized relative strength vs QQQ
    - rel_strength_qqq_spy_spread: Ratio of QQQ-relative to SPY-relative RS
    - rel_strength_sector: Raw relative strength vs sector ETF
    - rel_strength_sector_norm: Normalized relative strength vs sector ETF

    Enhanced features (if enhanced_mappings provided):
    - rel_strength_rsp: Raw relative strength vs RSP (equal-weight market)
    - rel_strength_subsector: Raw relative strength vs subsector ETF
    - etc.

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        sectors: Optional mapping of symbol -> sector name
        sector_to_etf: Optional mapping of lowercase sector name -> ETF symbol
        spy_symbol: Symbol to use as market benchmark
        qqq_symbol: Symbol to use as growth/liquidity benchmark (default: QQQ)
        enhanced_mappings: Optional enhanced sector/subsector mappings from sector_mapping module
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    import os
    logger.info(f"Computing relative strength features using {spy_symbol} and {qqq_symbol} as benchmarks (parallel)")

    # Prepare sector mappings (convert to lowercase)
    sector_to_etf = sector_to_etf or {}
    sector_to_etf_lc = {k.lower(): v for k, v in sector_to_etf.items()}

    # =========================================================================
    # Step 1: Pre-extract all benchmark price series as (index, numpy_array) tuples
    # This makes them picklable and efficiently shareable across parallel workers
    # =========================================================================
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # SPY benchmark
    spy_data = indicators_by_symbol.get(spy_symbol)
    if spy_data is not None and "adjclose" in spy_data.columns:
        spy_px = pd.to_numeric(spy_data["adjclose"], errors='coerce')
        benchmarks['SPY'] = (spy_px.index, spy_px.values)
        logger.debug(f"Loaded SPY benchmark: {len(spy_px)} data points")

    # RSP benchmark (equal-weight market)
    rsp_data = indicators_by_symbol.get('RSP')
    if rsp_data is not None and "adjclose" in rsp_data.columns:
        rsp_px = pd.to_numeric(rsp_data["adjclose"], errors='coerce')
        benchmarks['RSP'] = (rsp_px.index, rsp_px.values)
        logger.debug(f"Loaded RSP benchmark: {len(rsp_px)} data points")

    # QQQ benchmark (growth/liquidity factor)
    qqq_data = indicators_by_symbol.get(qqq_symbol)
    if qqq_data is not None and "adjclose" in qqq_data.columns:
        qqq_px = pd.to_numeric(qqq_data["adjclose"], errors='coerce')
        benchmarks['QQQ'] = (qqq_px.index, qqq_px.values)
        logger.debug(f"Loaded QQQ benchmark: {len(qqq_px)} data points")
    else:
        logger.warning(f"{qqq_symbol} missing or has no 'adjclose' column - QQQ features will be skipped")

    # Collect all ETFs needed (sector, equal-weight, subsector)
    all_etfs_needed = set()

    # From sector_to_etf
    if sectors and sector_to_etf:
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sector_to_etf_lc.get(sec.lower())
                if etf:
                    all_etfs_needed.add(etf)

    # From enhanced_mappings
    if enhanced_mappings:
        for mapping in enhanced_mappings.values():
            if mapping.get('sector_etf'):
                all_etfs_needed.add(mapping['sector_etf'])
            if mapping.get('equal_weight_etf'):
                all_etfs_needed.add(mapping['equal_weight_etf'])
            if mapping.get('subsector_etf'):
                all_etfs_needed.add(mapping['subsector_etf'])

    # Load all needed ETF benchmarks
    for etf in all_etfs_needed:
        if etf in benchmarks:
            continue
        etf_df = indicators_by_symbol.get(etf)
        if etf_df is not None and "adjclose" in etf_df.columns:
            etf_px = pd.to_numeric(etf_df["adjclose"], errors='coerce')
            benchmarks[etf] = (etf_px.index, etf_px.values)

    logger.info(f"Loaded {len(benchmarks)} benchmark price series for RS computation")

    # =========================================================================
    # Step 2: Build per-symbol mapping info
    # =========================================================================
    symbol_mappings: Dict[str, Dict[str, Any]] = {}

    for sym, df in indicators_by_symbol.items():
        if "adjclose" not in df.columns:
            continue

        mapping = {}

        # Sector ETF from sectors dict
        if sectors and sector_to_etf:
            sec = sectors.get(sym)
            if isinstance(sec, str):
                etf = sector_to_etf_lc.get(sec.lower())
                if etf:
                    mapping['sector_etf'] = etf

        # Enhanced mappings override
        if enhanced_mappings and sym in enhanced_mappings:
            em = enhanced_mappings[sym]
            if em.get('sector_etf'):
                mapping['sector_etf'] = em['sector_etf']
            if em.get('equal_weight_etf'):
                mapping['equal_weight_etf'] = em['equal_weight_etf']
            if em.get('subsector_etf'):
                mapping['subsector_etf'] = em['subsector_etf']

        symbol_mappings[sym] = mapping

    # =========================================================================
    # Step 3: Prepare work items (symbol, price_index, price_values, mapping)
    # =========================================================================
    work_items = []
    for sym, df in indicators_by_symbol.items():
        if "adjclose" not in df.columns:
            continue
        if sym in benchmarks:  # Skip benchmark symbols themselves
            continue

        px = pd.to_numeric(df["adjclose"], errors='coerce')
        work_items.append((sym, px.index, px.values, symbol_mappings.get(sym, {})))

    n_symbols = len(work_items)

    # =========================================================================
    # Step 4: Parallel computation with batching (stocks_per_worker based)
    # =========================================================================
    chunk_size = DEFAULT_STOCKS_PER_WORKER
    n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)
    n_chunks = max(1, (n_symbols + chunk_size - 1) // chunk_size)

    logger.info(f"Computing RS features for {n_symbols} symbols in {n_chunks} batches "
                f"({chunk_size} stocks/worker, {n_workers} workers)")

    # Use sequential processing for small datasets
    if n_symbols < 10 or n_workers == 1:
        logger.info("Using sequential processing for RS computation")
        results = []
        for sym, px_index, px_values, mapping in work_items:
            rs_features = _compute_rs_for_symbol(
                sym, px_index, px_values, benchmarks, mapping, spy_symbol, qqq_symbol
            )
            results.append((sym, rs_features))
    else:
        # Create batches for parallel processing with subsetted benchmarks
        work_chunks = _chunk_list(work_items, chunk_size)

        # Pre-compute subsetted benchmarks for each chunk to reduce serialization
        # Each worker only gets the benchmarks it needs (core + batch-specific ETFs)
        chunk_benchmarks = []
        for chunk in work_chunks:
            benchmark_subset = _get_required_benchmarks_for_rs_batch(
                chunk, benchmarks, spy_symbol, qqq_symbol
            )
            chunk_benchmarks.append(benchmark_subset)

        # Log benchmark reduction
        total_benchmarks = len(benchmarks)
        avg_benchmarks = sum(len(b) for b in chunk_benchmarks) / len(chunk_benchmarks) if chunk_benchmarks else 0
        logger.info(f"Benchmark subsetting: {total_benchmarks} total -> {avg_benchmarks:.1f} avg per batch")

        try:
            batch_results = Parallel(
                n_jobs=n_workers,
                backend='loky',
                verbose=0,
                prefer='processes'
            )(
                delayed(_compute_rs_batch)(chunk, chunk_benchmarks[i], spy_symbol, qqq_symbol)
                for i, chunk in enumerate(work_chunks)
            )

            # Flatten batch results
            results = []
            for batch in batch_results:
                results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel RS processing failed ({e}), falling back to sequential")
            results = []
            for sym, px_index, px_values, mapping in work_items:
                rs_features = _compute_rs_for_symbol(
                    sym, px_index, px_values, benchmarks, mapping, spy_symbol, qqq_symbol
                )
                results.append((sym, rs_features))

    # =========================================================================
    # Step 5: Apply results back to DataFrames
    # =========================================================================
    rs_spy_count = 0
    rs_qqq_count = 0
    rs_sector_count = 0
    rs_subsector_count = 0

    for sym, rs_features in results:
        if not rs_features:
            continue

        df = indicators_by_symbol[sym]
        for col_name, series in rs_features.items():
            df[col_name] = series

        # Count features added
        if "rel_strength_spy" in rs_features:
            rs_spy_count += 1
        if "rel_strength_qqq" in rs_features:
            rs_qqq_count += 1
        if "rel_strength_sector" in rs_features:
            rs_sector_count += 1
        if "rel_strength_subsector" in rs_features:
            rs_subsector_count += 1

    logger.info(f"Added SPY relative strength to {rs_spy_count} symbols, "
                f"QQQ relative strength to {rs_qqq_count} symbols, "
                f"sector relative strength to {rs_sector_count} symbols, "
                f"subsector relative strength to {rs_subsector_count} symbols")