"""
Alpha momentum features based on CAPM residuals and risk-adjusted returns.

This module computes alpha (excess return) features by calculating rolling CAPM
beta and alpha against market and sector benchmarks, then deriving momentum
features from these alpha streams.

Parallelization: Per-symbol alpha computation is embarrassingly parallel since
each symbol only depends on its own return series and pre-computed benchmark returns.
"""
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def _rolling_beta_alpha(ret: pd.Series, bench_ret: pd.Series, win: int, min_periods: int = None) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling CAPM beta and alpha using rolling covariance and variance.

    Args:
        ret: Return series for the security
        bench_ret: Benchmark return series
        win: Rolling window for beta/alpha calculation
        min_periods: Minimum observations in window (default: win//3, at least 20)

    Returns:
        Tuple of (beta, alpha) where:
        - beta_t = Cov(ret, bench_ret) / Var(bench_ret)
        - alpha_t = ret - beta_t * bench_ret
    """
    ret = pd.to_numeric(ret, errors='coerce')
    bench_ret = pd.to_numeric(bench_ret, errors='coerce')

    # Default min_periods to allow for some NaN values in the window
    # This is critical when benchmark has holiday gaps that don't align with stock data
    if min_periods is None:
        min_periods = max(20, win // 3)

    # Rolling covariance and variance with min_periods to handle NaN gaps
    # Suppress numpy warnings from covariance calculation (expected when variance is 0 or NaN present)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cov = ret.rolling(win, min_periods=min_periods).cov(bench_ret)
        var = bench_ret.rolling(win, min_periods=min_periods).var()

    # Beta calculation
    beta = cov / var.replace(0, np.nan)

    # Alpha: actual return minus expected return (beta * benchmark return)
    alpha = ret - beta * bench_ret

    return beta.astype('float32'), alpha.astype('float32')


def _alpha_momentum_from_residual(
    alpha_ret: pd.Series, 
    windows: Tuple[int, ...] = (20, 60, 120), 
    ema_span: int = 10, 
    prefix: str = "alpha_mom"
) -> Dict[str, pd.Series]:
    """
    Create momentum features from alpha residual return series.
    
    Args:
        alpha_ret: Alpha residual return series
        windows: Windows for cumulative alpha momentum
        ema_span: EMA span for smoothing
        prefix: Prefix for feature names
        
    Returns:
        Dictionary of momentum features
    """
    out = {}
    
    # EMA of residual returns (fast "flow" indicator)
    out[f"{prefix}_ema{ema_span}"] = (
        alpha_ret.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype('float32')
    )
    
    # Windowed cumulative alpha with EMA smoothing
    for w in windows:
        # Cumulative alpha over horizon (sum of residual returns)
        mom = alpha_ret.rolling(w, min_periods=max(3, w//3)).sum()
        # EMA smoothing of cumulative alpha
        out[f"{prefix}_{w}_ema{ema_span}"] = (
            mom.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype('float32')
        )
    
    return out


def _compute_alpha_batch(
    work_items_batch: List[Tuple[str, pd.Index, np.ndarray, Optional[str]]],
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    market_symbol: str,
    beta_win: int,
    windows: Tuple[int, ...],
    ema_span: int,
    qqq_symbol: str = "QQQ"
) -> List[Tuple[str, Dict[str, pd.Series]]]:
    """
    Process a batch of symbols for alpha computation to reduce IPC overhead.

    Args:
        work_items_batch: List of (symbol, ret_index, ret_values, sector_etf) tuples
        benchmarks: Pre-extracted benchmark return series
        market_symbol: Symbol used as market benchmark
        beta_win: Rolling window for beta/alpha calculation
        windows: Momentum calculation windows
        ema_span: EMA span for smoothing
        qqq_symbol: Symbol used as QQQ benchmark (growth/liquidity factor)

    Returns:
        List of (symbol, alpha_features_dict) tuples
    """
    results = []
    for sym, ret_index, ret_values, sector_etf in work_items_batch:
        alpha_features = _compute_alpha_for_symbol(
            sym, ret_index, ret_values, benchmarks, sector_etf,
            market_symbol, beta_win, windows, ema_span, qqq_symbol
        )
        results.append((sym, alpha_features))
    return results


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _get_required_benchmarks_for_alpha_batch(
    work_items_batch: List[Tuple[str, pd.Index, np.ndarray, Optional[str]]],
    all_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ"
) -> Dict[str, Tuple[pd.Index, np.ndarray]]:
    """
    Extract only the benchmarks needed by a specific batch for alpha computation.

    This reduces serialization overhead by only sending relevant benchmark data to
    each worker instead of the entire benchmark dictionary.

    Args:
        work_items_batch: List of (symbol, ret_index, ret_values, sector_etf) tuples
        all_benchmarks: Complete benchmark dictionary
        market_symbol: Market benchmark symbol (always included)
        qqq_symbol: QQQ benchmark symbol (always included)

    Returns:
        Subset of benchmarks needed by this batch
    """
    # Core benchmarks always needed for alpha computation
    required = {market_symbol, qqq_symbol}

    # Add sector ETFs referenced by symbols in this batch
    for item in work_items_batch:
        if len(item) >= 4 and item[3]:  # item[3] is symbol_sector_etf
            required.add(item[3])

    # Extract only required benchmarks that exist
    return {k: v for k, v in all_benchmarks.items() if k in required}


def _compute_alpha_for_symbol(
    sym: str,
    ret_index: pd.Index,
    ret_values: np.ndarray,
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    symbol_sector_etf: Optional[str],
    market_symbol: str,
    beta_win: int,
    windows: Tuple[int, ...],
    ema_span: int,
    qqq_symbol: str = "QQQ"
) -> Dict[str, pd.Series]:
    """
    Compute alpha momentum features for a single symbol.

    This is a pure function designed for parallel execution.

    Args:
        sym: Symbol ticker
        ret_index: DatetimeIndex for the symbol's return data
        ret_values: Numpy array of returns
        benchmarks: Dict mapping benchmark name -> (index, values) tuple
        symbol_sector_etf: ETF symbol for this stock's sector (or None)
        market_symbol: Symbol used as market benchmark
        beta_win: Rolling window for beta/alpha calculation
        windows: Momentum calculation windows
        ema_span: EMA span for smoothing
        qqq_symbol: Symbol for QQQ benchmark (growth/liquidity factor)

    Returns:
        Dict mapping column_name -> pd.Series with computed alpha features
    """
    result = {}
    r = pd.Series(ret_values, index=ret_index, dtype='float64')

    # Skip if this is a benchmark symbol
    if sym == market_symbol or sym == qqq_symbol:
        return result

    alpha_mkt = None
    alpha_sec = None
    alpha_qqq = None

    # Market alpha (vs SPY)
    if market_symbol in benchmarks:
        mkt_idx, mkt_vals = benchmarks[market_symbol]
        mkt_ret = pd.Series(mkt_vals, index=mkt_idx, dtype='float64').reindex(ret_index)
        beta_mkt, alpha_mkt = _rolling_beta_alpha(r, mkt_ret, win=beta_win)
        # Note: beta_spy_simple is univariate cov/var, distinct from beta_market (joint factor model)
        result['beta_spy_simple'] = beta_mkt
        result['alpha_resid_spy'] = alpha_mkt

        # Create momentum features from market alpha
        mkt_feats = _alpha_momentum_from_residual(
            alpha_mkt, windows=windows, ema_span=ema_span, prefix="alpha_mom_spy"
        )
        result.update(mkt_feats)

    # QQQ alpha (vs QQQ - growth/liquidity factor)
    # Applied to ALL stocks regardless of sector to capture growth sensitivity
    if qqq_symbol in benchmarks:
        qqq_idx, qqq_vals = benchmarks[qqq_symbol]
        qqq_ret = pd.Series(qqq_vals, index=qqq_idx, dtype='float64').reindex(ret_index)
        beta_qqq_val, alpha_qqq = _rolling_beta_alpha(r, qqq_ret, win=beta_win)
        # Note: beta_qqq_simple is univariate cov/var, distinct from beta_qqq (joint factor model)
        result['beta_qqq_simple'] = beta_qqq_val
        result['alpha_resid_qqq'] = alpha_qqq

        # Create momentum features from QQQ alpha
        qqq_feats = _alpha_momentum_from_residual(
            alpha_qqq, windows=windows, ema_span=ema_span, prefix="alpha_mom_qqq"
        )
        result.update(qqq_feats)

    # Sector alpha (vs sector ETF)
    if symbol_sector_etf and symbol_sector_etf in benchmarks:
        sec_idx, sec_vals = benchmarks[symbol_sector_etf]
        sec_ret = pd.Series(sec_vals, index=sec_idx, dtype='float64').reindex(ret_index)
        beta_sec, alpha_sec = _rolling_beta_alpha(r, sec_ret, win=beta_win)
        result['beta_sector'] = beta_sec
        result['alpha_resid_sector'] = alpha_sec

        # Create momentum features from sector alpha
        sec_feats = _alpha_momentum_from_residual(
            alpha_sec, windows=windows, ema_span=ema_span, prefix="alpha_mom_sector"
        )
        result.update(sec_feats)

    # Blended alpha features (if both market and sector exist)
    if alpha_mkt is not None and alpha_sec is not None:
        alpha_combo = 0.5 * alpha_mkt + 0.5 * alpha_sec
        combo_feats = _alpha_momentum_from_residual(
            alpha_combo, windows=windows, ema_span=ema_span, prefix="alpha_mom_combo"
        )
        result.update(combo_feats)

    # SPY-QQQ spread alpha (captures value vs growth positioning)
    # This is the difference in alpha residuals - positive means stock outperforms
    # its expected QQQ-based return MORE than its expected SPY-based return
    if alpha_mkt is not None and alpha_qqq is not None:
        # Difference shows growth vs broad market tilt in alpha generation
        alpha_qqq_vs_spy = alpha_qqq - alpha_mkt
        result['alpha_qqq_vs_spy'] = alpha_qqq_vs_spy.astype('float32')

        # Momentum of the spread
        spread_feats = _alpha_momentum_from_residual(
            alpha_qqq_vs_spy, windows=windows, ema_span=ema_span, prefix="alpha_mom_qqq_spread"
        )
        result.update(spread_feats)

    return result


def add_alpha_momentum_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    beta_win: int = 60,
    windows: Tuple[int, ...] = (20, 60, 120),
    ema_span: int = 10,
    n_jobs: int = -1
) -> None:
    """
    Add alpha momentum features based on CAPM residuals vs market, QQQ, and sector benchmarks.

    This function is parallelized across symbols using joblib.

    Features added (if benchmarks available):
    - beta_spy_simple, alpha_resid_spy: Univariate CAPM beta/alpha vs market (cov/var method)
    - alpha_mom_spy_ema{ema_span}: EMA of market alpha residuals
    - alpha_mom_spy_{w}_ema{ema_span}: EMA of {w}-day cumulative market alpha
    - beta_qqq_simple, alpha_resid_qqq: Univariate CAPM beta/alpha vs QQQ (cov/var method)
    - alpha_mom_qqq_ema{ema_span}: EMA of QQQ alpha residuals
    - alpha_mom_qqq_{w}_ema{ema_span}: EMA of {w}-day cumulative QQQ alpha
    - alpha_qqq_vs_spy: Difference in alpha (QQQ - SPY) capturing growth vs value tilt
    - alpha_mom_qqq_spread_*: Momentum of the QQQ-SPY alpha spread
    - beta_sector, alpha_resid_sector: CAPM beta/alpha residuals vs sector
    - alpha_mom_sector_ema{ema_span}: EMA of sector alpha residuals
    - alpha_mom_sector_{w}_ema{ema_span}: EMA of {w}-day cumulative sector alpha
    - alpha_mom_combo_*: Blended features (50/50 market/sector) if both exist

    Note: beta_spy_simple and beta_qqq_simple use simple rolling cov/var, distinct from
    beta_market and beta_qqq in factor_regression.py which use orthogonalized joint regression.

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        sectors: Optional mapping of symbol -> sector name
        sector_to_etf: Optional mapping of lowercase sector name -> ETF symbol
        market_symbol: Symbol to use as market benchmark (default: SPY)
        qqq_symbol: Symbol to use as growth/liquidity benchmark (default: QQQ)
        beta_win: Rolling window for beta/alpha calculation
        windows: Momentum calculation windows
        ema_span: EMA span for smoothing momentum features
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    logger.info(f"Computing alpha momentum features using {market_symbol} and {qqq_symbol} as benchmarks (parallel)")

    # Pre-build sector ETF mapping (lowercase)
    sec_map_lc = {k.lower(): v for k, v in (sector_to_etf or {}).items()}

    # =========================================================================
    # Step 1: Pre-extract all benchmark return series as (index, numpy_array) tuples
    # =========================================================================
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # Market benchmark (SPY)
    mkt_df = indicators_by_symbol.get(market_symbol)
    if mkt_df is not None and 'ret' in mkt_df.columns:
        mkt_ret = pd.to_numeric(mkt_df['ret'], errors='coerce')
        benchmarks[market_symbol] = (mkt_ret.index, mkt_ret.values)
        logger.debug(f"Loaded {market_symbol} returns: {len(mkt_ret)} data points")
    else:
        logger.warning(f"{market_symbol} missing or has no 'ret' column")

    # QQQ benchmark (growth/liquidity factor)
    qqq_df = indicators_by_symbol.get(qqq_symbol)
    if qqq_df is not None and 'ret' in qqq_df.columns:
        qqq_ret = pd.to_numeric(qqq_df['ret'], errors='coerce')
        benchmarks[qqq_symbol] = (qqq_ret.index, qqq_ret.values)
        logger.debug(f"Loaded {qqq_symbol} returns: {len(qqq_ret)} data points")
    else:
        logger.warning(f"{qqq_symbol} missing or has no 'ret' column - QQQ features will be skipped")

    # Sector ETF returns
    if sectors and sector_to_etf:
        needed_etfs = set()
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sec_map_lc.get(sec.lower())
                if etf:
                    needed_etfs.add(etf)

        for etf in needed_etfs:
            if etf in benchmarks:
                continue
            df_etf = indicators_by_symbol.get(etf)
            if df_etf is not None and 'ret' in df_etf.columns:
                etf_ret = pd.to_numeric(df_etf['ret'], errors='coerce')
                benchmarks[etf] = (etf_ret.index, etf_ret.values)

        logger.debug(f"Loaded {len(benchmarks) - 1} sector benchmark returns")

    # =========================================================================
    # Step 2: Build per-symbol sector ETF mapping
    # =========================================================================
    symbol_sector_etfs: Dict[str, Optional[str]] = {}
    for sym in indicators_by_symbol:
        if sectors and sector_to_etf:
            sec = sectors.get(sym)
            if isinstance(sec, str):
                symbol_sector_etfs[sym] = sec_map_lc.get(sec.lower())
            else:
                symbol_sector_etfs[sym] = None
        else:
            symbol_sector_etfs[sym] = None

    # =========================================================================
    # Step 3: Prepare work items
    # =========================================================================
    work_items = []
    for sym, df in indicators_by_symbol.items():
        if 'ret' not in df.columns or 'adjclose' not in df.columns:
            continue
        # Skip benchmark symbols (SPY and QQQ)
        if sym == market_symbol or sym == qqq_symbol:
            continue

        r = pd.to_numeric(df['ret'], errors='coerce')
        work_items.append((sym, r.index, r.values, symbol_sector_etfs.get(sym)))

    n_symbols = len(work_items)

    # =========================================================================
    # Step 4: Parallel computation with batching
    # =========================================================================
    if n_jobs == -1:
        import multiprocessing
        effective_jobs = multiprocessing.cpu_count()
    else:
        effective_jobs = n_jobs

    # Calculate optimal chunk size: minimum 100 symbols per chunk to amortize IPC overhead
    min_chunk_size = 100
    chunk_size = max(min_chunk_size, n_symbols // effective_jobs)
    n_chunks = max(1, n_symbols // chunk_size)
    actual_workers = min(effective_jobs, n_chunks)

    logger.info(f"Computing alpha features for {n_symbols} symbols in {n_chunks} batches "
                f"(~{chunk_size} symbols/batch, {actual_workers} workers)")

    # Use sequential processing for small datasets
    if n_symbols < 10 or actual_workers == 1:
        logger.info("Using sequential processing for alpha computation")
        results = []
        for sym, ret_index, ret_values, sector_etf in work_items:
            alpha_features = _compute_alpha_for_symbol(
                sym, ret_index, ret_values, benchmarks, sector_etf,
                market_symbol, beta_win, windows, ema_span, qqq_symbol
            )
            results.append((sym, alpha_features))
    else:
        # Create batches for parallel processing with subsetted benchmarks
        work_chunks = _chunk_list(work_items, chunk_size)

        # Pre-compute subsetted benchmarks for each chunk to reduce serialization
        # Each worker only gets the benchmarks it needs (core + batch-specific sector ETFs)
        # See Section 4 of FEATURE_PIPELINE_ARCHITECTURE.md for parallelism best practices
        chunk_benchmarks = []
        for chunk in work_chunks:
            benchmark_subset = _get_required_benchmarks_for_alpha_batch(
                chunk, benchmarks, market_symbol, qqq_symbol
            )
            chunk_benchmarks.append(benchmark_subset)

        # Log benchmark reduction
        total_benchmarks = len(benchmarks)
        avg_benchmarks = sum(len(b) for b in chunk_benchmarks) / len(chunk_benchmarks) if chunk_benchmarks else 0
        logger.info(f"Benchmark subsetting: {total_benchmarks} total -> {avg_benchmarks:.1f} avg per batch")

        try:
            batch_results = Parallel(
                n_jobs=actual_workers,
                backend='loky',
                verbose=0,
                prefer='processes'
            )(
                delayed(_compute_alpha_batch)(
                    chunk, chunk_benchmarks[i], market_symbol, beta_win, windows, ema_span, qqq_symbol
                )
                for i, chunk in enumerate(work_chunks)
            )

            # Flatten batch results
            results = []
            for batch in batch_results:
                results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel alpha processing failed ({e}), falling back to sequential")
            results = []
            for sym, ret_index, ret_values, sector_etf in work_items:
                alpha_features = _compute_alpha_for_symbol(
                    sym, ret_index, ret_values, benchmarks, sector_etf,
                    market_symbol, beta_win, windows, ema_span, qqq_symbol
                )
                results.append((sym, alpha_features))

    # =========================================================================
    # Step 5: Apply results back to DataFrames
    # =========================================================================
    added_count = 0
    for sym, alpha_features in results:
        if not alpha_features:
            continue

        df = indicators_by_symbol[sym]
        # Use pd.concat instead of repeated column insertion to avoid fragmentation
        new_cols_df = pd.DataFrame(alpha_features, index=df.index)
        indicators_by_symbol[sym] = pd.concat([df, new_cols_df], axis=1)
        added_count += 1

    logger.info(f"Alpha momentum features added for {added_count} symbols "
                f"(beta window={beta_win}, horizons={list(windows)}, EMA={ema_span})")