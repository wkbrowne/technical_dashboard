"""
Cross-sectional feature computation requiring data from multiple stocks.

This module provides a centralized function for computing all features that require
comparing a stock against other stocks in the universe (cross-sectional features).

Parallelization Strategy:
- All per-symbol cross-sectional features (alpha, relative strength) are computed
  together in a single parallel job to maximize CPU utilization
- Benchmarks are pre-extracted once and shared across workers
- Features that require the full universe (breadth, xsec momentum) run sequentially
"""
import logging
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Feature module imports with fallback for both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from .volatility import add_vol_regime_cs_context
    from .alpha import (
        _compute_alpha_for_symbol,
        _rolling_beta_alpha,
        _alpha_momentum_from_residual
    )
    from .relstrength import (
        _compute_rs_for_symbol,
        _compute_relative_strength_block
    )
    from .breadth import add_breadth_series
    from .xsec import (
        add_xsec_momentum_panel,
        add_weekly_xsec_momentum_panel,
        add_xsec_percentile_rank,
        add_weekly_xsec_percentile_rank
    )
    from .macro import add_vix_regime_features, add_volatility_term_structure, add_weekly_vix_features
    from .cross_asset import add_cross_asset_features
    from .factor_regression import (
        add_joint_factor_features,
        compute_best_match_mappings,
        compute_best_match_ew_mappings,
        FactorRegressionConfig,
        BEST_MATCH_CANDIDATES,
        BEST_MATCH_EW_CANDIDATES
    )
    from ..data.fred import add_fred_features, download_fred_series
    from ..config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER
except ImportError:
    # Fallback to absolute imports (when run directly)
    from src.features.volatility import add_vol_regime_cs_context
    from src.features.alpha import (
        _compute_alpha_for_symbol,
        _rolling_beta_alpha,
        _alpha_momentum_from_residual
    )
    from src.features.relstrength import (
        _compute_rs_for_symbol,
        _compute_relative_strength_block
    )
    from src.features.breadth import add_breadth_series
    from src.features.xsec import (
        add_xsec_momentum_panel,
        add_weekly_xsec_momentum_panel,
        add_xsec_percentile_rank,
        add_weekly_xsec_percentile_rank
    )
    from src.features.macro import add_vix_regime_features, add_volatility_term_structure, add_weekly_vix_features
    from src.features.cross_asset import add_cross_asset_features
    from src.features.factor_regression import (
        add_joint_factor_features,
        compute_best_match_mappings,
        compute_best_match_ew_mappings,
        FactorRegressionConfig,
        BEST_MATCH_CANDIDATES,
        BEST_MATCH_EW_CANDIDATES
    )
    from src.data.fred import add_fred_features, download_fred_series
    from src.config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER

logger = logging.getLogger(__name__)


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _get_required_benchmarks_for_batch(
    work_items_batch: List[Tuple],
    all_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ"
) -> Dict[str, Tuple[pd.Index, np.ndarray]]:
    """
    Extract only the benchmarks needed by a specific batch of symbols.

    This reduces memory usage by only sending relevant benchmark data to each worker
    instead of the entire benchmark dictionary.

    Args:
        work_items_batch: List of work item tuples for this batch
        all_benchmarks: Complete benchmark dictionary
        market_symbol: Market benchmark symbol (always included)
        qqq_symbol: Growth benchmark symbol (always included)

    Returns:
        Subset of benchmarks needed by this batch
    """
    # Core benchmarks always needed
    required = {market_symbol, qqq_symbol, 'RSP'}

    # Add best-match ETFs referenced by symbols in this batch
    for work_item in work_items_batch:
        # work_item structure: (sym, ret_index, ret_values, px_index, px_values, sector_etf, rs_mapping)
        if len(work_item) >= 6 and work_item[5]:  # sector_etf
            required.add(work_item[5])
        if len(work_item) >= 7 and work_item[6]:  # rs_mapping dict
            rs_mapping = work_item[6]
            if isinstance(rs_mapping, dict):
                if rs_mapping.get('sector_etf'):
                    required.add(rs_mapping['sector_etf'])
                if rs_mapping.get('equal_weight_etf'):
                    required.add(rs_mapping['equal_weight_etf'])

    # Extract only required benchmarks
    subset = {}
    for key in required:
        if key in all_benchmarks:
            subset[key] = all_benchmarks[key]

    return subset


def _compute_all_per_symbol_features(
    sym: str,
    ret_index: pd.Index,
    ret_values: np.ndarray,
    px_index: pd.Index,
    px_values: np.ndarray,
    alpha_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    rs_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    symbol_sector_etf: Optional[str],
    symbol_rs_mapping: Dict[str, Any],
    market_symbol: str,
    beta_win: int,
    alpha_windows: Tuple[int, ...],
    alpha_ema_span: int,
    qqq_symbol: str = "QQQ"
) -> Dict[str, pd.Series]:
    """
    Compute ALL per-symbol cross-sectional features in one call.

    This combines alpha momentum and relative strength computation to allow
    a single parallel job to compute all features for each symbol batch.

    Args:
        sym: Symbol ticker
        ret_index: DatetimeIndex for return data
        ret_values: Numpy array of returns
        px_index: DatetimeIndex for price data
        px_values: Numpy array of adjusted close prices
        alpha_benchmarks: Dict mapping benchmark name -> (index, values) for alpha
        rs_benchmarks: Dict mapping benchmark name -> (index, values) for RS
        symbol_sector_etf: ETF symbol for this stock's sector (for alpha)
        symbol_rs_mapping: Dict with sector_etf, subsector_etf, equal_weight_etf keys
        market_symbol: Symbol used as market benchmark
        beta_win: Rolling window for beta/alpha calculation
        alpha_windows: Windows for alpha-momentum computation
        alpha_ema_span: EMA span for alpha smoothing
        qqq_symbol: Symbol used as QQQ benchmark (growth/liquidity factor)

    Returns:
        Dict mapping column_name -> pd.Series with all computed features
    """
    result = {}

    # Skip benchmark symbols
    if sym == market_symbol or sym == qqq_symbol:
        return result

    # 1) Compute alpha momentum features
    alpha_features = _compute_alpha_for_symbol(
        sym, ret_index, ret_values, alpha_benchmarks, symbol_sector_etf,
        market_symbol, beta_win, alpha_windows, alpha_ema_span, qqq_symbol
    )
    result.update(alpha_features)

    # 2) Compute relative strength features
    rs_features = _compute_rs_for_symbol(
        sym, px_index, px_values, rs_benchmarks, symbol_rs_mapping, market_symbol, qqq_symbol
    )
    result.update(rs_features)

    return result


def _compute_per_symbol_batch(
    work_items_batch: List[Tuple],
    alpha_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    rs_benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]],
    market_symbol: str,
    beta_win: int,
    alpha_windows: Tuple[int, ...],
    alpha_ema_span: int,
    qqq_symbol: str = "QQQ"
) -> List[Tuple[str, Dict[str, pd.Series]]]:
    """
    Process a batch of symbols computing ALL per-symbol CS features.

    Args:
        work_items_batch: List of tuples containing per-symbol data
        alpha_benchmarks: Benchmark data for alpha computation
        rs_benchmarks: Benchmark data for RS computation
        market_symbol: Market benchmark symbol
        beta_win: Beta calculation window
        alpha_windows: Alpha momentum windows
        alpha_ema_span: Alpha EMA span
        qqq_symbol: QQQ benchmark symbol (growth/liquidity factor)

    Returns:
        List of (symbol, features_dict) tuples
    """
    results = []
    for work_item in work_items_batch:
        sym, ret_index, ret_values, px_index, px_values, sector_etf, rs_mapping = work_item
        features = _compute_all_per_symbol_features(
            sym, ret_index, ret_values, px_index, px_values,
            alpha_benchmarks, rs_benchmarks, sector_etf, rs_mapping,
            market_symbol, beta_win, alpha_windows, alpha_ema_span, qqq_symbol
        )
        results.append((sym, features))
    return results


def _prepare_alpha_benchmarks(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    best_match_mappings: Dict[str, str],
    market_symbol: str,
    qqq_symbol: str = "QQQ"
) -> Tuple[Dict[str, Tuple[pd.Index, np.ndarray]], Dict[str, Optional[str]]]:
    """
    Prepare benchmarks and per-symbol mappings for alpha computation.

    Uses best-match ETF (sector or subsector, whichever has higher R²) instead
    of static sector mappings.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame
        best_match_mappings: Dict of symbol -> best-match ETF symbol
        market_symbol: Market benchmark symbol (e.g., 'SPY')
        qqq_symbol: Growth/liquidity benchmark symbol (e.g., 'QQQ')

    Returns:
        Tuple of (benchmarks dict, symbol_best_match_etfs dict)
    """
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # Market benchmark (SPY)
    mkt_df = indicators_by_symbol.get(market_symbol)
    if mkt_df is not None and 'ret' in mkt_df.columns:
        mkt_ret = pd.to_numeric(mkt_df['ret'], errors='coerce')
        benchmarks[market_symbol] = (mkt_ret.index, mkt_ret.values)

    # QQQ benchmark (growth/liquidity factor)
    qqq_df = indicators_by_symbol.get(qqq_symbol)
    if qqq_df is not None and 'ret' in qqq_df.columns:
        qqq_ret = pd.to_numeric(qqq_df['ret'], errors='coerce')
        benchmarks[qqq_symbol] = (qqq_ret.index, qqq_ret.values)
        logger.debug(f"Loaded {qqq_symbol} returns for alpha computation")
    else:
        logger.warning(f"{qqq_symbol} missing or has no 'ret' column - QQQ alpha features will be skipped")

    # Load all best-match ETF returns
    needed_etfs = set(best_match_mappings.values())
    for etf in needed_etfs:
        if etf in benchmarks:
            continue
        df_etf = indicators_by_symbol.get(etf)
        if df_etf is not None and 'ret' in df_etf.columns:
            etf_ret = pd.to_numeric(df_etf['ret'], errors='coerce')
            benchmarks[etf] = (etf_ret.index, etf_ret.values)

    # Symbol -> best-match ETF mapping (replaces sector_etf)
    symbol_best_match_etfs: Dict[str, Optional[str]] = {}
    for sym in indicators_by_symbol:
        symbol_best_match_etfs[sym] = best_match_mappings.get(sym)

    return benchmarks, symbol_best_match_etfs


def _prepare_rs_benchmarks(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    best_match_mappings: Dict[str, str],
    best_match_ew_mappings: Dict[str, str],
    market_symbol: str,
    qqq_symbol: str = "QQQ"
) -> Tuple[Dict[str, Tuple[pd.Index, np.ndarray]], Dict[str, Dict[str, Any]]]:
    """
    Prepare benchmarks and per-symbol mappings for RS computation.

    Uses best-match ETF (sector or subsector, whichever has higher R²) as the
    sector reference. Also uses equal-weight best-match for EW RS features.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame
        best_match_mappings: Dict of symbol -> best-match cap-weighted ETF symbol
        best_match_ew_mappings: Dict of symbol -> best-match equal-weight ETF symbol
        market_symbol: Market benchmark symbol (e.g., 'SPY')
        qqq_symbol: Growth/liquidity benchmark symbol (e.g., 'QQQ')

    Returns:
        Tuple of (benchmarks dict, symbol_mappings dict)
    """
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # SPY benchmark
    spy_data = indicators_by_symbol.get(market_symbol)
    if spy_data is not None and "adjclose" in spy_data.columns:
        spy_px = pd.to_numeric(spy_data["adjclose"], errors='coerce')
        benchmarks['SPY'] = (spy_px.index, spy_px.values)

    # RSP benchmark (equal-weight market - kept as breadth factor)
    rsp_data = indicators_by_symbol.get('RSP')
    if rsp_data is not None and "adjclose" in rsp_data.columns:
        rsp_px = pd.to_numeric(rsp_data["adjclose"], errors='coerce')
        benchmarks['RSP'] = (rsp_px.index, rsp_px.values)

    # QQQ benchmark (growth/liquidity factor)
    qqq_data = indicators_by_symbol.get(qqq_symbol)
    if qqq_data is not None and "adjclose" in qqq_data.columns:
        qqq_px = pd.to_numeric(qqq_data["adjclose"], errors='coerce')
        benchmarks['QQQ'] = (qqq_px.index, qqq_px.values)
        logger.debug(f"Loaded {qqq_symbol} prices for RS computation")
    else:
        logger.warning(f"{qqq_symbol} missing or has no 'adjclose' column - QQQ RS features will be skipped")

    # Load all best-match ETF benchmarks (cap-weighted)
    for etf in set(best_match_mappings.values()):
        if etf in benchmarks:
            continue
        etf_df = indicators_by_symbol.get(etf)
        if etf_df is not None and "adjclose" in etf_df.columns:
            etf_px = pd.to_numeric(etf_df["adjclose"], errors='coerce')
            benchmarks[etf] = (etf_px.index, etf_px.values)

    # Load all best-match EW ETF benchmarks (equal-weight)
    for etf in set(best_match_ew_mappings.values()):
        if etf in benchmarks:
            continue
        etf_df = indicators_by_symbol.get(etf)
        if etf_df is not None and "adjclose" in etf_df.columns:
            etf_px = pd.to_numeric(etf_df["adjclose"], errors='coerce')
            benchmarks[etf] = (etf_px.index, etf_px.values)

    # Build per-symbol mapping info with best-match as sector_etf
    # and best-match EW as equal_weight_etf (subsector_etf is removed)
    symbol_mappings: Dict[str, Dict[str, Any]] = {}

    for sym, df in indicators_by_symbol.items():
        if "adjclose" not in df.columns:
            continue

        mapping = {}
        best_match = best_match_mappings.get(sym)
        if best_match:
            # Best-match becomes the sector reference (could be sector OR subsector ETF)
            mapping['sector_etf'] = best_match

        best_match_ew = best_match_ew_mappings.get(sym)
        if best_match_ew:
            # Best-match EW becomes the equal-weight reference
            mapping['equal_weight_etf'] = best_match_ew

        symbol_mappings[sym] = mapping

    return benchmarks, symbol_mappings


def compute_per_symbol_cs_features_parallel(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    best_match_mappings: Dict[str, str],
    best_match_ew_mappings: Dict[str, str],
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    beta_win: int = 60,
    alpha_windows: Tuple[int, ...] = (20, 60, 120),
    alpha_ema_span: int = 10,
    n_jobs: int = -1
) -> None:
    """
    Compute all per-symbol cross-sectional features in a single parallel job.

    This function combines alpha momentum and relative strength features to
    maximize CPU utilization by computing all features for each symbol batch
    in a single parallel task.

    Uses best-match ETF (sector or subsector with highest R²) as the sector
    reference for all computations. Also uses equal-weight best-match for EW features.

    Features computed:
    - Alpha momentum (alpha_resid_spy, alpha_mom_spy_*, alpha_resid_qqq, alpha_mom_qqq_*, etc.)
    - Relative strength (rel_strength_spy, rel_strength_qqq, rel_strength_sector, rel_strength_sector_ew, etc.)

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame (modified in place)
        best_match_mappings: Dict of symbol -> best-match cap-weighted ETF symbol
        best_match_ew_mappings: Dict of symbol -> best-match equal-weight ETF symbol
        market_symbol: Symbol to use as market benchmark (default: SPY)
        qqq_symbol: Symbol to use as growth/liquidity benchmark (default: QQQ)
        beta_win: Window for beta calculation in alpha features
        alpha_windows: Windows for alpha-momentum computation
        alpha_ema_span: EMA span for alpha smoothing
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    logger.info(f"Computing per-symbol cross-sectional features using {market_symbol} and {qqq_symbol} (unified parallel)")

    # Prepare all benchmarks upfront using best-match mappings
    logger.info("Preparing benchmarks for alpha and RS computation...")
    alpha_benchmarks, symbol_best_match_etfs = _prepare_alpha_benchmarks(
        indicators_by_symbol, best_match_mappings, market_symbol, qqq_symbol
    )
    rs_benchmarks, symbol_rs_mappings = _prepare_rs_benchmarks(
        indicators_by_symbol, best_match_mappings, best_match_ew_mappings, market_symbol, qqq_symbol
    )

    logger.info(f"Loaded {len(alpha_benchmarks)} alpha benchmarks, {len(rs_benchmarks)} RS benchmarks")

    # Build work items with all data needed for each symbol
    work_items = []
    for sym, df in indicators_by_symbol.items():
        # Skip benchmarks and symbols missing required columns
        if sym == market_symbol or sym == qqq_symbol:
            continue
        if 'ret' not in df.columns or 'adjclose' not in df.columns:
            continue
        if sym in alpha_benchmarks or sym in rs_benchmarks:
            # Skip ETFs that are used as benchmarks
            continue

        ret = pd.to_numeric(df['ret'], errors='coerce')
        px = pd.to_numeric(df['adjclose'], errors='coerce')

        work_items.append((
            sym,
            ret.index,
            ret.values,
            px.index,
            px.values,
            symbol_best_match_etfs.get(sym),
            symbol_rs_mappings.get(sym, {})
        ))

    n_symbols = len(work_items)

    # Calculate workers based on stocks_per_worker (100 stocks/worker by default)
    chunk_size = DEFAULT_STOCKS_PER_WORKER
    n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)
    n_chunks = max(1, (n_symbols + chunk_size - 1) // chunk_size)

    logger.info(f"Processing {n_symbols} symbols in {n_chunks} batches "
                f"({chunk_size} stocks/worker, {n_workers} workers)")

    # Process sequentially for small datasets
    if n_symbols < 10 or n_workers == 1:
        logger.info("Using sequential processing for per-symbol CS features")
        results = []
        for work_item in work_items:
            sym, ret_index, ret_values, px_index, px_values, sector_etf, rs_mapping = work_item
            features = _compute_all_per_symbol_features(
                sym, ret_index, ret_values, px_index, px_values,
                alpha_benchmarks, rs_benchmarks, sector_etf, rs_mapping,
                market_symbol, beta_win, alpha_windows, alpha_ema_span, qqq_symbol
            )
            results.append((sym, features))
    else:
        # Batch parallel processing with subsetted benchmarks
        work_chunks = _chunk_list(work_items, chunk_size)

        # Pre-compute subsetted benchmarks for each chunk to reduce serialization
        # Each worker only gets the benchmarks it needs (core + batch-specific ETFs)
        chunk_alpha_benchmarks = []
        chunk_rs_benchmarks = []
        for chunk in work_chunks:
            # Get required benchmarks for this batch
            alpha_subset = _get_required_benchmarks_for_batch(
                chunk, alpha_benchmarks, market_symbol, qqq_symbol
            )
            rs_subset = _get_required_benchmarks_for_batch(
                chunk, rs_benchmarks, market_symbol, qqq_symbol
            )
            chunk_alpha_benchmarks.append(alpha_subset)
            chunk_rs_benchmarks.append(rs_subset)

        # Log benchmark reduction
        total_alpha = len(alpha_benchmarks)
        avg_alpha = sum(len(b) for b in chunk_alpha_benchmarks) / len(chunk_alpha_benchmarks) if chunk_alpha_benchmarks else 0
        logger.info(f"Benchmark subsetting: {total_alpha} total -> {avg_alpha:.1f} avg per batch")

        try:
            batch_results = Parallel(
                n_jobs=n_workers,
                backend='loky',
                verbose=0,
                prefer='processes'
            )(
                delayed(_compute_per_symbol_batch)(
                    chunk, chunk_alpha_benchmarks[i], chunk_rs_benchmarks[i],
                    market_symbol, beta_win, alpha_windows, alpha_ema_span, qqq_symbol
                )
                for i, chunk in enumerate(work_chunks)
            )

            # Flatten results
            results = []
            for batch in batch_results:
                results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel processing failed ({e}), falling back to sequential")
            results = []
            for work_item in work_items:
                sym, ret_index, ret_values, px_index, px_values, sector_etf, rs_mapping = work_item
                features = _compute_all_per_symbol_features(
                    sym, ret_index, ret_values, px_index, px_values,
                    alpha_benchmarks, rs_benchmarks, sector_etf, rs_mapping,
                    market_symbol, beta_win, alpha_windows, alpha_ema_span, qqq_symbol
                )
                results.append((sym, features))

    # Apply results to DataFrames
    alpha_count = 0
    rs_count = 0
    for sym, features in results:
        if not features:
            continue

        df = indicators_by_symbol[sym]
        for col_name, series in features.items():
            df[col_name] = series

        # Count features
        if any('alpha' in col for col in features.keys()):
            alpha_count += 1
        if any('rel_strength' in col for col in features.keys()):
            rs_count += 1

    logger.info(f"Per-symbol CS features completed: alpha added to {alpha_count} symbols, "
                f"RS added to {rs_count} symbols")


def compute_cross_sectional_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    sp500_tickers: Optional[List[str]] = None,
    beta_win: int = 60,
    alpha_windows: tuple = (20, 60, 120),
    alpha_ema_span: int = 10,
    xsec_lookbacks: tuple = (5, 20, 60),
    xsec_weekly_lookbacks: tuple = (1, 4, 13),
    price_col: str = "adjclose",
    vix_symbol: str = "^VIX",
    vxn_symbol: str = "^VXN",
    n_jobs: int = -1
) -> None:
    """
    Compute all cross-sectional features that require data from multiple stocks.

    This function modifies the indicators_by_symbol dictionary in place, adding
    cross-sectional features to each symbol's DataFrame.

    Uses R²-based best-match ETF selection (sector or subsector, whichever explains
    more variance) instead of static sector mappings. This replaces the broken
    subsector features that had 40-48% NaN rates.

    Features computed:
    1. Best-match ETF mappings (R²-based selection from sector + subsector candidates)
    2. Cross-sectional volatility regime context (vol_regime_cs_median, vol_regime_rel)
    3. Alpha-momentum features (beta-adjusted returns vs SPY, QQQ, best-match) [PARALLEL]
    4. Relative strength features (vs SPY, QQQ, best-match) [PARALLEL]
    5. Joint factor regression betas (market, QQQ, best-match, breadth) [PARALLEL]
    6. Market breadth features (advancing/declining stocks, new highs/lows)
    7. Cross-sectional momentum z-scores (xsec_mom_*d_z)
    8. Cross-sectional percentile ranks (xsec_pct_*d)
    9. Weekly cross-sectional momentum z-scores (w_xsec_mom_*w_z)
    10. Weekly cross-sectional percentile ranks (w_xsec_pct_*w)
    11. VIX regime features (vix_level, vix_percentile, vix_regime, etc.)
    12. Volatility term structure features from VXX/VIXY
    13. Weekly VIX regime features (w_vix_level, w_vix_percentile_52w, etc.)
    14. Cross-asset signal features (macro regime indicators)
    15. FRED macro features (treasury yields, credit spreads, etc.)

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame with features
                             (modified in place)
        sectors: Dictionary mapping symbol -> sector name (optional, used for xsec)
        sector_to_etf: Dictionary mapping sector name -> ETF symbol (optional, legacy)
        market_symbol: Symbol to use as market benchmark (default: "SPY")
        qqq_symbol: Symbol to use as growth/liquidity benchmark (default: "QQQ")
        sp500_tickers: List of S&P 500 tickers for breadth calculation (optional)
        beta_win: Window for beta calculation in alpha features (default: 60)
        alpha_windows: Windows for alpha-momentum computation (default: (20, 60, 120))
        alpha_ema_span: EMA span for alpha smoothing (default: 10)
        xsec_lookbacks: Lookback periods for cross-sectional momentum (default: (5, 20, 60))
        xsec_weekly_lookbacks: Lookback periods for weekly cross-sectional (default: (1, 4, 13))
        price_col: Column name for price data (default: "adjclose")
        vix_symbol: Symbol for VIX index (default: "^VIX")
        vxn_symbol: Symbol for VXN index (default: "^VXN")
        n_jobs: Number of parallel jobs for per-symbol features (default: -1)

    Returns:
        None (modifies indicators_by_symbol in place)
    """
    logger.info("Computing cross-sectional features for all symbols")

    n_symbols = len(indicators_by_symbol)
    logger.info(f"Processing {n_symbols} symbols")

    # 1) Compute best-match ETF mappings (R²-based, sector or subsector)
    # This replaces the broken subsector discovery with stable behavioral matching
    logger.info("Computing best-match ETF mappings (R²-based)...")
    best_match_mappings = compute_best_match_mappings(
        indicators_by_symbol,
        candidates=BEST_MATCH_CANDIDATES,
        window=252,  # 1 year lookback for stable matching
        min_overlap=100
    )
    logger.info(f"  ✓ Best-match (cap-weighted) mappings computed for {len(best_match_mappings)} symbols")

    # 1b) Compute best-match equal-weight ETF mappings
    logger.info("Computing best-match equal-weight ETF mappings (R²-based)...")
    best_match_ew_mappings = compute_best_match_ew_mappings(
        indicators_by_symbol,
        candidates=BEST_MATCH_EW_CANDIDATES,
        window=252,
        min_overlap=100
    )
    logger.info(f"  ✓ Best-match (equal-weight) mappings computed for {len(best_match_ew_mappings)} symbols")

    # 2) Cross-sectional volatility regime context
    logger.info("Adding cross-sectional volatility regime context...")
    add_vol_regime_cs_context(indicators_by_symbol)
    logger.debug("  ✓ Volatility context completed")

    # 3-4) Alpha-momentum AND Relative strength features (UNIFIED PARALLEL)
    # Uses best-match ETF instead of static sector mappings
    logger.info("Computing per-symbol CS features (alpha + RS) using best-match...")
    compute_per_symbol_cs_features_parallel(
        indicators_by_symbol,
        best_match_mappings=best_match_mappings,
        best_match_ew_mappings=best_match_ew_mappings,
        market_symbol=market_symbol,
        qqq_symbol=qqq_symbol,
        beta_win=beta_win,
        alpha_windows=alpha_windows,
        alpha_ema_span=alpha_ema_span,
        n_jobs=n_jobs
    )
    logger.debug("  ✓ Per-symbol CS features (alpha + RS) completed")

    # 5) Joint factor regression betas (market, QQQ, best-match, breadth)
    # Computes all betas jointly via multivariate ridge regression for cleaner estimates
    logger.info("Computing joint factor regression features (daily)...")
    factor_config = FactorRegressionConfig()
    add_joint_factor_features(
        indicators_by_symbol,
        best_match_mappings=best_match_mappings,
        config=factor_config,
        frequency='daily',
        n_jobs=n_jobs
    )
    logger.debug("  ✓ Joint factor features (daily) completed")

    # 5b) Joint factor regression betas (weekly)
    # Same factors but computed on weekly returns for smoother signals
    logger.info("Computing joint factor regression features (weekly)...")
    add_joint_factor_features(
        indicators_by_symbol,
        best_match_mappings=best_match_mappings,
        config=factor_config,
        frequency='weekly',
        n_jobs=n_jobs
    )
    logger.debug("  ✓ Joint factor features (weekly) completed")

    # 4) Breadth features (requires SP500 ticker list)
    if sp500_tickers:
        logger.info("Adding market breadth features...")
        add_breadth_series(indicators_by_symbol, sp500_tickers)
        logger.debug(f"  ✓ Breadth features completed ({len(sp500_tickers)} tickers)")
    else:
        logger.warning("Skipping breadth features (sp500_tickers not provided)")

    # 5) Cross-sectional momentum z-scores (daily)
    logger.info("Adding daily cross-sectional momentum z-scores...")
    add_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=xsec_lookbacks,
        price_col=price_col,
        sector_map=sectors
    )
    logger.debug(f"  ✓ Daily cross-sectional momentum z-scores completed (lookbacks={xsec_lookbacks})")

    # 6) Cross-sectional percentile ranks (daily)
    logger.info("Adding daily cross-sectional percentile ranks...")
    add_xsec_percentile_rank(
        indicators_by_symbol,
        lookbacks=xsec_lookbacks,
        price_col=price_col,
        sector_map=sectors
    )
    logger.debug(f"  ✓ Daily cross-sectional percentile ranks completed (lookbacks={xsec_lookbacks})")

    # 7) Weekly cross-sectional momentum z-scores
    logger.info("Adding weekly cross-sectional momentum z-scores...")
    add_weekly_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=xsec_weekly_lookbacks,
        price_col=price_col,
        sector_map=sectors
    )
    logger.debug(f"  ✓ Weekly cross-sectional momentum z-scores completed (lookbacks={xsec_weekly_lookbacks} weeks)")

    # 8) Weekly cross-sectional percentile ranks
    logger.info("Adding weekly cross-sectional percentile ranks...")
    add_weekly_xsec_percentile_rank(
        indicators_by_symbol,
        lookbacks=xsec_weekly_lookbacks,
        price_col=price_col,
        sector_map=sectors
    )
    logger.debug(f"  ✓ Weekly cross-sectional percentile ranks completed (lookbacks={xsec_weekly_lookbacks} weeks)")

    # 9) VIX regime features (all features with 1-day lag to prevent look-ahead bias)
    logger.info("Adding VIX regime features...")
    add_vix_regime_features(
        indicators_by_symbol,
        vix_symbol=vix_symbol,
        vxn_symbol=vxn_symbol
    )
    logger.debug("  ✓ VIX regime features completed")

    # 10) Volatility term structure features (VXX/VIXY)
    logger.info("Adding volatility term structure features...")
    add_volatility_term_structure(indicators_by_symbol)
    logger.debug("  ✓ Volatility term structure features completed")

    # 11) Weekly VIX regime features
    logger.info("Adding weekly VIX regime features...")
    add_weekly_vix_features(
        indicators_by_symbol,
        vix_symbol=vix_symbol,
        vxn_symbol=vxn_symbol
    )
    logger.debug("  ✓ Weekly VIX regime features completed")

    # 12) Cross-asset signal features (macro regime indicators)
    logger.info("Adding cross-asset signal features...")
    add_cross_asset_features(indicators_by_symbol)
    logger.debug("  ✓ Cross-asset features completed")

    # 13) FRED macro features (treasury yields, credit spreads, economic indicators)
    logger.info("Adding FRED macro features...")
    try:
        from pathlib import Path
        cache_path = Path(__file__).parent.parent.parent / "cache" / "fred_data.parquet"
        add_fred_features(
            indicators_by_symbol,
            cache_path=cache_path,
            additional_lag_days=0  # Publication lag is already built into each series
        )
        logger.debug("  ✓ FRED macro features completed")
    except Exception as e:
        logger.warning(f"FRED features skipped (may need FRED_API_KEY): {e}")

    logger.info(f"Cross-sectional feature computation completed for {n_symbols} symbols")


def compute_cross_sectional_features_safe(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    **kwargs
) -> bool:
    """
    Safe wrapper for compute_cross_sectional_features with error handling.

    This version catches exceptions and logs errors instead of raising them,
    allowing the pipeline to continue even if cross-sectional computation fails.

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame with features
        **kwargs: Additional arguments passed to compute_cross_sectional_features

    Returns:
        True if successful, False if computation failed
    """
    try:
        compute_cross_sectional_features(indicators_by_symbol, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Cross-sectional feature computation failed: {e}")
        return False


def compute_weekly_cross_sectional_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    sp500_tickers: Optional[List[str]] = None,
    prefix: str = "w_",
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
) -> None:
    """
    Compute weekly cross-sectional features by resampling daily data to weekly.

    This function creates weekly versions of key cross-sectional features:
    - Alpha momentum vs SPY, QQQ, and sector (w_alpha_mom_*)
    - Beta vs market (w_beta_spy)
    - Cross-asset ratios (w_copper_gold_ratio, w_gold_spy_ratio, etc.)
    - FRED macro features (w_fred_*)
    - Breadth features (w_ad_ratio_universe, w_mcclellan_oscillator)
    - Relative strength vs benchmarks (w_rel_strength_*)

    These features are computed on weekly-resampled data and then merged back
    to daily with forward-fill to provide multi-timeframe signals.

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame (modified in place)
        market_symbol: Symbol to use as market benchmark (default: "SPY")
        qqq_symbol: Symbol to use as growth/liquidity benchmark (default: "QQQ")
        sp500_tickers: List of S&P 500 tickers for breadth calculation
        prefix: Prefix for weekly feature names (default: "w_")
        sectors: Dict mapping symbol -> sector name (for sector-relative features)
        sector_to_etf: Dict mapping sector name -> ETF symbol (for sector benchmarks)
    """
    logger.info(f"Computing weekly cross-sectional features (prefix={prefix})")

    n_symbols = len(indicators_by_symbol)

    # Step 1: Create weekly-resampled data for the entire universe
    logger.info("Resampling daily data to weekly for cross-sectional computation...")
    weekly_by_symbol: Dict[str, pd.DataFrame] = {}

    for sym, df in indicators_by_symbol.items():
        if df.empty:
            continue

        # Need DatetimeIndex for resampling
        if isinstance(df.index, pd.DatetimeIndex):
            df_work = df.copy()
        elif 'date' in df.columns:
            df_work = df.set_index('date').copy()
        else:
            continue

        # Resample to weekly (Friday close)
        # Only keep OHLCV columns needed for cross-sectional computation
        ohlcv_cols = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
        available_cols = [c for c in ohlcv_cols if c in df_work.columns]

        if not available_cols or 'adjclose' not in available_cols:
            continue

        try:
            # Standard OHLCV aggregation rules
            agg_rules = {}
            if 'open' in df_work.columns:
                agg_rules['open'] = lambda x: x.iloc[0] if len(x) > 0 else np.nan
            if 'high' in df_work.columns:
                agg_rules['high'] = 'max'
            if 'low' in df_work.columns:
                agg_rules['low'] = 'min'
            if 'close' in df_work.columns:
                agg_rules['close'] = lambda x: x.iloc[-1] if len(x) > 0 else np.nan
            if 'adjclose' in df_work.columns:
                agg_rules['adjclose'] = lambda x: x.iloc[-1] if len(x) > 0 else np.nan
            if 'volume' in df_work.columns:
                agg_rules['volume'] = 'sum'

            weekly_df = df_work[list(agg_rules.keys())].resample('W-FRI').agg(agg_rules)
            weekly_df = weekly_df.dropna(subset=['adjclose'] if 'adjclose' in weekly_df.columns else ['close'])

            # Compute returns on weekly data
            if 'adjclose' in weekly_df.columns:
                weekly_df['ret'] = weekly_df['adjclose'].pct_change()
            elif 'close' in weekly_df.columns:
                weekly_df['ret'] = weekly_df['close'].pct_change()

            if len(weekly_df) >= 10:  # Need minimum data for features
                weekly_by_symbol[sym] = weekly_df

        except Exception as e:
            logger.debug(f"Weekly resampling failed for {sym}: {e}")
            continue

    logger.info(f"Created weekly data for {len(weekly_by_symbol)} symbols")

    if len(weekly_by_symbol) < 10:
        logger.warning("Insufficient weekly data for cross-sectional features")
        return

    # Step 2: Compute weekly cross-asset features
    logger.info("Computing weekly cross-asset features...")
    try:
        _add_weekly_cross_asset_features(weekly_by_symbol, prefix=prefix)
    except Exception as e:
        logger.warning(f"Weekly cross-asset features failed: {e}")

    # Step 3: Compute weekly FRED features
    logger.info("Computing weekly FRED features...")
    try:
        _add_weekly_fred_features(weekly_by_symbol, prefix=prefix)
    except Exception as e:
        logger.warning(f"Weekly FRED features failed: {e}")

    # Step 4: Compute weekly alpha/beta features
    logger.info("Computing weekly alpha/beta features...")
    try:
        _add_weekly_alpha_features(
            weekly_by_symbol,
            market_symbol=market_symbol,
            qqq_symbol=qqq_symbol,
            prefix=prefix,
            sectors=sectors,
            sector_to_etf=sector_to_etf,
        )
    except Exception as e:
        logger.warning(f"Weekly alpha features failed: {e}")

    # Step 5: Compute weekly breadth features
    if sp500_tickers:
        logger.info("Computing weekly breadth features...")
        try:
            _add_weekly_breadth_features(weekly_by_symbol, sp500_tickers, prefix=prefix)
        except Exception as e:
            logger.warning(f"Weekly breadth features failed: {e}")

    # Step 6: Compute weekly relative strength features
    if sectors and sector_to_etf:
        logger.info("Computing weekly relative strength features...")
        try:
            _add_weekly_relative_strength_features(
                weekly_by_symbol,
                market_symbol=market_symbol,
                qqq_symbol=qqq_symbol,
                sectors=sectors,
                sector_to_etf=sector_to_etf,
                prefix=prefix
            )
        except Exception as e:
            logger.warning(f"Weekly relative strength features failed: {e}")

    # Step 7: Merge weekly cross-sectional features back to daily data
    logger.info("Merging weekly cross-sectional features back to daily...")
    _merge_weekly_cs_to_daily(indicators_by_symbol, weekly_by_symbol, prefix=prefix)

    logger.info(f"Weekly cross-sectional feature computation completed")


def _add_weekly_cross_asset_features(
    weekly_by_symbol: Dict[str, pd.DataFrame],
    prefix: str = "w_"
) -> None:
    """Add weekly cross-asset features to weekly data."""
    from .cross_asset import (
        _get_price_series, _compute_ratio, _compute_zscore,
        _compute_momentum, _compute_rolling_corr, CROSS_ASSET_ETFS
    )

    # Extract weekly prices for ETFs
    prices = {}
    for name, symbols in CROSS_ASSET_ETFS.items():
        for symbol in symbols:
            if symbol in weekly_by_symbol:
                df = weekly_by_symbol[symbol]
                if 'adjclose' in df.columns:
                    series = pd.to_numeric(df['adjclose'], errors='coerce')
                    if series is not None and not series.isna().all():
                        prices[name] = series
                        break

    if len(prices) < 3:
        logger.warning("Insufficient ETF data for weekly cross-asset features")
        return

    features = {}

    # Compute key cross-asset ratios on weekly data
    if 'gold' in prices and 'equity' in prices:
        gold_spy = _compute_ratio(prices['gold'], prices['equity'], f'{prefix}gold_spy_ratio')
        features[f'{prefix}gold_spy_ratio'] = gold_spy
        # Use window=12, min_periods=4 for weekly data (12 weeks ~ 60 days)
        gold_spy_zscore = gold_spy.rolling(12, min_periods=4).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) >= 4 else np.nan,
            raw=False
        ).rename(f'{prefix}gold_spy_ratio_zscore')
        features[f'{prefix}gold_spy_ratio_zscore'] = gold_spy_zscore

    if 'copper' in prices and 'gold' in prices:
        copper_gold = _compute_ratio(prices['copper'], prices['gold'], f'{prefix}copper_gold_ratio')
        features[f'{prefix}copper_gold_ratio'] = copper_gold

    if 'equity' in prices and 'long_bond' in prices:
        # Compute rolling correlation with appropriate min_periods for weekly data
        equity_ret = prices['equity'].pct_change()
        bond_ret = prices['long_bond'].pct_change()
        features[f'{prefix}equity_bond_corr_60d'] = equity_ret.rolling(12, min_periods=6).corr(bond_ret).rename(f'{prefix}equity_bond_corr_60d')

    if 'financials' in prices and 'utilities' in prices:
        features[f'{prefix}financials_utilities_ratio'] = _compute_ratio(
            prices['financials'], prices['utilities'], f'{prefix}financials_utilities_ratio'
        )

    if 'dollar' in prices:
        features[f'{prefix}dollar_momentum_20d'] = _compute_momentum(prices['dollar'], 4).rename(f'{prefix}dollar_momentum_20d')  # 4 weeks ~ 20 days

    # Attach to ALL weekly symbols
    for sym, df in weekly_by_symbol.items():
        new_cols = {}
        for name, series in features.items():
            if len(series) > 0:
                reindexed = series.reindex(df.index)
                new_cols[name] = pd.to_numeric(reindexed, errors='coerce').astype('float32')
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            weekly_by_symbol[sym] = pd.concat([df, new_df], axis=1)

    logger.info(f"  Added {len(features)} weekly cross-asset features")


def _add_weekly_fred_features(
    weekly_by_symbol: Dict[str, pd.DataFrame],
    prefix: str = "w_"
) -> None:
    """Add weekly FRED features to weekly data.

    Extended to compute all EXPANSION_CANDIDATES weekly FRED variants:
    - Credit spreads: BAMLH0A0HYM2, BAMLC0A4CBBB
    - Treasury yields: DGS10, DGS2
    - Yield curves: T10Y2Y, T10Y3M
    - Fed policy: DFEDTARU
    - Labor: ICSA, CCSA
    - Financial conditions: NFCI
    - Volatility: VIXCLS
    """
    from pathlib import Path

    try:
        from ..data.fred import download_fred_series, FRED_SERIES
    except ImportError:
        from src.data.fred import download_fred_series, FRED_SERIES

    # Load FRED data
    cache_path = Path(__file__).parent.parent.parent / "cache" / "fred_data.parquet"
    try:
        fred_df = download_fred_series(cache_path=cache_path)
    except Exception as e:
        logger.warning(f"Could not load FRED data: {e}")
        return

    # Resample FRED data to weekly (Friday)
    fred_weekly = fred_df.resample('W-FRI').last()

    # Helper functions for computing features
    def compute_change(series: pd.Series, periods: int) -> pd.Series:
        """Compute change over N periods."""
        return series.diff(periods)

    def compute_zscore(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
        """Compute z-score over rolling window."""
        if min_periods is None:
            min_periods = window // 4
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()
        return (series - mean) / std.replace(0, np.nan)

    def compute_percentile(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
        """Compute percentile rank over rolling window."""
        if min_periods is None:
            min_periods = window // 4
        return series.rolling(window, min_periods=min_periods).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x.dropna()) > min_periods else np.nan,
            raw=False
        )

    features = {}

    # ==========================================================================
    # Credit Spreads
    # ==========================================================================

    # High Yield OAS (BAMLH0A0HYM2)
    if 'BAMLH0A0HYM2' in fred_weekly.columns:
        series = fred_weekly['BAMLH0A0HYM2']
        # Weekly equivalents: 5d ~ 1w, 20d ~ 4w, 60d ~ 12w, 252d ~ 52w
        features[f'{prefix}fred_bamlh0a0hym2_chg5d'] = compute_change(series, 1).astype('float32')
        features[f'{prefix}fred_bamlh0a0hym2_chg20d'] = compute_change(series, 4).astype('float32')
        features[f'{prefix}fred_bamlh0a0hym2_z60'] = compute_zscore(series, 12).astype('float32')
        features[f'{prefix}fred_bamlh0a0hym2_pct252'] = compute_percentile(series, 52).astype('float32')

    # BBB Corporate OAS (BAMLC0A4CBBB)
    if 'BAMLC0A4CBBB' in fred_weekly.columns:
        series = fred_weekly['BAMLC0A4CBBB']
        features[f'{prefix}fred_bamlc0a4cbbb_chg5d'] = compute_change(series, 1).astype('float32')
        features[f'{prefix}fred_bamlc0a4cbbb_z60'] = compute_zscore(series, 12).astype('float32')

    # ==========================================================================
    # Treasury Yields
    # ==========================================================================

    # 10-Year Treasury (DGS10)
    if 'DGS10' in fred_weekly.columns:
        series = fred_weekly['DGS10']
        features[f'{prefix}fred_dgs10_chg5d'] = compute_change(series, 1).astype('float32')
        features[f'{prefix}fred_dgs10_chg20d'] = compute_change(series, 4).astype('float32')
        features[f'{prefix}fred_dgs10_z60'] = compute_zscore(series, 12).astype('float32')
        features[f'{prefix}fred_dgs10_pct252'] = compute_percentile(series, 52).astype('float32')

    # 2-Year Treasury (DGS2)
    if 'DGS2' in fred_weekly.columns:
        series = fred_weekly['DGS2']
        features[f'{prefix}fred_dgs2_chg5d'] = compute_change(series, 1).astype('float32')
        features[f'{prefix}fred_dgs2_chg20d'] = compute_change(series, 4).astype('float32')

    # ==========================================================================
    # Yield Curves
    # ==========================================================================

    # 10Y-2Y Spread (T10Y2Y)
    if 'T10Y2Y' in fred_weekly.columns:
        series = fred_weekly['T10Y2Y']
        features[f'{prefix}fred_t10y2y_chg5d'] = compute_change(series, 1).astype('float32')
        features[f'{prefix}fred_t10y2y_z60'] = compute_zscore(series, 12).astype('float32')
        features[f'{prefix}fred_t10y2y_pct252'] = compute_percentile(series, 52).astype('float32')

    # 10Y-3M Spread (T10Y3M)
    if 'T10Y3M' in fred_weekly.columns:
        series = fred_weekly['T10Y3M']
        features[f'{prefix}fred_t10y3m_z60'] = compute_zscore(series, 12).astype('float32')

    # ==========================================================================
    # Fed Policy
    # ==========================================================================

    # Fed Funds Target (DFEDTARU)
    if 'DFEDTARU' in fred_weekly.columns:
        series = fred_weekly['DFEDTARU']
        features[f'{prefix}fred_dfedtaru_chg20d'] = compute_change(series, 4).astype('float32')

    # ==========================================================================
    # Labor Market
    # ==========================================================================

    # Initial Jobless Claims (ICSA)
    if 'ICSA' in fred_weekly.columns:
        series = fred_weekly['ICSA']
        features[f'{prefix}fred_icsa_chg4w'] = compute_change(series, 4).astype('float32')
        features[f'{prefix}fred_icsa_z52w'] = compute_zscore(series, 52).astype('float32')
        features[f'{prefix}fred_icsa_pct104w'] = compute_percentile(series, 104).astype('float32')

    # Continued Claims (CCSA)
    if 'CCSA' in fred_weekly.columns:
        series = fred_weekly['CCSA']
        features[f'{prefix}fred_ccsa_chg4w'] = compute_change(series, 4).astype('float32')
        features[f'{prefix}fred_ccsa_z52w'] = compute_zscore(series, 52).astype('float32')

    # ==========================================================================
    # Financial Conditions
    # ==========================================================================

    # Chicago Fed NFCI
    if 'NFCI' in fred_weekly.columns:
        series = fred_weekly['NFCI']
        features[f'{prefix}fred_nfci_chg4w'] = compute_change(series, 4).astype('float32')
        features[f'{prefix}fred_nfci_z52w'] = compute_zscore(series, 52).astype('float32')

    # ==========================================================================
    # Volatility
    # ==========================================================================

    # VIX (VIXCLS)
    if 'VIXCLS' in fred_weekly.columns:
        series = fred_weekly['VIXCLS']
        features[f'{prefix}fred_vixcls_chg5d'] = compute_change(series, 1).astype('float32')
        features[f'{prefix}fred_vixcls_z60'] = compute_zscore(series, 12).astype('float32')
        features[f'{prefix}fred_vixcls_pct252'] = compute_percentile(series, 52).astype('float32')

    if not features:
        logger.warning("No weekly FRED features computed")
        return

    # Attach to all weekly symbols
    for sym, df in weekly_by_symbol.items():
        new_cols = {}
        for name, series in features.items():
            if len(series) > 0:
                reindexed = series.reindex(df.index)
                # Apply publication lag (shift by 1 week for safety)
                lagged = reindexed.shift(1)
                new_cols[name] = pd.to_numeric(lagged, errors='coerce').astype('float32')
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            weekly_by_symbol[sym] = pd.concat([df, new_df], axis=1)

    logger.info(f"  Added {len(features)} weekly FRED features")


def _add_weekly_alpha_features(
    weekly_by_symbol: Dict[str, pd.DataFrame],
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    prefix: str = "w_",
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
) -> None:
    """Add weekly alpha/beta features to weekly data.

    Extended to compute:
    - w_beta_spy, w_beta_qqq
    - w_alpha_mom_spy_{20,60,120}_ema10
    - w_alpha_mom_qqq_{20,60}_ema10
    - w_alpha_mom_sector_{20,60}_ema10
    - w_alpha_mom_combo_{20,60}_ema10
    - w_alpha_resid_spy, w_alpha_resid_qqq
    """
    # Get market returns
    if market_symbol not in weekly_by_symbol:
        logger.warning(f"Market symbol {market_symbol} not in weekly data")
        return

    spy_df = weekly_by_symbol[market_symbol]
    if 'ret' not in spy_df.columns:
        logger.warning(f"No returns column in {market_symbol}")
        return

    spy_ret = pd.to_numeric(spy_df['ret'], errors='coerce')

    # Get QQQ returns if available
    qqq_ret = None
    if qqq_symbol in weekly_by_symbol:
        qqq_df = weekly_by_symbol[qqq_symbol]
        if 'ret' in qqq_df.columns:
            qqq_ret = pd.to_numeric(qqq_df['ret'], errors='coerce')

    # Build sector returns dict
    sector_rets = {}
    if sectors and sector_to_etf:
        for sector_name, etf_symbol in sector_to_etf.items():
            if etf_symbol in weekly_by_symbol:
                etf_df = weekly_by_symbol[etf_symbol]
                if 'ret' in etf_df.columns:
                    sector_rets[sector_name.lower()] = pd.to_numeric(etf_df['ret'], errors='coerce')

    # Window configurations (in weeks)
    beta_window = 12  # ~60 days = 12 weeks
    alpha_windows = [4, 12, 24]  # ~20, 60, 120 days
    ema_span = 2  # ~10 days = 2 weeks

    count = 0
    for sym, df in weekly_by_symbol.items():
        if sym == market_symbol or sym == qqq_symbol:
            continue
        if sym in [etf for etf in sector_to_etf.values()] if sector_to_etf else False:
            continue
        if 'ret' not in df.columns:
            continue

        stock_ret = pd.to_numeric(df['ret'], errors='coerce')

        # Align returns with SPY
        aligned = pd.DataFrame({
            'stock': stock_ret,
            'spy': spy_ret
        }).dropna()

        if len(aligned) < beta_window:
            continue

        new_features = {}

        # Rolling beta vs SPY
        try:
            cov = aligned['stock'].rolling(beta_window, min_periods=beta_window//2).cov(aligned['spy'])
            var = aligned['spy'].rolling(beta_window, min_periods=beta_window//2).var()
            beta = cov / var.replace(0, np.nan)
            new_features[f'{prefix}beta_spy'] = beta.reindex(df.index).astype('float32')

            # Alpha residual (stock return - beta * market return)
            alpha_resid = aligned['stock'] - beta * aligned['spy']
            new_features[f'{prefix}alpha_resid_spy'] = alpha_resid.reindex(df.index).astype('float32')

            # Alpha momentum for multiple windows
            for win in alpha_windows:
                if len(aligned) >= win:
                    alpha_cum = alpha_resid.rolling(win, min_periods=win//2).sum()
                    alpha_mom = alpha_cum.ewm(span=ema_span, min_periods=1).mean()
                    days_equiv = win * 5  # Convert weeks to approx days
                    new_features[f'{prefix}alpha_mom_spy_{days_equiv}_ema10'] = alpha_mom.reindex(df.index).astype('float32')

        except Exception as e:
            logger.debug(f"SPY alpha computation failed for {sym}: {e}")
            continue

        # Alpha vs QQQ if available
        if qqq_ret is not None:
            try:
                aligned_qqq = pd.DataFrame({
                    'stock': stock_ret,
                    'spy': spy_ret,
                    'qqq': qqq_ret
                }).dropna()

                if len(aligned_qqq) >= beta_window:
                    cov_qqq = aligned_qqq['stock'].rolling(beta_window, min_periods=beta_window//2).cov(aligned_qqq['qqq'])
                    var_qqq = aligned_qqq['qqq'].rolling(beta_window, min_periods=beta_window//2).var()
                    beta_qqq = cov_qqq / var_qqq.replace(0, np.nan)
                    new_features[f'{prefix}beta_qqq'] = beta_qqq.reindex(df.index).astype('float32')

                    alpha_qqq = aligned_qqq['stock'] - beta_qqq * aligned_qqq['qqq']
                    new_features[f'{prefix}alpha_resid_qqq'] = alpha_qqq.reindex(df.index).astype('float32')

                    # Alpha momentum vs QQQ for different windows
                    for win in [4, 12]:  # 20, 60 days equiv
                        if len(aligned_qqq) >= win:
                            alpha_cum = alpha_qqq.rolling(win, min_periods=win//2).sum()
                            alpha_mom = alpha_cum.ewm(span=ema_span, min_periods=1).mean()
                            days_equiv = win * 5
                            new_features[f'{prefix}alpha_mom_qqq_{days_equiv}_ema10'] = alpha_mom.reindex(df.index).astype('float32')

                    # Alpha spread (longer window for stability)
                    cov_spy = aligned_qqq['stock'].rolling(beta_window, min_periods=beta_window//2).cov(aligned_qqq['spy'])
                    var_spy = aligned_qqq['spy'].rolling(beta_window, min_periods=beta_window//2).var()
                    beta_spy_qqq = cov_spy / var_spy.replace(0, np.nan)
                    alpha_spy_qqq = aligned_qqq['stock'] - beta_spy_qqq * aligned_qqq['spy']

                    alpha_spread = alpha_spy_qqq.rolling(12, min_periods=6).sum() - alpha_qqq.rolling(12, min_periods=6).sum()
                    alpha_spread_ema = alpha_spread.ewm(span=ema_span, min_periods=1).mean()
                    new_features[f'{prefix}alpha_mom_qqq_spread_60_ema10'] = alpha_spread_ema.reindex(df.index).astype('float32')

            except Exception as e:
                logger.debug(f"QQQ alpha computation failed for {sym}: {e}")

        # Alpha vs Sector if available
        if sectors and sym in sectors:
            sector_name = sectors[sym].lower()
            if sector_name in sector_rets:
                try:
                    sector_ret = sector_rets[sector_name]
                    aligned_sect = pd.DataFrame({
                        'stock': stock_ret,
                        'sector': sector_ret
                    }).dropna()

                    if len(aligned_sect) >= beta_window:
                        cov_sect = aligned_sect['stock'].rolling(beta_window, min_periods=beta_window//2).cov(aligned_sect['sector'])
                        var_sect = aligned_sect['sector'].rolling(beta_window, min_periods=beta_window//2).var()
                        beta_sect = cov_sect / var_sect.replace(0, np.nan)
                        new_features[f'{prefix}beta_sector'] = beta_sect.reindex(df.index).astype('float32')

                        alpha_sect = aligned_sect['stock'] - beta_sect * aligned_sect['sector']

                        # Sector alpha momentum
                        for win in [4, 12]:  # 20, 60 days equiv
                            if len(aligned_sect) >= win:
                                alpha_cum = alpha_sect.rolling(win, min_periods=win//2).sum()
                                alpha_mom = alpha_cum.ewm(span=ema_span, min_periods=1).mean()
                                days_equiv = win * 5
                                new_features[f'{prefix}alpha_mom_sector_{days_equiv}_ema10'] = alpha_mom.reindex(df.index).astype('float32')

                        # Combo alpha (average of SPY and sector alpha)
                        if f'{prefix}alpha_resid_spy' in new_features:
                            alpha_spy_vals = new_features[f'{prefix}alpha_resid_spy']
                            for win in [4, 12]:
                                if len(aligned_sect) >= win:
                                    combo_alpha = (alpha_resid.rolling(win, min_periods=win//2).sum() +
                                                   alpha_sect.rolling(win, min_periods=win//2).sum()) / 2
                                    combo_mom = combo_alpha.ewm(span=ema_span, min_periods=1).mean()
                                    days_equiv = win * 5
                                    new_features[f'{prefix}alpha_mom_combo_{days_equiv}_ema10'] = combo_mom.reindex(df.index).astype('float32')

                except Exception as e:
                    logger.debug(f"Sector alpha computation failed for {sym}: {e}")

        # Add features to dataframe
        if new_features:
            new_df = pd.DataFrame(new_features, index=df.index)
            weekly_by_symbol[sym] = pd.concat([df, new_df], axis=1)
            count += 1

    logger.info(f"  Added weekly alpha features to {count} symbols")


def _add_weekly_breadth_features(
    weekly_by_symbol: Dict[str, pd.DataFrame],
    sp500_tickers: List[str],
    prefix: str = "w_"
) -> None:
    """Add weekly breadth features to weekly data.

    Extended to compute:
    - w_ad_ratio_universe: Raw advance-decline ratio
    - w_ad_ratio_ema10: EMA-smoothed A/D ratio (2-week EMA for weekly data)
    - w_mcclellan_oscillator: Weekly McClellan oscillator (4/8 week EMAs)
    - w_ad_thrust_4w: A/D thrust indicator (4-week sum of extreme readings)
    """
    # Count advancing and declining stocks each week
    available_tickers = [t for t in sp500_tickers if t in weekly_by_symbol]

    if len(available_tickers) < 50:
        logger.warning(f"Only {len(available_tickers)} SP500 tickers available for weekly breadth")
        return

    # Build returns matrix
    returns_list = []
    for ticker in available_tickers:
        df = weekly_by_symbol[ticker]
        if 'ret' in df.columns:
            returns_list.append(df['ret'].rename(ticker))

    if len(returns_list) < 50:
        logger.warning("Insufficient returns data for weekly breadth features")
        return

    returns_df = pd.concat(returns_list, axis=1)

    # Compute advancing/declining
    advancing = (returns_df > 0).sum(axis=1)
    declining = (returns_df < 0).sum(axis=1)
    total = advancing + declining

    # Build breadth features
    features = {}

    # AD ratio (raw)
    ad_ratio = (advancing / total.replace(0, np.nan)).astype('float32')
    features[f'{prefix}ad_ratio_universe'] = ad_ratio

    # AD ratio EMA (2-week EMA ~ 10 days)
    ad_ratio_ema = ad_ratio.ewm(span=2, min_periods=1).mean().astype('float32')
    features[f'{prefix}ad_ratio_ema10'] = ad_ratio_ema

    # A/D line (cumulative advancing - declining)
    ad_line = (advancing - declining).cumsum()

    # McClellan Oscillator (4-week EMA minus 8-week EMA of A/D diff)
    # Weekly: 4 weeks ~ 19 days, 8 weeks ~ 39 days
    ad_diff = advancing - declining
    ema_fast = ad_diff.ewm(span=4, min_periods=2).mean()
    ema_slow = ad_diff.ewm(span=8, min_periods=4).mean()
    mcclellan = (ema_fast - ema_slow).astype('float32')
    features[f'{prefix}mcclellan_oscillator'] = mcclellan

    # A/D thrust (4-week rolling count of high A/D ratio weeks)
    # High A/D = more than 60% advancing
    high_ad_signal = (ad_ratio > 0.60).astype(float)
    ad_thrust = high_ad_signal.rolling(4, min_periods=2).sum().astype('float32')
    features[f'{prefix}ad_thrust_4w'] = ad_thrust

    # Attach to all symbols
    for sym, df in weekly_by_symbol.items():
        new_cols = {}
        for name, series in features.items():
            reindexed = series.reindex(df.index)
            new_cols[name] = reindexed
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            weekly_by_symbol[sym] = pd.concat([df, new_df], axis=1)

    logger.info(f"  Added {len(features)} weekly breadth features")


def _add_weekly_relative_strength_features(
    weekly_by_symbol: Dict[str, pd.DataFrame],
    market_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    prefix: str = "w_"
) -> None:
    """Add weekly relative strength features.

    Computes RS metrics on weekly data for:
    - vs SPY (market benchmark)
    - vs QQQ (growth benchmark)
    - vs Sector ETF (sector benchmark)

    Features per benchmark:
    - rel_strength_{bench}: Raw RS ratio (price / benchmark)
    - rel_strength_{bench}_norm: Normalized RS (vs 12-week rolling mean)
    - rel_strength_{bench}_zscore: Z-scored RS (12-week window)
    """
    # Get benchmark prices
    spy_close = None
    qqq_close = None

    if market_symbol in weekly_by_symbol:
        spy_df = weekly_by_symbol[market_symbol]
        if 'close' in spy_df.columns:
            spy_close = pd.to_numeric(spy_df['close'], errors='coerce')
        else:
            logger.warning(f"No 'close' column in {market_symbol} weekly data")

    if qqq_symbol in weekly_by_symbol:
        qqq_df = weekly_by_symbol[qqq_symbol]
        if 'close' in qqq_df.columns:
            qqq_close = pd.to_numeric(qqq_df['close'], errors='coerce')

    if spy_close is None:
        logger.warning("Cannot compute weekly RS without SPY benchmark")
        return

    # Get sector ETF prices
    sector_etf_prices = {}
    if sector_to_etf:
        for sector_name, etf_symbol in sector_to_etf.items():
            if etf_symbol in weekly_by_symbol:
                etf_df = weekly_by_symbol[etf_symbol]
                if 'close' in etf_df.columns:
                    sector_etf_prices[sector_name.lower()] = pd.to_numeric(
                        etf_df['close'], errors='coerce'
                    )

    # Helper functions
    def compute_rs(price: pd.Series, benchmark: pd.Series) -> pd.Series:
        """Compute raw relative strength ratio."""
        bench_aligned = benchmark.reindex(price.index).replace(0, np.nan)
        return (price / bench_aligned).astype('float32')

    def compute_rs_norm(rs: pd.Series, window: int = 12) -> pd.Series:
        """Compute normalized RS (current/rolling_mean - 1)."""
        roll_mean = rs.rolling(window, min_periods=max(3, window // 4)).mean()
        return ((rs / roll_mean) - 1.0).astype('float32')

    def compute_rs_zscore(rs: pd.Series, window: int = 12) -> pd.Series:
        """Compute z-scored RS."""
        roll_mean = rs.rolling(window, min_periods=max(3, window // 4)).mean()
        roll_std = rs.rolling(window, min_periods=max(3, window // 4)).std()
        return ((rs - roll_mean) / roll_std.replace(0, np.nan)).astype('float32')

    # Get stock-to-sector mapping
    stock_sectors = {}
    if sectors:
        stock_sectors = {sym: sec.lower() for sym, sec in sectors.items()}

    feature_count = 0
    processed_symbols = 0

    # Compute RS for each stock
    for sym, df in weekly_by_symbol.items():
        # Skip benchmarks
        if sym == market_symbol or sym == qqq_symbol:
            continue
        if sector_to_etf and sym in sector_to_etf.values():
            continue

        if 'close' not in df.columns:
            continue

        stock_close = pd.to_numeric(df['close'], errors='coerce')

        new_features = {}

        # RS vs SPY
        rs_spy = compute_rs(stock_close, spy_close)
        new_features[f'{prefix}rel_strength_spy'] = rs_spy
        new_features[f'{prefix}rel_strength_spy_norm'] = compute_rs_norm(rs_spy)
        new_features[f'{prefix}rel_strength_spy_zscore'] = compute_rs_zscore(rs_spy)

        # RS vs QQQ (if available)
        if qqq_close is not None:
            rs_qqq = compute_rs(stock_close, qqq_close)
            new_features[f'{prefix}rel_strength_qqq'] = rs_qqq
            new_features[f'{prefix}rel_strength_qqq_norm'] = compute_rs_norm(rs_qqq)
            new_features[f'{prefix}rel_strength_qqq_zscore'] = compute_rs_zscore(rs_qqq)

        # RS vs Sector ETF (if available)
        stock_sector = stock_sectors.get(sym)
        if stock_sector and stock_sector in sector_etf_prices:
            sector_close = sector_etf_prices[stock_sector]
            rs_sector = compute_rs(stock_close, sector_close)
            new_features[f'{prefix}rel_strength_sector'] = rs_sector
            new_features[f'{prefix}rel_strength_sector_norm'] = compute_rs_norm(rs_sector)
            new_features[f'{prefix}rel_strength_sector_zscore'] = compute_rs_zscore(rs_sector)

        # Add features to DataFrame
        if new_features:
            new_df = pd.DataFrame(new_features, index=df.index)
            weekly_by_symbol[sym] = pd.concat([df, new_df], axis=1)
            feature_count = max(feature_count, len(new_features))
            processed_symbols += 1

    logger.info(f"  Added up to {feature_count} weekly RS features for {processed_symbols} symbols")


def _merge_weekly_cs_to_daily(
    daily_by_symbol: Dict[str, pd.DataFrame],
    weekly_by_symbol: Dict[str, pd.DataFrame],
    prefix: str = "w_"
) -> None:
    """Merge weekly cross-sectional features back to daily data."""
    # Identify weekly CS feature columns (those that start with prefix)
    sample_sym = next(iter(weekly_by_symbol.keys()))
    sample_weekly = weekly_by_symbol[sample_sym]
    weekly_cs_cols = [c for c in sample_weekly.columns if c.startswith(prefix)]

    if not weekly_cs_cols:
        logger.warning("No weekly cross-sectional features to merge")
        return

    logger.info(f"  Merging {len(weekly_cs_cols)} weekly CS features to daily")

    merged_count = 0
    for sym, daily_df in daily_by_symbol.items():
        if sym not in weekly_by_symbol:
            continue

        weekly_df = weekly_by_symbol[sym]

        # Get only the weekly CS columns
        weekly_cs_cols_available = [c for c in weekly_cs_cols if c in weekly_df.columns]
        if not weekly_cs_cols_available:
            continue

        weekly_features = weekly_df[weekly_cs_cols_available].copy()

        # Ensure DatetimeIndex for merge_asof
        if isinstance(daily_df.index, pd.DatetimeIndex):
            daily_dates = daily_df.index
            had_datetime_index = True
        elif 'date' in daily_df.columns:
            daily_dates = pd.to_datetime(daily_df['date'])
            had_datetime_index = False
        else:
            continue

        # Reset index for merge_asof
        weekly_features = weekly_features.reset_index()
        weekly_features.columns = ['week_end'] + list(weekly_features.columns[1:])

        # Prepare daily for merge
        if had_datetime_index:
            daily_for_merge = daily_df.reset_index()
            daily_for_merge.columns = ['date'] + list(daily_for_merge.columns[1:])
        else:
            daily_for_merge = daily_df.copy()
            daily_for_merge['date'] = daily_dates

        # Drop rows with null dates before merge_asof
        null_dates_mask = daily_for_merge['date'].isna()
        if null_dates_mask.any():
            logger.debug(f"  Dropping {null_dates_mask.sum()} rows with null dates for {sym}")
            daily_for_merge = daily_for_merge[~null_dates_mask]

        daily_for_merge = daily_for_merge.sort_values('date')
        weekly_features = weekly_features.sort_values('week_end')

        # Skip if no data left
        if daily_for_merge.empty or weekly_features.empty:
            continue

        # Merge with backward fill (weekly value applies to current and future days until next week)
        merged = pd.merge_asof(
            daily_for_merge,
            weekly_features,
            left_on='date',
            right_on='week_end',
            direction='backward'
        )

        # Drop the week_end column
        if 'week_end' in merged.columns:
            merged = merged.drop(columns=['week_end'])

        # Restore original index structure
        if had_datetime_index:
            merged = merged.set_index('date')
        else:
            # Keep date column as is
            pass

        daily_by_symbol[sym] = merged
        merged_count += 1

    logger.info(f"  Merged weekly CS features to {merged_count} daily symbols")


if __name__ == "__main__":
    # Example usage for testing
    import logging
    import numpy as np

    logging.basicConfig(level=logging.DEBUG)

    # Create sample multi-stock data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']

    indicators_by_symbol = {}
    for sym in symbols:
        df = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'adjclose': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'ret': np.random.randn(len(dates)) * 0.02,
            'rv_ratio_20_100': np.random.randn(len(dates)) * 0.5 + 1.0,
            'vol_regime': np.random.randn(len(dates)) * 0.1
        }, index=dates)
        indicators_by_symbol[sym] = df

    # Sector mapping
    sectors = {
        'AAPL': 'technology',
        'MSFT': 'technology',
        'GOOGL': 'technology',
        'SPY': 'market'
    }

    sector_to_etf = {
        'technology': 'XLK',
        'market': 'SPY'
    }

    # Compute cross-sectional features
    compute_cross_sectional_features(
        indicators_by_symbol,
        sectors=sectors,
        sector_to_etf=sector_to_etf,
        market_symbol='SPY',
        sp500_tickers=['AAPL', 'MSFT', 'GOOGL']
    )

    # Show what was added
    sample_df = indicators_by_symbol['AAPL']
    print(f"\nTotal columns in AAPL: {len(sample_df.columns)}")

    # Show cross-sectional feature columns
    cs_cols = [col for col in sample_df.columns if any(x in col for x in
               ['_cs_', 'vol_regime_rel', 'alpha_', 'rs_', 'breadth_', 'xmom_'])]
    print(f"Cross-sectional feature columns: {len(cs_cols)}")
    print(f"Sample: {cs_cols[:10]}")
