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
    from ..data.fred import add_fred_features, download_fred_series
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
    from src.data.fred import add_fred_features, download_fred_series

logger = logging.getLogger(__name__)


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


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
    alpha_ema_span: int
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

    Returns:
        Dict mapping column_name -> pd.Series with all computed features
    """
    result = {}

    # Skip benchmark symbols
    if sym == market_symbol:
        return result

    # 1) Compute alpha momentum features
    alpha_features = _compute_alpha_for_symbol(
        sym, ret_index, ret_values, alpha_benchmarks, symbol_sector_etf,
        market_symbol, beta_win, alpha_windows, alpha_ema_span
    )
    result.update(alpha_features)

    # 2) Compute relative strength features
    rs_features = _compute_rs_for_symbol(
        sym, px_index, px_values, rs_benchmarks, symbol_rs_mapping, market_symbol
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
    alpha_ema_span: int
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

    Returns:
        List of (symbol, features_dict) tuples
    """
    results = []
    for work_item in work_items_batch:
        sym, ret_index, ret_values, px_index, px_values, sector_etf, rs_mapping = work_item
        features = _compute_all_per_symbol_features(
            sym, ret_index, ret_values, px_index, px_values,
            alpha_benchmarks, rs_benchmarks, sector_etf, rs_mapping,
            market_symbol, beta_win, alpha_windows, alpha_ema_span
        )
        results.append((sym, features))
    return results


def _prepare_alpha_benchmarks(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]],
    sector_to_etf: Optional[Dict[str, str]],
    market_symbol: str
) -> Tuple[Dict[str, Tuple[pd.Index, np.ndarray]], Dict[str, Optional[str]]]:
    """
    Prepare benchmarks and per-symbol mappings for alpha computation.

    Returns:
        Tuple of (benchmarks dict, symbol_sector_etfs dict)
    """
    sec_map_lc = {k.lower(): v for k, v in (sector_to_etf or {}).items()}
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # Market benchmark
    mkt_df = indicators_by_symbol.get(market_symbol)
    if mkt_df is not None and 'ret' in mkt_df.columns:
        mkt_ret = pd.to_numeric(mkt_df['ret'], errors='coerce')
        benchmarks[market_symbol] = (mkt_ret.index, mkt_ret.values)

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

    # Build per-symbol sector ETF mapping
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

    return benchmarks, symbol_sector_etfs


def _prepare_rs_benchmarks(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]],
    sector_to_etf: Optional[Dict[str, str]],
    enhanced_mappings: Optional[Dict[str, Dict]],
    market_symbol: str
) -> Tuple[Dict[str, Tuple[pd.Index, np.ndarray]], Dict[str, Dict[str, Any]]]:
    """
    Prepare benchmarks and per-symbol mappings for RS computation.

    Returns:
        Tuple of (benchmarks dict, symbol_mappings dict)
    """
    sector_to_etf = sector_to_etf or {}
    sector_to_etf_lc = {k.lower(): v for k, v in sector_to_etf.items()}
    benchmarks: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # SPY benchmark
    spy_data = indicators_by_symbol.get(market_symbol)
    if spy_data is not None and "adjclose" in spy_data.columns:
        spy_px = pd.to_numeric(spy_data["adjclose"], errors='coerce')
        benchmarks['SPY'] = (spy_px.index, spy_px.values)

    # RSP benchmark
    rsp_data = indicators_by_symbol.get('RSP')
    if rsp_data is not None and "adjclose" in rsp_data.columns:
        rsp_px = pd.to_numeric(rsp_data["adjclose"], errors='coerce')
        benchmarks['RSP'] = (rsp_px.index, rsp_px.values)

    # Collect all ETFs needed
    all_etfs_needed = set()

    if sectors and sector_to_etf:
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sector_to_etf_lc.get(sec.lower())
                if etf:
                    all_etfs_needed.add(etf)

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

    # Build per-symbol mapping info
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

    return benchmarks, symbol_mappings


def compute_per_symbol_cs_features_parallel(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    enhanced_mappings: Optional[Dict[str, Dict]] = None,
    market_symbol: str = "SPY",
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

    Features computed:
    - Alpha momentum (alpha_resid_spy, alpha_mom_spy_*, alpha_resid_sector, etc.)
    - Relative strength (rel_strength_spy, rel_strength_sector, etc.)

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame (modified in place)
        sectors: Optional mapping of symbol -> sector name
        sector_to_etf: Optional mapping of sector name -> ETF symbol
        enhanced_mappings: Optional enhanced sector/subsector mappings
        market_symbol: Symbol to use as market benchmark
        beta_win: Window for beta calculation in alpha features
        alpha_windows: Windows for alpha-momentum computation
        alpha_ema_span: EMA span for alpha smoothing
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    logger.info("Computing per-symbol cross-sectional features (unified parallel)")

    # Prepare all benchmarks upfront
    logger.info("Preparing benchmarks for alpha and RS computation...")
    alpha_benchmarks, symbol_sector_etfs = _prepare_alpha_benchmarks(
        indicators_by_symbol, sectors, sector_to_etf, market_symbol
    )
    rs_benchmarks, symbol_rs_mappings = _prepare_rs_benchmarks(
        indicators_by_symbol, sectors, sector_to_etf, enhanced_mappings, market_symbol
    )

    logger.info(f"Loaded {len(alpha_benchmarks)} alpha benchmarks, {len(rs_benchmarks)} RS benchmarks")

    # Build work items with all data needed for each symbol
    work_items = []
    for sym, df in indicators_by_symbol.items():
        # Skip benchmarks and symbols missing required columns
        if sym == market_symbol:
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
            symbol_sector_etfs.get(sym),
            symbol_rs_mappings.get(sym, {})
        ))

    n_symbols = len(work_items)

    # Calculate optimal parallel parameters
    if n_jobs == -1:
        effective_jobs = multiprocessing.cpu_count()
    else:
        effective_jobs = n_jobs

    min_chunk_size = 100
    chunk_size = max(min_chunk_size, n_symbols // effective_jobs)
    n_chunks = max(1, n_symbols // chunk_size)
    actual_workers = min(effective_jobs, n_chunks)

    logger.info(f"Processing {n_symbols} symbols in {n_chunks} batches "
                f"(~{chunk_size} symbols/batch, {actual_workers} workers)")

    # Process sequentially for small datasets
    if n_symbols < 10 or actual_workers == 1:
        logger.info("Using sequential processing for per-symbol CS features")
        results = []
        for work_item in work_items:
            sym, ret_index, ret_values, px_index, px_values, sector_etf, rs_mapping = work_item
            features = _compute_all_per_symbol_features(
                sym, ret_index, ret_values, px_index, px_values,
                alpha_benchmarks, rs_benchmarks, sector_etf, rs_mapping,
                market_symbol, beta_win, alpha_windows, alpha_ema_span
            )
            results.append((sym, features))
    else:
        # Batch parallel processing
        work_chunks = _chunk_list(work_items, chunk_size)

        try:
            batch_results = Parallel(
                n_jobs=actual_workers,
                backend='loky',
                verbose=0,
                prefer='processes'
            )(
                delayed(_compute_per_symbol_batch)(
                    chunk, alpha_benchmarks, rs_benchmarks,
                    market_symbol, beta_win, alpha_windows, alpha_ema_span
                )
                for chunk in work_chunks
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
                    market_symbol, beta_win, alpha_windows, alpha_ema_span
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
    sp500_tickers: Optional[List[str]] = None,
    enhanced_mappings: Optional[Dict[str, Dict]] = None,
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

    Features computed:
    1. Cross-sectional volatility regime context (vol_regime_cs_median, vol_regime_rel)
    2. Alpha-momentum features (beta-adjusted returns vs market and sectors) [PARALLEL]
    3. Relative strength features (vs SPY, sectors, and subsectors) [PARALLEL]
    4. Market breadth features (advancing/declining stocks, new highs/lows)
    5. Cross-sectional momentum z-scores (xsec_mom_*d_z)
    6. Cross-sectional percentile ranks (xsec_pct_*d)
    7. Weekly cross-sectional momentum z-scores (w_xsec_mom_*w_z)
    8. Weekly cross-sectional percentile ranks (w_xsec_pct_*w)
    9. VIX regime features (vix_level, vix_percentile, vix_regime, etc.)
    10. Volatility term structure features from VXX/VIXY
    11. Weekly VIX regime features (w_vix_level, w_vix_percentile_52w, etc.)
    12. Cross-asset signal features (macro regime indicators)
    13. FRED macro features (treasury yields, credit spreads, etc.)

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame with features
                             (modified in place)
        sectors: Dictionary mapping symbol -> sector name (optional)
        sector_to_etf: Dictionary mapping sector name -> ETF symbol (optional)
        market_symbol: Symbol to use as market benchmark (default: "SPY")
        sp500_tickers: List of S&P 500 tickers for breadth calculation (optional)
        enhanced_mappings: Enhanced sector/subsector mappings dictionary (optional)
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

    # 1) Cross-sectional volatility regime context
    logger.info("Adding cross-sectional volatility regime context...")
    add_vol_regime_cs_context(indicators_by_symbol)
    logger.debug("  ✓ Volatility context completed")

    # 2-3) Alpha-momentum AND Relative strength features (UNIFIED PARALLEL)
    # This is the key optimization: compute all per-symbol features in ONE parallel job
    if sectors is not None and sector_to_etf is not None:
        logger.info("Computing per-symbol CS features (alpha + RS) in unified parallel...")
        compute_per_symbol_cs_features_parallel(
            indicators_by_symbol,
            sectors=sectors,
            sector_to_etf=sector_to_etf,
            enhanced_mappings=enhanced_mappings,
            market_symbol=market_symbol,
            beta_win=beta_win,
            alpha_windows=alpha_windows,
            alpha_ema_span=alpha_ema_span,
            n_jobs=n_jobs
        )
        logger.debug("  ✓ Per-symbol CS features (alpha + RS) completed")
    else:
        logger.warning("Skipping alpha and RS features (sectors or sector_to_etf not provided)")

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
