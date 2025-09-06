"""
Main orchestration pipeline for feature computation.

This module contains the high-level workflow that coordinates data loading,
feature computation, and output generation. It includes parallel processing
of individual symbols and cross-sectional feature computation.
"""
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Relative imports for feature modules
from ..features.assemble import assemble_indicators_from_wide
from ..features.trend import add_trend_features, add_rsi_features, add_macd_features
from ..features.volatility import add_multiscale_vol_regime, add_vol_regime_cs_context
from ..features.hurst import add_hurst_features
from ..features.distance import add_distance_to_ma_features
from ..features.range_breakout import add_range_breakout_features
from ..features.volume import add_volume_features, add_volume_shock_features
from ..features.relstrength import add_relative_strength
from ..features.alpha import add_alpha_momentum_features
from ..features.breadth import add_breadth_series
from ..features.xsec import add_xsec_momentum_panel
from ..features.postprocessing import interpolate_internal_gaps
from ..features.ohlc_adjustment import adjust_ohlc_to_adjclose
from ..features.sector_mapping import build_enhanced_sector_mappings, get_required_etfs
from ..features.target_generation import generate_targets_parallel

logger = logging.getLogger(__name__)


def _feature_worker(sym: str, df: pd.DataFrame, cs_ratio_median: Optional[pd.Series] = None) -> Tuple[str, pd.DataFrame]:
    """
    Core feature computation worker for a single symbol.
    
    This function runs the complete feature stack for one symbol, including:
    - Trend features (MA slopes, agreement, etc.)
    - Multi-scale volatility regime features
    - Hurst exponent features
    - Distance-to-MA features
    - Range/breakout features (including ATR14)
    - Volume features and shocks
    
    Args:
        sym: Symbol ticker
        df: Input DataFrame with OHLCV data
        cs_ratio_median: Cross-sectional median of volatility ratios (optional)
        
    Returns:
        Tuple of (symbol, enriched_dataframe)
    """
    try:
        logger.debug(f"Processing features for {sym}")
        out = df.copy()

        # 0) First: Adjust OHLC to match adjusted close for consistent price data
        out = adjust_ohlc_to_adjclose(out)

        # Ensure returns exist (calculated from adjclose for consistency)
        if "ret" not in out.columns:
            if "adjclose" in out.columns:
                out["ret"] = np.log(pd.to_numeric(out["adjclose"], errors="coerce")).diff()
            else:
                logger.warning(f"{sym}: No adjclose column for returns calculation")
                return sym, out
        else:
            # Recalculate returns from adjclose to ensure consistency after OHLC adjustment
            if "adjclose" in out.columns:
                out["ret"] = np.log(pd.to_numeric(out["adjclose"], errors="coerce")).diff()

        # 1) Trend features (MA slopes, agreement, etc.)
        out = add_trend_features(
            out,
            src_col='adjclose',
            ma_periods=(10, 20, 30, 50, 75, 100, 150, 200),
            slope_window=20,
            eps=1e-5
        )

        # 1.5) RSI features (momentum oscillator)
        out = add_rsi_features(
            out,
            src_col='adjclose',
            periods=(14, 21, 30)
        )

        # 1.6) MACD features (histogram and derivative)
        out = add_macd_features(
            out,
            src_col='adjclose',
            fast=12,
            slow=26,
            signal=9,
            derivative_ema_span=3
        )

        # 2) Enhanced multi-scale vol regime
        out = add_multiscale_vol_regime(
            out,
            ret_col="ret",
            short_windows=(10, 20),
            long_windows=(60, 100),
            z_window=60,
            ema_span=10,
            slope_win=20,
            cs_ratio_median=cs_ratio_median,  # Will be None on first pass
        )

        # 3) Hurst exponent features
        out = add_hurst_features(out, ret_col="ret", windows=(64, 128), ema_halflife=5)

        # 4) Distance-to-MA + z-scores
        out = add_distance_to_ma_features(
            out,
            src_col='adjclose',
            ma_lengths=(20, 50, 100, 200),
            z_window=60
        )

        # 5) Range / breakout features
        out = add_range_breakout_features(out, win_list=(5, 10, 20))

        # 6) Volume features
        out = add_volume_features(out)

        # 7) Volume shock + alignment features
        out = add_volume_shock_features(
            out,
            vol_col="volume",
            price_col_primary="close",
            price_col_fallback="adjclose",
            lookback=20,
            ema_span=10,
            prefix="volshock"
        )

        logger.debug(f"Completed feature processing for {sym}")
        return sym, out

    except Exception as e:
        logger.error(f"Feature processing failed for {sym}: {e}")
        return sym, df


def build_feature_universe(
    max_stocks: Optional[int] = None,
    rate_limit: float = 1.0,
    interval: str = "1d",
    default_etfs: List[str] = None,
    spy_symbol: str = "SPY",
    sector_to_etf: Dict[str, str] = None,
    sp500_tickers: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Build the complete feature universe by loading data and computing all features.
    
    Args:
        max_stocks: Maximum number of stocks to load (None for all)
        rate_limit: Rate limit for data requests (requests per second)
        interval: Data interval (e.g., "1d")
        default_etfs: List of ETF symbols to load
        spy_symbol: Symbol to use as market benchmark
        sector_to_etf: Mapping of sector names to ETF symbols
        sp500_tickers: List of S&P 500 ticker symbols for breadth calculation
        
    Returns:
        Tuple of (features_dict, targets_dataframe) where:
        - features_dict: Dictionary mapping symbol -> DataFrame with all computed features  
        - targets_dataframe: DataFrame with triple barrier targets
        
    Raises:
        RuntimeError: If data loading fails
    """
    logger.info("Building feature universe")
    
    # Import data loading functions (assuming they exist in the original structure)
    try:
        from ...data.loader import load_stock_universe, load_etf_universe
    except ImportError:
        from src.data.loader import load_stock_universe, load_etf_universe

    # Set defaults - keep as None to use loader's DEFAULT_ETFS
    # default_etfs parameter is passed through to load_etf_universe()
    
    if sector_to_etf is None:
        sector_to_etf = {}

    # 1) Load stocks + ETFs
    logger.info("Loading stock universe")
    stocks, sectors = load_stock_universe(
        max_symbols=max_stocks, 
        update=False,
        rate_limit=rate_limit, 
        interval=interval, 
        include_sectors=True
    )
    if not stocks:
        raise RuntimeError("Failed to load stock universe.")

    logger.info("Loading ETF universe")
    etfs = load_etf_universe(
        etf_symbols=default_etfs,  # None uses loader's DEFAULT_ETFS
        update=False,
        rate_limit=rate_limit, 
        interval=interval
    )
    if not etfs:
        raise RuntimeError("Failed to load ETF universe.")

    # 2) Assemble per-symbol frames
    logger.info("Assembling indicators from wide data")
    data = {
        k: pd.concat([stocks.get(k, pd.DataFrame()),
                      etfs.get(k, pd.DataFrame())], axis=1).sort_index()
        for k in (set(stocks) | set(etfs))
    }
    indicators_by_symbol = assemble_indicators_from_wide(data, adjust_ohlc_with_factor=False)

    # 2.5) Enhanced sector/subsector mapping
    try:
        from ...data.loader import _discover_universe_csv
    except ImportError:
        from src.data.loader import _discover_universe_csv
    
    universe_csv = _discover_universe_csv(str(Path(__file__).parent.parent.parent / "cache" / "stock_data.pkl"))
    enhanced_mappings = build_enhanced_sector_mappings(
        universe_csv=universe_csv,
        stock_data=indicators_by_symbol,
        etf_data=etfs,
        base_sectors=sectors
    )
    
    # Ensure we have all required ETFs loaded
    required_etfs = get_required_etfs(enhanced_mappings)
    missing_etfs = set(required_etfs) - set(etfs.keys())
    if missing_etfs:
        logger.warning(f"Missing {len(missing_etfs)} required ETFs for enhanced mappings: {list(missing_etfs)[:5]}...")

    # 3) Core features (parallelized)
    logger.info(f"Processing {len(indicators_by_symbol)} symbols in parallel...")
    results = Parallel(
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        backend="loky",
        batch_size=8,
        verbose=0,
    )(
        delayed(_feature_worker)(sym, df) for sym, df in indicators_by_symbol.items()
    )

    indicators_by_symbol = {sym: df for sym, df in results}
    logger.info(f"Core features completed for {len(indicators_by_symbol)} symbols")

    # 4) Cross-sectional volatility regime context
    logger.info("Adding cross-sectional volatility regime context...")
    add_vol_regime_cs_context(indicators_by_symbol)

    # 5) Alpha-momentum features
    logger.info("Adding alpha-momentum features...")
    add_alpha_momentum_features(
        indicators_by_symbol,
        sectors=sectors,
        sector_to_etf=sector_to_etf,
        market_symbol=spy_symbol,
        beta_win=60,
        windows=(20, 60, 120),
        ema_span=10
    )

    # 6) Relative strength vs SPY, sectors, and subsectors
    logger.info("Adding relative strength features...")
    add_relative_strength(
        indicators_by_symbol, 
        sectors=sectors, 
        sector_to_etf=sector_to_etf, 
        spy_symbol=spy_symbol,
        enhanced_mappings=enhanced_mappings
    )

    # 7) Breadth features
    if sp500_tickers:
        logger.info("Adding breadth series...")
        add_breadth_series(indicators_by_symbol, sp500_tickers)

    # 8) Cross-sectional momentum
    logger.info("Adding cross-sectional momentum features...")
    add_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=(5, 20, 60),
        price_col="adjclose",
        sector_map=sectors
    )

    # 9) Generate triple barrier targets (parallel)
    logger.info("Generating triple barrier targets...")
    targets_df = _generate_triple_barrier_targets(indicators_by_symbol)
    
    # 10) Interpolate internal gaps (NaNs between observed values only)
    indicators_by_symbol = interpolate_internal_gaps(indicators_by_symbol)

    logger.info("Feature universe construction completed")
    return indicators_by_symbol, targets_df


def _generate_triple_barrier_targets(indicators_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert indicators to long format and generate triple barrier targets in parallel.
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames with features
        
    Returns:
        DataFrame with triple barrier targets
    """
    # Convert to long format for target generation
    long_format_data = []
    
    for symbol, df in indicators_by_symbol.items():
        if df.empty:
            continue
            
        # Extract required columns for triple barrier generation
        required_cols = ['adjclose', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Symbol {symbol} missing columns for target generation: {missing_cols}")
            continue
        
        # Check if ATR exists (should be computed in range_breakout features)
        if 'atr14' not in df.columns:
            logger.warning(f"Symbol {symbol} missing ATR for target generation")
            continue
        
        # Create long format record for this symbol
        symbol_data = df[['adjclose', 'high', 'low', 'atr14']].copy()
        symbol_data = symbol_data.dropna()
        
        if symbol_data.empty:
            continue
            
        # Add symbol column and rename for target generation
        symbol_data['symbol'] = symbol
        symbol_data['date'] = symbol_data.index
        symbol_data = symbol_data.rename(columns={'adjclose': 'close', 'atr14': 'atr'})
        
        long_format_data.append(symbol_data[['symbol', 'date', 'close', 'high', 'low', 'atr']])
    
    if not long_format_data:
        logger.warning("No valid data for triple barrier target generation")
        return pd.DataFrame()
    
    # Combine all symbol data
    df_long = pd.concat(long_format_data, ignore_index=True)
    logger.info(f"Prepared long format data: {len(df_long)} rows across {df_long['symbol'].nunique()} symbols")
    
    # Triple barrier configuration
    config = {
        'up_mult': 3.0,
        'dn_mult': 2.5,
        'max_horizon': 20,
        'start_every': 3,
    }
    
    # Generate targets in parallel
    targets_df = generate_targets_parallel(df_long, config, n_jobs=-1, chunk_size=32)
    
    if not targets_df.empty:
        logger.info(f"Generated {len(targets_df)} triple barrier targets")
    else:
        logger.warning("No triple barrier targets generated")
    
    return targets_df


def run_pipeline(
    max_stocks: Optional[int] = None,
    rate_limit: float = 1.0,
    interval: str = "1d",
    default_etfs: List[str] = None,
    spy_symbol: str = "SPY",
    sector_to_etf: Dict[str, str] = None,
    sp500_tickers: List[str] = None,
    output_dir: Path = Path("./artifacts"),
    include_sectors: bool = True
) -> None:
    """
    Run the complete feature computation pipeline and save outputs.
    
    This function orchestrates the entire process:
    1. Load stock and ETF data
    2. Compute all features in parallel
    3. Add cross-sectional features
    4. Save single long-format parquet file
    
    Args:
        max_stocks: Maximum number of stocks to process (None for all)
        rate_limit: Rate limit for data requests (requests per second)
        interval: Data interval (e.g., "1d")  
        default_etfs: List of ETF symbols to include
        spy_symbol: Market benchmark symbol
        sector_to_etf: Mapping of sector names to ETF symbols
        sp500_tickers: List of S&P 500 tickers for breadth calculation
        output_dir: Directory for output files
        include_sectors: Whether to include sector information in processing
    """
    logger.info("Starting feature computation pipeline")
    
    # Set up output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build feature universe (now returns features and targets)
    indicators_by_symbol, targets_df = build_feature_universe(
        max_stocks=max_stocks,
        rate_limit=rate_limit,
        interval=interval,
        default_etfs=default_etfs,
        spy_symbol=spy_symbol,
        sector_to_etf=sector_to_etf,
        sp500_tickers=sp500_tickers
    )

    # Save outputs (import saving functions)
    try:
        from ..io.saving import save_long_parquet
    except ImportError:
        # If relative import fails, try absolute import
        from src.io.saving import save_long_parquet

    logger.info("Saving output files...")
    save_long_parquet(indicators_by_symbol, out_path=output_dir / "features_long.parquet")
    
    # Save triple barrier targets if generated
    if not targets_df.empty:
        targets_df.to_parquet(output_dir / "targets_triple_barrier.parquet", index=False)
        logger.info(f"Saved {len(targets_df)} triple barrier targets")
    
    logger.info(f"Pipeline completed. Outputs saved to {output_dir}")


def main(include_sectors: bool = True) -> None:
    """
    Main entry point for the pipeline (backward compatibility).
    
    Args:
        include_sectors: Whether to include sector-based features
    """
    # Import SP500 tickers
    try:
        from ...cache.sp500_list import SP500_TICKERS
    except ImportError:
        from cache.sp500_list import SP500_TICKERS

    # Default sector mapping
    sector_to_etf = {
        "technology services": "XLK",
        "electronic technology": "XLK", 
        "finance": "XLF",
        "retail trade": "XRT",
        "health technology": "XLV",
        "consumer non-durables": "XLP",
        "producer manufacturing": "XLI",
        "energy minerals": "XLE",
        "consumer services": "XLY",
        "consumer durables": "XLY",
        "utilities": "XLU",
        "non-energy minerals": "XLB",
        "industrial services": "XLI",
        "transportation": "IYT",
        "commercial services": "XLC",
        "process industries": "XLB", 
        "communications": "XLC",
        "health services": "XLV",
        "distribution services": "XLI",
        "miscellaneous": "SPY",
    }

    run_pipeline(
        max_stocks=None,
        rate_limit=1.0,
        interval="1d",
        spy_symbol="SPY",
        sector_to_etf=sector_to_etf if include_sectors else None,
        sp500_tickers=SP500_TICKERS,
    )