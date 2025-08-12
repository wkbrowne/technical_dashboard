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
from ..features.trend import add_trend_features
from ..features.volatility import add_multiscale_vol_regime, add_vol_regime_cs_context
from ..features.hurst import add_hurst_features
from ..features.distance import add_distance_to_ma_features
from ..features.range_breakout import add_range_breakout_features
from ..features.volume import add_volume_features, add_volume_shock_features
from ..features.relstrength import add_relative_strength
from ..features.alpha import add_alpha_momentum_features
from ..features.breadth import add_breadth_series
from ..features.xsec import add_xsec_momentum_panel
from ..data.postprocessing import interpolate_internal_gaps

logger = logging.getLogger(__name__)


def _feature_worker(sym: str, df: pd.DataFrame, cs_ratio_median: Optional[pd.Series] = None) -> Tuple[str, pd.DataFrame]:
    """
    Core feature computation worker for a single symbol.
    
    This function runs the complete feature stack for one symbol, including:
    - Trend features (MA slopes, agreement, etc.)
    - Multi-scale volatility regime features
    - Hurst exponent features
    - Distance-to-MA features
    - Range/breakout features
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

        # Ensure returns exist
        if "ret" not in out.columns:
            if "adjclose" in out.columns:
                import numpy as np
                out["ret"] = np.log(pd.to_numeric(out["adjclose"], errors="coerce")).diff()
            else:
                logger.warning(f"{sym}: No adjclose column for returns calculation")
                return sym, out

        # 1) Trend features (MA slopes, agreement, etc.)
        out = add_trend_features(
            out,
            src_col='adjclose',
            ma_periods=(10, 20, 30, 50, 75, 100, 150, 200),
            slope_window=20,
            eps=1e-5
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
        Dictionary mapping symbol -> DataFrame with all computed features
        
    Raises:
        RuntimeError: If data loading fails
    """
    logger.info("Building feature universe")
    
    # Import data loading functions (assuming they exist in the original structure)
    try:
        from ...data.loader import load_stock_universe, load_etf_universe
    except ImportError:
        from src.data.loader import load_stock_universe, load_etf_universe

    # Set defaults
    if default_etfs is None:
        default_etfs = [
            "SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "HYG", "LQD",
            "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLC",
            "EFA", "EEM", "GLD", "SLV", "USO", "UNG",
            "SMH", "XRT", "ITA", "KBE", "KRE", "IBB", "IHE", "IYT", "XLRE"
        ]
    
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
        etf_symbols=default_etfs, 
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
    indicators_by_symbol = assemble_indicators_from_wide(data, adjust_ohlc_with_factor=True)

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

    # 6) Relative strength vs SPY and sectors
    logger.info("Adding relative strength features...")
    add_relative_strength(
        indicators_by_symbol, 
        sectors=sectors, 
        sector_to_etf=sector_to_etf, 
        spy_symbol=spy_symbol
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

    # 9) Interpolate internal gaps (NaNs between observed values only)
    indicators_by_symbol = interpolate_internal_gaps(indicators_by_symbol)

    logger.info("Feature universe construction completed")
    return indicators_by_symbol


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

    # Build feature universe
    indicators_by_symbol = build_feature_universe(
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