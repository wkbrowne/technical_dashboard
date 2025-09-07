"""
Main orchestration pipeline for feature computation.

This module contains the high-level workflow that coordinates data loading,
feature computation, and output generation. It includes parallel processing
of individual symbols and cross-sectional feature computation.
"""
import logging
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Feature module imports with fallback for both relative and absolute imports
try:
    # Try relative imports first (when run as module)
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
    from ..features.weekly import add_weekly_features_to_daily
    from ..features.lagging import apply_configurable_lags
except ImportError:
    # Fallback to absolute imports (when run directly)
    from src.features.assemble import assemble_indicators_from_wide
    from src.features.trend import add_trend_features, add_rsi_features, add_macd_features
    from src.features.volatility import add_multiscale_vol_regime, add_vol_regime_cs_context
    from src.features.hurst import add_hurst_features
    from src.features.distance import add_distance_to_ma_features
    from src.features.range_breakout import add_range_breakout_features
    from src.features.volume import add_volume_features, add_volume_shock_features
    from src.features.relstrength import add_relative_strength
    from src.features.alpha import add_alpha_momentum_features
    from src.features.breadth import add_breadth_series
    from src.features.xsec import add_xsec_momentum_panel
    from src.features.postprocessing import interpolate_internal_gaps
    from src.features.ohlc_adjustment import adjust_ohlc_to_adjclose
    from src.features.sector_mapping import build_enhanced_sector_mappings, get_required_etfs
    from src.features.target_generation import generate_targets_parallel
    from src.features.weekly import add_weekly_features_to_daily
    from src.features.lagging import apply_configurable_lags

logger = logging.getLogger(__name__)

# Global storage for profiling data
_pipeline_timings = []
_profiling_enabled = True


@contextmanager
def profile_stage(stage_name: str):
    """
    Context manager for timing pipeline stages.
    
    Args:
        stage_name: Name of the pipeline stage for logging and reporting
        
    Yields:
        None (context manager)
    """
    if not _profiling_enabled:
        # Just yield without timing if profiling is disabled
        yield
        return
        
    start_time = time.time()
    start_timestamp = datetime.now()
    logger.info(f"⏱️  Starting {stage_name}...")
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        end_timestamp = datetime.now()
        logger.info(f"✅ Completed {stage_name} in {elapsed:.2f}s")
        
        # Store timing data for summary
        _pipeline_timings.append({
            'stage': stage_name,
            'wall_time': elapsed,
            'start': start_timestamp,
            'end': end_timestamp
        })


def _log_profiling_summary():
    """Log a summary of all pipeline stage timings."""
    if not _pipeline_timings:
        logger.info("No profiling data available")
        return
    
    total_time = sum(timing['wall_time'] for timing in _pipeline_timings)
    
    logger.info("📊 Pipeline Timing Summary:")
    logger.info("  " + "─" * 50)
    
    for timing in _pipeline_timings:
        stage_name = timing['stage']
        wall_time = timing['wall_time']
        percentage = (wall_time / total_time * 100) if total_time > 0 else 0
        logger.info(f"  {stage_name:<30} {wall_time:>7.2f}s ({percentage:>5.1f}%)")
    
    logger.info("  " + "─" * 50)
    logger.info(f"  {'Total Pipeline':<30} {total_time:>7.2f}s (100.0%)")


def _clear_profiling_data():
    """Clear stored profiling data for a fresh pipeline run."""
    global _pipeline_timings
    _pipeline_timings = []


def _set_profiling_enabled(enabled: bool):
    """Set global profiling enabled state."""
    global _profiling_enabled
    _profiling_enabled = enabled


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
    sp500_tickers: List[str] = None,
    interpolation_n_jobs: int = -1,
    triple_barrier_config: Dict[str, float] = None,
    daily_lags: List[int] = None,
    weekly_lags: List[int] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Dict]]:
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
        interpolation_n_jobs: Number of parallel jobs for NaN interpolation (-1 for all cores)
        triple_barrier_config: Configuration for triple barrier target generation
        daily_lags: List of daily lag periods in trading days (None for no daily lags)
        weekly_lags: List of weekly lag periods in weeks (None for no weekly lags)
        
    Returns:
        Tuple of (features_dict, targets_dataframe, enhanced_mappings) where:
        - features_dict: Dictionary mapping symbol -> DataFrame with all computed features  
        - targets_dataframe: DataFrame with triple barrier targets
        - enhanced_mappings: Enhanced sector/subsector mappings dictionary
        
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
    with profile_stage("Data Loading"):
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
    with profile_stage("Data Assembly"):
        logger.info("Assembling indicators from wide data")
        data = {
            k: pd.concat([stocks.get(k, pd.DataFrame()),
                          etfs.get(k, pd.DataFrame())], axis=1).sort_index()
            for k in (set(stocks) | set(etfs))
        }
        indicators_by_symbol = assemble_indicators_from_wide(data, adjust_ohlc_with_factor=False)

    # 2.5) Enhanced sector/subsector mapping - moved here to be available for both daily and weekly features
    with profile_stage("Enhanced Sector Mappings"):
        try:
            from ...data.loader import _discover_universe_csv
        except ImportError:
            from src.data.loader import _discover_universe_csv
        
        logger.info("Building enhanced sector/subsector mappings...")
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
        
        logger.info(f"Enhanced mappings completed: {len(enhanced_mappings)} symbols mapped")

    # 3) Core features (parallelized)
    with profile_stage("Core Feature Computation"):
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

    # 4-8) Cross-sectional features (grouped for profiling)
    with profile_stage("Cross-Sectional Features"):
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
    with profile_stage("Triple Barrier Target Generation"):
        logger.info("Generating triple barrier targets...")
        targets_df = _generate_triple_barrier_targets(indicators_by_symbol, triple_barrier_config)
    
    # 10) Interpolate internal gaps (NaNs between observed values only) - parallelized
    with profile_stage("NaN Interpolation"):
        logger.info(f"Starting NaN interpolation (n_jobs={interpolation_n_jobs})...")
        indicators_by_symbol = interpolate_internal_gaps(
            indicators_by_symbol, 
            n_jobs=interpolation_n_jobs,
            batch_size=32  # Process 32 symbols per batch
        )

    # 11) Apply configurable feature lags (optional)
    if (daily_lags and len(daily_lags) > 0) or (weekly_lags and len(weekly_lags) > 0):
        with profile_stage("Feature Lag Application"):
            logger.info("Applying configurable feature lags...")
            indicators_by_symbol = apply_configurable_lags(
                indicators_by_symbol,
                daily_lags=daily_lags,
                weekly_lags=weekly_lags,
                n_jobs=max(1, (os.cpu_count() or 4) - 1)
            )

    logger.info("Feature universe construction completed")
    return indicators_by_symbol, targets_df, enhanced_mappings


def _generate_triple_barrier_targets(indicators_by_symbol: Dict[str, pd.DataFrame], config: Dict[str, float] = None) -> pd.DataFrame:
    """
    Convert indicators to long format and generate triple barrier targets in parallel.
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames with features
        config: Triple barrier configuration dictionary
        
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
    
    # Triple barrier configuration (use provided config or defaults)
    if config is None:
        config = {
            'up_mult': 3.0,
            'dn_mult': 1.5,  # Updated default
            'max_horizon': 20,
            'start_every': 3,
        }
    
    logger.info(f"Triple barrier config: up_mult={config['up_mult']}, dn_mult={config['dn_mult']}, "
                f"max_horizon={config['max_horizon']}, start_every={config['start_every']}")
    
    # Generate targets in parallel
    targets_df = generate_targets_parallel(df_long, config, n_jobs=-1, chunk_size=32)
    
    if not targets_df.empty:
        logger.info(f"Generated {len(targets_df)} triple barrier targets")
        # Log class distribution
        _log_target_class_distribution(targets_df)
    else:
        logger.warning("No triple barrier targets generated")
    
    return targets_df


def _log_target_class_distribution(targets_df: pd.DataFrame) -> None:
    """Log distribution of triple barrier target classes."""
    if targets_df.empty:
        logger.info("No targets generated for class distribution analysis")
        return
    
    hit_counts = targets_df['hit'].value_counts().sort_index()
    total_targets = len(targets_df)
    
    logger.info("Triple Barrier Target Class Distribution:")
    logger.info(f"  Total targets: {total_targets:,}")
    
    for hit_value in [-1, 0, 1]:
        count = hit_counts.get(hit_value, 0)
        percentage = (count / total_targets * 100) if total_targets > 0 else 0
        class_name = {-1: "Lower barrier hit", 0: "Time expired", 1: "Upper barrier hit"}[hit_value]
        logger.info(f"  {class_name}: {count:,} ({percentage:.1f}%)")


def run_pipeline(
    max_stocks: Optional[int] = None,
    rate_limit: float = 1.0,
    interval: str = "1d",
    default_etfs: List[str] = None,
    spy_symbol: str = "SPY",
    sector_to_etf: Dict[str, str] = None,
    sp500_tickers: List[str] = None,
    output_dir: Path = Path("./artifacts"),
    include_sectors: bool = True,
    include_weekly: bool = True,
    interpolation_n_jobs: int = -1,
    triple_barrier_config: Dict[str, float] = None,
    enable_profiling: bool = True,
    daily_lags: List[int] = None,
    weekly_lags: List[int] = None
) -> None:
    """
    Run the complete feature computation pipeline and save outputs.
    
    This function orchestrates the entire process:
    1. Load stock and ETF data
    2. Compute all daily features in parallel
    3. Add cross-sectional features
    4. Generate triple barrier targets
    5. Apply configurable daily and weekly feature lags (optional)
    6. Add comprehensive weekly features (optional)
    7. Save feature and target files
    
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
        include_weekly: Whether to add comprehensive weekly features (default: True)
        interpolation_n_jobs: Number of parallel jobs for NaN interpolation (-1 for all cores)
        triple_barrier_config: Configuration for triple barrier target generation
        enable_profiling: Whether to enable pipeline stage profiling (default: True)
        daily_lags: List of daily lag periods in trading days (None for no daily lags)
        weekly_lags: List of weekly lag periods in weeks (None for no weekly lags)
    """
    # Set profiling state and clear any previous profiling data
    _set_profiling_enabled(enable_profiling)
    if enable_profiling:
        _clear_profiling_data()
    
    logger.info("Starting feature computation pipeline")
    
    # Set up output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build feature universe (now returns features, targets, and enhanced mappings)
    indicators_by_symbol, targets_df, enhanced_mappings = build_feature_universe(
        max_stocks=max_stocks,
        rate_limit=rate_limit,
        interval=interval,
        default_etfs=default_etfs,
        spy_symbol=spy_symbol,
        sector_to_etf=sector_to_etf,
        sp500_tickers=sp500_tickers,
        interpolation_n_jobs=interpolation_n_jobs,
        triple_barrier_config=triple_barrier_config,
        daily_lags=daily_lags,
        weekly_lags=weekly_lags
    )

    # Save outputs (import saving functions)
    try:
        from ..io.saving import save_long_parquet
    except ImportError:
        # If relative import fails, try absolute import
        from src.io.saving import save_long_parquet

    logger.info("Saving output files...")
    
    # Save basic daily features
    daily_features_path = output_dir / "features_daily.parquet"
    save_long_parquet(indicators_by_symbol, out_path=daily_features_path)
    logger.info(f"Saved daily features to {daily_features_path}")
    
    # Add comprehensive weekly features if requested
    final_features = None
    if include_weekly:
        with profile_stage("Weekly Features Computation"):
            logger.info("Adding comprehensive weekly features...")
            try:
                # Load the daily features we just saved
                from ..io.saving import load_long_parquet
                daily_df = load_long_parquet(daily_features_path)
                
                # Add weekly features using the enhanced mappings computed earlier
                final_features = add_weekly_features_to_daily(
                    daily_df,
                    spy_symbol=spy_symbol,
                    n_jobs=-1,
                    enhanced_mappings=enhanced_mappings
                )
                
                weekly_count = len([col for col in final_features.columns if col.startswith('w_')])
                logger.info(f"Added {weekly_count} weekly features")
                
            except Exception as e:
                logger.error(f"Error adding weekly features: {e}")
                logger.info("Continuing with daily features only...")
    
    # File I/O operations (group remaining saves together)  
    with profile_stage("File I/O Operations"):
        # Save complete feature set if weekly features were added
        if final_features is not None:
            complete_features_path = output_dir / "features_complete.parquet"
            final_features.to_parquet(complete_features_path, index=False)
            logger.info(f"Saved complete feature set to {complete_features_path}")
            
            # Save in long format for legacy compatibility
            save_long_parquet(indicators_by_symbol, out_path=output_dir / "features_long.parquet")
        
        # Save triple barrier targets if generated
        if not targets_df.empty:
            targets_df.to_parquet(output_dir / "targets_triple_barrier.parquet", index=False)
            logger.info(f"Saved {len(targets_df)} triple barrier targets")
            # Log final class distribution summary
            _log_target_class_distribution(targets_df)
    
    # Summary
    if include_weekly and final_features is not None:
        daily_feature_count = len([col for col in final_features.columns 
                                 if not col.startswith('w_') and col not in ['symbol', 'date']])
        weekly_feature_count = len([col for col in final_features.columns if col.startswith('w_')])
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"  - Daily features: {daily_feature_count}")
        logger.info(f"  - Weekly features: {weekly_feature_count}")
        logger.info(f"  - Total features: {daily_feature_count + weekly_feature_count}")
        logger.info(f"  - Symbols processed: {final_features['symbol'].nunique()}")
        logger.info(f"  - Outputs saved to {output_dir}")
    else:
        logger.info(f"Pipeline completed. Daily features and targets saved to {output_dir}")
    
    # Log profiling summary if enabled
    if enable_profiling:
        _log_profiling_summary()


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


if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive financial feature computation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--max-stocks", 
        type=int, 
        default=None,
        help="Maximum number of stocks to process (None for all)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts",
        help="Output directory for feature and target files"
    )
    
    parser.add_argument(
        "--no-weekly",
        action="store_true",
        help="Skip weekly features computation (faster but less comprehensive)"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Rate limit for data requests (requests per second)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting financial features pipeline...")
        logger.info(f"Max stocks: {args.max_stocks or 'All'}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Weekly features: {'No' if args.no_weekly else 'Yes'}")
        
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
            "retail trade": "XLY",
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
        
        # Run pipeline with command-line arguments
        run_pipeline(
            max_stocks=args.max_stocks,
            rate_limit=args.rate_limit,
            interval="1d",
            spy_symbol="SPY",
            sector_to_etf=sector_to_etf,
            sp500_tickers=SP500_TICKERS,
            output_dir=Path(args.output_dir),
            include_weekly=not args.no_weekly
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Check {args.output_dir} for output files:")
        logger.info("  - features_daily.parquet: Daily features only")
        if not args.no_weekly:
            logger.info("  - features_complete.parquet: Daily + weekly features")
        logger.info("  - targets_triple_barrier.parquet: Triple barrier targets")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)