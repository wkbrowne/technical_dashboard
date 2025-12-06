"""
Main orchestration pipeline for feature computation.

This module contains the high-level workflow that coordinates data loading,
feature computation, and output generation. It includes parallel processing
of individual symbols and cross-sectional feature computation.

Refactored to use:
- FeatureConfig for feature selection and toggling
- TimeframeResampler for unified D/W/M handling
- Feature registry for dynamic feature instantiation
"""
import logging
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Feature module imports with fallback for both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from ..features.assemble import assemble_indicators_from_wide
    from ..features.single_stock import compute_single_stock_features
    from ..features.cross_sectional import compute_cross_sectional_features
    from ..features.postprocessing import interpolate_internal_gaps, drop_rows_with_excessive_nans
    from ..features.ohlc_adjustment import adjust_ohlc_to_adjclose
    from ..features.sector_mapping import build_enhanced_sector_mappings, get_required_etfs
    from ..features.target_generation import generate_targets_parallel
    from ..features.lagging import apply_configurable_lags
    # New unified timeframe handler
    from ..features.timeframe import (
        TimeframeResampler, TimeframeType, TimeframeConfig,
        partition_by_symbol, combine_to_long, compute_features_at_timeframe,
        add_prefix_to_features
    )
    # Config system
    from ..config.features import FeatureConfig, FeatureSpec, Timeframe
    from ..config.parallel import ParallelConfig
except ImportError:
    # Fallback to absolute imports (when run directly)
    from src.features.assemble import assemble_indicators_from_wide
    from src.features.single_stock import compute_single_stock_features
    from src.features.cross_sectional import compute_cross_sectional_features
    from src.features.postprocessing import interpolate_internal_gaps, drop_rows_with_excessive_nans
    from src.features.ohlc_adjustment import adjust_ohlc_to_adjclose
    from src.features.sector_mapping import build_enhanced_sector_mappings, get_required_etfs
    from src.features.target_generation import generate_targets_parallel
    from src.features.lagging import apply_configurable_lags
    # New unified timeframe handler
    from src.features.timeframe import (
        TimeframeResampler, TimeframeType, TimeframeConfig,
        partition_by_symbol, combine_to_long, compute_features_at_timeframe,
        add_prefix_to_features
    )
    # Config system
    from src.config.features import FeatureConfig, FeatureSpec, Timeframe
    from src.config.parallel import ParallelConfig

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


def _feature_worker(sym: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Core feature computation worker for a single symbol.

    This function runs the complete single-stock feature stack, including:
    - OHLC adjustment to match adjusted close
    - All single-stock features (trend, volatility, distance, range, volume, etc.)

    Note: Cross-sectional features (volatility context, relative strength, etc.)
    are added separately after all symbols are processed.

    Args:
        sym: Symbol ticker
        df: Input DataFrame with OHLCV data

    Returns:
        Tuple of (symbol, enriched_dataframe)
    """
    try:
        logger.debug(f"Processing features for {sym}")

        # First: Adjust OHLC to match adjusted close for consistent price data
        out = adjust_ohlc_to_adjclose(df)

        # Compute all single-stock features using the centralized function
        out = compute_single_stock_features(
            out,
            price_col='adjclose',
            ret_col='ret',
            vol_col='volume',
            ensure_returns=True
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
    weekly_lags: List[int] = None,
    weight_min_clip: float = 0.01,
    weight_max_clip: float = 10.0
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
        weight_min_clip: Minimum weight value for target generation (prevents zero weights)
        weight_max_clip: Maximum weight value for target generation (prevents extreme weights)
        
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

    # Import new pipeline runner
    try:
        from ..features.runner import run_pipeline_parallel
    except ImportError:
        from src.features.runner import run_pipeline_parallel

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

    # 3) Core features (parallelized using new pipeline)
    with profile_stage("Core Feature Computation"):
        logger.info(f"Processing {len(indicators_by_symbol)} symbols in parallel using FeaturePipeline...")
        
        # Convert combined DataFrame back to dict for pipeline if needed, 
        # but run_pipeline_parallel takes a dict.
        
        # Run the new pipeline
        combined_features_df = run_pipeline_parallel(
            indicators_by_symbol,
            config={}, # Pass any config if needed
            n_jobs=max(1, (os.cpu_count() or 4) - 1)
        )
        
        # Convert back to dict for compatibility with downstream steps (cross-sectional, etc.)
        # The new pipeline returns a single combined DataFrame, but existing logic expects a dict.
        # We need to pivot or split it back.
        
        indicators_by_symbol = {}
        if not combined_features_df.empty:
            for sym, group in combined_features_df.groupby('symbol'):
                # Set date as index again
                if 'date' in group.columns:
                    group = group.set_index('date')
                indicators_by_symbol[sym] = group
        
        logger.info(f"Core features completed for {len(indicators_by_symbol)} symbols")

    # 4) Cross-sectional features (all computed together)
    with profile_stage("Cross-Sectional Features"):
        compute_cross_sectional_features(
            indicators_by_symbol,
            sectors=sectors,
            sector_to_etf=sector_to_etf,
            market_symbol=spy_symbol,
            sp500_tickers=sp500_tickers,
            enhanced_mappings=enhanced_mappings,
            beta_win=60,
            alpha_windows=(20, 60, 120),
            alpha_ema_span=10,
            xsec_lookbacks=(5, 20, 60),
            price_col="adjclose"
        )

    # 5) Generate triple barrier targets (parallel)
    with profile_stage("Triple Barrier Target Generation"):
        logger.info("Generating triple barrier targets...")
        targets_df = _generate_triple_barrier_targets(indicators_by_symbol, triple_barrier_config,
                                                     weight_min_clip, weight_max_clip)

    # 6) Interpolate internal gaps (NaNs between observed values only) - parallelized
    with profile_stage("NaN Interpolation"):
        logger.info(f"Starting NaN interpolation (n_jobs={interpolation_n_jobs})...")
        indicators_by_symbol = interpolate_internal_gaps(
            indicators_by_symbol,
            n_jobs=interpolation_n_jobs,
            batch_size=32  # Process 32 symbols per batch
        )

    # 7) Apply configurable feature lags (optional)
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


def _generate_triple_barrier_targets(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    config: Dict[str, float] = None,
    weight_min_clip: float = 0.01,
    weight_max_clip: float = 10.0,
    parallel_config: Optional[ParallelConfig] = None
) -> pd.DataFrame:
    """
    Convert indicators to long format and generate triple barrier targets in parallel.

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames with features
        config: Triple barrier configuration dictionary
        weight_min_clip: Minimum weight value (prevents zero weights)
        weight_max_clip: Maximum weight value (prevents extreme weights)
        parallel_config: ParallelConfig for parallel processing

    Returns:
        DataFrame with triple barrier targets and combined weights
    """
    if parallel_config is None:
        parallel_config = ParallelConfig.default()
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
    
    # Generate targets in parallel with combined weights
    targets_df = generate_targets_parallel(
        df_long, config,
        weight_min_clip=weight_min_clip,
        weight_max_clip=weight_max_clip,
        parallel_config=parallel_config
    )
    
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


def compute_higher_timeframe_features(
    daily_df: pd.DataFrame,
    timeframes: List[str] = None,
    spy_symbol: str = "SPY",
    enhanced_mappings: Optional[Dict] = None,
    n_jobs: int = -1,
    parallel_config: Optional[ParallelConfig] = None
) -> pd.DataFrame:
    """
    Compute features at higher timeframes (weekly/monthly) using unified TimeframeResampler.

    This function replaces the complex fallback chain of weekly implementations with
    a single, clean implementation using the new TimeframeResampler.

    Args:
        daily_df: Long-format daily DataFrame with features
        timeframes: List of timeframes to compute ('W', 'M'). Default: ['W']
        spy_symbol: Market benchmark symbol
        enhanced_mappings: Enhanced sector mappings for relative strength
        n_jobs: Number of parallel jobs (deprecated, use parallel_config)
        parallel_config: ParallelConfig for unified parallel processing

    Returns:
        Daily DataFrame with higher timeframe features merged in
    """
    # Use ParallelConfig if provided
    if parallel_config is None:
        parallel_config = ParallelConfig(n_jobs=n_jobs, batch_size=16)

    if timeframes is None:
        timeframes = ['W']

    logger.info(f"Computing higher timeframe features: {timeframes}")

    # Initialize resampler with ParallelConfig
    resampler = TimeframeResampler(config=parallel_config)

    # Partition daily data by symbol
    daily_by_symbol = partition_by_symbol(daily_df, optimize_dtypes=True)

    result_df = daily_df.copy()

    for tf_str in timeframes:
        tf_type = TimeframeType.WEEKLY if tf_str == 'W' else TimeframeType.MONTHLY
        prefix = 'w_' if tf_str == 'W' else 'm_'

        logger.info(f"Processing {tf_str} timeframe...")

        try:
            # Step 1: Resample to higher timeframe
            higher_tf_data = resampler.resample_symbols(daily_by_symbol, tf_type)

            if not higher_tf_data:
                logger.warning(f"No data resampled for {tf_str} timeframe")
                continue

            # Step 2: Compute features at higher timeframe
            def compute_tf_features(df: pd.DataFrame) -> pd.DataFrame:
                """Compute basic features for higher timeframe."""
                result = df.copy()
                close = pd.to_numeric(result['close'], errors='coerce')

                if len(close.dropna()) < 10:
                    return result

                # RSI
                try:
                    import pandas_ta as ta
                    for period in [14, 21]:
                        rsi = ta.rsi(close, length=period)
                        if rsi is not None:
                            result[f'rsi_{period}'] = rsi.astype('float32')
                except Exception:
                    pass

                # SMAs and distance
                for period in [20, 50]:
                    sma = close.rolling(period, min_periods=period//2).mean()
                    result[f'sma{period}'] = sma.astype('float32')
                    if sma is not None:
                        dist = (close - sma) / sma
                        result[f'dist_sma{period}'] = dist.astype('float32')

                # Trend slope
                for period in [20]:
                    if f'sma{period}' in result.columns:
                        slope = result[f'sma{period}'].diff(4) / result[f'sma{period}'].shift(4)
                        result[f'sma{period}_slope'] = slope.astype('float32')

                return result

            higher_tf_features = compute_features_at_timeframe(
                higher_tf_data,
                compute_tf_features,
                tf_type,
                config=parallel_config
            )

            # Step 3: Add prefix to feature columns
            for symbol, df in higher_tf_features.items():
                higher_tf_features[symbol] = add_prefix_to_features(df, prefix)

            # Step 4: Merge back to daily (need to work with result_df)
            # Convert result_df to dict, merge, then convert back
            result_by_symbol = partition_by_symbol(result_df, optimize_dtypes=False)
            merged_by_symbol = resampler.merge_symbols_to_daily(
                result_by_symbol,
                higher_tf_features,
                tf_type
            )

            # Combine back to long format
            result_df = combine_to_long(merged_by_symbol)

            feature_count = len([c for c in result_df.columns if c.startswith(prefix)])
            logger.info(f"Added {feature_count} {tf_str} features")

        except Exception as e:
            logger.error(f"Failed to compute {tf_str} features: {e}")
            logger.info(f"Continuing without {tf_str} features...")

    return result_df


def run_pipeline_v2(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    feature_config: Optional[FeatureConfig] = None,
    timeframes: List[str] = None,
    spy_symbol: str = "SPY",
    enhanced_mappings: Optional[Dict] = None,
    include_targets: bool = True,
    triple_barrier_config: Optional[Dict] = None,
    n_jobs: int = -1,
    parallel_config: Optional[ParallelConfig] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Simplified pipeline using new config system.

    This is the new recommended entry point for feature computation.
    It uses FeatureConfig for feature selection and TimeframeResampler
    for unified timeframe handling.

    Args:
        indicators_by_symbol: Dict of symbol -> daily OHLCV DataFrame
        feature_config: Feature configuration (None for defaults)
        timeframes: List of timeframes ['D', 'W', 'M'] (default: ['D', 'W'])
        spy_symbol: Market benchmark symbol
        enhanced_mappings: Enhanced sector mappings
        include_targets: Whether to generate triple barrier targets
        triple_barrier_config: Config for target generation
        n_jobs: Number of parallel jobs (deprecated, use parallel_config)
        parallel_config: ParallelConfig for unified parallel processing

    Returns:
        Tuple of (features_df, targets_df)
    """
    # Use ParallelConfig if provided, otherwise create from n_jobs
    if parallel_config is None:
        parallel_config = ParallelConfig(n_jobs=n_jobs, batch_size=16)

    if feature_config is None:
        feature_config = FeatureConfig.default()

    if timeframes is None:
        timeframes = ['D', 'W']

    logger.info(f"Running pipeline v2 with {len(indicators_by_symbol)} symbols")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Parallel config: {parallel_config.summary()}")
    logger.info(feature_config.summary())

    # Get enabled features
    enabled_single = feature_config.get_enabled_features(single_stock_only=True)
    enabled_cs = feature_config.get_enabled_features(cross_sectional_only=True)

    logger.info(f"Enabled single-stock features: {[f.name for f in enabled_single]}")
    logger.info(f"Enabled cross-sectional features: {[f.name for f in enabled_cs]}")

    # Step 1: Compute daily single-stock features (parallel)
    with profile_stage("Daily Single-Stock Features"):
        results = Parallel(
            n_jobs=parallel_config.n_jobs,
            backend=parallel_config.backend,
            batch_size=parallel_config.batch_size,
            verbose=parallel_config.verbose,
            prefer=parallel_config.prefer
        )(
            delayed(_feature_worker)(sym, df)
            for sym, df in indicators_by_symbol.items()
        )
        indicators_by_symbol = dict(results)

    # Step 2: Compute cross-sectional features (sequential - needs full universe)
    if enabled_cs:
        with profile_stage("Cross-Sectional Features"):
            compute_cross_sectional_features(
                indicators_by_symbol,
                enhanced_mappings=enhanced_mappings,
                market_symbol=spy_symbol
            )

    # Step 3: Add higher timeframe features (W/M) - compute alongside daily features
    higher_tfs = [tf for tf in timeframes if tf in ['W', 'M']]
    if higher_tfs:
        with profile_stage(f"Higher Timeframe Features ({','.join(higher_tfs)})"):
            # Convert to long format for higher TF processing
            daily_df = combine_to_long(indicators_by_symbol)
            daily_df = compute_higher_timeframe_features(
                daily_df,
                timeframes=higher_tfs,
                spy_symbol=spy_symbol,
                enhanced_mappings=enhanced_mappings,
                parallel_config=parallel_config
            )
            # Convert back to dict format for subsequent steps
            indicators_by_symbol = partition_by_symbol(daily_df, optimize_dtypes=False)

    # Step 4: Interpolate NaNs (applies to daily AND higher TF features)
    with profile_stage("NaN Interpolation"):
        indicators_by_symbol = interpolate_internal_gaps(
            indicators_by_symbol,
            parallel_config=parallel_config
        )

    # Step 5: Generate targets (after all features computed and interpolated)
    targets_df = None
    if include_targets:
        with profile_stage("Triple Barrier Targets"):
            targets_df = _generate_triple_barrier_targets(
                indicators_by_symbol,
                triple_barrier_config,
                parallel_config=parallel_config
            )

    # Step 6: Convert to final long format
    daily_df = combine_to_long(indicators_by_symbol)

    logger.info(f"Pipeline v2 completed: {len(daily_df)} rows, {len(daily_df.columns)} columns")

    return daily_df, targets_df


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
    include_monthly: bool = False,
    interpolation_n_jobs: int = -1,
    triple_barrier_config: Dict[str, float] = None,
    enable_profiling: bool = True,
    daily_lags: List[int] = None,
    weekly_lags: List[int] = None,
    weight_min_clip: float = 0.01,
    weight_max_clip: float = 10.0,
    feature_config: Optional[FeatureConfig] = None
) -> None:
    """
    Run the complete feature computation pipeline and save outputs.

    This function orchestrates the entire process:
    1. Load stock and ETF data
    2. Compute all daily features in parallel
    3. Add cross-sectional features
    4. Add higher timeframe features (weekly/monthly)
    5. Interpolate NaNs
    6. Generate triple barrier targets
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
        include_monthly: Whether to add monthly features (default: False)
        interpolation_n_jobs: Number of parallel jobs for NaN interpolation (-1 for all cores)
        triple_barrier_config: Configuration for triple barrier target generation
        enable_profiling: Whether to enable pipeline stage profiling (default: True)
        daily_lags: List of daily lag periods in trading days (None for no daily lags)
        weekly_lags: List of weekly lag periods in weeks (None for no weekly lags)
        weight_min_clip: Minimum weight value for target generation (prevents zero weights)
        weight_max_clip: Maximum weight value for target generation (prevents extreme weights)
        feature_config: Optional FeatureConfig for feature selection (None for all features)
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
        weekly_lags=weekly_lags,
        weight_min_clip=weight_min_clip,
        weight_max_clip=weight_max_clip
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
    
    # Add higher timeframe features (weekly/monthly) if requested
    final_features = None
    higher_timeframes = []
    if include_weekly:
        higher_timeframes.append('W')
    if include_monthly:
        higher_timeframes.append('M')

    if higher_timeframes:
        stage_name = f"Higher Timeframe Features ({','.join(higher_timeframes)})"
        with profile_stage(stage_name):
            try:
                # Load the daily features we just saved
                try:
                    from ..io.saving import load_long_parquet
                except ImportError:
                    from src.io.saving import load_long_parquet

                daily_df = load_long_parquet(daily_features_path)
                n_symbols = daily_df['symbol'].nunique()

                logger.info(f"Using unified TimeframeResampler for {higher_timeframes}")
                final_features = compute_higher_timeframe_features(
                    daily_df,
                    timeframes=higher_timeframes,
                    spy_symbol=spy_symbol,
                    enhanced_mappings=enhanced_mappings,
                    n_jobs=interpolation_n_jobs
                )
                feature_counts = {
                    'weekly': len([c for c in final_features.columns if c.startswith('w_')]),
                    'monthly': len([c for c in final_features.columns if c.startswith('m_')])
                }
                logger.info(f"Added features: {feature_counts}")

            except Exception as e:
                logger.error(f"Higher timeframe feature computation failed: {e}")
                logger.info("Continuing with daily features only...")
                final_features = None
    
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
            logger.info(f"Saved {len(targets_df)} triple barrier targets with combined weights")
            
            # Log weight information
            if 'weight_final' in targets_df.columns:
                weight_stats = targets_df['weight_final'].describe()
                logger.info(f"Final weight statistics: min={weight_stats['min']:.4f}, "
                           f"mean={weight_stats['mean']:.4f}, max={weight_stats['max']:.4f}")
                
                # Count columns for comprehensive summary
                weight_cols = [col for col in targets_df.columns if col.startswith('weight_')]
                logger.info(f"Saved weight columns: {weight_cols}")
            
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
        "--monthly",
        action="store_true",
        help="Include monthly timeframe features (m_ prefix)"
    )

    parser.add_argument(
        "--legacy-weekly",
        action="store_true",
        help="Use legacy weekly implementation instead of unified TimeframeResampler"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to features.yaml config file for feature selection"
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Rate limit for data requests (requests per second)"
    )
    
    parser.add_argument(
        "--daily-lag",
        type=int,
        action="append",
        help="Daily lag periods in trading days (can be specified multiple times, e.g., --daily-lag 1 --daily-lag 5)"
    )
    
    parser.add_argument(
        "--weekly-lag",
        type=int,
        action="append",
        help="Weekly lag periods in weeks (can be specified multiple times, e.g., --weekly-lag 1 --weekly-lag 2)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting financial features pipeline...")
        logger.info(f"Max stocks: {args.max_stocks or 'All'}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Weekly features: {'No' if args.no_weekly else 'Yes'}")
        logger.info(f"Monthly features: {'Yes' if args.monthly else 'No'}")
        logger.info(f"Unified timeframe: {'No (legacy)' if args.legacy_weekly else 'Yes'}")
        if args.config:
            logger.info(f"Config file: {args.config}")
        if args.daily_lag:
            logger.info(f"Daily lags: {args.daily_lag}")
        if args.weekly_lag:
            logger.info(f"Weekly lags: {args.weekly_lag}")
        
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
        
        # Load feature config if provided
        feature_config = None
        if args.config:
            feature_config = FeatureConfig.from_yaml(args.config)
            logger.info(feature_config.summary())

        # Run pipeline with command-line arguments
        run_pipeline(
            max_stocks=args.max_stocks,
            rate_limit=args.rate_limit,
            interval="1d",
            spy_symbol="SPY",
            sector_to_etf=sector_to_etf,
            sp500_tickers=SP500_TICKERS,
            output_dir=Path(args.output_dir),
            include_weekly=not args.no_weekly,
            include_monthly=args.monthly,
            daily_lags=args.daily_lag,
            weekly_lags=args.weekly_lag,
            feature_config=feature_config,
            use_unified_timeframe=not args.legacy_weekly
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Check {args.output_dir} for output files:")
        logger.info("  - features_daily.parquet: Daily features only")
        if not args.no_weekly or args.monthly:
            timeframes = []
            if not args.no_weekly:
                timeframes.append("weekly")
            if args.monthly:
                timeframes.append("monthly")
            logger.info(f"  - features_complete.parquet: Daily + {' + '.join(timeframes)} features")
        logger.info("  - targets_triple_barrier.parquet: Triple barrier targets")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)