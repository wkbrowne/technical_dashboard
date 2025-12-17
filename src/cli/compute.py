#!/usr/bin/env python3
"""
CLI for computing features from cached data.

This module provides commands for computing features across different timeframes.
It uses cached data from the download CLI, allowing you to iterate on feature
computation without re-downloading data.

Usage:
    python -m src.cli.compute --config config/features.yaml
    python -m src.cli.compute --config config/features_minimal.yaml --timeframes D
    python -m src.cli.compute --output artifacts/ --no-targets
    python -m src.cli.compute --timeframes D,W,M --max-stocks 50
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# Force unbuffered output for progress visibility (especially when piped to tee/file)
# This ensures print() and logger output is flushed immediately
if not sys.stdout.line_buffering:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        # Python < 3.7 fallback
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file if present
# This allows storing API keys (FRED_API_KEY, RAPIDAPI_KEY) in .env
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Ensure handler is reconfigured with line buffering
)
logger = logging.getLogger(__name__)

# Ensure all logging handlers flush immediately
for handler in logging.root.handlers:
    if hasattr(handler, 'stream'):
        handler.stream = sys.stderr  # Use line-buffered stderr


def load_cached_data(cache_dir: Path) -> dict:
    """Load cached stock data from parquet files.

    Args:
        cache_dir: Directory containing cached data

    Returns:
        Dict with 'stocks' and 'etfs' DataFrames
    """
    import pandas as pd

    data = {}

    # Load stock data - prefer newer file (combined vs universe)
    combined_file = cache_dir / "stock_data_combined.parquet"
    universe_file = cache_dir / "stock_data_universe.parquet"

    # Determine which file to use based on modification time
    stock_file = None
    if combined_file.exists() and universe_file.exists():
        # Prefer whichever is newer
        if combined_file.stat().st_mtime > universe_file.stat().st_mtime:
            stock_file = combined_file
            logger.info(f"Using combined cache (newer than universe)")
        else:
            stock_file = universe_file
            logger.info(f"Using universe cache (newer than combined)")
    elif combined_file.exists():
        stock_file = combined_file
    elif universe_file.exists():
        stock_file = universe_file

    if stock_file and stock_file.exists():
        data['stocks'] = pd.read_parquet(stock_file)
        logger.info(f"Loaded {len(data['stocks'])} stock records from {stock_file.name}")
    else:
        logger.warning("No stock data found in cache")
        data['stocks'] = pd.DataFrame()

    # Load ETF data (includes VIX/VXN if downloaded via load_etf_universe)
    etf_file = cache_dir / "stock_data_etf.parquet"
    if etf_file.exists():
        data['etfs'] = pd.read_parquet(etf_file)
        logger.info(f"Loaded {len(data['etfs'])} ETF records")

        # Check if VIX is present
        if 'symbol' in data['etfs'].columns:
            etf_symbols = data['etfs']['symbol'].unique()
            has_vix = '^VIX' in etf_symbols or 'VIX' in etf_symbols
            if not has_vix:
                logger.warning("VIX not found in ETF cache. Run 'python -m src.cli.download --universe etf' to download VIX data.")
    else:
        data['etfs'] = pd.DataFrame()

    return data


def _pivot_long_to_wide(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pivot long-format data (date, symbol, value, metric) to wide format per metric.

    Args:
        df: Long-format DataFrame with columns: date, symbol, value, metric

    Returns:
        Dict mapping metric name (e.g., 'Open', 'AdjClose') -> wide DataFrame (dates x symbols)
    """
    if df.empty:
        return {}

    # Check if it's already in wide format (no 'metric' column)
    if 'metric' not in df.columns:
        logger.debug("Data appears to already be in wide format")
        return {}

    result = {}
    for metric, chunk in df.groupby('metric'):
        wide = chunk.pivot(index='date', columns='symbol', values='value').sort_index()
        result[str(metric)] = wide

    return result


def prepare_indicators_from_cached(data: Dict) -> Dict[str, pd.DataFrame]:
    """Convert cached long-format data to indicators_by_symbol dict.

    Args:
        data: Dict with 'stocks' and 'etfs' DataFrames in long format
              (columns: date, symbol, value, metric)

    Returns:
        Dict mapping symbol -> DataFrame with date index and columns:
        open, high, low, close, adjclose, volume, ret
    """
    from src.features.assemble import assemble_indicators_from_wide

    # Convert long format to wide format per metric
    stocks_wide = {}
    etfs_wide = {}

    if 'stocks' in data and not data['stocks'].empty:
        stocks_wide = _pivot_long_to_wide(data['stocks'])
        logger.debug(f"Pivoted stocks data to wide format: {list(stocks_wide.keys())}")

    if 'etfs' in data and not data['etfs'].empty:
        etfs_wide = _pivot_long_to_wide(data['etfs'])
        logger.debug(f"Pivoted ETFs data to wide format: {list(etfs_wide.keys())}")

    # Combine stocks and ETFs into single wide dict
    # Concatenate symbols from both sources for each metric
    combined_wide = {}
    all_metrics = set(stocks_wide.keys()) | set(etfs_wide.keys())

    for metric in all_metrics:
        dfs_to_concat = []
        if metric in stocks_wide:
            dfs_to_concat.append(stocks_wide[metric])
        if metric in etfs_wide:
            dfs_to_concat.append(etfs_wide[metric])

        if dfs_to_concat:
            combined_wide[metric] = pd.concat(dfs_to_concat, axis=1).sort_index()

    if not combined_wide:
        logger.warning("No data to process after pivoting")
        return {}

    # Use assemble_indicators_from_wide to convert to per-symbol DataFrames
    # with proper lowercase column names and returns calculation
    indicators = assemble_indicators_from_wide(combined_wide, adjust_ohlc_with_factor=True)

    return indicators


def _discover_universe_csv(cache_dir: Path) -> Optional[str]:
    """Find the universe CSV file in cache directory.

    Args:
        cache_dir: Directory containing cached data

    Returns:
        Path to universe CSV or None if not found
    """
    candidates = [f for f in cache_dir.iterdir()
                  if f.name.startswith("US universe_") and f.suffix == ".csv"]
    if not candidates:
        return None
    return str(sorted(candidates)[0])


def _load_sectors_from_cache(cache_dir: Path, max_stocks: Optional[int] = None) -> Dict[str, str]:
    """Load sector mappings from universe CSV in cache directory.

    Args:
        cache_dir: Directory containing cached data and universe CSV
        max_stocks: Optional limit on number of stocks

    Returns:
        Dict mapping symbol -> sector name
    """
    try:
        universe_csv = _discover_universe_csv(cache_dir)
        if not universe_csv:
            logger.warning("No universe CSV found in cache directory")
            return {}

        df = pd.read_csv(universe_csv)

        # Find symbol and sector columns
        sym_col = "Symbol" if "Symbol" in df.columns else next(
            (c for c in df.columns if "symbol" in c.lower()), None)
        sec_col = "Sector" if "Sector" in df.columns else next(
            (c for c in df.columns if "sector" in c.lower()), None)

        if not sym_col or not sec_col:
            logger.warning(f"No Symbol/Sector columns in {universe_csv}")
            return {}

        # Build symbol -> sector mapping
        result = (df[[sym_col, sec_col]]
                  .dropna(subset=[sym_col])
                  .assign(**{sym_col: df[sym_col].astype(str).str.strip(),
                            sec_col: df[sec_col].astype(str).str.strip()}))

        if max_stocks and max_stocks > 0:
            result = result.iloc[:max_stocks]

        sectors = dict(zip(result[sym_col], result[sec_col]))
        logger.info(f"Loaded sector mappings for {len(sectors)} symbols")
        return sectors

    except Exception as e:
        logger.warning(f"Could not load sector mappings: {e}")
        return {}


def _get_default_sector_to_etf() -> Dict[str, str]:
    """Return default mapping of sector names to ETF symbols."""
    return {
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
        "transportation": "XLI",
        "commercial services": "XLC",
        "process industries": "XLB",
        "communications": "XLC",
        "health services": "XLV",
        "distribution services": "XLI",
        "miscellaneous": "SPY",
    }


def run_feature_pipeline(
    config_path: Optional[str],
    output_dir: Path,
    cache_dir: Path,
    timeframes: List[str],
    include_targets: bool = True,
    max_stocks: Optional[int] = None,
    n_jobs: int = -1,
    batch_size: int = 16,
    full_output: bool = False
):
    """Run the feature computation pipeline.

    Args:
        config_path: Path to features.yaml config file
        output_dir: Directory to save output
        cache_dir: Directory with cached data
        timeframes: List of timeframes to compute (D, W, M)
        include_targets: Whether to compute triple barrier targets
        max_stocks: Maximum number of stocks to process
        n_jobs: Number of parallel jobs
        batch_size: Number of symbols per parallel batch
        full_output: If True, output all computed features. If False (default),
            filter to curated feature set (~200 features)
    """
    from src.config.features import FeatureConfig, Timeframe
    from src.config.parallel import ParallelConfig
    from src.pipelines.orchestrator import run_pipeline_v2, compute_higher_timeframe_features
    from src.features.single_stock import compute_single_stock_features
    from src.features.cross_sectional import compute_cross_sectional_features
    from src.features.postprocessing import interpolate_internal_gaps
    from src.features.timeframe import combine_to_long, partition_by_symbol
    from src.features.sector_mapping import build_enhanced_sector_mappings
    from joblib import Parallel, delayed

    # Create unified parallel config
    parallel_config = ParallelConfig(n_jobs=n_jobs, batch_size=batch_size)
    logger.info(f"Parallel config: {parallel_config.summary()}")

    start_time = time.time()

    # Load configuration
    if config_path:
        config = FeatureConfig.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = FeatureConfig.default()
        logger.info("Using default feature configuration")

    logger.info(config.summary())

    # Parse timeframes
    tf_map = {'D': Timeframe.DAILY, 'W': Timeframe.WEEKLY, 'M': Timeframe.MONTHLY}
    active_timeframes = [tf_map[t] for t in timeframes if t in tf_map]
    logger.info(f"Active timeframes: {[t.value for t in active_timeframes]}")

    # Load cached data
    data = load_cached_data(cache_dir)
    if data['stocks'].empty:
        logger.error("No stock data available. Run download first.")
        sys.exit(1)

    # Apply max_stocks limit
    if max_stocks:
        symbols = data['stocks']['symbol'].unique()[:max_stocks]
        data['stocks'] = data['stocks'][data['stocks']['symbol'].isin(symbols)]
        logger.info(f"Limited to {max_stocks} stocks")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sector mappings for cross-sectional features
    sectors = _load_sectors_from_cache(cache_dir, max_stocks)
    sector_to_etf = _get_default_sector_to_etf()

    # Load SP500 tickers for breadth features
    try:
        from cache.sp500_list import SP500_TICKERS
        sp500_tickers = SP500_TICKERS
        logger.info(f"Loaded {len(sp500_tickers)} S&P 500 tickers for breadth features")
    except ImportError:
        sp500_tickers = None
        logger.warning("Could not load S&P 500 ticker list for breadth features")

    try:
        # Convert cached data to indicators dict
        logger.info("Preparing indicators from cached data...")
        indicators_by_symbol = prepare_indicators_from_cached(data)

        if not indicators_by_symbol:
            logger.error("No indicators prepared from cached data")
            sys.exit(1)

        logger.info(f"Prepared {len(indicators_by_symbol)} symbols for feature computation")

        # Build enhanced sector/subsector mappings for relative strength
        enhanced_mappings = None
        if sectors:
            try:
                universe_csv = _discover_universe_csv(cache_dir)

                if universe_csv:
                    # Get ALL ETF data from indicators (includes sector + subsector ETFs)
                    # The ETF parquet should contain all ETFs loaded by prepare_indicators_from_cached
                    # Filter to just ETF-like symbols (typically 2-5 letter uppercase, no numbers)
                    from src.features.sector_mapping import SUBSECTOR_ETF_KEYWORDS
                    all_known_etfs = (
                        set(sector_to_etf.values()) |  # Sector ETFs
                        set(SUBSECTOR_ETF_KEYWORDS.keys()) |  # Subsector ETFs
                        {'SPY', 'RSP', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'}  # Common benchmarks
                    )
                    etf_data = {sym: df for sym, df in indicators_by_symbol.items()
                                if sym in all_known_etfs}
                    logger.info(f"Found {len(etf_data)} ETFs for enhanced mappings: {sorted(etf_data.keys())[:10]}...")

                    enhanced_mappings = build_enhanced_sector_mappings(
                        universe_csv=universe_csv,
                        stock_data=indicators_by_symbol,
                        etf_data=etf_data,
                        base_sectors=sectors
                    )
                    logger.info(f"Built enhanced mappings for {len(enhanced_mappings)} symbols")
                else:
                    logger.warning("No universe CSV found for enhanced mappings")
            except Exception as e:
                logger.warning(f"Could not build enhanced mappings: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                enhanced_mappings = None

        # Use the new pipeline_v2 for feature computation
        logger.info("Running feature pipeline v2...")
        features_df, targets_df, features_complete, features_filtered = run_pipeline_v2(
            indicators_by_symbol=indicators_by_symbol,
            feature_config=config,
            timeframes=timeframes,
            include_targets=include_targets,
            parallel_config=parallel_config,
            sectors=sectors,
            sector_to_etf=sector_to_etf,
            enhanced_mappings=enhanced_mappings,
            sp500_tickers=sp500_tickers,
            full_output=full_output
        )

        # Save BOTH complete and filtered feature files
        # 1. Complete file with ALL features (~600+)
        complete_output = output_dir / "features_complete.parquet"
        features_complete.to_parquet(complete_output, index=False)
        logger.info(f"Saved complete features ({len(features_complete.columns)} cols) to {complete_output}")

        # 2. Filtered file with curated ML-ready features (~250)
        filtered_output = output_dir / "features_filtered.parquet"
        features_filtered.to_parquet(filtered_output, index=False)
        logger.info(f"Saved filtered features ({len(features_filtered.columns)} cols) to {filtered_output}")

        # Count features by timeframe
        daily_features = [c for c in features_df.columns
                         if not c.startswith(('w_', 'm_')) and c not in ['symbol', 'date']]
        weekly_features = [c for c in features_df.columns if c.startswith('w_')]
        monthly_features = [c for c in features_df.columns if c.startswith('m_')]

        # Save targets if generated
        if targets_df is not None and not targets_df.empty:
            targets_output = output_dir / "targets_triple_barrier.parquet"
            targets_df.to_parquet(targets_output, index=False)
            logger.info(f"Saved {len(targets_df)} targets to {targets_output}")

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.1f}s")

        # Print summary
        print("\n" + "=" * 50)
        print("Feature Computation Summary")
        print("=" * 50)
        print(f"Symbols processed:  {features_df['symbol'].nunique()}")
        print(f"Total rows:         {len(features_df):,}")

        # Get date range from features DataFrame
        if 'date' in features_df.columns:
            date_min = features_df['date'].min()
            date_max = features_df['date'].max()
            print(f"Date range:         {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
        elif features_df.index.name == 'date' or hasattr(features_df.index, 'min'):
            date_min = features_df.index.min()
            date_max = features_df.index.max()
            if hasattr(date_min, 'strftime'):
                print(f"Date range:         {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")

        print(f"Daily features:     {len(daily_features)}")
        if weekly_features:
            print(f"Weekly features:    {len(weekly_features)}")
        if monthly_features:
            print(f"Monthly features:   {len(monthly_features)}")
        print(f"Total features:     {len(daily_features) + len(weekly_features) + len(monthly_features)}")
        if targets_df is not None and not targets_df.empty:
            print(f"Targets generated:  {len(targets_df):,}")
        print("")
        print("Output Files:")
        print(f"  features_complete.parquet:  {len(features_complete.columns)} cols (ALL features)")
        print(f"  features_filtered.parquet:  {len(features_filtered.columns)} cols (ML-ready)")
        if targets_df is not None and not targets_df.empty:
            print(f"  targets_triple_barrier.parquet")
        print(f"Output directory:   {output_dir}")
        print(f"Total time:         {elapsed:.1f}s")
        print("=" * 50)

    except ImportError as e:
        logger.error(f"Could not import pipeline components: {e}")
        logger.error("Make sure you're running from the project root")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from cached market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute with default config
    python -m src.cli.compute

    # Compute with custom config
    python -m src.cli.compute --config config/features.yaml

    # Compute minimal features for fast iteration
    python -m src.cli.compute --config config/features_minimal.yaml

    # Daily only (fastest)
    python -m src.cli.compute --timeframes D

    # Skip target generation
    python -m src.cli.compute --no-targets

    # Limit to 50 stocks for testing
    python -m src.cli.compute --max-stocks 50
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to features.yaml config file (default: use built-in defaults)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="artifacts/",
        help="Output directory (default: artifacts/)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="cache/",
        help="Cache directory with downloaded data (default: cache/)"
    )
    parser.add_argument(
        "--timeframes", "-t",
        type=str,
        default="D,W,M",
        help="Comma-separated timeframes: D,W,M (default: D,W,M)"
    )
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="Skip triple barrier target generation"
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Maximum number of stocks to process"
    )
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 for all cores)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Symbols per parallel batch (default: 16). Larger = less IPC overhead, more memory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Output all computed features (default: curated subset ~200 features)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse timeframes
    timeframes = [t.strip().upper() for t in args.timeframes.split(",")]
    valid_tf = {'D', 'W', 'M'}
    invalid = [t for t in timeframes if t not in valid_tf]
    if invalid:
        logger.error(f"Invalid timeframes: {invalid}. Use D, W, or M.")
        sys.exit(1)

    run_feature_pipeline(
        config_path=args.config,
        output_dir=Path(args.output),
        cache_dir=Path(args.cache),
        timeframes=timeframes,
        include_targets=not args.no_targets,
        max_stocks=args.max_stocks,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        full_output=args.full_output
    )


if __name__ == "__main__":
    main()
