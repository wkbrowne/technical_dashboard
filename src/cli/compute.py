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
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cached_data(cache_dir: Path) -> dict:
    """Load cached stock data from parquet files.

    Args:
        cache_dir: Directory containing cached data

    Returns:
        Dict with 'stocks' and 'etfs' DataFrames
    """
    import pandas as pd

    data = {}

    # Load stock data
    stock_file = cache_dir / "stock_data_universe.parquet"
    if stock_file.exists():
        data['stocks'] = pd.read_parquet(stock_file)
        logger.info(f"Loaded {len(data['stocks'])} stock records")
    else:
        # Try combined file
        combined_file = cache_dir / "stock_data_combined.parquet"
        if combined_file.exists():
            data['stocks'] = pd.read_parquet(combined_file)
            logger.info(f"Loaded {len(data['stocks'])} stock records from combined file")
        else:
            logger.warning("No stock data found in cache")
            data['stocks'] = pd.DataFrame()

    # Load ETF data
    etf_file = cache_dir / "stock_data_etf.parquet"
    if etf_file.exists():
        data['etfs'] = pd.read_parquet(etf_file)
        logger.info(f"Loaded {len(data['etfs'])} ETF records")
    else:
        data['etfs'] = pd.DataFrame()

    return data


def prepare_indicators_from_cached(data: Dict) -> Dict[str, pd.DataFrame]:
    """Convert cached long-format data to indicators_by_symbol dict.

    Args:
        data: Dict with 'stocks' and 'etfs' DataFrames in long format

    Returns:
        Dict mapping symbol -> DataFrame with date index
    """
    from src.features.assemble import assemble_indicators_from_wide
    from src.features.ohlc_adjustment import adjust_ohlc_to_adjclose

    indicators = {}

    # Process stocks
    if 'stocks' in data and not data['stocks'].empty:
        stock_df = data['stocks']
        if 'symbol' in stock_df.columns:
            for symbol, group in stock_df.groupby('symbol'):
                df = group.copy()
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                df = df.drop(columns=['symbol'], errors='ignore')
                # Adjust OHLC if adjclose is present
                if 'adjclose' in df.columns:
                    df = adjust_ohlc_to_adjclose(df)
                indicators[symbol] = df

    # Process ETFs similarly
    if 'etfs' in data and not data['etfs'].empty:
        etf_df = data['etfs']
        if 'symbol' in etf_df.columns:
            for symbol, group in etf_df.groupby('symbol'):
                df = group.copy()
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                df = df.drop(columns=['symbol'], errors='ignore')
                if 'adjclose' in df.columns:
                    df = adjust_ohlc_to_adjclose(df)
                indicators[symbol] = df

    return indicators


def run_feature_pipeline(
    config_path: Optional[str],
    output_dir: Path,
    cache_dir: Path,
    timeframes: List[str],
    include_targets: bool = True,
    max_stocks: Optional[int] = None,
    n_jobs: int = -1,
    batch_size: int = 16
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
    """
    from src.config.features import FeatureConfig, Timeframe
    from src.config.parallel import ParallelConfig
    from src.pipelines.orchestrator import run_pipeline_v2, compute_higher_timeframe_features
    from src.features.single_stock import compute_single_stock_features
    from src.features.cross_sectional import compute_cross_sectional_features
    from src.features.postprocessing import interpolate_internal_gaps
    from src.features.timeframe import combine_to_long, partition_by_symbol
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

    try:
        # Convert cached data to indicators dict
        logger.info("Preparing indicators from cached data...")
        indicators_by_symbol = prepare_indicators_from_cached(data)

        if not indicators_by_symbol:
            logger.error("No indicators prepared from cached data")
            sys.exit(1)

        logger.info(f"Prepared {len(indicators_by_symbol)} symbols for feature computation")

        # Use the new pipeline_v2 for feature computation
        logger.info("Running feature pipeline v2...")
        features_df, targets_df = run_pipeline_v2(
            indicators_by_symbol=indicators_by_symbol,
            feature_config=config,
            timeframes=timeframes,
            include_targets=include_targets,
            parallel_config=parallel_config
        )

        # Save features
        daily_output = output_dir / "features_daily.parquet"
        features_df.to_parquet(daily_output, index=False)
        logger.info(f"Saved features to {daily_output}")

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
        print(f"Daily features:     {len(daily_features)}")
        if weekly_features:
            print(f"Weekly features:    {len(weekly_features)}")
        if monthly_features:
            print(f"Monthly features:   {len(monthly_features)}")
        print(f"Total features:     {len(daily_features) + len(weekly_features) + len(monthly_features)}")
        if targets_df is not None and not targets_df.empty:
            print(f"Targets generated:  {len(targets_df):,}")
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
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
