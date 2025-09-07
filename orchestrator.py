#!/usr/bin/env python3
"""
Top-level orchestrator script for running the comprehensive financial features pipeline.

This script can be run directly from the project root directory:
    python orchestrator.py

It handles the import path setup and delegates to the actual orchestrator module.
"""

import sys
from pathlib import Path
import argparse
import logging

# Add the project root to Python path so relative imports work
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import the orchestrator module
from src.pipelines.orchestrator import run_pipeline

def main():
    """Main entry point for the pipeline."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
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
    
    parser.add_argument(
        "--interp-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for NaN interpolation (-1 for all cores, 1 for sequential)"
    )
    
    # Triple barrier ATR configuration
    parser.add_argument(
        "--atr-up-mult",
        type=float,
        default=3.0,
        help="Upper barrier ATR multiplier for triple barrier targets (default: 3.0)"
    )
    
    parser.add_argument(
        "--atr-dn-mult", 
        type=float,
        default=1.5,
        help="Lower barrier ATR multiplier for triple barrier targets (default: 1.5)"
    )
    
    parser.add_argument(
        "--atr-horizon",
        type=int,
        default=20,
        help="Maximum days to track triple barrier targets (default: 20)"
    )
    
    parser.add_argument(
        "--atr-start-every",
        type=int,
        default=3,
        help="Days between new triple barrier target starts (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting financial features pipeline...")
        logger.info(f"Max stocks: {args.max_stocks or 'All'}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Weekly features: {'No' if args.no_weekly else 'Yes'}")
        logger.info(f"Interpolation jobs: {args.interp_jobs}")
        logger.info(f"Triple barrier config: up_mult={args.atr_up_mult}, dn_mult={args.atr_dn_mult}, horizon={args.atr_horizon}, start_every={args.atr_start_every}")
        
        # Import SP500 tickers
        try:
            from cache.sp500_list import SP500_TICKERS
        except ImportError:
            logger.warning("Could not import SP500_TICKERS, using empty list")
            SP500_TICKERS = []

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
        
        # Create triple barrier configuration
        triple_barrier_config = {
            'up_mult': args.atr_up_mult,
            'dn_mult': args.atr_dn_mult, 
            'max_horizon': args.atr_horizon,
            'start_every': args.atr_start_every
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
            include_weekly=not args.no_weekly,
            interpolation_n_jobs=args.interp_jobs,
            triple_barrier_config=triple_barrier_config
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Check {args.output_dir} for output files:")
        logger.info("  - features_daily.parquet: Daily features only")
        if not args.no_weekly:
            logger.info("  - features_complete.parquet: Daily + weekly features")
        logger.info("  - targets_triple_barrier.parquet: Triple barrier targets")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()