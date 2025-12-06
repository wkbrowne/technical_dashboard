#!/usr/bin/env python3
"""
CLI for downloading market data.

This module provides commands for downloading stock and ETF data from Yahoo Finance.
It can be run independently from feature computation, allowing you to iterate on
compute logic while a long download runs in another terminal.

Usage:
    python -m src.cli.download --universe sp500 --output cache/
    python -m src.cli.download --symbols AAPL,MSFT,GOOGL
    python -m src.cli.download --symbols-file symbols.txt --rate-limit 2.0
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from time import sleep
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sp500_symbols() -> List[str]:
    """Get S&P 500 symbols from universe CSV."""
    from src.data.loader import _discover_universe_csv, _symbols_from_csv
    from src.config import CACHE_FILE

    try:
        csv_path = _discover_universe_csv(str(CACHE_FILE))
        return _symbols_from_csv(csv_path)
    except FileNotFoundError:
        logger.error("Universe CSV not found. Please ensure US universe_*.csv is in cache/")
        return []


def get_default_etfs() -> List[str]:
    """Get default ETF list for benchmarks."""
    return [
        # Core benchmarks
        "SPY", "QQQ", "IWM", "DIA",
        # Fixed income
        "TLT", "IEF", "HYG", "LQD",
        # Sector cap-weighted
        "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLC", "XLRE",
        # Sector equal-weight
        "RSP", "RYT", "RYF", "RYE", "RGI", "RHS", "RYH", "RYU",
        # International
        "EFA", "EEM", "VEU",
        # Commodities
        "GLD", "SLV", "USO", "UNG",
        # Subsector ETFs
        "SMH", "KBE", "KRE", "IBB", "XBI", "ITA", "XRT", "XHB",
        "SOXX", "IGV", "ITB", "XME", "XOP", "OIH",
    ]


def download_symbols(
    symbols: List[str],
    output_dir: Path,
    rate_limit: float = 1.0,
    interval: str = "1d",
    force_update: bool = False
) -> dict:
    """Download data for a list of symbols.

    Args:
        symbols: List of symbols to download
        output_dir: Directory to save data
        rate_limit: Requests per second
        interval: Data interval (1d, 1h, etc.)
        force_update: If True, re-download even if cached

    Returns:
        Dict with download statistics
    """
    from src.data.loader import get_stock_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "skipped": 0, "total": len(symbols)}
    all_data = {}

    logger.info(f"Downloading {len(symbols)} symbols to {output_dir}")
    logger.info(f"Rate limit: {rate_limit} req/sec, Interval: {interval}")

    for i, symbol in enumerate(symbols, 1):
        # Check cache
        cache_path = output_dir / f"{symbol}.parquet"
        if cache_path.exists() and not force_update:
            stats["skipped"] += 1
            logger.debug(f"[{i}/{len(symbols)}] {symbol}: cached")
            continue

        # Download
        try:
            df = get_stock_data(symbol, interval=interval)
            if df is not None and not df.empty:
                df.to_parquet(cache_path)
                all_data[symbol] = df
                stats["success"] += 1
            else:
                stats["failed"] += 1
                logger.warning(f"[{i}/{len(symbols)}] {symbol}: no data")
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"[{i}/{len(symbols)}] {symbol}: {e}")

        # Rate limiting
        if i < len(symbols):
            sleep(1.0 / rate_limit)

    # Save combined file
    if all_data:
        combined_path = output_dir / "stock_data_combined.parquet"
        import pandas as pd
        combined = pd.concat(all_data.values(), ignore_index=True)
        combined.to_parquet(combined_path)
        logger.info(f"Saved combined data to {combined_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download market data for the technical dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download S&P 500 stocks
    python -m src.cli.download --universe sp500

    # Download specific symbols
    python -m src.cli.download --symbols AAPL,MSFT,GOOGL,AMZN

    # Download from file
    python -m src.cli.download --symbols-file my_symbols.txt

    # Download ETFs only
    python -m src.cli.download --universe etf

    # Force update with custom rate limit
    python -m src.cli.download --universe sp500 --force --rate-limit 0.5
        """
    )

    # Symbol sources (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--universe",
        choices=["sp500", "etf", "all"],
        help="Download predefined universe"
    )
    source_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols"
    )
    source_group.add_argument(
        "--symbols-file",
        type=str,
        help="File with one symbol per line"
    )

    # Options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="cache/",
        help="Output directory (default: cache/)"
    )
    parser.add_argument(
        "--rate-limit", "-r",
        type=float,
        default=1.0,
        help="Requests per second (default: 1.0)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=str,
        default="1d",
        help="Data interval (default: 1d)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to download"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine symbols to download
    symbols = []
    if args.universe == "sp500":
        symbols = get_sp500_symbols()
    elif args.universe == "etf":
        symbols = get_default_etfs()
    elif args.universe == "all":
        symbols = get_sp500_symbols() + get_default_etfs()
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.symbols_file:
        with open(args.symbols_file) as f:
            symbols = [line.strip() for line in f if line.strip()]

    if not symbols:
        logger.error("No symbols to download")
        sys.exit(1)

    # Apply max symbols limit
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]

    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    symbols = unique_symbols

    logger.info(f"Will download {len(symbols)} symbols")

    # Download
    stats = download_symbols(
        symbols=symbols,
        output_dir=Path(args.output),
        rate_limit=args.rate_limit,
        interval=args.interval,
        force_update=args.force
    )

    # Report results
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    print(f"Total symbols:   {stats['total']}")
    print(f"Downloaded:      {stats['success']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Skipped (cached): {stats['skipped']}")
    print("=" * 50)

    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
