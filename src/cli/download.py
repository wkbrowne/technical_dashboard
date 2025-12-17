#!/usr/bin/env python3
"""
CLI for downloading market data.

This module provides commands for downloading stock, ETF, and FRED data.
It can be run independently from feature computation, allowing you to iterate on
compute logic while a long download runs in another terminal.

Usage:
    python -m src.cli.download --universe stocks --output cache/
    python -m src.cli.download --universe fred --output cache/
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

# Load .env file for RAPIDAPI_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass  # dotenv not installed, rely on environment variables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_stock_universe_symbols() -> List[str]:
    """Get stock symbols from universe CSV."""
    from src.data.loader import _discover_universe_csv, _symbols_from_csv
    from src.config import CACHE_FILE

    try:
        csv_path = _discover_universe_csv(str(CACHE_FILE))
        return _symbols_from_csv(csv_path)
    except FileNotFoundError:
        logger.error("Universe CSV not found. Please ensure US universe_*.csv is in cache/")
        return []


def download_fred_data(output_dir: Path, force_update: bool = False) -> dict:
    """Download FRED economic data.

    Args:
        output_dir: Directory to save data
        force_update: If True, re-download even if cached

    Returns:
        Dict with download statistics
    """
    from src.data.fred import download_fred_series

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "fred_data.parquet"

    stats = {"success": 0, "failed": 0, "skipped": 0, "total": 1}

    # Check cache
    if cache_path.exists() and not force_update:
        logger.info(f"FRED data already cached at {cache_path}")
        stats["skipped"] = 1
        return stats

    # Download
    try:
        logger.info("Downloading FRED economic data...")
        df = download_fred_series(cache_path=cache_path, force_refresh=force_update)
        if df is not None and not df.empty:
            stats["success"] = 1
            logger.info(f"Downloaded FRED data: {df.shape[0]} rows, {df.shape[1]} series")
            logger.info(f"Saved to: {cache_path}")
        else:
            stats["failed"] = 1
            logger.error("FRED download returned empty data")
    except Exception as e:
        stats["failed"] = 1
        logger.error(f"FRED download failed: {e}")

    return stats


def get_default_etfs() -> List[str]:
    """Get comprehensive ETF list including all sector and subsector ETFs."""
    return [
        # Core market benchmarks
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",

        # Fixed income
        "TLT", "IEF", "SHY", "HYG", "LQD", "AGG", "BND",

        # Sector cap-weighted (Select Sector SPDRs)
        "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLC", "XLRE",

        # Sector equal-weight (Invesco S&P 500 Equal Weight Sector ETFs)
        # These are used by BEST_MATCH_EW_CANDIDATES in factor_regression.py
        "RSP",   # S&P 500 Equal Weight (market benchmark)
        "RSPT",  # Technology Equal Weight Sector
        "RSPF",  # Financials Equal Weight Sector
        "RSPH",  # Health Care Equal Weight Sector
        "RSPE",  # Energy Equal Weight Sector
        "RSPS",  # Consumer Staples Equal Weight Sector
        "RSPC",  # Consumer Discretionary Equal Weight Sector
        "RSPU",  # Utilities Equal Weight Sector
        "RSPM",  # Materials Equal Weight Sector
        "RSPD",  # Communication Services Equal Weight Sector
        "RSPN",  # Industrials Equal Weight Sector

        # Older Rydex/Invesco equal-weight ETFs (alternative mappings)
        "RYT",   # Technology Equal Weight (legacy)
        "RYF",   # Financial Equal Weight (legacy)
        "RYE",   # Energy Equal Weight (legacy)
        "RYH",   # Healthcare Equal Weight (legacy)
        "RYU",   # Utilities Equal Weight (legacy)
        "RHS",   # Consumer Staples Equal Weight (legacy)
        "RTM",   # Materials Equal Weight (legacy)
        "EWRE",  # Real Estate Equal Weight

        # Technology subsectors
        "SMH",   # VanEck Semiconductor ETF
        "SOXX",  # iShares Semiconductor ETF
        "IGV",   # iShares Expanded Tech-Software ETF
        "SKYY",  # First Trust Cloud Computing ETF
        "HACK",  # ETFMG Prime Cyber Security ETF
        "WCLD",  # WisdomTree Cloud Computing ETF
        "CLOU",  # Global X Cloud Computing ETF

        # Financial subsectors
        "KBE",   # SPDR S&P Bank ETF
        "KRE",   # SPDR S&P Regional Banking ETF
        "IAI",   # iShares U.S. Broker-Dealers & Securities Exchanges ETF
        "KIE",   # SPDR S&P Insurance ETF

        # Healthcare/Biotech subsectors
        "IBB",   # iShares Biotechnology ETF
        "XBI",   # SPDR S&P Biotech ETF
        "IHE",   # iShares U.S. Pharmaceuticals ETF
        "PJP",   # Invesco Dynamic Pharmaceuticals ETF
        "XLV",   # Health Care Select Sector (also sector)

        # Industrial subsectors
        "ITA",   # iShares U.S. Aerospace & Defense ETF
        "XAR",   # SPDR S&P Aerospace & Defense ETF
        "ITB",   # iShares U.S. Home Construction ETF
        "XHB",   # SPDR S&P Homebuilders ETF
        "IYT",   # iShares U.S. Transportation ETF
        "XTN",   # SPDR S&P Transportation ETF

        # Energy subsectors
        "XOP",   # SPDR S&P Oil & Gas Exploration & Production ETF
        "OIH",   # VanEck Oil Services ETF
        "XES",   # SPDR S&P Oil & Gas Equipment & Services ETF
        "AMLP",  # Alerian MLP ETF (midstream)

        # Consumer subsectors
        "XRT",   # SPDR S&P Retail ETF
        "XHB",   # SPDR S&P Homebuilders ETF (also construction)
        "IBUY",  # Amplify Online Retail ETF
        "PBJ",   # Invesco Food & Beverage ETF

        # Materials/Mining subsectors
        "XME",   # SPDR S&P Metals & Mining ETF
        "COPX",  # Global X Copper Miners ETF
        "GDX",   # VanEck Gold Miners ETF
        "GDXJ",  # VanEck Junior Gold Miners ETF
        "SIL",   # Global X Silver Miners ETF

        # Clean energy/Green subsectors
        "TAN",   # Invesco Solar ETF
        "ICLN",  # iShares Global Clean Energy ETF
        "QCLN",  # First Trust NASDAQ Clean Edge Green Energy ETF
        "PBW",   # Invesco WilderHill Clean Energy ETF
        "FAN",   # First Trust Global Wind Energy ETF

        # Nuclear/Uranium
        "URA",   # Global X Uranium ETF
        "URNM",  # Sprott Uranium Miners ETF

        # Lithium/Battery/EV
        "LIT",   # Global X Lithium & Battery Tech ETF
        "DRIV",  # Global X Autonomous & Electric Vehicles ETF
        "IDRV",  # iShares Self-Driving EV and Tech ETF

        # International
        "EFA", "EEM", "VEU", "IEFA", "IEMG",

        # Commodities
        "GLD", "SLV", "USO", "UNG", "DBC", "PDBC",

        # Volatility/Alternatives
        "VXX", "VIXY",

        # Currency
        "UUP",  # Dollar index (for cross-asset features)

        # Volatility Indices (VIX = S&P 500 implied vol, VXN = Nasdaq-100 implied vol)
        "^VIX", "^VXN",
    ]


def download_symbols(
    symbols: List[str],
    output_dir: Path,
    rate_limit: float = 1.0,
    interval: str = "1d",
    force_update: bool = False,
    is_etf: bool = False,
    cleanup: bool = False
) -> dict:
    """Download data for a list of symbols.

    Args:
        symbols: List of symbols to download
        output_dir: Base cache directory (e.g., cache/)
        rate_limit: Requests per second
        interval: Data interval (1d, 1h, etc.)
        force_update: If True, re-download even if cached
        is_etf: If True, save to etfs/ subfolder, otherwise stocks/ subfolder
        cleanup: If True, remove individual symbol parquet files after creating combined file

    Returns:
        Dict with download statistics
    """
    from src.data.loader import get_stock_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use subfolders for stocks and ETFs
    subfolder = "etfs" if is_etf else "stocks"
    symbol_dir = output_dir / subfolder
    symbol_dir.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "skipped": 0, "total": len(symbols)}
    all_data = {}

    logger.info(f"Downloading {len(symbols)} {'ETF' if is_etf else 'stock'} symbols to {symbol_dir}")
    logger.info(f"Rate limit: {rate_limit} req/sec, Interval: {interval}")

    for i, symbol in enumerate(symbols, 1):
        # Check cache - normalize symbol for file path (remove ^ prefix)
        file_symbol = symbol.replace("^", "")
        cache_path = symbol_dir / f"{file_symbol}.parquet"
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

    # Save combined file in long format (compatible with loader's _load_long_parquet)
    # ETFs go to stock_data_etf.parquet, stocks go to stock_data_combined.parquet
    if all_data:
        from src.data.loader import _save_long_parquet
        import pandas as pd

        # Convert individual symbol DataFrames to wide format dict by metric
        # all_data is {symbol: DataFrame with OHLCV + symbol column}
        combined = pd.concat(all_data.values(), ignore_index=False).reset_index().rename(columns={'index': 'date'})
        combined['date'] = pd.to_datetime(combined['date'])
        combined['symbol'] = combined['symbol'].astype('string')

        # Pivot to {metric: wide DataFrame} format expected by _save_long_parquet
        result_dict = {}
        for metric in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
            if metric in combined.columns:
                result_dict[metric] = combined.pivot(index='date', columns='symbol', values=metric).sort_index()

        # Choose output file based on whether these are ETFs or stocks
        # Combined files go in the subfolder (stocks/ or etfs/)
        if is_etf:
            combined_path = symbol_dir / "stock_data_etf.parquet"
        else:
            combined_path = symbol_dir / "stock_data_universe.parquet"

        _save_long_parquet(result_dict, str(combined_path))
        logger.info(f"Saved combined {'ETF' if is_etf else 'stock'} data ({len(all_data)} symbols) to {combined_path}")

        # Cleanup individual symbol files if requested
        if cleanup:
            cleaned = 0
            for symbol in symbols:
                file_symbol = symbol.replace("^", "")
                cache_path = symbol_dir / f"{file_symbol}.parquet"
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        cleaned += 1
                    except Exception as e:
                        logger.warning(f"Could not remove {cache_path}: {e}")
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} individual symbol files")

    return stats


def generate_weekly_cache(output_dir: Path, is_etf: bool = False, source_path: str = None) -> bool:
    """Generate weekly resampled cache from daily data.

    Args:
        output_dir: Base cache directory (e.g., cache/)
        is_etf: If True, process ETF data from etfs/ subfolder; otherwise stocks/ subfolder
        source_path: Optional explicit path to daily data file

    Returns:
        True if successful, False otherwise
    """
    from src.data.loader import (
        _load_long_parquet,
        resample_daily_to_weekly,
        save_weekly_cache,
    )

    output_dir = Path(output_dir)

    # Determine input/output paths - use subfolders
    if source_path:
        daily_path = Path(source_path)
        # Try to infer subfolder from source path
        subfolder_dir = daily_path.parent
        data_type = "custom"
    elif is_etf:
        subfolder_dir = output_dir / "etfs"
        daily_path = subfolder_dir / "stock_data_etf.parquet"
        data_type = "ETF"
    else:
        subfolder_dir = output_dir / "stocks"
        # Try universe file first, then combined
        for fname in ["stock_data_universe.parquet", "stock_data_combined.parquet"]:
            daily_path = subfolder_dir / fname
            if daily_path.exists():
                break
        data_type = "stock"

    if not daily_path.exists():
        logger.warning(f"No {data_type} daily cache found at {daily_path}")
        return False

    # Weekly cache goes in same subfolder as daily
    if is_etf:
        weekly_path = subfolder_dir / "etf_data_weekly.parquet"
    else:
        weekly_path = subfolder_dir / "stock_data_weekly.parquet"

    logger.info(f"Loading daily {data_type} data from {daily_path}")
    daily_data = _load_long_parquet(str(daily_path))

    if not daily_data:
        logger.error(f"Failed to load daily {data_type} data")
        return False

    logger.info(f"Resampling {data_type} data to weekly (W-FRI)...")
    weekly_data = resample_daily_to_weekly(daily_data, drop_incomplete_week=True)

    if not weekly_data:
        logger.error(f"Failed to resample {data_type} data to weekly")
        return False

    # Log sample stats
    first_metric = list(weekly_data.keys())[0]
    n_weeks = len(weekly_data[first_metric])
    n_symbols = len(weekly_data[first_metric].columns)
    logger.info(f"Generated weekly {data_type} data: {n_symbols} symbols x {n_weeks} weeks")

    save_weekly_cache(weekly_data, str(weekly_path))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download market data for the technical dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download stocks from universe CSV
    python -m src.cli.download --universe stocks

    # Download specific symbols
    python -m src.cli.download --symbols AAPL,MSFT,GOOGL,AMZN

    # Download from file
    python -m src.cli.download --symbols-file my_symbols.txt

    # Download ETFs only
    python -m src.cli.download --universe etf

    # Download FRED economic data (requires FRED_API_KEY)
    python -m src.cli.download --universe fred

    # Force update with custom rate limit
    python -m src.cli.download --universe stocks --force --rate-limit 0.5

    # Download all and cleanup individual files
    python -m src.cli.download --universe all --cleanup
        """
    )

    # Symbol sources (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--universe",
        choices=["stocks", "etf", "fred", "all"],
        help="Download predefined universe (fred requires FRED_API_KEY env var)"
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
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove individual symbol parquet files after creating combined file"
    )
    parser.add_argument(
        "--resample-weekly",
        action="store_true",
        help="Generate pre-computed weekly resampled cache files (stock_data_weekly.parquet, etf_data_weekly.parquet)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Helper to deduplicate symbols while preserving order
    def deduplicate(symbols_list):
        seen = set()
        unique = []
        for s in symbols_list:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    # Determine symbols to download based on source
    # Note: ETFs go to stock_data_etf.parquet, stocks go to stock_data_combined.parquet
    # FRED data is special - it's economic data, not symbols
    if args.universe == "fred":
        # FRED download is special - doesn't use symbols
        stats = download_fred_data(
            output_dir=Path(args.output),
            force_update=args.force
        )
    elif args.universe == "all":
        # Download stocks and ETFs separately to maintain file separation
        stock_symbols = deduplicate(get_stock_universe_symbols())
        etf_symbols = deduplicate(get_default_etfs())

        if args.max_symbols:
            stock_symbols = stock_symbols[:args.max_symbols]
            etf_symbols = etf_symbols[:args.max_symbols]

        logger.info(f"Will download {len(stock_symbols)} stocks and {len(etf_symbols)} ETFs")

        # Download stocks first
        stats_stocks = download_symbols(
            symbols=stock_symbols,
            output_dir=Path(args.output),
            rate_limit=args.rate_limit,
            interval=args.interval,
            force_update=args.force,
            is_etf=False,
            cleanup=args.cleanup
        )

        # Then download ETFs
        stats_etfs = download_symbols(
            symbols=etf_symbols,
            output_dir=Path(args.output),
            rate_limit=args.rate_limit,
            interval=args.interval,
            force_update=args.force,
            is_etf=True,
            cleanup=args.cleanup
        )

        # Combine stats for reporting
        stats = {
            "success": stats_stocks["success"] + stats_etfs["success"],
            "failed": stats_stocks["failed"] + stats_etfs["failed"],
            "skipped": stats_stocks["skipped"] + stats_etfs["skipped"],
            "total": stats_stocks["total"] + stats_etfs["total"]
        }
    else:
        # Single universe download
        symbols = []
        is_etf = False

        if args.universe == "stocks":
            symbols = get_stock_universe_symbols()
            is_etf = False
        elif args.universe == "etf":
            symbols = get_default_etfs()
            is_etf = True
        elif args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",")]
            # Custom symbols go to stock file by default
            is_etf = False
        elif args.symbols_file:
            with open(args.symbols_file) as f:
                symbols = [line.strip() for line in f if line.strip()]
            is_etf = False

        if not symbols:
            logger.error("No symbols to download")
            sys.exit(1)

        # Apply max symbols limit
        if args.max_symbols:
            symbols = symbols[:args.max_symbols]

        symbols = deduplicate(symbols)
        logger.info(f"Will download {len(symbols)} {'ETF' if is_etf else 'stock'} symbols")

        # Download
        stats = download_symbols(
            symbols=symbols,
            output_dir=Path(args.output),
            rate_limit=args.rate_limit,
            interval=args.interval,
            force_update=args.force,
            is_etf=is_etf,
            cleanup=args.cleanup
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

    # Generate weekly cache if requested
    if getattr(args, 'resample_weekly', False) and args.universe != "fred":
        print("\n" + "=" * 50)
        print("Generating Weekly Cache")
        print("=" * 50)

        output_path = Path(args.output)
        weekly_success = True

        if args.universe == "all":
            # Generate both stock and ETF weekly caches
            if not generate_weekly_cache(output_path, is_etf=False):
                weekly_success = False
            if not generate_weekly_cache(output_path, is_etf=True):
                weekly_success = False
        elif args.universe == "etf":
            weekly_success = generate_weekly_cache(output_path, is_etf=True)
        else:
            # stocks or custom symbols
            weekly_success = generate_weekly_cache(output_path, is_etf=False)

        if weekly_success:
            print("Weekly cache generation complete")
        else:
            print("Weekly cache generation had errors")
        print("=" * 50)

    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
