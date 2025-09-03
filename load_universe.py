#!/usr/bin/env python3
"""
Command-line interface for loading stock and ETF universe data.
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def load_env():
    """Load .env file if available."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
    except Exception:
        pass

def require_api_key():
    """Ensure RAPIDAPI_KEY is set."""
    key = os.getenv("RAPIDAPI_KEY")
    if not key:
        raise SystemExit("RAPIDAPI_KEY not set. Put it in your environment or .env file")
    return key

def load_stocks(max_symbols=None, update=False, rate_limit=1.0, include_sectors=False):
    """Load stock universe data."""
    from src.data.loader import load_stock_universe
    
    print(f"Loading stock universe (max={max_symbols or 'ALL'}, update={update}, rate={rate_limit}/s)")
    
    result = load_stock_universe(
        max_symbols=max_symbols,
        update=update,
        rate_limit=rate_limit,
        include_sectors=include_sectors
    )
    
    if include_sectors:
        stocks, sectors = result
        print(f"✅ Loaded {len(stocks['Close'].columns) if stocks else 0} stocks with sector info")
        print(f"📊 Sectors: {len(sectors)} mapped symbols")
        return stocks, sectors
    else:
        print(f"✅ Loaded {len(result['Close'].columns) if result else 0} stocks")
        return result

def load_etfs(etf_symbols=None, etf_csv_path=None, update=False, rate_limit=1.0):
    """Load ETF universe data."""
    from src.data.loader import load_etf_universe
    
    source = "CSV" if etf_csv_path else "list" if etf_symbols else "default"
    print(f"Loading ETF universe from {source} (update={update}, rate={rate_limit}/s)")
    
    result = load_etf_universe(
        etf_symbols=etf_symbols,
        etf_csv_path=etf_csv_path,
        update=update,
        rate_limit=rate_limit
    )
    
    print(f"✅ Loaded {len(result['Close'].columns) if result else 0} ETFs")
    return result

def load_both(max_symbols=None, etf_symbols=None, etf_csv_path=None, 
              update=False, rate_limit=1.0, include_sectors=False):
    """Load both stocks and ETFs."""
    print("Loading complete universe (stocks + ETFs)")
    
    # Load stocks
    if include_sectors:
        stocks, sectors = load_stocks(max_symbols, update, rate_limit, include_sectors)
    else:
        stocks = load_stocks(max_symbols, update, rate_limit, include_sectors)
        sectors = None
    
    # Load ETFs  
    etfs = load_etfs(etf_symbols, etf_csv_path, update, rate_limit)
    
    return stocks, etfs, sectors

def main():
    parser = argparse.ArgumentParser(description="Load stock and ETF universe data")
    parser.add_argument("--mode", choices=["stocks", "etfs", "both"], default="both",
                       help="What to load (default: both)")
    parser.add_argument("--max-symbols", type=int, help="Max stock symbols to load")
    parser.add_argument("--etf-symbols", nargs="+", help="Specific ETF symbols to load")
    parser.add_argument("--etf-csv", help="Path to CSV file with ETF symbols")
    parser.add_argument("--update", action="store_true", help="Force refresh (ignore cache)")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="API requests per second")
    parser.add_argument("--include-sectors", action="store_true", help="Include sector mapping for stocks")
    parser.add_argument("--status", action="store_true", help="Show cache status only")
    
    args = parser.parse_args()
    
    if args.status:
        from src.data.loader import load_stock_universe, load_etf_universe
        print("=== CACHE STATUS ===")
        try:
            stocks = load_stock_universe(update=False, rate_limit=0)
            print(f"Stocks: {len(stocks['Close'].columns) if stocks else 0} symbols cached")
        except:
            print("Stocks: No cache or error")
        
        try:
            etfs = load_etf_universe(update=False, rate_limit=0)
            print(f"ETFs: {len(etfs['Close'].columns) if etfs else 0} symbols cached")
        except:
            print("ETFs: No cache or error")
        return
    
    load_env()
    require_api_key()
    
    if args.mode == "stocks":
        load_stocks(args.max_symbols, args.update, args.rate_limit, args.include_sectors)
    elif args.mode == "etfs":
        load_etfs(args.etf_symbols, args.etf_csv, args.update, args.rate_limit)
    else:  # both
        load_both(args.max_symbols, args.etf_symbols, args.etf_csv, 
                 args.update, args.rate_limit, args.include_sectors)

if __name__ == "__main__":
    main()