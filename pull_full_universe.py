#!/usr/bin/env python3
"""
Pull and cache the US universe from RapidAPI (Yahoo Finance 15).
Minimal CLI: choose refresh/cache mode, max symbols, and rate limit.
"""

import os
import sys
import argparse
from datetime import datetime

# Make `src` importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def load_env():
    """Load .env if python-dotenv is available; otherwise rely on existing env."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
    except Exception:
        # No dotenv? No problem—just use whatever env is already set.
        pass

def require_api_key():
    key = os.getenv("RAPIDAPI_KEY")
    if not key:
        raise SystemExit("RAPIDAPI_KEY not set. Put it in your environment or .env")
    return key

def pull_universe(max_symbols=None, rate_limit=2.0, refresh=False):
    """Fetch or load cached universe data via src.data.loader."""
    from src.data.loader import load_universe_data
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Loading US universe (max={max_symbols or 'ALL'}, "
          f"{'refresh' if refresh else 'cache-if-available'}, rate={rate_limit}/s)")
    data = load_universe_data(max_symbols=max_symbols, update=refresh, rate_limit=rate_limit)
    # quick sanity summary
    if not data or "Close" not in data:
        raise SystemExit("No data returned.")
    n_syms = len(data["Close"].columns)
    n_days = len(data["Close"])
    print(f"✔ Loaded {n_syms} symbols over {n_days} trading days")
    return data

def status():
    """Very small status helper—tries to load from cache without refresh."""
    from src.data.loader import load_universe_data
    try:
        data = load_universe_data(max_symbols=None, update=False, rate_limit=0.0)
        if data and "Close" in data:
            print(f"Cache OK: {len(data['Close'].columns)} symbols, {len(data['Close'])} days")
        else:
            print("No cache or invalid cache.")
    except Exception as e:
        print(f"Cache load error: {e}")

def parse_args():
    p = argparse.ArgumentParser(description="Pull & cache US universe via RapidAPI")
    p.add_argument("--max-symbols", type=int, default=None, help="Limit number of symbols")
    p.add_argument("--rate-limit", type=float, default=2.0, help="Requests per second")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch (ignore cache)")
    p.add_argument("--status", action="store_true", help="Show cache status and exit")
    return p.parse_args()

def main():
    load_env()
    args = parse_args()
    if args.status:
        status()
        return
    require_api_key()
    pull_universe(max_symbols=args.max_symbols, rate_limit=args.rate_limit, refresh=args.refresh)

if __name__ == "__main__":
    main()