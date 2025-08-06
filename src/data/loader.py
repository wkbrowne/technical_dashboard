# Data loading and caching

# -------------------- data/loader.py --------------------
# Module to load and cache price data from Yahoo Finance API via RapidAPI

import os, sys
import time
import pickle
import requests
import pandas as pd
from datetime import datetime

# Import config which loads .env variables
try:
    from ..config import CACHE_FILE
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import CACHE_FILE


def get_multiple_stocks(symbols, interval="1d", update=False, cache_file=CACHE_FILE, rate_limit=1.0, 
                       universe_file=None, max_symbols=None):
    """
    Fetch data for multiple stocks or load from cache.

    Parameters
    ----------
    symbols : list or str
        List of stock symbols to fetch, or 'universe' to load from universe_file
    interval : str
        Time interval for the data (e.g., "1d")
    update : bool
        If False, loads from cache file. If True, fetches fresh data.
    cache_file : str
        Path to the pickle file for saving/loading data
    rate_limit : float
        Maximum number of API requests per second
    universe_file : str, optional
        Path to CSV file containing universe of stocks (used when symbols='universe')
    max_symbols : int, optional
        Maximum number of symbols to load from universe file

    Returns
    -------
    dict
        Dictionary with keys 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume' mapping to DataFrames
    """
    # Handle universe file loading
    if symbols == 'universe' or isinstance(symbols, str) and symbols.lower() == 'universe':
        if universe_file is None:
            # Auto-detect US universe file in cache directory
            cache_dir = os.path.dirname(cache_file)
            universe_files = [f for f in os.listdir(cache_dir) if f.startswith('US universe_2025-08-05') and f.endswith('.csv')]
            if universe_files:
                universe_file = os.path.join(cache_dir, universe_files[0])
                print(f"🌍 Auto-detected universe file: {universe_file}")
            else:
                raise ValueError("No universe file found. Please specify universe_file parameter.")
        
        if not os.path.exists(universe_file):
            raise ValueError(f"Universe file not found: {universe_file}")
        
        print(f"📊 Loading symbols from universe file: {universe_file}")
        universe_df = pd.read_csv(universe_file)
        
        # Extract symbols from the Symbol column
        if 'Symbol' in universe_df.columns:
            symbols = universe_df['Symbol'].dropna().tolist()
        else:
            # Try to find symbol column with different names
            symbol_cols = [col for col in universe_df.columns if 'symbol' in col.lower()]
            if symbol_cols:
                symbols = universe_df[symbol_cols[0]].dropna().tolist()
            else:
                raise ValueError(f"No 'Symbol' column found in universe file. Available columns: {list(universe_df.columns)}")
        
        # Apply max_symbols limit if specified
        if max_symbols and max_symbols > 0:
            symbols = symbols[:max_symbols]
            print(f"🎯 Limited to first {len(symbols)} symbols from universe")
        
        print(f"✅ Loaded {len(symbols)} symbols from universe file")
        
        # Update cache file name to include universe info
        if hasattr(cache_file, 'with_suffix'):
            # Path object
            cache_file = cache_file.with_name(cache_file.stem + '_universe.pkl')
        else:
            # String
            cache_file = str(cache_file).replace('.pkl', '_universe.pkl')

    if not update and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    all_data = []
    delay = 1.0 / rate_limit if rate_limit > 0 else 0

    for idx, symbol in enumerate(symbols):
        print(f"Fetching data for {symbol}...")
        # Replace "." and "/" with "-" for API lookup
        api_symbol = symbol.replace(".", "-").replace("/", "-")
        df = get_stock_data(api_symbol, interval, original_symbol=symbol)
        if df is not None:
            all_data.append(df)

        if idx < len(symbols) - 1:
            time.sleep(delay)

    if not all_data:
        print("No data was successfully fetched for any of the symbols")
        return None

    combined_df = pd.concat(all_data)
    result = {}
    metrics = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']

    for metric in metrics:
        metric_df = combined_df.pivot(columns='symbol', values=metric)
        result[metric] = metric_df

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        print(f"Saved fresh data to {cache_file}")

    return result

def get_etf_data(etf_symbols, interval="1d", update=False, cache_file=None, rate_limit=1.0):
    """
    Fetch data for multiple ETFs or load from cache.
    
    Parameters
    ----------
    etf_symbols : list
        List of ETF symbols to fetch (e.g., ['SPY', 'QQQ', 'IWM'])
    interval : str
        Time interval for the data (e.g., "1d")
    update : bool
        If False, loads from cache file. If True, fetches fresh data.
    cache_file : str
        Path to the pickle file for saving/loading ETF data
    rate_limit : float
        Maximum number of API requests per second
        
    Returns
    -------
    dict
        Dictionary with ETF symbols as keys mapping to DataFrames with OHLC data
    """
    if cache_file is None:
        # Handle Path object properly
        if hasattr(CACHE_FILE, 'with_suffix'):
            # Path object - use with_stem to change name
            cache_file = CACHE_FILE.with_name(CACHE_FILE.stem + '_etf.pkl')
        else:
            # String - use replace
            cache_file = str(CACHE_FILE).replace('.pkl', '_etf.pkl')
    
    if not update and os.path.exists(cache_file):
        print(f"Loading cached ETF data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    print(f"🏦 Fetching ETF data for: {etf_symbols}")
    etf_data = {}
    delay = 1.0 / rate_limit if rate_limit > 0 else 0
    
    for idx, symbol in enumerate(etf_symbols):
        print(f"Fetching ETF data for {symbol}...")
        df = get_stock_data(symbol, interval)
        if df is not None:
            # Remove the 'symbol' column and clean up for ETF-specific processing
            etf_df = df.drop('symbol', axis=1) if 'symbol' in df.columns else df
            etf_data[symbol] = etf_df
            print(f"✅ {symbol}: {len(etf_df)} observations")
        else:
            print(f"❌ Failed to fetch data for {symbol}")
        
        if idx < len(etf_symbols) - 1:
            time.sleep(delay)
    
    if etf_data:
        with open(cache_file, "wb") as f:
            pickle.dump(etf_data, f)
            print(f"💾 Saved ETF data to {cache_file}")
    else:
        print("⚠️ No ETF data was successfully fetched")
    
    return etf_data

def get_stock_data(symbol, interval="1d", original_symbol=None):
    """
    Fetch historical stock data from Yahoo Finance via RapidAPI.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (transformed for API lookup)
    interval : str
        Data interval (e.g., "1d")
    original_symbol : str, optional
        Original symbol to preserve in the returned DataFrame

    Returns
    -------
    pd.DataFrame or None
        Cleaned DataFrame with historical data
    """
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"

    querystring = {
        "symbol": symbol,
        "interval": interval,
        "diffandsplits": "false"
    }

    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        print(f"⚠️ Error: RAPIDAPI_KEY environment variable not set. Cannot fetch data for {symbol}")
        return None
        
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()

        if 'body' in data and data['body']:
            records = []
            for timestamp, values in data['body'].items():
                record = values.copy()
                record['date'] = values.get('date')
                records.append(record)

            df = pd.DataFrame(records)
            # Fix: Use format='mixed' to handle various date formats automatically
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            df.set_index('date', inplace=True)
            df['symbol'] = original_symbol if original_symbol is not None else symbol

            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjclose': 'AdjClose',
                'volume': 'Volume'
            }

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            required_fields = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
            missing_fields = [field for field in required_fields if field not in df.columns]

            if missing_fields:
                print(f"⚠️ Warning: Missing data for {symbol} — missing fields: {missing_fields}")

            desired_columns = required_fields + ['symbol']
            df = df[[col for col in desired_columns if col in df.columns]]

            return df
        else:
            print(f"No data found for {symbol}")
            return None

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None


def load_universe_data(max_symbols=None, update=False, rate_limit=1.0):
    """
    Convenience function to load data from the US universe file.
    
    Parameters
    ----------
    max_symbols : int, optional
        Maximum number of symbols to load from universe
    update : bool
        If True, fetches fresh data. If False, loads from cache.
    rate_limit : float
        Maximum number of API requests per second
        
    Returns
    -------
    dict
        Dictionary with OHLC data for universe stocks
    """
    print(f"🌍 Loading US universe data...")
    print(f"   Max symbols: {max_symbols or 'All'}")
    print(f"   Update: {'Fresh data' if update else 'Cache if available'}")
    
    return get_multiple_stocks(
        symbols='universe',
        update=update,
        rate_limit=rate_limit,
        max_symbols=max_symbols
    )

