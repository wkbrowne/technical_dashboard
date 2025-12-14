# -------------------- src/data/loader.py (lean + sectors) --------------------
import os
import sys
from time import monotonic, sleep
from typing import Dict, Optional, List, Tuple, Union

import requests
import pandas as pd

# Import config (for CACHE_FILE path)
try:
    from ..config import CACHE_FILE
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import CACHE_FILE


# ---------- Parquet I/O (long-format) ----------
def _save_long_parquet(data_dict: Dict[str, pd.DataFrame], parquet_path: str) -> None:
    if not data_dict:
        return
    pieces = []
    for metric, df in data_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        stacked = df.stack(dropna=False).reset_index()
        stacked.columns = ['date', 'symbol', 'value']
        stacked['metric'] = metric
        pieces.append(stacked)
    if not pieces:
        return
    long_df = pd.concat(pieces, ignore_index=True)
    long_df['symbol'] = long_df['symbol'].astype('string')
    long_df['metric'] = long_df['metric'].astype('string')
    long_df['date'] = pd.to_datetime(long_df['date'])
    long_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)


def _load_long_parquet(parquet_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    if not os.path.exists(parquet_path):
        return None
    df = pd.read_parquet(parquet_path)
    if df.empty or not {'date', 'symbol', 'metric', 'value'}.issubset(df.columns):
        return None
    out: Dict[str, pd.DataFrame] = {}
    for metric, chunk in df.groupby('metric'):
        out[str(metric)] = chunk.pivot(index='date', columns='symbol', values='value').sort_index()
    return out


# ---------- Robust GET w/ retries ----------
def _request_with_retries(url: str, headers: dict, params: dict,
                          max_retries: int = 4, base_sleep: float = 2.0) -> requests.Response:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=(10, 30))
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                wait = float(ra) if (ra and str(ra).isdigit()) else base_sleep * (2 ** attempt)
                print(f"   ⏳ 429 throttled. Sleeping {wait:.1f}s...")
                sleep(wait); continue
            if 500 <= resp.status_code < 600:
                wait = base_sleep * (2 ** attempt)
                print(f"   ⏳ {resp.status_code} server error. Retry in {wait:.1f}s...")
                sleep(wait); continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            wait = base_sleep * (2 ** attempt)
            print(f"   ⏳ Network error ({type(e).__name__}): {e}. Retry in {wait:.1f}s...")
            sleep(wait)
    raise last_err


# ---------- Yahoo (RapidAPI) fetch ----------
def get_stock_data(symbol: str, interval: str = "1d", original_symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV from Yahoo Finance (via RapidAPI yahoo-finance15).
    Returns DataFrame indexed by date with columns: Open/High/Low/Close/AdjClose/Volume + symbol
    """
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        print(f"⚠️  RAPIDAPI_KEY not set. Skipping {symbol}")
        return None

    params = {"symbol": symbol, "interval": interval, "diffandsplits": "false"}
    headers = {"x-rapidapi-key": rapidapi_key, "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"}

    print(f"↪ {original_symbol or symbol} …", end="", flush=True)
    try:
        resp = _request_with_retries(url, headers, params, max_retries=4, base_sleep=2.0)
        data = resp.json()
    except Exception as e:
        print(f" fail ({type(e).__name__})")
        return None
    print(" ok")

    if not isinstance(data, dict) or not data.get("body"):
        print(f"   ⚠️  no body for {symbol}")
        return None

    records = []
    for _, v in data["body"].items():
        rec = v.copy()
        rec["date"] = v.get("date")
        records.append(rec)
    if not records:
        print(f"   ⚠️  empty records for {symbol}")
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")

    colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
              "adjclose": "AdjClose", "volume": "Volume"}
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    keep = [c for c in ["Open","High","Low","Close","AdjClose","Volume"] if c in df.columns]
    if not keep:
        print(f"   ⚠️  missing OHLCV for {symbol}")
        return None

    df = df[keep]
    df["symbol"] = original_symbol or symbol
    return df


# ---------- Symbol sources ----------
def _discover_universe_csv(cache_file: str) -> str:
    cache_dir = os.path.dirname(cache_file)
    cands = [f for f in os.listdir(cache_dir) if f.startswith("US universe_") and f.endswith(".csv")]
    if not cands:
        raise FileNotFoundError("No universe CSV found in cache directory.")
    return os.path.join(cache_dir, sorted(cands)[0])

def _symbols_from_csv(path: str, max_symbols: Optional[int] = None) -> List[str]:
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else next((c for c in df.columns if "symbol" in c.lower()), None)
    if not col:
        raise ValueError(f"No symbol column in {path}. Columns: {list(df.columns)}")
    syms = df[col].dropna().astype(str).tolist()
    if max_symbols and max_symbols > 0:
        syms = syms[:max_symbols]
    return syms

def _sectors_from_universe(path: str, max_symbols: Optional[int] = None) -> Dict[str, str]:
    """
    Returns a dict: {symbol -> sector} from the universe CSV.
    Keeps the sector text as-is (strip/normalize whitespace & capitalization lightly).
    """
    df = pd.read_csv(path)
    # Detect columns (robust to case/spacing)
    sym_col = "Symbol" if "Symbol" in df.columns else next((c for c in df.columns if "symbol" in c.lower()), None)
    sec_col = "Sector" if "Sector" in df.columns else next((c for c in df.columns if "sector" in c.lower()), None)
    if not sym_col or not sec_col:
        raise ValueError(f"No Symbol/Sector columns in {path}. Columns: {list(df.columns)}")

    out = (df[[sym_col, sec_col]]
           .dropna(subset=[sym_col])
           .assign(**{sym_col: df[sym_col].astype(str).str.strip(),
                      sec_col: df[sec_col].astype(str).str.strip()}))
    if max_symbols and max_symbols > 0:
        out = out.iloc[:max_symbols]

    # Keep original sector names (e.g., "Electronic technology")
    return dict(zip(out[sym_col], out[sec_col]))

DEFAULT_ETFS = [
    # Core market benchmarks
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",

    # Fixed income
    "TLT", "IEF", "SHY", "HYG", "LQD", "AGG", "BND",

    # Sector cap-weighted (Select Sector SPDRs)
    "XLF", "XLK", "XLE", "XLY", "XLI", "XLP", "XLV", "XLU", "XLB", "XLC", "XLRE",

    # Sector equal-weight (Invesco)
    "RSP", "RYT", "RYF", "RYE", "RYH", "RYU", "RHS", "RTM", "EWRE",

    # Technology subsectors
    "SMH", "SOXX", "IGV", "SKYY", "HACK", "WCLD", "CLOU",

    # Financial subsectors
    "KBE", "KRE", "IAI", "KIE",

    # Healthcare/Biotech subsectors
    "IBB", "XBI", "IHE", "PJP",

    # Industrial subsectors
    "ITA", "XAR", "ITB", "XHB", "IYT", "XTN",

    # Energy subsectors
    "XOP", "OIH", "XES", "AMLP",

    # Consumer subsectors
    "XRT", "IBUY", "PBJ",

    # Materials/Mining subsectors
    "XME", "COPX", "GDX", "GDXJ", "SIL",

    # Clean energy/Green subsectors
    "TAN", "ICLN", "QCLN", "PBW", "FAN",

    # Nuclear/Uranium
    "URA", "URNM",

    # Lithium/Battery/EV
    "LIT", "DRIV", "IDRV",

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


# ---------- Core fetcher (shared) ----------
def _fetch_symbols(symbols: List[str], interval: str = "1d", rate_limit: float = 1.0) -> Optional[Dict[str, pd.DataFrame]]:
    if not symbols:
        return None
    delay = 1.0 / rate_limit if rate_limit > 0 else 0.0
    frames = []
    for i, sym in enumerate(symbols):
        api_sym = sym.replace(".", "-").replace("/", "-")
        df = get_stock_data(api_sym, interval=interval, original_symbol=sym)
        if df is not None:
            frames.append(df)
        if i < len(symbols) - 1 and delay > 0:
            sleep(delay)

    if not frames:
        return None

    combined = pd.concat(frames, axis=0, ignore_index=False).reset_index().rename(columns={'index': 'date'})
    combined['date'] = pd.to_datetime(combined['date'])
    combined['symbol'] = combined['symbol'].astype('string')

    result: Dict[str, pd.DataFrame] = {}
    for metric in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if metric in combined.columns:
            result[metric] = combined.pivot(index='date', columns='symbol', values=metric).sort_index()
    return result


# ---------- Public API ----------
def load_stock_universe(max_symbols: Optional[int] = None,
                        update: bool = False,
                        rate_limit: float = 1.0,
                        interval: str = "1d",
                        include_sectors: bool = False
                        ) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict[str, str]]]:
    """
    Load stock OHLCV into dict of wide DataFrames. Caches to *_universe.parquet.
    If include_sectors=True, returns a tuple: (data, sectors_map)
    where sectors_map is {symbol -> sector} from the universe CSV.
    """
    # Cache path
    cache_parquet = (CACHE_FILE.with_name(CACHE_FILE.stem + "_universe.parquet")
                     if hasattr(CACHE_FILE, "with_name") else str(CACHE_FILE).replace(".pkl", "_universe.parquet"))

    # Try cache (OHLCV only; sectors always re-read from CSV for freshness)
    if not update and os.path.exists(cache_parquet):
        print(f"💾 stocks: loading cache {cache_parquet}")
        cached = _load_long_parquet(cache_parquet)
        if cached:
            if include_sectors:
                universe_csv = _discover_universe_csv(CACHE_FILE if isinstance(CACHE_FILE, str) else str(CACHE_FILE))
                sectors = _sectors_from_universe(universe_csv, max_symbols=max_symbols)
                
                # Apply max_symbols limit to cached data to match sector mappings
                if max_symbols:
                    limited_symbols = list(sectors.keys())  # These are already limited by max_symbols
                    cached_limited = {}
                    for metric, df in cached.items():
                        # Only keep columns (symbols) that are in the limited sectors list
                        available_symbols = [sym for sym in limited_symbols if sym in df.columns]
                        cached_limited[metric] = df[available_symbols] if available_symbols else df.iloc[:, :0]
                    cached = cached_limited
                    print(f"📊 Applied max_symbols limit: reduced to {len(limited_symbols)} symbols")
                
                return cached, sectors
            
            # Apply max_symbols limit even when not including sectors
            if max_symbols:
                universe_csv = _discover_universe_csv(CACHE_FILE if isinstance(CACHE_FILE, str) else str(CACHE_FILE))
                limited_symbols = _symbols_from_csv(universe_csv, max_symbols=max_symbols)
                cached_limited = {}
                for metric, df in cached.items():
                    available_symbols = [sym for sym in limited_symbols if sym in df.columns]
                    cached_limited[metric] = df[available_symbols] if available_symbols else df.iloc[:, :0]
                cached = cached_limited
                
            return cached
        print("⚠️  cache empty/corrupt — fetching")

    # Discover symbols & sectors from stock universe CSV
    universe_csv = _discover_universe_csv(CACHE_FILE if isinstance(CACHE_FILE, str) else str(CACHE_FILE))
    print(f"📊 stocks CSV: {universe_csv}")
    symbols = _symbols_from_csv(universe_csv, max_symbols=max_symbols)
    sectors = _sectors_from_universe(universe_csv, max_symbols=max_symbols)
    print(f"✅ stocks queued: {len(symbols)}")

    # Fetch & cache
    data = _fetch_symbols(symbols, interval=interval, rate_limit=rate_limit)
    if not data:
        print("❌ no stock data fetched"); 
        return (None, sectors) if include_sectors else None

    print(f"💾 stocks: writing {cache_parquet}")
    _save_long_parquet(data, cache_parquet)
    return (data, sectors) if include_sectors else data


def load_etf_universe(etf_symbols: Optional[List[str]] = None,
                      etf_csv_path: Optional[str] = None,
                      update: bool = False,
                      rate_limit: float = 1.0,
                      interval: str = "1d",
                      incremental: bool = True) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load ETF OHLCV into dict of wide DataFrames. Caches to *_etf.parquet.
    Choose one of:
      - etf_symbols (list)
      - etf_csv_path (CSV with Symbol column)
      - default curated list (if both are None)

    Args:
        incremental: If True, only fetch missing ETFs and merge with existing cache.
                    If False, re-fetch all ETFs. Default True.
    """
    # Cache path
    cache_parquet = (CACHE_FILE.with_name(CACHE_FILE.stem + "_etf.parquet")
                     if hasattr(CACHE_FILE, "with_name") else str(CACHE_FILE).replace(".pkl", "_etf.parquet"))

    # Resolve target ETF symbols
    if etf_csv_path and os.path.exists(etf_csv_path):
        print(f"📈 ETF CSV: {etf_csv_path}")
        target_symbols = _symbols_from_csv(etf_csv_path, max_symbols=None)
    elif etf_symbols:
        target_symbols = list(dict.fromkeys(etf_symbols))  # dedupe, keep order
        print(f"📈 ETF list provided ({len(target_symbols)})")
    else:
        target_symbols = DEFAULT_ETFS
        print(f"📈 ETF default set ({len(target_symbols)})")

    # Load existing cache
    cached = None
    if os.path.exists(cache_parquet):
        print(f"💾 ETFs: loading cache {cache_parquet}")
        cached = _load_long_parquet(cache_parquet)

    # If not updating and cache exists, return it (but check for missing symbols)
    if not update and cached:
        # Check which symbols are missing from cache
        cached_symbols = set()
        for metric_df in cached.values():
            cached_symbols.update(metric_df.columns.tolist())

        missing = [s for s in target_symbols if s not in cached_symbols]

        if not missing:
            print(f"✅ All {len(target_symbols)} ETFs found in cache")
            return cached
        elif incremental:
            print(f"⏳ {len(missing)} ETFs missing from cache, fetching incrementally...")
            print(f"   Missing: {missing}")
            new_data = _fetch_symbols(missing, interval=interval, rate_limit=rate_limit)

            if new_data:
                # Merge with existing cache
                for metric, new_df in new_data.items():
                    if metric in cached:
                        # Concat along columns (symbols), aligning on date index
                        cached[metric] = pd.concat([cached[metric], new_df], axis=1).sort_index()
                    else:
                        cached[metric] = new_df

                # Save updated cache
                print(f"💾 ETFs: writing updated cache {cache_parquet}")
                _save_long_parquet(cached, cache_parquet)

            return cached
        else:
            print("⚠️  Missing ETFs but incremental=False, will re-fetch all")

    # Full fetch (update=True or no valid cache)
    print(f"✅ ETFs queued: {len(target_symbols)}")
    data = _fetch_symbols(target_symbols, interval=interval, rate_limit=rate_limit)
    if not data:
        print("❌ no ETF data fetched"); return cached  # Return cached if available

    print(f"💾 ETFs: writing {cache_parquet}")
    _save_long_parquet(data, cache_parquet)
    return data


# Note: VIX/VXN should be downloaded via load_etf_universe() which includes
# ^VIX and ^VXN in DEFAULT_ETFS. The RapidAPI yahoo-finance15 endpoint
# supports these symbols. No yfinance fallback is needed.