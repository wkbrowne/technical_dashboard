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
        stacked = df.stack(future_stack=True).reset_index()
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
    """
    Load parquet file in either long format or raw OHLCV format.

    Supports two formats:
    1. Long format: columns = [date, symbol, metric, value] - pivot to {metric: wide DataFrame}
    2. Raw format: columns = [Open, High, Low, Close, AdjClose, Volume, symbol, date] - convert to same structure

    Returns None if format is unrecognized or data is invalid.
    """
    if not os.path.exists(parquet_path):
        return None
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return None

    # Format 1: Long format with [date, symbol, metric, value]
    if {'date', 'symbol', 'metric', 'value'}.issubset(df.columns):
        # Filter out invalid symbols (file names like 'fred_data', lowercase, underscores)
        valid_symbol_mask = df['symbol'].str.match(r'^[A-Z\^][A-Z0-9\.]*$', na=False)
        invalid_symbols = df.loc[~valid_symbol_mask, 'symbol'].unique()
        if len(invalid_symbols) > 0:
            print(f"⚠️  Filtering {len(invalid_symbols)} invalid symbols: {list(invalid_symbols)[:5]}")
            df = df[valid_symbol_mask]

        out: Dict[str, pd.DataFrame] = {}
        for metric, chunk in df.groupby('metric'):
            out[str(metric)] = chunk.pivot(index='date', columns='symbol', values='value').sort_index()
        return out

    # Format 2: Raw OHLCV format with symbol and date columns (from new download CLI)
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    if 'symbol' in df.columns and 'date' in df.columns and any(col in df.columns for col in ohlcv_cols):
        print("📊 Converting raw format with dates to wide format...")
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = df['symbol'].astype('string')

        out: Dict[str, pd.DataFrame] = {}
        for metric in ohlcv_cols:
            if metric in df.columns:
                out[metric] = df.pivot(index='date', columns='symbol', values=metric).sort_index()
        return out if out else None

    # Format 3: Raw OHLCV format WITHOUT date column - invalid, skip this file
    if 'symbol' in df.columns and any(col in df.columns for col in ohlcv_cols):
        print(f"⚠️  Cache file {parquet_path} missing date column - falling back to legacy cache")
        return None

    return None


# ---------- Robust GET w/ retries ----------
def _request_with_retries(url: str, headers: dict, params: dict,
                          max_retries: int = 4, base_sleep: float = 2.0,
                          verify_ssl: bool = True) -> requests.Response:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=(10, 30), verify=verify_ssl)
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
# Set to False to bypass SSL verification (use if RapidAPI has cert issues)
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() not in ("false", "0", "no")

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
        resp = _request_with_retries(url, headers, params, max_retries=4, base_sleep=2.0, verify_ssl=VERIFY_SSL)
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


def _filter_untradeable_securities(df: pd.DataFrame, sym_col: str, desc_col: Optional[str] = None) -> pd.DataFrame:
    """
    Filter out SPACs, ADRs, warrants, units, and other untradeable securities.

    Filtering rules:
    1. SPACs: Description contains "Acquisition" (case-insensitive)
    2. ADRs: Description contains "ADR" or symbol ends with common ADR suffixes
    3. Warrants: Symbol contains "W" suffix pattern (e.g., FOOW, FOO.W, FOO/W)
    4. Units: Symbol contains "U" suffix pattern or "/U"
    5. Foreign listings: Symbol ends with "F" (foreign ordinary shares)
    6. Rights: Symbol contains "R" suffix pattern
    7. Preferred shares: Symbol contains "P" suffix (e.g., BAC.PR.A, BAC-P)

    Args:
        df: DataFrame with at least a symbol column
        sym_col: Name of the symbol column
        desc_col: Name of the description column (optional, for SPAC/ADR detection)

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    mask = pd.Series(True, index=df.index)

    # Get symbols for pattern matching
    symbols = df[sym_col].astype(str).str.upper()

    # 1. Filter by symbol patterns (warrants, units, rights, preferred, foreign)
    # Be CONSERVATIVE - only filter obvious derivative patterns with separators

    # Warrants: .W, /W, -W, or WS suffix (require separator to avoid false positives like ACIW, CDW)
    warrant_pattern = r'[.\-/]W[S]?$'
    mask &= ~symbols.str.contains(warrant_pattern, regex=True, na=False)

    # Units: .U, /U, -U (require separator)
    unit_pattern = r'[.\-/]U$'
    mask &= ~symbols.str.contains(unit_pattern, regex=True, na=False)

    # Rights: .R, /R, -R (require separator)
    rights_pattern = r'[.\-/]R$'
    mask &= ~symbols.str.contains(rights_pattern, regex=True, na=False)

    # Preferred: .PR, -PR, /PR, or .P, -P, /P followed by letter (e.g., BAC.PRA)
    preferred_pattern = r'[.\-/]PR?[A-Z]?$'
    mask &= ~symbols.str.contains(preferred_pattern, regex=True, na=False)

    # Foreign ordinary shares: .F or /F only (require separator)
    foreign_pattern = r'[.\-/]F$'
    mask &= ~symbols.str.contains(foreign_pattern, regex=True, na=False)

    # 2. Filter by description if available
    if desc_col and desc_col in df.columns:
        descriptions = df[desc_col].astype(str).str.upper()

        # SPACs: Description contains "Acquisition Corp/Inc/Company" pattern (more specific to avoid "GE Aerospace")
        # Must have "Acquisition" followed by Corp/Corporation/Co/Company/Inc/Limited
        spac_pattern = r'ACQUISITION\s+(?:CORP|CO\b|COMPANY|LIMITED|INC)'
        mask &= ~descriptions.str.contains(spac_pattern, regex=True, na=False)

        # ADRs: Description contains "ADR", "American Depositary", "Depositary Share/Receipt"
        # More specific - require word boundary or "ADS-" prefix, not just "ADS" anywhere
        adr_pattern = r'\bADR\b|AMERICAN DEPOSITARY|DEPOSITARY SHARE|DEPOSITARY RECEIPT|^ADS-'
        mask &= ~descriptions.str.contains(adr_pattern, regex=True, na=False)

        # Blank Check Companies
        blank_check_pattern = r'BLANK CHECK'
        mask &= ~descriptions.str.contains(blank_check_pattern, regex=True, na=False)

    filtered_df = df[mask].copy()
    removed_count = original_count - len(filtered_df)

    if removed_count > 0:
        print(f"   🚫 Filtered {removed_count} untradeable securities (SPACs, ADRs, warrants, etc.)")

    return filtered_df


def filter_suspicious_price_symbols(
    data: Dict[str, pd.DataFrame],
    max_price: float = 50000.0,
    min_price: float = 0.01,
    max_price_range_ratio: float = 10000.0,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Filter out symbols with suspicious price data (likely unadjusted reverse splits).

    Detection rules:
    1. Any close price > max_price (default $50k) - extreme price suggests data issue
    2. Any close price < min_price (default $0.01) - penny stock data often unreliable
    3. Price range ratio (max/min) > max_price_range_ratio - suggests split adjustment issues

    Args:
        data: Dict of {metric: DataFrame with symbols as columns}
        max_price: Maximum allowed close price (default $50k)
        min_price: Minimum allowed close price (default $0.01)
        max_price_range_ratio: Maximum allowed ratio of max/min price (default 10000x)

    Returns:
        Tuple of (filtered_data, list of removed symbols)
    """
    if 'Close' not in data or data['Close'].empty:
        return data, []

    close_df = data['Close']
    symbols_to_remove = []

    for sym in close_df.columns:
        prices = close_df[sym].dropna()
        if len(prices) == 0:
            continue

        price_max = prices.max()
        price_min = prices.min()

        # Check for extreme prices
        if price_max > max_price:
            symbols_to_remove.append(sym)
            continue

        # Check for penny stocks
        if price_min < min_price:
            symbols_to_remove.append(sym)
            continue

        # Check for suspicious price range (split adjustment issues)
        if price_min > 0 and (price_max / price_min) > max_price_range_ratio:
            symbols_to_remove.append(sym)
            continue

    if symbols_to_remove:
        print(f"   ⚠️  Filtering {len(symbols_to_remove)} symbols with suspicious prices: {symbols_to_remove[:5]}{'...' if len(symbols_to_remove) > 5 else ''}")
        # Filter from all metrics
        filtered_data = {}
        for metric, df in data.items():
            keep_cols = [c for c in df.columns if c not in symbols_to_remove]
            filtered_data[metric] = df[keep_cols]
        return filtered_data, symbols_to_remove

    return data, []


def _symbols_from_csv(path: str, max_symbols: Optional[int] = None, filter_untradeable: bool = True) -> List[str]:
    """
    Load symbols from CSV, optionally filtering out untradeable securities.

    Args:
        path: Path to CSV file
        max_symbols: Maximum number of symbols to return
        filter_untradeable: If True, filter out SPACs, ADRs, warrants, etc.

    Returns:
        List of symbol strings
    """
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else next((c for c in df.columns if "symbol" in c.lower()), None)
    if not col:
        raise ValueError(f"No symbol column in {path}. Columns: {list(df.columns)}")

    # Apply filtering if requested
    if filter_untradeable:
        desc_col = "Description" if "Description" in df.columns else next(
            (c for c in df.columns if "description" in c.lower()), None
        )
        df = _filter_untradeable_securities(df, col, desc_col)

    syms = df[col].dropna().astype(str).tolist()
    if max_symbols and max_symbols > 0:
        syms = syms[:max_symbols]
    return syms

def _sectors_from_universe(path: str, max_symbols: Optional[int] = None, filter_untradeable: bool = True) -> Dict[str, str]:
    """
    Returns a dict: {symbol -> sector} from the universe CSV.
    Keeps the sector text as-is (strip/normalize whitespace & capitalization lightly).

    Args:
        path: Path to universe CSV file
        max_symbols: Maximum number of symbols to return
        filter_untradeable: If True, filter out SPACs, ADRs, warrants, etc.

    Returns:
        Dictionary mapping symbol -> sector
    """
    df = pd.read_csv(path)
    # Detect columns (robust to case/spacing)
    sym_col = "Symbol" if "Symbol" in df.columns else next((c for c in df.columns if "symbol" in c.lower()), None)
    sec_col = "Sector" if "Sector" in df.columns else next((c for c in df.columns if "sector" in c.lower()), None)
    if not sym_col or not sec_col:
        raise ValueError(f"No Symbol/Sector columns in {path}. Columns: {list(df.columns)}")

    # Apply filtering if requested
    if filter_untradeable:
        desc_col = "Description" if "Description" in df.columns else next(
            (c for c in df.columns if "description" in c.lower()), None
        )
        df = _filter_untradeable_securities(df, sym_col, desc_col)

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
    Load stock OHLCV into dict of wide DataFrames. Caches to *_universe.parquet or *_combined.parquet.
    If include_sectors=True, returns a tuple: (data, sectors_map)
    where sectors_map is {symbol -> sector} from the universe CSV.

    Cache selection logic:
    - Prefers *_combined.parquet (created by download CLI) if it exists and is newer
    - Falls back to *_universe.parquet (legacy cache format)
    """
    # Cache paths - check both legacy and combined formats
    cache_dir = CACHE_FILE.parent if hasattr(CACHE_FILE, "parent") else os.path.dirname(str(CACHE_FILE))
    cache_stem = CACHE_FILE.stem if hasattr(CACHE_FILE, "stem") else os.path.splitext(os.path.basename(str(CACHE_FILE)))[0]

    universe_parquet = os.path.join(cache_dir, f"{cache_stem}_universe.parquet")
    combined_parquet = os.path.join(cache_dir, f"{cache_stem}_combined.parquet")

    # Determine cache preference order (prefer combined if newer)
    cache_preference = []
    if os.path.exists(combined_parquet) and os.path.exists(universe_parquet):
        combined_mtime = os.path.getmtime(combined_parquet)
        universe_mtime = os.path.getmtime(universe_parquet)
        if combined_mtime > universe_mtime:
            cache_preference = [combined_parquet, universe_parquet]
        else:
            cache_preference = [universe_parquet, combined_parquet]
    elif os.path.exists(combined_parquet):
        cache_preference = [combined_parquet]
    elif os.path.exists(universe_parquet):
        cache_preference = [universe_parquet]

    # Try cache files in preference order (OHLCV only; sectors always re-read from CSV for freshness)
    cached = None
    cache_parquet = None
    if not update:
        for cache_path in cache_preference:
            print(f"💾 stocks: trying cache {cache_path}")
            cached = _load_long_parquet(cache_path)
            if cached:
                cache_parquet = cache_path
                print(f"✅ stocks: loaded from {cache_path}")
                break
            else:
                print(f"⚠️  Cache {cache_path} invalid, trying next...")

    if not update and cached:
        if include_sectors:
            universe_csv = _discover_universe_csv(CACHE_FILE if isinstance(CACHE_FILE, str) else str(CACHE_FILE))
            sectors = _sectors_from_universe(universe_csv, max_symbols=max_symbols)

            # ALWAYS filter cached data to match filtered sectors (removes SPACs, ADRs, etc.)
            # The sectors dict has already filtered out untradeable securities
            filtered_symbols = list(sectors.keys())
            cached_limited = {}
            for metric, df in cached.items():
                available_symbols = [sym for sym in filtered_symbols if sym in df.columns]
                cached_limited[metric] = df[available_symbols] if available_symbols else df.iloc[:, :0]
            cached = cached_limited
            print(f"📊 Filtered to {len(filtered_symbols)} tradeable symbols (removed SPACs, ADRs, etc.)")

            # Filter symbols with suspicious prices (unadjusted reverse splits, etc.)
            cached, removed_price = filter_suspicious_price_symbols(cached)
            if removed_price:
                # Also remove from sectors
                sectors = {k: v for k, v in sectors.items() if k not in removed_price}

            return cached, sectors

        # When not including sectors, still filter out untradeable securities
        universe_csv = _discover_universe_csv(CACHE_FILE if isinstance(CACHE_FILE, str) else str(CACHE_FILE))
        filtered_symbols = _symbols_from_csv(universe_csv, max_symbols=max_symbols)
        cached_limited = {}
        for metric, df in cached.items():
            available_symbols = [sym for sym in filtered_symbols if sym in df.columns]
            cached_limited[metric] = df[available_symbols] if available_symbols else df.iloc[:, :0]
        cached = cached_limited
        print(f"📊 Filtered to {len(filtered_symbols)} tradeable symbols")

        # Filter symbols with suspicious prices (unadjusted reverse splits, etc.)
        cached, _ = filter_suspicious_price_symbols(cached)

        return cached

    # No valid cache found - fetch from API
    print("⚠️  No valid cache found — fetching from API...")

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

    # Save to universe cache (default location for API-fetched data)
    save_path = universe_parquet
    print(f"💾 stocks: writing {save_path}")
    _save_long_parquet(data, save_path)
    return (data, sectors) if include_sectors else data


def load_etf_universe(etf_symbols: Optional[List[str]] = None,
                      etf_csv_path: Optional[str] = None,
                      update: bool = False,
                      rate_limit: float = 1.0,
                      interval: str = "1d",
                      incremental: bool = True) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load ETF OHLCV into dict of wide DataFrames. Caches to etfs/stock_data_etf.parquet.
    Choose one of:
      - etf_symbols (list)
      - etf_csv_path (CSV with Symbol column)
      - default curated list (if both are None)

    Args:
        incremental: If True, only fetch missing ETFs and merge with existing cache.
                    If False, re-fetch all ETFs. Default True.
    """
    # Cache path - use etfs/ subfolder
    cache_dir = CACHE_FILE.parent if hasattr(CACHE_FILE, "parent") else os.path.dirname(str(CACHE_FILE))
    etf_cache_dir = os.path.join(cache_dir, "etfs")
    os.makedirs(etf_cache_dir, exist_ok=True)
    cache_parquet = os.path.join(etf_cache_dir, "stock_data_etf.parquet")

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


# ---------- Weekly Resampling ----------
import numpy as np

def resample_daily_to_weekly(
    data: Dict[str, pd.DataFrame],
    drop_incomplete_week: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Resample daily OHLCV data to weekly (W-FRI).

    Resampling rules:
    - open: first of week
    - high: max of week
    - low: min of week
    - close/adjclose: last of week (Friday)
    - volume: sum of week

    Args:
        data: Dict of {metric: wide DataFrame with symbols as columns}
        drop_incomplete_week: If True, drop the last week if incomplete

    Returns:
        Dict of {metric: weekly DataFrame with symbols as columns}
    """
    if not data:
        return {}

    # Get the common date index from any metric
    sample_df = next(iter(data.values()))
    if sample_df.empty:
        return {}

    # Ensure DatetimeIndex
    if not isinstance(sample_df.index, pd.DatetimeIndex):
        sample_df.index = pd.to_datetime(sample_df.index)

    # Define aggregation rules per metric
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'AdjClose': 'last',
        'Volume': 'sum',
    }

    result = {}
    for metric, df in data.items():
        if df.empty:
            continue

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        # Get agg rule for this metric
        agg_rule = agg_rules.get(metric, 'last')

        # Resample to weekly (Friday end-of-week)
        weekly_df = df.resample('W-FRI').agg(agg_rule)

        result[metric] = weekly_df

    # Drop incomplete last week if requested
    if drop_incomplete_week and result:
        today = pd.Timestamp.now().normalize()
        # Find last Friday
        days_since_friday = (today.weekday() - 4) % 7
        last_friday = today - pd.Timedelta(days=days_since_friday)

        # If today is not Friday, the last week is incomplete
        if today.weekday() != 4:
            for metric in result:
                # Keep only weeks up to and including the previous Friday
                result[metric] = result[metric][result[metric].index <= last_friday]

    return result


def save_weekly_cache(
    weekly_data: Dict[str, pd.DataFrame],
    output_path: str
) -> None:
    """
    Save weekly resampled data to parquet in long format.

    Args:
        weekly_data: Dict of {metric: weekly DataFrame}
        output_path: Path to save parquet file
    """
    _save_long_parquet(weekly_data, output_path)
    print(f"💾 Saved weekly cache to {output_path}")


def load_weekly_cache(cache_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load pre-computed weekly cache if it exists.

    Args:
        cache_path: Path to weekly parquet cache

    Returns:
        Dict of {metric: weekly DataFrame} or None if not found
    """
    if not os.path.exists(cache_path):
        return None

    data = _load_long_parquet(cache_path)
    if data:
        print(f"✅ Loaded weekly cache from {cache_path}")
    return data


def get_weekly_cache_path(cache_dir: str, is_etf: bool = False) -> str:
    """
    Get the standard path for weekly cache file.

    Args:
        cache_dir: Base cache directory path (e.g., cache/)
        is_etf: If True, return ETF weekly cache path in etfs/ subfolder

    Returns:
        Full path to weekly cache file
    """
    subfolder = "etfs" if is_etf else "stocks"
    filename = "etf_data_weekly.parquet" if is_etf else "stock_data_weekly.parquet"
    return os.path.join(cache_dir, subfolder, filename)