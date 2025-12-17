# Downloader Architecture

This document describes the data download system for the Technical Dashboard feature pipeline.

## Cache Structure

```
cache/
├── stocks/                         # Individual stock data
│   ├── stock_data_combined.parquet # All stocks combined (long format)
│   ├── stock_data_universe.parquet # Alternative name
│   └── stock_data_weekly.parquet   # Pre-computed weekly resampled data
├── etfs/                           # ETF and index data
│   ├── stock_data_etf.parquet      # All ETFs combined (long format)
│   └── etf_data_weekly.parquet     # Pre-computed weekly resampled data
├── fred_data.parquet               # FRED economic data (wide format)
└── US universe_*.csv               # Symbol list with sector mappings
```

## Quick Start

```bash
# Activate environment
conda activate stocks_predictor

# Download stocks from universe CSV (all symbols in US universe_*.csv)
python -m src.cli.download --universe stocks --output cache/

# Download a limited number of stocks (for debugging)
python -m src.cli.download --universe stocks --max-symbols 50

# Download ETFs only
python -m src.cli.download --universe etf

# Download FRED economic data
python -m src.cli.download --universe fred --output cache/

# Download everything (stocks + ETFs)
python -m src.cli.download --universe all

# Force refresh (ignore cache)
python -m src.cli.download --universe stocks --force

# Custom rate limiting
python -m src.cli.download --universe stocks --rate-limit 0.5

# Generate weekly resampled cache after download
python -m src.cli.download --universe all --resample-weekly
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--universe` | Required | Universe to download: `stocks`, `etf`, `fred`, `all` |
| `--symbols` | None | Specific symbols to download (comma-separated) |
| `--symbols-file` | None | File with one symbol per line |
| `--output` | `cache/` | Output directory for cache files |
| `--force` | False | Force re-download even if cache exists |
| `--rate-limit` | 1.0 | Requests per second |
| `--max-symbols` | None | Maximum number of symbols to download (useful for debugging) |
| `--interval` | `1d` | Data interval (1d, 1h, etc.) |
| `--cleanup` | False | Remove individual symbol parquet files after creating combined file |
| `--resample-weekly` | False | Generate pre-computed weekly cache files |
| `--verbose` | False | Enable verbose logging |

### Universe Options

| Value | Description | Source |
|-------|-------------|--------|
| `stocks` | All stocks from universe CSV | `cache/US universe_*.csv` |
| `etf` | Curated ETF list (~100 symbols) | Hardcoded in `get_default_etfs()` |
| `fred` | FRED economic data series | FRED API (requires `FRED_API_KEY`) |
| `all` | Both stocks and ETFs | Combined stocks + etf |

**Note:** The `stocks` option reads from `US universe_*.csv`. Use `--max-symbols` to limit downloads for faster debugging.

## Module Architecture

### 1. CLI Entry Point: src/cli/download.py

The main command-line interface that orchestrates downloads.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `get_stock_universe_symbols()` | Load symbols from `US universe_*.csv` |
| `get_default_etfs()` | Return curated ETF list (~100 symbols) |
| `download_symbols()` | Main download loop with rate limiting |
| `download_fred_data()` | Download FRED economic series |
| `generate_weekly_cache()` | Pre-compute weekly resampled data |

**Download Flow:**
```
CLI Arguments
    ↓
get_stock_universe_symbols() or get_default_etfs()
    ↓
download_symbols() [with rate limiting]
    ↓
Save to cache/{stocks,etfs}/stock_data_*.parquet
    ↓
generate_weekly_cache() [optional]
```

### 2. Data Fetching: src/data/loader.py

Core data fetching via RapidAPI Yahoo Finance.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `get_stock_data()` | Fetch single symbol via RapidAPI |
| `_request_with_retries()` | HTTP with exponential backoff |
| `_filter_untradeable_securities()` | Remove SPACs, ADRs, warrants |
| `_save_long_parquet()` | Save data in long format |
| `_load_long_parquet()` | Load cached data |
| `resample_daily_to_weekly()` | OHLCV aggregation to weekly |
| `load_stock_universe()` | Public API for stocks |
| `load_etf_universe()` | Public API for ETFs |

**Retry Logic:**
- Exponential backoff: 1s, 2s, 4s, 8s...
- Max 3 retries by default
- Handles rate limits (429) and server errors (5xx)

### 3. FRED Data: src/data/fred.py

Federal Reserve Economic Data integration.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `download_fred_series()` | Fetch FRED series via API |
| `compute_fred_features()` | Generate derived features |
| `add_fred_features()` | Merge FRED features to stock data |

## Data Formats

### Long Format (Stocks and ETFs)

```
date        symbol  value       metric
2024-01-02  AAPL    185.64      Close
2024-01-02  AAPL    186.74      High
2024-01-02  AAPL    184.35      Low
2024-01-02  AAPL    185.14      Open
2024-01-02  AAPL    185.64      AdjClose
2024-01-02  AAPL    45678900    Volume
```

**Metrics:** Open, High, Low, Close, AdjClose, Volume

### Wide Format (FRED)

```
date        DGS10   DGS2    T10Y2Y  BAMLH0A0HYM2  ...
2024-01-02  3.95    4.32    -0.37   3.54          ...
2024-01-03  3.91    4.28    -0.37   3.52          ...
```

## ETF Universe

The downloader includes approximately 100 curated ETFs across categories:

### Market Benchmarks
SPY, QQQ, IWM, DIA, VTI, VOO, RSP

### Sector Cap-Weighted (Select Sector SPDRs)
XLF, XLK, XLE, XLY, XLI, XLP, XLV, XLU, XLB, XLC, XLRE

### Sector Equal-Weight (Invesco S&P 500 Equal Weight)
RSP, RSPT, RSPF, RSPH, RSPE, RSPS, RSPC, RSPU, RSPM, RSPD, RSPN

### Subsector ETFs

| ETF | Sector |
|-----|--------|
| SMH, SOXX | Semiconductors |
| IGV | Software |
| KBE, KRE | Banks |
| IBB, XBI | Biotech |
| ITA | Aerospace/Defense |
| XOP | Oil & Gas E&P |
| XRT | Retail |
| ITB, XHB | Homebuilders |
| TAN | Solar |
| URA | Uranium |
| LIT | Lithium/Battery |
| COPX | Copper Miners |

### Fixed Income
TLT, IEF, SHY, HYG, LQD, AGG, BND, TIP, EMB

### Commodities
GLD, SLV, USO, UNG, DBC, PDBC

### Volatility
^VIX, ^VXN, VXX, VIXY, UVXY, SVXY

### International
EFA, EEM, FXI, EWJ, EWZ, VEU, IEMG

## FRED Series

| Series | Name | Pub Lag | Use Case |
|--------|------|---------|----------|
| DGS10 | 10Y Treasury Yield | 1 day | Interest rate regime |
| DGS2 | 2Y Treasury Yield | 1 day | Short-term rates |
| T10Y2Y | 10Y-2Y Spread | 1 day | Yield curve shape |
| BAMLH0A0HYM2 | High Yield OAS | 1 day | Credit risk |
| ICSA | Initial Jobless Claims | 5 days | Labor market |
| CCSA | Continued Claims | 12 days | Labor market |
| NFCI | Chicago Fed Financial Conditions | 7 days | Financial stress |
| VIXCLS | VIX Close | 1 day | Volatility regime |

**Feature Transforms:**
- `level` - Raw value
- `change_5d` / `change_20d` - Period change
- `zscore_60d` - 60-day z-score
- `percentile_252d` - 1-year percentile rank

## Symbol Filtering

The loader automatically filters out untradeable securities:

| Pattern | Filtered |
|---------|----------|
| Description contains "Acquisition Corp/Inc/Company" | SPACs |
| Description contains "ADR", "American Depositary" | ADRs |
| Symbol ends with `.W`, `/U`, `-R` | Warrants/Units/Rights |
| Symbol contains `.PR`, `-P` | Preferred shares |
| Description contains "Blank Check" | Blank check companies |

## Weekly Resampling

Weekly data is aggregated using standard OHLCV rules:

```python
agg_rules = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'adjclose': 'last',
    'volume': 'sum'
}
```

**Week ending:** Friday (W-FRI convention)

## API Keys

### Required

| Key | Purpose | How to Get |
|-----|---------|------------|
| `RAPIDAPI_KEY` | Yahoo Finance data | rapidapi.com |
| `FRED_API_KEY` | FRED economic data | fred.stlouisfed.org |

### Configuration

**Option 1: Environment variables**
```bash
export RAPIDAPI_KEY="your-rapidapi-key"
export FRED_API_KEY="your-fred-key"
```

**Option 2: .env file (recommended)**
```bash
# .env file in project root
RAPIDAPI_KEY=your-rapidapi-key
FRED_API_KEY=your-fred-key
```

## Best Practices

1. **Rate Limiting**: Use `--rate-limit 0.5` for large downloads to avoid API throttling

2. **Debugging**: Use `--max-symbols 50` to quickly test with a subset of data

3. **Incremental Updates**: Don't use `--force` unless necessary; the downloader skips existing symbols

4. **Weekly Cache**: Use `--resample-weekly` to pre-compute weekly data and speed up pipeline compute

5. **Symbol Validation**: Check for invalid symbols in cache:
   ```python
   import pandas as pd
   df = pd.read_parquet('cache/stocks/stock_data_combined.parquet')
   invalid = df[~df['symbol'].str.match(r'^[A-Z\^][A-Z0-9\.]*$')]
   print(invalid['symbol'].unique())
   ```

6. **FRED Cache**: FRED data is cached separately and doesn't auto-refresh. Delete `cache/fred_data.parquet` to force re-download.

## Troubleshooting

### Rate Limit Errors (429)
```bash
# Increase delay between requests
python -m src.cli.download --universe stocks --rate-limit 0.5
```

### Missing API Key
```bash
# Check if key is set
echo $RAPIDAPI_KEY
echo $FRED_API_KEY

# Set via export or .env file
```

### Stale FRED Data
```bash
# Force refresh
rm cache/fred_data.parquet
python -m src.cli.download --universe fred --force
```

### Invalid Symbols in Cache
```python
# Clean invalid entries
import pandas as pd
df = pd.read_parquet('cache/stocks/stock_data_combined.parquet')
df_clean = df[df['symbol'].str.match(r'^[A-Z\^][A-Z0-9\.]*$', na=False)]
df_clean.to_parquet('cache/stocks/stock_data_combined.parquet', index=False)
```
