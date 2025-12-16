# Claude Code Context - Technical Dashboard

A financial features pipeline for computing technical indicators across daily, weekly, and monthly timeframes with triple barrier labeling for machine learning.

## Quick Commands

**Important:** All commands must run in the `stocks_predictor` conda environment.

```bash
# Activate environment (or prefix commands with conda run)
conda activate stocks_predictor

# Download data (run in separate terminal)
python -m src.cli.download --universe sp500 --output cache/

# Download ETFs only
python -m src.cli.download --universe etf

# Download FRED economic data (requires FRED_API_KEY)
python -m src.cli.download --universe fred --output cache/

# Force refresh FRED data
python -m src.cli.download --universe fred --output cache/ --force

# Compute features with default config
python -m src.cli.compute --output artifacts/

# Compute with custom config (faster iteration)
python -m src.cli.compute --config config/features_minimal.yaml

# Compute daily only (fastest)
python -m src.cli.compute --timeframes D --max-stocks 50

# Run tests
pytest tests/ -v

# Run specific test module
pytest tests/test_features.py -v

# Data quality check (after pipeline run)
python run_data_quality.py
python run_data_quality.py --verbose  # Detailed feature descriptions

# Alternative: Use conda run without activating
conda run -n stocks_predictor python -m src.cli.compute --output artifacts/
conda run -n stocks_predictor python run_data_quality.py
```

## Architecture Overview

```
src/
├── config/                 # Configuration
│   ├── features.py         # FeatureSpec, FeatureConfig dataclasses
│   └── regime_config.py    # Trend/volatility regime definitions
│
├── features/               # Feature computation
│   ├── base.py             # BaseFeature, CrossSectionalFeature ABCs
│   ├── registry.py         # Feature factory and registration
│   ├── timeframe.py        # Unified D/W/M resampling
│   ├── single_stock.py     # Orchestrates single-stock features
│   ├── cross_sectional.py  # Orchestrates cross-sectional features
│   ├── trend.py            # MA slopes, RSI, MACD
│   ├── volatility.py       # Multi-scale volatility regime
│   ├── volume.py           # Volume ratios and shocks
│   ├── distance.py         # Distance to MA with z-scores
│   ├── range_breakout.py   # ATR, ranges, breakouts
│   ├── alpha.py            # Alpha momentum vs market/sectors
│   ├── relstrength.py      # Relative strength features
│   ├── breadth.py          # Market breadth indicators
│   ├── xsec.py             # Cross-sectional momentum
│   ├── target_generation.py # Triple barrier targets
│   └── postprocessing.py   # NaN interpolation, lags
│
├── pipelines/              # Orchestration
│   ├── orchestrator.py     # Main pipeline coordinator
│   └── visualization.py    # Feature visualization utilities
│
├── data/                   # Data loading
│   └── loader.py           # Yahoo Finance via RapidAPI + SPAC/ADR filtering
│
└── cli/                    # Command-line interface
    ├── download.py         # Data download command
    └── compute.py          # Feature computation command

config/
├── features.yaml           # Default feature configuration
└── features_minimal.yaml   # Minimal config for fast iteration
```

## BASE_FEATURES (Golden Reference)

The **golden reference** for required features is `src/feature_selection/base_features.py`. This file contains:

- **BASE_FEATURES**: 53 curated features selected via feature selection (0.6932 AUC)
- **FEATURE_CATEGORIES**: Category-to-feature mapping for documentation
- **EXPANSION_CANDIDATES**: Additional features for forward selection experiments
- **EXCLUDED_FEATURES**: Raw values not suitable for ML (e.g., raw prices, raw MAs)

### Validating BASE_FEATURES

```python
from src.feature_selection.base_features import validate_features, BASE_FEATURES
import pandas as pd

df = pd.read_parquet('artifacts/features_complete.parquet')
result = validate_features(df)

print(f"Valid: {len(result['valid'])}/{len(BASE_FEATURES)}")
print(f"Missing: {result['missing']}")
for feat, nan_rate in sorted(result['nan_rates'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat}: {nan_rate:.1f}% NaN")
```

### Running BASE_FEATURES Tests

```bash
# Run BASE_FEATURES validation tests
pytest tests/test_base_features.py -v

# Key tests:
# - test_base_features_contract: Critical features must be present
# - test_pipeline_output_contains_base_features: Pipeline produces all BASE_FEATURES
```

## Adding New Features

Features are computed directly in `single_stock.py` (single-stock) or `cross_sectional.py` (cross-sectional). The registry is legacy and rarely used.

### Step 1: Create the feature function

```python
# src/features/my_feature.py
import pandas as pd
import numpy as np

def add_my_feature(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add my custom feature.

    Args:
        df: DataFrame with 'close' column (and 'ret' for return-based features)
        window: Lookback window

    Returns:
        DataFrame with 'my_feature' column added (in place)

    NOTE: Use pd.concat() for multiple columns to avoid DataFrame fragmentation.
    """
    # Single column: direct assignment is fine
    df['my_feature'] = df['close'].pct_change(window)
    return df
```

### Step 2: Call from the appropriate orchestrator

**For single-stock features** (computed per symbol independently):

```python
# src/features/single_stock.py
from .my_feature import add_my_feature

def compute_single_stock_features(df, ...):
    # ... existing features ...

    # Add your feature (near the end, before return)
    df = add_my_feature(df, window=20)

    return df
```

**For cross-sectional features** (computed across all symbols):

```python
# src/features/cross_sectional.py
from .my_feature import add_my_feature_cross_sectional

def add_cross_sectional_features(indicators_by_symbol, ...):
    # ... existing features ...

    # Cross-sectional features need access to all symbols
    add_my_feature_cross_sectional(indicators_by_symbol)
```

### Step 3: Add weekly version (if needed)

Weekly features are auto-computed via `orchestrator.py:_compute_higher_tf_for_symbol()`. Features computed on daily data are re-computed on weekly-resampled data and prefixed with `w_`.

If your feature needs special weekly handling, add it to the weekly flow in `orchestrator.py`.

### Step 4: Add to BASE_FEATURES (if selected by model)

If your feature proves valuable in feature selection:

```python
# src/feature_selection/base_features.py

BASE_FEATURES = [
    # ... existing features ...

    # === YOUR CATEGORY ===
    "my_feature",        # Short description
    "w_my_feature",      # Weekly version
]

FEATURE_CATEGORIES = {
    # ... existing categories ...
    "my_category": [
        "my_feature",
        "w_my_feature",
    ],
}
```

### Step 5: Add description to data quality script

```python
# run_data_quality.py

FEATURE_DESCRIPTIONS = {
    # ... existing descriptions ...
    "my_feature": "My feature description (units, range, interpretation)",
}
```

### Feature Design Guidelines

1. **Normalize features**: Use z-scores, percentiles, or ratios (not raw prices)
2. **Avoid look-ahead bias**: Lag features appropriately (use `shift()`)
3. **Use pd.concat() for multiple columns**: Avoid DataFrame fragmentation warnings
4. **Test with small data first**: `--max-stocks 10 --timeframes D`
5. **Check NaN rates**: Run `python run_data_quality.py --verbose` after adding

## Factor Model

The pipeline computes factor exposures via joint multivariate ridge regression (`src/features/factor_regression.py`).

### Orthogonalized 4-Factor Model

```
R_stock = α + β_market × R_SPY
            + β_qqq × (R_QQQ - R_SPY)
            + β_bestmatch × (R_bestmatch - R_SPY)
            + β_breadth × (R_RSP - R_SPY)
            + ε
```

| Factor | Formula | Interpretation |
|--------|---------|----------------|
| `beta_market` | R_SPY | Broad market exposure |
| `beta_qqq` | R_QQQ - R_SPY | Growth/tech premium over market |
| `beta_bestmatch` | R_bestmatch - R_SPY | Sector/subsector premium over market |
| `beta_breadth` | R_RSP - R_SPY | Equal-weight vs cap-weight spread |

### Best-Match ETF Selection

Each stock is matched to the ETF (sector or subsector) with highest R² from the candidate pool:
- Sector ETFs: XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLC, XLRE
- Subsector ETFs: SMH, IGV, KBE, KRE, IBB, XBI, ITA, ITB, XOP, XRT, TAN, URA, LIT, COPX

Example: NVDA matches SMH (semiconductors) not XLK (broad tech) because SMH explains more variance.

### Features Output

Daily (`beta_*`) and weekly (`w_beta_*`) versions:
- `beta_market`, `beta_qqq`, `beta_bestmatch`, `beta_breadth` - factor loadings
- `residual_mean`, `residual_cumret`, `residual_vol` - idiosyncratic return statistics

## Key Patterns

### DataFrame Column Insertion

**NEVER** insert columns one at a time in a loop - this causes `PerformanceWarning: DataFrame is highly fragmented`. Instead, collect all new columns and use `pd.concat()`:

```python
# WRONG: Repeated column insertion causes fragmentation
for col_name, series in features.items():
    df[col_name] = series  # PerformanceWarning!

# CORRECT: Collect columns and concat once
new_cols = {col_name: series for col_name, series in features.items()}
new_df = pd.DataFrame(new_cols, index=df.index)
result = pd.concat([df, new_df], axis=1)
```

### NaN Handling

Never drop NaNs to preserve history. Use interpolation only for internal gaps:

```python
# CORRECT: Preserve leading/trailing NaNs
df[col].interpolate(method='linear', limit_area='inside')

# WRONG: Don't do this
df.dropna()
```

### Timeframe Resampling

Use the TimeframeResampler for consistent D/W/M handling:

```python
from src.features.timeframe import TimeframeResampler, TimeframeType

# Resample to weekly
weekly_df = TimeframeResampler.resample_single(daily_df, TimeframeType.WEEKLY)

# Merge back to daily (leakage-safe)
merged = TimeframeResampler.merge_to_daily(daily_df, weekly_df, TimeframeType.WEEKLY)
```

### Parallel Processing

Use joblib with loky backend for symbol-level parallelization. **Two key rules:**

1. **Worker count is determined by `stocks_per_worker`** (default 100):
   - 100 stocks → 1 worker
   - 500 stocks → 5 workers
   - 3000 stocks → 30 workers (capped at 32 max)

2. **Workers should only receive data they need** to minimize serialization overhead:
   - Pass subsetted benchmark/ETF data, not the entire dataset
   - Each batch gets: core benchmarks (SPY, QQQ, RSP) + batch-specific best-match ETFs

```python
from joblib import Parallel, delayed
from src.config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER

# Calculate workers based on stocks_per_worker
n_symbols = len(indicators_by_symbol)
chunk_size = DEFAULT_STOCKS_PER_WORKER  # 100
n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)

# Create batches
work_chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

# IMPORTANT: Subset shared data (benchmarks, ETFs) for each batch
chunk_benchmarks = []
for chunk in work_chunks:
    # Only include benchmarks this batch actually needs
    subset = get_required_benchmarks_for_batch(chunk, all_benchmarks)
    chunk_benchmarks.append(subset)

# Run parallel with subsetted data
results = Parallel(n_jobs=n_workers, backend='loky')(
    delayed(process_batch)(chunk, chunk_benchmarks[i])
    for i, chunk in enumerate(work_chunks)
)
```

See `src/config/parallel.py` for `ParallelConfig` class and helper functions.

### Feature Prefixes

- Daily features: no prefix (e.g., `rsi_14`)
- Weekly features: `w_` prefix (e.g., `w_rsi_14`)
- Monthly features: `m_` prefix (e.g., `m_rsi_14`)

## Configuration System

Toggle features on/off via YAML:

```yaml
features:
  rsi:
    enabled: true     # Compute this feature
    params:
      periods: [14, 21, 30]

  macd:
    enabled: false    # Skip this feature
```

Or programmatically:

```python
from src.config import FeatureConfig

config = FeatureConfig.from_yaml('config/features.yaml')
config.disable_feature('breadth')
config.set_param('rsi', 'periods', [14])
```

## Data Quality Check

After running the pipeline, use `run_data_quality.py` to validate output:

```bash
# Basic quality check
python run_data_quality.py

# Verbose mode (detailed feature descriptions)
python run_data_quality.py --verbose

# Check a specific file
python run_data_quality.py --features artifacts/features_complete.parquet
```

**Output includes:**
- Feature coverage by category (trend, momentum, volatility, etc.)
- NaN rates vs expected ranges
- Broken features (100% NaN) - indicates pipeline bugs
- Target file anomalies (invalid price data)
- Actionable recommendations for fixing issues

**Example output:**
```
FEATURES SUMMARY
Category              Count  Avg NaN  Max NaN Status
alpha_beta               40    27.0%   100.0% BROKEN (7)
breadth                   6     0.3%     1.5% OK
vix_macro                20     1.1%     4.6% OK

RECOMMENDATIONS
1. Weekly factor regression features are 100% NaN.
   Check: src/features/factor_regression.py weekly computation
```

## Visualization

```python
from src.pipelines.visualization import FeatureVisualizer
import pandas as pd

df = pd.read_parquet('artifacts/features_daily.parquet')
viz = FeatureVisualizer(df)

# Static matplotlib plot
viz.plot_single_stock('AAPL', ['rsi_14', 'macd_histogram'])

# Interactive plotly
fig = viz.interactive_stock_explorer('AAPL')
fig.show()

# Check feature coverage
coverage = viz.coverage_summary()
print(coverage[coverage['nan_pct_mean'] > 10])
```

## Testing

Run tests before committing:

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_features.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Common Issues

### Memory errors

Reduce dataset size:
```bash
python -m src.cli.compute --max-stocks 100 --timeframes D
```

### Rate limiting (download)

Slow down requests:
```bash
python -m src.cli.download --rate-limit 0.5 --universe sp500
```

### Missing features

Check if feature is enabled in config:
```python
from src.config import FeatureConfig
config = FeatureConfig.from_yaml('config/features.yaml')
print(config.features['rsi'].enabled)
```

### NaN issues

Use visualizer to check coverage:
```python
viz = FeatureVisualizer(df)
coverage = viz.validate_feature_coverage()
print(coverage[coverage['nan_pct'] > 50])
```

## Files to Ignore (Legacy/Deprecated)

These files are superseded by the new architecture but kept for reference:

- `src/features/weekly.py` - Use timeframe.py instead
- `src/features/weekly_enhanced.py` - Merged into timeframe.py
- `src/features/weekly_optimized.py` - Merged into timeframe.py
- `src/features/weekly_multiprocessing.py` - Best patterns extracted to timeframe.py
- `src/features/pipeline.py` - Use single_stock.py directly
- `src/features/runner.py` - Use CLI commands instead

## Environment Setup

The pipeline requires API keys for data access. These can be set via environment variables or a `.env` file in the project root.

### Required API Keys

| Key | Purpose | Required For |
|-----|---------|--------------|
| `RAPIDAPI_KEY` | Yahoo Finance data via RapidAPI | `src.cli.download` |
| `FRED_API_KEY` | Federal Reserve Economic Data | FRED macro features (optional if cache exists) |

### Option 1: Environment Variables

```bash
export RAPIDAPI_KEY="your-rapidapi-key"
export FRED_API_KEY="your-fred-key"
```

### Option 2: .env File (Recommended)

Create a `.env` file in the project root:

```bash
# .env file (automatically loaded by CLI commands)
RAPIDAPI_KEY=your-rapidapi-key
FRED_API_KEY=your-fred-key
```

The CLI commands (`src.cli.compute`, `src.cli.download`) automatically load `.env` via `python-dotenv`.

### Getting API Keys

- **RapidAPI**: Sign up at [rapidapi.com](https://rapidapi.com) and subscribe to Yahoo Finance API
- **FRED**: Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

**Note**: FRED features use cached data from `cache/fred_data.parquet` if available. The API key is only needed for initial download or manual refresh.

## Data Files

### Directory Structure

```
cache/                              # Input data (raw OHLCV, macro)
├── stock_data_combined.parquet     # Individual stocks (S&P 500, etc.)
├── stock_data_etf.parquet          # ETFs (SPY, sector ETFs, VIX indices)
├── fred_data.parquet               # FRED macro data (cached, auto-downloaded)
├── US universe_*.csv               # Symbol list with sectors
└── *.parquet                       # Individual symbol files (temporary)

artifacts/                          # Output data (computed features)
├── features_daily.parquet          # Daily features
├── features_complete.parquet       # Daily + Weekly features
└── targets_triple_barrier.parquet  # ML targets
```

### Cache Files (Input)

#### 1. Stock Data: `stock_data_combined.parquet`

**Purpose**: Individual stock OHLCV data (S&P 500, other tradeable stocks)

**Format**: Long format with columns `[date, symbol, value, metric]`
- `date`: Trading date (datetime)
- `symbol`: Ticker symbol (string, UPPERCASE only, e.g., "AAPL", "MSFT")
- `value`: Price/volume value (float)
- `metric`: One of "Open", "High", "Low", "Close", "AdjClose", "Volume"

**Symbol validation**: Only valid ticker symbols (regex: `^[A-Z\^][A-Z0-9\.]*$`)
- Valid: `AAPL`, `BRK.B`, `^VIX`
- Invalid: `fred_data`, `stock_data`, `aapl` (lowercase)

**IMPORTANT**: Must NOT contain:
- ETF symbols (SPY, XLF, etc.) - those go in `stock_data_etf.parquet`
- File names accidentally loaded as symbols (e.g., `fred_data`)
- Lowercase or underscore-containing entries

#### 2. ETF Data: `stock_data_etf.parquet`

**Purpose**: ETF and index OHLCV data for benchmarking and cross-sectional features

**Format**: Same long format as stocks: `[date, symbol, value, metric]`

**Contents**:
- Market benchmarks: SPY, QQQ, IWM, DIA, VTI, VOO
- Fixed income: TLT, IEF, SHY, HYG, LQD, AGG, BND
- Sector cap-weighted: XLF, XLK, XLE, XLY, XLI, XLP, XLV, XLU, XLB, XLC, XLRE
- Sector equal-weight (RSP series): RSP, RSPT, RSPF, RSPH, RSPE, RSPS, RSPC, RSPU, RSPM, RSPD, RSPN
- Subsector ETFs: SMH, SOXX, IGV, KBE, KRE, IBB, XBI, ITA, XOP, etc.
- Commodities: GLD, SLV, USO, UNG, DBC
- Volatility indices: ^VIX, ^VXN, VXX, VIXY

#### 3. FRED Data: `fred_data.parquet`

**Purpose**: Federal Reserve Economic Data (macro indicators) - cached separately from price data

**Format**: Wide format with date index and series as columns
- Index: Trading date (datetime)
- Columns: FRED series IDs (DGS10, DGS2, T10Y2Y, BAMLH0A0HYM2, ICSA, etc.)

**Cache behavior**: Uses cached data if `fred_data.parquet` exists (does NOT auto-refresh)
- Only downloads if cache file is missing
- To force refresh: delete `cache/fred_data.parquet` and ensure `FRED_API_KEY` is set
- Downloaded by `src/data/fred.py:download_fred_series(force_refresh=True)`

**Contents** (with publication lags):
| Series | Name | Pub Lag | Transforms |
|--------|------|---------|------------|
| DGS10 | 10Y Treasury Yield | 1 day | level, change_5d, change_20d, zscore_60d, percentile_252d |
| DGS2 | 2Y Treasury Yield | 1 day | level, change_5d, change_20d |
| T10Y2Y | 10Y-2Y Yield Spread | 1 day | level, change_5d, zscore_60d, percentile_252d |
| BAMLH0A0HYM2 | High Yield OAS | 1 day | level, change_5d, change_20d, zscore_60d, percentile_252d |
| ICSA | Initial Jobless Claims | 5 days | level, change_4w, zscore_52w, percentile_104w |
| CCSA | Continued Claims | 12 days | level, change_4w, zscore_52w |
| NFCI | Chicago Fed Financial Conditions | 7 days | level, change_4w, zscore_52w |
| VIXCLS | VIX Close | 1 day | level, change_5d, zscore_60d, percentile_252d |

**IMPORTANT**: FRED data is NEVER mixed with stock/ETF data. It lives in its own file.

#### 4. Universe CSV: `US universe_*.csv`

**Purpose**: Symbol list with sector/industry mappings

**Format**: CSV with columns:
- `Symbol`: Ticker symbol
- `Sector`: GICS sector (e.g., "Technology", "Health Care")
- `Description`: Company name (used for filtering SPACs/ADRs)

### Artifacts (Output)

| File | Description |
|------|-------------|
| `features_daily.parquet` | Daily features only (~300 features) |
| `features_complete.parquet` | Daily + Weekly features (~600 features) |
| `targets_triple_barrier.parquet` | ML targets (forward returns, barrier hits) |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOWNLOAD PHASE                           │
│  python -m src.cli.download --universe sp500 → stock_data_combined.parquet
│  python -m src.cli.download --universe etf   → stock_data_etf.parquet
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        COMPUTE PHASE                            │
│  Loader: stock_data_combined.parquet → indicators_by_symbol     │
│  Loader: stock_data_etf.parquet → etf_data (benchmarks)         │
│  FRED:   fred_data.parquet (auto-downloaded if missing)         │
│                              ↓                                  │
│  Single-stock features → Cross-sectional features               │
│                              ↓                                  │
│  Output: features_complete.parquet, targets_triple_barrier.parquet
└─────────────────────────────────────────────────────────────────┘
```

### File Selection Logic

When both `stock_data_combined.parquet` and `stock_data_universe.parquet` exist, the pipeline uses the **newer** file (by modification time).

### Troubleshooting Data Issues

**Invalid symbol in data** (e.g., `fred_data` as a symbol):
```python
# Check for invalid symbols
import pandas as pd
df = pd.read_parquet('cache/stock_data_combined.parquet')
invalid = df[~df['symbol'].str.match(r'^[A-Z\^][A-Z0-9\.]*$', na=False)]
print(invalid['symbol'].unique())

# Remove invalid symbols
df_clean = df[df['symbol'].str.match(r'^[A-Z\^][A-Z0-9\.]*$', na=False)]
df_clean.to_parquet('cache/stock_data_combined.parquet', index=False)
```

**FRED data not loading**:
```bash
# Ensure FRED_API_KEY is set
export FRED_API_KEY="your-key-here"

# Delete stale cache to force re-download
rm cache/fred_data.parquet
```

## Symbol Filtering

The data loader automatically filters out untradeable securities:

- **SPACs**: Description contains "Acquisition Corp/Inc/Company"
- **ADRs**: Description contains "ADR", "American Depositary", etc.
- **Warrants/Units/Rights**: Symbols with `.W`, `/U`, `-R` patterns
- **Preferred shares**: Symbols with `.PR`, `-P` patterns
- **Blank check companies**: Description contains "Blank Check"

This filtering is enabled by default in `_symbols_from_csv()` and `_sectors_from_universe()`. To disable:

```python
from src.data.loader import _symbols_from_csv
symbols = _symbols_from_csv(csv_path, filter_untradeable=False)
```
