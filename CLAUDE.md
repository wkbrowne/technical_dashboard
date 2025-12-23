# Claude Code Context - Technical Dashboard

A financial features pipeline for computing technical indicators across daily, weekly, and monthly timeframes with triple barrier labeling for machine learning.

## Key Documentation

For detailed implementation guides, see:

| Document | Description |
|----------|-------------|
| [docs/DOWNLOADER.md](docs/DOWNLOADER.md) | Data download CLI, cache structure, ETF/FRED lists, API keys |
| [docs/FEATURE_PIPELINE_ARCHITECTURE.md](docs/FEATURE_PIPELINE_ARCHITECTURE.md) | Pipeline stages, parallelism, data flow, target generation |
| [docs/FEATURE_SELECTION.md](docs/FEATURE_SELECTION.md) | Feature selection methodology, sample weighting, CV strategy |

## Quick Commands

```bash
conda activate stocks_predictor

# Download
python -m src.cli.download --universe stocks      # Download stocks
python -m src.cli.download --universe etf         # Download ETFs
python -m src.cli.download --universe all         # Download everything
python -m src.cli.download --universe fred        # FRED macro data

# Compute features
python -m src.cli.compute                                    # Full pipeline
python -m src.cli.compute --timeframes D --max-stocks 50     # Fast iteration
python -m src.cli.compute --checkpoint-dir artifacts/checkpoints  # Resumable

# Quality checks
python run_data_quality.py --verbose
pytest tests/ -v
```

## Architecture Overview

```
src/
├── config/                 # Configuration (features.py, parallel.py)
├── features/               # Feature computation modules
│   ├── single_stock.py     # Per-symbol features (RSI, MACD, ATR, etc.)
│   ├── cross_sectional.py  # Cross-symbol features (alpha, relative strength)
│   ├── factor_regression.py # Joint 4-factor model (beta, alpha)
│   ├── macro.py            # FRED macro features
│   └── target_generation.py # Triple barrier targets
├── pipelines/orchestrator.py  # Main pipeline coordinator
├── data/loader.py          # Data loading + SPAC/ADR filtering
├── cli/                    # CLI commands (download.py, compute.py)
└── feature_selection/      # Feature selection pipeline
    └── base_features.py    # BASE_FEATURES golden reference
```

## Critical Principles

### 1. DataFrame Column Insertion

**NEVER** insert columns one at a time in a loop - causes `PerformanceWarning: DataFrame is highly fragmented`.

```python
# WRONG
for col_name, series in features.items():
    df[col_name] = series  # PerformanceWarning!

# CORRECT
new_df = pd.DataFrame(features, index=df.index)
result = pd.concat([df, new_df], axis=1)
```

### 2. Column Names Must Be Lowercase

All column names MUST be lowercase immediately after loading any data source.

```python
df.columns = [c.lower() for c in df.columns]
```

### 3. NaN Handling

Never drop NaNs. Use interpolation only for internal gaps:

```python
df[col].interpolate(method='linear', limit_area='inside')
```

### 4. Weekly Feature Leakage Prevention

Weekly features must not leak future information:

```python
# Friday's weekly feature applies to NEXT week's daily rows
merged = pd.merge_asof(daily, weekly, on='date', by='symbol', direction='backward')
```

### 5. Parallel Processing

Workers should only receive data they need:

```python
# CORRECT: Subset benchmarks per batch
chunk_benchmarks = get_required_benchmarks_for_batch(chunk, all_benchmarks)

# WRONG: Send entire dataset to each worker
delayed(process)(sym, full_universe_df)  # BAD: Serializes everything N times
```

### 6. Row Alignment with Non-Unique Indices

When data has multiple rows per date (e.g., multiple symbols), **NEVER** use `.loc[date_index]` for row alignment - it causes a silent cross-join explosion.

```python
# WRONG: Causes cross-join when dates are not unique
X_aligned = X.loc[y.index]  # 25k rows → 500k rows silently!

# CORRECT: Use a unique row identifier
df['_row_id'] = range(len(df))  # Add unique ID during data loading
row_ids = valid_rows['_row_id'].values  # Get IDs for target subset
X_aligned = X[X['_row_id'].isin(row_ids)]  # Filter by unique ID
```

This is critical for panel data where (date, symbol) pairs are unique but date alone is not.

## BASE_FEATURES (Golden Reference)

The curated feature set lives in `src/feature_selection/base_features.py`:

```python
from src.feature_selection.base_features import validate_features, BASE_FEATURES
import pandas as pd

df = pd.read_parquet('artifacts/features_complete.parquet')
result = validate_features(df)
print(f"Valid: {len(result['valid'])}/{len(BASE_FEATURES)}")
print(f"Missing: {result['missing']}")
```

## Factor Model (4-Factor Orthogonalized)

```
R_stock = α + β_market × R_SPY
            + β_qqq × (R_QQQ - R_SPY)
            + β_bestmatch × (R_bestmatch - R_SPY)
            + β_breadth × (R_RSP - R_SPY)
            + ε
```

| Factor | Interpretation |
|--------|----------------|
| `beta_market` | Broad market exposure |
| `beta_qqq` | Growth/tech premium over market |
| `beta_bestmatch` | Sector premium over market (R²-based ETF match) |
| `beta_breadth` | Equal-weight vs cap-weight spread |

## Data Files

### Cache (Input)
| File | Description |
|------|-------------|
| `cache/stock_data_combined.parquet` | Stocks OHLCV (long format) |
| `cache/stock_data_etf.parquet` | ETFs OHLCV (long format) |
| `cache/fred_data.parquet` | FRED macro data |
| `cache/US universe_*.csv` | Symbol list with sectors |

### Artifacts (Output)
| File | Description |
|------|-------------|
| `artifacts/features_complete.parquet` | All computed features |
| `artifacts/targets_triple_barrier.parquet` | ML targets with sample weights |

## Environment Setup

Create `.env` in project root:

```bash
RAPIDAPI_KEY=your-rapidapi-key
FRED_API_KEY=your-fred-key
```

## Common Issues

| Issue | Solution |
|-------|----------|
| DataFrame fragmented warning | Use `pd.concat()` instead of loop insertion |
| Features all NaN | Check column case - must be lowercase |
| Memory errors | Use `--max-stocks 100 --timeframes D` |
| Rate limiting | Use `--rate-limit 0.5` |
| Missing features | Check `run_data_quality.py --verbose` |
| Row count explosion after `.loc[]` | Use `_row_id` with `.isin()` for panel data alignment |

## Documentation Guidelines

### File Naming Convention

All documentation files in `docs/` must use **UPPER_SNAKE_CASE** with `.md` extension:
- `FEATURE_SELECTION.md` (correct)
- `POSITION_SIZING.md` (correct)
- `PositionSizing.md` (wrong)
- `feature-selection.md` (wrong)

### Avoid Explicit Feature Lists

Features evolve frequently. **Never hardcode specific feature names or counts** in documentation. Instead:
- Reference the source of truth: `src/feature_selection/base_features.py`
- Use dynamic counts like "current feature set" instead of "53 features"
- Document the validation pattern, not the feature list

```python
# CORRECT: Reference the source
from src.feature_selection.base_features import BASE_FEATURES
print(f"Using {len(BASE_FEATURES)} features")

# WRONG: Hardcoded in docs
"The model uses 53 features including rsi_14, macd_histogram..."
```

## Removed Files

These legacy files no longer exist:
- `src/features/weekly*.py` → merged into `timeframe.py`
- `src/features/pipeline.py` → use `single_stock.py`
- `src/features/breadth.py` → replaced by `sector_breadth.py`
