# Claude Code Context - Technical Dashboard

A financial features pipeline for computing technical indicators across daily, weekly, and monthly timeframes with triple barrier labeling for machine learning.

## Quick Commands

```bash
# Download data (run in separate terminal)
python -m src.cli.download --universe sp500 --output cache/

# Download ETFs only
python -m src.cli.download --universe etf

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
│   └── loader.py           # Yahoo Finance via RapidAPI
│
└── cli/                    # Command-line interface
    ├── download.py         # Data download command
    └── compute.py          # Feature computation command

config/
├── features.yaml           # Default feature configuration
└── features_minimal.yaml   # Minimal config for fast iteration
```

## Adding New Features

### Step 1: Create the feature function

```python
# src/features/my_feature.py
def add_my_feature(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add my custom feature.

    Args:
        df: DataFrame with 'close' column
        window: Lookback window

    Returns:
        DataFrame with 'my_feature' column added
    """
    df = df.copy()
    df['my_feature'] = df['close'].rolling(window).mean()
    return df
```

### Step 2: Register the feature

```python
# src/features/registry.py
from .my_feature import add_my_feature

register_legacy_feature(
    'my_feature',
    add_my_feature,
    category='custom',
    params={'window': 20}
)
```

### Step 3: Add to config

```yaml
# config/features.yaml
features:
  my_feature:
    category: custom
    timeframes: [D, W]
    enabled: true
    params:
      window: 20
```

### Step 4: Call from single_stock.py or cross_sectional.py

```python
# src/features/single_stock.py
from .my_feature import add_my_feature

# In compute_single_stock_features():
out = add_my_feature(out, window=20)
```

## Key Patterns

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

Use joblib with loky backend for symbol-level parallelization:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1, backend='loky')(
    delayed(process_symbol)(sym, df)
    for sym, df in symbols_dict.items()
)
```

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

```bash
# Required environment variable
export RAPIDAPI_KEY="your-key-here"

# Or use .env file
echo "RAPIDAPI_KEY=your-key-here" > .env
```

## Data Files

- `cache/stock_data_universe.parquet` - Cached stock OHLCV data
- `cache/stock_data_etf.parquet` - Cached ETF OHLCV data
- `cache/US universe_*.csv` - Symbol list with sectors
- `artifacts/features_daily.parquet` - Computed daily features
- `artifacts/features_complete.parquet` - Features with weekly
- `artifacts/targets_triple_barrier.parquet` - ML targets
