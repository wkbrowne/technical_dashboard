# Model-Aware Featurization (Phase 1)

This document describes the 4-model feature architecture for the momentum trading system.

## Overview

The system supports four distinct models, each targeting a specific trade type:

| Model Key | Direction | Style | Target Setup |
|-----------|-----------|-------|--------------|
| `LONG_NORMAL` | Long | Standard | Impulse/transition, gap behavior, 1.5 ATR style |
| `LONG_PARABOLIC` | Long | Extended | Trend persistence, continuation plays |
| `SHORT_NORMAL` | Short | Standard | Breakdown, fragility, liquidity stress |
| `SHORT_PARABOLIC` | Short | Panic | Regime shift, vol-of-vol, capitulation |

## Feature Architecture

### CORE_FEATURES (Shared Backbone)

All models share a common backbone of curated features selected via the Loose-Tight pipeline. This includes:

- **Relative Performance / Alpha**: Cross-sectional momentum, alpha vs benchmarks
- **Macro / Intermarket**: VIX regime, credit spreads, yield curve, FRED data
- **Trend Strength**: MA slopes, trend scores, MACD
- **Price Position**: Distance to MAs, position in range
- **Volatility / Regime**: ATR, Bollinger width, vol regime
- **Sector Breadth**: McClellan oscillator, A/D line
- **Volume / Liquidity**: VWAP distance, volume shock
- **Momentum / Trend Quality**: RSI, Choppiness, ADX/DI

### HEAD_FEATURES (Model-Specific)

Each model has additive head features that augment the core:

#### LONG_NORMAL
Focus: Impulse/transition/gap behavior, squeeze setups
- Gap features: `overnight_ret`, `overnight_ratio`, `gap_fill_frac`
- Candlestick: `lower_shadow_ratio`
- Range: `range_efficiency`
- Squeeze: `squeeze_release_20`, `squeeze_intensity_20`, `days_in_squeeze_20`
- Trend quality: `adx_14`, `di_plus_14`
- Volume: `pv_divergence_5d`

#### LONG_PARABOLIC
Focus: Persistence, continuation, extended momentum
- Persistence: `trend_persist_ema`, `w_trend_persist_ema`
- Vol expansion: `atr_percent_chg_5`
- Cross-sectional: `xsec_mom_60d_z`, `w_xsec_mom_13w_z`
- Alpha (longer): `alpha_mom_spy_60_ema10`, `alpha_mom_qqq_60_ema10`
- Breakouts: `breakout_up_20d`, `range_expansion_20d`

#### SHORT_NORMAL
Focus: Breakdown, fragility, liquidity stress
- Drawdown: `drawdown_20d_z`, `drawdown_velocity_20d`, `drawdown_regime`
- Liquidity: `illiquidity_score`, `amihud_illiq_ratio`
- Volume: `volshock_dir`
- Weekly: `w_drawdown_60d_z`
- Candlestick: `lower_shadow_ratio` (inverse signal)
- Breakdown: `breakout_dn_20d`

#### SHORT_PARABOLIC
Focus: Panic, regime shift, vol-of-vol
- VIX: `vix_change_5d`, `vix_change_20d`, `vix_regime`, `w_vix_change_4w`
- Credit: `credit_spread_zscore`, `w_credit_spread_zscore`
- Correlation: `equity_bond_corr_60d`
- Gaps: `overnight_ret`, `gap_fill_frac`
- Vol explosion: `squeeze_release_20`

## Pipeline Output

The feature pipeline outputs a **single dataset** containing all computed features. Model-specific column selection happens at **training time**, not during feature production. This approach:

- Avoids redundant computation of shared features
- Allows flexible experimentation with different model configurations
- Keeps the feature pipeline simple and deterministic

```bash
# Compute features (outputs single dataset with ALL features)
python -m src.cli.compute --timeframes D,W

# Output files:
#   artifacts/features_complete.parquet  - All computed features (~600+)
#   artifacts/features_filtered.parquet  - BASE_FEATURES curated set
```

At training time, use `get_featureset()` to select the appropriate columns:

```python
import pandas as pd
from src.config.model_keys import ModelKey
from src.feature_selection.base_features import get_featureset

# Load the complete feature file
df = pd.read_parquet('artifacts/features_complete.parquet')

# Select columns for a specific model at training time
features = get_featureset(ModelKey.LONG_NORMAL)
df_model = df[['symbol', 'date'] + [f for f in features if f in df.columns]]
```

## Usage

### Getting Feature Sets

```python
from src.config.model_keys import ModelKey
from src.feature_selection.base_features import (
    get_core_features,
    get_head_features,
    get_featureset,
    get_all_selectable_features,
)

# Core features (shared across all models)
core = get_core_features()

# Head features for a specific model
head = get_head_features(ModelKey.LONG_NORMAL)

# Complete featureset for a model (CORE + HEAD)
features = get_featureset(ModelKey.LONG_NORMAL)

# With expansion candidates
features_expanded = get_featureset(ModelKey.LONG_NORMAL, include_expansion=True)

# All selectable features (union across all models)
all_features = get_all_selectable_features()

# All selectable for a specific model
model_features = get_all_selectable_features(ModelKey.SHORT_PARABOLIC)
```

### Structured Feature Set

```python
# Get structured breakdown instead of flat list
structured = get_featureset(ModelKey.LONG_NORMAL, flat=False)
# Returns: {'core': [...], 'head': [...]}

structured_expanded = get_featureset(ModelKey.LONG_NORMAL, flat=False, include_expansion=True)
# Returns: {'core': [...], 'head': [...], 'expansion': [...]}
```

### Validation

```python
import pandas as pd
from src.config.model_keys import ModelKey
from src.feature_selection.base_features import (
    validate_features,
    validate_model_featuresets,
    report_head_features_status,
)

# Load feature DataFrame
df = pd.read_parquet('artifacts/features_complete.parquet')

# Validate features for a specific model
result = validate_features(df, model_key=ModelKey.LONG_NORMAL)
print(f"Valid: {len(result['valid'])}, Missing: {result['missing']}")

# Validate all model featuresets for consistency
validation = validate_model_featuresets()
print(f"All valid: {validation['all_valid']}")

# Check head features specifically
status = report_head_features_status(df, ModelKey.SHORT_PARABOLIC)
print(status['summary'])
```

### Retired Features and Dependencies

```python
from src.feature_selection.base_features import (
    get_retired_features_safe_to_skip,
    get_features_required_for_model,
)

# Get retired features that can be skipped (excluding head features)
safe_to_skip = get_retired_features_safe_to_skip(ModelKey.LONG_NORMAL)

# Get all features required for a model (including intermediate dependencies)
required = get_features_required_for_model(ModelKey.LONG_NORMAL)
```

## Backwards Compatibility

For legacy code, the following aliases are preserved:

```python
from src.feature_selection.base_features import (
    BASE_FEATURES,      # Alias to get_featureset(LONG_NORMAL)
    get_base_features,  # Returns BASE_FEATURES (deprecated)
)
```

## Phase 2/3 Preview

- **Phase 2**: Separate model training per model_key with dedicated CV and hyperparameters
- **Phase 3**: Sizing integration combining predictions from all 4 models

## File Locations

| File | Description |
|------|-------------|
| `src/config/model_keys.py` | ModelKey enum and utilities |
| `src/feature_selection/base_features.py` | Feature registry and retrieval functions |
| `scripts/test_model_featuresets.py` | Smoke test for featureset validation |
