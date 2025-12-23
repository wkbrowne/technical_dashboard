# Triple Barrier Target Generation

This document describes the triple barrier target generation system, including barrier calibration and targets-only recomputation.

## Overview

The trading system uses a **4-model architecture** with distinct target definitions:

| Model Key | Direction | Trade Type | Profit Target | Stop Loss |
|-----------|-----------|------------|---------------|-----------|
| `long_normal` | Long | Standard momentum | Up barrier | Down barrier |
| `long_parabolic` | Long | Extended runners | Higher up barrier | Down barrier |
| `short_normal` | Short | Breakdown/fragility | Down barrier | Up barrier |
| `short_parabolic` | Short | Panic/capitulation | Lower down barrier | Up barrier |

### Triple Barrier Mechanics

For each entry point, we define three exit conditions:

1. **Upper barrier**: Entry price + `up_mult` × ATR
2. **Lower barrier**: Entry price - `dn_mult` × ATR
3. **Time barrier**: Maximum holding period (`max_horizon` days, default 20)

The target label is determined by which barrier is hit first:
- **+1**: Upper barrier hit first (profit for longs)
- **-1**: Lower barrier hit first (profit for shorts)
- **0**: Time expired without hitting either barrier

### Entry Sampling

Entries are sampled using `start_every` spacing (default 3 days), with adaptive advancement when barriers are hit early. This creates overlapping trajectories that are handled through sample weighting.

## Barrier Calibration

### Why Calibrate?

The default barrier multipliers (1.5-2.5 ATR) may not produce optimal label distributions for your trading objectives. Calibration allows you to:

1. **Control trade frequency**: Achieve target trades per week (e.g., 1-2)
2. **Balance classes**: Ensure sufficient positive samples for model training
3. **Adapt to market conditions**: Different volatility regimes may require different thresholds

### Running Calibration

```bash
# Standard calibration on your training data window
python scripts/calibrate_barriers.py --start 2018-01-01 --end 2024-12-31

# Target fewer trades per week (lower prevalence)
python scripts/calibrate_barriers.py --target-prevalence 0.08

# Custom stop loss (default is 1.0 ATR for risk normalization)
python scripts/calibrate_barriers.py --stop-mult 1.25

# Full options
python scripts/calibrate_barriers.py \
    --input artifacts/features_complete.parquet \
    --output artifacts/targets/barrier_calibration.json \
    --start 2018-01-01 \
    --end 2024-12-31 \
    --horizon 20 \
    --start-every 3 \
    --stop-mult 1.0 \
    --target-prevalence 0.16 \
    --parabolic-ratio 0.5 \
    --verbose
```

### Calibration Output

The calibration produces `artifacts/targets/barrier_calibration.json`:

```json
{
  "calibration_metadata": {
    "created_at": "2024-01-15T10:30:00Z",
    "date_range": {"start": "2018-01-01", "end": "2024-12-31"},
    "horizon": 20,
    "start_every": 3,
    "stop_mult": 1.0,
    "target_prevalence_normal": 0.16,
    "n_symbols": 500,
    "n_samples": 250000
  },
  "mfe_mae_statistics": {
    "long": {"mfe_mean": 1.85, "mfe_median": 1.42, ...},
    "short": {"mfe_mean": 1.72, "mfe_median": 1.35, ...}
  },
  "target_configs": {
    "long_normal": {"up_mult": 1.65, "dn_mult": 1.0, ...},
    "long_parabolic": {"up_mult": 2.35, "dn_mult": 1.0, ...},
    "short_normal": {"up_mult": 1.0, "dn_mult": 1.82, ...},
    "short_parabolic": {"up_mult": 1.0, "dn_mult": 2.48, ...}
  }
}
```

### Calibration Philosophy

The calibration computes:

1. **MFE (Maximum Favorable Excursion)**: Best price reached in favorable direction
2. **MAE (Maximum Adverse Excursion)**: Worst price reached in adverse direction

Both are measured in ATR units over the `max_horizon` window.

**Key design decisions:**

- **Fixed stop at 1.0 ATR**: Risk normalization across all trades. All losing trades lose approximately the same in volatility-adjusted terms.
- **Target-first rate calibration**: Profit targets are calibrated to achieve the desired **target-first rate** - the fraction of trajectories that hit the profit target BEFORE hitting the stop. This is the actual label prevalence in training data.
- **Asymmetric long/short**: Long and short distributions often differ; each is calibrated independently.

### Target-First vs MFE-Based Calibration

The calibration uses **target-first rates** rather than raw MFE quantiles. This distinction is important:

| Approach | What it measures | Problem |
|----------|------------------|---------|
| MFE quantile | % of trajectories where price *ever* reached target | Overestimates wins - some hit target AFTER being stopped out |
| Target-first rate | % of trajectories where target was hit BEFORE stop | Accurate measure of actual trading wins |

**Example**: At 3.0 ATR target with 1.0 ATR stop:
- MFE >= 3.0 ATR: ~15% of trajectories
- Target hit before stop: ~10% of trajectories

The 5% difference represents trajectories where the stock rallied to 3 ATR *after* first dropping 1 ATR (triggering our stop). Using MFE alone would produce inflated profit targets.

**Typical calibrated values** (for 16% target prevalence):
- Long normal: ~2.5-2.7 ATR profit target
- Short normal: ~2.5-2.7 ATR profit target
- Parabolic: ~3.5-4.5 ATR profit target

### Freezing Calibration

**Important**: Calibration should be computed ONCE on a fixed training window and then frozen. Do not recalibrate dynamically in live trading, as this introduces look-ahead bias.

Recommended workflow:
1. Run calibration on your training data window
2. Save the calibration JSON
3. Use the same calibration for all subsequent target generation
4. Only recalibrate when significantly expanding training data

## Targets-Only Recomputation

### When to Use

Recompute targets without running the full feature pipeline when:

1. You've recalibrated barrier thresholds
2. You want to test different target configurations
3. You need to regenerate targets quickly for experimentation

### Running Recomputation

```bash
# Using default TARGET_CONFIGS
python scripts/recompute_targets.py

# Using calibrated thresholds
python scripts/recompute_targets.py --target-config artifacts/targets/barrier_calibration.json

# Specific models only
python scripts/recompute_targets.py --model-keys long_normal,short_normal

# Custom date range and output
python scripts/recompute_targets.py \
    --input artifacts/features_complete.parquet \
    --output artifacts/targets/targets.parquet \
    --target-config artifacts/targets/barrier_calibration.json \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --verbose
```

### Output Schema

The targets parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Stock ticker |
| `t0` | datetime | Entry date |
| `t_hit` | datetime | Exit date (barrier or time) |
| `entry_px` | float | Entry price |
| `atr_at_entry` | float | ATR at entry |
| `hit_<model>` | int | Target label (-1, 0, +1) |
| `ret_<model>` | float | Log return from entry |
| `h_used_<model>` | int | Days until exit |
| `n_overlapping_trajs` | int | Overlap count |
| `weight_overlap` | float | Uniqueness weight |
| `weight_class_balance` | float | Class balance weight |
| `weight_final` | float | Combined sample weight |

## Trade Frequency Guidance

### Label Prevalence vs Trades per Week

Label prevalence is a **proxy** for trade frequency. Actual trades depend on:
- Model probability thresholds
- Top-N selection across universe
- Market conditions

Approximate mapping (for ~500 symbol universe, top-5 selection, weekly rebalance):

| Prevalence | Expected Trades/Week |
|------------|---------------------|
| 5-8% | 0.5-1 |
| 8-12% | 1-2 |
| 12-15% | 2-3 |
| 15-20% | 3-5 |

### Recommended Settings

For **rare trading (1-2 trades/week)**:
```bash
python scripts/calibrate_barriers.py --target-prevalence 0.10 --stop-mult 1.0
```

For **moderate trading (2-4 trades/week)**:
```bash
python scripts/calibrate_barriers.py --target-prevalence 0.15 --stop-mult 1.0
```

## Data Validation

All target generation uses centralized validation from `src/validation/target_data.py`. This ensures consistent filtering across:
- Calibration (`scripts/calibrate_barriers.py`)
- Target recomputation (`scripts/recompute_targets.py`)
- Main pipeline (`src/pipelines/orchestrator.py`)

### Validation Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| `max_price` | $50,000 | Maximum price per share |
| `min_price` | $0.01 | Minimum price per share |
| `max_price_ratio` | 1000 | Max price range ratio (catches unadjusted splits) |
| `min_atr` | 0.0 | ATR must be positive |
| `max_mfe_mae_atr` | 50 | Max MFE/MAE in ATR units (filters outliers) |
| `ret_zscore_threshold` | 5.0 | Z-score threshold for return outliers |
| `ret_abs_threshold` | 1.0 | Max absolute log return (~170% linear) |
| `min_rows_per_symbol` | 25 | Minimum data points per symbol |

### Using Custom Thresholds

```python
from src.validation.target_data import TargetDataValidator, ValidationConfig

# Create custom config
config = ValidationConfig(
    max_price=100_000,  # Allow higher priced stocks
    max_mfe_mae_atr=30,  # Stricter outlier filtering
)

# Use validator
validator = TargetDataValidator(config)
df, summary = validator.validate_input(df)
```

## Programmatic Usage

### Loading Calibrated Configs

```python
from src.config.model_keys import (
    load_target_configs_from_file,
    override_target_configs,
    ModelKey
)

# Load from calibration file
configs = load_target_configs_from_file("artifacts/targets/barrier_calibration.json")

# Or use the convenience function (returns defaults if path is None)
configs = override_target_configs("artifacts/targets/barrier_calibration.json")

# Access specific config
long_normal_config = configs[ModelKey.LONG_NORMAL]
print(f"Up mult: {long_normal_config['up_mult']}")
print(f"Down mult: {long_normal_config['dn_mult']}")
```

### Generating Targets with Custom Config

```python
from src.features.target_generation import generate_multi_targets_parallel
from src.config.model_keys import override_target_configs, ModelKey
import pandas as pd

# Load data
df = pd.read_parquet("artifacts/features_complete.parquet")
df = df[['symbol', 'date', 'close', 'high', 'low', 'atr14']].copy()
df = df.rename(columns={'atr14': 'atr'})

# Load calibrated configs
configs = override_target_configs("artifacts/targets/barrier_calibration.json")

# Generate targets
# Note: This temporarily modifies TARGET_CONFIGS, so use with care in production
targets = generate_multi_targets_parallel(
    df,
    model_keys=list(configs.keys()),
    n_jobs=-1
)
```

## Integration with Training Pipeline

After recomputing targets, ensure your training scripts use the new file:

```bash
# Default location (automatically used by training)
python scripts/recompute_targets.py --output artifacts/targets_triple_barrier.parquet

# Or copy custom output to standard location
cp artifacts/targets/targets.parquet artifacts/targets_triple_barrier.parquet
```

## Troubleshooting

### Low Positive Sample Count

If calibration warns about low positive samples:
1. Increase `--target-prevalence` (e.g., 0.15 instead of 0.10)
2. Expand the calibration date range
3. Consider using all 4 models instead of just 2

### Unstable Calibration

If calibration results vary significantly:
1. Ensure sufficient samples (>100,000 entry points)
2. Use a longer date range spanning multiple market regimes
3. Check for data quality issues (gaps, outliers)

### Memory Issues

For large datasets:
```bash
# Process with fewer parallel workers
python scripts/recompute_targets.py --n-jobs 4

# Or process date ranges separately
python scripts/recompute_targets.py --start 2018-01-01 --end 2020-12-31
python scripts/recompute_targets.py --start 2021-01-01 --end 2024-12-31
```

## Related Documentation

- [docs/FEATURE_PIPELINE_ARCHITECTURE.md](FEATURE_PIPELINE_ARCHITECTURE.md) - Full pipeline details
- [docs/FEATURE_SELECTION.md](FEATURE_SELECTION.md) - Feature selection for each model
- [src/config/model_keys.py](../src/config/model_keys.py) - Default TARGET_CONFIGS
