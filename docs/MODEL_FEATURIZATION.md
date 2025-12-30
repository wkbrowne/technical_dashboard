# Model-Aware Featurization

This document describes the 4-model feature architecture and the feature registry system for reproducible ML pipelines.

## Overview

The system supports four distinct models, each targeting a specific trade type:

| Model Key | Direction | Style | Target Setup |
|-----------|-----------|-------|--------------|
| `LONG_NORMAL` | Long | Standard | Impulse/transition, gap behavior, 1.5 ATR style |
| `LONG_PARABOLIC` | Long | Extended | Trend persistence, continuation plays |
| `SHORT_NORMAL` | Short | Standard | Breakdown, fragility, liquidity stress |
| `SHORT_PARABOLIC` | Short | Panic | Regime shift, vol-of-vol, capitulation |

## Feature Registry

The feature registry provides reproducible feature lists across the ML pipeline stages. Each model gets its own registry artifact that records exactly which features were selected.

### Registry Location

```
artifacts/<model_name>/features.json
```

Example: `artifacts/long_normal/features.json`

### Registry Schema

```json
{
  "schema_version": 1,
  "model": "long_normal",
  "created_at": "2025-01-15T10:30:00Z",
  "data_signature": "optional_dataset_version",
  "selection": {
    "baseline_k_of_n": {
      "alpha_momentum": {"k": 2, "chosen": ["w_rel_strength_sector", "xsec_mom_20d_z"]},
      "volatility_regime": {"k": 1, "chosen": ["vol_regime_ema10"]}
    },
    "include_features": [],
    "exclude_features": []
  },
  "selection_metadata": {
    "final_metric": 0.5803,
    "baseline_metric": 0.5287,
    "holdout_auc": 0.5621
  },
  "resolved_features": ["w_rel_strength_sector", "xsec_mom_20d_z", ...],
  "feature_signature": "sha256:abc123..."
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `resolved_features` | Final ordered list of features for training/inference |
| `feature_signature` | SHA256 hash of features for reproducibility checks |
| `selection.baseline_k_of_n` | K-of-N selection results per group |
| `selection_metadata` | Metrics from feature selection (CV AUC, holdout AUC) |

### API Usage

```python
from src.features.registry import (
    load_registry,
    registry_exists,
    get_registry_path,
    validate_registry,
    compare_registries,
)

# Check if registry exists
if registry_exists("long_normal"):
    registry = load_registry(get_registry_path("long_normal"))
    features = registry["resolved_features"]
    signature = registry["feature_signature"]

    # Validate integrity
    result = validate_registry(registry)
    assert result["valid"], result["issues"]
```

### Building a Registry

After feature selection, build and save a registry:

```python
from src.features.registry import (
    build_registry_from_selection,
    save_registry,
    get_registry_summary,
)

# After group selection completes
registry = build_registry_from_selection(
    model_name="long_normal",
    selected_groups=result.selected_groups,
    selection_metadata={
        "final_metric": result.final_metric,
        "baseline_metric": result.baseline_metric,
    },
)

# Save to standard location
save_registry(registry, "artifacts/long_normal/features.json")

# Print summary
print(get_registry_summary(registry))
# Output: long_normal: 23 features, sha256:abc123...
```

## Pipeline Integration

### Feature Selection Stage

`run_group_selection.py` automatically creates feature registries:

```bash
python run_group_selection.py --model long_normal
# Outputs:
#   artifacts/group_selection/group_selection_long_normal.json  (legacy)
#   artifacts/long_normal/features.json                          (registry)
```

### Training Stage

`run_training.py` reads from the registry if available:

```python
# In run_training.py
features, signature = load_model_features(model_key)
# Prefers registry, falls back to base_features.py
```

The model metadata includes the feature signature for traceability:

```json
{
  "training_date": "2025-01-15T12:00:00",
  "model_key": "long_normal",
  "n_features": 23,
  "feature_signature": "sha256:abc123...",
  "train_auc": 0.6543
}
```

### Hyperparameter Optimization

Hyperopt should load features from the registry:

```python
registry = load_registry("artifacts/long_normal/features.json")
features = registry["resolved_features"]
signature = registry["feature_signature"]

# Log signature for traceability
print(f"Optimizing with features: {signature}")
```

## Feature Architecture

### CORE_GROUPS (Shared Backbone)

All models share a common backbone of groups selected via group-first feature selection:

- **alpha_momentum**: Cross-sectional momentum, alpha vs SPY/sector benchmarks
- **macro_credit_labor**: Credit spreads, labor market indicators (FRED)
- **macro_intermarket**: Cross-asset correlations, intermarket signals
- **trend_strength**: MA slopes, trend scores, MACD
- **price_position**: Distance to MAs, position in range
- **sector_breadth**: McClellan oscillator, A/D line
- **momentum_quality**: RSI, Choppiness, ADX/DI
- **range_breakout**: Range position, efficiency, breakouts
- **volatility_regime**: VIX percentile, VIX zscore, vol regime state
- **volume_shock**: Volume shock signals, price-volume divergences
- **microstructure_position**: VWAP distance, overnight ratio
- **volatility_state**: Bollinger width, squeeze intensity, realized vol zscore
- **gap_dynamics**: Gap/ATR ratio, overnight return patterns

### HEAD_GROUPS (Model-Specific)

Each model can have additional head groups that augment the core. These are selected during model-specific feature selection.

### CANDIDATE_GROUPS

Groups available for forward selection experiments. See `src/feature_selection/base_features.py` for the full list.

## K-of-N Selection

Within each selected group, K-of-N selection chooses the most informative features:

```bash
python run_group_selection.py --model long_normal --group-k 2
```

This means: for each group, select up to 2 features (not all features in the group).

The registry records which features were chosen per group:

```json
{
  "selection": {
    "baseline_k_of_n": {
      "alpha_momentum": {
        "k": 2,
        "chosen": ["w_rel_strength_sector", "xsec_mom_20d_z"]
      }
    }
  }
}
```

## Deterministic Resolution

The registry ensures deterministic feature ordering:

1. Groups are processed in alphabetical order by name
2. Chosen features within groups preserve the selection order
3. `include_features` are sorted alphabetically
4. `exclude_features` are removed after deduplication

This means running the same selection twice produces identical `resolved_features` and `feature_signature`.

## Validation and Comparison

### Validate a Registry

```python
from src.features.registry import validate_registry

result = validate_registry(registry)
if not result["valid"]:
    print("Issues:", result["issues"])
```

### Compare Two Registries

```python
from src.features.registry import compare_registries

comparison = compare_registries(registry_old, registry_new)
print(f"Jaccard similarity: {comparison['jaccard_similarity']:.2%}")
print(f"Only in new: {comparison['only_in_b']}")
```

## Quick Commands

```bash
# Feature selection (creates registry)
python run_group_selection.py --model long_normal

# Training (reads from registry)
python run_training.py --model long_normal

# Train all models
python run_training.py --all-models
```

## File Locations

| File | Description |
|------|-------------|
| `src/features/registry.py` | Feature registry module |
| `src/feature_selection/base_features.py` | Group definitions and feature retrieval |
| `artifacts/<model>/features.json` | Per-model feature registry |
| `artifacts/group_selection/*.json` | Group selection results (legacy format) |
| `artifacts/models/<model>/model_metadata.json` | Trained model metadata with signature |

## Migration from Legacy

If you have existing `group_selection_*.json` files without registries:

1. Re-run feature selection to generate registries:
   ```bash
   python run_group_selection.py --model all
   ```

2. Or manually create registries from existing results:
   ```python
   import json
   from src.features.registry import build_registry_from_selection, save_registry

   # Load legacy result
   with open("artifacts/group_selection/group_selection_long_normal.json") as f:
       legacy = json.load(f)

   # Build registry
   registry = build_registry_from_selection(
       model_name="long_normal",
       selected_groups=legacy["selected_groups"],
   )

   save_registry(registry, "artifacts/long_normal/features.json")
   ```
