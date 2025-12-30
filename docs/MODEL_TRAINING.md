# Model Training & Hyperparameter Optimization

LightGBM training pipeline for the 4-model triple barrier classification system.

> **Prerequisites**: Complete feature selection first. See [FEATURE_SELECTION.md](FEATURE_SELECTION.md).

---

## 1. Overview

The training pipeline supports four distinct models, each targeting different market regimes:

| Model | Description | Target |
|-------|-------------|--------|
| `long_normal` | Standard long momentum | 1.5 ATR up barrier |
| `long_parabolic` | Extended momentum / runners | 2.5 ATR up barrier |
| `short_normal` | Breakdown / fragility | 2.0 ATR down barrier |
| `short_parabolic` | Panic / regime shift | 2.5 ATR down barrier |

Each model uses:
- **CORE features**: Shared backbone features across all models
- **HEAD features**: Model-specific additive features
- **Model-specific targets**: Per-model triple barrier labels (`hit_long_normal`, etc.)

---

## 2. Quick Start

```bash
# Full workflow for all 4 models (recommended)
python scripts/run_feature_selection_multimodel.py --update-registry
python run_model_tuning.py --all-models --rounds 2 --trials-per-round 100
python run_training.py --all-models

# Single model with multi-round HPO
python run_model_tuning.py --model long_normal --rounds 3 --trials-per-round 150
python run_training.py --model long_normal

# Quick iteration (single round, legacy behavior)
python run_model_tuning.py --model long_normal --rounds 1 --n-trials 100
```

**Workflow Summary:**
```
Feature Selection       Hyperopt                   Production Training
     │                     │                              │
     ▼                     ▼                              ▼
base_features.py → model_configs.json → training_registry.json
(CORE + HEAD)      (per-model params)   (trained models)
```

---

## 3. Architecture

### 3.1 Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     INPUTS      │     │   HYPEROPT      │     │    TRAINING     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────────────────────────────┐
│ features_       │     │          MULTI-ROUND HPO                │
│ complete.parquet│────▶│  ┌─────────────────────────────────┐    │
│ (all features)  │     │  │ Round 1: Broad Search Space     │    │
└─────────────────┘     │  │  - 100+ trials                  │    │
                        │  │  - Prune on folds [3,4]         │    │
┌─────────────────┐     │  │  - Identify elite trials        │    │
│ targets_triple_ │     │  └───────────────┬─────────────────┘    │
│ barrier.parquet │────▶│                  ▼                      │
│ (hit_*, weights)│     │  ┌─────────────────────────────────┐    │
└─────────────────┘     │  │ Refine: Narrow Search Space     │    │
                        │  │  - Quantile-based bounds        │    │
┌─────────────────┐     │  │  - Elite parameter analysis     │    │
│ base_features.py│     │  └───────────────┬─────────────────┘    │
│ (CORE + HEAD)   │────▶│                  ▼                      │
└─────────────────┘     │  ┌─────────────────────────────────┐    │
                        │  │ Round 2+: Focused Search        │    │
                        │  │  - Narrowed parameter ranges    │    │
                        │  │  - Better convergence           │    │
                        │  └─────────────────────────────────┘    │
                        └────────────────┬────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │           OUTPUTS                       │
                        │                                         │
                        │  artifacts/hyperopt/                    │
                        │  ├── model_configs.json   (registry)    │
                        │  └── {model_key}/                       │
                        │      ├── best_params.json               │
                        │      └── study_round_*.db               │
                        │                                         │
                        │  artifacts/hpo/{model_key}/{run_id}/    │
                        │  ├── search_space_round_*.json          │
                        │  ├── best_params_round_*.json           │
                        │  ├── pruning_stats_round_*.json         │
                        │  └── report.md                          │
                        └────────────────┬────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │       PRODUCTION TRAINING               │
                        │                                         │
                        │  - Load best hyperparams                │
                        │  - Full dataset training                │
                        │  - Sample weights applied               │
                        │                                         │
                        │  artifacts/models/{model_key}/          │
                        │  ├── production_model.pkl               │
                        │  ├── feature_importance.csv             │
                        │  └── model_metadata.json                │
                        └─────────────────────────────────────────┘
```

### 3.2 Sample Weights Strategy

Triple barrier targets create ~6-7 overlapping trajectories per time period. Sample weights correct for this correlation.

| Stage | Sample Weights? | Reason |
|-------|-----------------|--------|
| Feature selection | **No** | Need low variance to detect ε=0.0002 improvements |
| Hyperparameter optimization | **Yes** | Robust parameters on properly-weighted data |
| Production training | **Yes** | Correct for correlation, better generalization |

---

## 4. Hyperparameter Optimization

### 4.1 CLI

```bash
# Default: 2 rounds with future-biased pruning on folds [3,4]
python run_model_tuning.py --model long_normal

# Multi-round with explicit settings
python run_model_tuning.py --model long_normal --rounds 3 --trials-per-round 150

# All models with weighted CV scoring
python run_model_tuning.py --all-models --cv-score weighted

# Legacy mode (single round, prune on fold 0)
python run_model_tuning.py --model long_normal --rounds 1 --prune-folds 0

# AUC-only objective
python run_model_tuning.py --model long_normal --objective auc --variance-penalty 1.0

# Importance-based refinement (advanced)
python run_model_tuning.py --model long_normal --refine-method importance

# Resume interrupted study
python run_model_tuning.py --model long_normal --resume
```

### 4.2 Multi-Round HPO

The HPO pipeline runs in multiple rounds, refining the search space after each round:

1. **Round 1**: Explores the full search space
2. **Refinement**: Computes quantile bounds from top elite trials
3. **Round 2+**: Searches within narrowed bounds

This approach finds better hyperparameters with fewer total trials by focusing on promising regions.

**Suggested Budgets:**

| Scenario | Rounds | Trials/Round | Total | Command |
|----------|--------|--------------|-------|---------|
| Thorough | 3 | 150 | 450 | `--rounds 3 --trials-per-round 150` |
| Standard | 2 | 100 | 200 | `--rounds 2 --trials-per-round 100` |
| Quick | 1 | 100 | 100 | `--rounds 1 --n-trials 100` |

### 4.3 Future-Biased Pruning

By default, trials are pruned based on performance on the **most recent CV folds** (folds 3 and 4 of a 5-fold CV). This prevents optimizing for older, potentially stale market regimes.

```
CV Folds (chronological order):
  Fold 0 ─────┐
  Fold 1 ─────┤  Evaluated after prune folds pass
  Fold 2 ─────┘
  Fold 3 ─────┐  [PRUNE FOLDS] Evaluated first
  Fold 4 ─────┘  Pruning decision based on these
```

**Pruning Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--prune-folds` | `3,4` | Comma-separated fold indices for pruning |
| `--prune-metric` | `mean` | How to aggregate prune fold AUCs (`mean` or `min`) |
| `--prune-margin` | `0.002` | Margin for relative pruning |
| `--prune-min-completed` | `30` | Min trials before relative pruning |
| `--eval-all-folds` | `true` | Evaluate all folds for surviving trials |

### 4.4 CV Score Weighting

The final objective score can weight later folds more heavily:

```bash
# Mean scoring (default)
python run_model_tuning.py --model long_normal --cv-score mean

# Weighted scoring (later folds weighted more)
python run_model_tuning.py --model long_normal --cv-score weighted
```

Weighted scoring uses weights `[0.5, 0.75, 1.0, 1.25, 1.5]` for folds 0-4, favoring recent performance.

### 4.5 Search Space Refinement

After each round, the search space is narrowed based on elite trials:

**Refinement Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--refine-method` | `quantile` | `quantile` (simple) or `importance` (advanced) |
| `--elite-frac` | `0.15` | Fraction of trials to consider elite |
| `--refine-quantiles` | `0.1,0.9` | Quantile bounds for numeric params |
| `--refine-padding` | `0.10` | Padding factor to avoid premature collapse |
| `--min-range-frac` | `0.20` | Minimum range as fraction of original |

**Quantile Mode** (default): Simple narrowing based on parameter quantiles from elite trials.

**Importance Mode**: Uses Optuna's parameter importance to freeze low-importance parameters and focus on high-impact ones.

### 4.6 Cross-Validation with Embargo

The hyperopt script automatically applies a **20-day embargo gap** between training and test sets to prevent lookahead leakage. This matches the `max_horizon` used in triple barrier target generation.

```
Training Data          Gap        Test Data
[────────────────────] [20 days] [────────]
                       ↑
                    Embargo
```

Use `--gap` to customize the embargo period if your targets use a different horizon.

### 4.7 Composite Objective

The hyperopt objective uses a **"mean minus SE penalty"** formulation that aligns with the SNR-based gating in feature selection:

```
objective = S_mean - λ × S_se
```

Where:
- `S_fold[i]` = per-fold composite score (weighted sum of metrics)
- `S_mean` = mean of per-fold scores
- `S_se` = std(S_fold) / sqrt(n_folds) (standard error)
- `λ` = `lambda_stability` penalty coefficient (default: 0.5)

**Per-fold composite score** (computed before aggregation):

| Component | Weight | Metric |
|-----------|--------|--------|
| Discrimination | 40% | AUC (25%) + AUPR (15%) |
| Calibration | 15% | 1 - clamp(Brier/0.25, 0, 1) |
| Tail performance | 45% | Precision@10% (25%) + Spread (20%) |

**Why per-fold first, then aggregate?**
- Ensures the stability penalty reflects true fold-to-fold variance in the composite objective, not just AUC variance
- Aligns with financial intuition: we want stable *overall* performance, not just stable discrimination
- Consistent with SNR gating used in feature selection (SE-based, not CV-coefficient-based)

**AUC floor constraint**: Trials that sacrifice AUC below `baseline - 0.002` are pruned, preventing the composite from trading discrimination for other metrics.

**Fold weighting**: In `--cv-score weighted` mode, later folds get higher weight (recency bias) but SE penalty is skipped (proper weighted variance requires careful effective-N handling).

### 4.8 Hyperparameter Philosophy

The search space is designed to **prevent overfitting** on noisy financial data while allowing sufficient model capacity:

1. **Fixed n_estimators with early stopping**: Rather than tuning tree count, we fix `n_estimators=5000` and rely on early stopping (`stopping_rounds=50`). This lets the model find its natural stopping point based on validation performance, avoiding both underfitting (too few trees) and overfitting (too many).

2. **Aggressive regularization via tree structure**: The key regularization levers are `num_leaves` and `min_child_samples`. We enforce `min_child_samples >= 2 × num_leaves` to ensure each leaf has statistically meaningful sample sizes. This prevents the model from memorizing noise in thin leaves.

3. **Log-scale sampling for capacity parameters**: Both `num_leaves` and `min_child_samples` span orders of magnitude, so log-scale sampling explores the space more efficiently than linear.

4. **Categorical max_depth**: Including `-1` (unlimited) lets LightGBM's leaf-wise growth determine depth naturally, constrained only by `num_leaves`. The discrete options `[−1, 4, 6, 8, 10]` cover the realistic range without wasting trials on fine-grained depth tuning.

5. **Tight min_split_gain**: Capped at `[0, 0.2]` because higher values rarely help and can prevent useful splits. Most gains come from tree structure constraints, not split gain thresholds.

### 4.9 Search Space

| Parameter | Range | Scale | Notes |
|-----------|-------|-------|-------|
| n_estimators | 5000 (fixed) | — | Early stopping finds optimal count |
| max_depth | [-1, 4, 6, 8, 10] | categorical | -1 = unlimited (leaf-wise) |
| num_leaves | 16-256 | log (via exponent) | Constrained by max_depth when set |
| min_child_samples | 50-3000 | log | Enforced ≥ 2 × num_leaves |
| learning_rate | 0.01-0.15 | log | |
| reg_alpha, reg_lambda | 1e-4 to 10 | log | L1/L2 regularization |
| min_split_gain | 0-0.2 | linear | Tight range, rarely needs tuning |
| subsample, colsample_bytree | 0.5-1.0 | linear | Row/column subsampling |

**Complexity constraint**: `min_child_samples >= 2 × num_leaves` ensures adequate samples per leaf. With 256 leaves, you need at least 512 samples per leaf—this prevents overfitting on thin data slices common in financial time series.

### 4.10 Pruning Thresholds

Trials are pruned early based on absolute thresholds:

| Threshold | Value | Applied On |
|-----------|-------|------------|
| min_auc | 0.54 | Any prune fold |
| max_brier | 0.26 | Any prune fold |
| max_cv_coef | 0.20 | All folds (post-prune) |

**Relative pruning** also activates after `--prune-min-completed` trials, pruning trials that fall below `best_prune_metric - prune_margin`.

### 4.11 Output

HPO produces outputs in two locations:

**Legacy outputs** (backwards compatible):
```
artifacts/hyperopt/
├── model_configs.json        # Combined config registry (all models)
├── long_normal/
│   ├── best_params.json      # Best hyperparameters + metrics
│   ├── study_round_0.db      # Optuna SQLite per round
│   ├── study_round_1.db
│   └── ...
└── ...
```

**Detailed artifacts** (new):
```
artifacts/hpo/{model_key}/{run_id}/
├── search_space_round_0.json     # Search space at start of round
├── search_space_round_1.json     # Refined search space
├── best_params_round_0.json      # Best params per round
├── best_params_round_1.json
├── trials_round_0.csv            # All trials per round
├── trials_round_1.csv
├── param_importance_round_0.json # Parameter importance
├── pruning_stats_round_0.json    # Pruning behavior
├── final_best_params.json        # Global best across all rounds
└── report.md                     # Markdown summary report
```

### 4.12 Model Config Registry

The combined `model_configs.json` tracks all model configurations:

```json
{
  "models": {
    "long_normal": {
      "hyperparameters": { "max_depth": 6, ... },
      "metrics": { "auc_mean": 0.70, "aupr_mean": 0.39, ... },
      "features": { "count": 63, "core_count": 50, "head_count": 13 },
      "target_config": { "up_mult": 1.5, "dn_mult": 1.5, "max_horizon": 20 },
      "tuning_timestamp": "2025-01-15T10:30:00"
    },
    ...
  },
  "updated": "2025-01-15T12:00:00"
}
```

---

## 5. Production Training

### 5.1 CLI

```bash
# Train all 4 models
python run_training.py --all-models

# Train specific model
python run_training.py --model long_normal

# With options
python run_training.py --model long_normal --n-jobs 8 --balanced
python run_training.py --all-models --no-sample-weights
```

### 5.2 Output

Per-model outputs in `artifacts/models/{model_key}/`:

```
artifacts/models/
├── training_registry.json    # Combined training registry (all models)
├── long_normal/
│   ├── production_model.pkl      # Trained LightGBM model
│   ├── feature_importance.csv    # Feature importance ranking
│   └── model_metadata.json       # Training config, metrics
├── long_parabolic/
│   └── ...
├── short_normal/
│   └── ...
└── short_parabolic/
    └── ...
```

### 5.3 Model Metadata

Each model's `model_metadata.json` includes:

```json
{
  "training_date": "2025-01-15T10:30:00",
  "model_key": "long_normal",
  "n_features": 63,
  "features": ["rsi_14", "atr_percent", "..."],
  "params": {"max_depth": 6, "learning_rate": 0.09, "..."},
  "n_estimators": 479,
  "train_auc": 0.87,
  "train_aupr": 0.68,
  "n_samples": 682000,
  "positive_rate": 0.24,
  "date_range": ["2021-01-01", "2025-11-18"],
  "target_config": {"up_mult": 1.5, "dn_mult": 1.5, "max_horizon": 20},
  "feature_breakdown": {"core_count": 50, "head_count": 13},
  "sample_weights_used": true
}
```

### 5.4 Training Registry

The combined `training_registry.json` provides quick access to all trained models:

```json
{
  "models": {
    "long_normal": {
      "model_path": "artifacts/models/long_normal/production_model.pkl",
      "metadata_path": "artifacts/models/long_normal/model_metadata.json",
      "train_auc": 0.87,
      "train_aupr": 0.68,
      "n_samples": 682000,
      "training_date": "2025-01-15T10:30:00"
    },
    ...
  },
  "updated": "2025-01-15T12:00:00"
}
```

---

## 6. Reference

### 6.1 Feature Architecture

Features are loaded from the model-aware registry in `src/feature_selection/base_features.py`:

```python
from src.config.model_keys import ModelKey
from src.feature_selection.base_features import get_featureset, CORE_FEATURES, HEAD_FEATURES

# Get features for a specific model
features = get_featureset(ModelKey.LONG_NORMAL)  # CORE + HEAD

# Check feature counts
print(f"CORE: {len(CORE_FEATURES)} features")
print(f"HEAD[LONG_NORMAL]: {len(HEAD_FEATURES[ModelKey.LONG_NORMAL])} features")
```

See [MODEL_FEATURIZATION.md](MODEL_FEATURIZATION.md) for detailed feature documentation.

### 6.2 Metric Targets

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| AUC | 0.60-0.65 | 0.65-0.70 | > 0.70 |
| AUPR | 0.50-0.58 | 0.58-0.65 | > 0.65 |
| Brier | 0.23-0.25 | 0.20-0.23 | < 0.20 |
| Precision@10% | 0.50-0.55 | 0.55-0.60 | > 0.60 |
| S_se (composite SE) | > 0.02 | 0.01-0.02 | < 0.01 |

### 6.3 File Dependencies

```
Input files:
├── artifacts/features_complete.parquet         # Computed features
├── artifacts/targets_triple_barrier.parquet    # Triple barrier labels (all 4 model targets)
└── src/feature_selection/base_features.py      # CORE + HEAD feature registry

Intermediate files (from hyperopt):
└── artifacts/hyperopt/
    ├── model_configs.json                      # Combined hyperparameter registry
    └── {model_key}/best_params.json            # Per-model hyperparameters

Output files:
└── artifacts/models/
    ├── training_registry.json                  # Combined training registry
    └── {model_key}/
        ├── production_model.pkl
        ├── feature_importance.csv
        └── model_metadata.json
```

### 6.4 Loading Trained Models

```python
import pickle
import json
from pathlib import Path
from src.config.model_keys import ModelKey

def load_model(model_key: ModelKey):
    """Load a trained model and its metadata."""
    base_path = Path(f'artifacts/models/{model_key.value}')

    with open(base_path / 'production_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(base_path / 'model_metadata.json') as f:
        metadata = json.load(f)

    return model, metadata

# Load all models
models = {}
for mk in ModelKey.all_keys():
    models[mk] = load_model(mk)

# Or use the training registry
with open('artifacts/models/training_registry.json') as f:
    registry = json.load(f)

for model_key, info in registry['models'].items():
    print(f"{model_key}: AUC={info['train_auc']:.4f}")
```

### 6.5 Troubleshooting

| Issue | Solution |
|-------|----------|
| All trials pruned | Lower `min_auc` in PruningConfig (run_model_tuning.py) |
| High CV variance | Check data quality or reduce model complexity |
| OOM errors | Lower `--n-jobs` or cap `num_leaves` |
| Missing features warning | Re-run feature pipeline or check base_features.py |
| Model-specific target not found | Run target generation with 4-model support |
| KeyError on model_key | Ensure model key is one of: long_normal, long_parabolic, short_normal, short_parabolic |
| Hyperparams not found | Run `run_model_tuning.py` for the specific model first |
