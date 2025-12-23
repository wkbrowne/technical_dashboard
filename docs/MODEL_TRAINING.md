# Model Training & Hyperparameter Optimization

LightGBM training pipeline for triple barrier classification models. This document covers hyperparameter optimization and production model training.

> **Prerequisites**: Complete feature selection first. See [FEATURE_SELECTION.md](FEATURE_SELECTION.md).

---

## Workflow

```
Feature Selection       Hyperopt               Production Training
     │                     │                          │
     │                     │                          │
     ▼                     ▼                          ▼
selected_features.txt → best_params.json → production_model.pkl
```

```bash
# Full workflow
python run_feature_selection.py --n-jobs 8    # Step 1: Features
python run_model_tuning.py --n-trials 200     # Step 2: Hyperparams
python run_training.py                        # Step 3: Final model
```

---

## Sample Weights Strategy

Triple barrier targets create ~6-7 overlapping trajectories per time period. Sample weights correct for this correlation.

| Stage | Sample Weights? | Reason |
|-------|-----------------|--------|
| Feature selection | **No** | Need low variance to detect ε=0.0002 improvements |
| Hyperparameter optimization | **Yes** | Robust parameters on properly-weighted data |
| Production training | **Yes** | Correct for correlation, better generalization |

---

## Part 1: Hyperparameter Optimization

### Composite Objective

Optimizes multiple financial metrics (not just AUC):

| Component | Weight | Metric |
|-----------|--------|--------|
| Discrimination | 35% | AUC (20%) + AUPR (15%) |
| Calibration | 15% | 1 - Brier/0.25 |
| Tail performance | 30% | Precision@10% (20%) + Spread (10%) |
| Stability | 20% | 1 - CV coefficient×5 |

### Search Space

| Parameter | Range | Scale |
|-----------|-------|-------|
| max_depth | 3-10 | linear |
| num_leaves | 15-127 | ≤ 2^max_depth |
| min_child_samples | 20-500 | linear |
| learning_rate | 0.01-0.15 | log |
| n_estimators | 100-800 | linear |
| reg_alpha, reg_lambda | 1e-4 to 10 | log |
| subsample, colsample_bytree | 0.5-1.0 | linear |

### Pruning

Trials pruned early if:
- Any fold AUC < 0.54
- Brier > 0.26
- CV coefficient > 0.20 (after 3 folds)

### CLI

```bash
python run_model_tuning.py --n-trials 200 --n-jobs 8           # Composite (default)
python run_model_tuning.py --objective auc --variance-penalty 1.0  # AUC-only
python run_model_tuning.py --resume                            # Resume study
python run_model_tuning.py --timeout 3600                      # 1-hour limit
```

### Output

```
artifacts/hyperopt/
├── best_params.json      # Best hyperparameters + metrics
├── trials_history.csv    # All trial results
├── param_importance.json # Parameter importance ranking
└── study.db              # SQLite for resume capability
```

---

## Part 2: Production Training

Trains the final model using:
- Selected features from `artifacts/feature_selection/selected_features.txt`
- Best hyperparameters from `artifacts/hyperopt/best_params.json`
- All available historical data with sample weights

### CLI

```bash
python run_training.py                           # Default (with sample weights)
python run_training.py --n-jobs 8                # Parallel threads
python run_training.py --output-dir artifacts/models  # Custom output
python run_training.py --balanced                # Force class balancing
python run_training.py --no-balanced             # Force no class balancing
python run_training.py --no-sample-weights       # Disable overlap inverse weights
```

### Output

```
artifacts/models/
├── production_model.pkl      # Trained LightGBM model (pickle)
├── feature_importance.csv    # Feature importance ranking
└── model_metadata.json       # Training config, metrics, date range
```

### Model Metadata

```json
{
  "training_date": "2025-01-15T10:30:00",
  "n_features": 38,
  "features": ["rsi_14", "atr_percent", "..."],
  "params": {"max_depth": 9, "learning_rate": 0.09, "..."},
  "n_estimators": 479,
  "train_auc": 0.87,
  "train_aupr": 0.68,
  "n_samples": 682000,
  "positive_rate": 0.24,
  "date_range": ["2021-01-01", "2025-11-18"],
  "balanced": false,
  "scale_pos_weight": null,
  "sample_weights_used": true
}
```

---

## Metric Targets

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| AUC | 0.60-0.65 | 0.65-0.70 | > 0.70 |
| AUPR | 0.50-0.58 | 0.58-0.65 | > 0.65 |
| Brier | 0.23-0.25 | 0.20-0.23 | < 0.20 |
| Precision@10% | 0.50-0.55 | 0.55-0.60 | > 0.60 |
| CV(AUC) | 0.10-0.15 | 0.05-0.10 | < 0.05 |

---

## File Dependencies

```
Input files:
├── artifacts/features_daily.parquet              # Computed features
├── artifacts/targets_triple_barrier.parquet      # Triple barrier labels
├── artifacts/feature_selection/selected_features.txt  # From run_feature_selection.py
└── artifacts/hyperopt/best_params.json           # From run_model_tuning.py (optional)

Output files:
└── artifacts/models/
    ├── production_model.pkl
    ├── feature_importance.csv
    └── model_metadata.json
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| All trials pruned | Lower `min_auc` in PruningConfig (run_model_tuning.py) |
| High CV variance | Check data quality or reduce model complexity |
| OOM errors | Lower `--n-jobs` or cap `num_leaves` |
| Missing features warning | Re-run feature pipeline or check column names |
| KeyError on columns | Verify targets parquet schema matches expected |
| Training uses wrong features | Check `selected_features.txt` exists |
