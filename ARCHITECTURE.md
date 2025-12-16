# ML Pipeline Architecture

This document describes the architecture for the momentum trading ML pipeline, including component responsibilities, update frequencies, and operational guidelines.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ONE-TIME / RARE (months)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  1. FEATURE ENGINEERING                                                 │
│     - Add new feature categories (fundamentals, sentiment, etc.)        │
│     - Requires code changes + domain expertise                          │
│     - Trigger: New data sources, strategy pivots                        │
│                                                                         │
│  2. FEATURE SELECTION                                                   │
│     - Loose-then-tight pipeline on full candidate set                   │
│     - Very expensive (~hours)                                           │
│     - Trigger: New features added, major regime shift, yearly review    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PERIODIC (quarterly/monthly)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  3. HYPERPARAMETER TUNING                                               │
│     - Optuna on FIXED feature set                                       │
│     - Moderate cost (~30-60 min)                                        │
│     - Trigger: Quarterly, or when performance degrades                  │
│     - Output: best_params.json                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        REGULAR (weekly/daily)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  4. MODEL RETRAINING                                                    │
│     - Fixed features + fixed hyperparameters                            │
│     - Train on expanding window of recent data                          │
│     - Fast (~minutes)                                                   │
│     - Trigger: Weekly or daily, before predictions                      │
│                                                                         │
│  5. PREDICTION & MONITORING                                             │
│     - Generate predictions for tradeable universe                       │
│     - Track live performance metrics                                    │
│     - Trigger: Daily                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MONITORING (continuous)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  6. PERFORMANCE DEGRADATION DETECTION                                   │
│     - Rolling AUC on recent predictions                                 │
│     - Feature importance drift                                          │
│     - Prediction distribution shifts                                    │
│     - Trigger: Automatic alerts                                         │
│                                                                         │
│  Degradation detected? → Re-run hyperopt (step 3)                       │
│  Still degraded? → Re-run feature selection (step 2)                    │
│  Still degraded? → Review feature engineering (step 1)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Feature Engineering (Rare)

**Purpose**: Create new feature categories from raw data sources.

**Scripts**:
- `src/features/*.py` - Individual feature modules
- `src/pipelines/orchestrator.py` - Feature computation pipeline
- `python -m src.cli.compute` - CLI entry point

**When to run**:
- Adding new data sources (fundamentals, sentiment, etc.)
- Strategy changes requiring different signals
- Annual review of feature coverage

**Output**:
- `artifacts/features_daily.parquet`
- `artifacts/targets_triple_barrier.parquet`

---

### 2. Feature Selection (1-2x/year)

**Purpose**: Select optimal feature subset from all candidates using Loose-then-Tight pipeline.

**Script**: `python run_feature_selection.py`

**When to run**:
- After adding new features to candidate pool
- Major market regime shift (structural break)
- Annual review
- Sustained performance degradation not fixed by hyperopt

**Output**:
```
artifacts/feature_selection/
├── selected_features.txt    # Final feature list (52 features)
├── stage_summary.csv        # Performance at each selection stage
└── ...                      # Stage-by-stage details
```

**Configuration**:
- Uses conservative hyperparameters (max_depth=6, num_leaves=31)
- 5-fold expanding window CV with 20-day embargo
- AUC metric with stability requirements

**Do NOT re-run** just because hyperparameters changed. Features should be selected with conservative model settings to find robust signals.

---

### 3. Hyperparameter Tuning (Quarterly)

**Purpose**: Optimize LightGBM hyperparameters for the fixed feature set.

**Script**: `python run_hyperopt.py --n-trials 500`

**When to run**:
- Quarterly (calendar-based)
- 10%+ relative drop in rolling AUC vs baseline
- After feature selection (once, to establish baseline)

**Output**:
```
artifacts/hyperopt/
├── best_params.json         # Optimal hyperparameters
├── final_results.json       # CV results with best params
├── trials_history.csv       # All trial results
└── param_importance.json    # Which params matter most
```

**Configuration**:
- Same CV scheme as feature selection (expanding, 20-day embargo)
- Stability-adjusted objective: `mean_auc - variance_penalty * std_auc`
- Minimum fold AUC threshold to reject unstable configs

**Do NOT re-run feature selection** with tuned hyperparameters. This creates a feedback loop that overfits to CV folds.

---

### 4. Model Retraining (Weekly)

**Purpose**: Train production model on latest data with fixed features and hyperparameters.

**Script**: `python run_training.py` (TODO: create this)

**When to run**:
- Weekly (e.g., Sunday evening before market open)
- After any upstream component changes

**Output**:
```
artifacts/models/
├── production_model.pkl     # Trained model
├── model_metadata.json      # Training date, features, params
└── feature_importance.csv   # For monitoring drift
```

**Configuration**:
- Loads features from `selected_features.txt`
- Loads params from `best_params.json`
- Trains on all available data (no CV, just fit)

---

### 5. Prediction (Daily)

**Purpose**: Generate predictions for tradeable universe.

**Script**: `python run_predict.py` (TODO: create this)

**When to run**:
- Daily, before market open
- After new data is available

**Output**:
```
artifacts/predictions/
├── predictions_YYYYMMDD.parquet  # Daily predictions
└── rankings_YYYYMMDD.csv         # Sorted by probability
```

---

### 6. Monitoring (Continuous)

**Purpose**: Detect model degradation before it impacts trading.

**Metrics to track**:
- Rolling AUC (21-day, 63-day windows)
- Prediction distribution (mean, std, percentiles)
- Feature importance stability (correlation with baseline)
- Hit rate by decile

**Alerts**:
- AUC drops below 0.55 for 5+ consecutive days
- Feature importance correlation < 0.8 vs baseline
- Prediction mean shifts > 2 std from historical

---

## Update Frequency Summary

| Component | Frequency | Trigger | Duration |
|-----------|-----------|---------|----------|
| Feature Engineering | Rare | New data sources | Hours-days |
| Feature Selection | 1-2x/year | New features, major regime shift | 2-4 hours |
| Hyperparameter Tuning | Quarterly | Calendar, performance drop | 30-60 min |
| Model Retraining | Weekly | Before predictions | 2-5 min |
| Predictions | Daily | Market open | <1 min |
| Monitoring | Continuous | Automatic | N/A |

---

## Key Principles

### 1. Separate Concerns
- **Features** answer "what to look at"
- **Hyperparameters** answer "how to learn"
- **Retraining** answers "on what data"

### 2. Stability Over Optimization
A slightly suboptimal but stable model beats a perfectly tuned model that needs constant adjustment. The variance penalty in hyperopt enforces this.

### 3. Avoid Feedback Loops
Do NOT re-run feature selection with tuned hyperparameters. The hyperparameters were optimized FOR those features - re-selecting creates circular dependency that overfits to CV folds.

### 4. Monitor Before Re-tuning
Track rolling metrics continuously. Only re-tune when there's evidence of degradation, not on a fixed schedule. False positives waste compute and introduce instability.

### 5. Conservative Feature Selection
Use simple, regularized models for feature selection. This finds features with strong, robust signals rather than features that only work with specific configurations.

### 6. No Class Balancing by Default

We train on imbalanced data (approximately 25% positive, 75% negative) **without** class weighting by default. Reasoning:

1. **Reflects reality**: The imbalanced ratio reflects actual market conditions. About 1 in 4 setups hits the target before the stop - this is the real distribution we need the model to learn.

2. **Calibrated probabilities**: Without class weighting, predicted probabilities approximate true frequencies. A 30% predicted probability means ~30% of similar setups historically hit target. This is critical for position sizing and risk management.

3. **Break-even calculation**: Our asymmetric barriers (3× ATR target, 1.5× ATR stop) create a 2:1 reward/risk ratio. The break-even probability is 33.3%. With properly calibrated probabilities, we can directly compare predictions to this threshold.

4. **Threshold flexibility**: Unweighted models allow choosing decision thresholds post-training. Want higher precision? Raise the threshold. Want more trades? Lower it. Weighted models bake in a specific operating point.

**When to use balanced training** (`--balanced` flag):
- When AUC matters more than calibration
- When the minority class is very rare (<10%)
- When you'll use fixed thresholds rather than probability-based decisions

All scripts support `--balanced` for class-weighted training via `scale_pos_weight`.

### 7. Symbol Filtering (Automatic)

The data loader automatically filters out untradeable securities before they enter the pipeline:

**Filtered by symbol pattern:**
- Warrants: `.W`, `/W`, `-W` suffixes
- Units: `.U`, `/U`, `-U` suffixes
- Rights: `.R`, `/R`, `-R` suffixes
- Preferred shares: `.PR`, `-P` patterns
- Foreign shares: `.F`, `/F` suffixes

**Filtered by description:**
- SPACs: "Acquisition Corp/Inc/Company" in name
- ADRs: "ADR", "American Depositary", "Depositary Share/Receipt"
- Blank check companies

This filtering is enabled by default in `src/data/loader.py` and removes ~40 securities from a typical 3000-stock universe. To disable:

```python
from src.data.loader import _symbols_from_csv
symbols = _symbols_from_csv(csv_path, filter_untradeable=False)
```

**Rationale**: SPACs and ADRs have fundamentally different price dynamics (merger arbitrage, currency effects) that don't fit the momentum-based signals this pipeline is designed to capture.

---

## Directory Structure

```
artifacts/
├── feature_selection/
│   ├── selected_features.txt    # 52 features (FIXED between selections)
│   ├── stage_summary.csv
│   └── ...
├── hyperopt/
│   ├── best_params.json         # Tuned params (QUARTERLY updates)
│   ├── final_results.json
│   └── trials_history.csv
├── models/
│   ├── production_model.pkl     # Current model (WEEKLY updates)
│   ├── model_metadata.json
│   └── feature_importance.csv
├── predictions/
│   ├── predictions_YYYYMMDD.parquet  # Daily predictions
│   └── rankings_YYYYMMDD.csv
└── monitoring/
    ├── rolling_metrics.parquet   # Historical performance
    └── alerts.log
```

---

## Degradation Response Playbook

```
Performance drops detected
         │
         ▼
┌─────────────────────────┐
│ Check data quality      │ ← Missing data? Stale prices?
│ (quick sanity check)    │
└─────────────────────────┘
         │ Data OK
         ▼
┌─────────────────────────┐
│ Re-run hyperopt         │ ← Maybe model needs tuning for new regime
│ (30-60 min)             │
└─────────────────────────┘
         │ Still degraded
         ▼
┌─────────────────────────┐
│ Re-run feature selection│ ← Maybe different features needed
│ (2-4 hours)             │
└─────────────────────────┘
         │ Still degraded
         ▼
┌─────────────────────────┐
│ Review strategy         │ ← Fundamental market change?
│ (manual analysis)       │   Need new feature categories?
└─────────────────────────┘
```

---

## Scripts Reference

| Script | Purpose | Command |
|--------|---------|---------|
| Feature Selection | Select optimal features | `python run_feature_selection.py` |
| Hyperopt | Tune hyperparameters | `python run_hyperopt.py --n-trials 500` |
| Training | Train production model | `python run_training.py` |
| Prediction | Generate predictions | `python run_predict.py [--date YYYY-MM-DD]` |
| Dashboard | Interactive web UI | `streamlit run app.py` |

---

## Dashboard (Streamlit)

The dashboard (`app.py`) provides three main views:

### 1. Ticker Analysis
- Enter any ticker symbol
- See probability of hitting target, target price, stop-loss
- Adjust ATR multiples for custom targets/stops
- SHAP waterfall showing feature contributions
- Price chart with target/stop levels

### 2. Top Candidates
- Browse highest probability stocks
- Filter by minimum probability, reward/risk ratio
- Sort by probability, R/R, or target %
- Distribution plot of predictions

### 3. Model Monitoring
- Feature importance chart
- Rolling AUC over time (backtest)
- Calibration analysis
- Model metadata

### Running the Dashboard

```bash
# Install dependencies
pip install -r requirements-dashboard.txt

# Run locally
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8501

# Run for network access
streamlit run app.py --server.address 0.0.0.0
```

---

## Current State

As of the last run:

- **Features**: 52 selected (0.6932 AUC ± 0.0625)
- **Hyperparameters**: Run `python run_hyperopt.py` to generate
- **Model**: Run `python run_training.py` to train
- **Predictions**: Run `python run_predict.py` to generate
- **Dashboard**: Run `streamlit run app.py` to view
