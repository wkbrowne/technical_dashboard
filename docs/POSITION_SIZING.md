# Position Sizing System

A production-quality position sizing, optimization, and backtesting system for the triple-barrier momentum strategy.

## Table of Contents

1. [Overview](#overview)
2. [Weekly Trading Protocol](#weekly-trading-protocol)
3. [Overlapping Labels and Purge/Embargo Logic](#overlapping-labels-and-purgeembargo-logic)
4. [Model Training and Calibration](#model-training-and-calibration)
5. [Position Sizing Rules](#position-sizing-rules)
6. [NAV Exposure vs Volatility-Targeted Exposure](#nav-exposure-vs-volatility-targeted-exposure)
7. [Optimizer Design](#optimizer-design)
8. [Backtest Assumptions](#backtest-assumptions)
9. [Running the Pipeline](#running-the-pipeline)
10. [Configuration Examples](#configuration-examples)
11. [Known Failure Modes and Sanity Checks](#known-failure-modes-and-sanity-checks)

---

## Overview

The position sizing system sits on top of an existing LightGBM binary classifier that predicts the probability of hitting a triple-barrier target within a 20 trading-day horizon. The system:

- **Respects time-series structure**: Proper holdouts with purging and embargo
- **Avoids leakage**: All transforms and calibration fit only on training data
- **Uses the model only for ranking**: Extracts performance via sizing, regime awareness, and exposure control
- **Produces reproducible artifacts**: Models, parameters, and backtests are versioned

### Module Structure

```
src/alpha/
├── config/              # Configuration and settings
│   ├── default.yaml     # Default configuration
│   └── settings.py      # Dataclasses and loading
├── data/                # Data schemas and loaders
│   ├── schemas.py       # SignalRecord, PositionRecord, etc.
│   └── loaders.py       # Load targets, features, predictions
├── cv/                  # Cross-validation
│   ├── purged_walkforward.py   # Purged walk-forward CV
│   └── embargo.py       # Embargo and purge utilities
├── features/            # Feature transforms
│   ├── transforms.py    # Z-score, winsorize, etc.
│   └── pipeline.py      # Sizing feature pipeline
├── models/              # Model training
│   ├── train_lgbm.py    # LightGBM training with CV
│   ├── calibration.py   # Probability calibration
│   └── artifacts.py     # Model saving/loading
├── sizing/              # Position sizing
│   ├── rules.py         # Sizing rule implementations
│   ├── constraints.py   # Portfolio constraints
│   ├── exposure.py      # Exposure calculations
│   └── optimizer.py     # Optuna optimization
├── backtest/            # Backtesting
│   ├── engine.py        # Backtest engine
│   ├── costs.py         # Transaction cost model
│   └── metrics.py       # Performance metrics
└── reporting/           # Visualization
    ├── plots.py         # Plotting utilities
    └── tearsheet.py     # Tear sheet generation
```

---

## Weekly Trading Protocol

The strategy follows a strict weekly protocol:

| Event | Timing | Description |
|-------|--------|-------------|
| **Signal Generation** | Monday close | Model generates probability scores |
| **Entry** | Monday close | Positions opened at closing prices |
| **Holding** | Up to 20 trading days | Or until barrier hit |
| **Rebalance Decision** | Friday close | Evaluate current positions, plan next week |
| **Exit** | Barrier hit or horizon | Upper barrier, lower barrier, or time expiry |

### Trade Selection

Each week, a subset of the universe is selected for trading:

1. **Top-N Selection**: Select top N stocks by model score
2. **Threshold Selection**: Select stocks with probability ≥ threshold
3. **Percentile Selection**: Select top X% of universe

The selection method is configurable via `top_n` or `top_pct` parameters.

---

## Overlapping Labels and Purge/Embargo Logic

### The Problem

Labels overlap because:
- Signals are generated **weekly** (every Monday)
- But the holding horizon is **20 trading days** (~4 weeks)
- A signal generated on Week 1 may still be "live" during Week 4

This creates information leakage if validation samples overlap with training samples.

### The Solution: Purging and Embargo

**Purging**: Remove training samples whose label window extends into the validation period.

```
Training Period     Test Period
────────────────►  ────────────►
                   │
    These samples  │  All samples
    are PURGED     │  with labels
    ─────────────► │  still "live"
                   ▼
```

**Embargo**: Gap period after test to prevent reverse leakage.

### Implementation

```python
cv = WeeklySignalCV(
    n_splits=5,
    min_train_weeks=52,    # 1 year minimum training
    test_weeks=13,         # ~1 quarter test
    purge_days=25,         # >= 20-day horizon
    embargo_days=5,        # 1 week gap
)
```

**Key Assertions** (fail loudly if violated):

```python
# All training exits must be before test entry
assert max(train_exit_dates) < min(test_entry_dates)

# No overlapping label windows
for train_sample in training_set:
    for test_sample in test_set:
        assert not labels_overlap(train_sample, test_sample)
```

### Validation

Run leakage tests:

```bash
pytest tests/test_no_leakage.py -v
pytest tests/test_purged_cv.py -v
```

---

## Model Training and Calibration

### Sample Weighting Schemes

Training uses sample weights to handle class imbalance and overlapping events:

| Scheme | Formula | Use Case |
|--------|---------|----------|
| `overlap_inverse` | `1 / (n_overlapping + 0.5)` | Default, pre-computed in targets |
| `liquidity` | `log(ADV)` normalized | Weight liquid names higher |
| `inverse_volatility` | `1 / ATR%` | Weight stable names higher |
| `atr_percent` | `ATR%` normalized | Weight volatile names higher |

**Important**: Weights use only information known at decision time (no lookahead).

### Probability Calibration

Raw model probabilities are often miscalibrated. We calibrate using:

- **Isotonic Regression**: Non-parametric, preserves ranking
- **Platt Scaling**: Logistic fit on log-odds

**Critical**: Calibration is fit **within each fold** on the validation set, not globally.

```python
# Per-fold calibration
calibrator = CalibratedClassifier(method="isotonic")
calibrator.fit(y_val, y_prob_raw)
y_prob_calibrated = calibrator.transform(y_prob_raw)
```

### Fold-Level Artifacts

Each fold produces:
- Trained model
- Fitted calibrator
- Validation metrics (AUC, AUPR, calibration error)
- Feature importance

---

## Position Sizing Rules

### A) Monotone Probability Mapping

Linear transformation of calibrated probability to weight:

```
w_raw = slope × (p_cal − intercept)
w = clip(w_raw, 0, max_weight)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slope` | 2.0 | Sensitivity to probability |
| `intercept` | 0.5 | Probability threshold for positive weight |
| `max_weight` | 0.10 | Maximum 10% per position |

**Intuition**:
- At p=0.5, weight = 0
- At p=0.75, weight = 0.5 × slope = 0.10 (at max)

### B) Rank-Bucket Sizing

Divide universe into buckets by rank, assign monotonic weights:

```
Bucket 1 (top 20%):  weight = 1.0 × base
Bucket 2 (20-40%):   weight = 0.8 × base
Bucket 3 (40-60%):   weight = 0.6 × base
...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_buckets` | 5 | Number of rank buckets |
| `bucket_weights` | [1.0, 0.8, 0.6, 0.4, 0.2] | Weight per bucket |

### C) Threshold + Confidence Sizing

Two-stage approach:

1. **Entry Gate**: Only trade if `p_cal ≥ threshold`
2. **Size by Confidence**:

```
confidence = (p_cal − threshold) / (1 − threshold)
w = confidence × confidence_scale × max_weight
```

With optional **regime multipliers**:

| Regime | Multiplier | Effect |
|--------|------------|--------|
| Low VIX | 1.2 | Increase exposure |
| Normal VIX | 1.0 | Baseline |
| High VIX | 0.8 | Reduce exposure |

### D) Volatility-Targeted Overlay

Targets portfolio-level volatility, capped by NAV exposure:

```
vol_scale = target_vol / estimated_portfolio_vol
adjusted_weights = base_weights × vol_scale
final_weights = min(adjusted_weights, max_exposure × base_weights / base_weights.sum())
```

---

## NAV Exposure vs Volatility-Targeted Exposure

### NAV Fraction Mode (Default)

**Definition**: Gross exposure as a fixed fraction of NAV.

```
gross_exposure = sum(abs(weights)) = target_fraction × NAV
```

**Example**: With `max_gross_exposure = 0.6`, if NAV = $1M:
- Maximum position value = $600K
- Cash reserve = $400K

**Advantages**:
- Simple, predictable
- Easy risk budgeting
- Capital preservation focus

### Volatility-Targeted Mode

**Definition**: Scale exposure to achieve target portfolio volatility.

```
exposure = (target_vol / realized_vol) × base_exposure
exposure = min(exposure, max_exposure_cap)
```

**Example**: With `vol_target = 0.15` (15% annual) and `max_gross_exposure = 1.0`:
- If portfolio vol = 20%, scale = 0.75
- If portfolio vol = 10%, scale = 1.5 → capped at 1.0

**Advantages**:
- Constant risk profile
- Automatic deleveraging in volatile markets
- Risk parity approach

**Important**: NAV exposure cap always applies as a ceiling.

---

## Optimizer Design

### Optuna-Based Optimization

The optimizer uses Tree-structured Parzen Estimator (TPE) to find optimal sizing parameters:

```python
optimizer = SizingOptimizer(
    config=config,
    method=SizingMethod.MONOTONE_PROBABILITY,
    n_trials=100,
    metric="penalized_return",
)
result = optimizer.optimize(signals)
```

### Optimized Parameters

| Parameter | Search Range | Description |
|-----------|--------------|-------------|
| `slope` | [1.0, 5.0] | Probability mapping slope |
| `intercept` | [0.3, 0.6] | Entry threshold |
| `entry_threshold` | [0.45, 0.7] | For threshold-based sizing |
| `exposure_mult` | [0.5, 1.5] | Exposure scaling factor |
| `turnover_penalty` | [0, 0.01] | Cost of turnover |
| `regime_*` | [0.5, 1.5] | Regime multipliers |

### Objective Function

The objective is computed **only on validation folds**:

```python
objective = mean(penalized_returns) - 0.5 × std(penalized_returns)

penalized_return = portfolio_return - turnover_penalty × turnover
```

**Key Properties**:
- Higher is better
- Penalizes high variance across folds (stability)
- Accounts for trading costs

### Parameter Importance

Optuna provides importance ranking:

```
slope:              0.35
intercept:          0.28
exposure_mult:      0.22
turnover_penalty:   0.15
```

---

## Backtest Assumptions

### Market Assumptions

| Assumption | Value | Rationale |
|------------|-------|-----------|
| Transaction costs | 20 bps round-trip | 10 bps each way (spread + impact) |
| Slippage | Included in costs | No separate modeling |
| Fill rate | 100% | All orders execute at close |
| Short selling | Not supported | Long-only strategy |

### Execution Assumptions

| Assumption | Implementation |
|------------|----------------|
| Entry timing | Monday close price |
| Exit timing | Actual barrier hit date/price |
| Rebalance | Full rebalance allowed weekly |
| Partial fills | Not modeled |

### Position Management

| Rule | Implementation |
|------|----------------|
| Horizon limit | 20 trading days |
| Early exit | On barrier hit (target or stop) |
| Forced exit | At horizon if no barrier hit |
| Position averaging | Allowed within same week |

---

## Running the Pipeline

### Quick Start (Recommended)

The unified pipeline handles everything correctly:

```bash
conda activate stocks_predictor

# Full pipeline: CV predictions + optimization + backtest
python scripts/run_sizing_pipeline.py --n-trials 100

# Quick test run (fewer optimization trials)
python scripts/run_sizing_pipeline.py --n-trials 20
```

### Step-by-Step Execution

```bash
# Step 1: Generate out-of-sample predictions via walk-forward CV
# This creates TRUE out-of-sample predictions - no leakage!
python scripts/generate_cv_predictions.py --use-hyperopt-params

# Step 2: Optimize sizing parameters using CV predictions
python scripts/run_optimize_sizing.py --n-trials 100

# Step 3: Run backtest with optimized params
python scripts/run_backtest.py \
    --config artifacts/sizing/best_params_monotone_probability.json
```

### Understanding the Two Training Approaches

| Use Case | Script | Data Used | Valid For |
|----------|--------|-----------|-----------|
| **Backtesting/Evaluation** | `generate_cv_predictions.py` | Walk-forward CV folds | Sizing optimization, strategy validation |
| **Live Trading** | `run_training.py` | All historical data | Forward-looking predictions on new data |

**CRITICAL**: Never use predictions from `run_training.py` for backtesting - that would be information leakage!

### Full Pipeline (From Scratch)

```bash
# Step 1: Ensure features and targets exist
python -m src.cli.compute --timeframes D

# Step 2: Run feature selection (optional)
python run_feature_selection.py

# Step 3: Run hyperparameter tuning (optional)
python run_model_tuning.py

# Step 4: Run the sizing pipeline (CV + optimization)
python scripts/run_sizing_pipeline.py --n-trials 200

# Step 5: Generate tear sheet
python scripts/run_report.py --output-dir artifacts/reports
```

### Production Model (For Live Trading)

```bash
# Train production model on all data
python run_training.py

# Generate predictions for latest date
python run_predict.py
```

### Output Artifacts

```
artifacts/
├── models/
│   ├── sizing_model_fold*.pkl    # Per-fold models
│   ├── sizing_model_summary.json # CV metrics
│   └── feature_importance.csv
├── sizing/
│   ├── best_params_*.json        # Optimized parameters
│   ├── trials_*.csv              # Optimization history
│   └── param_importance_*.json
├── backtests/
│   ├── backtest_equity_curve.csv
│   ├── backtest_metrics.json
│   └── backtest_tearsheet.pdf
└── reports/
    ├── report_tearsheet.pdf
    └── monthly_returns.csv
```

---

## Configuration Examples

### Conservative Configuration

Low exposure, strict controls, high threshold:

```yaml
sizing:
  method: threshold_confidence
  max_gross_exposure: 0.3      # 30% max
  max_name_weight: 0.05        # 5% per position
  entry_threshold: 0.65        # High conviction only
  top_n: 10
  turnover_penalty: 0.001

costs:
  spread_bps: 7.5              # Higher cost assumption
  slippage_bps: 7.5
```

### Moderate Configuration (Default)

Balanced approach:

```yaml
sizing:
  method: monotone_probability
  max_gross_exposure: 0.6
  max_name_weight: 0.08
  prob_slope: 2.0
  prob_intercept: 0.5
  top_n: 15
  turnover_penalty: 0.0005

costs:
  spread_bps: 5.0
  slippage_bps: 5.0
```

### Aggressive Configuration

Higher exposure, lower threshold:

```yaml
sizing:
  method: volatility_targeted
  exposure_mode: vol_targeted
  max_gross_exposure: 1.0      # Can go to 100%
  max_name_weight: 0.10
  vol_target_annual: 0.20      # 20% vol target
  entry_threshold: 0.50
  top_n: 20
```

---

## Known Failure Modes and Sanity Checks

### Failure Modes

| Issue | Symptom | Solution |
|-------|---------|----------|
| Label leakage | Unrealistic backtest performance | Check purge_days >= horizon_days |
| Transform leakage | Test metrics too good | Ensure transforms fit only on train |
| Look-ahead bias | Impossible alpha | Check all features use lagged data |
| Survivorship bias | Missing delisted symbols | Use point-in-time universe |
| Overfitting | Poor out-of-sample | Reduce optimizer trials, increase folds |

### Sanity Checks

Run before trusting results:

```bash
# 1. Check for leakage
pytest tests/test_no_leakage.py -v

# 2. Check CV implementation
pytest tests/test_purged_cv.py -v

# 3. Check backtest alignment
pytest tests/test_backtest_alignment.py -v
```

### Manual Verification

1. **Date Alignment**: Print train/test date ranges, verify no overlap
2. **Return Attribution**: Check if returns come from alpha or beta
3. **Stability**: Compare fold-to-fold metrics variance
4. **Sensitivity**: Vary parameters ±10%, check performance stability
5. **Regime Analysis**: Check performance in high vs low VIX periods

### Red Flags

- Sharpe ratio > 3.0 in backtest (likely leakage)
- Zero turnover with changing weights (implementation bug)
- All folds have identical metrics (data leakage)
- Returns uncorrelated with hit rate (implementation bug)

---

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.

---

*Last updated: 2024*
