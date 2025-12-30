# Position Sizing System

A production-quality position sizing, optimization, and backtesting system for the triple-barrier momentum strategy.

## Table of Contents

1. [Overview](#overview)
2. [Multi-Model Sizing System](#multi-model-sizing-system)
3. [Weekly Trading Protocol](#weekly-trading-protocol)
4. [Overlapping Labels and Purge/Embargo Logic](#overlapping-labels-and-purgeembargo-logic)
5. [Model Training and Calibration](#model-training-and-calibration)
6. [Position Sizing Rules](#position-sizing-rules)
7. [NAV Exposure vs Volatility-Targeted Exposure](#nav-exposure-vs-volatility-targeted-exposure)
8. [Regime Gating](#regime-gating) (Direction-aware)
9. [Short Selectivity](#short-selectivity)
10. [Justification Outputs](#justification-outputs)
11. [Optimizer Design](#optimizer-design)
12. [Backtest Assumptions](#backtest-assumptions)
13. [Running the Pipeline](#running-the-pipeline)
14. [Configuration Examples](#configuration-examples)
15. [Known Failure Modes and Sanity Checks](#known-failure-modes-and-sanity-checks)

---

## Overview

The position sizing system sits on top of an existing LightGBM binary classifier that predicts the probability of hitting a triple-barrier target within a 20 trading-day horizon. The system:

---

## Multi-Model Sizing System

The multi-model sizing system supports four models with different targets and directions, combining their predictions into final position weights.

### The Four Models

| Model | Target | Direction | Probability Column | Use Case |
|-------|--------|-----------|-------------------|----------|
| LONG_NORMAL | Upper barrier hit (normal move) | Long (+1) | `p_long_normal` | Standard bullish bets, moderate conviction |
| LONG_PARABOLIC | Upper barrier hit (parabolic move) | Long (+1) | `p_long_parabolic` | High-conviction momentum, rare but larger moves |
| SHORT_NORMAL | Lower barrier hit (normal move) | Short (-1) | `p_short_normal` | Standard bearish bets, moderate conviction |
| SHORT_PARABOLIC | Lower barrier hit (parabolic move) | Short (-1) | `p_short_parabolic` | High-conviction reversals, extreme moves |

Parabolic models have higher thresholds (via `parabolic_threshold_offset`, default 0.05) because parabolic moves are rarer but have larger expected returns.

### Module Structure

```
src/sizing/
├── __init__.py          # Module exports
├── config.py            # Configuration dataclasses + ShortSelectivityConfig
├── predictions.py       # Multi-model prediction loading
├── multi_model.py       # Core MultiModelSizingEngine + justification outputs
├── regime_gating.py     # Direction-aware regime-based exposure gating
├── regime_data.py       # Regime data loading and joining utilities
└── optimizer.py         # TPE optimization with short selectivity
```

### Pipeline Flow: Predictions → Position Weights

The `MultiModelSizingEngine.compute_weights()` method processes predictions through this sequence:

```
Raw Predictions (4 probability columns)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Compute Model Weights (compute_model_weights)   │
│   For each model:                                       │
│   • raw_weight = slope × (probability - intercept)      │
│   • Clip to [0, max_weight]                             │
│   • Apply exposure_mult scaling                         │
│   • Apply direction sign (+1 for longs, -1 for shorts)  │
│   Outputs: w_long_normal, w_long_parabolic, etc.        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Compute Edge Scores (compute_edge_scores)       │
│   edge = probability - intercept                        │
│   Higher edge = stronger signal above threshold         │
│   Outputs: edge_long_normal, edge_long_parabolic, etc.  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Combine Model Signals (combine_model_weights)   │
│   MODE_PRIORITY (default):                              │
│     For each symbol, pick model with highest |edge|     │
│     Use that model's weight and direction               │
│   BLEND:                                                │
│     Average all model weights (simple mean)             │
│   Outputs: combined_weight, contributing_model          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Handle Long/Short Conflicts (_apply_netting)    │
│   STRONGEST (default):                                  │
│     Pick direction with larger absolute weight          │
│   NET:                                                  │
│     Subtract short weights from long weights            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Apply Regime Gating (if enabled)                │
│   Compute exposure multipliers from regime features:    │
│   • VIX high → reduce all exposure                      │
│   • Credit spread high → reduce all exposure            │
│   • Breadth poor → reduce long exposure only            │
│   final_mult = vix_mult × credit_mult × breadth_mult    │
│   weights = weights × final_mult                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Apply Portfolio Constraints (in order!)         │
│   1. Clip individual weights to max_weight_per_name     │
│   2. Filter by minimum weight threshold                 │
│   3. Enforce max_positions (keep top N by |weight|)     │
│   4. Scale to max_gross_exposure                        │
│   5. Enforce max_net_exposure                           │
│   Output: final_weight                                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final Position Weights (per symbol)
```

### Quick Start

```bash
# Optimize sizing for all 4 models
python scripts/run_multi_model_sizing.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --n-trials 200

# Run backtest with optimized config
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json
```

### Prediction Format

**Wide format** (required): Single parquet with all model predictions:
```
date        symbol  p_long_normal  p_long_parabolic  p_short_normal  p_short_parabolic
2024-01-01  AAPL    0.65           0.45              0.32            0.28
2024-01-01  MSFT    0.58           0.42              0.35            0.30
```

### Combining Policies

#### Mode Priority (Default)
For each symbol, pick the model with highest "edge score":
```python
edge_score = probability - intercept
```
The winning model's weight is used with its direction sign. This produces sparse allocations where each symbol gets contribution from 1-2 models.

#### Blend
Average all model weights (after applying direction signs):
```python
combined = mean(w_long_normal, w_long_parabolic, w_short_normal, w_short_parabolic)
```
This produces denser allocations where all symbols get contributions from all models.

### Direction and Netting

Long models contribute positive weights, short models negative weights.

When both long and short signals exist for a symbol:
- **Strongest (default)**: Compare aggregate long vs short weights, pick the direction with larger absolute value
- **Net**: Subtract short weight from long weight, can result in mixed/near-zero final weights

### Configuration Example

```json
{
  "models": ["long_normal", "long_parabolic", "short_normal", "short_parabolic"],
  "combine_policy": "mode_priority",
  "netting_policy": "strongest",
  "sizing_params": {
    "slope": 2.0,
    "intercept": 0.5,
    "exposure_mult": 0.9,
    "max_weight": 0.10
  },
  "parabolic_threshold_offset": 0.05,
  "regime_gating": {
    "enabled": true,
    "vix_high_threshold": 80.0,
    "vix_high_exposure_mult": 0.7,
    "credit_risk_threshold": 1.5,
    "credit_risk_exposure_mult": 0.8,
    "breadth_poor_threshold": 30.0,
    "breadth_poor_long_mult": 0.8
  },
  "max_gross_exposure": 1.0,
  "max_net_exposure": 0.5,
  "max_weight_per_name": 0.10,
  "max_positions": 50
}
```

### Optimization Parameters

The TPE optimizer searches over these parameter ranges:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `slope` | [1.0, 5.0] | Probability mapping sensitivity |
| `intercept` | [0.3, 0.7] | Probability threshold for positive weight |
| `exposure_mult` | [0.5, 1.5] | Global exposure scaling factor |
| `turnover_penalty` | [0.0, 0.02] | Cost of turnover in objective |
| `parabolic_threshold_offset` | [0.0, 0.15] | Extra threshold for parabolic models |

**Note**: All models currently share the same `slope`, `intercept`, and `exposure_mult`. Only `parabolic_threshold_offset` differentiates normal vs parabolic models.

### Backward Compatibility

Legacy single-model configs are automatically converted:
```json
{"slope": 1.5, "intercept": 0.58}
```
Becomes:
```json
{"models": ["long_normal"], "sizing_params": {"slope": 1.5, "intercept": 0.58}}
```

---

## Original Single-Model Overview

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

## Regime Gating

Regime gating is a **risk-control overlay** that adjusts exposure based on market conditions, independent of model predictions. The system is **direction-aware**, applying different multipliers to long and short positions.

### Why Gating If Regime Is in the Model?

| Aspect | Model Features | Gating Overlay |
|--------|---------------|----------------|
| Purpose | Alpha generation | Risk control |
| Effect | Affects predictions | Affects position sizes |
| Nature | Soft (influences output) | Hard (exposure limits) |
| Control | Learned from data | Configurable rules |

Models include regime features to predict returns better. Gating provides hard limits that override model confidence when risk is elevated.

### Direction-Aware Gating

The gating system applies **separate multipliers** to longs and shorts:

| Rule | Long Multiplier | Short Multiplier | Rationale |
|------|-----------------|------------------|-----------|
| VIX High | `vix_high_exposure_mult` | `vix_high_exposure_mult` | Both reduced equally |
| Credit Stress | `credit_risk_exposure_mult` | `credit_risk_exposure_mult` | Both reduced equally |
| Poor Breadth | `breadth_poor_long_mult` | `breadth_poor_short_mult` | Shorts may thrive in poor breadth |
| Short Regime | N/A | `short_regime_mult` | Additional short-specific dampening |

**Key Design**: Poor breadth reduces longs (they struggle when few stocks participate) but may not reduce shorts (they can profit from narrow markets).

### Gating Rules

```yaml
regime_gating:
  enabled: true
  strict_missing_features: false  # Warn on missing features instead of failing

  # VIX-based: reduce exposure when VIX is high
  vix_high_threshold: 80.0       # 80th percentile
  vix_high_exposure_mult: 0.7    # Scale to 70%

  # Credit spread: reduce when credit stress
  credit_risk_threshold: 1.5     # Z-score
  credit_risk_exposure_mult: 0.8 # Scale to 80%

  # Breadth: direction-aware reduction
  breadth_poor_threshold: 30.0       # 30th percentile
  breadth_poor_long_mult: 0.8        # Scale longs to 80%
  breadth_poor_short_mult: 1.0       # Shorts NOT reduced

  # Additional short dampening (always applied)
  short_regime_mult: 0.9             # Scale shorts to 90%
```

### Multiplier Computation

Rules combine multiplicatively with direction awareness:

```python
# Long positions
final_long_mult = vix_mult × credit_mult × breadth_long_mult

# Short positions
final_short_mult = vix_mult × credit_mult × breadth_short_mult × short_regime_mult
```

### Available Regime Features

| Feature | Description |
|---------|-------------|
| `vix_percentile_252d` | VIX relative to 252-day history |
| `vix_zscore_60d` | VIX z-score over 60 days |
| `fred_bamlh0a0hym2_z60` | Credit spread z-score |
| `sector_breadth_pct_above_ma200` | Breadth indicator |
| `w_equity_bond_corr_60d` | Stock-bond correlation |

### Disabling Gating

Set thresholds to never trigger and multipliers to 1:

```json
{
  "regime_gating": {
    "enabled": false,
    "vix_high_threshold": 100.0,
    "vix_high_exposure_mult": 1.0
  }
}
```

### TPE Optimization of Gating

Gating parameters can be optimized alongside sizing parameters:

```bash
python scripts/run_multi_model_sizing.py \
    --regime-gating on \
    --n-trials 300
```

Optimized parameters are saved in `best_config_*.json` and included in trial logs.

---

## Short Selectivity

Short positions are inherently riskier (unlimited loss, borrow costs, squeeze risk). The short selectivity system makes shorts **more selective** than longs.

### Short Selectivity Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_threshold_offset` | 0.05 | Added to intercept for short models |
| `short_max_weight_mult` | 0.8 | Multiplier for max weight on shorts |
| `short_exposure_mult` | 1.0 | Multiplier for short exposure |

### How It Works

When computing weights for short models, the sizing parameters are modified:

```python
# For SHORT_NORMAL and SHORT_PARABOLIC models:
effective_intercept = base_intercept + short_threshold_offset
effective_max_weight = base_max_weight × short_max_weight_mult
```

**Example**: With `intercept=0.55`, `short_threshold_offset=0.05`:
- Long models require p > 0.55 for positive weight
- Short models require p > 0.60 for positive weight

### Configuration

```yaml
short_selectivity:
  short_threshold_offset: 0.05    # Shorts need p > 0.60 vs longs p > 0.55
  short_max_weight_mult: 0.8      # Shorts max at 8% vs longs max at 10%
  short_exposure_mult: 1.0        # No additional exposure reduction
```

### TPE Optimization

Short selectivity parameters are optimized when `--short-selectivity on` is passed:

```bash
python scripts/run_multi_model_sizing.py \
    --short-selectivity on \
    --regime-gating on \
    --n-trials 300
```

---

## Justification Outputs

Every sizing decision can be accompanied by justification columns for dashboard display and audit trails. All justification columns use the `j_` prefix.

### Justification Columns

| Column | Description |
|--------|-------------|
| `j_winning_model` | Which model contributed (for MODE_PRIORITY) |
| `j_winning_edge` | Edge score of the winning model |
| `j_gated` | Whether regime gating was applied |
| `j_long_mult` | Regime gating multiplier for longs |
| `j_short_mult` | Regime gating multiplier for shorts |
| `j_vix_triggered` | Whether VIX rule triggered |
| `j_credit_triggered` | Whether credit rule triggered |
| `j_breadth_triggered` | Whether breadth rule triggered |

### Generating Justification Outputs

```python
engine = MultiModelSizingEngine(config)
weighted = engine.compute_weights(signals, include_justification=True)

# Get all justification columns
j_cols = [c for c in weighted.columns if c.startswith('j_')]
```

### Decision Log

The decision log is a parquet file containing all decisions with justifications:

```bash
# Generate during optimization
python scripts/run_multi_model_sizing.py \
    --decision-log artifacts/sizing/decision_log.parquet

# Generate during backtest
python scripts/run_multi_model_backtest.py \
    --decision-log artifacts/backtests/decision_log.parquet
```

### Decision Log Schema

```python
decision_log_columns = [
    'run_id',           # Unique run identifier
    'date',             # Trading date
    'symbol',           # Stock symbol
    'final_weight',     # Final position weight
    # ... all j_ columns ...
]
```

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

### Multi-Model Pipeline

The multi-model sizing pipeline requires predictions from all four models:

```bash
conda activate stocks_predictor

# Step 1: Optimize sizing parameters (basic)
python scripts/run_multi_model_sizing.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --n-trials 200

# Step 1b: Optimize with regime gating and short selectivity
python scripts/run_multi_model_sizing.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --regime-gating on \
    --short-selectivity on \
    --n-trials 300 \
    --decision-log artifacts/sizing/decision_log.parquet

# Step 2: Run backtest with optimized config
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json

# Step 2b: Run backtest with decision log output
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json \
    --decision-log artifacts/backtests/decision_log.parquet
```

### Pipeline Implementation Status

| Component | Status | Script/Module |
|-----------|--------|---------------|
| Multi-model sizing engine | ✅ Implemented | `src/sizing/multi_model.py` |
| Regime gating overlay | ✅ Implemented | `src/sizing/regime_gating.py` |
| TPE optimizer | ✅ Implemented | `src/sizing/optimizer.py` |
| Configuration management | ✅ Implemented | `src/sizing/config.py` |
| Multi-model optimization | ✅ Implemented | `scripts/run_multi_model_sizing.py` |
| Multi-model backtest | ✅ Implemented | `scripts/run_multi_model_backtest.py` |
| CV prediction generation | ⚠️ Not integrated | Requires manual multi-model CV setup |
| Pipeline orchestrator | ⚠️ Partial | `scripts/run_sizing_pipeline.py` references some scripts not yet available |

### Generating Multi-Model Predictions

To use the sizing pipeline, you need predictions from all four models. The predictions must include:

```python
# Required columns in cv_predictions_multi.parquet
required_columns = [
    'date',              # Trading date
    'symbol',            # Stock symbol
    'p_long_normal',     # Probability from LONG_NORMAL model
    'p_long_parabolic',  # Probability from LONG_PARABOLIC model
    'p_short_normal',    # Probability from SHORT_NORMAL model
    'p_short_parabolic', # Probability from SHORT_PARABOLIC model
]
```

### Understanding the Two Training Approaches

| Use Case | Script | Data Used | Valid For |
|----------|--------|-----------|-----------|
| **Backtesting/Evaluation** | Walk-forward CV | Out-of-sample folds | Sizing optimization, strategy validation |
| **Live Trading** | `run_training.py` | All historical data | Forward-looking predictions on new data |

**CRITICAL**: Never use predictions from full-sample training for backtesting - that would be information leakage!

### Full Pipeline (From Scratch)

```bash
# Step 1: Ensure features and targets exist
python -m src.cli.compute --timeframes D

# Step 2: Run feature selection (optional)
python run_feature_selection.py

# Step 3: Run hyperparameter tuning (optional)
python run_model_tuning.py

# Step 4: Train models and generate CV predictions for each target type
# (Run for each of: long_normal, long_parabolic, short_normal, short_parabolic)
python run_training.py --target-type long_normal --cv-predictions

# Step 5: Run the multi-model sizing optimization
python scripts/run_multi_model_sizing.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --n-trials 200

# Step 6: Run backtest with optimized params
python scripts/run_multi_model_backtest.py \
    --sizing-config artifacts/sizing/best_config_multi_model.json
```

### Production Model (For Live Trading)

```bash
# Train production models on all data (one per target type)
python run_training.py --target-type long_normal
python run_training.py --target-type long_parabolic
python run_training.py --target-type short_normal
python run_training.py --target-type short_parabolic

# Generate predictions for latest date
python run_predict.py
```

### Output Artifacts

```
artifacts/
├── models/
│   ├── model_long_normal_*.pkl      # LONG_NORMAL model
│   ├── model_long_parabolic_*.pkl   # LONG_PARABOLIC model
│   ├── model_short_normal_*.pkl     # SHORT_NORMAL model
│   ├── model_short_parabolic_*.pkl  # SHORT_PARABOLIC model
│   └── feature_importance.csv
├── predictions/
│   └── cv_predictions_multi.parquet # All 4 model predictions
├── sizing/
│   ├── best_config_multi_model.json # Optimized multi-model config
│   ├── trials_multi_model.csv       # Optimization history
│   └── param_importance.json
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

### Multi-Model Specific Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Shared parameters across models | `optimizer.py` | All 4 models use same slope/intercept - may not be optimal for each |
| High stability penalty | `optimizer.py` line 302 | 0.5 × std(folds) can push toward conservative parameters |
| Breadth-only gating | `regime_gating.py` | Breadth poor multiplier should only affect longs, logic may need review |
| Silent feature dropping | `regime_gating.py` | Missing regime features disable gating rules silently |
| Weight clipping order | `multi_model.py` | Constraints applied sequentially, order matters for final weights |

### Sanity Checks

Run before trusting results:

```bash
# 1. Check for leakage
pytest tests/test_no_leakage.py -v

# 2. Check CV implementation
pytest tests/test_purged_cv.py -v

# 3. Check backtest alignment
pytest tests/test_backtest_alignment.py -v

# 4. Check multi-model sizing
pytest tests/sizing_multi_model_test.py -v
```

### Manual Verification

1. **Date Alignment**: Print train/test date ranges, verify no overlap
2. **Return Attribution**: Check if returns come from alpha or beta
3. **Stability**: Compare fold-to-fold metrics variance
4. **Sensitivity**: Vary parameters ±10%, check performance stability
5. **Regime Analysis**: Check performance in high vs low VIX periods
6. **Model Contribution**: Check which model wins most often in MODE_PRIORITY

### Red Flags

- Sharpe ratio > 3.0 in backtest (likely leakage)
- Zero turnover with changing weights (implementation bug)
- All folds have identical metrics (data leakage)
- Returns uncorrelated with hit rate (implementation bug)
- Single model dominates all symbol/date pairs (model imbalance)
- Regime gating never activates (missing features or misconfigured thresholds)

---

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.

---

*Last updated: 2025-12-28 - Added direction-aware gating, short selectivity, and justification outputs*
