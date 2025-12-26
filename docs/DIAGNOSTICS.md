# Diagnostics

Three-stage diagnostic system for validating pipeline outputs, model training, and position sizing.

## Quick Start

```bash
conda activate stocks_predictor

# Stage 1: After feature computation - validate pipeline outputs
python run_data_quality.py --verbose

# Stage 2: After model training - validate model behavior
python run_diagnostics.py --model long_normal
python run_diagnostics.py --all-models

# Stage 3: After backtest - validate sizing decisions
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json \
    --diagnostics on
```

## When to Run Each Stage

| Stage | After Running | Script | Purpose |
|-------|---------------|--------|---------|
| 1 | `python -m src.cli.compute` | `run_data_quality.py` | Validate features computed correctly |
| 2 | `python run_model_tuning.py` | `run_diagnostics.py` | Validate model is learning properly |
| 3 | Backtest / before trading | `--diagnostics on` | Validate sizing, detect drift |

---

## Severity Levels

All stages use consistent severity levels:

| Severity | Meaning | Action |
|----------|---------|--------|
| **CRITICAL** / **ERROR** | Must fix before proceeding | Blocks pipeline / trading |
| **WARN** | Should investigate | May affect quality |
| **INFO** | Informational | No immediate action required |

---

## Stage 1: Data Quality Checks

Run after feature pipeline to validate outputs before training.

### Usage

```bash
# Default: check features_filtered.parquet
python run_data_quality.py

# Check complete features
python run_data_quality.py --features artifacts/features_complete.parquet

# Verbose output with detailed breakdowns
python run_data_quality.py --verbose
```

### What It Checks

| Check | Description | Action If Failed |
|-------|-------------|------------------|
| **BASE_FEATURES coverage** | Validates all CORE + HEAD features present | Re-run feature pipeline |
| **EXPANSION_CANDIDATES coverage** | Validates optional feature pool | Check specific modules |
| **Category NaN rates** | Groups features by type (trend, volatility, macro) | Investigate specific category |
| **Infinite values** | Detects data corruption | Fix division-by-zero in feature code |
| **Targets validation** | Checks target file for anomalies | Re-run target generation |

### Output

Prints to console with status per category:
- `OK` - Category healthy
- `HIGH NaN` - Average NaN rate above expected
- `DEGRADED` - Many features have >50% NaN
- `BROKEN` - Features are 100% NaN

**Exit codes:** 0 = OK, 1 = Issues found

### Example Output

```
================================================================================
DATA QUALITY REPORT
================================================================================

========================================
BASE_FEATURES V2 VALIDATION (~49 curated core features)
========================================
   Coverage: 49/49 (100.0%) [PASS]

========================================
FEATURES SUMMARY
========================================
Total features: 312

Category             Count   Avg NaN   Max NaN Status
------------------------------------------------------------
trend                   12     8.2%     15.3% OK
momentum                 8     6.1%     12.0% OK
volatility              14     7.5%     18.2% OK
fred_macro              18    12.3%     25.1% OK
alpha_beta              24    15.8%     35.2% OK

========================================
RECOMMENDATIONS
========================================
No critical issues found. Data quality is acceptable.
```

---

## Stage 2: Model Diagnostics

Run after hyperopt/training to validate model behavior.

### Usage

```bash
# Single model
python run_diagnostics.py --model long_normal

# All 4 models
python run_diagnostics.py --all-models

# Force model fitting (if no trained model exists)
python run_diagnostics.py --model long_normal --fit

# Custom CV settings
python run_diagnostics.py --model long_normal --n-folds 3 --gap 20
```

### What It Checks

#### Data Quality (integrated)
| Check | Threshold | Severity |
|-------|-----------|----------|
| Missing features (NaN/inf) | >5% | CRITICAL |
| Missing features (NaN/inf) | >1% | WARN |
| Constant features | variance < 1e-8 | WARN |
| Duplicate features | correlation >= 0.9999 | WARN |
| Infinite values | any | CRITICAL |

#### Leakage Detection
| Check | Threshold | Severity |
|-------|-----------|----------|
| CV date overlap | any overlap | CRITICAL |
| Embargo gap too small | gap < 20 days | WARN |
| Label leakage (high AUC) | mean AUC > 0.85 | CRITICAL |
| Label leakage (high AUC) | any fold > 0.85 | WARN |

#### CV Stability
| Check | Threshold | Severity |
|-------|-----------|----------|
| AUC instability | CV > 0.25 | CRITICAL |
| AUC instability | CV > 0.15 | WARN |
| Performance degradation | late folds 0.05 worse | WARN |
| Small training folds | < 2000 samples | WARN |
| Positive rate drift | max/min ratio > 1.5 | WARN |

#### Calibration & Ranking
| Check | Threshold | Severity |
|-------|-----------|----------|
| Brier worse than baseline | Brier > baseline × 1.1 | WARN |
| Extreme predictions | >10% outside [0.05, 0.95] | WARN |
| Low precision lift | P@5% < 1.5× base rate | WARN |

#### Sample Weighting
| Check | Threshold | Severity |
|-------|-----------|----------|
| Weight concentration | top 1% > 10% of weight | WARN |
| Low effective sample size | ESS < 50% of n | WARN |
| Weight-sensitive metrics | AUC diff > 0.03 | INFO |

#### Hyperopt Sanity
| Check | Threshold | Severity |
|-------|-----------|----------|
| High pruning rate | >50% trials pruned | WARN |
| Degenerate tree params | leaves > 128 & child < 100 | WARN |
| Learning rate extreme | lr < 0.01 or > 0.12 | INFO |

#### Feature Story
| Check | Threshold | Severity |
|-------|-----------|----------|
| Missing feature family | critical family absent | WARN |
| Importance concentration | top 5 > 50% importance | INFO |
| Story mismatch | expected families not in top 10 | INFO |

### Output

Reports saved to:
```
artifacts/diagnostics/{model_key}/
├── diagnostic_report.json    # Structured results for programmatic use
└── diagnostic_report.md      # Human-readable report
```

### Example Console Output

```
======================================================================
DIAGNOSTIC REPORT: LONG_NORMAL
======================================================================

Loading model configuration...
  Found 77 features
Loading features...
  Features shape: (1250000, 320)
Loading targets...
  Targets shape: (85000, 12)

Data prepared:
  Samples: 45,231
  Features: 77
  Positive rate: 52.3%
  Generated 5 CV folds

Running 5-fold CV...
  Fold 1: AUC=0.5834, Brier=0.2412, P@10=0.7156
  Fold 2: AUC=0.5789, Brier=0.2445, P@10=0.7023
  Fold 3: AUC=0.5812, Brier=0.2398, P@10=0.7234
  Fold 4: AUC=0.5856, Brier=0.2389, P@10=0.7189
  Fold 5: AUC=0.5801, Brier=0.2421, P@10=0.7098

Running diagnostic checks...
  Data quality checks...
  Leakage detection checks...
  CV stability checks...
  Calibration & ranking checks...
  Sample weighting checks...
  Hyperopt sanity checks...
  Feature sanity checks...

======================================================================
DIAGNOSTIC SUMMARY
======================================================================
  CRITICAL: 0
  WARN:     2
  INFO:     4
  STATUS:   PASSED

Reports saved to:
  JSON: artifacts/diagnostics/long_normal/diagnostic_report.json
  Markdown: artifacts/diagnostics/long_normal/diagnostic_report.md
```

### Programmatic Usage

```python
from src.diagnostics import ModelDiagnosticRunner, run_data_quality_checks
import numpy as np

# Run full model diagnostics
runner = ModelDiagnosticRunner(model_key='long_normal')
result = runner.run()
runner.save_report()

print(f"Passed: {result.passed}")
print(f"Critical: {result.n_critical}, Warn: {result.n_warn}")

# Run data quality checks on feature matrix
X = np.load('features.npy')
feature_names = ['feat1', 'feat2', ...]
flags, summary = run_data_quality_checks(X, feature_names)

for flag in flags:
    print(f"[{flag.severity}] {flag.check_name}: {flag.symptom}")
```

### Custom Thresholds

```python
from src.diagnostics import ModelDiagnosticRunner, DiagnosticThresholds

# Customize thresholds
thresholds = DiagnosticThresholds(
    nan_rate_warn=0.02,           # Stricter: warn at 2% instead of 1%
    nan_rate_critical=0.10,       # Looser: critical at 10% instead of 5%
    auc_cv_warn=0.20,             # Looser: warn at CV=0.20 instead of 0.15
    suspiciously_high_auc=0.80,   # Stricter: flag leakage at AUC=0.80
)

runner = ModelDiagnosticRunner(
    model_key='long_normal',
    thresholds=thresholds,
)
```

### Module Structure

```
src/diagnostics/
├── __init__.py              # Public API exports
├── core.py                  # DiagnosticFlag, Severity, thresholds
├── runner.py                # ModelDiagnosticRunner orchestrator
├── checks_data_quality.py   # NaN, constant, duplicate, coverage checks
├── checks_data.py           # CV leakage detection
├── checks_cv.py             # CV stability analysis
├── checks_calibration.py    # Calibration & ranking
├── checks_weights.py        # Sample weight validation
├── checks_hyperopt.py       # Hyperopt parameter analysis
└── checks_features.py       # Feature story consistency
```

---

## Stage 3: Sizing Diagnostics

Run during or after backtest to validate position sizing decisions and surface operational risks. Designed for a weekly "no-think" dashboard workflow.

### Usage

```bash
# Run backtest with diagnostics enabled
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json \
    --diagnostics on

# Compute baselines first (recommended for drift detection)
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json \
    --compute-diagnostic-baselines

# Use custom thresholds
python scripts/run_multi_model_backtest.py \
    --diagnostics on \
    --diagnostics-thresholds config/my_thresholds.json
```

### What It Checks

#### Data Integrity
| Check | WARN Threshold | ERROR Threshold |
|-------|---------------|-----------------|
| Missing predictions | > 1% | > 5% |
| NaN predictions | > 0.5% | > 2% |
| Duplicate (symbol, date) | - | any |
| Missing weeks | 1 consecutive | 2 consecutive |
| Stale predictions | > 3 days | > 7 days |

#### Signal Distribution / Drift
| Check | WARN Threshold | ERROR Threshold |
|-------|---------------|-----------------|
| PSI vs baseline | > 0.10 | > 0.25 |
| KS statistic vs baseline | > 0.10 | > 0.20 |
| Extreme signal rate | > 10% | - |
| Rank correlation vs prev week | < 0.50 | < 0.30 |

#### Portfolio Construction
| Check | WARN Threshold | ERROR Threshold |
|-------|---------------|-----------------|
| Gross exposure exceeded | - | max + epsilon |
| Max weight cap violated | - | max + epsilon |
| HHI concentration | > 0.10 | > 0.25 |
| Top 5 concentration | > 50% | > 70% |
| Turnover | > 50% | > 100% |
| Short dominance | > 70% of gross | - |
| Cash fraction low/high | < 10% or > 80% | - |
| Consecutive gating weeks | >= 3 | - |

#### Execution Risk
| Check | WARN Threshold | ERROR Threshold |
|-------|---------------|-----------------|
| Positions below liquidity | > 10% | > 30% |
| Positions above gap threshold | > 20% | > 40% |
| Slippage budget utilization | > 80% | > 100% |

#### Backtest Outcomes (backtest mode only)
| Check | WARN Threshold | ERROR Threshold |
|-------|---------------|-----------------|
| Rolling 13w Sharpe | < 0 | < -1 |
| Rolling drawdown | < -15% | < -25% |
| Tail week | < p5 historical or < -5% | - |

#### Behavioral Risk (Compound Conditions)
| Check | Condition | Severity |
|-------|-----------|----------|
| High exposure + high vol | Gross > 80% AND VIX > 80th pct | WARN |
| Fragile portfolio | HHI > 0.08 AND Turnover > 40% | WARN |

### Output

Each diagnostics run produces:

```
artifacts/diagnostics/
├── sizing_diagnostics_<run_id>.json       # Full structured report
├── sizing_diagnostics_weekly_<run_id>.csv # Per-week metrics for analysis
└── sizing_diagnostics_<run_id>.md         # Human-readable summary
```

### Example Terminal Output

```
============================================================
SIZING DIAGNOSTICS SUMMARY
============================================================

Run ID: multi_model
Period: 2024-01-01 to 2024-12-31
Weeks: 52
Models: long_normal, long_parabolic, short_normal, short_parabolic

Status: PASSED

--- WARNINGS ---
  ERRORS:   0
  WARNINGS: 3
  INFO:     8

--- WARNINGS ---
  [PSI_ELEVATED_LONG_NORMAL] PSI for long_normal shows moderate distribution shift
  [TURNOVER_ELEVATED] Turnover above threshold
  [TOP5_CONCENTRATION] Top 5 positions are significant fraction

--- AGGREGATE METRICS ---
  Mean Gross Exposure: 65.20%
  Mean Turnover:       28.50%
  Mean Positions:      35.2
  Mean HHI:            0.0423

--- BACKTEST METRICS ---
  Total Return:  45.30%
  Sharpe Ratio:  1.25
  Max Drawdown:  -12.50%
  Hit Rate:      58.20%
```

### Baselines for Drift Detection

Baselines capture the historical distribution of predictions for PSI/KS comparison:

```bash
# Compute baselines from historical OOS predictions
python scripts/run_multi_model_backtest.py \
    --prediction-path artifacts/predictions/cv_predictions_multi.parquet \
    --sizing-config artifacts/sizing/best_config_multi_model.json \
    --compute-diagnostic-baselines

# Baselines are saved to:
# artifacts/diagnostics/baselines/<model>_probability_baseline.json
# artifacts/diagnostics/baselines/<model>_edge_baseline.json
```

### Custom Thresholds

Create a JSON file with custom thresholds:

```json
{
  "missing_preds_warn": 0.02,
  "psi_warn": 0.15,
  "hhi_warn": 0.08,
  "turnover_error": 0.80,
  "drawdown_warn": -0.10
}
```

### Programmatic Usage

```python
from src.diagnostics.sizing import SizingDiagnosticsRunner

runner = SizingDiagnosticsRunner(
    weighted_signals=df,
    sizing_config=config.to_dict(),
    models=['long_normal', 'short_normal'],
    backtest_mode=True,
)

report = runner.run()
runner.save_report("artifacts/diagnostics")
runner.print_summary()

# Access warnings
for warning in report.all_warnings:
    print(f"[{warning.severity}] {warning.code}: {warning.message}")
    print(f"  Value: {warning.value}, Threshold: {warning.threshold}")
    print(f"  Action: {warning.suggested_action}")

# Check if passed (no ERROR severity)
if not report.passed:
    print("FAILED: Address errors before trading")
```

### Module Structure

```
src/diagnostics/sizing/
├── __init__.py      # Public API
├── schema.py        # Dataclasses for reports and warnings
├── thresholds.py    # Configurable thresholds
├── baselines.py     # Baseline computation and PSI/KS
├── warnings.py      # Warning rule engine
├── metrics.py       # Core metric computations
└── runner.py        # Main diagnostics orchestrator
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DIAGNOSTICS PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    │   STAGE 1        │      │   STAGE 2        │      │   STAGE 3        │
    │   Data Quality   │ ──▶  │   Model          │ ──▶  │   Sizing         │
    │                  │      │   Diagnostics    │      │   Diagnostics    │
    └──────────────────┘      └──────────────────┘      └──────────────────┘
           │                         │                         │
           ▼                         ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    │ ENTRY POINT      │      │ ENTRY POINT      │      │ ENTRY POINT      │
    │                  │      │                  │      │                  │
    │ run_data_        │      │ run_diagnostics  │      │ run_multi_model_ │
    │ quality.py       │      │ .py              │      │ backtest.py      │
    │                  │      │                  │      │ --diagnostics on │
    └──────────────────┘      └──────────────────┘      └──────────────────┘
           │                         │                         │
           ▼                         ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    │ INPUT            │      │ INPUT            │      │ INPUT            │
    │                  │      │                  │      │                  │
    │ • features_      │      │ • features_      │      │ • cv_predictions │
    │   complete.pq    │      │   complete.pq    │      │   _multi.pq      │
    │ • targets_       │      │ • targets_       │      │ • sizing_config  │
    │   triple_barrier │      │   triple_barrier │      │   .json          │
    │   .pq            │      │   .pq            │      │ • baselines/     │
    │                  │      │ • hyperopt/      │      │                  │
    │                  │      │   best_params    │      │                  │
    └──────────────────┘      └──────────────────┘      └──────────────────┘
           │                         │                         │
           ▼                         ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    │ CHECKS           │      │ CHECKS           │      │ CHECKS           │
    │                  │      │                  │      │                  │
    │ • BASE_FEATURES  │      │ • Data quality   │      │ • Data integrity │
    │   coverage       │      │ • Leakage detect │      │ • Signal drift   │
    │ • NaN rates by   │      │ • CV stability   │      │ • Portfolio      │
    │   category       │      │ • Calibration    │      │   construction   │
    │ • Infinite vals  │      │ • Sample weights │      │ • Execution risk │
    │ • Target anomaly │      │ • Hyperopt sanity│      │ • Backtest perf  │
    │                  │      │ • Feature story  │      │ • Behavioral     │
    └──────────────────┘      └──────────────────┘      └──────────────────┘
           │                         │                         │
           ▼                         ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    │ OUTPUT           │      │ OUTPUT           │      │ OUTPUT           │
    │                  │      │                  │      │                  │
    │ Console report   │      │ artifacts/       │      │ artifacts/       │
    │ with status per  │      │ diagnostics/     │      │ diagnostics/     │
    │ category         │      │ {model_key}/     │      │                  │
    │                  │      │ • diagnostic_    │      │ • sizing_diag_   │
    │ Exit codes:      │      │   report.json    │      │   {run_id}.json  │
    │ 0 = OK           │      │ • diagnostic_    │      │ • sizing_diag_   │
    │ 1 = Issues       │      │   report.md      │      │   weekly.csv     │
    └──────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No features found" | Hyperopt not run yet | Run `python run_model_tuning.py --model {key}` first |
| "No hyperopt results" | Missing best_params.json | Run hyperopt or use `--fit` flag |
| High pruning rate | Search space too aggressive | Adjust hyperopt bounds |
| Label leakage detected | Feature uses future data | Audit feature computation for lookahead |
| Low ESS | Overlap weighting too aggressive | Review weight_final computation |

### When to Re-run Diagnostics

1. After changing feature computation
2. After modifying target generation
3. After hyperopt with new search space
4. After changing CV strategy
5. Before deploying model to production

---

## Related Documentation

- [FEATURE_PIPELINE_ARCHITECTURE.md](FEATURE_PIPELINE_ARCHITECTURE.md) - Pipeline stages and data flow
- [FEATURE_SELECTION.md](FEATURE_SELECTION.md) - Feature selection methodology
- [TARGETS.md](TARGETS.md) - Triple barrier targets and sample weighting
- [MODEL_TRAINING.md](MODEL_TRAINING.md) - Model training workflow
