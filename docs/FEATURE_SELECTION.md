# Feature Selection Architecture

This document describes the feature selection methodology for the technical dashboard ML pipeline. The system is specifically designed for financial time-series with noisy labels from triple barrier targets.

---

## 1. Design Philosophy

### 1.1 Key Challenges in Financial Feature Selection

Financial prediction poses unique challenges for feature selection:

1. **Noisy Labels**: Triple barrier targets have inherent noise from market microstructure
2. **Overlapping Trajectories**: Non-IID samples due to temporal overlap
3. **Regime Changes**: Features that work in one regime may fail in another
4. **Overfitting Risk**: High feature dimensionality with limited effective sample size

### 1.2 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Inverse Overlap Weighting** | Down-weight correlated samples via `n_overlapping_trajs` |
| **Over-Regularization** | Choose slightly sub-optimal but robust feature sets |
| **Thematic Grouping** | Pairs/triples represent meaningful combinations |
| **Multi-Stage Filtering** | Loose recall then strict precision |
| **Best-of-Any-Stage** | Return best subset seen at any pipeline stage |

---

## 2. Sample Weighting from Triple Barrier Targets

### 2.1 Triple Barrier Target Generation

Each sample represents a potential trade with:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIPLE BARRIER                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Entry: close price at t0                                            │
│  Upper Barrier: entry + up_mult * ATR (profit target)                │
│  Lower Barrier: entry - dn_mult * ATR (stop loss)                    │
│  Time Barrier: max_horizon days (expiration)                         │
│                                                                      │
│  Label:                                                              │
│    hit = 1  -> upper barrier hit first (profit)                      │
│    hit = -1 -> lower barrier hit first (loss)                        │
│    hit = 0  -> time expired (no barrier hit)                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Default Configuration** (from `src/pipelines/orchestrator.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `up_mult` | 3.0 | Upper barrier = entry + 3x ATR |
| `dn_mult` | 1.5 | Lower barrier = entry - 1.5x ATR |
| `max_horizon` | 20 | Maximum days to track trajectory |
| `start_every` | 3 | Days between new trajectory starts |

**Note**: The asymmetric barriers (3.0 up vs 1.5 down) create a 2:1 risk-reward ratio, meaning profit targets are set wider than stop losses.

### 2.2 Overlapping Trajectory Counting

When trajectories overlap in time, their outcomes are correlated. We count overlaps using an O(n log n) sweep-line algorithm:

```python
# For each trajectory i starting at t0[i] with horizon h_used[i]:
n_overlapping_trajs[i] = count of trajectories active at t0[i]

# Active = any trajectory j where:
#   t0[j] <= t0[i] < t0[j] + h_used[j]
```

**Algorithm (from `_count_overlaps_vectorized`):**

```
1. Create events: (time, type, index) where type=0 (start) or 1 (end)
2. Sort events by time (starts before ends at same time)
3. Sweep through events tracking active set:
   - At start event: count = |active_set| + 1 (self)
   - Update all active trajectories' counts
   - Add to active set
   - At end event: remove from active set
```

### 2.3 Inverse Overlap Weighting

The overlap weight reduces influence of correlated samples:

```python
weight_overlap = 1.0 / (n_overlapping_trajs + 0.5)

# Examples:
# - 1 trajectory (no overlap):  weight = 1/(1+0.5) = 0.67
# - 3 overlapping trajectories: weight = 1/(3+0.5) = 0.29
# - 10 overlapping trajectories: weight = 1/(10+0.5) = 0.095
```

**Why +0.5 pseudocount?**
- Prevents division by zero
- Smooths weights at low overlap counts
- Standard Laplace smoothing approach

### 2.4 Class Balance Weighting

Uses sklearn-style inverse frequency weighting:

```python
# For each class c with count N_c:
weight_class_balance[c] = n_samples / (n_classes * N_c)
```

**Note**: Class balance weighting is available but NOT applied by default. The pipeline uses overlap weighting only unless explicitly configured.

### 2.5 Combined Sample Weight

```python
# Multiplicative combination
weight_raw = weight_overlap * weight_class_balance

# Clip extreme values to [0.01, 10.0]
weight_clipped = clip(weight_raw, min=0.01, max=10.0)

# Normalize to sum = n_samples (standard for ML frameworks)
weight_final = weight_clipped * n_samples / sum(weight_clipped)

# Re-clip after normalization
weight_final = clip(weight_final, min=0.01, max=10.0)
```

**Interpretation:**
- Low overlap + rare class -> high weight (unique, important sample)
- High overlap + common class -> low weight (redundant sample)
- Clipping prevents extreme weights from dominating training

---

## 3. Balancing Precision and Recall

### 3.1 The Core Tradeoff

Feature selection for stock prediction faces a fundamental tension:

| Goal | Risk if Over-Optimized |
|------|------------------------|
| **High Precision** (few false positives) | Model becomes too conservative, misses opportunities |
| **High Recall** (few false negatives) | Model becomes noisy, generates many low-quality signals |
| **Cross-Sectional Coverage** | Model fixates on single sector or market cap |
| **Probability Calibration** | Probabilities don't match actual hit rates |

### 3.2 Avoiding Single-Stock/Sector Concentration

The risk of feature selection converging on narrow patterns (e.g., "only works for tech mega-caps") is mitigated through several **implemented mechanisms**:

**1. Cross-Sectional Features in BASE_FEATURES** ✅ *Implemented*

The curated BASE_FEATURES explicitly includes features that measure relative position:

```python
# From src/feature_selection/base_features.py
BASE_FEATURES = [
    # Cross-sectional momentum (relative to universe)
    "xsec_mom_20d_z",       # Z-score of 20d return vs cross-section
    "w_xsec_mom_4w_z",      # Weekly 4-week z-score

    # Sector-relative performance
    "rel_strength_sector",       # Relative strength vs sector ETF
    "alpha_mom_sector_20_ema10", # Alpha vs sector (20d, smoothed)

    # Breadth indicators (inherently cross-sectional)
    "sector_breadth_pct_above_ma200",  # % of sector ETFs above 200d MA
    "sector_breadth_mcclellan_osc",    # Breadth momentum oscillator
    ...
]
```

These features measure *relative* performance, not absolute returns. A stock scoring high on `xsec_mom_20d_z` is outperforming peers regardless of sector.

**2. Multi-Domain Feature Categories** ✅ *Implemented*

BASE_FEATURES spans multiple domains ensuring the model doesn't rely on any single pattern:

| Domain | Example Features | Purpose |
|--------|------------------|---------|
| **Momentum** | `rsi_14`, `w_macd_histogram` | Trend direction |
| **Price Position** | `pct_dist_ma_20_z`, `pos_in_20d_range` | Mean reversion |
| **Volatility Regime** | `vol_regime_ema10`, `rv_z_60` | Risk state |
| **Alpha/Relative** | `alpha_mom_spy_20_ema10`, `rel_strength_sector` | Stock vs market |
| **Breadth** | `sector_breadth_mcclellan_osc` | Market structure |
| **Macro** | `copper_gold_zscore`, `w_fred_bamlh0a0hym2_z60` | Economic regime |

**3. Time-Series CV with Expanding Window** ✅ *Implemented*

```python
# From src/feature_selection/config.py
cv_config = CVConfig(
    n_splits=5,
    scheme=CVScheme.EXPANDING,  # All historical data in training
    gap=20,                      # Embargo matches max_horizon
    min_train_samples=1000,
)
```

Expanding window ensures:
- Each fold includes all sectors in training data
- Features that only work for one sector show high fold variance
- Fold-level consistency check rejects sector-specific features

**4. Fold-Level Acceptance Criteria** ✅ *Implemented*

The loose forward selection requires improvement across multiple folds:

```python
# From src/feature_selection/pipeline.py
min_fold_improvement_ratio_loose=0.6  # 60% of folds must improve
```

A feature that dramatically improves one sector but hurts others will fail this check because it won't improve 60% of folds consistently.

**5. Regime Metrics Infrastructure** ⚠️ *Infrastructure exists but not actively used*

The codebase includes regime-stratified evaluation:

```python
# From src/feature_selection/metrics.py
def compute_regime_metrics(y_true, y_pred, regime, metric_fn):
    """Compute metrics stratified by regime."""
    regime_metrics = {}
    for regime_label in np.unique(regime):
        mask = regime == regime_label
        if mask.sum() > 10:
            regime_metrics[str(regime_label)] = metric_fn(y_true[mask], y_pred[mask])
    return regime_metrics
```

**Future Enhancement**: Pass sector labels as regime to get sector-stratified AUC. See Section 15.

### 3.3 Probability Calibration Strategy

To ensure predicted probabilities are meaningful:

**1. Over-Regularization** ✅ *Implemented*
```python
epsilon_remove_strict = 0.0  # Remove if doesn't hurt at all
```
Aggressively removes features that don't clearly help, reducing overfitting.

**2. Sample Weighting** ✅ *Implemented*
```python
weight_overlap = 1.0 / (n_overlapping_trajs + 0.5)
```
Down-weights correlated samples, giving more honest uncertainty estimates.

**3. LightGBM Regularization** ✅ *Implemented*
```python
'reg_alpha': 0.1,   # L1 prevents extreme splits
'reg_lambda': 0.1,  # L2 smooths predictions
'min_child_samples': 20,  # Prevents fitting noise
```

**4. Brier Score Tracking** ✅ *Implemented*

The pipeline computes and displays Brier score for calibration assessment:
```python
# From src/feature_selection/evaluation.py
metrics['extended'] = {
    'auc': compute_auc(y_true_arr, y_pred),
    'aupr': compute_aupr(y_true_arr, y_pred),
    'brier': compute_brier(y_true_arr, y_pred),  # Calibration metric
    'log_loss': compute_log_loss(y_true_arr, y_pred),
}
```

### 3.4 Evaluation Metrics

The pipeline tracks multiple metrics for comprehensive evaluation:

| Metric | Purpose | Implementation Status |
|--------|---------|----------------------|
| **AUC-ROC** | Discrimination ability (primary) | ✅ Primary optimization metric |
| **AUPR** | Precision-recall for imbalanced data | ✅ Computed every fold |
| **Brier Score** | Probability calibration | ✅ Computed every fold |
| **Log Loss** | Likelihood calibration | ✅ Computed every fold |
| **Fold Std Dev** | CV stability | ✅ Displayed for all metrics |
| **Sector-Stratified AUC** | Check for sector concentration | ⚠️ Infrastructure exists, not exposed |

**Metrics Display Example:**

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     METRICS SUMMARY                              │
  ├──────────────────────────────────────────────────────────────────┤
  │ Stage: 3_strict_backward    Features:  42                        │
  ├──────────┬────────────┬────────────┬──────────┬─────────────────┤
  │  Metric  │    Mean    │    Std     │  Better  │ Interpretation  │
  ├──────────┼────────────┼────────────┼──────────┼─────────────────┤
  │   AUC    │   0.6932   │   0.0145   │ ↑ higher │  Discrimination │
  │   AUPR   │   0.6821   │   0.0198   │ ↑ higher │   Prec-Recall   │
  │  BRIER   │   0.2341   │   0.0087   │ ↓ lower  │   Calibration   │
  │ LOG_LOSS │   0.6543   │   0.0234   │ ↓ lower  │   Likelihood    │
  └──────────┴────────────┴────────────┴──────────┴─────────────────┘
```

---

## 4. Feature Selection Pipeline

### 4.1 Pipeline Overview

The **Loose-Then-Tight Pipeline** implements "high recall early, high precision later":

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LOOSE-THEN-TIGHT PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: Start with BASE_FEATURES (seed, not forced)                │
│          - 36 curated features from base_features.py                │
│                                                                      │
│  STEP 1b: Optional Base Feature Elimination (quick prune)           │
│           - Remove base features that don't contribute              │
│                                                                      │
│  STEP 2: Loose Forward Selection (HIGH RECALL)                      │
│          - Add features under lenient criteria                      │
│          - Goal: capture all potentially useful features            │
│                                                                      │
│  STEP 3: Strict Backward Elimination (HIGH PRECISION)               │
│          - Remove features aggressively                             │
│          - Base features NOT protected                              │
│                                                                      │
│  STEP 4: Light Interaction Pass (targeted)                          │
│          - Pairs of top importance features                         │
│          - Stricter threshold than forward selection                │
│                                                                      │
│  STEP 5: Hill Climbing / Swapping (local optimization)              │
│          - Swap borderline features with outside candidates         │
│                                                                      │
│  STEP 6: Final Cleanup Pass (stability)                             │
│          - One last backward elimination                            │
│                                                                      │
│  STEP 7: Best Subset Selection                                      │
│          - Return best subset from ANY stage                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Acceptance Criteria

**Loose Forward Selection (Step 2):**
Accept feature if ANY of:
- Mean CV metric improves by `epsilon_add_loose` (default: 0.0002)
- Median CV metric improves
- Feature improves in `min_fold_improvement_ratio` of folds (default: 60%)

**Strict Backward Elimination (Steps 3, 6):**
Remove feature if:
- Removal improves the metric, OR
- Removal doesn't worsen metric beyond `epsilon_remove_strict` (default: 0.0)

**Hill Climbing / Swapping (Step 5):**
Accept swap if:
- New metric > old metric + `epsilon_swap` (default: 0.0005)

### 4.3 Over-Regularization Strategy

We intentionally choose slightly sub-optimal but robust feature sets:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `epsilon_remove_strict` | 0.0 | Remove if doesn't hurt at all |
| `epsilon_swap` | 0.0005 | Small improvement required for swap |
| `max_features_loose` | 80 | Cap on features after loose FS |
| Model L1/L2 | 0.1/0.1 | Regularization in LightGBM |

**Why over-regularize?**
- Features that barely improve CV may not generalize
- Financial regimes change - robust features preferred
- Simpler models are easier to interpret and maintain

---

## 5. BASE_FEATURES Starting Point

### 5.1 Curated Feature Set

The selection starts from `BASE_FEATURES` in `src/feature_selection/base_features.py` (36 features, 0.6932 AUC):

| Category | Features | Description |
|----------|----------|-------------|
| **Trend/Momentum** | `rsi_14`, `w_macd_histogram`, `trend_score_sign`, `trend_score_slope` | Direction and strength |
| **Trend Slopes** | `pct_slope_ma_20`, `pct_slope_ma_100`, `w_pct_slope_ma_50` | Multi-timeframe trend |
| **Price Position** | `pct_dist_ma_20_z`, `pct_dist_ma_50_z`, `relative_dist_20_50_z`, `pos_in_20d_range`, `vwap_dist_20d_zscore` | Mean reversion signals |
| **Volatility** | `atr_percent`, `vol_regime_ema10`, `rv_z_60`, `vix_zscore_60d`, `w_vix_vxn_spread` | Regime detection |
| **Relative Performance** | `alpha_mom_spy_20_ema10`, `alpha_mom_sector_20_ema10`, `w_alpha_mom_spy_20_ema10`, `rel_strength_sector`, `xsec_mom_20d_z`, `w_xsec_mom_4w_z` | Stock vs market/sector |
| **Sector Breadth** | `sector_breadth_pct_above_ma200`, `sector_breadth_mcclellan_osc` | Market structure |
| **Liquidity** | `upper_shadow_ratio`, `w_volshock_ema` | Selling pressure |
| **Macro** | `copper_gold_zscore`, `gold_spy_ratio_zscore`, `w_equity_bond_corr_60d`, `w_fred_bamlh0a0hym2_z60`, `fred_dgs2_chg20d`, `fred_ccsa_z52w` | Economic regime |

### 5.2 Expansion Candidates

`EXPANSION_CANDIDATES` provides ~200 additional features organized by domain for forward selection experiments.

### 5.3 Excluded Features

`EXCLUDED_FEATURES` lists raw/unnormalized values not suitable for ML:
- Raw OHLCV prices
- Raw moving averages
- Raw ATR, VIX levels
- Raw FRED series values

---

## 6. Thematic Pair and Triple Selection

### 6.1 Design Philosophy

Pairs and triples should represent **meaningful thematic combinations**, not arbitrary interactions. Each combination should tell a coherent story about market state.

### 6.2 Example Thematic Combinations

**Momentum + Regime + Higher Timeframe:**
```python
# Story: Strong daily momentum, in uptrend regime, confirmed by weekly
("rsi_14", "vol_regime_ema10", "w_macd_histogram")
```

**Trend + Mean Reversion + Breadth:**
```python
# Story: Stock extended from MA, but sector breadth supportive
("pct_dist_ma_20_z", "trend_score_sign", "sector_breadth_mcclellan_osc")
```

**Alpha + Volatility + Macro:**
```python
# Story: Outperforming market, in low vol regime, favorable credit conditions
("alpha_mom_spy_20_ema10", "rv_z_60", "w_fred_bamlh0a0hym2_z60")
```

### 6.3 Domain-Aware Interaction Patterns

The pipeline uses domain knowledge to filter interaction candidates:

```python
DOMAIN_PATTERNS = [
    ('vol', 'momentum'),     # Volatility * momentum
    ('regime', 'rsi'),       # Regime * momentum oscillator
    ('alpha', 'vol'),        # Alpha * volatility
    ('vix', 'return'),       # Market fear * returns
    ('breadth', 'trend'),    # Market breadth * stock trend
    ('macro', 'alpha'),      # Economic regime * stock alpha
]
```

### 6.4 Implementation in Pipeline

**Step 4 (Light Interaction Pass):**
```python
# Generate interaction candidates from top importance features
interaction_candidates = []
for i, f1 in enumerate(top_features):
    for f2 in top_features[i+1:]:
        # Create product interaction
        interaction_name = f"{f1}_x_{f2}"
        interaction_values = X[f1] * X[f2]

        # Accept if improvement >= epsilon_add_interaction
        if improvement >= config.epsilon_add_interaction:
            current_set.add(interaction_name)
```

---

## 7. Parallelization Strategy

### 7.1 Core Principle: Optimize CPU Utilization

The parallelization strategy adapts based on dataset size:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLELIZATION DECISION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  IF single model fully utilizes CPU (large dataset):                │
│      -> Train models sequentially                                   │
│      -> n_jobs = 1, num_threads = -1 (all cores per model)          │
│                                                                      │
│  IF single model underutilizes CPU (small dataset):                 │
│      -> Train models in parallel                                    │
│      -> n_jobs = N, num_threads = CPU_count / N                     │
│                                                                      │
│  RULE: n_jobs * num_threads ~ CPU_count                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Configuration Parameters

```python
@dataclass
class SearchConfig:
    # Forward selection parallelism
    forward_selection_n_jobs: int = 8      # Parallel candidate evaluations
    forward_selection_model_threads: int = 1  # Threads per model

    # Interaction search parallelism
    interaction_n_jobs: int = -1           # Workers for interaction eval
    interaction_num_threads_model: int = 1 # Threads per model
```

### 7.3 Dataset Size Heuristics

| Dataset Rows | Single Model Utilization | Recommended Strategy |
|--------------|--------------------------|----------------------|
| < 10,000 | Low | `n_jobs=8, num_threads=1` |
| 10,000 - 50,000 | Medium | `n_jobs=4, num_threads=2` |
| 50,000 - 200,000 | High | `n_jobs=2, num_threads=4` |
| > 200,000 | Full | `n_jobs=1, num_threads=-1` |

### 7.4 Joblib Parallelization Pattern

All parallel evaluation uses joblib with the loky backend:

```python
from joblib import Parallel, delayed

# Parallel candidate evaluation
results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
    delayed(_evaluate_addition_joblib)(
        X, y, current_features, candidate,
        model_config, cv_config, metric_config, search_config
    )
    for candidate in candidates
)
```

**Key Design Decisions:**
- **Fresh evaluator per worker**: Each worker creates its own `SubsetEvaluator`
- **No shared state**: Workers are independent (avoids GIL issues)
- **Loky backend**: Process-based parallelism for true multi-core usage

---

## 8. Cross-Validation with Purging

### 8.1 Time-Series CV Configuration

```python
@dataclass
class CVConfig:
    n_splits: int = 5
    scheme: CVScheme = CVScheme.EXPANDING
    gap: int = 20       # Embargo: skip 20 samples between train/test
    purge_window: int = 0
    min_train_samples: int = 1000
```

### 8.2 Purging and Embargo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TIME-SERIES CV WITH PURGING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Train Period          Gap (Embargo)        Test Period             │
│  |------------------|<-- 20 samples -->|--------------|             │
│                                                                      │
│  gap = 20 matches max_horizon = 20 from triple barrier targets      │
│  This prevents overlapping outcome windows between train/test       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why gap = 20?**
- Triple barrier targets have `max_horizon = 20` days
- A sample in train could have outcome overlapping with test sample
- Gap ensures no information leakage from outcome windows

---

## 9. Model Configuration

### 9.1 LightGBM Settings

```python
@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.LIGHTGBM
    task_type: TaskType = TaskType.CLASSIFICATION
    params: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,      # L1 regularization
        'reg_lambda': 0.1,     # L2 regularization
        'verbosity': -1,
    })
    num_threads: int = 1
    early_stopping_rounds: int = 50
    num_boost_round: int = 500
```

### 9.2 Regularization for Robustness

The model intentionally uses moderate regularization:
- `reg_alpha = 0.1`: L1 penalty shrinks weak features to zero
- `reg_lambda = 0.1`: L2 penalty prevents large coefficients
- `min_child_samples = 20`: Prevents splits on small groups
- `max_depth = 6`: Limits tree complexity

---

## 10. Checkpointing and Resumption

### 10.1 Checkpoint Structure

The pipeline saves state after each stage to `artifacts/feature_selection/checkpoint.pkl`:

```python
checkpoint = {
    'stage': '3_strict_backward',      # Last completed stage
    'current_features': list,          # Features at checkpoint
    'completed_stages': list,          # All completed stages
    'snapshots': list,                 # All stage snapshots
    'best_snapshot': SubsetSnapshot,   # Best so far
    'result_metric_main': float,
    'result_metric_std': float,
    'result_fold_metrics': list,
    'config': LooseTightConfig,
    'model_config': ModelConfig,
    'cv_config': CVConfig,
    'metric_config': MetricConfig,
    'search_config': SearchConfig,
    'timestamp': time.time(),
}
```

### 10.2 Resumption

```bash
# Check checkpoint status
python run_feature_selection.py --checkpoint-info

# Resume from checkpoint
python run_feature_selection.py --resume
```

### 10.3 Stage Identifiers

```
1_base_features        # Initial base features evaluation
1b_base_elimination    # Optional base feature pruning
2_loose_forward        # Loose forward selection
3_strict_backward      # Strict backward elimination
4_interactions         # Light interaction pass
5_swapping             # Hill climbing / swapping
6_final_cleanup        # Final cleanup pass
```

---

## 11. Running Feature Selection

### 11.1 Basic Usage

```bash
# Full pipeline with default settings
python run_feature_selection.py --n-jobs 8 --n-folds 5

# With base feature pruning
python run_feature_selection.py --prune-base

# Resume from checkpoint
python run_feature_selection.py --resume
```

### 11.2 Programmatic Usage

```python
from src.feature_selection.pipeline import LooseTightPipeline, LooseTightConfig
from src.feature_selection.config import ModelConfig, CVConfig

# Configure pipeline
config = LooseTightConfig(
    n_jobs=8,
    epsilon_add_loose=0.0002,
    epsilon_remove_strict=0.0,
    max_features_loose=80,
    run_interactions=True,
)

# Create and run pipeline
pipeline = LooseTightPipeline(pipeline_config=config)
pipeline.run(X, y, verbose=True)

# Get results
best_features = pipeline.get_best_features()
best_metric = pipeline.get_best_metric()  # (mean, std)
stage_summary = pipeline.get_stage_summary()
```

### 11.3 Using Sample Weights

Sample weights from triple barrier targets (overlap inverse) are automatically loaded and passed through the pipeline when using `run_feature_selection.py`:

```python
# Sample weights are loaded from targets_triple_barrier.parquet
# and passed to the pipeline automatically
X, y, sample_weight, regime = load_and_prepare_data(
    max_symbols=5000,
    binary_target=True,
    use_filtered_features=True
)

# The pipeline applies weights during model training
pipeline = run_feature_selection(
    X, y,
    features=valid_features,
    sample_weight=sample_weight,  # Overlap inverse weights
    n_folds=5,
    n_jobs=8,
)
```

Weights are applied during LightGBM/XGBoost training:
```python
# Internal implementation
train_data = lgb.Dataset(
    X_train, y_train,
    weight=sample_weight,  # Down-weights overlapping trajectories
)
```

---

## 12. Output Files

### 12.1 Feature Selection Artifacts

```
artifacts/feature_selection/
├── checkpoint.pkl           # Pipeline checkpoint (auto-saved)
├── selected_features.txt    # Final selected features (one per line)
├── selection_summary.json   # Complete metrics, features, and config
├── stage_summary.csv        # Per-stage metrics in CSV format
└── feature_importance.csv   # Feature importance scores (if computed)
```

### 12.2 Results Format

```python
# selected_features.txt
# Best features from Loose-then-Tight pipeline
# Stage: 3_strict_backward
# AUC: 0.6932 +/- 0.0625

rsi_14
w_macd_histogram
trend_score_sign
...

# selection_summary.json
{
  "best_stage": "3_strict_backward",
  "n_features": 42,
  "metric_mean": 0.6932,
  "metric_std": 0.0625,
  "features": ["alpha_mom_spy_20_ema10", "atr_percent", "rsi_14", ...],
  "stages": [
    {"stage": "1_base_features", "n_features": 36, "metric_mean": 0.6821, "metric_std": 0.0512, "is_best": false},
    {"stage": "2_loose_forward", "n_features": 58, "metric_mean": 0.6905, "metric_std": 0.0498, "is_best": false},
    {"stage": "3_strict_backward", "n_features": 42, "metric_mean": 0.6932, "metric_std": 0.0625, "is_best": true}
  ],
  "config": {
    "n_folds": 5,
    "n_jobs": 8,
    "balanced": false,
    "prune_base": false,
    "max_symbols": 5000,
    "sample_weighting": "overlap_inverse"
  }
}
```

---

## 13. Key Files Reference

| File | Purpose |
|------|---------|
| `src/feature_selection/pipeline.py` | Loose-Then-Tight pipeline implementation |
| `src/feature_selection/base_features.py` | BASE_FEATURES, EXPANSION_CANDIDATES, EXCLUDED_FEATURES |
| `src/feature_selection/config.py` | Configuration dataclasses |
| `src/feature_selection/evaluation.py` | SubsetEvaluator for CV evaluation |
| `src/feature_selection/algorithms.py` | Forward/backward/swap algorithms |
| `src/feature_selection/cv.py` | Time-series CV with purging |
| `src/features/target_generation.py` | Triple barrier targets and weighting |
| `run_feature_selection.py` | Main entry point script |

---

## 14. Best Practices

### 14.1 Feature Design

1. **Normalize features**: Use z-scores, percentiles, or ratios (not raw prices)
2. **Include multi-timeframe**: Daily + weekly versions capture different signals
3. **Domain knowledge**: Group features by economic meaning
4. **Avoid redundancy**: Check correlation between candidate features

### 14.2 Selection Process

1. **Start with BASE_FEATURES**: Proven feature set as starting point
2. **Use sample weights**: Always weight by overlap inverse
3. **Validate on holdout**: Final validation on unseen time period
4. **Monitor stability**: Features should be consistent across CV folds

### 14.3 Parallelization

1. **Match to dataset size**: Adjust n_jobs based on data volume
2. **Avoid over-parallelization**: n_jobs * num_threads ~ CPU_count
3. **Use checkpointing**: Long-running pipelines should checkpoint frequently
4. **Monitor memory**: Parallel workers share data via serialization

---

## 15. Future Enhancements

### 15.1 Planned Features

**1. Sector-Stratified Evaluation** (Infrastructure exists, needs exposure)

The infrastructure for regime-stratified metrics exists in `compute_regime_metrics()`. To enable sector-stratified AUC:

```python
# In run_feature_selection.py - load sector labels
def load_and_prepare_data(...):
    # Load sector from universe CSV
    universe = pd.read_csv('cache/US universe_sp500.csv')
    sector_map = dict(zip(universe['Symbol'], universe['Sector']))
    merged['sector'] = merged['symbol'].map(sector_map)

    # Return sector as regime
    sector = merged['sector'].copy()
    return X, y, sample_weight, sector  # sector instead of vol_regime

# In pipeline - pass sector to evaluator
evaluator = SubsetEvaluator(
    X=X, y=y,
    model_config=model_config,
    cv_config=cv_config,
    metric_config=metric_config,
    search_config=search_config,
    sample_weight=sample_weight,
    regime=sector,  # Enable sector-stratified metrics
)
```

**Expected output** (new display section):
```
  SECTOR-STRATIFIED AUC
  ┌─────────────────────┬────────┐
  │  Sector             │  AUC   │
  ├─────────────────────┼────────┤
  │  Technology         │ 0.6921 │
  │  Health Care        │ 0.6845 │
  │  Financials         │ 0.7012 │
  │  Consumer Discret.  │ 0.6789 │
  │  ...                │ ...    │
  └─────────────────────┴────────┘
  Sector AUC Std Dev: 0.0087 (< 0.03 = consistent)
```

**Warning trigger**: If sector AUC std dev > 0.03, display warning about potential sector concentration.

**2. Automated Thematic Pair/Triple Generation**

Generate interaction candidates based on `FEATURE_CATEGORIES` metadata rather than pure importance ranking:

```python
# Proposed implementation
THEMATIC_GROUPS = {
    'momentum_regime': ['rsi_14', 'vol_regime_ema10', 'w_macd_histogram'],
    'trend_reversion': ['pct_dist_ma_20_z', 'trend_score_sign', 'sector_breadth_mcclellan_osc'],
    'alpha_macro': ['alpha_mom_spy_20_ema10', 'rv_z_60', 'w_fred_bamlh0a0hym2_z60'],
}

def generate_thematic_interactions(groups: dict) -> List[Tuple[str, ...]]:
    """Generate pairs/triples from predefined thematic groups."""
    interactions = []
    for group_name, features in groups.items():
        # Generate all pairs within group
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                interactions.append((f1, f2, group_name))
        # Generate triple if 3+ features
        if len(features) >= 3:
            interactions.append(tuple(features[:3]) + (group_name,))
    return interactions
```

**Expected benefit**: More interpretable interactions that represent coherent market narratives.

**2. Adaptive Parallelization Based on Runtime Profiling**

Profile single model training time and adjust parallelization dynamically:

```python
# Proposed implementation
def adaptive_n_jobs(X: pd.DataFrame, y: pd.Series) -> int:
    """Determine optimal n_jobs based on dataset size and profiled timing."""
    n_samples = len(X)
    n_features = len(X.columns)

    # Profile single model training
    start = time.time()
    _train_single_model(X.iloc[:1000], y.iloc[:1000])
    single_model_time = time.time() - start

    # Estimate full dataset time
    estimated_time = single_model_time * (n_samples / 1000)

    # If single model takes > 5 seconds, it likely saturates CPU
    if estimated_time > 5.0:
        return 1  # Sequential models
    elif estimated_time > 1.0:
        return max(1, cpu_count() // 4)
    else:
        return max(1, cpu_count() // 2)
```

**Expected benefit**: Automatic tuning without manual configuration.

**3. Feature Stability Scoring Across Regime Changes**

Evaluate feature importance stability across different market regimes (bull/bear/sideways):

```python
# Proposed implementation
def compute_regime_stability(
    X: pd.DataFrame,
    y: pd.Series,
    regime_column: str = 'market_regime'
) -> pd.DataFrame:
    """Compute feature importance by regime and stability score."""
    regimes = X[regime_column].unique()
    importance_by_regime = {}

    for regime in regimes:
        mask = X[regime_column] == regime
        model = train_model(X[mask], y[mask])
        importance_by_regime[regime] = model.feature_importances_

    # Stability = inverse of coefficient of variation across regimes
    df = pd.DataFrame(importance_by_regime)
    df['stability'] = df.mean(axis=1) / (df.std(axis=1) + 1e-8)
    return df
```

**Expected benefit**: Identify features that work consistently vs those that only work in specific conditions.

**4. Integration with Hyperparameter Tuning Pipeline**

Joint optimization of feature set and model hyperparameters:

```python
# Proposed implementation
def joint_feature_hyperparam_search(
    X: pd.DataFrame,
    y: pd.Series,
    feature_search_config: SearchConfig,
    hyperparam_space: dict,
    n_trials: int = 50
) -> Tuple[List[str], dict]:
    """
    Alternating optimization:
    1. Fix hyperparams, optimize features
    2. Fix features, optimize hyperparams
    3. Repeat until convergence
    """
    current_features = BASE_FEATURES
    current_params = default_params

    for iteration in range(n_iterations):
        # Feature selection with current params
        current_features = run_feature_selection(X, y, current_params)

        # Hyperparam tuning with current features
        current_params = run_hyperparam_search(X[current_features], y, hyperparam_space)

        # Check convergence
        if converged:
            break

    return current_features, current_params
```

**Expected benefit**: Features and hyperparameters are jointly optimized, avoiding local optima from sequential optimization.

---

## 16. Troubleshooting

### 16.1 Common Issues

**Pipeline stuck on one stage:**
- Check for infinite loop in backward elimination
- Verify epsilon values are reasonable
- Review checkpoint for last successful stage

**Memory errors:**
- Reduce n_jobs (fewer parallel workers)
- Use smaller batch sizes for interactions
- Check for DataFrame fragmentation

**Low AUC scores:**
- Verify sample weights are applied correctly
- Check for data leakage in CV
- Review feature NaN rates

**Model concentrates on single sector:**
- Add more cross-sectional features to BASE_FEATURES
- Check that sector-relative alpha features are included
- Verify sector distribution in training data

### 16.2 Debugging

```python
# Check checkpoint status
info = LooseTightPipeline.checkpoint_info()
print(info)

# Inspect stage snapshots
pipeline = LooseTightPipeline()
pipeline.resume_from_checkpoint(X, y)
for snapshot in pipeline.snapshots:
    print(f"{snapshot.stage}: {len(snapshot.features)} features, {snapshot.metric_mean:.4f}")
```
