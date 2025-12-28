# Feature Selection Architecture

This document describes the **group-first feature selection** methodology for the technical dashboard ML pipeline. The system is specifically designed for financial time-series with noisy labels from triple barrier targets.

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
| **Group-First Selection** | Entire hypothesis groups are the atomic unit of selection |
| **Inverse Overlap Weighting** | Down-weight correlated samples via `n_overlapping_trajs` |
| **Over-Regularization** | Choose slightly sub-optimal but robust feature sets |
| **Deterministic Selection** | No randomness in swaps or selection order |
| **Baseline Demotion Mode** | Optional removal of baseline groups during backward elimination |

### 1.3 Group-First vs Singleton Selection

**Why Group-First?**

Singleton selection (adding/removing individual features) has several drawbacks:
- Features within a hypothesis group are often correlated
- Adding one feature from a group may prevent related features from being selected
- Results are sensitive to feature ordering
- Difficult to interpret which "stories" the model relies on

**Group-First Benefits:**
- Each group represents a coherent hypothesis (e.g., "volatility regime", "momentum quality")
- Selection is more stable and interpretable
- Easier to reason about which market dynamics the model captures
- Natural regularization through group-level decisions

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

**Note**: Sample weighting (both overlap inverse and class balance) is available but NOT applied by default. Use `--use-weights` flag to enable overlap inverse weighting.

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

**Future Enhancement**: Pass sector labels as regime to get sector-stratified AUC. See Section 16.

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
  │ Stage: 3_strict_backward    Features: N                          │
  ├──────────┬────────────┬────────────┬──────────┬─────────────────┤
  │  Metric  │    Mean    │    Std     │  Better  │ Interpretation  │
  ├──────────┼────────────┼────────────┼──────────┼─────────────────┤
  │   AUC    │   X.XXXX   │   0.XXXX   │ ↑ higher │  Discrimination │
  │   AUPR   │   X.XXXX   │   0.XXXX   │ ↑ higher │   Prec-Recall   │
  │  BRIER   │   0.XXXX   │   0.XXXX   │ ↓ lower  │   Calibration   │
  │ LOG_LOSS │   0.XXXX   │   0.XXXX   │ ↓ lower  │   Likelihood    │
  └──────────┴────────────┴────────────┴──────────┴─────────────────┘
```

---

## 4. Group Structure

### 4.1 Group Categories

Groups are organized into four categories defined in `src/feature_selection/base_features.py`:

| Category | Description | Example Groups |
|----------|-------------|----------------|
| **CORE_GROUPS** | Global baseline groups (always included) | `alpha_momentum`, `volatility_regime`, `volatility_state`, `gap_dynamics` |
| **HEAD_GROUPS** | Per-model baseline groups (model-specific) | `price_action`, `trend_cross_sectional`, `relative_strength` |
| **CANDIDATE_GROUPS** | Groups available for forward selection | `atr_breakout`, `volume_liquidity`, `weekly_momentum` |
| **INTERACTION_TEMPLATES** | Template-based group interactions | `momentum_x_vol_gate`, `gap_x_vol_state` |

### 4.2 Group Design Rules

Each group must follow these design rules:
- **Size**: 3-12 features per group (validated at startup)
- **Coherence**: One hypothesis per group (features should tell a coherent story)
- **Independence**: Minimal overlap between groups

### 4.3 Mandatory Group Splits

Certain feature domains are split into multiple groups:

**Drawdown/Recovery (4 groups):**
- `drawdown_depth`: How far price has fallen from highs
- `drawdown_duration`: Time-based drawdown metrics
- `recovery_momentum`: Recovery strength after drawdowns
- `bounce_quality`: Quality of bounce from lows

**Volume/Liquidity (3 groups):**
- `volume_surge`: Abnormal volume patterns
- `volume_trend`: Volume moving averages and trends
- `liquidity_stress`: Bid-ask spread, impact measures

**Macro FRED (3 groups):**
- `macro_credit_labor`: Credit spreads and labor market
- `macro_intermarket`: Cross-asset correlations
- `macro_rates_curve`: Yield curve dynamics

---

## 5. Group-First Selection Pipeline

### 5.1 Pipeline Overview

The **Group-First Pipeline** implements selection at the group level:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GROUP-FIRST SELECTION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: Start with Baseline Groups                                 │
│          - CORE_GROUPS (global) + HEAD_GROUPS[model_key]            │
│          - Evaluate baseline metric                                  │
│                                                                      │
│  STEP 2: Grouped Forward Selection                                  │
│          - Add CANDIDATE_GROUPS if improvement > epsilon_add        │
│          - Entire group is added or rejected (atomic)               │
│                                                                      │
│  STEP 3: Enhanced Local Search (Swaps/Add/Drop)                     │
│          - Move types: swap, add, drop                              │
│          - Deterministic hill-climbing with caching                 │
│          - Optional tabu to avoid cycling                           │
│                                                                      │
│  STEP 4: Group Backward Elimination                                 │
│          - Remove groups if loss < epsilon_remove                   │
│          - Optional: allow_baseline_demotions to remove baseline    │
│                                                                      │
│  STEP 5: Template-Based Interaction Selection                       │
│          - Evaluate INTERACTION_TEMPLATES (group-to-group)          │
│          - Only eligible if both parent groups selected             │
│          - Max 3 interaction groups by default                      │
│                                                                      │
│  RESULT: Final selected groups with per-group metrics               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Signal-to-Noise Acceptance Criteria

All selection phases use a **two-gate SNR acceptance rule** to reduce false positives from noisy fold-level metrics. A move is accepted if and only if BOTH gates pass:

```
Gate 1 (Practical Minimum):  Δ_mean > epsilon
Gate 2 (Signal-to-Noise):    t_stat > t_threshold
```

Where:
- **Δ_mean**: Mean improvement across folds = `mean(metric_after[i] - metric_before[i])`
- **Δ_std**: Standard deviation of per-fold improvement deltas (using sample std, ddof=1)
- **SE**: Standard error of the mean = `Δ_std / sqrt(n_folds)`
- **t_stat**: Signal-to-noise ratio = `Δ_mean / (SE + 1e-12)`
- **epsilon**: Minimum practical improvement (move-type-specific)
- **t_threshold**: Minimum t-statistic for reliability (move-type-specific)

**Why Per-Fold Improvement Deltas?**

The key insight is measuring improvement RELATIVE TO THE BASELINE per fold:
- d_i = metric_after_fold_i - metric_before_fold_i

This measures the IMPROVEMENT signal, not absolute metric variability. A feature that improves every fold by 0.001 has low improvement noise, even if absolute AUC varies significantly across folds (0.58, 0.62, 0.60...).

**Why Two Gates?**

Simple threshold checks (`delta >= epsilon`) are susceptible to selection bias:
- With 5 folds and many candidate groups, random variation can exceed epsilon
- This leads to optimistically biased CV metrics that don't generalize

The two-gate approach ensures:
1. **Epsilon Gate**: The improvement has practical significance (not just statistical)
2. **t-stat Gate**: The improvement is reliable, not noise (high signal-to-noise ratio)

**Move-Type-Specific Thresholds:**

| Move Type | epsilon | t_threshold | Rationale |
|-----------|---------|-------------|-----------|
| `add` | 0.0001 | 0.5 | Lenient: want exploration |
| `swap` | 0.0005 | 1.0 | Moderate: meaningful swap |
| `drop` | 0.0005 | 1.0 | Moderate: meaningful drop |
| `add_interaction` | 0.0015 | 0.5 | Higher epsilon for interactions |

**Example (5 folds):**
```
Evaluating group 'momentum_quality' (add move):
  Δ_mean = 0.00035 (improvement)
  Δ_std  = 0.0080 (variance in improvement across folds)
  SE     = 0.0080 / sqrt(5) = 0.00358
  t_stat = 0.00035 / 0.00358 = 0.098

  Gate 1: Δ_mean (0.00035) > epsilon (0.0001) → PASS
  Gate 2: t_stat (0.098) > t_threshold (0.5) → FAIL

  Result: REJECTED
  Reason: t_stat too low - improvement not reliable enough
```

**Grouped Forward Selection (Step 2):**
Accept group if:
- `Δ_mean > epsilon_add` (default: 0.0001) AND `t_stat > t_add` (default: 0.5)
- The entire group is added atomically

**Enhanced Local Search (Step 3):**
Supports three move types in a deterministic hill-climbing loop:
- **swap**: `Δ_mean > epsilon_swap` (0.0005) AND `t_stat > t_swap` (1.0)
- **add**: `Δ_mean > epsilon_add` (0.0001) AND `t_stat > t_add` (0.5)
- **drop**: `Δ_mean > epsilon_drop` (0.0005) AND `t_stat > t_drop` (1.0)

Features:
- **Caching**: Evaluation results cached to avoid redundant CV calls
- **Tabu** (optional): Prevents cycling by forbidding recent moves
- **Deterministic**: No randomness, reproducible results
- **SNR-Adjusted**: All moves use per-fold delta statistics for acceptance

**Group Backward Elimination (Step 4):**
Remove group if:
- `Δ_mean >= -epsilon_remove` (loss within acceptable tolerance)
- If `allow_baseline_demotions=True`, baseline groups can also be removed
- Note: Backward elimination uses loss tolerance, not t-stat gate

**Template-Based Interactions (Step 5):**
- Templates define group-to-group interactions (not feature×feature)
- A template is **eligible** only if both parent groups are selected
- Add up to `max_interaction_groups` (default: 3) interaction groups
- Uses `epsilon_add_interaction` (0.0015) and `t_add_interaction` (0.5)

### 5.3 Holdout Evaluation

To detect selection bias, the pipeline supports temporal holdout evaluation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOLDOUT EVALUATION                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Full Dataset:  |──────────────────────────────────────────────|    │
│                                                                      │
│  Train (95%):   |─────────────────────────────────────────|         │
│                 ^                                         ^          │
│                 │  CV folds for feature selection         │          │
│                                                                      │
│  Holdout (5%):                                          |────|       │
│                                                          ^    ^      │
│                                                          │    │      │
│                              Unbiased evaluation ────────┘    │      │
│                                                               │      │
└─────────────────────────────────────────────────────────────────────┘
```

**CLI Usage:**
```bash
# Default: 5% holdout
python run_group_selection.py --model long_normal --holdout-pct 0.05

# Disable holdout (not recommended)
python run_group_selection.py --model long_normal --holdout-pct 0
```

**Interpreting Results:**
```
CV AUC:      0.8850
Holdout AUC: 0.8500
Gap:         +0.0350
```

| CV-Holdout Gap | Interpretation |
|----------------|----------------|
| < 0.05 | ✅ OK - Selection appears robust |
| 0.05 - 0.10 | ⚠️ NOTICE - Some selection bias present |
| > 0.10 | ❌ WARNING - Significant selection bias, investigate feature choices |

**Why Holdout Matters:**
- CV metrics are optimistically biased because features were selected to maximize them
- Holdout data was never seen during selection (neither train nor test folds)
- The gap between CV and holdout AUC reveals the degree of selection bias

### 5.4 Configuration

```python
@dataclass
class GroupSelectionConfig:
    # Epsilon thresholds (minimum improvement in metric units)
    epsilon_add: float = 0.0001              # Min improvement to add
    epsilon_swap: float = 0.0005             # Min improvement for swaps
    epsilon_remove: float = 0.001            # Max loss to remove
    epsilon_drop: float = 0.0005             # Min improvement for drop moves
    epsilon_add_interaction: float = 0.0015  # Min improvement for interactions

    # Signal-to-noise thresholds (t = Δ_mean / SE)
    t_add: float = 0.5                       # t-threshold for adds (lenient)
    t_swap: float = 1.0                      # t-threshold for swaps (moderate)
    t_drop: float = 1.0                      # t-threshold for drops (moderate)
    t_add_interaction: float = 0.5           # t-threshold for interaction adds

    # Debug flag for acceptance diagnostics
    debug_acceptance: bool = False           # Print per-move acceptance details

    # Group constraints
    allow_baseline_demotions: bool = False
    max_groups: int = 20
    max_interaction_groups: int = 3

    # Enhanced local search
    enable_add_drop_moves: bool = True       # Enable add/drop moves
    max_search_iterations: int = 50          # Max iterations

    # Tabu mechanism (optional)
    enable_tabu: bool = False                # Enable tabu list
    tabu_tenure: int = 5                     # Iterations to keep moves tabu
    tabu_aspiration_delta: float = 0.005     # Override tabu threshold

    # Caching
    enable_caching: bool = True              # Cache evaluation results

    verbose: bool = True
```

### 5.5 Over-Regularization Strategy

We intentionally choose slightly sub-optimal but robust group sets:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `epsilon_add` | 0.0001 | Minimum practical improvement for adds |
| `epsilon_swap` | 0.0005 | Meaningful improvement for swaps |
| `t_add` | 0.5 | Lenient t-threshold for exploration |
| `t_swap` | 1.0 | Moderate t-threshold for reliability |
| `epsilon_remove` | 0.001 | Only keep groups that clearly help |
| `max_groups` | 20 | Limit total model complexity |

**Why over-regularize?**
- Groups that barely improve CV may not generalize
- Financial regimes change - robust groups preferred
- Fewer groups = more interpretable model

---

## 6. Group Definitions

### 6.1 CORE_GROUPS (Global Baseline)

CORE_GROUPS are included for all models and represent universally predictive signals:

```python
CORE_GROUPS = {
    "alpha_momentum": [...],        # Alpha vs SPY/sector
    "macro_credit_labor": [...],    # Credit spreads + labor market
    "macro_intermarket": [...],     # Cross-asset correlations
    "trend_strength": [...],        # Trend direction signals
    "price_position": [...],        # Mean reversion signals
    "sector_breadth": [...],        # Market breadth
    "momentum_quality": [...],      # Momentum validation
    "range_breakout": [...],        # Range position
    # Refactored from market_regime (split for hypothesis purity):
    "volatility_regime": [...],     # VIX percentile, zscore, vol regime
    "volume_shock": [...],          # Volume shocks, divergences
    "microstructure_position": [...], # VWAP distance, overnight ratio
    # Refactored from volatility_squeeze (split for hypothesis purity):
    "volatility_state": [...],      # BB width, squeeze intensity, RV zscore
    "gap_dynamics": [...],          # Gap/ATR ratio, overnight return (LAGGED 1 day)
}
```

**Note on Gap Features:** All gap-related features (`gap_atr_ratio`, `gap_atr_ratio_raw`, `overnight_ret`, `gap_fill_frac`, `overnight_ratio`) are **lagged by 1 day** to prevent data leakage. When predicting day T returns, these features reflect the gap behavior from day T-1, not day T. This prevents using today's open price (which is part of today's trading activity) as a predictor.

### 6.2 HEAD_GROUPS (Per-Model Baseline)

HEAD_GROUPS are model-specific baseline features:

```python
HEAD_GROUPS = {
    ModelKey.LONG_NORMAL: {
        "drawdown_recovery": [...],      # Drawdown and recovery
        "relative_strength": [...],      # Relative performance (extended)
        "macro_sector": [...],           # Macro + sector signals
        # Refactored from price_momentum (split for hypothesis purity):
        "price_action": [...],           # Candlestick patterns, VWAP, RSI divergence
        "trend_cross_sectional": [...],  # Trend slope, MA slope, cross-sectional momentum
    },
    ModelKey.SHORT_NORMAL: {
        "breakdown_signals": [...],      # Breakdown detection
        "reversal_risk": [...],          # Reversal indicators
        ...
    },
    # ... other models
}
```

### 6.3 CANDIDATE_GROUPS

Groups available for forward selection from EXPANSION_CANDIDATES.

### 6.4 INTERACTION_GROUPS

Curated interaction feature groups (not generated pairwise):

```python
INTERACTION_GROUPS = {
    "momentum_vol_gate": [         # Momentum gated by volatility
        "interact_rsi_vol_regime",
        "interact_macd_vix_z",
    ],
    "breadth_trend_confirm": [     # Breadth confirming trend
        "interact_breadth_trend",
    ],
    ...
}
```

---

## 7. Thematic Pair and Triple Selection

### 7.1 Design Philosophy

Pairs and triples should represent **meaningful thematic combinations**, not arbitrary interactions. Each combination should tell a coherent story about market state.

### 7.2 Example Thematic Combinations

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

### 7.3 Domain-Aware Interaction Patterns

The pipeline uses comprehensive domain knowledge from quantitative trading research to filter and prioritize interaction candidates. Pattern matching is substring-based.

```python
# From src/feature_selection/interactions.py - DOMAIN_PATTERNS
# Curated from 20+ years of systematic trading experience

# === VOLATILITY REGIME × SIGNALS (The Regime Gate) ===
# Volatility regime is the single most important conditioning variable.
# In high-vol, mean-reversion dominates; in low-vol, trends persist.
('vol', 'momentum'),      # RSI, MACD effectiveness depends on vol regime
('vol', 'rsi'),           # RSI overbought/oversold zones widen in high vol
('regime', 'momentum'),   # Regime state gates momentum signals
('regime', 'alpha'),      # Alpha decay rate differs by regime

# === BREADTH × LOCAL STOCK GEOMETRY (Confirmation Patterns) ===
# Individual stock signals are more reliable when confirmed by market breadth
('breadth', 'momentum'),  # Stock momentum × market participation
('breadth', 'pos'),       # pos_in_*d_range features × breadth
('breadth', 'trend'),     # Trend strength × breadth confirmation

# === CROSS-SECTIONAL × TIME-SERIES (Compounding Effects) ===
# Top-decile stocks with strong TS momentum = "winners that are winning more"
('xsec', 'momentum'),     # XS momentum z-score × individual signals
('rank', 'trend'),        # Cross-sectional rank × trend strength

# === VIX/MACRO × SIGNALS (Fear Gate) ===
# High VIX = correlated selloffs, stock-specific signals less reliable
('vix', 'momentum'),      # Momentum signals × fear level
('vix', 'alpha'),         # Alpha signals fail in high VIX

# === ATR/RANGE × MOMENTUM (Volatility-Adjusted Signals) ===
# Raw momentum signals must be scaled by recent volatility
('atr', 'momentum'),      # ATR-scaled momentum
('atr', 'return'),        # ATR-scaled returns

# ... 100+ more patterns covering macro, liquidity, candlestick, etc.
```

See `src/feature_selection/interactions.py` for the complete `DOMAIN_PATTERNS` list with detailed rationale for each pattern category.

### 7.4 Interaction Types

The pipeline supports four interaction types based on the economic relationship between features:

| Type | Formula | Use Case | Example |
|------|---------|----------|---------|
| **PRODUCT** | `f1 × f2` | Multiplicative confirmation - both signals reinforce | `breadth × momentum` |
| **GATED** | `f1 × sign(f2)` | One feature conditions another's reliability | `momentum × sign(vol_regime)` |
| **RATIO** | `f1 / (\|f2\| + ε)` | Scale-invariant comparison | `momentum / ATR` |
| **THRESHOLD** | `I(f1 > med) × I(f2 > med)` | Non-linear regime switching | `squeeze × breakout` |

**Pattern-to-Type Mapping:**

```python
# From src/feature_selection/interactions.py - PATTERN_INTERACTION_TYPES
PATTERN_INTERACTION_TYPES = {
    # Regime-gating patterns (one feature conditions the other)
    ('vol', 'momentum'): [InteractionType.GATED, InteractionType.PRODUCT],
    ('vix', 'alpha'): [InteractionType.GATED],
    ('regime', 'momentum'): [InteractionType.GATED],

    # Confirmation patterns (multiplicative)
    ('breadth', 'momentum'): [InteractionType.PRODUCT],
    ('xsec', 'momentum'): [InteractionType.PRODUCT],
    ('vwap', 'trend'): [InteractionType.PRODUCT],

    # Scale-invariant patterns
    ('atr', 'momentum'): [InteractionType.RATIO, InteractionType.PRODUCT],
    ('atr', 'return'): [InteractionType.RATIO],

    # Threshold patterns (regime switching)
    ('squeeze', 'momentum'): [InteractionType.THRESHOLD, InteractionType.GATED],
    ('drawdown', 'momentum'): [InteractionType.THRESHOLD, InteractionType.GATED],
}
```

### 7.5 Template-Based Interaction System

The template-based interaction system replaces ad-hoc feature×feature interactions with
**thematic group-to-group templates**. Each template represents a coherent economic
hypothesis about how two feature groups interact.

#### Template Structure

```python
INTERACTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "momentum_x_vol_gate": {
        "parents": ("momentum_quality", "volatility_state"),  # volatility_state (was volatility_squeeze)
        "type": "gate",  # or "signed_gate" or "product"
        "base_features": ["rsi_14", "adx_14", "chop_14"],
        "gate_features": ["squeeze_intensity_20", "rv_z_60"],
        "description": "Momentum indicators gated by volatility state",
    },
    "gap_x_vol_state": {
        "parents": ("gap_dynamics", "volatility_state"),
        "type": "gate",
        "base_features": ["gap_atr_ratio", "overnight_ret"],
        "gate_features": ["squeeze_intensity_20", "bb_width_20_2"],
        "description": "Gap dynamics gated by volatility state",
    },
    # ... more templates
}
```

| Field | Description |
|-------|-------------|
| `parents` | Tuple of (group_A, group_B) from selection groups |
| `type` | `"gate"`, `"signed_gate"`, or `"product"` |
| `base_features` | Features from parent_A to use as signals |
| `gate_features` | Features from parent_B to use as gates/modifiers |
| `invert_gate` | (Optional) If True, invert gate logic |

#### Interaction Types

| Type | Formula | When to Use |
|------|---------|-------------|
| **gate** | `base × I(gate > median)` | Binary regime conditioning |
| **signed_gate** | `base × sign(gate)` | Direction matters (bullish/bearish) |
| **product** | `base × gate` | Multiplicative confirmation |

#### Naming Convention

Generated feature names follow a deterministic scheme:
```
ix__{template_name}__{base_feat}__gated__{gate_feat}   # for gate/signed_gate
ix__{template_name}__{feat_a}__x__{feat_b}             # for product
```

Example: `ix__momentum_x_vol_gate__rsi_14__gated__vol_regime_ema10`

#### Eligibility Rules

A template is **eligible** for selection only when **both parent groups** are
already in the selected feature set. This ensures:
1. No orphan interactions (interactions without base signals)
2. Interactions build on established relationships
3. Selection remains interpretable

#### Implemented Templates

| Template | Parents | Type | Hypothesis |
|----------|---------|------|------------|
| `momentum_x_vol_gate` | momentum_quality × volatility_state | gate | Momentum reliable in specific vol states |
| `trend_x_vol_gate` | trend_strength × volatility_state | signed_gate | Trends persist when vol is stable |
| `breadth_x_trend_confirm` | sector_breadth × trend_strength | product | Breadth confirms trend quality |
| `drawdown_x_breadth` | drawdown_level × breadth_motion | gate | Drawdown signals + breadth direction |
| `alpha_x_macro_regime` | alpha_momentum × macro_credit_labor | signed_gate | Alpha reliable in stable macro |
| `price_position_x_regime` | price_position × volatility_regime | gate | Mean reversion works in calm markets |
| `breakout_x_squeeze` | range_breakout × volatility_state | product | Breakouts amplified by squeeze release |
| `gap_x_vol_state` | gap_dynamics × volatility_state | gate | Gap behavior gated by volatility state |
| `microstructure_x_volume` | microstructure_position × volume_shock | product | VWAP/microstructure × volume shocks |
| `vol_regime_x_momentum` | volatility_regime × momentum_quality | signed_gate | Vol regime conditions momentum signals |

#### Example Output

```python
# Template: momentum_x_vol_gate
# Parents: momentum_quality, volatility_state
# Generated features (6 = 3 base × 2 gates):
[
    "ix__momentum_x_vol_gate__rsi_14__gated__squeeze_intensity_20",
    "ix__momentum_x_vol_gate__rsi_14__gated__rv_z_60",
    "ix__momentum_x_vol_gate__adx_14__gated__squeeze_intensity_20",
    "ix__momentum_x_vol_gate__adx_14__gated__rv_z_60",
    "ix__momentum_x_vol_gate__chop_14__gated__squeeze_intensity_20",
    "ix__momentum_x_vol_gate__chop_14__gated__rv_z_60",
]
```

---

## 8. Parallelization Strategy

### 8.1 Core Principle: Optimize CPU Utilization

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

### 8.2 Configuration Parameters

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

### 8.3 Dataset Size Heuristics

| Dataset Rows | Single Model Utilization | Recommended Strategy |
|--------------|--------------------------|----------------------|
| < 10,000 | Low | `n_jobs=8, num_threads=1` |
| 10,000 - 50,000 | Medium | `n_jobs=4, num_threads=2` |
| 50,000 - 200,000 | High | `n_jobs=2, num_threads=4` |
| > 200,000 | Full | `n_jobs=1, num_threads=-1` |

### 8.4 Joblib Parallelization Pattern

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

### 8.5 Group Selection Parallelism

Group selection (`grouped_swap_selection`) evaluates candidate moves in parallel.
The key configuration options in `GroupSelectionConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parallelize_moves` | `True` | Enable parallel move evaluation |
| `n_move_workers` | `-1` | Workers for move eval (-1 = use n_jobs) |
| `use_loky_for_moves` | `True` | Use loky (processes) instead of threading |
| `debug_parallelism` | `False` | Print PID from worker processes for verification |

**Why Loky (Processes) Instead of Threading?**

LightGBM training is CPU-bound. Python's Global Interpreter Lock (GIL) prevents
threads from running Python code in parallel. With threading backend, only one
worker effectively runs at a time, serializing all move evaluations.

Loky (process-based) parallelism spawns separate processes that each have their
own Python interpreter and bypass the GIL, enabling true parallel execution.

**Trade-offs:**
- **Pro**: True parallelism for CPU-bound work
- **Con**: Objects must be picklable (evaluator is pickled to workers)
- **Con**: In-memory cache cannot be shared (disabled in loky mode)
- **Con**: Slightly slower startup (process spawn vs thread)

**Debugging Parallelism:**

To verify that multiple processes are actually running in parallel:

```python
config = GroupSelectionConfig(
    n_jobs=4,
    use_loky_for_moves=True,
    debug_parallelism=True,  # Enable PID logging
)
```

This prints `[PID 12345] Evaluating move: swap:group_a->group_b` from each
worker, showing different PIDs confirms parallel execution.

---

## 9. Cross-Validation with Purging

### 9.1 Time-Series CV Configuration

```python
@dataclass
class CVConfig:
    n_splits: int = 5
    scheme: CVScheme = CVScheme.EXPANDING
    gap: int = 20       # Embargo: skip 20 samples between train/test
    purge_window: int = 0
    min_train_samples: int = 1000
```

### 9.2 Purging and Embargo

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

## 10. Model Configuration

### 10.1 LightGBM Settings

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

### 10.2 Regularization for Robustness

The model intentionally uses moderate regularization:
- `reg_alpha = 0.1`: L1 penalty shrinks weak features to zero
- `reg_lambda = 0.1`: L2 penalty prevents large coefficients
- `min_child_samples = 20`: Prevents splits on small groups
- `max_depth = 6`: Limits tree complexity

---

## 11. Checkpointing and Resumption

### 11.1 Checkpoint Structure

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

### 11.2 Resumption

```bash
# Check checkpoint status
python run_feature_selection.py --checkpoint-info

# Resume from checkpoint
python run_feature_selection.py --resume
```

### 11.3 Stage Identifiers

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

## 12. Running Group Selection

### 12.1 Basic Usage (Group-First)

```bash
# Run group selection for a single model
python run_group_selection.py --model long_normal

# Run for all 4 models
python run_group_selection.py --model all

# Allow baseline group demotions during backward elimination
python run_group_selection.py --model long_normal --allow-demotions

# Adjust selection thresholds
python run_group_selection.py --model long_normal --epsilon-add 0.003 --epsilon-swap 0.002
```

### 12.2 CLI Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `long_normal` | Model: `long_normal`, `long_parabolic`, `short_normal`, `short_parabolic`, or `all` |
| `--epsilon-add` | 0.0001 | Minimum improvement to add a group |
| `--epsilon-swap` | 0.0005 | Minimum improvement for swaps |
| `--allow-demotions` | False | Allow dropping baseline groups |
| `--max-groups` | 20 | Maximum total groups to select |
| `--max-symbols` | 5000 | Maximum symbols to use |
| `--balanced` | False | Use class weights (scale_pos_weight) |
| `--use-weights` | False | Use sample weights from overlap inverse weighting |
| `--n-folds` | 5 | Number of CV folds |
| `--holdout-pct` | 0.05 | Fraction of dates for holdout evaluation (0 to disable) |
| `--n-jobs` | 4 | Number of parallel jobs for CV |
| `--model-threads` | 1 | Threads per LightGBM model |
| `--output-dir` | `artifacts/group_selection` | Output directory |
| `--quiet` | False | Reduce verbosity |

**K-of-N Feature Selection:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--disable-k-of-n` | False | Disable K-of-N selection (use all features per group) |
| `--group-k` | 2 | Default K for K-of-N selection within groups |
| `--epsilon-add-feature` | 0.0005 | Minimum improvement to add a feature within group |

### 12.3 Programmatic Usage

```python
from src.feature_selection import (
    run_group_selection,
    GroupSelectionConfig,
    ModelConfig, ModelType, TaskType,
    CVConfig, CVScheme,
    MetricConfig, MetricType,
    SearchConfig,
)
from src.config.model_keys import ModelKey

# Configure model
model_config = ModelConfig(
    model_type=ModelType.LIGHTGBM,
    task_type=TaskType.CLASSIFICATION,
    params={'learning_rate': 0.03, 'max_depth': 5}
)

# Configure CV
cv_config = CVConfig(
    n_splits=5,
    scheme=CVScheme.EXPANDING,
    gap=5,
    purge_window=2,
)

# Configure metrics
metric_config = MetricConfig(
    primary_metric=MetricType.AUC,
    secondary_metrics=[MetricType.LOG_LOSS],
)

# Configure group selection (with SNR acceptance thresholds)
config = GroupSelectionConfig(
    # Epsilon thresholds (minimum improvement in metric units)
    epsilon_add=0.0001,           # Min improvement to add a group
    epsilon_swap=0.0005,          # Min improvement for swaps
    epsilon_remove=0.001,         # Max loss allowed when removing
    epsilon_drop=0.0005,          # Min improvement for drop moves
    epsilon_add_interaction=0.0015,  # Min improvement for interactions

    # Signal-to-noise thresholds (t = delta_mean / SE)
    t_add=0.5,                    # t-threshold for adds (lenient)
    t_swap=1.0,                   # t-threshold for swaps (moderate)
    t_drop=1.0,                   # t-threshold for drops (moderate)
    t_add_interaction=0.5,        # t-threshold for interaction adds

    # Group constraints
    allow_baseline_demotions=False,
    max_groups=20,
    max_interaction_groups=3,

    # Debug
    debug_acceptance=False,       # Print per-move acceptance details
)

# Run selection
result = run_group_selection(
    X=X,
    y=y,
    model_key=ModelKey.LONG_NORMAL,
    model_config=model_config,
    cv_config=cv_config,
    metric_config=metric_config,
    search_config=SearchConfig(),
    config=config,
)

# Access results
print(f"Selected groups: {list(result.selected_groups.keys())}")
print(f"Total features: {len(result.selected_features)}")
print(f"Baseline AUC: {result.baseline_metric:.4f}")
print(f"Final AUC: {result.final_metric:.4f}")
```

### 12.4 Legacy Singleton Selection

The legacy singleton-based selection (`run_feature_selection.py`) is still available for backwards compatibility but is deprecated in favor of group-first selection.

### 12.5 Using Sample Weights

Sample weights from triple barrier targets (overlap inverse) are **opt-in** and disabled by default. To enable them, use the `--use-weights` flag:

```bash
# Enable sample weights
python run_group_selection.py --model long_normal --use-weights
```

When enabled, weights are loaded from `targets_triple_barrier.parquet` and passed through the pipeline:

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

## 13. Output Files

### 13.1 Feature Selection Artifacts

```
artifacts/feature_selection/
├── checkpoint.pkl           # Pipeline checkpoint (auto-saved)
├── selected_features.txt    # Final selected features (one per line)
├── selection_summary.json   # Complete metrics, features, and config
├── stage_summary.csv        # Per-stage metrics in CSV format
└── feature_importance.csv   # Feature importance scores (if computed)
```

### 13.2 Results Format

```python
# selected_features.txt
# Best features from Loose-then-Tight pipeline
# Stage: 3_strict_backward
# Metric: [mean] +/- [std]

rsi_14
w_macd_histogram
trend_score_sign
...

# selection_summary.json
{
  "best_stage": "3_strict_backward",
  "n_features": ...,
  "metric_mean": ...,
  "metric_std": ...,
  "features": ["alpha_mom_spy_20_ema10", "atr_percent", "rsi_14", ...],
  "stages": [
    {"stage": "1_base_features", "n_features": ..., "metric_mean": ..., "metric_std": ..., "is_best": false},
    {"stage": "2_loose_forward", "n_features": ..., "metric_mean": ..., "metric_std": ..., "is_best": false},
    {"stage": "3_strict_backward", "n_features": ..., "metric_mean": ..., "metric_std": ..., "is_best": true}
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

## 14. Key Files Reference

| File | Purpose |
|------|---------|
| `src/feature_selection/base_features.py` | CORE_GROUPS, HEAD_GROUPS, CANDIDATE_GROUPS, INTERACTION_TEMPLATES |
| `src/feature_selection/group_selection.py` | Group-first selection algorithms, `_snr_acceptance()` |
| `src/feature_selection/config.py` | Configuration dataclasses including GroupSelectionConfig |
| `src/feature_selection/evaluation.py` | SubsetEvaluator for CV evaluation (returns `fold_metrics`) |
| `src/feature_selection/cv.py` | Time-series CV with purging |
| `src/feature_selection/pipeline.py` | Loose-Then-Tight pipeline (legacy) |
| `src/feature_selection/algorithms.py` | Singleton forward/backward/swap (legacy) |
| `src/feature_selection/interactions.py` | DOMAIN_PATTERNS, InteractionType |
| `src/features/target_generation.py` | Triple barrier targets and weighting |
| `run_group_selection.py` | Group-first selection entry point with holdout evaluation |
| `run_feature_selection.py` | Singleton selection entry point (legacy) |

---

## 15. Best Practices

### 15.1 Feature Design

1. **Normalize features**: Use z-scores, percentiles, or ratios (not raw prices)
2. **Include multi-timeframe**: Daily + weekly versions capture different signals
3. **Domain knowledge**: Group features by economic meaning
4. **Avoid redundancy**: Check correlation between candidate features

### 15.2 Selection Process

1. **Start with BASE_FEATURES**: Proven feature set as starting point
2. **Use sample weights**: Always weight by overlap inverse
3. **Validate on holdout**: Final validation on unseen time period
4. **Monitor stability**: Features should be consistent across CV folds

### 15.3 Parallelization

1. **Match to dataset size**: Adjust n_jobs based on data volume
2. **Avoid over-parallelization**: n_jobs * num_threads ~ CPU_count
3. **Use checkpointing**: Long-running pipelines should checkpoint frequently
4. **Monitor memory**: Parallel workers share data via serialization

---

## 16. Future Enhancements

### 16.1 Planned Features

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
  │  Technology         │ X.XXXX │
  │  Health Care        │ X.XXXX │
  │  Financials         │ X.XXXX │
  │  Consumer Discret.  │ X.XXXX │
  │  ...                │ ...    │
  └─────────────────────┴────────┘
  Sector AUC Std Dev: 0.XXXX (< 0.03 = consistent)
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

## 17. Troubleshooting

### 17.1 Common Issues

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

### 17.2 Debugging

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

---

## 18. Multi-Model Feature Selection

### 18.1 Overview

The multi-model feature selection system runs the Loose-Tight pipeline independently for each of the 4 model targets while ensuring:

- **Same CV splits**: All models use identical time-series CV splits for comparability
- **Same feature universe**: All models start from the same candidate features
- **Single data load**: Features are computed once and reused

**Model Keys:**

| Model Key | Target | Interpretation |
|-----------|--------|----------------|
| `LONG_NORMAL` | Upper barrier hit | Standard long momentum (1.5 ATR) |
| `LONG_PARABOLIC` | Extended upper | Trend persistence / parabolic moves |
| `SHORT_NORMAL` | Lower barrier hit | Breakdown / fragility setups |
| `SHORT_PARABOLIC` | Extended lower | Panic / regime shift |

### 18.2 Running Multi-Model Selection

**CLI Usage:**

```bash
# Run all 4 models (default)
python scripts/run_feature_selection_multimodel.py

# Run specific models
python scripts/run_feature_selection_multimodel.py --model-keys LONG_NORMAL,SHORT_NORMAL

# Limit max features per model
python scripts/run_feature_selection_multimodel.py --max-features 50

# Use cached data for faster iteration
python scripts/run_feature_selection_multimodel.py --use-cache

# Custom output directory
python scripts/run_feature_selection_multimodel.py --output-dir artifacts/my_selection
```

**Programmatic Usage:**

```python
from src.feature_selection.multimodel import (
    run_single_model_selection,
    compute_overlap_analysis,
    write_per_model_artifacts,
    write_global_summary,
)
from src.config.model_keys import ModelKey

# Run for a single model
result = run_single_model_selection(
    X=features_df,
    y=target_series,
    model_key=ModelKey.LONG_NORMAL,
    features=candidate_features,
    sample_weight=weights,
    verbose=True,
)

# Access results
print(f"Selected: {result.n_features} features")
print(f"AUC: {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f}")
print(f"Features: {result.selected_features}")
```

### 18.3 Output Artifacts

**Per-Model Artifacts:**

```
artifacts/feature_selection/
├── long_normal/
│   ├── selected_features.json    # Full metadata
│   └── selected_features.txt     # Feature list only
├── long_parabolic/
│   ├── selected_features.json
│   └── selected_features.txt
├── short_normal/
│   ├── selected_features.json
│   └── selected_features.txt
├── short_parabolic/
│   ├── selected_features.json
│   └── selected_features.txt
└── summary.json                  # Global summary with overlap analysis
```

**Per-Model JSON Schema (`selected_features.json`):**

```json
{
  "model_key": "long_normal",
  "selected_features": ["feat1", "feat2", "..."],
  "n_features": 42,
  "cv_auc_mean": 0.6789,
  "cv_auc_std": 0.0123,
  "fold_metrics": [0.68, 0.67, 0.69, 0.68, 0.68],
  "secondary_metrics": {
    "auc": {"mean": 0.6789, "std": 0.0123}
  },
  "best_stage": "3_strict_backward",
  "algorithm": "loose_tight_pipeline",
  "algorithm_params": {
    "epsilon_add_loose": 0.0002,
    "epsilon_remove_strict": 0.0,
    "max_features_loose": 80
  },
  "cv_config_hash": "a1b2c3d4e5f6",
  "date_range": ["2020-01-01", "2024-12-01"],
  "universe_hash": "f6e5d4c3b2a1",
  "timestamp": "2024-12-22T10:30:00Z",
  "run_signature": "abc123def456",
  "elapsed_seconds": 1234.5
}
```

**Global Summary Schema (`summary.json`):**

```json
{
  "results": {
    "long_normal": { "...per-model result..." },
    "long_parabolic": { "..." },
    "short_normal": { "..." },
    "short_parabolic": { "..." }
  },
  "core_features": ["feat1", "feat2", "..."],
  "head_features": {
    "long_normal": ["unique_feat1", "..."],
    "long_parabolic": ["unique_feat2", "..."],
    "short_normal": ["unique_feat3", "..."],
    "short_parabolic": ["unique_feat4", "..."]
  },
  "overlap_matrix": {
    "long_normal": {
      "long_normal": 1.0,
      "long_parabolic": 0.75,
      "short_normal": 0.60,
      "short_parabolic": 0.55
    }
  },
  "intersection_matrix": {
    "long_normal": {
      "long_normal": 42,
      "long_parabolic": 35,
      "short_normal": 30,
      "short_parabolic": 28
    }
  },
  "union_size": 85,
  "top_shared_features": [
    {"feature": "rsi_14", "count": 4},
    {"feature": "atr_percent", "count": 4}
  ],
  "run_signature": "combined_hash",
  "timestamp": "2024-12-22T10:30:00Z"
}
```

### 18.4 CORE and HEAD Features

**CORE_FEATURES**: Features selected by ALL 4 models (intersection). These represent the most universally predictive signals.

**HEAD_FEATURES[model_key]**: Features unique to each model (selected by that model but not in CORE). These capture model-specific patterns.

```
Total Features = CORE_FEATURES ∪ HEAD_FEATURES[model_key]
CORE_FEATURES  = ∩ (all selected feature sets)
HEAD_FEATURES[mk] = selected[mk] - CORE_FEATURES
```

**Example Analysis:**

```python
import json
from pathlib import Path

# Load summary
with open('artifacts/feature_selection/summary.json') as f:
    summary = json.load(f)

# Core features (shared by all models)
print(f"CORE: {len(summary['core_features'])} features")
for feat in summary['core_features'][:5]:
    print(f"  - {feat}")

# Model-specific head features
for model_key, head_feats in summary['head_features'].items():
    print(f"\n{model_key}: {len(head_feats)} unique features")
    for feat in head_feats[:3]:
        print(f"  - {feat}")
```

### 18.5 Overlap Analysis

**Jaccard Similarity**: Measures pairwise overlap between model feature sets:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

- 1.0 = identical feature sets
- 0.0 = completely disjoint

**Interpretation:**

| Jaccard Range | Interpretation |
|---------------|----------------|
| > 0.8 | Very similar models, may be redundant |
| 0.5 - 0.8 | Moderate overlap, models share core signals |
| 0.3 - 0.5 | Low overlap, models capture different patterns |
| < 0.3 | Very different, models may target distinct regimes |

### 18.6 Run Signature for Reproducibility

Each selection run generates a unique signature combining:

- Feature universe hash (which features were candidates)
- CV configuration hash (splits, gap, purge)
- Date range of data
- Algorithm parameters
- Model key

```python
from src.feature_selection.multimodel import compute_run_signature

signature = compute_run_signature(
    universe_hash="a1b2c3",
    cv_config_hash="d4e5f6",
    date_range=("2020-01-01", "2024-12-01"),
    algorithm_params={"epsilon_add_loose": 0.0002},
    model_key="long_normal",
)
# Returns: "abc123def456" (12-char hex hash)
```

This signature enables:
- Tracking which runs produced which feature sets
- Detecting when results need to be regenerated (data/config changed)
- Reproducibility auditing

### 18.7 Label Column Structure

Each model uses its own target column from triple barrier labeling with different ATR multiples:

| Model Key | Target Column | Barrier Config | Success Condition |
|-----------|---------------|----------------|-------------------|
| `LONG_NORMAL` | `hit_long_normal` | 1.5x ATR up/down | Upper barrier hit (1) |
| `LONG_PARABOLIC` | `hit_long_parabolic` | 2.5x ATR up, 1.5x down | Upper barrier hit (1) |
| `SHORT_NORMAL` | `hit_short_normal` | 1.5x up, 2.0x ATR down | Lower barrier hit (-1) |
| `SHORT_PARABOLIC` | `hit_short_parabolic` | 1.5x up, 2.5x ATR down | Lower barrier hit (-1) |

**Hit column values:**
- `1`: Upper barrier hit (profit for long)
- `-1`: Lower barrier hit (profit for short)
- `0`: Timeout (neither barrier hit within horizon)

The `get_model_labels()` function automatically:
1. Selects the correct column for each model
2. Filters out timeout samples (hit=0) for binary classification
3. Converts to binary labels (1=success, 0=failure)

### 18.8 Best Practices

1. **Run all 4 models together**: Ensures identical CV splits and feature universe
2. **Use caching for iteration**: `--use-cache` speeds up re-runs during tuning
3. **Check overlap matrix**: Very high overlap (>0.9) suggests models may be redundant
4. **Monitor CORE size**: A small CORE indicates models target very different signals
5. **Validate against registry**: Ensure selected features exist in `base_features.py`

### 18.9 Auto-Updating Feature Registry

After multi-model selection, you can automatically update `base_features.py` with the new CORE/HEAD features:

```bash
# Run selection and auto-update registry
python scripts/run_feature_selection_multimodel.py --update-registry

# Preview changes without modifying (dry run)
python scripts/run_feature_selection_multimodel.py --update-registry --dry-run

# Skip backup when updating
python scripts/run_feature_selection_multimodel.py --update-registry --no-backup
```

**What gets updated:**

| Registry Variable | Updated To |
|-------------------|------------|
| `CORE_FEATURES` | Intersection of all 4 model selections |
| `HEAD_FEATURES[model_key]` | Model-specific features (selected minus CORE) |

**Important notes:**

1. **Nothing is protected**: The backward elimination stage can remove ANY feature, including those currently in CORE_FEATURES. Features earn their place through CV performance.

2. **Backup created**: By default, a timestamped backup is created before modifying (`base_features.py.backup_YYYYMMDD_HHMMSS`).

3. **Diff displayed**: The script shows added/removed features before applying changes.

4. **Timestamp tracking**: Updated registry includes auto-generation timestamp and run signature for traceability.

**Programmatic usage:**

```python
from src.feature_selection.multimodel import (
    update_base_features_file,
    compute_feature_diff,
    print_feature_diff,
)
from src.feature_selection.base_features import CORE_FEATURES, HEAD_FEATURES

# After running selection and getting summary...
diff = compute_feature_diff(CORE_FEATURES, old_head_dict, summary)
print_feature_diff(diff)

# Apply update
result = update_base_features_file(
    summary=summary,
    backup=True,
    dry_run=False,
    verbose=True,
)
```

### 18.10 Integration with Training Pipeline

After multi-model selection, the results feed into model training:

```python
# Load selected features per model
import json

def load_selected_features(model_key: str) -> list:
    path = f'artifacts/feature_selection/{model_key}/selected_features.json'
    with open(path) as f:
        return json.load(f)['selected_features']

# Use in training
for model_key in ['long_normal', 'long_parabolic', 'short_normal', 'short_parabolic']:
    features = load_selected_features(model_key)
    X_train = features_df[features]
    # ... train model ...
```
