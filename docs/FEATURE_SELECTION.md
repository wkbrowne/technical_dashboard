# Feature Selection Architecture

## Edit Summary

- **Targets out of scope** — target semantics, label mapping, and barrier concepts deferred entirely to TARGETS.md
- **Softened hardcoded defaults** — code snippets show placeholders with comments directing to authoritative sources (`config.py`, `--help`); parallelism table labeled as "typical/example"
- **Clarified threshold precedence** — SNR thresholds table labeled as library defaults; explicitly noted that **only** `--epsilon-add` and `--epsilon-swap` are CLI-overridable; all other thresholds remain library defaults with no CLI override
- **Resolved BASE_FEATURES terminology** — defined as `flatten(Baseline Groups)` in terminology table; group-level terms used consistently elsewhere
- **Softened unverified claims** — removed "validated at startup" language; avoided guarantees about thread semantics; use convention language for naming/ordering
- **Added Section 6.5** — documented the interaction search algorithm (eligibility, enumeration, acceptance, caching, K-of-N interaction)

---

This document describes the **group-first feature selection** methodology for the technical dashboard ML pipeline. The system is designed for financial time-series panel data with noisy labels.

---

## 1. Design Philosophy

### 1.1 Key Challenges in Financial Feature Selection

Financial prediction poses unique challenges for feature selection:

1. **Noisy Labels**: Targets derived from price trajectories have inherent noise
2. **Regime Changes**: Features that work in one regime may fail in another
3. **Overfitting Risk**: High feature dimensionality with limited effective sample size
4. **Time-Series Panel Data**: Non-IID samples due to multiple symbols per date

### 1.2 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Group-First Selection** | Entire hypothesis groups are the atomic unit of selection |
| **Over-Regularization** | Choose slightly sub-optimal but robust feature sets |
| **Baseline Demotion Mode** | Optional removal of baseline groups during backward elimination |

**Note on Stability**: Selection is intended to be stable and repeatable given fixed data and splits, but strict determinism is not a design goal.

### 1.3 Out of Scope

This document covers feature selection architecture only. The following are **out of scope**:

| Topic | Reference |
|-------|-----------|
| **Target generation, calibration, label semantics** | [TARGETS.md](TARGETS.md) |
| **Feature computation & lagging** | Upstream feature engineering pipeline |
| **Leakage controls** | Feature engineering (`single_stock.py`, `timeframe.py`) |
| **Sample weighting** | Not used in feature selection |
| **Production inference scheduling** | Deployment/inference documentation |

### 1.4 Group-First vs Singleton Selection

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

## 2. Balancing Precision and Recall

### 2.1 The Core Tradeoff

Feature selection for stock prediction faces a fundamental tension:

| Goal | Risk if Over-Optimized |
|------|------------------------|
| **High Precision** (few false positives) | Model becomes too conservative, misses opportunities |
| **High Recall** (few false negatives) | Model becomes noisy, generates many low-quality signals |
| **Cross-Sectional Coverage** | Model fixates on single sector or market cap |

### 2.2 Avoiding Single-Stock/Sector Concentration

The risk of feature selection converging on narrow patterns (e.g., "only works for tech mega-caps") is mitigated through:

**1. Cross-Sectional Features in Baseline Groups** ✅ *Implemented*

The curated baseline groups explicitly include features that measure relative position:
- `xsec_mom_20d_z` — Z-score of 20d return vs cross-section
- `rel_strength_sector` — Relative strength vs sector ETF
- `sector_breadth_pct_above_ma200` — Breadth indicators

These features measure *relative* performance, not absolute returns.

**2. Multi-Domain Feature Categories** ✅ *Implemented*

Baseline groups span multiple domains ensuring the model doesn't rely on any single pattern:

| Domain | Example Features | Purpose |
|--------|------------------|---------|
| **Momentum** | `rsi_14`, `w_macd_histogram` | Trend direction |
| **Price Position** | `pct_dist_ma_20_z`, `pos_in_20d_range` | Mean reversion |
| **Volatility Regime** | `vol_regime_ema10`, `rv_z_60` | Risk state |
| **Alpha/Relative** | `alpha_mom_spy_20_ema10`, `rel_strength_sector` | Stock vs market |
| **Breadth** | `sector_breadth_mcclellan_osc` | Market structure |
| **Macro** | `copper_gold_zscore`, `w_fred_bamlh0a0hym2_z60` | Economic regime |

**3. Time-Series CV with Expanding Window** ✅ *Implemented*

Expanding window ensures:
- Each fold includes all sectors in training data
- Features that only work for one sector show high fold variance
- Fold-level consistency check rejects sector-specific features

**4. Fold-Level Acceptance Criteria** ✅ *Implemented*

The SNR acceptance criteria require improvement across multiple folds, filtering out sector-specific improvements.

**5. Regime Metrics Infrastructure** ⚠️ *Infrastructure exists but not actively used*

The codebase includes regime-stratified evaluation (`compute_regime_metrics`). See Section 13 for potential future use.

### 2.3 Metrics and Calibration

**Selection Metric:**

AUC is the **sole metric** used for feature selection decisions. The epsilon/t-stat acceptance gates (add, swap, drop) evaluate changes in AUC only.

**Monitoring Metrics (Logged, Not Used for Selection):**

| Metric | Purpose | Role in Selection |
|--------|---------|-------------------|
| **AUC-ROC** | Discrimination ability | ✅ **Primary optimization metric** |
| **AUPR** | Precision-recall for imbalanced data | ℹ️ Logged only |
| **Brier Score** | Probability calibration indicator | ℹ️ Logged only |
| **Log Loss** | Likelihood calibration indicator | ℹ️ Logged only |
| **Fold Std Dev** | CV stability | ℹ️ Logged for all metrics |

**Why single-metric optimization?**
- Weighted composite metrics add hyperparameters (weights) that are difficult to tune
- AUC is threshold-invariant and well-suited for ranking problems
- AUPR, Brier, and log loss are tracked to monitor for degradation during selection

**Calibration:**

The feature selection pipeline does **not** perform probability calibration (Platt scaling, isotonic regression, temperature scaling). It only tracks Brier score and log loss as indicators. Probability calibration, if needed, is performed downstream in the training/inference pipeline.

---

## 3. Group Structure

### 3.1 Terminology

| Term | Definition |
|------|------------|
| **CORE_GROUPS** | Global baseline groups included for all models |
| **HEAD_GROUPS** | Per-model baseline groups (model-specific) |
| **CANDIDATE_GROUPS** | Groups available for forward selection |
| **INTERACTION_TEMPLATES** | Template-based group-to-group interactions |
| **Baseline Groups** | CORE_GROUPS ∪ HEAD_GROUPS[model_key] |
| **BASE_FEATURES** | flatten(Baseline Groups) — the feature-level view of baseline groups |

All group definitions live in `src/feature_selection/base_features.py`.

### 3.2 Group Categories

| Category | Description | Example Groups |
|----------|-------------|----------------|
| **CORE_GROUPS** | Global baseline (always included) | `alpha_momentum`, `volatility_regime`, `volatility_state`, `gap_dynamics` |
| **HEAD_GROUPS** | Per-model baseline | `price_action`, `trend_cross_sectional`, `relative_strength` |
| **CANDIDATE_GROUPS** | Forward selection pool | `atr_breakout`, `volume_liquidity`, `weekly_momentum` |
| **INTERACTION_TEMPLATES** | Group-to-group interactions | `momentum_x_vol_gate`, `gap_x_vol_state` |

### 3.3 Group Design Rules

Each group should follow these design conventions:
- **Size**: Target 3-12 features per group
- **Coherence**: One hypothesis per group (features should tell a coherent story)
- **Independence**: Minimal overlap between groups

### 3.4 Mandatory Group Splits

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

## 4. Group-First Selection Pipeline

### 4.1 Pipeline Overview

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
│          - Hill-climbing with caching                               │
│          - Optional tabu to avoid cycling                           │
│                                                                      │
│  STEP 4: Group Backward Elimination                                 │
│          - Remove groups if loss < epsilon_remove                   │
│          - Optional: allow_baseline_demotions to remove baseline    │
│                                                                      │
│  STEP 5: Template-Based Interaction Selection                       │
│          - Enumerate eligible templates (both parents selected)     │
│          - Greedy sequential evaluation with SNR acceptance         │
│          - See Section 6.5 for algorithm details                    │
│                                                                      │
│  RESULT: Final selected groups with per-group metrics               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Signal-to-Noise Acceptance Criteria

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

This measures the IMPROVEMENT signal, not absolute metric variability.

**Why Two Gates?**

Simple threshold checks (`delta >= epsilon`) are susceptible to selection bias:
- With 5 folds and many candidate groups, random variation can exceed epsilon
- This leads to optimistically biased CV metrics that don't generalize

The two-gate approach ensures:
1. **Epsilon Gate**: The improvement has practical significance
2. **t-stat Gate**: The improvement is reliable, not noise

**Default Acceptance Thresholds** (library defaults in `GroupSelectionConfig`):

| Move Type | epsilon | t_threshold | Rationale |
|-----------|---------|-------------|-----------|
| `add` | 0.0003 | 0.1 | Lenient: want exploration |
| `swap` | 0.0003 | 0.6 | Moderate: meaningful swap |
| `drop` | 0.0005 | 0.7 | Moderate: meaningful drop |
| `remove` | 0.0005 | -0.25 | Accept unless confident harm (t < -0.25) |
| `add_interaction` | 0.0008 | 0.5 | Higher epsilon for interactions |

**CLI Override Coverage:**

The CLI script `run_group_selection.py` exposes **only** `--epsilon-add` and `--epsilon-swap` flags. These override the corresponding library defaults for `add` and `swap` moves.

**Not CLI-overridable** (remain at library defaults):
- All t_threshold values (t_add, t_swap, t_drop, t_remove, t_add_interaction)
- epsilon_drop, epsilon_remove, epsilon_add_interaction

Run `python run_group_selection.py --help` for current CLI defaults.

### 4.3 Holdout Evaluation

To detect selection bias, the pipeline supports temporal holdout evaluation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOLDOUT EVALUATION                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Full Dataset:  |──────────────────────────────────────────────|    │
│                                                                      │
│  Train:         |─────────────────────────────────────────|         │
│                 ^                                         ^          │
│                 │  CV folds for feature selection         │          │
│                                                                      │
│  Holdout:                                               |────|       │
│                                                          ^    ^      │
│                                                          │    │      │
│                              Unbiased evaluation ────────┘    │      │
│                                                               │      │
└─────────────────────────────────────────────────────────────────────┘
```

**Interpreting Results:**

| CV-Holdout Gap | Interpretation |
|----------------|----------------|
| < 0.05 | ✅ OK - Selection appears robust |
| 0.05 - 0.10 | ⚠️ NOTICE - Some selection bias present |
| > 0.10 | ❌ WARNING - Significant selection bias |

### 4.4 Over-Regularization Strategy

We intentionally choose slightly sub-optimal but robust group sets:

**Why over-regularize?**
- Groups that barely improve CV may not generalize
- Financial regimes change - robust groups preferred
- Fewer groups = more interpretable model

### 4.5 Outer CV for Stability (Default)

The pipeline uses **outer cross-validation** by default to reduce feature-selection bias. This runs feature selection multiple times on different temporal splits, then keeps only groups that are consistently selected.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTER CV WITH STABILITY AGGREGATION              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Outer Fold 1:                                                       │
│    Train: |──────────────────────────|                               │
│    Test:                              |────|                         │
│    → Run full group selection → Selected groups: {A, B, C, D}        │
│                                                                      │
│  Outer Fold 2:                                                       │
│    Train: |──────────────────────────────────|                       │
│    Test:                                      |────|                 │
│    → Run full group selection → Selected groups: {A, B, D, E}        │
│                                                                      │
│  Outer Fold 3:                                                       │
│    Train: |──────────────────────────────────────────|               │
│    Test:                                              |────|         │
│    → Run full group selection → Selected groups: {A, B, C}           │
│                                                                      │
│  Final Holdout:                                                |──|  │
│                                                                      │
│  Stability Aggregation:                                              │
│    - Core (3/3 folds): {A, B}                                        │
│    - Stable (≥2/3 folds): {A, B, C, D}                               │
│    - Choose based on holdout AUC                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Stability Thresholds:**

| Set | Threshold | Interpretation |
|-----|-----------|----------------|
| `core` | Selected in ALL folds | Most stable, may be conservative |
| `stable` | Selected in >= 2/3 folds | Good balance of stability and coverage |

**Selection Logic:**
1. Run group selection independently on each outer fold
2. Compute group selection frequency across folds
3. Build two candidate sets (core and stable)
4. Evaluate both on final holdout (never used during selection)
5. Choose stable only if holdout AUC improves sufficiently

---

## 5. Group Definitions

### 5.1 CORE_GROUPS (Global Baseline)

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
    "volatility_regime": [...],     # VIX percentile, zscore, vol regime
    "volume_shock": [...],          # Volume shocks, divergences
    "microstructure_position": [...], # VWAP distance, overnight ratio
    "volatility_state": [...],      # BB width, squeeze intensity, RV zscore
    "gap_dynamics": [...],          # Gap/ATR ratio, overnight return
}
```

See `src/feature_selection/base_features.py` for the current feature list.

### 5.2 HEAD_GROUPS (Per-Model Baseline)

HEAD_GROUPS are model-specific baseline features. See `base_features.py` for full definitions.

### 5.3 CANDIDATE_GROUPS

Groups available for forward selection from EXPANSION_CANDIDATES.

---

## 6. Template-Based Interaction System

The **current** interaction system uses template-based group-to-group interactions rather than ad-hoc feature×feature combinations. Each template represents a coherent economic hypothesis.

### 6.1 Template Structure

```python
INTERACTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "momentum_x_vol_gate": {
        "parents": ("momentum_quality", "volatility_state"),
        "type": "gate",  # or "signed_gate" or "product"
        "base_features": ["rsi_14", "adx_14", "chop_14"],
        "gate_features": ["squeeze_intensity_20", "rv_z_60"],
        "description": "Momentum indicators gated by volatility state",
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

### 6.2 Interaction Types

| Type | Formula | When to Use |
|------|---------|-------------|
| **gate** | `base × I(gate > median)` | Binary regime conditioning |
| **signed_gate** | `base × sign(gate)` | Direction matters (bullish/bearish) |
| **product** | `base × gate` | Multiplicative confirmation |

### 6.3 Eligibility Rules

A template is **eligible** for selection only when **both parent groups** are already in the selected feature set. This ensures:
1. No orphan interactions (interactions without base signals)
2. Interactions build on established relationships
3. Selection remains interpretable

### 6.4 Naming Convention

Generated feature names follow a consistent naming scheme:
```
ix__{template_name}__{base_feat}__gated__{gate_feat}   # for gate/signed_gate
ix__{template_name}__{feat_a}__x__{feat_b}             # for product
```

### 6.5 Interaction Search Algorithm

This section describes how interaction templates are evaluated and selected during Step 5 of the pipeline. The implementation lives in `template_interaction_selection()` in `group_selection.py`.

**Inputs:**

| Input | Description |
|-------|-------------|
| `selected_groups` | Groups selected from Steps 1-4 (dict of group_name → feature list) |
| `X` | Feature DataFrame (may be modified to add computed interaction features) |
| `evaluator` | Configured `SubsetEvaluator` for CV evaluation |
| `config` | `GroupSelectionConfig` with thresholds and limits |
| `cache` | Optional evaluation cache for reusing CV results |

**Algorithm:**

```
1. ELIGIBILITY FILTERING
   - For each template in INTERACTION_TEMPLATES:
     - Check if BOTH parent groups are in selected_groups
     - If yes, add to eligible_templates list
   - If no templates are eligible, return immediately (no interactions added)

2. BASELINE EVALUATION
   - Evaluate current selected features to establish baseline metric
   - Use cache if available to avoid recomputation

3. GREEDY SEQUENTIAL EVALUATION
   For each template_name in eligible_templates (in definition order):

     a. EARLY STOPPING
        - If max_interaction_groups reached, stop
        - If template already in selected, skip

     b. FEATURE GENERATION
        - Generate interaction feature names from template definition
        - Compute interaction features dynamically if not already in X
          (uses compute_template_interactions() from interaction_production.py)
        - Skip template if no features could be computed

     c. EVALUATOR UPDATE
        - Add newly computed features to evaluator's internal state
        - This allows subsequent CV evaluation to use them

     d. CANDIDATE EVALUATION
        - Create test feature set: current_features + template_features
        - Evaluate via CV (uses cache if available)

     e. SNR ACCEPTANCE
        - Apply two-gate acceptance rule (see Section 4.2):
          - epsilon_add_interaction (default: 0.0008)
          - t_add_interaction (default: 0.5)
        - Record result (accepted or rejected with reason)

     f. STATE UPDATE (if accepted)
        - Add template to selected_groups
        - Update baseline metric and result for next iteration
        - Increment interaction counter

4. RETURN
   - Return updated selected_groups with any added interaction templates
   - Return list of GroupResult objects (one per evaluated template)
   - Return final metric
```

**Key Characteristics:**

| Aspect | Behavior |
|--------|----------|
| **Iteration Order** | Templates are evaluated in their definition order in `INTERACTION_TEMPLATES` |
| **Atomicity** | Each template is evaluated as a single atomic unit; partial acceptance is not possible |
| **Greedy Strategy** | Accepted templates immediately become part of the baseline for subsequent evaluations |
| **No Batch Comparison** | Templates are not compared against each other; first to pass acceptance wins |
| **Dynamic Computation** | Interaction features are computed on-demand if not present in X |
| **Max Limit** | Controlled by `config.max_interaction_groups` (see `GroupSelectionConfig`) |

**K-of-N Interaction:**

Interaction templates use **raw template definitions** (`base_features` and `gate_features` lists), not K-of-N resolved features. K-of-N selection applies to base groups during Steps 1-4, but the interaction feature generation uses the template's explicit feature lists.

**Caching Behavior:**

- If `cache` is provided and enabled, evaluation results are stored keyed by `frozenset(features)`
- Cache lookups occur before CV evaluation to avoid redundant computation
- New evaluations are inserted into the cache after computation
- Cache is shared across all pipeline stages when enabled

**Failure Modes:**

| Condition | Behavior |
|-----------|----------|
| No eligible templates | Function returns immediately with no interaction groups added |
| Template already in selected | Template is skipped (no re-evaluation) |
| No features could be computed | Template is skipped with verbose log message |
| SNR acceptance fails | Template is rejected; reason recorded in `GroupResult` |
| `max_interaction_groups` reached | Remaining templates are not evaluated |

**Stability Note:**

The algorithm is intended to produce consistent results given identical inputs and configuration, but strict determinism is not guaranteed. Template iteration follows definition order, which provides implicit stability.

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
│      -> Low n_jobs, high threads per model                          │
│                                                                      │
│  IF single model underutilizes CPU (small dataset):                 │
│      -> Train models in parallel                                    │
│      -> Higher n_jobs, fewer threads per model                      │
│                                                                      │
│  RULE: n_jobs * model_threads ~ available CPU cores                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Dataset Size Heuristics (Typical Examples)

The following are **example guidelines** — actual optimal settings depend on hardware and data characteristics. Use CLI flags `--n-jobs` and `--model-threads` to adjust.

| Dataset Size | Single Model Utilization | Example Strategy |
|--------------|--------------------------|------------------|
| Small | Low | Higher n_jobs, lower threads/model |
| Medium | Medium | Balanced |
| Large | High | Lower n_jobs, more threads/model |
| Very Large | Full | Sequential models, max threads each |

### 7.3 Process-Based Parallelism (Loky)

LightGBM training is CPU-bound. Python's GIL prevents threads from running in parallel. The pipeline uses loky (process-based) parallelism by default.

---

## 8. Cross-Validation with Purging

### 8.1 CV Hierarchy

The pipeline supports multiple levels of cross-validation:

| Level | Purpose |
|-------|---------|
| **Inner CV** | Evaluate feature subsets during selection |
| **Outer CV** | Stability check (run selection N times) |
| **Final Holdout** | Unbiased evaluation (never seen during selection) |

Run `python run_group_selection.py --help` for current default fold counts and fractions.

### 8.2 Time-Series CV Configuration

```python
@dataclass
class CVConfig:
    n_splits: int = ...           # See config.py for default
    scheme: CVScheme = CVScheme.EXPANDING
    gap: int = ...                # Embargo between train/test (in dates)
    purge_window: int = ...       # See config.py for default
    min_train_samples: int = ...  # See config.py for default
```

See `src/feature_selection/config.py` for authoritative defaults.

**IMPORTANT**: The `gap` parameter is measured in **unique dates**, not rows/samples. For panel data with multiple symbols per date, the CV splitter:
1. Groups all rows by their date
2. Splits DATES into train/test periods
3. Applies embargo gap in terms of dates
4. Maps back to row indices

### 8.3 Purging and Embargo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TIME-SERIES CV WITH PURGING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Train Period          Gap (Embargo)        Test Period             │
│  |------------------|<-- N dates --->|--------------|               │
│                                                                      │
│  Gap/embargo reduces temporal leakage between train and test        │
│  periods. See TARGETS.md for label-window details.                  │
│                                                                      │
│  For panel data (N symbols × T dates):                              │
│  - Dates are the unit of splitting, not rows                        │
│  - All symbols on a given date stay together                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Model Configuration

### 9.1 LightGBM Settings

The selection pipeline uses LightGBM for evaluation. Hyperparameters are configured via `ModelConfig`:

```python
@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.LIGHTGBM
    task_type: TaskType = TaskType.CLASSIFICATION
    params: Dict[str, Any] = ...  # See config.py for defaults
    num_threads: int = ...        # See config.py / CLI --model-threads
    early_stopping_rounds: int = ...
    num_boost_round: int = ...
```

See `src/feature_selection/config.py` for authoritative defaults. The `run_group_selection.py` script may override some parameters via CLI flags.

### 9.2 Regularization for Robustness

The model intentionally uses moderate regularization:
- **L1 penalty** (`reg_alpha`): Shrinks weak features to zero
- **L2 penalty** (`reg_lambda`): Prevents large coefficients
- **Min samples per leaf** (`min_child_samples`): Prevents splits on small groups
- **Max depth**: Limits tree complexity

---

## 10. Running Group Selection

### 10.1 Basic Usage

```bash
# Run group selection for a single model
python run_group_selection.py --model long_normal

# Run for all 4 models
python run_group_selection.py --model all

# Allow baseline group demotions during backward elimination
python run_group_selection.py --model long_normal --allow-demotions

# Disable outer CV for faster iteration
python run_group_selection.py --model long_normal --disable-outer-cv
```

### 10.2 CLI Options

**`run_group_selection.py` is the authoritative source for defaults. Run `--help` for current values.**

Key options:

| Parameter | Description |
|-----------|-------------|
| `--model` | Model: `long_normal`, `long_parabolic`, `short_normal`, `short_parabolic`, or `all` |
| `--epsilon-add` | Minimum improvement to add a group (overrides library default) |
| `--epsilon-swap` | Minimum improvement for swaps (overrides library default) |
| `--allow-demotions` | Allow dropping baseline groups |
| `--max-groups` | Maximum total groups to select |
| `--max-symbols` | Maximum symbols to use |
| `--balanced` | Use class weights (scale_pos_weight) |
| `--n-folds` | Number of inner CV folds |
| `--holdout-pct` | Fraction of dates for final holdout |
| `--disable-outer-cv` | Disable outer CV (use single-run selection) |
| `--n-outer-folds` | Number of outer CV folds |
| `--disable-k-of-n` | Disable K-of-N selection (use all features per group) |
| `--group-k` | Default K for K-of-N selection within groups |
| `--n-jobs` | Number of parallel jobs for CV |
| `--model-threads` | Threads per LightGBM model |
| `--output-dir` | Output directory for results |
| `--quiet` | Reduce verbosity |

---

## 11. Output Files

### 11.1 Feature Selection Artifacts

```
artifacts/group_selection/
├── group_selection_{model}.json   # Complete results per model
├── selected_features_{model}.txt  # Feature list (one per line)
├── holdout_evaluation.json        # Holdout metrics (if enabled)
└── outer_cv_results.json          # Outer CV details (if enabled)

artifacts/{model}/
└── features.json                  # Feature registry for training
```

### 11.2 Feature Registry

The registry (`artifacts/{model}/features.json`) provides:
- **resolved_features**: An explicitly ordered list of features for training
- **feature_signature**: SHA256 hash for reproducibility verification
- **selection_metadata**: Metrics from feature selection (CV AUC, holdout AUC)

See [MODEL_FEATURIZATION.md](MODEL_FEATURIZATION.md) for full registry documentation.

---

## 12. Key Files Reference

| File | Purpose |
|------|---------|
| `src/feature_selection/base_features.py` | CORE_GROUPS, HEAD_GROUPS, CANDIDATE_GROUPS, INTERACTION_TEMPLATES |
| `src/feature_selection/group_selection.py` | Group-first selection algorithms, `_snr_acceptance()`, `template_interaction_selection()` |
| `src/feature_selection/config.py` | Configuration dataclasses (ModelConfig, CVConfig, etc.) |
| `src/feature_selection/evaluation.py` | SubsetEvaluator for CV evaluation |
| `src/feature_selection/cv.py` | Time-series CV with purging |
| `src/feature_selection/interactions.py` | DOMAIN_PATTERNS, InteractionType |
| `src/features/interaction_production.py` | `compute_template_interactions()` for dynamic feature generation |
| `run_group_selection.py` | Group-first selection entry point |

**Legacy Files (Singleton Selection):**

| File | Status |
|------|--------|
| `src/feature_selection/pipeline.py` | Deprecated - Loose-Then-Tight pipeline |
| `src/feature_selection/algorithms.py` | Deprecated - Singleton forward/backward/swap |
| `run_feature_selection.py` | Deprecated - Singleton selection entry point |

---

## 13. Future Enhancements

### 13.1 Sector-Stratified Evaluation

The infrastructure for regime-stratified metrics exists in `compute_regime_metrics()`. To enable sector-stratified AUC:

```python
# In run_group_selection.py - load sector labels
sector_map = dict(zip(universe['Symbol'], universe['Sector']))
merged['sector'] = merged['symbol'].map(sector_map)
sector = merged['sector'].copy()

# Pass to evaluator
evaluator = SubsetEvaluator(..., regime=sector)
```

### 13.2 Adaptive Parallelization

Profile single model training time and adjust parallelization dynamically based on dataset size.

### 13.3 Feature Stability Scoring

Evaluate feature importance stability across different market regimes (bull/bear/sideways).

---

## 14. Troubleshooting

### 14.1 Common Issues

**Pipeline stuck on one stage:**
- Check for infinite loop in backward elimination
- Verify epsilon values are reasonable

**Memory errors:**
- Reduce n_jobs (fewer parallel workers)
- Use smaller batch sizes for interactions

**Low AUC scores:**
- Check for data leakage in CV
- Review feature NaN rates
- Verify embargo gap is appropriate (see TARGETS.md for label-window details)

**Model concentrates on single sector:**
- Add more cross-sectional features to baseline groups
- Check that sector-relative alpha features are included
- Verify sector distribution in training data

### 14.2 Debugging

```python
# Check outer CV results
import json
with open('artifacts/group_selection/outer_cv_results.json') as f:
    results = json.load(f)
print(results['long_normal']['stability']['group_frequency'])
```

---

## 15. Multi-Model Feature Selection

### 15.1 Overview

The multi-model feature selection system runs the group selection pipeline independently for each of the 4 model targets while ensuring:

- **Same CV splits**: All models use identical time-series CV splits for comparability
- **Same feature universe**: All models start from the same candidate features
- **Single data load**: Features are computed once and reused

**Model Keys:**

| Model Key | Interpretation |
|-----------|----------------|
| `LONG_NORMAL` | Standard long momentum |
| `LONG_PARABOLIC` | Trend persistence / parabolic moves |
| `SHORT_NORMAL` | Breakdown / fragility setups |
| `SHORT_PARABOLIC` | Panic / regime shift |

### 15.2 Running Multi-Model Selection

```bash
# Run all 4 models
python run_group_selection.py --model all

# Run specific model
python run_group_selection.py --model long_normal
```

### 15.3 CORE and HEAD Features

**CORE_FEATURES**: Features selected by ALL 4 models (intersection). These represent the most universally predictive signals.

**HEAD_FEATURES[model_key]**: Features unique to each model (selected by that model but not in CORE). These capture model-specific patterns.

### 15.4 Best Practices

1. **Run all 4 models together**: Ensures identical CV splits and feature universe
2. **Check overlap matrix**: Very high overlap (>0.9) suggests models may be redundant
3. **Monitor CORE size**: A small CORE indicates models target very different signals
4. **Validate against registry**: Ensure selected features exist in `base_features.py`
