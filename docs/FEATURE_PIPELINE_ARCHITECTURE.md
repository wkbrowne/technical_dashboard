# Feature Pipeline Architecture

Technical architecture of the feature computation pipeline, including data flow, parallelism strategies, and data conventions.

For the high-level ML pipeline (feature selection, hyperparameter tuning, model training), see [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 1. Pipeline Overview

The pipeline executes stages in a fixed order, with checkpointing for resumability:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LOADING                                        │
│  cache/stock_data_universe.parquet  →  Daily OHLCV                              │
│  cache/stock_data_etf.parquet       →  ETF OHLCV (long format)                  │
│  cache/fred_data.parquet            →  FRED macro data                          │
│  cache/US universe_*.csv            →  Symbol list with sectors                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: SINGLE-STOCK FEATURES (parallel, batched by symbol)                   │
│  - OHLC adjustment to match adjclose                                            │
│  - Trend: RSI, MACD, ADX, trend scores                                          │
│  - Volatility: ATR, realized vol, Bollinger/Keltner squeeze                     │
│  - Volume: volume ratios, OBV                                                   │
│  - Distance: percent distance to MAs, position in range                         │
│  - Range/Gap: overnight returns, gap behavior                                   │
│  Checkpoint: 01_single_stock                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CROSS-SECTIONAL FEATURES (parallel)                                   │
│  - Factor regression: beta_market, beta_qqq, beta_bestmatch, alpha              │
│  - Sector-relative: alpha vs sector ETF, relative strength                      │
│  - Cross-sectional momentum: z-score ranks                                      │
│  Checkpoint: 02_cross_sectional                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: DAILY SPREAD FEATURES (global, broadcast to all symbols)              │
│  - Market spreads: QQQ, SPY, QQQ-SPY, RSP-SPY cumulative returns                │
│  - Per-symbol spreads: bestmatch-SPY, bestmatch_ew-RSP                          │
│  Checkpoint: 03_spread_features                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: WEEKLY FEATURE COMPUTATION (parallel)                                 │
│  - Resample daily to weekly (W-FRI)                                             │
│  - Compute w_* versions of single-stock features                                │
│  - Forward-fill back to daily index                                             │
│  Checkpoint: 04_weekly_features                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: WEEKLY CROSS-SECTIONAL FEATURES                                       │
│  - Weekly alpha, beta (w_alpha_mom_*, w_beta_*)                                 │
│  - Weekly sector breadth proxy (w_sector_breadth_*)                             │
│  - Weekly FRED macro features                                                   │
│  Checkpoint: 05_weekly_cs                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: WEEKLY SPREAD FEATURES                                                │
│  - Weekly versions of market spreads (w_qqq_spy_*, w_rsp_spy_*)                 │
│  Checkpoint: 06_weekly_spread                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: NaN INTERPOLATION (parallel)                                          │
│  - Internal gap interpolation (limit_area='inside')                             │
│  - No leading/trailing NaN filling                                              │
│  Checkpoint: 07_interpolated                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 8: INTERACTION FEATURES                                                  │
│  - Registered interactions from HEAD_FEATURES/CORE_FEATURES                     │
│  - Computed from base features post-interpolation                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 9: TRIPLE BARRIER TARGETS (parallel)                                     │
│  - Multi-model targets: LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARA   │
│  - Sample weights: overlap + class balance                                      │
│  Checkpoint: 08_targets                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 10: FINAL ASSEMBLY & OUTPUT                                              │
│  - Convert dict-of-DataFrames to long format                                    │
│  - Filter: features_complete.parquet (all features)                             │
│  - Filter: features_filtered.parquet (curated ML-ready set)                     │
│  - targets_triple_barrier.parquet                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Conventions

### 2.1 Column Naming

**All column names MUST be lowercase.** This prevents bugs from case mismatches between data sources.

```python
# IMMEDIATELY after loading any DataFrame:
df.columns = [c.lower() for c in df.columns]
```

Sources that require normalization:
- Yahoo Finance: `AdjClose`, `Close`, `High` → `adjclose`, `close`, `high`
- ETF cache after pivot: Mixed case → lowercase
- Feature columns: Always snake_case (`rsi_14`, `atr_percent`, `w_beta_spy`)

### 2.2 File Format Standards

| File | Format | Key Columns |
|------|--------|-------------|
| `stock_data_universe.parquet` | Wide (symbols as columns) | DatetimeIndex, per-metric |
| `stock_data_etf.parquet` | Long | `date`, `symbol`, `value`, `metric` |
| `features_*.parquet` | Long | `symbol`, `date`, + feature columns |
| `targets_triple_barrier.parquet` | Long | `symbol`, `t0`, `hit`, weights |

**DataFrame Requirements:**
- Dtypes: `float32` for features (memory efficiency), `category` for symbols
- Sorting: Always `['symbol', 'date']` before saving

### 2.3 NaN Handling

Never drop NaNs. Use interpolation only for internal gaps:

```python
df[col].interpolate(method='linear', limit_area='inside')
```

### 2.4 Row Alignment with Panel Data

When data has multiple rows per date (multiple symbols), **never** use `.loc[date_index]` for alignment - it causes a silent cross-join explosion.

```python
# WRONG: Causes cross-join when dates are not unique
X_aligned = X.loc[y.index]  # 25k rows → 500k rows!

# CORRECT: Use unique row identifier
df['_row_id'] = range(len(df))
X_aligned = X[X['_row_id'].isin(valid_row_ids)]
```

---

## 3. Pipeline Stages

### 3.1 Data Loading

**Entry Point:** `src/data/loader.py`

**Untradeable Asset Filtering:**

The pipeline automatically filters problematic securities at load time:

| Category | Detection | Examples |
|----------|-----------|----------|
| SPACs | Description contains "Acquisition" | PSTH, CCIV |
| ADRs | Description contains "ADR" | BABA, TSM |
| Warrants | Symbol `.W`, `/W`, `-W`, `WS` | FOOW, FOO.WS |
| Preferred | Symbol `.PR`, `-P` | BAC.PR.A |

**Suspicious Price Filter:**

| Rule | Threshold | Reason |
|------|-----------|--------|
| Max price | > $50,000 | Data error or unadjusted split |
| Min price | < $0.01 | Penny stock artifacts |
| Price range ratio | max/min > 10,000x | Split adjustment issues |

### 3.2 OHLC Adjustment

Before feature computation, OHLC prices are adjusted to match adjclose:

```python
df_adjusted = adjust_ohlc_to_adjclose(df)
```

This ensures technical indicators use split/dividend-adjusted prices.

### 3.3 Single-Stock Features

**Module:** `src/features/single_stock.py`

| Category | Module | Key Features |
|----------|--------|--------------|
| Trend | `trend.py` | `rsi_14`, `macd_histogram`, `trend_score_*`, `chop_14`, `adx_14` |
| Volatility | `volatility.py` | `atr_percent`, `vol_regime_*`, `bb_width_*`, `squeeze_*` |
| Vol Acceleration | `volatility.py` | `rv_delta_*`, `rv_accel_*`, `rv_impulse_*` |
| Volume | `volume.py` | `volshock_ema`, `obv_z_60` |
| Distance | `distance.py` | `pct_dist_ma_*_z`, `pos_in_*_range` |
| Range/Gap | `range_breakout.py` | `overnight_ret`, `gap_atr_ratio*`, `gap_fill_frac` |

**Trend Quality / Chop Filter:**

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `chop_14` | Choppiness Index | > 61.8 choppy, < 38.2 trending |
| `adx_14` | Average Directional Index | < 20 weak trend, > 40 strong |

**Volatility Squeeze:**

```
squeeze_on = (BB_upper < KC_upper) AND (BB_lower > KC_lower)
```

When Bollinger Bands contract inside Keltner Channels, volatility compression often precedes breakout.

**Volatility Acceleration:**

Features that capture directional change and acceleration in volatility, complementing level-based features:

| Feature | Description | Signal |
|---------|-------------|--------|
| `rv_delta_10_60` | rv_10 - rv_60 (signed difference) | Vol premium direction |
| `rv_delta_10_60_z` | Z-scored (60d) delta | Normalized expansion signal |
| `rv_accel_20` | 5-day change in rv_20, z-scored | Vol speeding up/down |
| `rv_accel_60` | 10-day change in rv_60, z-scored | Longer-term acceleration |
| `rv_impulse_5d_z` | 5-day pct_change of rv_10, z-scored | Sudden vol spikes |

Weekly versions: `w_rv_delta_10_60_z`, `w_rv_accel_20`

### 3.4 Cross-Sectional Features

**Modules:** `src/features/cross_sectional.py`, `src/features/factor_regression.py`

**Joint 4-Factor Model:**

```
R_stock = α + β_market × R_SPY
            + β_qqq × (R_QQQ - R_SPY)
            + β_bestmatch × (R_bestmatch - R_SPY)
            + β_breadth × (R_RSP - R_SPY)
            + ε
```

| Factor | Interpretation |
|--------|----------------|
| `beta_market` | Broad market exposure |
| `beta_qqq` | Growth/tech premium over market |
| `beta_bestmatch` | Sector premium (R²-based ETF match) |
| `beta_breadth` | Equal-weight vs cap-weight spread |

**Best-Match ETF Selection:**

Each stock is matched to the ETF with highest R² from univariate regression:
- Cap-weighted: XLK, SMH, XBI, etc. (25 candidates)
- Equal-weight: RSPT, RSPF, RSPH, etc. (10 candidates)

### 3.5 Spread Features

**Module:** `src/features/spread_features.py`

**Global Spreads (same for all symbols):**

| Spread | Formula | Signal |
|--------|---------|--------|
| `qqq_spy` | QQQ - SPY | Growth premium |
| `rsp_spy` | RSP - SPY | Breadth spread (equal vs cap weight) |

**Per-Symbol Spreads:**

| Spread | Formula | Description |
|--------|---------|-------------|
| `bestmatch_spy_*` | bestmatch - SPY | Sector vs market premium |
| `bestmatch_ew_rsp_*` | bestmatch_ew - RSP | EW sector vs EW market |

**Metrics per spread:** `cumret_{20,60,120}`, `zscore_60`, `slope_{20,60}`

### 3.6 Weekly Features

Weekly features are computed from resampled data and forward-filled to daily:

**Resampling Rules (W-FRI):**
- open: first of week
- high: max of week
- low: min of week
- close/adjclose: last (Friday)
- volume: sum

**Leakage Prevention:**

```python
# Friday's weekly feature applies to NEXT week's daily rows
merged = pd.merge_asof(daily, weekly, on='date', by='symbol', direction='backward')
```

### 3.7 Sector ETF Breadth Proxy

**Module:** `src/features/sector_breadth.py`

Uses 11 Select Sector SPDR ETFs (XLK, XLF, XLE, etc.) as survivorship-bias-free breadth proxy.

| Feature | Description |
|---------|-------------|
| `sector_breadth_net_adv` | Net advancing sectors |
| `sector_breadth_pct_above_ma50` | Pct above 50-day MA |
| `sector_breadth_mcclellan_osc` | EMA(19) - EMA(39) of net adv |
| `sector_breadth_mcclellan_sum` | Cumulative McClellan |

**Breadth Motion:**

Features that capture momentum and slope of breadth indicators, complementing static levels:

| Feature | Description | Signal |
|---------|-------------|--------|
| `sector_breadth_ad_chg_10d` | 10-day change in AD line | Breadth momentum direction |
| `sector_breadth_ad_slope_20d` | 20-day linear regression slope | Breadth trend strength |
| `sector_breadth_mcclellan_chg_5d` | 5-day change in McClellan | McClellan momentum |
| `sector_breadth_mcclellan_slope_10d` | 10-day slope of McClellan | McClellan trend |
| `sector_breadth_pct_ma50_chg_10d` | 10-day change in % above MA50 | Participation trend |

Weekly versions: `w_sector_breadth_ad_slope_8w`, `w_sector_breadth_mcclellan_chg_2w`

### 3.8 FRED Macro Features

**Module:** `src/data/fred.py`

FRED data is NOT resampled - forward-filled to daily with publication lag applied.

| Series | Frequency | `pub_lag_days` | Reason |
|--------|-----------|----------------|--------|
| DGS10, DGS2 | Daily | 1 | Available next morning |
| BAMLH0A0HYM2 | Daily | 1 | ICE BofA 1-day lag |
| ICSA | Weekly | 5 | Week ends Sat, released Thu |
| CCSA | Weekly | 12 | Extra week lag vs ICSA |

**Features per series:** `level`, `chg5d`, `chg20d`, `z60`, `pct252`

---

## 4. Parallelism Architecture

**Core Principle:** Only send data needed for each computation to workers.

### 4.1 Symbol Batching

Symbols are batched to reduce IPC overhead:

```python
BATCH_SIZE = 100  # ~100 symbols per worker

batch_results = Parallel(n_jobs=-1, backend='loky')(
    delayed(_feature_worker_batch)(chunk)
    for chunk in symbol_chunks
)
```

| Dataset Size | Strategy |
|--------------|----------|
| < 100 symbols | 1 symbol/task |
| 100-500 symbols | ~25/chunk |
| > 500 symbols | ~100/chunk |

### 4.2 Memory Management

**Avoid O(N²) patterns:**

```python
# BAD: O(N²) - creates N full DataFrame copies
symbol_groups = [(sym, df[df['symbol'] == sym]) for sym in symbols]

# GOOD: O(N) - single-pass groupby
symbol_data_dict = {sym: group.copy() for sym, group in df.groupby('symbol')}
```

**DataFrame construction:**

```python
# BAD: Fragmentation from loop insertion
for col in features:
    df[col] = values[col]

# GOOD: Single concat
new_df = pd.DataFrame(features, index=df.index)
result = pd.concat([df, new_df], axis=1)
```

### 4.3 Correlation Matrices

**Never use pandas `.corr()` for large matrices** - use numpy:

```python
# GOOD: Numpy (orders of magnitude faster)
X = df[feature_cols].values
X_clean = np.nan_to_num(X, nan=0.0)
X_centered = X_clean - X_clean.mean(axis=0)
X_normed = X_centered / (np.std(X_centered, axis=0) + 1e-10)
corr_matrix = (X_normed.T @ X_normed) / X_normed.shape[0]
```

---

## 5. Checkpointing

**Module:** `src/pipelines/checkpoint.py`

### 5.1 Checkpoint Stages

| Stage | Checkpoint | ~Memory (3000 sym) |
|-------|------------|-------------------|
| Single-Stock | `01_single_stock` | ~4 GB |
| Cross-Sectional | `02_cross_sectional` | ~5 GB |
| Daily Spread | `03_spread_features` | ~5.5 GB |
| Weekly Features | `04_weekly_features` | ~6 GB |
| Weekly CS | `05_weekly_cs` | ~6.5 GB |
| Weekly Spread | `06_weekly_spread` | ~6.5 GB |
| NaN Interpolation | `07_interpolated` | ~7 GB |
| Targets | `08_targets` | ~7 GB |

### 5.2 Usage

```bash
# Enable checkpointing
python -m src.cli.compute --checkpoint-dir artifacts/checkpoints

# Resume from checkpoint
python -m src.cli.compute --resume-from 04_weekly_features

# List available checkpoints
python -m src.cli.compute --list-checkpoints
```

---

## 6. Target Generation

**Module:** `src/features/target_generation.py`

### 6.1 Triple Barrier Labeling

| Barrier | Condition | Label |
|---------|-----------|-------|
| Upper | Price ≥ Entry + ATR×mult | `hit=1` (profit) |
| Lower | Price ≤ Entry - ATR×mult | `hit=0` (stop) |
| Time | Max days elapsed | `hit=-1` (expired) |

**4-Model System:**

| Model | up_mult | dn_mult | Style |
|-------|---------|---------|-------|
| LONG_NORMAL | 1.5 | 1.5 | Standard |
| LONG_PARABOLIC | 2.5 | 1.5 | Extended |
| SHORT_NORMAL | 1.5 | 2.0 | Breakdown |
| SHORT_PARABOLIC | 1.5 | 2.5 | Panic |

### 6.2 Sample Weights

| Weight | Formula | Purpose |
|--------|---------|---------|
| `weight_overlap` | 1 / (n_overlapping + 0.5) | Reduce correlated samples |
| `weight_class_balance` | n / (n_classes × class_count) | Handle imbalanced classes |
| `weight_final` | overlap × class_balance (clipped) | Combined training weight |

---

## 7. Output Files

### 7.1 Features

| File | Contents | Use Case |
|------|----------|----------|
| `features_complete.parquet` | All computed features | Debugging, exploration |
| `features_filtered.parquet` | Curated ML-ready set | Model training, production |

### 7.2 Feature Classification

**Module:** `src/feature_selection/base_features.py`

| Category | Purpose |
|----------|---------|
| `BASE_FEATURES` | Core features for production models |
| `EXPANSION_CANDIDATES` | Features for selection experiments |
| `RETIRED_FEATURES` | Tested but consistently not selected |
| `INTERMEDIATE_FEATURES` | Required to compute kept features |
| `META_COLUMNS` | Always kept: `symbol`, `date`, `ret` |

**Filtering:**

```python
from src.feature_selection.base_features import filter_output_columns, validate_features

# Filter to curated set
df_filtered = filter_output_columns(df, keep_all=False)

# Validate presence of required features
result = validate_features(df)
```

### 7.3 CLI Options

```bash
# Normal run
python -m src.cli.compute --timeframes D,W

# Exclude retired features (saves disk)
python -m src.cli.compute --exclude-retired

# Fast iteration
python -m src.cli.compute --timeframes D --max-stocks 50
```

---

## 8. Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Features all NaN | Column case mismatch | Lowercase immediately after loading |
| Row count explosion | `.loc[]` on non-unique dates | Use `_row_id` with `.isin()` |
| Memory errors | Large universe | Use `--max-stocks 100 --timeframes D` |
| Slow correlations | Pandas `.corr()` | Use numpy for matrices |
| Weekly leakage | Same-week features | Use `direction='backward'` in merge |
| DataFrame fragmented | Loop column insertion | Use `pd.concat()` |

---

## 9. Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - High-level ML pipeline
- [FEATURE_SELECTION.md](FEATURE_SELECTION.md) - Feature selection methodology
- [TARGETS.md](TARGETS.md) - Triple barrier target details
- [MODEL_FEATURIZATION.md](MODEL_FEATURIZATION.md) - Model-specific feature sets
