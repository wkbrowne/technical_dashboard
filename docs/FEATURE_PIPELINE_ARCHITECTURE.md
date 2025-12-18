# Feature Pipeline Architecture

This document describes the technical architecture of the feature computation pipeline, including data flow, parallelism strategies, and architectural conventions.

For the high-level ML pipeline (feature selection, hyperparameter tuning, model training), see [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LOADING                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  cache/stock_data_universe.parquet  →  Daily OHLCV (wide format)                │
│  cache/stock_data_etf.parquet       →  ETF OHLCV (long format)                  │
│  cache/US universe_*.csv            →  Symbol list with sectors                 │
│  cache/fred_*.parquet               →  FRED macro data (various frequencies)    │
│                                                                                  │
│  Column normalization: ALL COLUMNS LOWERCASE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WEEKLY DATA GENERATION [FULLY IMPLEMENTED]                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  [DONE] --resample-weekly flag → cache/stock_data_weekly.parquet                │
│  [DONE] --resample-weekly flag → cache/etf_data_weekly.parquet                  │
│  [DONE] Pipeline auto-loads cached weekly data (skips inline resampling)        │
│                                                                                  │
│  Usage:                                                                          │
│    1. Generate cache: python -m src.cli.download --universe sp500 --resample-weekly
│    2. Run pipeline: python -m src.cli.compute --timeframes D,W                  │
│       → Automatically loads cached weekly data if available                     │
│       → Falls back to inline resampling if cache not found                      │
│                                                                                  │
│  Resampling rules:                                                               │
│  - open: first of week                                                           │
│  - high: max of week                                                             │
│  - low: min of week                                                              │
│  - close/adjclose: last of week (Friday)                                         │
│  - volume: sum of week                                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│     DAILY FEATURE COMPUTATION     │   │    WEEKLY FEATURE COMPUTATION     │
├───────────────────────────────────┤   ├───────────────────────────────────┤
│  Single-Stock Features:           │   │  Weekly-Only Features:            │
│  - trend.py (MA slopes, RSI)      │   │  - Weekly RSI, MACD               │
│  - volatility.py (ATR, regimes)   │   │  - Weekly sector breadth proxy    │
│  - volume.py (volume ratios)      │   │  - Weekly cross-asset             │
│  - distance.py (distance to MA)   │   │  - Weekly alpha momentum          │
│  - range_breakout.py              │   │                                   │
│                                   │   │  [DONE: loads cached weekly]      │
│  Cross-Sectional Features:        │   │  Falls back to inline if no cache │
│  - alpha.py (alpha vs benchmarks) │   │                                   │
│  - xsec.py (cross-sec momentum)   │   │                                   │
│  - sector_breadth.py (ETF proxy)  │   │                                   │
│                                   │   │                                   │
│                                   │   │                                   │
└───────────────────────────────────┘   └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE MERGING                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Weekly features merged to daily timeline (leakage-safe)                         │
│  - Weekly feature assigned to Friday                                             │
│  - Propagated to next week's daily rows via forward-fill                         │
│  - Monday sees previous Friday's weekly features (no leakage)                    │
│                                                                                  │
│  Output: artifacts/features_complete.parquet                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TARGET GENERATION                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Triple barrier labeling using atr_percent                                       │
│  - Upper barrier: +3 ATR (profit target)                                         │
│  - Lower barrier: -1.5 ATR (stop loss)                                           │
│  - Time barrier: 10 days max holding                                             │
│                                                                                  │
│  Output: artifacts/targets_triple_barrier.parquet                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Format Standards

### 2.1 Column Naming Convention

**CRITICAL: All column names MUST be lowercase.**

This convention prevents bugs from case mismatches between data sources:

| Source | Original Format | Required Format |
|--------|----------------|-----------------|
| Yahoo Finance API | `AdjClose`, `Close`, `High` | `adjclose`, `close`, `high` |
| ETF Cache (long format) | `metric` values: `AdjClose`, `Close` | Lowercase after pivot |
| Feature columns | Already lowercase | `rsi_14`, `atr_percent`, `w_beta_spy` |

**Enforcement Points:**

1. **Data Loading** (`src/data/loader.py`):
   ```python
   # After loading any DataFrame:
   df.columns = [c.lower() for c in df.columns]
   ```

2. **ETF Cache Loading** (`src/features/cross_sectional.py:876`):
   ```python
   etf_df.columns = [c.lower() for c in etf_df.columns]
   ```

3. **Output Validation** (future):
   ```python
   assert all(c.islower() or c in ['symbol', 'date'] for c in df.columns)
   ```

**Common Violations:**
- ETF cache uses `AdjClose` → must lowercase after pivot
- Yahoo Finance returns `Close` → lowercase immediately
- Mixed-case feature names → always use snake_case

### 2.2 File Format Standards

| File | Format | Index | Required Columns |
|------|--------|-------|------------------|
| `stock_data_universe.parquet` | Wide (symbols as columns) | DatetimeIndex | Per metric: symbol columns |
| `stock_data_etf.parquet` | Long | None | `date`, `symbol`, `value`, `metric` |
| `stock_data_weekly.parquet` | Wide per symbol | DatetimeIndex | `open`, `high`, `low`, `close`, `adjclose`, `volume` |
| `features_daily.parquet` | Long | None | `symbol`, `date`, + feature columns |
| `features_weekly.parquet` | Long | None | `symbol`, `date`, + weekly feature columns |
| `features_complete.parquet` | Long | None | `symbol`, `date`, + all feature columns |

**DataFrame Requirements:**
- Index: DatetimeIndex for time-series operations, or reset to columns for parquet
- Dtypes: `float32` for features (memory efficiency), `category` for symbols
- Sorting: Always sorted by `['symbol', 'date']` before saving

---

## 3. Pipeline Stages

### 3.1 Data Loading

**Entry Point:** `src/data/loader.py`

**Cache Structure:**
```
cache/
├── stock_data_universe.parquet   # ~3000 stocks, 5 years daily OHLCV
├── stock_data_etf.parquet        # ~100 ETFs, long format
├── US universe_*.csv             # Symbol metadata with sectors
├── stock_data_weekly.parquet     # Pre-computed weekly (NEW)
└── etf_data_weekly.parquet       # Pre-computed ETF weekly (NEW)
```

**⚠️ CRITICAL: Lowercase Column Names Immediately After Loading**

Column names MUST be lowercased immediately after loading any data source. This is the **first operation** performed on raw data before any other processing:

```python
# IMMEDIATELY after loading any DataFrame:
df.columns = [c.lower() for c in df.columns]
```

**Why this matters:**
- Yahoo Finance returns CamelCase columns (`AdjClose`, `Close`, `Volume`)
- Feature code expects lowercase (`adjclose`, `close`, `volume`)
- Case mismatches cause silent bugs (features compute as NaN)
- Example bug: ETF cache had `AdjClose`, weekly beta looked for `adjclose` → all NaN

**Enforcement points in the codebase:**
1. `src/data/loader.py` - `load_cached_data()` normalizes immediately
2. `src/features/cross_sectional.py:876` - ETF cache columns lowercased
3. Any new data loader MUST lowercase before returning

**Never assume incoming data has correct case. Always normalize.**

**Data Cleaning: Untradeable Asset Filtering**

The pipeline automatically filters out untradeable and problematic securities at load time via `_filter_untradeable_securities()` and `filter_suspicious_price_symbols()` in [src/data/loader.py](../src/data/loader.py).

**Step 1: Untradeable Securities Filter** (by symbol pattern and description)

| Category | Detection Method | Examples Filtered |
|----------|------------------|-------------------|
| SPACs | Description contains "Acquisition Corp/Inc/Company" | PSTH, CCIV |
| ADRs | Description contains "ADR" or "American Depositary" | BABA, TSM |
| Warrants | Symbol pattern `.W`, `/W`, `-W`, or `WS` suffix | FOOW, FOO.WS |
| Units | Symbol pattern `/U`, `-U` suffix | FOO/U, FOO-U |
| Rights | Symbol pattern `/R`, `-R` suffix | FOO/R |
| Preferred | Symbol pattern `.PR`, `-P` suffix | BAC.PR.A |
| Foreign | Symbol ends with `F` (foreign ordinary) | RYDAF |

**Step 2: Suspicious Price Filter** (data quality)

| Rule | Threshold | Reason |
|------|-----------|--------|
| Max price | > $50,000 | Data error or unadjusted split |
| Min price | < $0.01 | Penny stock data unreliable |
| Price range ratio | max/min > 10,000x | Split adjustment issues |

**Implementation:**

```python
# Automatic at load time - no user action required
data, sectors = load_stock_universe(include_sectors=True)
# Output: "📊 Filtered to 2847 tradeable symbols (removed SPACs, ADRs, etc.)"
# Output: "⚠️ Filtering 3 symbols with suspicious prices: ['XYZ', ...]"
```

**To disable filtering** (not recommended):

```python
from src.data.loader import _symbols_from_csv
symbols = _symbols_from_csv(csv_path, filter_untradeable=False)
```

### 3.2 FRED Macro Data

**Source:** `cache/fred_data.parquet` (downloaded via `src/data/fred.py`)

FRED provides macroeconomic indicators at various frequencies. Each series has a defined **publication lag** (`pub_lag_days`) indicating when data becomes available after its reference date.

**IMPORTANT: FRED Data is NOT Resampled**

Unlike stock/ETF data, FRED data is **not resampled to weekly**. The pipeline:
1. Loads raw FRED data (daily and weekly series mixed)
2. Forward-fills weekly series to daily granularity
3. Applies publication lag via `shift()`
4. Attaches directly to daily stock data

Weekly FRED series (ICSA, CCSA, NFCI) are forward-filled to appear on every trading day, not aggregated.

**FRED Series with Publication Lags:**

| Series | Frequency | Description | `pub_lag_days` | Reason |
|--------|-----------|-------------|----------------|--------|
| DGS10, DGS2 | Daily | Treasury yields | 1 | Available next morning |
| T10Y2Y, T10Y3M | Daily | Yield spreads | 1 | Available next morning |
| BAMLH0A0HYM2 | Daily | High yield OAS | 1 | ICE BofA 1-day lag |
| DFEDTARU | Daily | Fed funds target | 1 | Conservative buffer |
| VIXCLS | Daily | VIX close | 1 | FRED updates next day |
| ICSA | Weekly | Initial claims | 5 | Week ends Sat, released Thu |
| CCSA | Weekly | Continued claims | 12 | Extra week lag vs ICSA |
| NFCI | Weekly | Financial conditions | 7 | ~1 week after reference |

**Publication Lag Handling:**

Each series is automatically shifted by its `pub_lag_days` to prevent lookahead bias:

```python
# From src/data/fred.py - compute_fred_features()

# Get publication lag for this series (defined per series)
pub_lag = config.get('pub_lag_days', 1)
total_lag = pub_lag + additional_lag_days

# Apply publication lag to ALL features from this series
for feat_name, feat_series in series_features.items():
    features[feat_name] = feat_series.shift(total_lag)
```

**Example: Initial Jobless Claims (ICSA)**
- Week ending Saturday
- Released Thursday (5 days later)
- `pub_lag_days = 5`
- On Friday, you see data from week ending 6 days ago (correct, no lookahead)

**Alignment to Daily Data:**

FRED data is forward-filled to align with daily stock dates:

```python
# Raw FRED data has gaps (weekends, holidays, different frequencies)
df = df.ffill()  # Forward-fill to daily (no resampling)

# Then reindex to match each symbol's trading dates
symbol_fred = fred_features.reindex(df.index)
```

**Output Features (per series):**

| Transform | Feature Name | Description |
|-----------|--------------|-------------|
| `level` | `fred_{series}` | Raw lagged value |
| `change_5d` | `fred_{series}_chg5d` | 5-day change |
| `change_20d` | `fred_{series}_chg20d` | 20-day change |
| `change_4w` | `fred_{series}_chg4w` | 4-week change (weekly series) |
| `zscore_60d` | `fred_{series}_z60` | 60-day z-score |
| `zscore_52w` | `fred_{series}_z52w` | 52-week z-score (weekly series) |
| `percentile_252d` | `fred_{series}_pct252` | 1-year percentile rank |
| `percentile_104w` | `fred_{series}_pct104w` | 2-year percentile rank |

### 3.3 OHLC Adjustment

**IMPORTANT: OHLC prices are adjusted to match adjclose before feature computation.**

The pipeline uses `adjust_ohlc_to_adjclose()` ([src/features/ohlc_adjustment.py](../src/features/ohlc_adjustment.py)) to scale Open, High, Low, Close prices so they're consistent with adjusted close throughout history. This ensures:
- Technical indicators use split/dividend-adjusted prices
- Price relationships (high > close > low) are preserved
- Volume is NOT adjusted (remains raw)

```python
# Applied per-symbol before feature computation
df_adjusted = adjust_ohlc_to_adjclose(df)
```

### 3.4 Daily Feature Computation

Daily features are computed in parallel at the symbol level. Symbols are batched together to reduce IPC overhead - see [Section 4: Parallelism Architecture](#4-parallelism-architecture) for details.

**Feature Categories:**

| Category | Module | Features | Typical NaN % |
|----------|--------|----------|---------------|
| Trend | `trend.py` | `rsi_14`, `macd_histogram`, `trend_score_*` | 5-10% |
| Volatility | `volatility.py` | `atr_percent`, `vol_regime_*`, `rv_z_*` | 5-10% |
| Volume | `volume.py` | `volshock_ema`, `obv_z_60` | 5-10% |
| Distance | `distance.py` | `pct_dist_ma_*_z`, `pos_in_*_range` | 5-10% |
| Alpha | `alpha.py` | `alpha_mom_spy_*`, `alpha_mom_sector_*` | 15-25% |
| Cross-Sectional | `xsec.py` | `xsec_mom_*_z` | 10-15% |

### 3.5 Spread Features

**Module:** [src/features/spread_features.py](../src/features/spread_features.py)

Factor spread features capture market-level signals from ETF relationships. These are **global features** (same value for all symbols) computed once and broadcast.

**Global Spreads:**

| Spread Name | Formula | Signal |
|-------------|---------|--------|
| `qqq` | QQQ returns | Absolute tech momentum |
| `spy` | SPY returns | Absolute market momentum |
| `qqq_spy` | QQQ - SPY | Growth premium (tech vs broad market) |
| `rsp_spy` | RSP - SPY | Breadth spread (equal-weight vs cap-weight) |

**Metrics Computed for Each Spread:**

| Feature | Description | Windows |
|---------|-------------|---------|
| `{spread}_cumret_{w}` | Cumulative return over window | 20, 60, 120 days |
| `{spread}_zscore_{w}` | Z-score of current level | 60 days |
| `{spread}_slope_{w}` | Rolling OLS slope (momentum direction) | 20, 60 days |

**Per-Symbol Spreads (Best-Match):**

In addition to global spreads, each stock gets TWO per-symbol spreads:

| Spread | Formula | Description |
|--------|---------|-------------|
| `bestmatch_spy_*` | bestmatch - SPY | Cap-weighted sector ETF vs cap-weighted market |
| `bestmatch_ew_rsp_*` | bestmatch_ew - RSP | Equal-weight sector ETF vs equal-weight market |

**Why both?**
- `bestmatch_spy` captures sector-vs-market premium in cap-weighted terms (influenced by mega-caps)
- `bestmatch_ew_rsp` captures sector-vs-market premium in equal-weight terms (removes large-cap concentration effects)

The equal-weight spread is particularly useful for detecting breadth divergences where large caps mask underlying sector weakness/strength.

**Lag Handling:**

Daily spreads are lagged by 1 day to prevent look-ahead bias.

### 3.6 Weekly Data Generation

**[PARTIALLY IMPLEMENTED] Pre-computed weekly cache files**

> **Status:** Cache generation implemented via `--resample-weekly` flag.
> **Remaining:** Pipeline integration to load cached weekly data instead of inline resampling.

Generate pre-computed weekly cache files from daily data:

**Command:**
```bash
# Generate weekly cache for both stocks and ETFs
python -m src.cli.download --universe all --resample-weekly

# Or generate after download
python -m src.cli.download --universe sp500 --resample-weekly
python -m src.cli.download --universe etf --resample-weekly
```

**Output Files:**
- `cache/stock_data_weekly.parquet` - Weekly OHLCV for stocks
- `cache/etf_data_weekly.parquet` - Weekly OHLCV for ETFs

**Resampling Rules (W-FRI):**
```python
agg_rules = {
    'open': lambda x: x.iloc[0] if len(x) > 0 else np.nan,
    'high': 'max',
    'low': 'min',
    'close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
    'adjclose': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
    'volume': 'sum'
}
weekly_df = daily_df.resample('W-FRI').agg(agg_rules)
```

**Data Integrity - No Partial Periods:**

Weekly files must NOT contain partially completed weeks. A partial week occurs when:
- The current week hasn't ended yet (today is not Friday)
- A week has fewer than expected trading days due to data cutoff

```python
# CORRECT: Drop the last week if it's incomplete
today = pd.Timestamp.now().normalize()
last_friday = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
if today.weekday() != 4:  # If today is not Friday
    weekly_df = weekly_df[weekly_df.index < last_friday]
```

Similarly, daily files must NOT contain partially completed days:
- No intraday partial data
- Last row should be a complete market close

**Benefits:**
- 60% faster pipeline (skip resampling on re-runs)
- Cleaner separation of concerns
- Easier debugging of weekly-specific issues
- Consistent weekly data across all components

### 3.7 Weekly Feature Computation

**[PARTIAL]** Currently computes weekly features inline from daily data.

> **Current behavior:** Weekly features are computed by resampling daily data during pipeline execution.
> **Target architecture:** Load from pre-computed `stock_data_weekly.parquet` and `etf_data_weekly.parquet`.

**Uses (planned):** Pre-computed `stock_data_weekly.parquet` and `etf_data_weekly.parquet`

**Weekly-Only Features:**
- `w_rsi_14`, `w_rsi_21` - Weekly RSI
- `w_macd_histogram`, `w_macd_signal` - Weekly MACD
- `w_beta_spy`, `w_beta_qqq` - Rolling beta vs benchmarks
- `w_alpha_mom_spy_*` - Weekly alpha momentum
- `w_sector_breadth_*` - Weekly sector ETF breadth proxy
- `w_vix_vxn_spread` - Weekly volatility spreads

**Weekly Spread Features:**

Like daily spreads, weekly spread features are computed via `add_weekly_spread_features()`:

| Feature | Description | Windows (weeks) |
|---------|-------------|-----------------|
| `w_{spread}_cumret_{w}` | Weekly cumulative return | 4, 12, 24 |
| `w_{spread}_zscore_{w}` | Weekly z-score | 12 |
| `w_{spread}_slope_{w}` | Weekly momentum slope | 4, 12 |

**Output:** `artifacts/features_weekly.parquet`

### 3.8 Feature Merging

**Leakage Prevention:**

Weekly features must not leak future information to daily data:

```python
# CORRECT: Friday's weekly feature applies to NEXT week's daily rows
# Monday sees last Friday's feature (no leakage)
merged = pd.merge_asof(
    daily_df.sort_values('date'),
    weekly_df.sort_values('date'),
    on='date',
    by='symbol',
    direction='backward'  # Use previous weekly observation
)
```

### 3.9 Pipeline Output Files

The pipeline produces **two output files** at the end:

| File | Contents | Use Case |
|------|----------|----------|
| `features_complete.parquet` | ALL computed features (~600+) | Debugging, feature exploration, re-running selection |
| `features_filtered.parquet` | Curated ML-ready features (~250) | Model training, production |

**Why Two Files:**

1. **Full file** (`features_complete.parquet`) - Contains every computed feature including intermediate values (raw MAs, raw ATR, etc.). Useful for:
   - Debugging feature computation issues
   - Exploring new feature ideas
   - Re-running feature selection without recomputing

2. **Filtered file** (`features_filtered.parquet`) - Contains only ML-ready features after applying `filter_output_columns()`. Useful for:
   - Model training (smaller file, faster loading)
   - Production predictions
   - Sharing with downstream systems

### 3.10 Output Feature Filtering

**Module:** [src/feature_selection/base_features.py](../src/feature_selection/base_features.py)

The filtering step removes intermediate/raw features not suitable for ML.

**Feature Categories:**

| Category | Purpose | Count |
|----------|---------|-------|
| `BASE_FEATURES` | Core features for model training | ~38 |
| `EXPANSION_CANDIDATES` | Additional features for selection experiments | ~200 |
| `EXCLUDED_FEATURES` | Raw/unnormalized values (dropped) | ~50 |
| `META_COLUMNS` | Always kept: `symbol`, `date`, `ret` | 3 |
| `REQUIRED_FEATURES` | Always kept: `atr_percent` (for targets) | 1 |

**Excluded Feature Types (examples):**

- Raw OHLCV: `open`, `high`, `low`, `close`, `adjclose`, `volume`
- Raw moving averages: `ma_20`, `ma_50`, `ma_100`, `w_ma_20`, etc.
- Raw volatility: `atr14`, `rv_20`, `rv_60`
- Raw FRED levels: `fred_dgs10`, `fred_bamlh0a0hym2`
- Raw VIX levels: `vix_level`, `vxn_level`

These are excluded because:
1. **Not comparable across stocks** - raw prices vary by price level
2. **Not normalized** - use z-scores or percent changes instead
3. **Intermediate values** - only transformed versions are useful

**Filtering Call:**

```python
from src.feature_selection.base_features import filter_output_columns

# After all features computed, filter to curated set
df_filtered = filter_output_columns(df, keep_all=False)
```

**Output Validation:**

The output should contain:
- All 38 `BASE_FEATURES` (verify with `validate_features()`)
- All ~200 `EXPANSION_CANDIDATES` (for selection experiments)
- Required columns: `symbol`, `date`, `ret`, `atr_percent`

---

## 3.11 Intermediate Checkpoint Architecture

**[FULLY IMPLEMENTED] Checkpoint system for staged processing**

> **Implementation:** `src/pipelines/checkpoint.py` (CheckpointManager, CheckpointConfig)
> **CLI flags:** `--checkpoint-dir`, `--resume-from`, `--cleanup-checkpoints`, `--list-checkpoints`

For large datasets or memory-constrained environments, the pipeline should output intermediate results after each major thematic computation stage. This allows:

1. **Memory clearing** between stages (via `del` + `gc.collect()`)
2. **Resumption** from any checkpoint if pipeline fails
3. **Debugging** of specific stages in isolation
4. **Incremental development** - modify one stage without re-running all

**Checkpoint Files:**

```
artifacts/checkpoints/
├── 01_ohlcv_adjusted.parquet      # After OHLC adjustment
├── 02_single_stock.parquet        # After single-stock features (RSI, MACD, ATR, etc.)
├── 03_spread_features.parquet     # After spread features (QQQ-SPY, RSP-SPY, etc.)
├── 04_factor_regression.parquet   # After factor regression (beta, alpha, bestmatch spreads)
├── 05_cross_sectional.parquet     # After cross-sectional features (xsec momentum, ranks)
├── 06_breadth.parquet             # After sector ETF breadth proxy features
├── 07_weekly_features.parquet     # After weekly feature computation
└── 08_merged_complete.parquet     # Final merged output
```

**Checkpoint Pattern:**

```python
def pipeline_with_checkpoints(indicators_by_symbol, output_dir, resume_from=None):
    """
    Pipeline with intermediate outputs after each thematic step.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame
        output_dir: Directory for checkpoint files
        resume_from: Optional checkpoint name to resume from (e.g., '03_spread_features')
    """
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Stage 1: OHLC Adjustment
    if not resume_from or resume_from < '01':
        indicators_by_symbol = adjust_all_ohlc(indicators_by_symbol)
        save_checkpoint(indicators_by_symbol, checkpoint_dir / '01_ohlcv_adjusted.parquet')
    else:
        indicators_by_symbol = load_checkpoint(checkpoint_dir / '01_ohlcv_adjusted.parquet')

    # Stage 2: Single-stock features
    if not resume_from or resume_from < '02':
        indicators_by_symbol = add_single_stock_features(indicators_by_symbol)
        save_checkpoint(indicators_by_symbol, checkpoint_dir / '02_single_stock.parquet')
        gc.collect()  # Clear intermediate memory
    else:
        indicators_by_symbol = load_checkpoint(checkpoint_dir / '02_single_stock.parquet')

    # Stage 3: Spread features (global)
    if not resume_from or resume_from < '03':
        add_spread_features(indicators_by_symbol, lag_days=1)
        add_weekly_spread_features(indicators_by_symbol, prefix='w_')
        save_checkpoint(indicators_by_symbol, checkpoint_dir / '03_spread_features.parquet')
        gc.collect()
    else:
        indicators_by_symbol = load_checkpoint(checkpoint_dir / '03_spread_features.parquet')

    # Stage 4: Factor regression (per-symbol beta, alpha, bestmatch spreads)
    if not resume_from or resume_from < '04':
        add_joint_factor_features(indicators_by_symbol, ...)
        save_checkpoint(indicators_by_symbol, checkpoint_dir / '04_factor_regression.parquet')
        gc.collect()
    else:
        indicators_by_symbol = load_checkpoint(checkpoint_dir / '04_factor_regression.parquet')

    # ... continue for remaining stages
```

**Thematic Steps (in order):**

| Step | Checkpoint File | Features Added | ~Memory (3000 sym) |
|------|-----------------|----------------|-------------------|
| 1. OHLC Adjustment | `01_ohlcv_adjusted.parquet` | Adjusted OHLC columns | ~2 GB |
| 2. Single-Stock | `02_single_stock.parquet` | RSI, MACD, ATR, MA slopes, volume ratios | ~4 GB |
| 3. Spread Features | `03_spread_features.parquet` | QQQ/SPY/RSP spread metrics | ~4.5 GB |
| 4. Factor Regression | `04_factor_regression.parquet` | Beta, alpha, bestmatch spreads | ~5 GB |
| 5. Cross-Sectional | `05_cross_sectional.parquet` | Xsec momentum, ranks | ~5.5 GB |
| 6. Breadth | `06_breadth.parquet` | Sector ETF breadth proxy, McClellan | ~6 GB |
| 7. Weekly Features | `07_weekly_features.parquet` | All w_* features | ~7 GB |
| 8. Final Merge | `08_merged_complete.parquet` | Combined daily + weekly | ~7 GB |

**Memory Clearing Pattern:**

```python
# After saving checkpoint, clear intermediate data structures
del intermediate_results
del large_panel
gc.collect()

# Read back only what's needed for next stage
indicators_by_symbol = load_checkpoint(checkpoint_path)
```

**Resumption Usage:**

```bash
# Resume from factor regression (skip stages 1-3)
python -m src.cli.compute --resume-from 04_factor_regression
```

**Benefits:**

1. **Memory efficiency** - Peak memory reduced by ~40% (don't hold all stages simultaneously)
2. **Fault tolerance** - Resume from last successful checkpoint after crash
3. **Development speed** - Iterate on later stages without re-running early expensive stages
4. **Debugging** - Inspect intermediate outputs to diagnose issues

**Note:** Checkpoint files should be cleaned up after successful pipeline completion to save disk space.

---

## 4. Parallelism Architecture

**Note:** We use `joblib.Parallel` for all parallel processing. Worker lifecycle (spawning, pooling, shutdown) is managed entirely by joblib - we do not explicitly manage worker processes.

### 4.1 Core Principle: Batch-Level Parallelism with Minimal Data Transfer

**CRITICAL: Only send data needed for each computation to workers.**

Each worker should receive:
- Only the symbols it needs to process
- Only the additional reference data required (e.g., SPY for beta calculation)
- Never the entire dataset

```python
# CORRECT: Send only what's needed to each worker
def compute_beta_for_symbol(symbol_data: pd.DataFrame, spy_data: pd.DataFrame) -> pd.Series:
    """Worker receives only: (1) single symbol's data, (2) benchmark data."""
    # Compute rolling beta
    return beta_series

# Dispatch minimal data per batch
results = Parallel(n_jobs=-1)(
    delayed(compute_beta_for_symbol)(
        symbol_dict[sym],      # Only this symbol's data
        spy_df                 # Only the benchmark needed
    )
    for sym in symbols_to_process
)
```

```python
# WRONG: Sending entire dataset to each worker
results = Parallel(n_jobs=-1)(
    delayed(process_symbol)(
        sym,
        full_universe_df      # BAD: Serializes entire dataset N times
    )
    for sym in symbols
)
```

**Data Transfer Patterns by Feature Type:**

| Feature Type | Data Sent to Worker | Example |
|--------------|---------------------|---------|
| Single-stock (RSI, MACD) | Symbol's OHLCV only | `symbol_dict[sym]` |
| Beta vs benchmark | Symbol + benchmark ETF | `symbol_dict[sym]`, `spy_df` |
| Sector-relative | Symbol + sector ETF | `symbol_dict[sym]`, `sector_etf_df` |
| Cross-sectional rank | Symbol + precomputed ranks | `symbol_dict[sym]`, `rank_series` |

### 4.2 Memory Management

**CRITICAL: Avoid O(N²) memory patterns**

**Bad Pattern (O(N²) memory):**
```python
# Creates N full DataFrame copies
symbol_groups = [(sym, df_long[df_long['symbol'] == sym]) for sym in symbols]
```

**Good Pattern (O(N) memory):**
```python
# Single-pass groupby, no duplication
symbol_data_dict = {}
for sym, group in df_long.groupby('symbol', observed=True):
    symbol_data_dict[sym] = group.copy()
```

**DataFrame Fragmentation Prevention:**

**Bad Pattern:**
```python
# Causes memory fragmentation
panel = pd.DataFrame()
for s in symbols:
    panel[s] = data[s]  # Memory reallocation each time
```

**Good Pattern:**
```python
# Single memory allocation
series_list = [pd.Series(data[s], name=s) for s in symbols]
panel = pd.concat(series_list, axis=1)
```

### 4.3 Multiprocessing Strategy

**Symbol Batching:**

Symbols are batched together before sending to workers to reduce inter-process communication (IPC) overhead. Instead of dispatching one symbol at a time, we group symbols into batches:

```python
from joblib import Parallel, delayed

# Batch symbols to reduce IPC overhead
BATCH_SIZE = 100  # ~100 symbols per worker

def _feature_worker_batch(symbol_data_list):
    """Process multiple symbols in a single worker call."""
    results = []
    for sym, df in symbol_data_list:
        out = compute_single_stock_features(df)
        results.append((sym, out))
    return results

# Create batches and dispatch
symbol_chunks = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

batch_results = Parallel(n_jobs=-1, backend='loky')(
    delayed(_feature_worker_batch)(chunk)
    for chunk in symbol_chunks
)
```

**Adaptive Chunking by Dataset Size:**

| Dataset Size | Strategy | Rationale |
|-------------|----------|-----------|
| < 100 symbols | Direct parallel (1 symbol/task) | Low overhead |
| 100-500 symbols | Batched parallel (~25/chunk) | Balance memory/speed |
| > 500 symbols | Batched parallel (~100/chunk) | Minimize IPC overhead |

**Backend Selection:**
- `loky`: CPU-bound feature computation (default)
- `threading`: I/O-bound merge operations
- `multiprocessing`: Legacy fallback

### 4.4 Performance Benchmarks

| Stage | Symbols/sec | Memory | Notes |
|-------|-------------|--------|-------|
| Single-stock features | 40-60 | O(N) | Parallel, loky |
| Cross-sectional panel | 200-250 | O(N) | pd.concat optimized |
| Weekly resampling | 100-150 | O(N) | Pre-computed (skip on re-run) |
| Weekly merge | 100-120 | O(N) | Threading backend |

**Full Pipeline (3000 symbols, D+W):**
- Without weekly cache: ~17 minutes
- With weekly cache: ~7 minutes (estimated)

---

## 5. Cross-Sectional Features

### 5.1 Enhanced Sector Mappings

**Module:** [src/features/sector_mapping.py](../src/features/sector_mapping.py)

Each stock is assigned multiple ETF benchmarks for relative performance features:

**Mapping Types:**

| Mapping | Description | Coverage |
|---------|-------------|----------|
| `sector_etf` | Base SPDR sector ETF (XLK, XLF, etc.) | ~100% |
| `equal_weight_etf` | Equal-weight sector ETF (RSPT, RSPF, etc.) | ~60% |
| `subsector_etf` | Industry-specific ETF (SOXX, XBI, etc.) | ~40% |
| `bestmatch_etf` | Best correlating cap-weighted ETF | ~100% |
| `bestmatch_ew_etf` | Best correlating equal-weight ETF (RSPT, RSPF, etc.) | ~80% |

**Subsector Discovery:**

The pipeline uses **correlation-based subsector discovery** to find the best benchmark ETF for each stock:

```python
# From sector_mapping.py - finds ETF with highest correlation improvement
def _find_best_subsector_by_correlation(symbol, sector_etf, stock_data, etf_data):
    """Tests ALL subsector ETFs and returns the one with highest correlation."""
    # Requires min 3% correlation improvement over sector ETF
    # Requires min 100 data points for reliable correlation
```

**Enhanced Mapping Structure:**

```python
# Example mapping for AAPL
enhanced_mappings['AAPL'] = {
    'sector_etf': 'XLK',           # Technology sector
    'equal_weight_etf': 'RSPT',    # Equal-weight tech
    'subsector_etf': 'SOXX',       # Semiconductors (found via correlation)
    'bestmatch_etf': 'SOXX',       # Best correlating ETF
    'industry': 'consumer electronics',
    'market_cap': 3000000000000
}
```

**Usage in Features:**

- `alpha_mom_sector_*` uses `sector_etf`
- `rel_strength_sector_ew_*` uses `equal_weight_etf`
- `rel_strength_subsector_*` uses `subsector_etf`
- `bestmatch_spy_*` spread uses `bestmatch_etf` (cap-weighted sector vs cap-weighted market)
- `bestmatch_ew_rsp_*` spread uses `bestmatch_ew_etf` (equal-weight sector vs equal-weight market)

### 5.2 Benchmark Requirements

Cross-sectional features require benchmark ETF data from the ETF cache:

| Benchmark | Symbol | Purpose | Source |
|-----------|--------|---------|--------|
| Market | SPY | Beta, alpha vs market | ETF cache or universe |
| Growth | QQQ | Beta, alpha vs tech | ETF cache |
| Sector | XLK, XLF, etc. | Sector-relative alpha | ETF cache |
| Breadth | RSP | Equal-weight spread | ETF cache |

**Loading Priority:**
1. Check if symbol in `indicators_by_symbol` (stock universe)
2. Fall back to `cache/stock_data_etf.parquet`
3. Fall back to `cache/etf_data_weekly.parquet` (for weekly)

### 5.3 Sector ETF Breadth Proxy (Global-by-Date)

**Module:** [src/features/sector_breadth.py](../src/features/sector_breadth.py)

Traditional market breadth data (NYSE advance/decline) is not reliably available via free APIs. The Sector ETF Breadth Proxy provides a survivorship-bias-free alternative using the 11 Select Sector SPDR ETFs.

**Purpose:**

Approximate traditional market-level breadth indicators (advance/decline, percent above moving averages) using the fixed set of 11 sector ETFs. These cover the entire S&P 500 without survivorship bias since the ETF set is static.

**Inputs:**

The 11 Select Sector SPDR ETFs (must be present in ETF cache):

| ETF | Sector |
|-----|--------|
| XLB | Materials |
| XLC | Communication Services |
| XLE | Energy |
| XLF | Financials |
| XLI | Industrials |
| XLK | Technology |
| XLP | Consumer Staples |
| XLRE | Real Estate |
| XLU | Utilities |
| XLV | Health Care |
| XLY | Consumer Discretionary |

**Daily Outputs:**

| Feature | Description | Computation |
|---------|-------------|-------------|
| `sector_breadth_adv` | Count of advancing sectors | Count where daily return > 0 |
| `sector_breadth_dec` | Count of declining sectors | Count where daily return < 0 |
| `sector_breadth_net_adv` | Net advancers | adv - dec |
| `sector_breadth_ad_line` | Cumulative A/D line | Cumulative sum of net_adv |
| `sector_breadth_pct_above_ma50` | Pct of sectors above 50-day MA | Count(close > ma50) / 11 |
| `sector_breadth_pct_above_ma200` | Pct of sectors above 200-day MA | Count(close > ma200) / 11 |
| `sector_breadth_mcclellan_osc` | McClellan Oscillator | EMA(19, net_adv) - EMA(39, net_adv) |
| `sector_breadth_mcclellan_sum` | McClellan Summation Index | Cumulative sum of McClellan Oscillator |

**Weekly Outputs:**

Weekly versions are computed on weekly-resampled ETF data and prefixed with `w_`:

| Feature | Description |
|---------|-------------|
| `w_sector_breadth_adv` | Weekly advancing sectors count |
| `w_sector_breadth_dec` | Weekly declining sectors count |
| `w_sector_breadth_net_adv` | Weekly net advancers |
| `w_sector_breadth_ad_line` | Cumulative weekly A/D line |
| `w_sector_breadth_pct_above_ma10` | Pct above 10-week MA (≈50-day) |
| `w_sector_breadth_pct_above_ma40` | Pct above 40-week MA (≈200-day) |
| `w_sector_breadth_mcclellan_osc` | Weekly McClellan Oscillator (EMA(4) - EMA(8)) |
| `w_sector_breadth_mcclellan_sum` | Weekly McClellan Summation Index |

**McClellan Oscillator:**

The McClellan Oscillator is a breadth momentum indicator computed from the A/D line:

```python
# Daily: traditional EMA spans (19 and 39 days)
ema_fast = net_adv.ewm(span=19, adjust=False).mean()
ema_slow = net_adv.ewm(span=39, adjust=False).mean()
mcclellan_osc = ema_fast - ema_slow

# Weekly: adjusted spans (4 and 8 weeks ≈ 19 and 39 days)
w_ema_fast = w_net_adv.ewm(span=4, adjust=False).mean()
w_ema_slow = w_net_adv.ewm(span=8, adjust=False).mean()
w_mcclellan_osc = w_ema_fast - w_ema_slow

# Summation index (cumulative)
mcclellan_sum = mcclellan_osc.cumsum()
```

Interpretation:
- Positive oscillator → breadth momentum improving
- Negative oscillator → breadth momentum weakening
- Summation index shows longer-term breadth trend

**Joining Rule:**

These features are **global-by-date** (same value for all symbols on a given date). Computed once per date, then broadcast to all symbols via a left join on date.

```python
# Global features computed once
breadth_features = compute_sector_breadth(etf_data)  # DatetimeIndex → features

# Broadcast to all symbols
for sym, df in indicators_by_symbol.items():
    df = df.join(breadth_features, on='date', how='left')
```

**Replacement Notice:**

This feature set **replaces** the prior `breadth.py` logic that depended on external advance/decline data sources (which are unavailable via free APIs). The old `ad_ratio_universe` and similar features are deprecated.

**Non-Goals:**

- Does NOT replace per-symbol sector-relative features (alpha, relative strength)
- Does NOT provide true NYSE advance/decline counts (only sector-level proxy with 11 data points per day)

### 5.4 Joint Factor Model (Per-Symbol)

**Module:** [src/features/factor_regression.py](../src/features/factor_regression.py)

The joint factor model computes factor exposures and alpha jointly via multivariate rolling ridge regression. Instead of computing betas one factor at a time (which produces correlated estimates), this approach computes all factor exposures simultaneously, producing cleaner estimates that properly account for factor correlations.

**Orthogonalized 4-Factor Design:**

```
R_stock = α + β_market × R_SPY
            + β_qqq × (R_QQQ - R_SPY)
            + β_bestmatch × (R_bestmatch - R_SPY)
            + β_breadth × (R_RSP - R_SPY)
            + ε
```

| Factor | Formula | Interpretation |
|--------|---------|----------------|
| `beta_market` | R_SPY | Broad market exposure |
| `beta_qqq` | R_QQQ - R_SPY | Growth/tech premium over market |
| `beta_bestmatch` | R_bestmatch - R_SPY | Sector/subsector premium over market |
| `beta_breadth` | R_RSP - R_SPY | Equal-weight vs cap-weight spread |

**Why Orthogonalized Factors:**

The spread factors (QQQ-SPY, bestmatch-SPY, RSP-SPY) are orthogonalized versus market to:
1. Avoid multicollinearity between correlated factors
2. Provide cleaner interpretation of factor loadings
3. Isolate the *premium* over market rather than the absolute exposure

For example, `beta_qqq` measures exposure to the *growth premium* (how much more/less the stock moves when QQQ outperforms SPY), not absolute QQQ correlation.

**Best-Match ETF Selection:**

Each stock is matched to TWO best-match ETFs based on highest R² from univariate regression:

| Mapping | Candidates | Purpose |
|---------|------------|---------|
| `bestmatch_etf` | Cap-weighted sector/subsector ETFs (XLK, SMH, XBI, etc.) | Sector-relative exposure |
| `bestmatch_ew_etf` | Equal-weight sector ETFs (RSPT, RSPF, RSPH, etc.) | Remove large-cap concentration |

**Cap-Weighted Candidates (25 ETFs):**
- Sector: XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLC, XLRE
- Tech subsectors: SMH, IGV, SKYY, HACK
- Finance: KBE, KRE
- Healthcare: IBB, XBI, IHE
- Industrial: ITA, ITB
- Energy: XOP
- Consumer: XRT
- Materials: TAN, URA, LIT, COPX

**Equal-Weight Candidates (10 ETFs):**
- RSPT (Tech), RSPF (Financials), RSPH (Healthcare), RSPE (Energy)
- RSPN (Industrials), RSPC (Consumer Disc), RSPS (Consumer Staples)
- RSPU (Utilities), RSPM (Materials), RSPD (Communication)

**Example:** NVDA matches SMH (semiconductors) rather than XLK (broad tech) because SMH explains more variance.

**Features Output:**

For daily (`beta_*`) and weekly (`w_beta_*`) frequencies:

| Feature | Description |
|---------|-------------|
| `alpha` | Regression intercept (idiosyncratic return) |
| `beta_market` | Market factor loading |
| `beta_qqq` | Growth premium factor loading |
| `beta_bestmatch` | Sector/subsector premium factor loading |
| `beta_breadth` | Equal-weight spread factor loading |
| `residual_mean` | Rolling mean of regression residuals |
| `residual_cumret` | Cumulative residual return over window |
| `residual_vol` | Rolling volatility of residuals |

**Spread Features (from factor regression):**

The factor regression module also computes best-match spread features:

| Feature | Formula | Description |
|---------|---------|-------------|
| `bestmatch_spy_cumret_*` | bestmatch - SPY | Cap-weighted sector vs market cumulative return |
| `bestmatch_spy_zscore_*` | Z-score | Normalized spread level |
| `bestmatch_spy_slope_*` | OLS slope | Spread momentum direction |
| `bestmatch_ew_rsp_cumret_*` | bestmatch_ew - RSP | Equal-weight sector vs equal-weight market |
| `bestmatch_ew_rsp_zscore_*` | Z-score | Normalized EW spread level |
| `bestmatch_ew_rsp_slope_*` | OLS slope | EW spread momentum direction |

**Configuration:**

```python
@dataclass
class FactorRegressionConfig:
    daily_windows: List[int] = [60]      # 60-day rolling window
    weekly_windows: List[int] = [12]     # 12-week (~60 days)
    ridge_alpha: float = 0.01            # L2 regularization
    min_periods_ratio: float = 0.33      # min_periods = window × ratio
    min_periods_floor: int = 20          # Absolute minimum observations
```

**Parallelism:**

Uses stocks_per_worker batching (200 symbols per worker) with factor subsetting:
- Each worker only receives the factor returns it needs (core factors + batch-specific best-match ETFs)
- Reduces serialization overhead by ~70% for large universes

**Weekly Computation:**

Weekly factor features handle resampling inside workers to avoid sequential bottleneck:
1. Factor returns are pre-resampled to weekly in main thread (once)
2. Each worker resamples its batch of stock returns to weekly in parallel
3. Results are forward-filled back to daily index

**Beta Feature Naming Convention:**

Two beta computation methods exist with distinct names:

| Feature | Method | Module | Description |
|---------|--------|--------|-------------|
| `beta_market` / `w_beta_market` | Joint factor (orthogonalized) | `factor_regression.py` | Market beta from 4-factor ridge regression |
| `beta_qqq` / `w_beta_qqq` | Joint factor (orthogonalized) | `factor_regression.py` | Growth premium (QQQ-SPY spread exposure) |
| `beta_spy_simple` / `w_beta_spy_simple` | Simple rolling cov/var | `cross_sectional.py` | Univariate beta vs SPY |
| `beta_qqq_simple` / `w_beta_qqq_simple` | Simple rolling cov/var | `cross_sectional.py` | Univariate beta vs QQQ |

**Key Differences:**

- **Joint factor betas** (`beta_market`, `beta_qqq`, etc.): Orthogonalized and computed simultaneously via ridge regression. `beta_qqq` measures exposure to the *growth premium* (QQQ outperformance vs SPY), not absolute QQQ correlation.

- **Simple betas** (`beta_spy_simple`, `beta_qqq_simple`): Traditional univariate rolling cov/var. `beta_qqq_simple` measures absolute correlation to QQQ returns.

Both are useful: joint factor betas provide cleaner factor attribution, simple betas provide intuitive market sensitivity measures.

### 5.5 Date Alignment

Weekly features have date alignment challenges:

**Problem:** Stock trades Monday, but weekly closes Friday. Dates may not match exactly.

**Solution - align_to_index:**
```python
def align_to_index(source: pd.Series, target_index: pd.Index) -> pd.Series:
    """Align source series to target index using ffill/bfill."""
    if len(source) == 0 or len(target_index) == 0:
        return pd.Series(index=target_index, dtype='float64')
    combined_idx = source.index.union(target_index).sort_values()
    filled = source.reindex(combined_idx).ffill().bfill()
    return filled.reindex(target_index)
```

This handles:
- Different trading calendars (ETFs vs stocks)
- Holiday mismatches
- Missing data gaps

---

## 6. Known Issues & Patterns

### 6.1 Expected NaN Rates

| Feature Category | Expected NaN % | Reason |
|-----------------|----------------|--------|
| Basic technical | 5-10% | Lookback warmup period |
| Alpha vs SPY | 10-15% | Lookback + alignment |
| Alpha vs sector | 20-25% | ~25% stocks missing sector mapping |
| Alpha vs subsector | 40-50% | Limited subsector ETF coverage |
| QQQ features | 13-15% | Similar to SPY |

**Acceptable Ranges:**
- If NaN > 50%: Feature may not be useful
- If NaN > 30%: Document and consider fallback
- If NaN < 15%: Normal for technical features

### 6.2 Common Bugs

**1. Case Sensitivity:**
```python
# Bug: ETF cache has 'AdjClose', code expects 'adjclose'
# Fix: Lowercase immediately after loading
df.columns = [c.lower() for c in df.columns]
```

**2. Date Alignment Failures:**
```python
# Bug: Direct merge fails on mismatched weekly dates
merged = pd.merge(daily, weekly, on=['symbol', 'date'])  # Many NaNs!

# Fix: Use merge_asof with backward direction
merged = pd.merge_asof(daily, weekly, on='date', by='symbol', direction='backward')
```

**3. Memory Exhaustion:**
```python
# Bug: Creating full DataFrame copies in list comprehension
symbol_data = [(s, df[df['symbol'] == s]) for s in symbols]  # O(N²)

# Fix: Use groupby
symbol_data = {s: g for s, g in df.groupby('symbol')}  # O(N)
```

**4. Future Leakage:**
```python
# Bug: Weekly feature from same week leaks to Monday
weekly_df.loc[friday_date, 'feature']  # Contains Friday data!

# Fix: Use previous week's Friday for current week
direction='backward'  # in merge_asof
```

**5. Suspicious Price Data:**
```python
# Bug: Symbols with unadjusted reverse splits have extreme prices
# Example: SMX showed max price of $217M (should be ~$150 after adjustment)
# These corrupt features and target generation

# Detection criteria (applied at multiple points):
max_price > 50000        # Extreme price
min_price < 0.01         # Penny stock artifacts
max/min ratio > 1000     # Unadjusted split (normal stocks rarely exceed 100x)

# Filtering locations (defense-in-depth):
# 1. src/data/loader.py:filter_suspicious_price_symbols() - data load time
# 2. src/cli/compute.py:_filter_suspicious_symbols() - CLI entry point
# 3. src/pipelines/orchestrator.py:_generate_triple_barrier_targets() - final check
```

---

## 7. Target Generation

**Module:** [src/features/target_generation.py](../src/features/target_generation.py)

The pipeline generates triple barrier targets with sample weights for training.

### 7.1 Triple Barrier Labeling

Each sample is labeled based on which barrier is hit first:

| Barrier | Condition | Label | Typical % |
|---------|-----------|-------|-----------|
| Upper | Price ≥ Entry + 3×ATR | `hit=1` (profit target) | ~17% |
| Lower | Price ≤ Entry - 1.5×ATR | `hit=0` (stop loss) | ~51% |
| Time | 10 days elapsed | `hit=-1` (expired) | ~32% |

**Configuration:**
- Uses `atr_percent` from features (REQUIRED)
- Upper barrier: 3× ATR
- Lower barrier: 1.5× ATR (2:1 reward/risk)
- Max holding period: 10 trading days

### 7.2 Sample Weights

The pipeline computes three weight columns for ML training:

**Weight Components:**

| Weight Column | Formula | Purpose |
|---------------|---------|---------|
| `weight_overlap` | 1 / (n_overlapping_trajs + 0.5) | Reduce correlated samples |
| `weight_class_balance` | n_samples / (n_classes × class_count) | Handle imbalanced classes |
| `weight_final` | overlap × class_balance (clipped & normalized) | Combined training weight |

**Overlap Weight:**

When multiple trades overlap in time, their outcomes are correlated. The overlap weight reduces the influence of these correlated samples:

```python
# Example: 3 overlapping trades = weight 1/(3+0.5) = 0.29
# Single trade (no overlap) = weight 1/(1+0.5) = 0.67
```

**Class Balance Weight:**

Uses sklearn-style class weighting for imbalanced data:

```python
# If 51% lower barrier, 32% expired, 17% upper:
# weight_lower = n / (3 × 0.51n) ≈ 0.65
# weight_expired = n / (3 × 0.32n) ≈ 1.04
# weight_upper = n / (3 × 0.17n) ≈ 1.96
```

**Final Weight:**

Combined multiplicatively, clipped to [0.01, 10.0], and normalized to sum to n_samples:

```python
weight_final = clip(overlap × class_balance, 0.01, 10.0)
weight_final = weight_final × n_samples / sum(weight_final)
```

### 7.3 Price Validation (Defense-in-Depth)

Target generation includes final-stage price validation to catch symbols with suspicious prices that may have slipped through earlier filters. This is a defense-in-depth measure since corrupted prices produce meaningless targets.

**Validation Criteria:**
```python
# Skip symbol if ANY of these are true:
max_price > 50000           # Extreme price (likely data error)
min_price < 0.01            # Penny stock artifacts
max_price / min_price > 1000  # Unadjusted reverse split
```

**Location:** `src/pipelines/orchestrator.py:_generate_triple_barrier_targets()`

**Why This Matters:**
- `entry_px` values of $217M produce meaningless barrier levels
- ATR calculated from corrupt prices amplifies the problem
- Extreme values skew class weights and corrupt model training

**Symptoms of Missing Validation:**
```
ANOMALIES:
- entry_px: max value 217880880 (suspiciously large)
- top: max value 299464722 (suspiciously large)
```

### 7.4 Target File Output

**Output:** `artifacts/targets_triple_barrier.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Stock symbol |
| `t0` | datetime | Entry date |
| `t_hit` | datetime | Barrier hit date |
| `hit` | int8 | Barrier hit: 1=upper, 0=lower, -1=time |
| `entry_px` | float32 | Entry price (adjclose) |
| `top` | float32 | Upper barrier price |
| `bot` | float32 | Lower barrier price |
| `h_used` | int16 | Trading days until barrier |
| `price_hit` | float32 | Price at barrier hit |
| `ret_from_entry` | float32 | Return from entry to hit |
| `n_overlapping_trajs` | int16 | Overlapping trade count |
| `weight_overlap` | float32 | Overlap uniqueness weight |
| `weight_class_balance` | float32 | Class frequency weight |
| `weight_final` | float32 | Combined training weight |

---

## 8. File Reference

### Source Files

| File | Purpose |
|------|---------|
| `src/pipelines/orchestrator.py` | Main pipeline coordinator |
| `src/features/single_stock.py` | Single-stock feature orchestration |
| `src/features/cross_sectional.py` | Cross-sectional feature computation |
| `src/features/timeframe.py` | D/W/M resampling utilities |
| `src/data/loader.py` | Data loading and caching |

### Output Files

| File | Contents |
|------|----------|
| `artifacts/features_complete.parquet` | ALL computed features (~600+) |
| `artifacts/features_filtered.parquet` | Curated ML-ready features (~250) |
| `artifacts/targets_triple_barrier.parquet` | ML target labels with sample weights |

**Note:** Both feature files are produced on every pipeline run:
- Use `features_complete.parquet` for debugging, exploration, and re-running feature selection
- Use `features_filtered.parquet` for model training and production

### Cache Files

| File | Contents |
|------|----------|
| `cache/stock_data_universe.parquet` | Daily stock OHLCV |
| `cache/stock_data_etf.parquet` | Daily ETF OHLCV (long format) |
| `cache/fred_data.parquet` | FRED macro data (daily/weekly mixed, not resampled) |
| `cache/stock_data_weekly.parquet` | Weekly stock OHLCV (NEW) |
| `cache/etf_data_weekly.parquet` | Weekly ETF OHLCV (NEW) |

---

## 9. Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - High-level ML pipeline
- [CLAUDE.md](../CLAUDE.md) - Developer quick reference
- [PIPELINE_ISSUES.md](../PIPELINE_ISSUES.md) - Known issues tracking

### Archived/Consolidated Docs

The following docs have been consolidated into this document:
- `docs/WEEKLY_MULTIPROCESSING_OPTIMIZATION.md` - Parallelism patterns
- `docs/WEEKLY_MERGE_OPTIMIZATION.md` - Merge strategies
- `docs/WEEKLY_FEATURES_ENHANCEMENT.md` - Error handling
- `docs/XSEC_FRAGMENTATION_FIX.md` - DataFrame patterns
