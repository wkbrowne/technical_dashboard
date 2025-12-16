# Pipeline Issues Log

Generated: 2025-12-14
Pipeline Run: Full feature computation (D,W timeframes)
Data: 3045 symbols, 4,144,245 rows, 617 columns
Time: 1049.2s (~17.5 minutes)

## Summary

| Category | Count |
|----------|-------|
| Total features | 607 |
| SEVERE NaN (>50%) | 0 |
| HIGH NaN (20-50%) | 65 |
| MODERATE NaN (5-20%) | 445 |
| LOW NaN (0-5%) | 36 |
| Zero NaN | 61 |
| QQQ features | 40 |
| BASE_FEATURES found | 52/52 |

---

## 1. PERFORMANCE WARNINGS (HIGH PRIORITY)

### 1.1 DataFrame Fragmentation (Major Issue)
**Files affected:**
- `src/data/fred.py:336` - 114,119 warnings
- `src/features/xsec.py:198` - 30,576 warnings
- `src/features/xsec.py:179` - 28,730 warnings
- `src/features/breadth.py:141` - 6,690 warnings
- `src/pipelines/orchestrator.py:580` - 3,045 warnings
- `src/features/timeframe.py:221,223,227` - 3,045 warnings each

**Root cause:** Repeated `df[col] = value` assignments cause memory fragmentation.

**Fix:** Use `pd.concat(axis=1)` to join columns at once, or `df = df.copy()` to defragment.

**Priority:** HIGH - affects memory usage and speed

### 1.2 FutureWarning: pct_change fill_method deprecated
**Files affected:**
- `src/features/liquidity.py:138,139,148` - 3,045 warnings each
- `src/features/sector_mapping.py:298,305,325,413,417,429,441,453,465` - 77-769 warnings each

**Fix:** Add `fill_method=None` to all `pct_change()` calls:
```python
# Before
df['ret'] = df['close'].pct_change()

# After
df['ret'] = df['close'].pct_change(fill_method=None)
```

**Priority:** MEDIUM - will break in future pandas versions

### 1.3 RuntimeWarning: divide by zero / invalid value in log
**Files affected:** pandas arraylike operations

**Count:** 18 divide by zero, 8 invalid value warnings

**Fix:** Add protection before log operations:
```python
# Before
np.log(prices)

# After
np.log(prices.clip(lower=1e-8))
```

**Priority:** LOW - produces NaN but doesn't crash

---

## 2. NaN ISSUES

### 2.1 BASE_FEATURES with >15% NaN (CRITICAL)
These are in BASE_FEATURES but have high NaN rates:

| Feature | NaN Rate | Issue |
|---------|----------|-------|
| rel_strength_sector | 25.4% | Missing sector ETF data |
| alpha_mom_sector_20_ema10 | 23.4% | Missing sector ETF data |

**Root cause:** ~25% of stocks don't have sector ETF mapping or sector ETF data is missing.

**Fix options:**
1. Replace missing sector features with SPY-relative features (fallback)
2. Remove these from BASE_FEATURES
3. Improve sector ETF coverage in sector_mapping.py

### 2.2 HIGH NaN Features (20-50%) - 65 features
Most are sector/subsector relative strength features:

**Subsector features (45-48% NaN):**
- rel_strength_subsector_* - Subsector ETF coverage is poor

**Equal-weight sector features (40-44% NaN):**
- rel_strength_sector_ew_* - Equal-weight ETF coverage is limited

**Sector features (22-27% NaN):**
- rel_strength_sector_*, alpha_*_sector_*, beta_sector

**Fix:** These are expected - not all stocks have subsector/EW sector ETF mappings. Consider:
1. Documenting expected NaN rates
2. Removing rarely-available features from EXPANSION_CANDIDATES
3. Adding fallback logic to use SPY when sector unavailable

### 2.3 QQQ Features (13-15% NaN)
All 40 QQQ features have 13-15% NaN, which is acceptable (matches other per-symbol features).

**Features working correctly:**
- alpha_resid_qqq: 15.2%
- alpha_qqq_vs_spy: 15.2%
- rel_strength_qqq_*: 13-15%
- alpha_mom_qqq_*: 13-14%

---

## 3. RECOMMENDED FIXES

### Priority 1: Performance (do first)
1. **fred.py:336** - Refactor to use pd.concat instead of column-by-column assignment
2. **xsec.py:179,198** - Same refactor
3. **breadth.py:141** - Same refactor
4. **timeframe.py:221-227** - Build dict of columns, concat at end
5. **orchestrator.py:580** - Add `.copy()` after heavy modification

### Priority 2: FutureWarning fixes
1. **liquidity.py** - Add `fill_method=None` to pct_change calls (lines 138, 139, 148)
2. **sector_mapping.py** - Add `fill_method=None` to all pct_change calls

### Priority 3: BASE_FEATURES NaN
Options:
1. **Option A:** Keep rel_strength_sector and alpha_mom_sector_20_ema10, accept ~25% NaN
2. **Option B:** Replace with fallback features:
   - rel_strength_sector -> rel_strength_spy (if sector missing)
   - alpha_mom_sector_20_ema10 -> alpha_mom_spy_20_ema10 (if sector missing)
3. **Option C:** Remove from BASE_FEATURES, keep in EXPANSION_CANDIDATES

**Recommendation:** Option A is fine for now. The feature selection pipeline handles NaN gracefully.

---

## 4. FILES TO MODIFY

### High Priority
- [ ] `src/data/fred.py` - Fix fragmentation (line 336)
- [ ] `src/features/xsec.py` - Fix fragmentation (lines 179, 198)
- [ ] `src/features/breadth.py` - Fix fragmentation (line 141)
- [ ] `src/features/timeframe.py` - Fix fragmentation (lines 221-227)
- [ ] `src/pipelines/orchestrator.py` - Add defragmentation (line 580)

### Medium Priority
- [ ] `src/features/liquidity.py` - FutureWarning fix (lines 138, 139, 148)
- [ ] `src/features/sector_mapping.py` - FutureWarning fix (multiple lines)

### Low Priority
- [ ] Consider removing subsector/EW features from EXPANSION_CANDIDATES due to low coverage

---

## 5. PIPELINE METRICS

**Run Statistics:**
- Symbols processed: 3045
- Total rows: 4,144,245
- Daily features: 299
- Weekly features: 316
- Total features: 615
- Targets generated: 1,247,073
- Total time: 1049.2s

**Target Distribution:**
- Lower barrier hit: 633,791 (50.8%)
- Time expired: 402,787 (32.3%)
- Upper barrier hit: 210,495 (16.9%)

**Memory/Performance:**
- Heavy DataFrame fragmentation warnings (~200k total)
- Parallel processing working (30 workers)
- NaN interpolation filled 12,699,653 values
