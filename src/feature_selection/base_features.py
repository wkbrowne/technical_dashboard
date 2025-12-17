"""
Base features for momentum strategy forward selection.

These features represent the recommended starting point for feature selection,
chosen by an expert momentum trader to capture the core signals:
- Trend direction and strength
- Price position relative to moving averages
- Volatility regime
- Relative performance vs market/sector
- Breakout signals

Use these as the initial feature set for forward selection, then expand
via reverse selection, pair interactions, and hill climbing.

TODO: Three most promising feature areas to explore next:

1. LIQUIDITY & MARKET MICROSTRUCTURE
   IMPLEMENTED (src/features/liquidity.py):
   ✓ Spread proxies: hl_spread_proxy, cs_spread_est, roll_spread_est
   ✓ Intraday patterns: overnight_ratio, range_efficiency, upper_shadow_ratio, lower_shadow_ratio
   ✓ VWAP distance: vwap_dist_{5,10,20}d, vwap_dist_{5,10,20}d_zscore
   ✓ Volume-price: volume_direction, rel_volume_{5,10,20}d, volume_trend_10d, pv_divergence_5d
   ✓ Illiquidity: amihud_illiq, amihud_illiq_ratio, zero_vol_pct_20d, illiquidity_score, liquidity_regime

   SELECTED BY MODEL (0.6932 AUC):
   → upper_shadow_ratio (selling pressure signal)
   → vwap_dist_20d_zscore (deviation from volume-weighted price)
   → w_vwap_dist_20d_zscore (weekly version - smoother)

   GAPS TO EXPLORE:
   - Order flow imbalance: up-tick vs down-tick volume approximation (sign of return * volume cumsum)
   - Volume profile concepts: price levels with highest traded volume (support/resistance)
   - Institutional footprint: block trade detection (large volume + small price impact)
   - Intraday momentum: first/last hour patterns (not available in daily data, but could proxy via gaps)
   - Cross-sectional liquidity rank: relative liquidity vs universe (liquidity beta)
   - Liquidity shocks: sudden spread widening or volume spikes as regime change signals

   Why promising: Only 3 liquidity features selected despite rich implementation.
   upper_shadow_ratio captures selling pressure; VWAP distance captures
   deviation from fair value. Missing: flow-based signals and regime transitions.

2. FUNDAMENTALS & FACTOR CHARACTERISTICS
   NOT YET IMPLEMENTED - Requires external data sources

   STATIC FUNDAMENTALS (stock characteristics):
   - Valuation: P/E, P/B, P/S, EV/EBITDA (raw + sector-relative z-scores)
   - Size: Market cap, market cap percentile, small/mid/large classification
   - Quality: ROE, profit margins, debt/equity, earnings stability
   - Growth: Revenue growth, earnings growth, estimate revisions

   DYNAMIC FACTOR LOADINGS (rolling betas):
   - SMB loading: sensitivity to small vs large cap factor
   - HML loading: sensitivity to value vs growth factor
   - MOM loading: sensitivity to momentum factor
   - Sector factor loadings: tech exposure, cyclical exposure, etc.

   Note: Fundamentals are INPUTS, factor loadings are CORRELATIONS.
   Both valuable - fundamentals for stock selection, loadings for
   understanding regime sensitivity and factor rotation timing.

   DATA SOURCES TO EXPLORE:
   - Yahoo Finance fundamentals (quarterly, delayed)
   - Quandl/Sharadar for fundamentals
   - Factor returns from Ken French data library
   - Sector ETF returns for sector factor proxies

   Why promising: Current model relies heavily on price-based features.
   Fundamentals could provide orthogonal information, especially for
   distinguishing quality momentum from junk rallies.

3. SENTIMENT & POSITIONING
   NOT YET IMPLEMENTED - Requires external data sources

   OPTIONS-DERIVED:
   - Put-call ratio (equity + index level)
   - Implied volatility skew (put vs call IV)
   - IV term structure (front month vs back month)
   - Options volume vs open interest changes

   POSITIONING DATA:
   - Short interest ratio (days to cover)
   - Short interest momentum (weekly changes)
   - COT data for related futures (S&P, sector ETFs)
   - ETF flows as sentiment proxy

   ANALYST/INSIDER:
   - Earnings estimate revisions (FY1/FY2 changes)
   - Recommendation changes momentum
   - Insider buy/sell ratio
   - Insider transaction size

   DATA SOURCES TO EXPLORE:
   - CBOE for options data (expensive)
   - FINRA for short interest (bi-monthly, delayed)
   - Quandl for COT data
   - Yahoo Finance for analyst estimates

   Why promising: Positioning extremes precede reversals. High short interest
   + positive momentum = squeeze potential. Put-call extremes signal
   sentiment exhaustion. Currently the model has no sentiment inputs.

================================================================================
FEATURE SELECTION RESULTS SUMMARY (52 features, 0.6932 AUC ± 0.0625)
================================================================================

TOP PERFORMING CATEGORIES (by # of features selected):

1. MACRO/INTERMARKET (12 features) - 23% of model
   Strongest signal: copper_gold_ratio, fred_* series, w_equity_bond_corr_60d
   → Economic regime and credit conditions matter more than expected
   → Weekly macro features (w_fred_*) complement daily well

2. TREND STRENGTH (8 features) - 15% of model
   Core: trend_score_sign, trend_score_slope, trend_persist_ema
   Slopes: pct_slope_ma_20, pct_slope_ma_100, w_pct_slope_ma_20
   → Multi-timeframe slope alignment is key signal

3. RELATIVE PERFORMANCE (7 features) - 13% of model
   Alpha: alpha_mom_spy_20_ema10, alpha_mom_sector_20_ema10, w_alpha_mom_spy_20_ema10
   Beta: beta_spy, w_beta_spy
   Strength: rel_strength_sector, xsec_mom_20d_z
   → Stock vs market/sector differentiation crucial

4. PRICE POSITION (6 features) - 12% of model
   Distance: pct_dist_ma_20, pct_dist_ma_50, w_pct_dist_ma_20, w_pct_dist_ma_100_z
   Range: pos_in_20d_range, w_pos_in_5d_range
   → Mean reversion signals complement momentum

5. VOLATILITY REGIME (6 features) - 12% of model
   Stock: vol_regime, vol_regime_ema10, atr_percent
   Market: vix_regime, w_vix_ma4_ratio, w_vix_vxn_spread
   → Both micro (stock) and macro (VIX) vol matter

6. TREND DIRECTION (4 features) - 8% of model
   Daily: rsi_14
   Weekly: w_rsi_14, w_macd_histogram
   → Weekly momentum filters noise vs daily

7. VOLUME/LIQUIDITY (3 features) - 6% of model
   Selected: upper_shadow_ratio, vwap_dist_20d_zscore, w_vwap_dist_20d_zscore
   → Candlestick patterns and VWAP useful; spread/illiquidity not selected

UNDERPERFORMING CATEGORIES (features available but not selected):
- Most spread proxies (cs_spread_est, roll_spread_est, hl_spread_proxy)
- Amihud illiquidity measures
- Most breakout signals (only breakout_up_20d selected)
- Most relative strength variants (MACD-based RS not selected)
- Cross-sectional percentiles (z-scores preferred over percentiles)

KEY INSIGHTS:
1. Weekly features are critical - 26 of 52 features have w_ prefix (50%)
2. Z-scores preferred over raw values for mean reversion signals
3. Macro regime (FRED, intermarket) unexpectedly important
4. Simple trend indicators (sign, slope) beat complex ones
5. Stock-level vol regime + market vol regime both matter
"""

# =============================================================================
# BASE FEATURES V2 - Curated core features (~38 total)
# =============================================================================
# Streamlined set of high-signal features chosen for:
# - Multi-timeframe coverage (daily + weekly)
# - Diverse signal types (trend, mean reversion, volatility, relative)
# - Low correlation within category
# - Strong feature selection performance

BASE_FEATURES = [
    # === TREND / MOMENTUM (4) ===
    "rsi_14",                    # Daily momentum oscillator
    "w_macd_histogram",          # Weekly momentum trend (filters noise)
    "trend_score_sign",          # Multi-MA alignment (+1/-1 per MA)
    "trend_score_slope",         # Trend slope composite

    # === TREND SLOPES (3) ===
    "pct_slope_ma_20",           # Short-term trend direction
    "pct_slope_ma_100",          # Medium-term trend direction
    "w_pct_slope_ma_50",         # Weekly trend direction

    # === PRICE POSITION / MEAN REVERSION (5) ===
    "pct_dist_ma_20_z",          # Z-scored distance from 20d MA
    "pct_dist_ma_50_z",          # Z-scored distance from 50d MA
    "relative_dist_20_50_z",     # Relative position between MAs
    "pos_in_20d_range",          # Position in recent range (0-1)
    "vwap_dist_20d_zscore",      # Z-scored distance from 20d VWAP

    # === VOLATILITY / REGIME (5) ===
    "atr_percent",               # Normalized volatility (REQUIRED for targets)
    "vol_regime_ema10",          # Smoothed vol regime
    "rv_z_60",                   # Realized vol z-score
    "vix_zscore_60d",            # VIX z-score (market fear)
    "w_vix_vxn_spread",          # Weekly VIX-VXN spread (tech vs broad)

    # === RELATIVE PERFORMANCE / CROSS-SECTION (6) ===
    "alpha_mom_spy_20_ema10",    # Alpha vs market (20d, smoothed)
    "alpha_mom_sector_20_ema10", # Alpha vs sector (20d, smoothed)
    "w_alpha_mom_spy_20_ema10",  # Weekly alpha vs market
    "rel_strength_sector",       # Relative strength vs sector
    "xsec_mom_20d_z",            # Cross-sectional momentum z-score
    "w_xsec_mom_4w_z",           # Weekly cross-sectional z-score

    # === BREADTH (1) ===
    "w_ad_ratio_universe",       # Weekly advance-decline ratio

    # === LIQUIDITY / VOLUME (2) ===
    "upper_shadow_ratio",        # Selling pressure (candlestick)
    "w_volshock_ema",            # Weekly volume shock indicator

    # === MACRO / INTERMARKET (6) ===
    "copper_gold_zscore",        # Copper/gold z-score (growth indicator)
    "gold_spy_ratio_zscore",     # Gold/SPY z-score (risk-off)
    "w_equity_bond_corr_60d",    # Weekly equity-bond correlation
    "w_fred_bamlh0a0hym2_z60",   # High yield spread z-score
    "fred_dgs2_chg20d",          # 2-year Treasury rate change
    "fred_ccsa_z52w",            # Continued claims z-score
]

# Feature categories for reference (updated to match config/features.yaml)
# Categories align with src/config/features.py FeatureCategory enum
FEATURE_CATEGORIES = {
    # === SINGLE-STOCK FEATURES ===
    "trend": [
        "trend_score_sign",
        "trend_score_granular",
        "trend_score_slope",
        "trend_persist_ema",
        "pct_slope_ma_20",
        "pct_slope_ma_100",
        "w_pct_slope_ma_20",
    ],
    "momentum": [
        "rsi_14",
        "w_rsi_14",
        "macd_histogram",
        "w_macd_histogram",
        "macd_hist_deriv_ema3",
    ],
    "volatility": [
        "vol_regime",
        "vol_regime_ema10",
        "atr_percent",
        "rv_z_60",
        "rvol_20",
        "w_rv60_slope_norm",
        "w_rv100_slope_norm",
    ],
    "price_position": [
        "pct_dist_ma_20",
        "pct_dist_ma_50",
        "pct_dist_ma_100_z",
        "w_pct_dist_ma_20",
        "w_pct_dist_ma_100_z",
        "min_pct_dist_ma",
        "relative_dist_20_50_z",
    ],
    "range_breakout": [
        "atr_percent",
        "pos_in_5d_range",
        "pos_in_10d_range",
        "pos_in_20d_range",
        "w_pos_in_5d_range",
        "breakout_up_5d",
        "breakout_up_10d",
        "breakout_up_20d",
        "range_expansion_20d",
    ],
    "volume": [
        "obv_z_60",
        "rdollar_vol_20",
        "volshock_z",
        "volshock_ema",
        "w_volshock_ema",
    ],
    "liquidity": [
        "hl_spread_proxy",
        "cs_spread_est",
        "roll_spread_est",
        "overnight_ratio",
        "range_efficiency",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "vwap_dist_20d_zscore",
        "w_vwap_dist_20d_zscore",
        "amihud_illiq",
        "illiquidity_score",
    ],

    # === CROSS-SECTIONAL FEATURES ===
    "cross_sectional": [
        "vol_regime_cs_median",
        "vol_regime_rel",
        "xsec_mom_5d_z",
        "xsec_mom_20d_z",
        "xsec_mom_60d_z",
        "xsec_pct_20d",
        "w_xsec_mom_4w_z",
    ],
    "alpha": [
        "beta_spy",
        "beta_qqq",
        "beta_sector",
        "w_beta_spy",
        "alpha_mom_spy_20_ema10",
        "alpha_mom_spy_60_ema10",
        "alpha_mom_qqq_20_ema10",
        "alpha_mom_sector_20_ema10",
        "alpha_resid_spy",
        "alpha_qqq_vs_spy",
        "w_alpha_mom_spy_20_ema10",
        # Factor regression
        "beta_market",
        "beta_bestmatch",
        "beta_breadth",
        "residual_cumret",
        "residual_vol",
    ],
    "factor_spreads": [
        # Daily QQQ spread (absolute)
        "qqq_cumret_20", "qqq_cumret_60", "qqq_cumret_120",
        "qqq_zscore_60", "qqq_slope_20", "qqq_slope_60",
        # Daily SPY spread (absolute)
        "spy_cumret_20", "spy_cumret_60", "spy_cumret_120",
        "spy_zscore_60", "spy_slope_20", "spy_slope_60",
        # Daily QQQ-SPY spread (growth premium)
        "qqq_spy_cumret_20", "qqq_spy_cumret_60", "qqq_spy_cumret_120",
        "qqq_spy_zscore_60", "qqq_spy_slope_20", "qqq_spy_slope_60",
        # Daily RSP-SPY spread (breadth/concentration)
        "rsp_spy_cumret_20", "rsp_spy_cumret_60", "rsp_spy_cumret_120",
        "rsp_spy_zscore_60", "rsp_spy_slope_20", "rsp_spy_slope_60",
        # Daily bestmatch-SPY spread (per-symbol sector premium)
        "bestmatch_spy_cumret_20", "bestmatch_spy_cumret_60", "bestmatch_spy_cumret_120",
        "bestmatch_spy_zscore_60", "bestmatch_spy_slope_20", "bestmatch_spy_slope_60",
        # Weekly QQQ spread
        "w_qqq_cumret_4", "w_qqq_cumret_12", "w_qqq_cumret_24",
        "w_qqq_zscore_12", "w_qqq_slope_4", "w_qqq_slope_12",
        # Weekly SPY spread
        "w_spy_cumret_4", "w_spy_cumret_12", "w_spy_cumret_24",
        "w_spy_zscore_12", "w_spy_slope_4", "w_spy_slope_12",
        # Weekly QQQ-SPY spread
        "w_qqq_spy_cumret_4", "w_qqq_spy_cumret_12", "w_qqq_spy_cumret_24",
        "w_qqq_spy_zscore_12", "w_qqq_spy_slope_4", "w_qqq_spy_slope_12",
        # Weekly RSP-SPY spread
        "w_rsp_spy_cumret_4", "w_rsp_spy_cumret_12", "w_rsp_spy_cumret_24",
        "w_rsp_spy_zscore_12", "w_rsp_spy_slope_4", "w_rsp_spy_slope_12",
        # Weekly bestmatch-SPY spread
        "w_bestmatch_spy_cumret_4", "w_bestmatch_spy_cumret_12", "w_bestmatch_spy_cumret_24",
        "w_bestmatch_spy_zscore_12", "w_bestmatch_spy_slope_4", "w_bestmatch_spy_slope_12",
    ],
    "relative_strength": [
        "rel_strength_spy",
        "rel_strength_spy_zscore",
        "rel_strength_qqq",
        "rel_strength_qqq_zscore",
        "rel_strength_sector",
        "rel_strength_sector_zscore",
        "w_rel_strength_spy",
        "w_rel_strength_sector",
    ],
    "breadth": [
        "ad_ratio_ema10",
        "ad_ratio_universe",
        "ad_thrust_10d",
        "mcclellan_oscillator",
        "pct_universe_above_ma20",
        "pct_universe_above_ma50",
        "w_ad_ratio_universe",
    ],

    # === MACRO/INTERMARKET FEATURES ===
    "macro": [
        # VIX regime
        "vix_regime",
        "vix_percentile_252d",
        "vix_zscore_60d",
        "vix_ma20_ratio",
        "vix_vxn_spread",
        "w_vix_ma4_ratio",
        "w_vix_vxn_spread",
        "w_vix_percentile_52w",
        # FRED data
        "fred_dgs10_chg20d",
        "fred_dgs2_chg20d",
        "fred_t10y2y_z60",
        "fred_bamlh0a0hym2_z60",
        "fred_bamlh0a0hym2_pct252",
        "fred_icsa_chg4w",
        "fred_icsa_z52w",
        "fred_ccsa_z52w",
        "fred_nfci_chg4w",
        "w_fred_bamlh0a0hym2_z60",
        "w_fred_icsa_chg4w",
        "w_fred_nfci_chg4w",
    ],
    "intermarket": [
        "copper_gold_ratio",
        "copper_gold_zscore",
        "gold_spy_ratio",
        "gold_spy_ratio_zscore",
        "dollar_momentum_20d",
        "dollar_percentile_252d",
        "oil_momentum_20d",
        "cyclical_defensive_ratio",
        "financials_utilities_ratio",
        "tech_spy_ratio",
        "equity_bond_corr_60d",
        "credit_spread_zscore",
        "yield_curve_zscore",
        "w_copper_gold_ratio",
        "w_gold_spy_ratio",
        "w_gold_spy_ratio_zscore",
        "w_dollar_momentum_20d",
        "w_financials_utilities_ratio",
        "w_equity_bond_corr_60d",
    ],
}

# =============================================================================
# EXPANSION CANDIDATES V2 - Curated features for forward selection (~200 total)
# =============================================================================
# Streamlined feature set excluding intermediate/raw features.
# Only normalized, z-scored, or otherwise ML-ready features.

EXPANSION_CANDIDATES = {
    # --- Momentum (6) ---
    "momentum": [
        "rsi_21",
        "macd_hist_deriv_ema3",
        "w_rsi_14",
        "w_rsi_21",
        "w_macd_hist_deriv_ema3",
        "trend_persist_ema",
    ],

    # --- Trend Shape & Scores (6) ---
    "trend_shape": [
        "trend_score_granular",
        "w_trend_score_sign",
        "w_trend_score_granular",
        "w_trend_persist_ema",
        "quiet_trend",
        "trend_alignment",
    ],

    # --- Trend Slopes (10) ---
    "trend_slopes": [
        "pct_slope_ma_10",
        "pct_slope_ma_30",
        "pct_slope_ma_50",
        "pct_slope_ma_150",
        "pct_slope_ma_200",
        "rv60_slope_norm",
        "w_pct_slope_ma_20",
        "w_pct_slope_ma_100",
        "w_rv60_slope_norm",
        "w_trend_score_slope",
    ],

    # --- Distance to MA (12) ---
    "distance_to_ma": [
        "pct_dist_ma_100",
        "pct_dist_ma_100_z",
        "pct_dist_ma_200",
        "pct_dist_ma_200_z",
        "min_pct_dist_ma",
        "relative_dist_20_50",
        "w_pct_dist_ma_20",
        "w_pct_dist_ma_20_z",
        "w_pct_dist_ma_50_z",
        "w_pct_dist_ma_100_z",
        "w_min_pct_dist_ma",
        "w_relative_dist_20_50_z",
    ],

    # --- Range & Breakout (16) ---
    "range_breakout": [
        "pos_in_5d_range",
        "pos_in_10d_range",
        "breakout_up_5d",
        "breakout_up_10d",
        "breakout_up_20d",
        "breakout_dn_20d",
        "range_expansion_20d",
        "range_z_20d",
        "w_pos_in_5d_range",
        "w_pos_in_10d_range",
        "w_pos_in_20d_range",
        "w_breakout_up_20d",
        "w_breakout_dn_20d",
        "w_range_expansion_20d",
        "w_range_z_20d",
        "gap_atr_ratio",
    ],

    # --- Volatility (14) ---
    "volatility": [
        "vol_regime",
        "rv_ratio_10_60",
        "rv_ratio_20_100",
        "vol_z_20",
        "vol_z_60",
        "rvol_20",
        "vol_regime_cs_median",
        "vol_regime_rel",
        "w_rv_z_60",
        "w_vol_z_60",
        "w_rvol_20",
        "w_vol_regime",
        "w_vol_regime_ema10",
        "w_vol_regime_rel",
    ],

    # --- VIX / Implied Vol (12) ---
    "vix_implied_vol": [
        "vix_percentile_252d",
        "vix_ma20_ratio",
        "vix_vxn_spread",
        "vix_change_5d",
        "vix_change_20d",
        "vix_regime",
        "w_vix_percentile_52w",
        "w_vix_zscore_12w",
        "w_vix_regime",
        "w_vix_ma4_ratio",
        "w_vix_change_4w",
        "w_vxn_percentile_252d",
    ],

    # --- Alpha Momentum (16) ---
    "alpha_momentum": [
        "alpha_mom_spy_60_ema10",
        "alpha_mom_spy_120_ema10",
        "alpha_mom_qqq_20_ema10",
        "alpha_mom_qqq_60_ema10",
        "alpha_mom_sector_60_ema10",
        "alpha_mom_combo_20_ema10",
        "alpha_mom_combo_60_ema10",
        "beta_spy",
        "beta_qqq",
        "beta_sector",
        "w_alpha_mom_spy_60_ema10",
        "w_alpha_mom_qqq_60_ema10",
        "w_alpha_mom_sector_60_ema10",
        "w_alpha_mom_combo_60_ema10",
        "w_beta_spy",
        "w_beta_qqq",
    ],

    # --- Factor Betas (joint regression) (10) ---
    "factor_betas": [
        "beta_market",
        "beta_bestmatch",
        "beta_breadth",
        "residual_cumret",
        "residual_vol",
        "w_beta_market",
        "w_beta_bestmatch",
        "w_beta_breadth",
        "w_residual_cumret",
        "w_residual_vol",
    ],

    # --- Factor Spreads (18) ---
    "factor_spreads": [
        "qqq_spy_cumret_20",
        "qqq_spy_cumret_60",
        "qqq_spy_zscore_60",
        "qqq_spy_slope_20",
        "rsp_spy_cumret_20",
        "rsp_spy_cumret_60",
        "rsp_spy_zscore_60",
        "rsp_spy_slope_20",
        "bestmatch_spy_cumret_60",
        "bestmatch_spy_zscore_60",
        "w_qqq_spy_cumret_12",
        "w_qqq_spy_zscore_12",
        "w_qqq_spy_slope_4",
        "w_rsp_spy_cumret_12",
        "w_rsp_spy_zscore_12",
        "w_rsp_spy_slope_4",
        "w_bestmatch_spy_cumret_12",
        "w_bestmatch_spy_zscore_12",
    ],

    # --- Relative Strength (14) ---
    "relative_strength": [
        "rel_strength_spy",
        "rel_strength_spy_zscore",
        "rel_strength_spy_rsi",
        "rel_strength_qqq",
        "rel_strength_qqq_zscore",
        "rel_strength_sector_zscore",
        "rel_strength_sector_rsi",
        "rel_strength_sector_vs_market",
        "w_rel_strength_spy",
        "w_rel_strength_spy_zscore",
        "w_rel_strength_qqq",
        "w_rel_strength_sector",
        "w_rel_strength_sector_zscore",
        "rel_strength_qqq_spy_spread",
    ],

    # --- Cross-Sectional Momentum (12) ---
    "cross_sectional_momentum": [
        "xsec_mom_5d_z",
        "xsec_mom_60d_z",
        "xsec_mom_5d_sect_neutral_z",
        "xsec_mom_20d_sect_neutral_z",
        "xsec_pct_20d",
        "xsec_pct_60d",
        "w_xsec_mom_1w_z",
        "w_xsec_mom_13w_z",
        "w_xsec_mom_4w_sect_neutral_z",
        "w_xsec_pct_4w",
        "w_xsec_pct_13w",
        "w_xsec_pct_4w_sect",
    ],

    # --- Liquidity (14) ---
    "liquidity": [
        "vwap_dist_5d_zscore",
        "vwap_dist_10d_zscore",
        "lower_shadow_ratio",
        "overnight_ratio",
        "range_efficiency",
        "rel_volume_20d",
        "volume_direction",
        "pv_divergence_5d",
        "amihud_illiq_ratio",
        "illiquidity_score",
        "w_vwap_dist_20d_zscore",
        "w_range_efficiency",
        "w_rel_volume_20d",
        "w_illiquidity_score",
    ],

    # --- Market Breadth (6) ---
    "market_breadth": [
        "ad_ratio_ema10",
        "ad_ratio_universe",
        "mcclellan_oscillator",
        "w_ad_ratio_ema10",
        "w_mcclellan_oscillator",
        "w_ad_thrust_4w",
    ],

    # --- Intermarket Ratios (10) ---
    "intermarket_ratios": [
        "copper_gold_ratio",
        "gold_spy_ratio",
        "cyclical_defensive_ratio",
        "tech_spy_ratio",
        "financials_utilities_ratio",
        "w_copper_gold_ratio",
        "w_gold_spy_ratio",
        "w_cyclical_defensive_ratio",
        "w_tech_spy_ratio",
        "w_financials_utilities_ratio",
    ],

    # --- Macro (FRED) (20) ---
    "macro_fred": [
        "fred_bamlh0a0hym2_z60",
        "fred_bamlh0a0hym2_chg20d",
        "fred_dgs10_chg20d",
        "fred_dgs10_z60",
        "fred_dgs2_chg5d",
        "fred_t10y2y_z60",
        "fred_t10y3m_z60",
        "fred_nfci_chg4w",
        "fred_nfci_z52w",
        "fred_icsa_chg4w",
        "fred_icsa_z52w",
        "fred_ccsa_chg4w",
        "w_fred_bamlh0a0hym2_chg20d",
        "w_fred_dgs10_z60",
        "w_fred_dgs2_chg20d",
        "w_fred_t10y2y_z60",
        "w_fred_nfci_chg4w",
        "w_fred_icsa_chg4w",
        "w_fred_icsa_z52w",
        "w_fred_ccsa_z52w",
    ],

    # --- Volume Analysis (8) ---
    "volume_analysis": [
        "obv_z_60",
        "rdollar_vol_20",
        "volshock_z",
        "volshock_dir",
        "w_obv_z_60",
        "w_rdollar_vol_20",
        "w_volshock_z",
        "w_volshock_dir",
    ],

    # --- Regime & Correlation (8) ---
    "regime_correlation": [
        "credit_spread_zscore",
        "yield_curve_zscore",
        "equity_bond_corr_60d",
        "w_credit_spread_zscore",
        "w_yield_curve_zscore",
        "w_quiet_trend",
        "w_trend_alignment",
        "dollar_momentum_20d",
    ],
}

# =============================================================================
# EXCLUDED FEATURES - Not suitable for ML (raw values, not normalized)
# =============================================================================
# These are intentionally excluded from forward selection:
# - Raw OHLCV prices (not comparable across stocks)
# - Raw moving average values (absolute price levels)
# - Raw ATR (varies by price level)
# - Raw high/low values (use pos_in_range instead)
# - Raw FRED levels (use changes/z-scores instead)
# - Raw VIX/VXN levels (use percentiles/z-scores instead)

EXCLUDED_FEATURES = [
    # Raw OHLCV
    "open", "high", "low", "close", "adjclose", "volume", "ret", "w_ret",
    # Raw moving averages
    "ma_10", "ma_20", "ma_30", "ma_50", "ma_75", "ma_100", "ma_150", "ma_200",
    "w_ma_10", "w_ma_20", "w_ma_30", "w_ma_50", "w_ma_75", "w_ma_100", "w_ma_150", "w_ma_200",
    "w_sma20", "w_sma50",
    # Raw ATR
    "atr14", "w_atr14",
    # Raw high/low values
    "5d_high", "5d_low", "10d_high", "10d_low", "20d_high", "20d_low",
    "w_5d_high", "w_5d_low", "w_10d_high", "w_10d_low", "w_20d_high", "w_20d_low",
    # Raw ranges (use pct_close versions)
    "5d_range", "10d_range", "20d_range", "hl_range", "true_range",
    "w_5d_range", "w_10d_range", "w_20d_range", "w_hl_range", "w_true_range",
    # Raw realized vol
    "rv_10", "rv_20", "rv_60", "rv_100", "vol_ma_20", "vol_ma_50",
    "vol_rolling_20d", "vol_rolling_60d",
    "w_rv_10", "w_rv_20", "w_rv_60", "w_rv_100", "w_vol_ma_20", "w_vol_ma_50",
    "w_vol_rolling_20d", "w_vol_rolling_60d",
    # Raw VIX/VXN levels
    "vix_level", "vix_ema10", "vxn_level",
    "w_vix_level", "w_vix_ema10", "w_vix_ema4", "w_vxn_level",
    # Raw FRED levels
    "fred_bamlc0a4cbbb", "fred_bamlh0a0hym2", "fred_dgs10", "fred_dgs2",
    "fred_t10y2y", "fred_t10y3m", "fred_dfedtaru", "fred_nfci",
    "fred_icsa", "fred_ccsa", "fred_vixcls",
    "w_fred_bamlc0a4cbbb", "w_fred_bamlh0a0hym2", "w_fred_dgs10", "w_fred_dgs2",
    "w_fred_t10y2y", "w_fred_t10y3m", "w_fred_dfedtaru", "w_fred_nfci",
    "w_fred_icsa", "w_fred_ccsa", "w_fred_vixcls",
    # Raw OBV/dollar vol
    "obv", "dollar_vol_ma_20", "w_obv", "w_dollar_vol_ma_20",
    # Raw credit/yield proxies (use z-scores)
    "credit_spread_proxy", "yield_curve_proxy",
    "w_credit_spread_proxy", "w_yield_curve_proxy",
]


def get_base_features():
    """Return the list of base features for forward selection."""
    return BASE_FEATURES.copy()


def get_expansion_candidates(flat=False):
    """
    Return expansion candidate features.

    Args:
        flat: If True, return flat list. If False, return dict by category.

    Returns:
        List or dict of expansion features
    """
    if flat:
        candidates = []
        for category_features in EXPANSION_CANDIDATES.values():
            candidates.extend(category_features)
        return candidates
    return EXPANSION_CANDIDATES.copy()


def get_excluded_features():
    """Return list of features excluded from selection (raw values)."""
    return EXCLUDED_FEATURES.copy()


def validate_features(df, features=None):
    """
    Validate that features exist in DataFrame and report NaN rates.

    Args:
        df: DataFrame with features
        features: List of feature names (default: BASE_FEATURES)

    Returns:
        Dict with 'valid', 'missing', and 'nan_rates' keys
    """
    if features is None:
        features = BASE_FEATURES

    valid = []
    missing = []
    nan_rates = {}

    for feat in features:
        if feat in df.columns:
            valid.append(feat)
            nan_rates[feat] = df[feat].isna().mean() * 100
        else:
            missing.append(feat)

    return {
        "valid": valid,
        "missing": missing,
        "nan_rates": nan_rates,
    }


def get_all_selectable_features():
    """Return all features suitable for selection (base + expansion)."""
    return get_base_features() + get_expansion_candidates(flat=True)


# =============================================================================
# OUTPUT FILTERING - Features to include in pipeline output
# =============================================================================

# Meta columns always included in output
META_COLUMNS = ['symbol', 'date', 'ret']

# Required features (needed for downstream processing)
REQUIRED_FEATURES = ['atr_percent']  # Required by target_generation.py


def get_output_features():
    """
    Get the curated list of features to include in pipeline output.

    Returns a set of feature names that should be kept in the final output.
    This excludes intermediate features (raw MAs, raw ATR, etc.) while
    keeping all normalized/transformed features suitable for ML.

    Returns:
        set: Feature names to include in output
    """
    # Combine base + expansion
    base = get_base_features()
    expansion = get_expansion_candidates(flat=True)

    # Include meta columns and required features
    output_features = set(META_COLUMNS + REQUIRED_FEATURES + base + expansion)

    return output_features


def filter_output_columns(df, keep_all=False):
    """
    Filter DataFrame columns to only include curated output features.

    Args:
        df: DataFrame with computed features
        keep_all: If True, return all columns (no filtering)

    Returns:
        DataFrame with filtered columns
    """
    if keep_all:
        return df

    output_features = get_output_features()

    # Keep columns that are in our output set
    keep_cols = [c for c in df.columns if c in output_features]

    # Log what we're filtering
    filtered_count = len(df.columns) - len(keep_cols)
    if filtered_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Filtered {filtered_count} intermediate columns, keeping {len(keep_cols)}")

    return df[keep_cols]
