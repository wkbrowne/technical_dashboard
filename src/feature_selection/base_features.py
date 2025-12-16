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

# Core momentum features for forward selection starting point
# Updated based on feature selection results (52 features, 0.6932 AUC)
BASE_FEATURES = [
    # === TREND DIRECTION (3) ===
    "rsi_14",                    # Daily momentum oscillator
    "w_rsi_14",                  # Weekly momentum (slower, filters noise)
    "w_macd_histogram",          # Weekly momentum trend

    # === TREND STRENGTH (8) ===
    "trend_score_sign",          # Multi-MA alignment (+1/-1 per MA)
    "trend_score_slope",         # Trend slope composite
    "trend_persist_ema",         # Consecutive up/down days
    "pct_slope_ma_20",           # Short-term trend direction
    "pct_slope_ma_100",          # Medium-term trend direction
    "w_pct_slope_ma_20",         # Weekly trend direction
    "w_rv60_slope_norm",         # Realized vol slope (60d)
    "w_rv100_slope_norm",        # Realized vol slope (100d)

    # === PRICE POSITION (8) ===
    "pct_dist_ma_20",            # Mean reversion signal
    "pct_dist_ma_50",            # Trend distance
    "w_pct_dist_ma_20",          # Weekly mean reversion
    "w_pct_dist_ma_100_z",       # Weekly distance to 100d MA z-score
    "pos_in_20d_range",          # Position in recent range (0-1)
    "w_pos_in_5d_range",         # Weekly position in 5d range
    "vwap_dist_20d_zscore",      # Z-scored distance from 20d VWAP
    "w_vwap_dist_20d_zscore",    # Weekly VWAP distance (smoother)

    # === VOLATILITY REGIME (6) ===
    "vol_regime",                # Stock-specific vol regime
    "vol_regime_ema10",          # Smoothed vol regime
    "atr_percent",               # Normalized volatility
    "vix_regime",                # Market-wide fear/greed
    "w_vix_ma4_ratio",           # VIX vs 4-week MA ratio
    "w_vix_vxn_spread",          # VIX-VXN spread (equity vs tech vol)
    "w_alpha_mom_qqq_spread_60_ema10",  # Weekly QQQ-SPY alpha spread (growth vs value)

    # === RELATIVE PERFORMANCE (7) ===
    "alpha_mom_spy_20_ema10",    # Alpha vs market
    "alpha_mom_sector_20_ema10", # Alpha vs sector
    "beta_spy",                  # Market beta (CAPM)
    "rel_strength_sector",       # RS vs sector
    "xsec_mom_20d_z",            # Cross-sectional rank
    "w_alpha_mom_spy_20_ema10",  # Weekly alpha
    "w_beta_spy",                # Weekly market beta

    # === MACRO/INTERMARKET (15) ===
    "copper_gold_ratio",         # Economic growth indicator
    "copper_gold_zscore",        # Copper/gold z-score
    "gold_spy_ratio",            # Gold vs SPY (risk-off indicator)
    "fred_ccsa_z52w",            # Continued claims z-score (labor market)
    "fred_dgs2_chg20d",          # 2-year Treasury rate change
    "fred_icsa_chg4w",           # Initial claims 4-week change
    "w_copper_gold_ratio",       # Weekly copper/gold
    "w_gold_spy_ratio",          # Weekly gold/SPY
    "w_gold_spy_ratio_zscore",   # Weekly gold/SPY z-score
    "w_dollar_momentum_20d",     # Dollar momentum
    "w_equity_bond_corr_60d",    # Stock-bond correlation
    "w_financials_utilities_ratio", # Risk-on/off sector ratio
    "w_fred_bamlh0a0hym2_pct252", # High yield spread percentile
    "w_fred_bamlh0a0hym2_z60",   # High yield spread z-score
    "w_fred_icsa_chg4w",         # Weekly initial claims change
    "w_fred_nfci_chg4w",         # Financial conditions change

    # === MARKET BREADTH (1) ===
    "w_ad_ratio_universe",       # Advance-decline ratio

    # === VOLUME/LIQUIDITY (2) ===
    "upper_shadow_ratio",        # Selling pressure candlestick
    "w_volshock_ema",            # Volume shock indicator

    # === BREAKOUT (1) ===
    "breakout_up_20d",           # 20-day high breakout
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
        "residual_cumret",
        "residual_vol",
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
# EXPANSION CANDIDATES - Features for forward selection
# =============================================================================
# These are properly normalized/transformed features suitable for ML.
# Excludes: raw prices, raw MAs, raw ATR, raw OHLCV, raw levels

EXPANSION_CANDIDATES = {
    # --- RSI variants (different lookbacks) ---
    "rsi": [
        "rsi_21",
        "rsi_30",
        "vxx_rsi_14",
        "w_rsi_21",
        "w_rsi_30",
        "w_vxx_rsi_14",
    ],

    # --- MACD derivatives ---
    "macd": [
        "macd_hist_deriv_ema3",       # Momentum of momentum
        "w_macd_hist_deriv_ema3",
    ],

    # --- Trend slopes (all normalized as pct change) ---
    "trend_slopes": [
        "pct_slope_ma_10",
        "pct_slope_ma_30",
        "pct_slope_ma_50",
        "pct_slope_ma_75",
        "pct_slope_ma_100",
        "pct_slope_ma_150",
        "pct_slope_ma_200",
        "rv60_slope_norm",
        "rv100_slope_norm",
        "w_pct_slope_ma_10",
        "w_pct_slope_ma_30",
        "w_pct_slope_ma_50",
        "w_pct_slope_ma_75",
        "w_pct_slope_ma_100",
        "w_pct_slope_ma_150",
        "w_pct_slope_ma_200",
        "w_rv60_slope_norm",
        "w_rv100_slope_norm",
        "w_sma20_slope",
        "w_trend_score_slope",
    ],

    # --- Trend scores ---
    "trend_scores": [
        "trend_score_granular",
        "w_trend_score_granular",
        "w_trend_score_sign",
        "w_trend_persist_ema",
    ],

    # --- Price vs MA binary signals ---
    "price_vs_ma_binary": [
        "sign_ma_10",
        "sign_ma_20",
        "sign_ma_30",
        "sign_ma_50",
        "sign_ma_75",
        "sign_ma_100",
        "sign_ma_150",
        "sign_ma_200",
        "w_sign_ma_10",
        "w_sign_ma_20",
        "w_sign_ma_30",
        "w_sign_ma_50",
        "w_sign_ma_75",
        "w_sign_ma_100",
        "w_sign_ma_150",
        "w_sign_ma_200",
    ],

    # --- Distance to MA (normalized + z-scores) ---
    "distance_to_ma": [
        "pct_dist_ma_20_z",
        "pct_dist_ma_50_z",
        "pct_dist_ma_100",
        "pct_dist_ma_100_z",
        "pct_dist_ma_200",
        "pct_dist_ma_200_z",
        "min_pct_dist_ma",
        "relative_dist_20_50",
        "relative_dist_20_50_z",
        "w_pct_dist_ma_50",
        "w_pct_dist_ma_20_z",
        "w_pct_dist_ma_50_z",
        "w_pct_dist_ma_100",
        "w_pct_dist_ma_100_z",
        "w_pct_dist_ma_200",
        "w_pct_dist_ma_200_z",
        "w_min_pct_dist_ma",
        "w_dist_sma20",
        "w_dist_sma50",
        "w_relative_dist_20_50",
        "w_relative_dist_20_50_z",
    ],

    # --- ATR (normalized only) ---
    "atr": [
        "gap_atr_ratio",
        "w_atr_percent",
        "w_gap_atr_ratio",
    ],

    # --- Range & Breakout (normalized) ---
    "range_breakout": [
        # Ranges as pct of close
        "5d_range_pct_close",
        "10d_range_pct_close",
        "20d_range_pct_close",
        "hl_range_pct_close",
        "tr_pct_close",
        # Breakout signals
        "breakout_up_5d",
        "breakout_up_10d",
        "breakout_dn_5d",
        "breakout_dn_10d",
        "breakout_dn_20d",
        # Position in range (0-1)
        "pos_in_5d_range",
        "pos_in_10d_range",
        # Range expansion/contraction
        "range_expansion_5d",
        "range_expansion_10d",
        "range_expansion_20d",
        "range_x_rvol20",
        # Range z-scores
        "range_z_5d",
        "range_z_10d",
        "range_z_20d",
        # Weekly variants
        "w_5d_range_pct_close",
        "w_10d_range_pct_close",
        "w_20d_range_pct_close",
        "w_hl_range_pct_close",
        "w_tr_pct_close",
        "w_breakout_up_5d",
        "w_breakout_up_10d",
        "w_breakout_up_20d",
        "w_breakout_dn_5d",
        "w_breakout_dn_10d",
        "w_breakout_dn_20d",
        "w_pos_in_5d_range",
        "w_pos_in_10d_range",
        "w_pos_in_20d_range",
        "w_range_expansion_5d",
        "w_range_expansion_10d",
        "w_range_expansion_20d",
        "w_range_x_rvol20",
        "w_range_z_5d",
        "w_range_z_10d",
        "w_range_z_20d",
    ],

    # --- Gap analysis ---
    "gap_analysis": [
        "gap_pct",
        "w_gap_pct",
    ],

    # --- Realized volatility (normalized/relative) ---
    "realized_volatility": [
        # Ratios
        "rv_ratio_10_60",
        "rv_ratio_20_100",
        # Z-scores
        "rv_z_60",
        "vol_z_20",
        "vol_z_60",
        # Relative measures
        "rvol_20",
        "rvol_50",
        "vol_regime_cs_median",
        "vol_regime_rel",
        "vol_of_vol_20d",
        # Weekly
        "w_rv_ratio_10_60",
        "w_rv_ratio_20_100",
        "w_rv_z_60",
        "w_vol_z_20",
        "w_vol_z_60",
        "w_rvol_20",
        "w_rvol_50",
        "w_vol_regime",
        "w_vol_regime_cs_median",
        "w_vol_regime_ema10",
        "w_vol_regime_rel",
        "w_vol_of_vol_20d",
    ],

    # --- VIX/Implied vol (normalized only) ---
    "vix_implied_vol": [
        # Percentiles and z-scores
        "vix_percentile_252d",
        "vix_zscore_60d",
        "vxn_percentile_252d",
        # Ratios
        "vix_ma20_ratio",
        "vix_vxn_ratio",
        "vix_vxn_spread",
        # Changes
        "vix_change_5d",
        "vix_change_20d",
        # VXX signals
        "vxx_ret_5d",
        "vxx_ret_20d",
        # Weekly
        "w_vix_percentile_252d",
        "w_vix_percentile_52w",
        "w_vix_zscore_60d",
        "w_vix_zscore_12w",
        "w_vix_regime",
        "w_vix_ma20_ratio",
        "w_vix_ma4_ratio",
        "w_vix_vxn_ratio",
        "w_vix_vxn_spread",
        "w_vix_change_5d",
        "w_vix_change_20d",
        "w_vix_change_1w",
        "w_vix_change_4w",
        "w_vxn_percentile_252d",
        "w_vxx_ret_5d",
        "w_vxx_ret_20d",
    ],

    # --- Alpha momentum & Beta ---
    # NOTE: Weekly alpha_mom features (w_alpha_*) are NOT implemented except specific BASE_FEATURES.
    # Alpha is computed on daily data; weekly alpha needs separate cross-sectional computation.
    "alpha_momentum": [
        # Market beta (CAPM) - daily only implemented
        "beta_spy",
        "beta_qqq",
        "beta_sector",
        # SPY alpha (different lookbacks - all implemented)
        "alpha_mom_spy_ema10",
        "alpha_mom_spy_20_ema10",
        "alpha_mom_spy_60_ema10",
        "alpha_mom_spy_120_ema10",
        # QQQ alpha (all implemented)
        "alpha_mom_qqq_ema10",
        "alpha_mom_qqq_20_ema10",
        "alpha_mom_qqq_60_ema10",
        "alpha_mom_qqq_120_ema10",
        # QQQ vs SPY spread (all implemented)
        "alpha_mom_qqq_spread_ema10",
        "alpha_mom_qqq_spread_20_ema10",
        "alpha_mom_qqq_spread_60_ema10",
        "alpha_mom_qqq_spread_120_ema10",
        # Sector alpha (all implemented)
        "alpha_mom_sector_ema10",
        "alpha_mom_sector_20_ema10",
        "alpha_mom_sector_60_ema10",
        "alpha_mom_sector_120_ema10",
        # Combo alpha (all implemented)
        "alpha_mom_combo_ema10",
        "alpha_mom_combo_20_ema10",
        "alpha_mom_combo_60_ema10",
        "alpha_mom_combo_120_ema10",
        # Weekly (only specific BASE_FEATURES implemented)
        "w_alpha_mom_spy_20_ema10",
        "w_alpha_mom_qqq_spread_60_ema10",
    ],

    # --- Joint Factor Regression (from factor_regression.py) ---
    # NOTE: Weekly joint_factor features (w_*) are NOT fully implemented.
    # Factor regression is computed on daily data; weekly needs separate implementation.
    "joint_factor": [
        # Daily factor betas (from joint regression - implemented)
        "beta_market",           # SPY beta (joint regression)
        "beta_bestmatch",        # Best-match sector/subsector ETF beta
        "beta_breadth",          # RSP-SPY spread beta (concentration factor)
        # Daily residual statistics (implemented)
        "residual_cumret",       # Cumulative residual (alpha signal)
        "residual_vol",          # Idiosyncratic volatility
        # Weekly (only w_beta_breadth implemented via weekly cross-sectional)
        "w_beta_breadth",        # Weekly breadth beta
    ],

    # --- Relative strength ---
    # NOTE: Weekly relative_strength features (w_rel_strength_*) are NOT implemented.
    # RS is a cross-sectional feature computed on daily data only.
    # The weekly resampling doesn't currently support cross-sectional recomputation.
    "relative_strength": [
        # vs SPY (all implemented)
        "rel_strength_spy",
        "rel_strength_spy_norm",
        "rel_strength_spy_zscore",
        "rel_strength_spy_rsi",
        "rel_strength_spy_macd",
        "rel_strength_spy_macd_hist",
        "rel_strength_spy_macd_signal",
        # vs QQQ (growth/tech benchmark - all implemented)
        "rel_strength_qqq",
        "rel_strength_qqq_norm",
        "rel_strength_qqq_zscore",
        "rel_strength_qqq_rsi",
        "rel_strength_qqq_macd",
        "rel_strength_qqq_macd_hist",
        "rel_strength_qqq_macd_signal",
        # QQQ vs SPY spread (implemented)
        "rel_strength_qqq_spy_spread",
        "rel_strength_qqq_spy_spread_norm",
        # vs RSP (equal weight - all implemented)
        "rel_strength_rsp",
        "rel_strength_rsp_norm",
        "rel_strength_rsp_zscore",
        "rel_strength_rsp_rsi",
        "rel_strength_rsp_macd",
        "rel_strength_rsp_macd_hist",
        # vs Sector (all implemented)
        "rel_strength_sector",
        "rel_strength_sector_norm",
        "rel_strength_sector_zscore",
        "rel_strength_sector_rsi",
        "rel_strength_sector_macd",
        "rel_strength_sector_macd_hist",
        "rel_strength_sector_macd_signal",
        "rel_strength_sector_vs_market",
        "rel_strength_sector_vs_market_norm",
        # vs Sector EW (Equal-Weighted - all implemented)
        "rel_strength_sector_ew",
        "rel_strength_sector_ew_norm",
        "rel_strength_sector_ew_zscore",
        "rel_strength_sector_ew_rsi",
        "rel_strength_sector_ew_macd",
        "rel_strength_sector_ew_macd_hist",
        # NOTE: Subsector features REMOVED (had 40-48% NaN due to unstable correlation-based discovery)
    ],

    # --- Cross-sectional momentum ---
    # NOTE: Weekly xsec features ARE implemented via weekly cross-sectional computation
    "cross_sectional_momentum": [
        # Daily z-scores (all implemented)
        "xsec_mom_5d_z",
        "xsec_mom_5d_sect_neutral_z",
        "xsec_mom_20d_sect_neutral_z",
        "xsec_mom_60d_z",
        "xsec_mom_60d_sect_neutral_z",
        # Daily percentiles (all implemented)
        "xsec_pct_5d",
        "xsec_pct_5d_sect",
        "xsec_pct_20d",
        "xsec_pct_20d_sect",
        "xsec_pct_60d",
        "xsec_pct_60d_sect",
        # Weekly z-scores (implemented via weekly cross-sectional stage)
        "w_xsec_mom_1w_z",
        "w_xsec_mom_1w_sect_neutral_z",
        "w_xsec_mom_4w_z",
        "w_xsec_mom_4w_sect_neutral_z",
        "w_xsec_mom_13w_z",
        "w_xsec_mom_13w_sect_neutral_z",
        # Weekly percentiles (implemented)
        "w_xsec_pct_1w",
        "w_xsec_pct_1w_sect",
        "w_xsec_pct_4w",
        "w_xsec_pct_4w_sect",
        "w_xsec_pct_13w",
        "w_xsec_pct_13w_sect",
    ],

    # --- Liquidity & Microstructure ---
    "liquidity": [
        # Spread proxies
        "hl_spread_proxy",
        "cs_spread_est",
        "roll_spread_est",
        # Intraday patterns
        "overnight_ratio",
        "range_efficiency",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        # VWAP distance
        "vwap_dist_5d",
        "vwap_dist_10d",
        "vwap_dist_20d",
        "vwap_dist_5d_zscore",
        "vwap_dist_10d_zscore",
        "vwap_dist_20d_zscore",
        # Volume-price relationships
        "volume_direction",
        "rel_volume_5d",
        "rel_volume_10d",
        "rel_volume_20d",
        "volume_trend_10d",
        "pv_divergence_5d",
        # Illiquidity measures
        "amihud_illiq",
        "amihud_illiq_ratio",
        "zero_vol_pct_20d",
        # Composite
        "illiquidity_score",
        "liquidity_regime",
        # Weekly
        "w_hl_spread_proxy",
        "w_cs_spread_est",
        "w_roll_spread_est",
        "w_overnight_ratio",
        "w_range_efficiency",
        "w_vwap_dist_5d",
        "w_vwap_dist_10d",
        "w_vwap_dist_20d",
        "w_vwap_dist_5d_zscore",
        "w_vwap_dist_10d_zscore",
        "w_vwap_dist_20d_zscore",
        "w_rel_volume_5d",
        "w_rel_volume_10d",
        "w_rel_volume_20d",
        "w_amihud_illiq",
        "w_amihud_illiq_ratio",
        "w_illiquidity_score",
        "w_liquidity_regime",
    ],

    # --- Market breadth ---
    # NOTE: Weekly breadth features (w_*) are NOT implemented except w_ad_ratio_universe.
    # Breadth is computed on daily data; weekly breadth needs separate implementation.
    "market_breadth": [
        # Daily breadth (implemented)
        "ad_ratio_ema10",
        "ad_ratio_universe",
        "mcclellan_oscillator",
        # Weekly breadth (only w_ad_ratio_universe implemented)
        "w_ad_ratio_universe",
    ],

    # --- Intermarket ratios ---
    "intermarket_ratios": [
        "cyclical_defensive_ratio",
        "financials_utilities_ratio",
        "gold_spy_ratio",
        "gold_spy_ratio_zscore",
        "tech_spy_ratio",
        "w_copper_gold_ratio",
        "w_copper_gold_zscore",
        "w_cyclical_defensive_ratio",
        "w_financials_utilities_ratio",
        "w_gold_spy_ratio",
        "w_gold_spy_ratio_zscore",
        "w_tech_spy_ratio",
    ],

    # --- Macro (FRED) - changes and z-scores only ---
    "macro_fred": [
        # Credit spreads
        "fred_bamlc0a4cbbb_chg5d",
        "fred_bamlc0a4cbbb_z60",
        "fred_bamlh0a0hym2_chg5d",
        "fred_bamlh0a0hym2_chg20d",
        "fred_bamlh0a0hym2_pct252",
        "fred_bamlh0a0hym2_z60",
        # Treasury rates
        "fred_dgs10_chg5d",
        "fred_dgs10_chg20d",
        "fred_dgs10_pct252",
        "fred_dgs10_z60",
        "fred_dgs2_chg5d",
        "fred_dgs2_chg20d",
        # Yield curve
        "fred_t10y2y_chg5d",
        "fred_t10y2y_pct252",
        "fred_t10y2y_z60",
        "fred_t10y3m_z60",
        # Fed funds
        "fred_dfedtaru_chg20d",
        # Financial conditions
        "fred_nfci_chg4w",
        "fred_nfci_z52w",
        # Jobless claims
        "fred_icsa_chg4w",
        "fred_icsa_pct104w",
        "fred_icsa_z52w",
        "fred_ccsa_chg4w",
        "fred_ccsa_z52w",
        # VIX from FRED
        "fred_vixcls_chg5d",
        "fred_vixcls_pct252",
        "fred_vixcls_z60",
        # Weekly variants
        "w_fred_bamlc0a4cbbb_chg5d",
        "w_fred_bamlc0a4cbbb_z60",
        "w_fred_bamlh0a0hym2_chg5d",
        "w_fred_bamlh0a0hym2_chg20d",
        "w_fred_bamlh0a0hym2_pct252",
        "w_fred_bamlh0a0hym2_z60",
        "w_fred_dgs10_chg5d",
        "w_fred_dgs10_chg20d",
        "w_fred_dgs10_pct252",
        "w_fred_dgs10_z60",
        "w_fred_dgs2_chg5d",
        "w_fred_dgs2_chg20d",
        "w_fred_t10y2y_chg5d",
        "w_fred_t10y2y_pct252",
        "w_fred_t10y2y_z60",
        "w_fred_t10y3m_z60",
        "w_fred_dfedtaru_chg20d",
        "w_fred_nfci_chg4w",
        "w_fred_nfci_z52w",
        "w_fred_icsa_chg4w",
        "w_fred_icsa_pct104w",
        "w_fred_icsa_z52w",
        "w_fred_ccsa_chg4w",
        "w_fred_ccsa_z52w",
        "w_fred_vixcls_chg5d",
        "w_fred_vixcls_pct252",
        "w_fred_vixcls_z60",
    ],

    # --- Volume analysis (normalized) ---
    "volume_analysis": [
        "obv_z_60",
        "rdollar_vol_20",
        "volshock_z",
        "volshock_ema",
        "volshock_dir",
        "w_obv_z_60",
        "w_rdollar_vol_20",
        "w_volshock_z",
        "w_volshock_ema",
        "w_volshock_dir",
    ],

    # --- Regime & correlation ---
    "regime_correlation": [
        "credit_spread_zscore",
        "yield_curve_zscore",
        "equity_bond_corr_60d",
        "quiet_trend",
        "trend_alignment",
        "w_credit_spread_zscore",
        "w_yield_curve_zscore",
        "w_equity_bond_corr_60d",
        "w_quiet_trend",
        "w_trend_alignment",
    ],

    # --- Dollar/Oil momentum ---
    "dollar_oil": [
        "dollar_momentum_20d",
        "dollar_percentile_252d",
        "oil_momentum_20d",
        "w_dollar_momentum_20d",
        "w_dollar_percentile_252d",
        "w_oil_momentum_20d",
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
