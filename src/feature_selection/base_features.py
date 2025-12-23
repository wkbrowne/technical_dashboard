"""
Model-aware feature registry for the 4-model momentum strategy.

This module provides a structured feature registry supporting different feature sets
per model key (LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC).

Architecture:
- CORE_FEATURES: Shared backbone features across all models (from BASE_FEATURES V4)
- HEAD_FEATURES: Model-specific additive features per model_key
- EXPANSION_CANDIDATES: Additional features for selection experiments

Feature Set Retrieval:
    from src.config.model_keys import ModelKey
    from src.feature_selection.base_features import get_featureset, get_core_features

    # Get features for a specific model
    features = get_featureset(ModelKey.LONG_NORMAL)  # CORE + HEAD[LONG_NORMAL]

    # Get all selectable features for a model
    all_features = get_all_selectable_features(ModelKey.LONG_NORMAL)

Backwards Compatibility:
    - BASE_FEATURES remains as alias to get_featureset(LONG_NORMAL)
    - get_base_features() returns BASE_FEATURES (deprecated, use get_featureset)
    - All existing filter_output_columns behavior preserved

Phase 2/3 will add:
- Separate model training per model_key
- Sizing integration based on model predictions
"""

from typing import Dict, List, Optional, Set, Union
import warnings

# Import ModelKey - handle both direct and late imports
try:
    from ..config.model_keys import ModelKey, DEFAULT_MODEL_KEY
except ImportError:
    # Fallback for scripts that import base_features directly
    from src.config.model_keys import ModelKey, DEFAULT_MODEL_KEY


# =============================================================================
# CORE FEATURES - Shared backbone across all models (50 features)
# =============================================================================
# Auto-generated from multi-model selection: 2025-12-23 15:51:09 UTC
# Run signature: 9cee9ff7f14c
# These features represent the intersection of all 4 model selections.

CORE_FEATURES: List[str] = [
    # === RELATIVE PERFORMANCE / ALPHA (8 features) ===
    "alpha_mom_qqq_20_ema10",
    "alpha_mom_sector_20_ema10",
    "rel_strength_sector",
    "w_alpha_mom_qqq_60_ema10",
    "w_alpha_mom_spy_20_ema10",
    "w_rel_strength_sector",
    "w_xsec_mom_4w_z",
    "xsec_mom_20d_z",

    # === MACRO / INTERMARKET (12 features) ===
    "copper_gold_zscore",
    "fred_bamlh0a0hym2_z60",
    "fred_ccsa_z52w",
    "fred_t10y2y_z60",
    "gold_spy_ratio_zscore",
    "qqq_spy_cumret_20",
    "qqq_spy_cumret_60",
    "qqq_spy_slope_20",
    "vix_percentile_252d",
    "vix_zscore_60d",
    "w_equity_bond_corr_60d",
    "w_fred_icsa_z52w",

    # === TREND STRENGTH (5 features) ===
    "pct_slope_ma_100",
    "pct_slope_ma_20",
    "trend_score_sign",
    "trend_score_slope",
    "w_macd_histogram",

    # === PRICE POSITION / MEAN REVERSION (5 features) ===
    "days_since_high_20d_norm",
    "pct_dist_ma_20_z",
    "pct_dist_ma_50_z",
    "pos_in_20d_range",
    "relative_dist_20_50_z",

    # === VOLATILITY / REGIME (7 features) ===
    "atr_percent",
    "bb_width_20_2",
    "days_in_squeeze_20",
    "gap_atr_ratio_raw",
    "rv_z_60",
    "squeeze_intensity_20",
    "vol_regime_ema10",

    # === SECTOR BREADTH (3 features) ===
    "sector_breadth_ad_line",
    "sector_breadth_mcclellan_osc",
    "w_sector_breadth_mcclellan_osc",

    # === VOLUME / LIQUIDITY (5 features) ===
    "lower_shadow_ratio",
    "pv_divergence_5d",
    "vwap_dist_20d_zscore",
    "vwap_dist_5d_zscore",
    "w_volshock_ema",

    # === MOMENTUM / TREND QUALITY (4 features) ===
    "adx_14",
    "chop_14",
    "di_plus_14",
    "rsi_14",

    # === RANGE / BREAKOUT (1 features) ===
    "overnight_ret",

]


# =============================================================================
# HEAD FEATURES - Model-specific additive features
# =============================================================================
# Auto-generated from multi-model selection: 2025-12-23 15:51:09 UTC
# These are features selected by specific models but not in CORE.

HEAD_FEATURES: Dict[ModelKey, List[str]] = {
    # LONG_NORMAL: Standard long momentum (1.5 ATR style)
    ModelKey.LONG_NORMAL: [
        "di_minus_14",
        "gap_atr_ratio",
        "gap_fill_frac",
        "overnight_ratio",
        "range_efficiency",
        "sector_breadth_pct_above_ma200",
        "squeeze_release_20",
        "upper_shadow_ratio",
        "w_alpha_mom_sector_60_ema10",
        "w_beta_qqq",
        "w_cyclical_defensive_ratio",
        "w_fred_bamlh0a0hym2_z60",
        "w_pct_slope_ma_50",
    ],

    # LONG_PARABOLIC: Extended momentum / trend persistence
    ModelKey.LONG_PARABOLIC: [
        "credit_spread_zscore",
        "di_minus_14",
        "fred_ccsa_chg4w",
        "fred_dgs2_chg20d",
        "overnight_ratio",
        "pct_dist_ma_100_z",
        "range_efficiency",
        "recovery_120d",
        "rel_strength_spy",
        "rsp_spy_cumret_60",
        "rsp_spy_cumret_60_x_qqq_spy_cumret_20",
        "rsp_spy_cumret_60_x_w_fred_icsa_z52w",
        "sector_breadth_pct_above_ma200",
        "squeeze_release_20",
        "upper_shadow_ratio",
        "w_alpha_mom_sector_60_ema10",
        "w_beta_qqq",
        "w_pct_slope_ma_50",
        "w_vix_vxn_spread",
    ],

    # SHORT_NORMAL: Breakdown / fragility / liquidity stress
    ModelKey.SHORT_NORMAL: [
        "alpha_mom_spy_120_ema10",
        "credit_spread_zscore",
        "di_minus_14",
        "fred_ccsa_chg4w",
        "fred_dgs10_chg20d",
        "fred_dgs2_chg20d",
        "fred_nfci_z52w",
        "gap_fill_frac",
        "range_efficiency",
        "rel_strength_spy",
        "sector_breadth_ad_line_x_w_fred_bamlh0a0hym2_z60",
        "sector_breadth_pct_above_ma50",
        "squeeze_release_20",
        "upper_shadow_ratio",
        "w_alpha_mom_sector_60_ema10",
        "w_beta_qqq",
        "w_credit_spread_zscore",
        "w_cyclical_defensive_ratio",
        "w_fred_bamlh0a0hym2_z60",
        "w_fred_ccsa_z52w",
        "w_pct_slope_ma_50",
        "w_vix_vxn_spread",
        "w_xsec_pct_4w",
        "w_yield_curve_zscore",
    ],

    # SHORT_PARABOLIC: Panic / regime shift / vol-of-vol
    ModelKey.SHORT_PARABOLIC: [
        "drawdown_60d_z",
        "fred_dgs10_chg20d",
        "fred_dgs10_z60",
        "fred_dgs2_chg20d",
        "gap_fill_frac",
        "min_pct_dist_ma",
        "recovery_20d",
        "rsp_spy_cumret_60",
        "sector_breadth_pct_above_ma200",
        "sector_breadth_pct_above_ma50",
        "w_cyclical_defensive_ratio",
        "w_fred_bamlh0a0hym2_z60",
        "w_rel_strength_spy_zscore",
        "w_rsp_spy_cumret_12",
        "w_vix_vxn_spread",
        "yield_curve_zscore",
    ],

}


# =============================================================================
# FEATURE CATEGORIES - For reference and validation
# =============================================================================

FEATURE_CATEGORIES = {
    "trend": [
        "trend_score_sign", "trend_score_granular", "trend_score_slope",
        "trend_persist_ema", "pct_slope_ma_20", "pct_slope_ma_100",
        "w_pct_slope_ma_20", "w_pct_slope_ma_50", "w_trend_persist_ema",
    ],
    "momentum": [
        "rsi_14", "w_rsi_14", "macd_histogram", "w_macd_histogram",
        "macd_hist_deriv_ema3", "chop_14", "adx_14", "di_plus_14", "di_minus_14",
    ],
    "volatility": [
        "vol_regime", "vol_regime_ema10", "atr_percent", "rv_z_60", "rvol_20",
        "w_rv60_slope_norm", "w_rv100_slope_norm", "bb_width_20_2",
        "bb_width_20_2_z60", "squeeze_on_20", "squeeze_on_wide_20",
        "squeeze_intensity_20", "squeeze_release_20", "days_in_squeeze_20",
        "atr_percent_chg_5",
    ],
    "price_position": [
        "pct_dist_ma_20", "pct_dist_ma_50", "pct_dist_ma_100_z",
        "w_pct_dist_ma_20", "w_pct_dist_ma_100_z", "min_pct_dist_ma",
        "relative_dist_20_50_z", "pct_dist_ma_20_z", "pct_dist_ma_50_z",
    ],
    "range_breakout": [
        "atr_percent", "pos_in_5d_range", "pos_in_10d_range", "pos_in_20d_range",
        "w_pos_in_5d_range", "breakout_up_5d", "breakout_up_10d", "breakout_up_20d",
        "breakout_dn_5d", "breakout_dn_10d", "breakout_dn_20d",
        "range_expansion_5d", "range_expansion_10d", "range_expansion_20d",
        "overnight_ret", "gap_atr_ratio_raw", "gap_fill_frac",
    ],
    "volume": [
        "obv_z_60", "rdollar_vol_20", "volshock_z", "volshock_ema",
        "volshock_dir", "w_volshock_ema", "pv_divergence_5d",
    ],
    "liquidity": [
        "hl_spread_proxy", "cs_spread_est", "roll_spread_est",
        "overnight_ratio", "range_efficiency", "upper_shadow_ratio",
        "lower_shadow_ratio", "vwap_dist_5d_zscore", "vwap_dist_20d_zscore",
        "w_vwap_dist_20d_zscore", "amihud_illiq", "amihud_illiq_ratio",
        "illiquidity_score",
    ],
    "drawdown": [
        "drawdown_20d", "drawdown_60d", "drawdown_120d", "drawdown_expanding",
        "drawdown_20d_z", "drawdown_60d_z", "drawdown_regime",
        "days_since_high_20d_norm", "days_since_high_60d_norm",
        "recovery_20d", "recovery_60d", "drawdown_velocity_20d",
        "drawdown_velocity_60d", "hl_range_position_60d",
        "w_drawdown_60d", "w_drawdown_60d_z",
    ],
    "divergence": [
        "rsi_price_div_10d", "rsi_price_div_20d",
        "rsi_price_div_cum_10d", "rsi_price_div_cum_20d",
        "macd_price_div_10d", "macd_price_div_20d",
        "trend_rsi_div_10d", "trend_rsi_div_20d",
        "vol_trend_div_10d", "vol_trend_div_20d",
    ],
    "cross_sectional": [
        "vol_regime_cs_median", "vol_regime_rel", "xsec_mom_5d_z",
        "xsec_mom_20d_z", "xsec_mom_60d_z", "xsec_pct_20d",
        "w_xsec_mom_4w_z", "w_xsec_mom_13w_z",
    ],
    "alpha": [
        "beta_spy_simple", "beta_qqq_simple", "beta_sector",
        "w_beta_spy_simple", "w_beta_qqq_simple", "w_beta_qqq",
        "alpha_mom_spy_20_ema10", "alpha_mom_spy_60_ema10",
        "alpha_mom_qqq_20_ema10", "alpha_mom_qqq_60_ema10",
        "alpha_mom_sector_20_ema10", "alpha_resid_spy", "alpha_qqq_vs_spy",
        "w_alpha_mom_spy_20_ema10", "w_alpha_mom_qqq_60_ema10",
        "w_alpha_mom_sector_60_ema10",
        "beta_market", "beta_qqq", "beta_bestmatch", "beta_breadth",
        "residual_cumret", "residual_vol",
    ],
    "factor_spreads": [
        "qqq_cumret_20", "qqq_cumret_60", "qqq_cumret_120",
        "qqq_zscore_60", "qqq_slope_20", "qqq_slope_60",
        "spy_cumret_20", "spy_cumret_60", "spy_cumret_120",
        "spy_zscore_60", "spy_slope_20", "spy_slope_60",
        "qqq_spy_cumret_20", "qqq_spy_cumret_60", "qqq_spy_cumret_120",
        "qqq_spy_zscore_60", "qqq_spy_slope_20", "qqq_spy_slope_60",
        "rsp_spy_cumret_20", "rsp_spy_cumret_60", "rsp_spy_cumret_120",
        "rsp_spy_zscore_60", "rsp_spy_slope_20", "rsp_spy_slope_60",
        "bestmatch_spy_cumret_20", "bestmatch_spy_cumret_60",
        "bestmatch_spy_cumret_120", "bestmatch_spy_zscore_60",
        "bestmatch_spy_slope_20", "bestmatch_spy_slope_60",
    ],
    "relative_strength": [
        "rel_strength_spy", "rel_strength_spy_zscore",
        "rel_strength_qqq", "rel_strength_qqq_zscore",
        "rel_strength_sector", "rel_strength_sector_zscore",
        "w_rel_strength_spy", "w_rel_strength_sector",
    ],
    "breadth": [
        "sector_breadth_pct_above_ma50", "sector_breadth_pct_above_ma200",
        "sector_breadth_mcclellan_osc", "sector_breadth_ad_line",
        "w_sector_breadth_pct_above_ma10", "w_sector_breadth_pct_above_ma40",
        "w_sector_breadth_mcclellan_osc",
    ],
    "macro": [
        "vix_regime", "vix_percentile_252d", "vix_zscore_60d",
        "vix_ma20_ratio", "vix_vxn_spread", "vix_change_5d", "vix_change_20d",
        "w_vix_ma4_ratio", "w_vix_vxn_spread", "w_vix_percentile_52w",
        "w_vix_change_4w", "w_vix_regime",
        "fred_dgs10_chg20d", "fred_dgs2_chg20d", "fred_t10y2y_z60",
        "fred_bamlh0a0hym2_z60", "fred_bamlh0a0hym2_pct252",
        "fred_icsa_chg4w", "fred_icsa_z52w", "fred_ccsa_z52w", "fred_nfci_chg4w",
        "w_fred_bamlh0a0hym2_z60", "w_fred_icsa_chg4w", "w_fred_icsa_z52w",
        "w_fred_nfci_chg4w",
    ],
    "intermarket": [
        "copper_gold_ratio", "copper_gold_zscore",
        "gold_spy_ratio", "gold_spy_ratio_zscore",
        "dollar_momentum_20d", "dollar_percentile_252d", "oil_momentum_20d",
        "cyclical_defensive_ratio", "financials_utilities_ratio", "tech_spy_ratio",
        "equity_bond_corr_60d", "credit_spread_zscore", "yield_curve_zscore",
        "w_copper_gold_ratio", "w_gold_spy_ratio", "w_gold_spy_ratio_zscore",
        "w_dollar_momentum_20d", "w_financials_utilities_ratio",
        "w_equity_bond_corr_60d", "w_cyclical_defensive_ratio",
        "w_credit_spread_zscore",
    ],
}


# =============================================================================
# EXPANSION CANDIDATES V3
# =============================================================================

EXPANSION_CANDIDATES = {
    "distance_to_ma": [
        "pct_dist_ma_100_z", "pct_dist_ma_200_z", "min_pct_dist_ma",
        "w_pct_dist_ma_20_z", "w_pct_dist_ma_50_z", "w_pct_dist_ma_100_z",
        "w_min_pct_dist_ma", "w_relative_dist_20_50_z",
    ],
    "alpha_momentum": [
        "alpha_mom_spy_60_ema10", "alpha_mom_spy_120_ema10",
        "alpha_mom_qqq_60_ema10", "alpha_mom_sector_60_ema10",
        "alpha_mom_combo_20_ema10", "alpha_mom_combo_60_ema10",
        "w_alpha_mom_spy_60_ema10",
    ],
    "factor_spreads": [
        "qqq_spy_zscore_60", "rsp_spy_cumret_20", "rsp_spy_cumret_60",
        "rsp_spy_zscore_60", "bestmatch_spy_cumret_60", "bestmatch_spy_zscore_60",
        "w_rsp_spy_cumret_12", "w_bestmatch_spy_cumret_12",
    ],
    "relative_strength": [
        "rel_strength_spy", "rel_strength_spy_zscore",
        "rel_strength_qqq", "rel_strength_qqq_zscore",
        "rel_strength_sector_zscore", "w_rel_strength_spy",
        "w_rel_strength_spy_zscore", "w_rel_strength_qqq",
    ],
    "cross_sectional_momentum": [
        "xsec_mom_5d_z", "xsec_mom_60d_z", "xsec_mom_20d_sect_neutral_z",
        "xsec_pct_20d", "xsec_pct_60d", "w_xsec_mom_1w_z",
        "w_xsec_mom_13w_z", "w_xsec_pct_4w",
    ],
    "sector_breadth": [
        "sector_breadth_pct_above_ma50",
        "w_sector_breadth_pct_above_ma10", "w_sector_breadth_pct_above_ma40",
    ],
    "macro_fred": [
        "fred_dgs10_chg20d", "fred_dgs10_z60", "fred_nfci_z52w",
        "fred_icsa_z52w", "fred_ccsa_chg4w", "w_fred_dgs10_z60",
        "w_fred_t10y2y_z60", "w_fred_ccsa_z52w", "w_fred_nfci_chg4w",
    ],
    "regime_correlation": [
        "credit_spread_zscore", "yield_curve_zscore",
        "w_credit_spread_zscore", "w_yield_curve_zscore",
    ],
    "drawdown_recovery": [
        "drawdown_20d", "drawdown_60d", "drawdown_120d", "drawdown_expanding",
        "drawdown_20d_z", "drawdown_60d_z", "drawdown_120d_z",
        "days_since_high_60d_norm", "days_since_high_120d_norm",
        "recovery_20d", "recovery_60d", "recovery_120d",
        "recovery_20d_z", "recovery_60d_z", "recovery_120d_z",
        "drawdown_velocity_20d", "drawdown_velocity_60d", "drawdown_velocity_120d",
        "drawdown_regime", "hl_range_position_60d",
        "w_drawdown_60d", "w_drawdown_60d_z", "w_days_since_high_60d_norm",
        "w_recovery_60d", "w_drawdown_velocity_60d",
    ],
    "gaps": [
        "gap_atr_ratio", "overnight_ret", "gap_fill_frac", "atr_percent_chg_5",
    ],
    "trend_quality": ["adx_14", "di_plus_14"],
    "volatility_squeeze": [
        "bb_width_20_2_z60", "squeeze_on_20", "squeeze_on_wide_20",
        "squeeze_intensity_20", "squeeze_release_20", "days_in_squeeze_20",
    ],
    "divergence": [
        "rsi_price_div_10d", "rsi_price_div_20d",
        "rsi_price_div_cum_10d", "rsi_price_div_cum_20d",
        "macd_price_div_10d", "macd_price_div_20d",
        "trend_rsi_div_10d", "trend_rsi_div_20d",
        "vol_trend_div_10d", "vol_trend_div_20d",
        "w_rsi_price_div_20d", "w_macd_price_div_20d",
    ],
    "volume_liquidity": ["pv_divergence_5d"],
}


# =============================================================================
# EXCLUDED FEATURES - Raw values not suitable for ML
# =============================================================================

EXCLUDED_FEATURES = [
    "open", "high", "low", "close", "adjclose", "volume", "ret", "w_ret",
    "ma_10", "ma_20", "ma_30", "ma_50", "ma_75", "ma_100", "ma_150", "ma_200",
    "w_ma_10", "w_ma_20", "w_ma_30", "w_ma_50", "w_ma_75", "w_ma_100",
    "w_ma_150", "w_ma_200", "w_sma20", "w_sma50",
    "atr14", "w_atr14",
    "5d_high", "5d_low", "10d_high", "10d_low", "20d_high", "20d_low",
    "w_5d_high", "w_5d_low", "w_10d_high", "w_10d_low", "w_20d_high", "w_20d_low",
    "5d_range", "10d_range", "20d_range", "hl_range", "true_range",
    "w_5d_range", "w_10d_range", "w_20d_range", "w_hl_range", "w_true_range",
    "rv_10", "rv_20", "rv_60", "rv_100", "vol_ma_20", "vol_ma_50",
    "vol_rolling_20d", "vol_rolling_60d",
    "w_rv_10", "w_rv_20", "w_rv_60", "w_rv_100", "w_vol_ma_20", "w_vol_ma_50",
    "vix_level", "vix_ema10", "vxn_level",
    "w_vix_level", "w_vix_ema10", "w_vix_ema4", "w_vxn_level",
    "fred_bamlc0a4cbbb", "fred_bamlh0a0hym2", "fred_dgs10", "fred_dgs2",
    "fred_t10y2y", "fred_t10y3m", "fred_dfedtaru", "fred_nfci",
    "fred_icsa", "fred_ccsa", "fred_vixcls",
    "w_fred_bamlc0a4cbbb", "w_fred_bamlh0a0hym2", "w_fred_dgs10", "w_fred_dgs2",
    "w_fred_t10y2y", "w_fred_t10y3m", "w_fred_dfedtaru", "w_fred_nfci",
    "w_fred_icsa", "w_fred_ccsa", "w_fred_vixcls",
    "obv", "dollar_vol_ma_20", "w_obv", "w_dollar_vol_ma_20",
    "credit_spread_proxy", "yield_curve_proxy",
    "w_credit_spread_proxy", "w_yield_curve_proxy",
    "sector_breadth_adv", "sector_breadth_dec", "sector_breadth_net_adv",
    "sector_breadth_mcclellan_sum",
    "w_sector_breadth_adv", "w_sector_breadth_dec", "w_sector_breadth_net_adv",
    "w_sector_breadth_ad_line", "w_sector_breadth_mcclellan_sum",
    "ad_ratio_ema10", "ad_ratio_universe", "ad_thrust_10d", "mcclellan_oscillator",
    "pct_universe_above_ma20", "pct_universe_above_ma50",
    "w_ad_ratio_universe", "w_ad_ratio_ema10", "w_mcclellan_oscillator",
    "w_ad_thrust_4w",
]


# =============================================================================
# RETIRED FEATURES - Can be excluded from computation
# =============================================================================

RETIRED_FEATURES_BY_MODULE = {
    "trend": [
        "macd_hist_deriv_ema3", "w_macd_hist_deriv_ema3",
        "rsi_21", "rsi_30", "w_rsi_14", "w_rsi_21",
        "trend_score_granular", "w_trend_score_granular",
        "trend_persist_ema", "w_trend_persist_ema",
        "quiet_trend", "w_quiet_trend", "trend_alignment", "w_trend_alignment",
        "pct_slope_ma_10", "pct_slope_ma_30", "pct_slope_ma_50",
        "pct_slope_ma_75", "pct_slope_ma_150", "pct_slope_ma_200",
        "w_pct_slope_ma_10", "w_pct_slope_ma_20", "w_pct_slope_ma_30",
        "w_pct_slope_ma_75", "w_pct_slope_ma_100", "w_pct_slope_ma_150",
        "w_pct_slope_ma_200", "w_trend_score_slope",
    ],
    "range_breakout": [
        "breakout_up_5d", "breakout_up_10d", "breakout_up_20d",
        "breakout_dn_5d", "breakout_dn_10d", "breakout_dn_20d",
        "w_breakout_up_5d", "w_breakout_up_10d", "w_breakout_up_20d",
        "w_breakout_dn_5d", "w_breakout_dn_10d", "w_breakout_dn_20d",
        "range_expansion_5d", "range_expansion_10d", "range_expansion_20d",
        "w_range_expansion_5d", "w_range_expansion_10d", "w_range_expansion_20d",
        "range_z_5d", "range_z_10d", "range_z_20d",
        "w_range_z_5d", "w_range_z_10d", "w_range_z_20d",
        "pos_in_5d_range", "pos_in_10d_range",
        "w_pos_in_5d_range", "w_pos_in_10d_range", "w_pos_in_20d_range",
    ],
    "volatility": [
        "vol_regime", "rv_ratio_10_60", "rv_ratio_20_100",
        "vol_z_20", "vol_z_60", "rvol_20", "w_rvol_20",
        "vol_regime_cs_median", "vol_regime_rel",
        "w_rv_z_60", "w_vol_z_60",
        "w_vol_regime", "w_vol_regime_ema10", "w_vol_regime_rel",
        "rv60_slope_norm", "rv100_slope_norm",
        "w_rv60_slope_norm", "w_rv100_slope_norm",
    ],
    "volume": [
        "obv_z_60", "w_obv_z_60", "volshock_z", "volshock_dir",
        "w_volshock_z", "w_volshock_dir", "rdollar_vol_20", "w_rdollar_vol_20",
    ],
    "liquidity": [
        "vwap_dist_10d_zscore", "w_vwap_dist_20d_zscore",
        # NOTE: lower_shadow_ratio, overnight_ratio, range_efficiency are now
        # HEAD_FEATURES for LONG_NORMAL/SHORT_NORMAL - do not retire them
        "w_range_efficiency", "amihud_illiq_ratio",
        "rel_volume_5d", "rel_volume_10d", "rel_volume_20d",
        "w_rel_volume_5d", "w_rel_volume_10d", "w_rel_volume_20d",
        "volume_direction", "volume_trend_10d",
        "illiquidity_score", "w_illiquidity_score",
    ],
    "alpha": [
        "beta_market", "beta_qqq", "beta_bestmatch", "beta_breadth",
        "beta_spy_simple", "beta_qqq_simple", "beta_sector",
        "residual_cumret", "residual_vol", "residual_mean",
        "w_beta_market", "w_beta_bestmatch", "w_beta_breadth",
        "w_beta_spy_simple", "w_beta_qqq_simple",
        "w_residual_cumret", "w_residual_vol",
    ],
    "macro": [
        "vix_ma20_ratio", "vix_vxn_spread",
        "vix_change_5d", "vix_change_20d", "vix_regime",
        "w_vix_percentile_52w", "w_vix_zscore_12w", "w_vix_regime",
        "w_vix_ma4_ratio", "w_vix_change_4w", "w_vxn_percentile_252d",
    ],
    "spread_features": [
        "copper_gold_ratio", "w_copper_gold_ratio",
        "gold_spy_ratio", "w_gold_spy_ratio",
        "cyclical_defensive_ratio",
        "financials_utilities_ratio", "w_financials_utilities_ratio",
        "tech_spy_ratio", "w_tech_spy_ratio", "oil_momentum_20d",
        "dollar_momentum_20d", "w_dollar_momentum_20d", "dollar_percentile_252d",
    ],
}

# Build RETIRED_FEATURES from modules, but exclude any HEAD_FEATURES
# (HEAD_FEATURES are active for the 4-model system and should not be retired)
_all_head_features: Set[str] = set()
for _head_list in HEAD_FEATURES.values():
    _all_head_features.update(_head_list)

RETIRED_FEATURES: Set[str] = set()
for module_features in RETIRED_FEATURES_BY_MODULE.values():
    RETIRED_FEATURES.update(module_features)
# Remove any features that are now HEAD_FEATURES (reactivated for 4-model system)
RETIRED_FEATURES -= _all_head_features


# =============================================================================
# INTERMEDIATE FEATURES - Required for derived features
# =============================================================================

INTERMEDIATE_FEATURES = {
    "ma_10", "ma_20", "ma_30", "ma_50", "ma_75", "ma_100", "ma_150", "ma_200",
    "w_ma_10", "w_ma_20", "w_ma_30", "w_ma_50", "w_ma_75", "w_ma_100",
    "w_ma_150", "w_ma_200",
    "atr14", "w_atr14",
    "5d_high", "5d_low", "10d_high", "10d_low", "20d_high", "20d_low",
    "5d_range", "10d_range", "20d_range",
    "rv_10", "rv_20", "rv_60", "rv_100",
    "open", "high", "low", "close", "adjclose", "volume",
    "sign_ma_10", "sign_ma_20", "sign_ma_30", "sign_ma_50",
    "sign_ma_75", "sign_ma_100", "sign_ma_150", "sign_ma_200",
    "pct_dist_ma_20", "pct_dist_ma_50", "pct_dist_ma_100", "pct_dist_ma_200",
    "sector_breadth_adv", "sector_breadth_dec", "sector_breadth_net_adv",
    "w_sector_breadth_adv", "w_sector_breadth_dec", "w_sector_breadth_net_adv",
    "trend_score_granular",
}


# =============================================================================
# FEATURE DEPENDENCIES
# =============================================================================

FEATURE_DEPENDENCIES = {
    "atr_percent": {"atr14"},
    "vol_regime": {"rv_20", "rv_100"},
    "vol_regime_ema10": {"vol_regime", "rv_20", "rv_100"},
    "rv_z_60": {"rv_20"},
    "trend_score_sign": {
        "sign_ma_10", "sign_ma_20", "sign_ma_30", "sign_ma_50",
        "sign_ma_75", "sign_ma_100", "sign_ma_150", "sign_ma_200"
    },
    "trend_score_slope": {"trend_score_granular"},
    "pct_slope_ma_20": {"ma_20"},
    "pct_slope_ma_100": {"ma_100"},
    "pct_dist_ma_20_z": {"pct_dist_ma_20", "ma_20"},
    "pct_dist_ma_50_z": {"pct_dist_ma_50", "ma_50"},
    "relative_dist_20_50_z": {"ma_20", "ma_50"},
    "pos_in_20d_range": {"20d_high", "20d_low"},
    "gap_atr_ratio": {"atr_percent"},
}


# =============================================================================
# BACKWARDS COMPATIBILITY - BASE_FEATURES alias
# =============================================================================

def _compute_base_features() -> List[str]:
    """Compute BASE_FEATURES as CORE + HEAD[LONG_NORMAL] with deduplication."""
    seen = set()
    result = []
    for feat in CORE_FEATURES:
        if feat not in seen:
            seen.add(feat)
            result.append(feat)
    for feat in HEAD_FEATURES.get(ModelKey.LONG_NORMAL, []):
        if feat not in seen:
            seen.add(feat)
            result.append(feat)
    return result


BASE_FEATURES: List[str] = _compute_base_features()


# =============================================================================
# MODEL-AWARE FEATURE RETRIEVAL FUNCTIONS
# =============================================================================

def get_core_features() -> List[str]:
    """Return the list of core features shared across all models."""
    return CORE_FEATURES.copy()


def get_head_features(model_key: ModelKey) -> List[str]:
    """
    Return the head features for a specific model key.

    Args:
        model_key: The model key (LONG_NORMAL, LONG_PARABOLIC, etc.)

    Returns:
        List of head feature names for this model
    """
    if model_key not in HEAD_FEATURES:
        raise ValueError(
            f"Unknown model_key: {model_key}. Valid: {list(HEAD_FEATURES.keys())}"
        )
    return HEAD_FEATURES[model_key].copy()


def get_featureset(
    model_key: ModelKey,
    include_expansion: bool = False,
    flat: bool = True
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Return the complete feature set for a given model key.

    Args:
        model_key: The model key (LONG_NORMAL, LONG_PARABOLIC, etc.)
        include_expansion: If True, also include EXPANSION_CANDIDATES
        flat: If True, return flat list. If False, return dict.

    Returns:
        If flat=True: List of feature names (CORE + HEAD + optional expansion)
        If flat=False: Dict with 'core', 'head', and optionally 'expansion' keys
    """
    if model_key not in HEAD_FEATURES:
        raise ValueError(
            f"Unknown model_key: {model_key}. Valid: {list(HEAD_FEATURES.keys())}"
        )

    core = CORE_FEATURES.copy()
    head = HEAD_FEATURES[model_key].copy()

    if flat:
        seen = set()
        result = []
        for feat in core:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
        for feat in head:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
        if include_expansion:
            for category_features in EXPANSION_CANDIDATES.values():
                for feat in category_features:
                    if feat not in seen:
                        seen.add(feat)
                        result.append(feat)
        return result
    else:
        result = {'core': core, 'head': head}
        if include_expansion:
            result['expansion'] = get_expansion_candidates(flat=True)
        return result


def get_all_selectable_features(model_key: Optional[ModelKey] = None) -> List[str]:
    """
    Return all features suitable for selection.

    Args:
        model_key: If provided, return CORE + HEAD[model_key] + expansion.
                   If None, return union across all models.

    Returns:
        List of feature names
    """
    seen = set()
    result = []

    for feat in CORE_FEATURES:
        if feat not in seen:
            seen.add(feat)
            result.append(feat)

    if model_key is not None:
        for feat in HEAD_FEATURES.get(model_key, []):
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
    else:
        for head_list in HEAD_FEATURES.values():
            for feat in head_list:
                if feat not in seen:
                    seen.add(feat)
                    result.append(feat)

    for category_features in EXPANSION_CANDIDATES.values():
        for feat in category_features:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)

    return result


def get_all_head_features() -> Set[str]:
    """Return the union of all head features across all models."""
    all_heads = set()
    for head_list in HEAD_FEATURES.values():
        all_heads.update(head_list)
    return all_heads


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def get_base_features() -> List[str]:
    """
    Return the list of base features for forward selection.

    DEPRECATED: Use get_featureset(ModelKey.LONG_NORMAL) instead.
    """
    return BASE_FEATURES.copy()


def get_retired_features() -> Set[str]:
    """Return the set of retired feature names."""
    return RETIRED_FEATURES.copy()


def get_retired_features_by_module() -> Dict[str, List[str]]:
    """Return retired features organized by module."""
    return {k: list(v) for k, v in RETIRED_FEATURES_BY_MODULE.items()}


def get_intermediate_features() -> Set[str]:
    """Return the set of intermediate feature names."""
    return INTERMEDIATE_FEATURES.copy()


def get_feature_dependencies() -> Dict[str, Set[str]]:
    """Return the feature dependency map."""
    return {k: v.copy() for k, v in FEATURE_DEPENDENCIES.items()}


def get_expansion_candidates(flat: bool = False) -> Union[Dict[str, List[str]], List[str]]:
    """
    Return expansion candidate features.

    Args:
        flat: If True, return flat list. If False, return dict by category.
    """
    if flat:
        candidates = []
        for category_features in EXPANSION_CANDIDATES.values():
            candidates.extend(category_features)
        return candidates
    return {k: list(v) for k, v in EXPANSION_CANDIDATES.items()}


def get_excluded_features() -> List[str]:
    """Return list of features excluded from selection (raw values)."""
    return EXCLUDED_FEATURES.copy()


def get_features_required_for_model(model_key: ModelKey) -> Set[str]:
    """
    Get features that MUST be computed for a given model.

    Includes CORE + HEAD features plus intermediate dependencies.
    """
    required = set(CORE_FEATURES)
    required.update(HEAD_FEATURES.get(model_key, []))

    to_check = list(required)
    while to_check:
        feat = to_check.pop()
        if feat in FEATURE_DEPENDENCIES:
            for dep in FEATURE_DEPENDENCIES[feat]:
                if dep not in required:
                    required.add(dep)
                    to_check.append(dep)

    return required


def get_retired_features_safe_to_skip(model_key: Optional[ModelKey] = None) -> Set[str]:
    """
    Get retired features that can be safely skipped for a given model.

    Features used as HEAD_FEATURES are NOT safe to skip.
    """
    retired = get_retired_features()

    if model_key is not None:
        heads_to_preserve = set(HEAD_FEATURES.get(model_key, []))
    else:
        heads_to_preserve = get_all_head_features()

    return retired - heads_to_preserve


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_features(
    df,
    features: Optional[List[str]] = None,
    model_key: Optional[ModelKey] = None
) -> Dict[str, any]:
    """
    Validate that features exist in DataFrame and report NaN rates.

    Args:
        df: DataFrame with features
        features: List of feature names (default: use model_key or BASE_FEATURES)
        model_key: If provided and features is None, validate for this model

    Returns:
        Dict with 'valid', 'missing', 'nan_rates', and 'model_key' keys
    """
    if features is None:
        if model_key is not None:
            features = get_featureset(model_key)
        else:
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
        "model_key": model_key,
    }


def validate_model_featuresets() -> Dict[str, any]:
    """
    Validate all model featuresets for consistency.

    Returns:
        Dict with validation results per model and overlap statistics
    """
    results = {"models": {}, "overlaps": {}, "all_valid": True}

    all_known_features = set()
    for cat_features in FEATURE_CATEGORIES.values():
        all_known_features.update(cat_features)

    for model_key in ModelKey.all_keys():
        featureset = get_featureset(model_key)
        unique_features = set(featureset)

        duplicates = len(featureset) - len(unique_features)
        unknown = unique_features - all_known_features

        results["models"][model_key.value] = {
            "total": len(featureset),
            "unique": len(unique_features),
            "duplicates": duplicates,
            "unknown": list(unknown),
            "core_count": len(CORE_FEATURES),
            "head_count": len(HEAD_FEATURES.get(model_key, [])),
        }

        if duplicates > 0 or unknown:
            results["all_valid"] = False

    for mk1 in ModelKey.all_keys():
        for mk2 in ModelKey.all_keys():
            if mk1.value >= mk2.value:
                continue
            set1 = set(get_featureset(mk1))
            set2 = set(get_featureset(mk2))
            overlap = len(set1 & set2)
            union = len(set1 | set2)
            overlap_pct = (overlap / union * 100) if union > 0 else 0
            key = f"{mk1.value}_vs_{mk2.value}"
            results["overlaps"][key] = {
                "overlap_count": overlap,
                "overlap_pct": round(overlap_pct, 1),
            }

    return results


def report_head_features_status(df, model_key: ModelKey) -> Dict[str, any]:
    """
    Report which head features for a model are present/missing/have high NaN.
    """
    head_features = get_head_features(model_key)

    present = []
    missing = []
    high_nan = []

    for feat in head_features:
        if feat in df.columns:
            nan_rate = df[feat].isna().mean() * 100
            present.append(feat)
            if nan_rate > 30:
                high_nan.append((feat, nan_rate))
        else:
            missing.append(feat)

    return {
        "model_key": model_key.value,
        "present": present,
        "missing": missing,
        "high_nan": high_nan,
        "summary": f"{len(present)}/{len(head_features)} present, "
                   f"{len(missing)} missing, {len(high_nan)} high NaN",
    }


# =============================================================================
# OUTPUT FILTERING
# =============================================================================

META_COLUMNS = ['symbol', 'date', 'ret']
REQUIRED_FEATURES = ['atr_percent']


def get_output_features(model_key: Optional[ModelKey] = None) -> Set[str]:
    """
    Get the curated list of features to include in pipeline output.

    Args:
        model_key: If provided, include CORE + HEAD for this model.
                   If None, include CORE + all heads (legacy behavior).

    Returns:
        Set of feature names to include in output
    """
    if model_key is not None:
        base = get_featureset(model_key)
    else:
        base = get_all_selectable_features(model_key=None)

    expansion = get_expansion_candidates(flat=True)
    output_features = set(META_COLUMNS + REQUIRED_FEATURES + base + expansion)
    return output_features


def filter_output_columns(
    df,
    keep_all: bool = False,
    exclude_retired: bool = False,
    model_key: Optional[ModelKey] = None
):
    """
    Filter DataFrame columns to only include curated output features.

    Args:
        df: DataFrame with computed features
        keep_all: If True, return all columns (no filtering for curated set)
        exclude_retired: If True, also exclude retired features
        model_key: If provided, filter to CORE + HEAD for this model

    Returns:
        DataFrame with filtered columns
    """
    import logging
    logger = logging.getLogger(__name__)

    if keep_all and not exclude_retired:
        return df

    cols_to_keep = set(df.columns)
    retired_removed = 0

    if exclude_retired:
        retired = get_retired_features_safe_to_skip(model_key)
        retired_in_df = cols_to_keep & retired
        retired_removed = len(retired_in_df)
        cols_to_keep -= retired_in_df
        if retired_removed > 0:
            logger.info(f"Excluded {retired_removed} retired features from output")

    if not keep_all:
        output_features = get_output_features(model_key)
        cols_to_keep &= output_features

    keep_cols = [c for c in df.columns if c in cols_to_keep]

    filtered_count = len(df.columns) - len(keep_cols)
    if filtered_count > 0:
        logger.debug(
            f"Filtered {filtered_count} columns "
            f"(including {retired_removed} retired), keeping {len(keep_cols)}"
        )

    return df[keep_cols]


def drop_retired_columns(
    df,
    inplace: bool = False,
    model_key: Optional[ModelKey] = None
):
    """
    Drop retired feature columns from a DataFrame.

    Args:
        df: DataFrame with computed features
        inplace: If True, modify DataFrame in place
        model_key: If provided, preserve head features for this model

    Returns:
        DataFrame with retired columns removed (or None if inplace=True)
    """
    retired = get_retired_features_safe_to_skip(model_key)
    cols_to_drop = [c for c in df.columns if c in retired]

    if cols_to_drop:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Dropping {len(cols_to_drop)} retired columns")

        if inplace:
            df.drop(columns=cols_to_drop, inplace=True)
            return None
        else:
            return df.drop(columns=cols_to_drop)

    return df if not inplace else None


def get_feature_exclusion_report(
    computed_features: Set[str],
    exclude_retired: bool = False,
    model_key: Optional[ModelKey] = None
) -> Dict[str, any]:
    """
    Generate a report of what features were/would be excluded.
    """
    output_features = get_output_features(model_key)
    retired = get_retired_features_safe_to_skip(model_key) if exclude_retired else set()
    intermediate = get_intermediate_features()

    kept_output = computed_features & output_features
    kept_intermediate = computed_features & intermediate
    excluded_retired = computed_features & retired if exclude_retired else set()

    other = computed_features - output_features - intermediate - retired

    return {
        "total_computed": len(computed_features),
        "kept_output": len(kept_output),
        "kept_intermediate": len(kept_intermediate),
        "excluded_retired": len(excluded_retired),
        "other": len(other),
        "retired_list": sorted(excluded_retired) if exclude_retired else [],
        "model_key": model_key.value if model_key else None,
    }
