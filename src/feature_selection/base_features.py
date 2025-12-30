"""
Group-first feature registry for the 4-model momentum strategy.

This module provides a structured feature registry organized by hypothesis groups,
supporting different feature sets per model key (LONG_NORMAL, LONG_PARABOLIC,
SHORT_NORMAL, SHORT_PARABOLIC).

Architecture:
- CORE_GROUPS: Dict[str, List[str]] - Shared backbone groups across all models
- HEAD_GROUPS: Dict[ModelKey, Dict[str, List[str]]] - Model-specific additive groups
- CANDIDATE_GROUPS: Dict[str, List[str]] - Groups available for selection
- INTERACTION_TEMPLATES: Dict[str, Dict] - Template-based group-to-group interactions

Group Design Rules:
- Each group contains 3-12 features
- One hypothesis per group (coherent semantic meaning)
- Groups are the atomic unit of selection (all-or-nothing)

Feature Set Retrieval:
    from src.config.model_keys import ModelKey
    from src.feature_selection.base_features import (
        get_core_groups, get_head_groups, get_candidate_groups
    )

    # Get all core groups
    core = get_core_groups()  # Dict[str, List[str]]

    # Get head groups for a model
    head = get_head_groups(ModelKey.LONG_NORMAL)  # Dict[str, List[str]]

    # Get all features for a model (flattened)
    features = get_featureset(ModelKey.LONG_NORMAL)

Backwards Compatibility:
    - BASE_FEATURES remains as alias (flattened CORE + HEAD[LONG_NORMAL])
    - get_base_features() returns BASE_FEATURES (deprecated)
    - CORE_FEATURES computed from CORE_GROUPS for legacy code
    - HEAD_FEATURES computed from HEAD_GROUPS for legacy code
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

# Import ModelKey - handle both direct and late imports
try:
    from ..config.model_keys import ModelKey, DEFAULT_MODEL_KEY
except ImportError:
    # Fallback for scripts that import base_features directly
    from src.config.model_keys import ModelKey, DEFAULT_MODEL_KEY


# =============================================================================
# CORE_GROUPS - Shared backbone groups across all models
# =============================================================================
# Auto-generated from multi-model selection: 2025-12-24 15:44:29 UTC
# Run signature: 6bf1994a6423
# These groups represent the intersection of all 4 model selections, organized
# by hypothesis. Each group is 3-12 features with coherent semantic meaning.

CORE_GROUPS: Dict[str, List[str]] = {
    # --- RELATIVE PERFORMANCE / ALPHA (7 features) ---
    "alpha_momentum": [
        "alpha_mom_qqq_20_ema10",     # Alpha vs QQQ, 20d window
        "alpha_mom_sector_20_ema10",  # Alpha vs sector ETF, 20d window
        "w_alpha_mom_qqq_60_ema10",   # Weekly alpha vs QQQ, 60d window
        "w_alpha_mom_spy_20_ema10",   # Weekly alpha vs SPY, 20d window
        "w_beta_qqq",                 # Weekly beta to QQQ
        "w_rel_strength_sector",      # Weekly relative strength vs sector
        "xsec_mom_20d_z",             # Cross-sectional momentum z-score
    ],

    # --- MACRO FRED: CREDIT & LABOR STRESS (6 features) ---
    # Combined rates curve and labor stress indicators
    "macro_credit_labor": [
        "fred_t10y2y_z60",            # Yield curve slope z-score (10Y-2Y)
        "fred_bamlh0a0hym2_z60",      # HY spread z-score (credit stress)
        "fred_ccsa_z52w",             # Continued claims z-score
        "w_fred_bamlh0a0hym2_z60",    # Weekly HY spread z-score
        "w_equity_bond_corr_60d",     # Weekly equity-bond correlation
        "w_fred_icsa_z52w",           # Weekly initial claims z-score
    ],

    # --- MACRO / INTERMARKET (5 features) ---
    "macro_intermarket": [
        "copper_gold_zscore",         # Copper/gold ratio (growth signal)
        "qqq_spy_cumret_20",          # QQQ vs SPY cumulative return 20d
        "qqq_spy_cumret_60",          # QQQ vs SPY cumulative return 60d
        "qqq_spy_slope_20",           # QQQ vs SPY slope 20d
        "w_cyclical_defensive_ratio", # Weekly cyclical/defensive ratio
    ],

    # --- VOLATILITY REGIME (3 features) ---
    # VIX-based fear/complacency indicators + RV regime
    # Refactored from market_regime: pure volatility regime hypothesis
    "volatility_regime": [
        "vix_percentile_252d",        # VIX percentile over 1 year (fear level)
        "vix_zscore_60d",             # VIX z-score 60d window (stress spike)
        "vol_regime_ema10",           # RV20/RV100 regime, smoothed (vol state)
    ],

    # --- VOLUME SHOCK (3 features) ---
    # Unusual volume activity indicators
    # Refactored from market_regime: pure volume anomaly hypothesis
    "volume_shock": [
        "volshock_ema",               # Daily volume shock EMA
        "w_volshock_ema",             # Weekly volume shock EMA
        "pv_divergence_5d",           # Price-volume divergence (5d)
    ],

    # --- MICROSTRUCTURE POSITION (3 features) ---
    # VWAP/liquidity-based institutional flow indicators
    # Refactored from market_regime: pure microstructure hypothesis
    "microstructure_position": [
        "vwap_dist_5d_zscore",        # VWAP distance z-score (5d) - flow position
        "vwap_dist_20d_zscore",       # VWAP distance z-score (20d) - longer flow
        "overnight_ratio",            # Overnight vs intraday ratio (lagged 1 day)
    ],

    # --- TREND STRENGTH (3 features) ---
    "trend_strength": [
        "pct_slope_ma_100",           # Slope of 100-day MA (%)
        "pct_slope_ma_20",            # Slope of 20-day MA (%)
        "w_macd_histogram",           # Weekly MACD histogram
    ],

    # --- PRICE POSITION / MEAN REVERSION (4 features) ---
    "price_position": [
        "pct_dist_ma_20_z",           # Distance to 20-day MA (z-scored)
        "pct_dist_ma_50_z",           # Distance to 50-day MA (z-scored)
        "pos_in_20d_range",           # Position in 20-day range [0,1]
        "relative_dist_20_50_z",      # Relative distance MA20 vs MA50
    ],

    # --- VOLATILITY STATE (5 features) ---
    # Squeeze dynamics and realized volatility levels
    # Refactored from volatility_squeeze: pure vol compression/expansion
    "volatility_state": [
        "bb_width_20_2",              # Bollinger band width (squeeze measure)
        "days_in_squeeze_20",         # Days in volatility squeeze
        "rv_z_60",                    # Realized vol z-score (60d)
        "squeeze_intensity_20",       # Squeeze intensity measure
        "squeeze_release_20",         # Squeeze release signal
    ],

    # --- GAP DYNAMICS (3 features) ---
    # Overnight gap behavior and normalization
    # Refactored from volatility_squeeze: pure gap hypothesis
    # NOTE: All gap features are LAGGED by 1 day to prevent leakage.
    # When predicting day T, these features reflect day T-1's gap.
    "gap_dynamics": [
        "gap_atr_ratio",              # Gap normalized by ATR (lagged 1 day)
        "gap_atr_ratio_raw",          # Raw gap/ATR ratio (lagged 1 day)
        "overnight_ret",              # Overnight return (lagged 1 day)
    ],

    # --- SECTOR BREADTH (3 features) ---
    "sector_breadth": [
        "sector_breadth_ad_line",         # Sector A/D line
        "sector_breadth_pct_above_ma200", # % of sector above 200-day MA
        "w_sector_breadth_mcclellan_osc", # Weekly McClellan oscillator
    ],

    # --- MOMENTUM / TREND QUALITY (5 features) ---
    "momentum_quality": [
        "adx_14",                     # Average Directional Index
        "chop_14",                    # Choppiness Index
        "di_minus_14",                # Directional Indicator (-)
        "di_plus_14",                 # Directional Indicator (+)
        "rsi_14",                     # RSI 14-period
    ],

    # --- RANGE / BREAKOUT (3 features) ---
    "range_breakout": [
        "gap_fill_frac",              # Gap fill fraction (moved from momentum_quality)
        "range_efficiency",           # Range efficiency (directional)
        "pos_in_5d_range",            # Position in 5-day range
    ],

    # --- DRAWDOWN RECOVERY (merged from LONG_NORMAL + LONG_PARABOLIC) ---
    # --- DRAWDOWN DEPTH ---
    "drawdown_depth": [
        "drawdown_60d_z",             # Drawdown from 60d high, z-scored
        "drawdown_120d_z",            # Drawdown from 120d high, z-scored
    ],
}


# =============================================================================
# HEAD_GROUPS - Model-specific additive feature groups
# =============================================================================
# Auto-generated from multi-model selection: 2025-12-24 15:44:29 UTC
# These are feature groups selected by specific models but not in CORE_GROUPS.

HEAD_GROUPS: Dict[ModelKey, Dict[str, List[str]]] = {
    # =========================================================================
    # LONG_NORMAL: Standard long momentum (1.5 ATR style)
    # Consolidated into 5 coherent hypothesis groups
    # =========================================================================
    ModelKey.LONG_NORMAL: {
        # Drawdown state and recovery dynamics
        
    },

    # =========================================================================
    # LONG_PARABOLIC: Extended momentum / trend persistence
    # Consolidated into 4 coherent hypothesis groups
    # =========================================================================
    ModelKey.LONG_PARABOLIC: {
        
    },

    # =========================================================================
    # SHORT_NORMAL: Breakdown / fragility / liquidity stress
    # Consolidated into 3 coherent hypothesis groups
    # =========================================================================
    ModelKey.SHORT_NORMAL: {
        
    },

    # =========================================================================
    # SHORT_PARABOLIC: Panic / regime shift / vol-of-vol
    # Consolidated into 3 coherent hypothesis groups
    # =========================================================================
    ModelKey.SHORT_PARABOLIC: {
        
    },
}


# =============================================================================
# CANDIDATE_GROUPS - Groups available for selection experiments
# =============================================================================
# These groups can be added or swapped during group-first selection.
# Organized by hypothesis with mandatory splits for specific domains.

CANDIDATE_GROUPS: Dict[str, List[str]] = {
    # --- DISTANCE TO MA (8 features) ---
    # --- STRUCTURAL DISTANCE TO TREND ---
    "ma_distance_structural": [
        "pct_dist_ma_100_z",
        "pct_dist_ma_200_z",
    ],

    # --- SHORT / MEDIUM HORIZON MA DISTANCE ---
    "ma_distance_intermediate": [
        "w_pct_dist_ma_20_z",
        "w_pct_dist_ma_50_z",
        "w_pct_dist_ma_100_z",
    ],

    # --- MA RELATIVE POSITIONING ---
    "ma_relative_position": [
        "w_relative_dist_20_50_z",
    ],

    # --- EXTREME STRETCH / EXHAUSTION ---
    "ma_distance_extremes": [
        "min_pct_dist_ma",
        "w_min_pct_dist_ma",
    ],

    # --- ALPHA MOMENTUM (12 features) ---
    # --- ALPHA MOMENTUM VS BENCHMARKS ---
    "alpha_momentum_core": [
        "alpha_mom_spy_60_ema10",
        "alpha_mom_spy_120_ema10",
        "alpha_mom_qqq_60_ema10",
        "alpha_mom_sector_60_ema10",
        "alpha_mom_combo_20_ema10",
        "alpha_mom_combo_60_ema10",
        "w_alpha_mom_spy_60_ema10",
    ],

    # --- RESIDUAL RETURN STATE ---
    "alpha_residual_return": [
        "residual_cumret",
        "w_residual_cumret",
    ],

    # --- RESIDUAL RISK / DISPERSION ---
    "alpha_residual_risk": [
        "residual_vol",
        "w_residual_vol",
        "residual_mean",
    ],

    # --- FACTOR SPREADS (8 features) ---
    # --- INDEX FACTOR SPREAD ---
    "factor_spread_index": [
        "qqq_spy_zscore_60",
    ],

    # --- EQUAL-WEIGHT / BREADTH LEADERSHIP ---
    "factor_spread_rsp": [
        "rsp_spy_cumret_20",
        "rsp_spy_cumret_60",
        "rsp_spy_zscore_60",
        "w_rsp_spy_cumret_12",
    ],

    # --- BESTMATCH / MAPPING-BASED SPREAD ---
    "factor_spread_bestmatch": [
        "bestmatch_spy_cumret_60",
        "bestmatch_spy_zscore_60",
        "w_bestmatch_spy_cumret_12",
    ],
    # --- RELATIVE STRENGTH (8 features) ---
    # --- RELATIVE STRENGTH VS SPY ---
    "relative_strength_spy": [
        "rel_strength_spy",
        "rel_strength_spy_zscore",
        "w_rel_strength_spy",
        "w_rel_strength_spy_zscore",
    ],

    # --- RELATIVE STRENGTH VS QQQ ---
    "relative_strength_qqq": [
        "rel_strength_qqq",
        "rel_strength_qqq_zscore",
        "w_rel_strength_qqq",
    ],

    # --- RELATIVE STRENGTH VS SECTOR ---
    "relative_strength_sector": [
        "rel_strength_sector_zscore",
    ],
    # --- CROSS-SECTIONAL MOMENTUM (8 features) ---
    # --- CROSS-SECTIONAL MOMENTUM (RETURNS) ---
    "xsec_momentum_multi": [
        "xsec_mom_5d_z",
        "xsec_mom_60d_z",
        "w_xsec_mom_1w_z",
        "w_xsec_mom_13w_z",
        "xsec_mom_20d_sect_neutral_z",
    ],

    # --- CROSS-SECTIONAL RANK / DISTRIBUTION ---
    "xsec_rank": [
        "xsec_pct_20d",
        "xsec_pct_60d",
        "w_xsec_pct_4w",
    ],

    # --- SECTOR BREADTH (3 features) ---
    "sector_breadth_candidates": [
        "sector_breadth_pct_above_ma50",
        "w_sector_breadth_pct_above_ma10", "w_sector_breadth_pct_above_ma40",
    ],

    # =========================================================================
    # MANDATORY SPLIT: MACRO FRED (3 groups)
    # =========================================================================
    "macro_rates_curve_extended": [
        "fred_dgs10_chg20d", "fred_dgs10_z60", "w_fred_dgs10_z60",
        "w_fred_t10y2y_z60",
    ],
    "macro_labor_stress_extended": [
        "fred_icsa_z52w", "fred_ccsa_chg4w", "w_fred_ccsa_z52w",
    ],
    "macro_financial_conditions_broad": [
        "fred_nfci_z52w", "w_fred_nfci_chg4w", "fred_bamlh0a0hym2_pct252",
    ],

    # --- REGIME CORRELATION (4 features) ---
    "regime_correlation": [
        "credit_spread_zscore", "yield_curve_zscore",
        "w_credit_spread_zscore", "w_yield_curve_zscore",
    ],

    # =========================================================================
    # MANDATORY SPLIT: DRAWDOWN/RECOVERY (4 groups)
    # =========================================================================
    "drawdown_level": [
        "drawdown_20d", "drawdown_60d", "drawdown_120d", "drawdown_expanding",
        "drawdown_20d_z", "drawdown_60d_z", "drawdown_120d_z",
        "w_drawdown_60d", "w_drawdown_60d_z",
    ],
    "drawdown_timing": [
        "days_since_high_20d_norm", "days_since_high_60d_norm",
        "days_since_high_120d_norm", "w_days_since_high_60d_norm",
    ],
    "recovery": [
        "recovery_20d", "recovery_60d", "recovery_120d",
        "recovery_20d_z", "recovery_60d_z", "recovery_120d_z",
        "w_recovery_60d",
    ],
    "drawdown_dynamics": [
        "drawdown_velocity_20d", "drawdown_velocity_60d", "drawdown_velocity_120d",
        "drawdown_regime", "hl_range_position_60d", "w_drawdown_velocity_60d",
    ],

    # --- GAPS EXTENDED (4 features) ---
    # Extended gap features beyond CORE gap_dynamics
    # NOTE: gap_fill_frac and overnight_ratio are LAGGED 1 day to prevent leakage
    "gaps_extended": [
        "gap_fill_frac",         # Gap fill behavior (lagged 1 day)
        "atr_percent_chg_5",     # ATR change (gap context)
        "overnight_ratio",       # Overnight vs intraday (lagged 1 day)
        "upper_shadow_ratio",    # Upper shadow (gap rejection)
    ],

    # --- TREND QUALITY (8 features) ---
    "trend_quality": [
        "adx_14", "di_plus_14",
        "trend_persist_ema", "quiet_trend", "trend_alignment",
        "w_trend_persist_ema", "w_quiet_trend", "w_trend_alignment",
    ],

    # --- VOLATILITY SQUEEZE (8 features) ---
    "volatility_squeeze_extended": [
        "bb_width_20_2_z60", "squeeze_on_20", "squeeze_on_wide_20",
        "squeeze_intensity_20", "squeeze_release_20", "days_in_squeeze_20",
        "rv_ratio_10_60", "rv_ratio_20_100",
    ],

    # --- DIVERGENCE (12 features) ---
        "divergence_rsi": [
        "rsi_price_div_10d",
        "rsi_price_div_20d",
        "rsi_price_div_cum_10d",
        "rsi_price_div_cum_20d",
        "trend_rsi_div_10d",
        "trend_rsi_div_20d",
        "w_rsi_price_div_20d",
    ],

    "divergence_macd": [
        "macd_price_div_10d",
        "macd_price_div_20d",
        "w_macd_price_div_20d",
    ],

    "divergence_volume": [
        "vol_trend_div_10d",
        "vol_trend_div_20d",
    ],

    # =========================================================================
    # MANDATORY SPLIT: VOLUME/LIQUIDITY (3 groups)
    # =========================================================================
    "illiquidity": [
        "amihud_illiq_ratio", "illiquidity_score", "w_illiquidity_score",
    ],
    "relative_volume": [
        "rel_volume_5d", "rel_volume_10d", "rel_volume_20d",
        "w_rel_volume_5d", "w_rel_volume_10d", "w_rel_volume_20d",
    ],
    "pv_divergence": [
        "pv_divergence_5d", "volume_direction", "volume_trend_10d",
    ],

    # --- VOLATILITY LEVEL / TERM STRUCTURE ---
    "vol_level_structure": [
        "rv_delta_10_60",
        "rv_delta_10_60_z",
        "w_rv_delta_10_60_z",
    ],

    # --- VOLATILITY ACCELERATION ---
    "vol_acceleration": [
        "rv_accel_20",
        "rv_accel_60",
        "w_rv_accel_20",
    ],

    # --- VOLATILITY IMPULSE / SHOCK ---
    "vol_impulse": [
        "rv_impulse_5d_z",
    ],

    # --- BREADTH ADVANCE–DECLINE ---
    "breadth_ad": [
        "sector_breadth_ad_chg_10d",        # short-term participation momentum
        "sector_breadth_ad_slope_20d",      # medium-term participation trend
        "w_sector_breadth_ad_slope_8w",     # longer-term participation trend
    ],

    # --- BREADTH THRUST / MOMENTUM ---
    "breadth_mcclellan": [
        "sector_breadth_mcclellan_chg_5d",      # fast thrust signal
        "sector_breadth_mcclellan_slope_10d",   # smoothed thrust
        "w_sector_breadth_mcclellan_chg_2w",    # weekly thrust confirmation
    ],

    # --- BREADTH PARTICIPATION LEVEL ---
    "breadth_participation": [
        "sector_breadth_pct_ma50_chg_10d",
    ],

    # =========================================================================
    # HEAD GROUPS AS CANDIDATES (merged from all 4 models)
    # These were previously model-specific HEAD groups, now available for all
    # models to select during forward selection.
    # =========================================================================

    

    
    # --- TIME SINCE PEAK ---
    "drawdown_time_since_high": [
        "days_since_high_20d_norm",   # Days since 20d high, normalized
        "w_days_since_high_60d_norm", # Weekly days since 60d high
    ],

    # --- RECOVERY STATE ---
    "drawdown_recovery": [
        "recovery_20d",               # Recovery from 20d low
        "recovery_120d",              # Recovery from 120d low
        "recovery_120d_z",            # Recovery from 120d low, z-scored
    ],
    # --- RELATIVE STRENGTH (merged from LONG_NORMAL + LONG_PARABOLIC) ---
    # --- RELATIVE STRENGTH VS BENCHMARKS ---
    "rel_strength_core": [
        "rel_strength_qqq",            # Relative strength vs QQQ
        "w_rel_strength_qqq",          # Weekly relative strength vs QQQ
        "rel_strength_sector",         # Relative strength vs sector
    ],

    # --- EQUAL-WEIGHT LEADERSHIP / MARKET INTERNALS ---
    "rel_strength_rsp": [
        "rsp_spy_cumret_20",           # RSP vs SPY cumret (20d)
        "rsp_spy_cumret_60",           # RSP vs SPY cumret (60d)
        "w_rsp_spy_cumret_12",         # Weekly RSP vs SPY (12w)
    ],

    # --- CROSS-ASSET REGIME PROXIES ---
    "rel_strength_regime": [
        "gold_spy_ratio_zscore",       # Gold/SPY ratio z-score
    ],

    # --- BESTMATCH / MAPPING-BASED STRENGTH ---
    "rel_strength_bestmatch": [
        "bestmatch_spy_zscore_60",     # Best-match ETF vs SPY z-score
        "w_alpha_mom_sector_60_ema10", # Weekly sector alpha (60d)
    ],
    # --- MACRO SECTOR (merged from LONG_NORMAL + LONG_PARABOLIC) ---
    # Macro and sector conditions
    # --- LABOR MARKET CONDITIONS ---
    "macro_labor": [
        "fred_ccsa_chg4w",      # Claims momentum
        "w_fred_ccsa_z52w",     # Claims regime
    ],

    # --- FINANCIAL CONDITIONS ---
    "head_macro_financial_conditions_nfci": [
        "w_fred_nfci_chg4w",    # NFCI momentum
        "fred_nfci_z52w",       # NFCI regime
    ],

    # --- RATES REGIME ---
    "macro_rates": [
        "fred_dgs10_chg20d",    # Yield momentum
    ],

    # --- SECTOR PARTICIPATION / BREADTH ---
    "macro_sector_participation": [
        "sector_breadth_pct_above_ma50",
        "w_sector_breadth_pct_above_ma10",
        "w_sector_breadth_pct_above_ma40",
    ],

    # --- CANDLESTICK PRESSURE ---
    "price_action_candles": [
        "lower_shadow_ratio",
        "upper_shadow_ratio",
    ],

    # --- VWAP FLOW ---
    "price_action_vwap": [
        "vwap_dist_10d_zscore",
    ],

    # --- PRICE–MOMENTUM DIVERGENCE ---
    "price_action_divergence": [
        "rsi_price_div_20d",
        "w_rsi_price_div_20d",
    ],
    # --- TREND CROSS-SECTIONAL (from LONG_NORMAL) ---
    # --- TREND SLOPE ---
    "trend_slope": [
        "trend_score_slope",      # composite trend slope
        "w_pct_slope_ma_50",      # weekly MA slope
    ],

    # --- CROSS-SECTIONAL MOMENTUM ---
    "xsec_momentum_4w": [
        "w_xsec_mom_4w_z",
    ],

    # --- VOLATILITY CONTEXT ---
    "vol_context": [
        "atr_percent",
    ],
    # --- TREND QUALITY PARABOLIC (from LONG_PARABOLIC) ---
    # --- PARABOLIC: CANDLE PRESSURE ---
    "parabolic_candles": [
        "lower_shadow_ratio",
        "upper_shadow_ratio",
    ],

    # --- PARABOLIC: FLOW / EXTENSION ---
    "parabolic_vwap_extension": [
        "vwap_dist_20d_zscore",
    ],

    # --- PARABOLIC: TREND QUALITY ---
    "parabolic_trend_quality": [
        "trend_score_sign",
        "trend_score_slope",
        "w_pct_slope_ma_50",
    ],

    # --- MOMENTUM DIVERGENCE (from LONG_PARABOLIC) ---
    # --- MA POSITIONING / EXTENSION ---
    "momentum_ma_position": [
        "w_pct_dist_ma_20_z",
        "w_relative_dist_20_50_z",
    ],

    # --- CROSS-SECTIONAL MOMENTUM ---
    "momentum_xsec": [
        "w_xsec_mom_4w_z",
    ],

    # --- MOMENTUM DIVERGENCE ---
    "momentum_divergence": [
        "rsi_price_div_20d",
        "w_rsi_price_div_20d",
    ],

    # --- VOLATILITY CONTEXT ---
    "momentum_vol_context": [
        "atr_percent",
        "bb_width_20_2_z60",
    ],

    # --- RELATIVE WEAKNESS (from SHORT_NORMAL) ---
    # --- TOPPING / FAILURE TIMING ---
    "short_topping_state": [
        "days_since_high_20d_norm",
    ],

    # --- RELATIVE UNDERPERFORMANCE ---
    "short_relative_strength": [
        "rel_strength_sector",
        "w_rel_strength_spy",
    ],

    # --- LEADERSHIP / ROTATION WEAKNESS ---
    "short_leadership_decay": [
        "rsp_spy_cumret_60",
        "w_alpha_mom_sector_60_ema10",
    ],

    # --- RISK-OFF REGIME ---
    "short_macro_risk": [
        "gold_spy_ratio_zscore",
    ],

    # --- MACRO STRESS (from SHORT_NORMAL) ---
    # Macro stress and breadth deterioration
    "head_macro_stress": [
        "w_fred_nfci_chg4w",          # Weekly NFCI 4-week change
        "sector_breadth_mcclellan_osc", # McClellan oscillator
        "pv_divergence_5d",           # Price-volume divergence (5d)
    ],

    # --- BREAKDOWN SIGNALS (from SHORT_NORMAL) ---
    # --- BREAKDOWN: CANDLE PRESSURE ---
    "breakdown_candles": [
        "lower_shadow_ratio",
        "upper_shadow_ratio",
    ],

    # --- BREAKDOWN: TREND STATE ---
    "breakdown_trend_state": [
        "trend_score_sign",
        "w_pct_slope_ma_50",
    ],

    # --- BREAKDOWN: CROSS-SECTIONAL WEAKNESS ---
    "breakdown_xsec_weakness": [
        "w_xsec_mom_4w_z",
    ],

    # --- BREAKDOWN: VOLATILITY CONTEXT ---
    "breakdown_vol_context": [
        "atr_percent",
    ],
    # --- DRAWDOWN PANIC (from SHORT_PARABOLIC) ---
    # Panic and drawdown acceleration for parabolic shorts
    "head_drawdown_panic": [
        "drawdown_60d_z",             # Drawdown from 60d high, z-scored
        "drawdown_velocity_60d",      # Speed of drawdown (60d window)
        "pct_dist_ma_100_z",          # Distance to 100-day MA
    ],

    # --- RELATIVE STRESS (from SHORT_PARABOLIC) ---
    # --- CORE RELATIVE WEAKNESS ---
    "relative_weakness_core": [
        "rel_strength_spy",
        "w_rel_strength_qqq",
    ],

    # --- ALPHA / EXCESS RETURN DECAY ---
    "relative_alpha_decay": [
        "alpha_mom_qqq_60_ema10",
        "w_alpha_mom_sector_60_ema10",
    ],

    # --- BESTMATCH / MAPPING STRESS ---
    "relative_bestmatch_stress": [
        "bestmatch_spy_cumret_60",
    ],

    # --- CROSS-SECTIONAL RANK / PEER CONTEXT ---
    "relative_xsec_rank": [
        "xsec_pct_20d",
    ],
    # --- MACRO REGIME (from SHORT_PARABOLIC) ---
    # Macro regime shift signals
    "head_macro_regime": [
        "fred_dgs10_z60",             # 10Y yield z-score (60d)
        "w_fred_nfci_chg4w",          # Weekly NFCI 4-week change
        "w_fred_t10y2y_z60",          # Weekly yield curve z-score
        "sector_breadth_mcclellan_osc", # McClellan oscillator
        "w_sector_breadth_pct_above_ma40", # Weekly % above 40-day MA
        "vwap_dist_20d_zscore",       # VWAP distance z-score (20d)
        "trend_score_sign",           # Trend score sign
    ],
    # --- TREND: CURVATURE / ACCELERATION ---
    "trend_curvature": [
        "macd_hist_deriv_ema3",
        "w_macd_hist_deriv_ema3",
    ],

    # --- TREND: GRANULAR COMPOSITE STATE ---
    # Use if your current trend_score_sign/slope feel too coarse.
    "trend_state_granular": [
        "trend_score_granular",
        "w_trend_score_granular",
    ],

    # --- TREND: SHORT-END SLOPE (adds a 10d “kick” you don’t have) ---
    # You already have pct_slope_ma_20 and pct_slope_ma_100 in CORE.
    "trend_slope_short_end": [
        "pct_slope_ma_10",
        "w_pct_slope_ma_10",
    ],

    # --- TREND: LONG-END SLOPE (adds true long trend anchor you don’t have) ---
    # You have 100d slope; this adds 200d slope as a separate hypothesis.
    "trend_slope_long_end": [
        "pct_slope_ma_200",
        "w_pct_slope_ma_200",
    ],
      # --- BREAKOUT STATE: UPSIDE/ DOWNSIDE PRESSURE (daily) ---
    "breakout_state_daily": [
        "breakout_up_10d",
        "breakout_up_20d",
        "breakout_dn_10d",
        "breakout_dn_20d",
    ],

    # --- BREAKOUT STATE: UPSIDE/DOWNSIDE PRESSURE (weekly) ---
    "breakout_state_weekly": [
        "w_breakout_up_10d",
        "w_breakout_up_20d",
        "w_breakout_dn_10d",
        "w_breakout_dn_20d",
    ],

    # --- RANGE EXPANSION (daily + weekly) ---
    "range_expansion_regime": [
        "range_expansion_10d",
        "range_expansion_20d",
        "w_range_expansion_10d",
        "w_range_expansion_20d",
    ],

    # --- RANGE EXTREMENESS (z-scored range as “compression/expansion level”) ---
    "range_extremeness": [
        "range_z_10d",
        "range_z_20d",
        "w_range_z_10d",
        "w_range_z_20d",
    ],

    # --- MULTI-HORIZON RANGE POSITIONING (adds 10d/20d positioning) ---
    # NOTE: You already use pos_in_5d_range in CORE.
    "range_position_multi": [
        "pos_in_10d_range",
        "w_pos_in_10d_range",
        "w_pos_in_20d_range",
    ],
    # --- VOL: TREND / SLOPE OF REALIZED VOL ---
    "vol_trend_slopes": [
        "rv60_slope_norm",
        "rv100_slope_norm",
        "w_rv60_slope_norm",
        "w_rv100_slope_norm",
    ],

    # --- VOL: CROSS-SECTIONAL / RELATIVE REGIME ---
    # These are different from simple level z-scores: they encode “vol relative to peers”.
    "vol_regime_relative": [
        "vol_regime_cs_median",
        "vol_regime_rel",
    ],
    # --- VOLUME FLOW: ACCUMULATION / DISTRIBUTION ---
    "flow_obv_state": [
        "obv_z_60",
        "w_obv_z_60",
    ],

    # --- VOLUME SHOCK: DIRECTIONAL / NORMALIZED ---
    # You have volshock_ema; this adds explicit “sign” and standardized magnitude.
    "volshock_directional": [
        "volshock_z",
        "volshock_dir",
        "w_volshock_z",
        "w_volshock_dir",
    ],

    # --- LIQUIDITY CAPACITY: DOLLAR VOLUME ---
    "dollar_volume_capacity": [
        "rdollar_vol_20",
        "w_rdollar_vol_20",
    ],
    # --- EFFICIENCY (weekly) ---
    "range_efficiency_weekly": [
        "w_range_efficiency",
    ],
    # --- VOL REGIME: TERM STRUCTURE / RELATIVE FEAR ---
  "vol_term_structure": [
      "vix_vxn_spread",
    ],

    # --- VOL REGIME: MOMENTUM / RATE OF CHANGE ---
    "vol_momentum": [
        "vix_change_5d",
        "vix_change_20d",
        "w_vix_change_4w",
    ],

    # --- VOL REGIME: LEVEL VS MA (different from z-score) ---
    "vol_level_vs_trend": [
        "vix_ma20_ratio",
        "w_vix_ma4_ratio",
    ],
    # --- SECTOR ROTATION RATIOS (risk-on/off within equities) ---
    "sector_rotation_ratios": [
        "financials_utilities_ratio",
        "w_financials_utilities_ratio",
        "tech_spy_ratio",
        "w_tech_spy_ratio",
    ],

    # --- COMMODITY / FX PRESSURE (macro impulse channel) ---
    "macro_impulse_commodity_fx": [
        "oil_momentum_20d",
        "dollar_momentum_20d",
        "w_dollar_momentum_20d",
        "dollar_percentile_252d",
    ],
}


# =============================================================================
# INTERACTION_TEMPLATES - Template-based group-to-group interactions
# =============================================================================
# Interaction templates define hypothesis-driven feature interactions between
# existing selection groups. Each template specifies:
#   - parents: tuple of (group_name_A, group_name_B) from CORE/HEAD/CANDIDATE groups
#   - type: "gate", "product", or "signed_gate"
#   - base_features: specific features from parent groups to use
#   - gate_features: features to use as gates/conditions (for gate types)
#   - threshold: optional threshold for gate activation (default: median)
#
# Interaction Types:
#   - "gate": base_feature * I(gate_feature > threshold) - binary gating
#   - "signed_gate": base_feature * sign(gate_feature) - direction gating
#   - "product": base_feature_a * base_feature_b - multiplicative confirmation
#
# Naming Convention: ix__{template_name}__{base_feat}__gated__{gate_feat}
#                    ix__{template_name}__{feat_a}__x__{feat_b}

INTERACTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # MOMENTUM × VOLATILITY STATE GATE
    # =========================================================================
    "momentum_x_vol_gate": {
        "parents": ("momentum_quality", "volatility_state"),
        "type": "gate",
        "base_features": [
            "rsi_14",
            "adx_14",
            "chop_14",
        ],
        "gate_features": [
            "squeeze_intensity_20",
            "rv_z_60",
        ],
        "description": "Momentum indicators gated by volatility state",
    },

    # =========================================================================
    # TREND × VOLATILITY STATE GATE
    # =========================================================================
    "trend_x_vol_gate": {
        "parents": ("trend_strength", "volatility_state"),
        "type": "signed_gate",
        "base_features": [
            "pct_slope_ma_20",
            "pct_slope_ma_100",
            "w_macd_histogram",
        ],
        "gate_features": [
            "rv_z_60",
            "squeeze_intensity_20",
        ],
        "description": "Trend strength gated by volatility state (signed)",
    },

    # =========================================================================
    # BREADTH × TREND CONFIRMATION
    # =========================================================================
    "breadth_x_trend_confirm": {
        "parents": ("sector_breadth", "trend_strength"),
        "type": "product",
        "base_features": [
            "sector_breadth_ad_line",
            "sector_breadth_pct_above_ma200",
            "w_sector_breadth_mcclellan_osc",
        ],
        "gate_features": [
            "pct_slope_ma_20",
            "w_macd_histogram",
        ],
        "description": "Breadth multiplied by trend confirmation",
    },

    # =========================================================================
    # DRAWDOWN DEPTH × BREADTH MOTION
    # =========================================================================
    # Aligned to your documented drawdown splits + existing breadth group.
    # If you later add a dedicated "breadth_motion" group, you can revert parents.
    "drawdown_depth_x_breadth": {
        "parents": ("drawdown_depth", "sector_breadth"),
        "type": "gate",
        "base_features": [
            "drawdown_expanding",
            "drawdown_60d_z",
        ],
        "gate_features": [
            "sector_breadth_ad_chg_10d",
            "sector_breadth_mcclellan_chg_5d",
        ],
        "description": "Drawdown depth gated by breadth momentum",
    },

    # =========================================================================
    # ALPHA × MACRO REGIME (CREDIT/LABOR)
    # =========================================================================
    "alpha_x_macro_regime": {
        "parents": ("alpha_momentum", "macro_credit_labor"),
        "type": "signed_gate",
        "base_features": [
            "alpha_mom_qqq_20_ema10",
            "alpha_mom_sector_20_ema10",
            "xsec_mom_20d_z",
        ],
        "gate_features": [
            "fred_bamlh0a0hym2_z60",
            "w_fred_icsa_z52w",
        ],
        # Your compute code must honor this if you want inverted behavior.
        "invert_gate": True,
        "description": "Alpha/relative signals gated by macro stress (inverted)",
    },

    # =========================================================================
    # PRICE POSITION × VOLATILITY REGIME (Mean Reversion Context)
    # =========================================================================
    "price_position_x_regime": {
        "parents": ("price_position", "volatility_regime"),
        "type": "gate",
        "base_features": [
            "pct_dist_ma_20_z",
            "pct_dist_ma_50_z",
            "pos_in_20d_range",
        ],
        "gate_features": [
            "vix_zscore_60d",
            "vix_percentile_252d",
        ],
        "invert_gate": True,  # low VIX => mean reversion works better
        "description": "Price position/mean reversion gated by VIX regime (inverted)",
    },

    # =========================================================================
    # RANGE/BREAKOUT × VOLATILITY STATE (Squeeze Release)
    # =========================================================================
    "breakout_x_squeeze": {
        "parents": ("range_breakout", "volatility_state"),
        "type": "product",
        "base_features": [
            "gap_fill_frac",
            "range_efficiency",
            "pos_in_5d_range",
        ],
        "gate_features": [
            "squeeze_release_20",
            "days_in_squeeze_20",
        ],
        "description": "Breakout/range signals amplified by squeeze release",
    },

    # =========================================================================
    # GAP DYNAMICS × VOLATILITY STATE
    # =========================================================================
    "gap_x_vol_state": {
        "parents": ("gap_dynamics", "volatility_state"),
        "type": "gate",
        "base_features": [
            "gap_atr_ratio",
            "gap_atr_ratio_raw",
            "overnight_ret",
        ],
        "gate_features": [
            "squeeze_release_20",
            "bb_width_20_2",
        ],
        "description": "Gap signals gated by volatility state",
    },

    # =========================================================================
    # MICROSTRUCTURE × VOLUME SHOCK
    # =========================================================================
    "microstructure_x_volume": {
        "parents": ("microstructure_position", "volume_shock"),
        "type": "product",
        "base_features": [
            "vwap_dist_5d_zscore",
            "vwap_dist_20d_zscore",
        ],
        "gate_features": [
            "volshock_ema",
            "w_volshock_ema",
        ],
        "description": "VWAP distance amplified by volume shock",
    },

    # =========================================================================
    # MOMENTUM QUALITY × VOLATILITY REGIME (VIX)
    # =========================================================================
    # Fixed direction: momentum signals gated by VIX regime.
    "vol_regime_x_momentum": {
        "parents": ("momentum_quality", "volatility_regime"),
        "type": "signed_gate",
        "base_features": [
            "rsi_14",
            "adx_14",
        ],
        "gate_features": [
            "vix_percentile_252d",
            "vix_zscore_60d",
        ],
        "description": "Momentum gated by VIX regime (signed)",
    },
}

# ============================================================================
# Public API (backward-compatible with your current call sites)
# ============================================================================

def get_interaction_templates() -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all interaction templates."""
    return {k: dict(v) for k, v in INTERACTION_TEMPLATES.items()}


def generate_interaction_feature_names(
    template_name: str,
    template: Dict[str, Any],
) -> List[str]:
    """Generate deterministic feature names from an interaction template."""
    interaction_type = template["type"]
    base_features: List[str] = template.get("base_features", [])
    gate_features: List[str] = template.get("gate_features", [])

    feature_names: List[str] = []

    if interaction_type in ("gate", "signed_gate"):
        # ix__{template}__{base}__gated__{gate}
        for base_feat in base_features:
            for gate_feat in gate_features:
                feature_names.append(
                    f"ix__{template_name}__{base_feat}__gated__{gate_feat}"
                )

    elif interaction_type == "product":
        # ix__{template}__{base}__x__{gate}
        for base_feat in base_features:
            for gate_feat in gate_features:
                feature_names.append(
                    f"ix__{template_name}__{base_feat}__x__{gate_feat}"
                )
    else:
        raise ValueError(
            f"Unknown interaction type '{interaction_type}' "
            f"for template '{template_name}'."
        )

    return feature_names


def get_all_template_feature_names() -> Dict[str, List[str]]:
    """Get all interaction feature names from all templates."""
    return {
        template_name: generate_interaction_feature_names(template_name, template)
        for template_name, template in INTERACTION_TEMPLATES.items()
    }


def get_template_parents() -> Dict[str, Tuple[str, str]]:
    """Get parent group pairs for each template."""
    return {
        name: template["parents"]
        for name, template in INTERACTION_TEMPLATES.items()
    }


def get_eligible_templates(selected_groups: Set[str]) -> List[str]:
    """Get templates eligible for selection based on selected parent groups.

    A template is eligible if BOTH parent groups are present in selected_groups.
    Ordering follows INTERACTION_TEMPLATES definition order.
    """
    eligible: List[str] = []
    for template_name, template in INTERACTION_TEMPLATES.items():
        parent_a, parent_b = template["parents"]
        if parent_a in selected_groups and parent_b in selected_groups:
            eligible.append(template_name)
    return eligible


# ============================================================================
# Optional utilities (non-breaking): validation + introspection
# ============================================================================

def validate_templates(
    available_groups: Optional[Set[str]] = None,
    available_features: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """Validate template definitions against known groups/features.

    Returns:
        Dict[template_name, list_of_problems]
    """
    problems: Dict[str, List[str]] = {}

    for name, t in INTERACTION_TEMPLATES.items():
        errs: List[str] = []

        # Type
        ttype = t.get("type")
        if ttype not in ("gate", "signed_gate", "product"):
            errs.append(f"Invalid type: {ttype!r}")

        # Parents
        parents = t.get("parents")
        if not isinstance(parents, tuple) or len(parents) != 2:
            errs.append(f"Invalid parents: {parents!r}")
        else:
            if available_groups is not None:
                pa, pb = parents
                if pa not in available_groups:
                    errs.append(f"Unknown parent group: {pa!r}")
                if pb not in available_groups:
                    errs.append(f"Unknown parent group: {pb!r}")

        # Features lists
        base = t.get("base_features", [])
        gate = t.get("gate_features", [])
        if not isinstance(base, list) or not all(isinstance(x, str) for x in base):
            errs.append("base_features must be a list[str]")
        if not isinstance(gate, list) or not all(isinstance(x, str) for x in gate):
            errs.append("gate_features must be a list[str]")
        if len(base) == 0:
            errs.append("base_features is empty")
        if len(gate) == 0:
            errs.append("gate_features is empty")

        if available_features is not None:
            missing_base = [f for f in base if f not in available_features]
            missing_gate = [f for f in gate if f not in available_features]
            if missing_base:
                errs.append(f"Missing base features: {missing_base}")
            if missing_gate:
                errs.append(f"Missing gate features: {missing_gate}")

        if errs:
            problems[name] = errs

    return problems


def get_template_flag(template: Dict[str, Any], flag_name: str, default: Any = None) -> Any:
    """Convenience accessor for optional template flags (e.g., invert_gate)."""
    return template.get(flag_name, default)

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
        # Volatility acceleration (2nd derivatives)
        "rv_delta_10_60", "rv_delta_10_60_z", "rv_accel_20", "rv_accel_60",
        "rv_impulse_5d_z", "w_rv_delta_10_60_z", "w_rv_accel_20",
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
        # Breadth motion (slopes and changes)
        "sector_breadth_ad_chg_10d", "sector_breadth_ad_slope_20d",
        "sector_breadth_mcclellan_chg_5d", "sector_breadth_mcclellan_slope_10d",
        "sector_breadth_pct_ma50_chg_10d",
        "w_sector_breadth_ad_slope_8w", "w_sector_breadth_mcclellan_chg_2w",
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
        # Keep curvature + granular score active (removed from retired):
        # "macd_hist_deriv_ema3", "w_macd_hist_deriv_ema3",
        # "trend_score_granular", "w_trend_score_granular",
        # "pct_slope_ma_10", "w_pct_slope_ma_10",
        # "pct_slope_ma_200", "w_pct_slope_ma_200",

        # RSI variants are redundant given rsi_14 + other momentum/quality features
        "rsi_21", "rsi_30", "w_rsi_14", "w_rsi_21",

        # Mid-slope grid is mostly redundant once you have 20/100/200 + MA distance families
        "pct_slope_ma_30", "pct_slope_ma_50", "pct_slope_ma_75", "pct_slope_ma_150",
        "w_pct_slope_ma_20", "w_pct_slope_ma_30", "w_pct_slope_ma_75",
        "w_pct_slope_ma_100", "w_pct_slope_ma_150",

        # Often redundant / unstable if you already have trend_score_slope and sign proxies
        "w_trend_score_slope",
    ],

    "range_breakout": [
        # You reintroduced most of the 10d/20d variants; keep the 5d grid retired as redundancy/noise.
        "breakout_up_5d", "breakout_dn_5d",
        "w_breakout_up_5d", "w_breakout_dn_5d",
        "range_expansion_5d", "w_range_expansion_5d",
        "range_z_5d", "w_range_z_5d",

        # Already in CORE, so keep retired to avoid duplicate computation lists:
        # (Core should be the source of truth for computed features.)
        "pos_in_5d_range",
        "w_pos_in_5d_range",
    ],

    "volatility": [
        # You reintroduced relative regime + vol slopes; remove from retired:
        # "vol_regime_cs_median", "vol_regime_rel",
        # "rv60_slope_norm", "rv100_slope_norm",
        # "w_rv60_slope_norm", "w_rv100_slope_norm",

        # vol_regime base is redundant with vol_regime_ema10 + RV ratios
        "vol_regime",

        # These are redundant with rv_z_60, vol_level_structure, vol_acceleration, vol_impulse, squeeze sets
        "vol_z_20", "vol_z_60", "rvol_20", "w_rvol_20",
        "w_rv_z_60", "w_vol_z_60",
        "w_vol_regime", "w_vol_regime_ema10", "w_vol_regime_rel",
    ],

    "volume": [
        # You reintroduced these; remove from retired:
        # "obv_z_60", "w_obv_z_60",
        # "volshock_z", "volshock_dir", "w_volshock_z", "w_volshock_dir",
        # "rdollar_vol_20", "w_rdollar_vol_20",
    ],

    "liquidity": [
        # w_range_efficiency was reintroduced; remove it from retired:
        # "w_range_efficiency",

        # vwap_dist_10d_zscore already exists in candidate groups; don't double-list as retired unless you truly want it excluded
        # If you want to exclude it, keep it retired. Otherwise remove it.
        "w_vwap_dist_20d_zscore",
    ],

    "alpha": [
        # If you want to stay “hypothesis-light” here, keep beta surface retired.
        # You already have w_beta_qqq in CORE; bringing the full beta surface back often adds redundancy.
        "beta_market", "beta_qqq", "beta_bestmatch", "beta_breadth",
        "beta_spy_simple", "beta_qqq_simple", "beta_sector",
        "w_beta_market", "w_beta_bestmatch", "w_beta_breadth",
        "w_beta_spy_simple", "w_beta_qqq_simple",
    ],

    "macro": [
        # You reintroduced these; remove from retired:
        # "vix_ma20_ratio", "vix_vxn_spread", "vix_change_5d", "vix_change_20d", "w_vix_ma4_ratio", "w_vix_change_4w",

        # These remain redundant given your VIX percentile/z and the reintroduced momentum/term-structure set
        "vix_regime",
        "w_vix_percentile_52w", "w_vix_zscore_12w", "w_vix_regime",
        "w_vxn_percentile_252d",
    ],

    "spread_features": [
        # Already represented in zscore forms or core:
        "copper_gold_ratio", "w_copper_gold_ratio",
        "gold_spy_ratio", "w_gold_spy_ratio",
        "cyclical_defensive_ratio",
        # These were reintroduced; remove from retired:
        # "financials_utilities_ratio", "w_financials_utilities_ratio",
        # "tech_spy_ratio", "w_tech_spy_ratio",
        # "oil_momentum_20d", "dollar_momentum_20d", "w_dollar_momentum_20d", "dollar_percentile_252d",
    ],
}


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
# HELPER FUNCTIONS - Flatten groups to feature lists
# =============================================================================

def _flatten_groups(groups: Dict[str, List[str]]) -> List[str]:
    """Flatten a dict of groups to a deduplicated list of features."""
    seen = set()
    result = []
    for group_features in groups.values():
        for feat in group_features:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
    return result


def _flatten_head_groups(model_key: ModelKey) -> List[str]:
    """Flatten HEAD_GROUPS for a specific model to a deduplicated list."""
    if model_key not in HEAD_GROUPS:
        return []
    return _flatten_groups(HEAD_GROUPS[model_key])


# =============================================================================
# LEGACY COMPUTED VARIABLES - For backwards compatibility
# =============================================================================

# CORE_FEATURES: Flattened list from CORE_GROUPS (legacy interface)
CORE_FEATURES: List[str] = _flatten_groups(CORE_GROUPS)

# HEAD_FEATURES: Flattened dict from HEAD_GROUPS (legacy interface)
HEAD_FEATURES: Dict[ModelKey, List[str]] = {
    mk: _flatten_head_groups(mk) for mk in ModelKey.all_keys()
}

# Build RETIRED_FEATURES from modules, but exclude any HEAD_FEATURES
_all_head_features: Set[str] = set()
for _head_list in HEAD_FEATURES.values():
    _all_head_features.update(_head_list)

RETIRED_FEATURES: Set[str] = set()
for module_features in RETIRED_FEATURES_BY_MODULE.values():
    RETIRED_FEATURES.update(module_features)
# Remove any features that are now HEAD_FEATURES (reactivated for 4-model system)
RETIRED_FEATURES -= _all_head_features

# EXPANSION_CANDIDATES: Flattened from CANDIDATE_GROUPS (legacy interface)
EXPANSION_CANDIDATES: Dict[str, List[str]] = {
    k: list(v) for k, v in CANDIDATE_GROUPS.items()
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
# GROUP-FIRST RETRIEVAL FUNCTIONS (NEW API)
# =============================================================================

def get_core_groups() -> Dict[str, List[str]]:
    """Return all core groups (shared across all models)."""
    return {k: list(v) for k, v in CORE_GROUPS.items()}


def get_head_groups(model_key: ModelKey) -> Dict[str, List[str]]:
    """
    Return the head groups for a specific model key.

    Args:
        model_key: The model key (LONG_NORMAL, LONG_PARABOLIC, etc.)

    Returns:
        Dict mapping group name to list of features
    """
    if model_key not in HEAD_GROUPS:
        raise ValueError(
            f"Unknown model_key: {model_key}. Valid: {list(HEAD_GROUPS.keys())}"
        )
    return {k: list(v) for k, v in HEAD_GROUPS[model_key].items()}


def get_candidate_groups() -> Dict[str, List[str]]:
    """Return all candidate groups available for selection."""
    return {k: list(v) for k, v in CANDIDATE_GROUPS.items()}


def get_all_groups(model_key: ModelKey) -> Dict[str, List[str]]:
    """
    Return all groups for a model (CORE + HEAD + CANDIDATE).

    Args:
        model_key: The model key

    Returns:
        Dict mapping group name to list of features
    """
    result = get_core_groups()
    result.update(get_head_groups(model_key))
    result.update(get_candidate_groups())
    return result


def get_baseline_groups(model_key: ModelKey) -> Dict[str, List[str]]:
    """
    Return baseline groups for a model (CORE + HEAD only).

    Args:
        model_key: The model key

    Returns:
        Dict mapping group name to list of features
    """
    result = get_core_groups()
    result.update(get_head_groups(model_key))
    return result


def get_group_names(group_type: str = "all", model_key: Optional[ModelKey] = None) -> List[str]:
    """
    Get list of group names.

    Args:
        group_type: One of "core", "head", "candidate", "interaction", "all"
        model_key: Required if group_type is "head" or "all"

    Returns:
        List of group names
    """
    if group_type == "core":
        return list(CORE_GROUPS.keys())
    elif group_type == "head":
        if model_key is None:
            raise ValueError("model_key required for head groups")
        return list(HEAD_GROUPS.get(model_key, {}).keys())
    elif group_type == "candidate":
        return list(CANDIDATE_GROUPS.keys())
    elif group_type == "template":
        return list(INTERACTION_TEMPLATES.keys())
    elif group_type == "all":
        if model_key is None:
            raise ValueError("model_key required for all groups")
        return (
            list(CORE_GROUPS.keys()) +
            list(HEAD_GROUPS.get(model_key, {}).keys()) +
            list(CANDIDATE_GROUPS.keys())
        )
    else:
        raise ValueError(f"Unknown group_type: {group_type}")


def validate_group_sizes(
    min_size: int = 3,
    max_size: int = 12,
    interaction_min_size: int = 1
) -> Dict[str, any]:
    """
    Validate that all groups meet size requirements.

    Args:
        min_size: Minimum features for regular groups (default: 3)
        max_size: Maximum features for any group (default: 12)
        interaction_min_size: Minimum for interaction groups (default: 1)

    Returns:
        Dict with validation results
    """
    issues = []

    # Check CORE_GROUPS
    for name, features in CORE_GROUPS.items():
        size = len(features)
        if size < min_size:
            issues.append(f"CORE_GROUPS['{name}']: {size} features (min {min_size})")
        elif size > max_size:
            issues.append(f"CORE_GROUPS['{name}']: {size} features (max {max_size})")

    # Check HEAD_GROUPS (interaction groups allowed to be smaller)
    for model_key, groups in HEAD_GROUPS.items():
        for name, features in groups.items():
            size = len(features)
            # Interaction groups can be smaller
            effective_min = interaction_min_size if 'interaction' in name.lower() else min_size
            if size < effective_min:
                issues.append(f"HEAD_GROUPS[{model_key}]['{name}']: {size} features (min {effective_min})")
            elif size > max_size:
                issues.append(f"HEAD_GROUPS[{model_key}]['{name}']: {size} features (max {max_size})")

    # Check CANDIDATE_GROUPS
    for name, features in CANDIDATE_GROUPS.items():
        size = len(features)
        if size < min_size:
            issues.append(f"CANDIDATE_GROUPS['{name}']: {size} features (min {min_size})")
        elif size > max_size:
            issues.append(f"CANDIDATE_GROUPS['{name}']: {size} features (max {max_size})")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_core_groups": len(CORE_GROUPS),
        "total_candidate_groups": len(CANDIDATE_GROUPS),
        "total_interaction_templates": len(INTERACTION_TEMPLATES),
    }


# =============================================================================
# MODEL-AWARE FEATURE RETRIEVAL FUNCTIONS (LEGACY API)
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

META_COLUMNS = ['symbol', 'date']
REQUIRED_FEATURES = ['atr_percent']


def get_registry_features(model_keys: Optional[List[ModelKey]] = None) -> Set[str]:
    """
    Get features from feature registries.

    This function reads the resolved_features from feature registry files
    (artifacts/<model>/features.json) and returns the union of all features.

    Args:
        model_keys: List of models to include. If None, uses all 4 models.

    Returns:
        Set of feature names from all specified registries.
        Returns empty set if no registries exist.
    """
    from src.features.registry import load_registry, registry_exists, get_registry_path

    if model_keys is None:
        model_keys = list(ModelKey.all_keys())

    all_features: Set[str] = set()

    for model_key in model_keys:
        model_name = model_key.value if isinstance(model_key, ModelKey) else model_key
        if registry_exists(model_name):
            try:
                registry = load_registry(get_registry_path(model_name))
                features = registry.get("resolved_features", [])
                all_features.update(features)
            except Exception:
                pass  # Silently skip invalid registries

    return all_features


def get_output_features(
    model_key: Optional[ModelKey] = None,
    use_registry: bool = False
) -> Set[str]:
    """
    Get the curated list of features to include in pipeline output.

    Args:
        model_key: If provided, include CORE + HEAD for this model.
                   If None, include CORE + all heads (legacy behavior).
        use_registry: If True, use feature registries as source of truth.
                      Falls back to base_features.py if no registries exist.

    Returns:
        Set of feature names to include in output
    """
    # Try registry first if requested
    if use_registry:
        registry_features = get_registry_features(
            [model_key] if model_key else None
        )
        if registry_features:
            # Add meta columns and required features
            output_features = set(META_COLUMNS + REQUIRED_FEATURES)
            output_features.update(registry_features)
            return output_features
        # Fall through to base_features.py if no registries

    # Legacy behavior: use base_features.py definitions
    if model_key is not None:
        base = get_featureset(model_key)
    else:
        base = get_all_selectable_features(model_key=None)

    expansion = get_expansion_candidates(flat=True)
    output_features = set(META_COLUMNS + REQUIRED_FEATURES + base + expansion)

    # Remove excluded raw data columns (ret, OHLC, volume, etc.)
    output_features -= set(EXCLUDED_FEATURES)

    return output_features


def filter_output_columns(
    df,
    keep_all: bool = False,
    exclude_retired: bool = False,
    model_key: Optional[ModelKey] = None,
    use_registry: bool = False
):
    """
    Filter DataFrame columns to only include curated output features.

    Args:
        df: DataFrame with computed features
        keep_all: If True, return all columns (no filtering for curated set)
        exclude_retired: If True, also exclude retired features
        model_key: If provided, filter to CORE + HEAD for this model
        use_registry: If True, use feature registries as source of truth.
                      This filters to only features needed by the ML models.

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
        output_features = get_output_features(model_key, use_registry=use_registry)
        cols_to_keep &= output_features
        if use_registry:
            logger.info(f"Filtering to {len(output_features)} registry features")

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
