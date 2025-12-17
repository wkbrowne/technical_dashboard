#!/usr/bin/env python3
"""
Data Quality Check Script

Analyzes pipeline output files and provides actionable insights:
- BASE_FEATURES V2 validation (~38 curated core features from base_features.py)
- EXPANSION_CANDIDATES V2 validation (~200 features for forward selection)
- Feature coverage and NaN rates by category
- Data quality issues (infinite values, missing features)
- Targets file validation
- Recommendations for fixing issues

V2 Feature Set (updated Dec 2024):
- BASE_FEATURES: 38 curated features covering trend, volatility, relative perf, macro
- EXPANSION_CANDIDATES: ~200 features organized by category for feature selection
- Output filtering: Pipeline now outputs curated ~200 features (vs ~480 raw)

Usage:
    conda run -n stocks_predictor python run_data_quality.py
    conda run -n stocks_predictor python run_data_quality.py --verbose
    conda run -n stocks_predictor python run_data_quality.py --features artifacts/features_complete.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Import BASE_FEATURES and EXPANSION_CANDIDATES as the golden references
try:
    from src.feature_selection.base_features import (
        BASE_FEATURES,
        FEATURE_CATEGORIES as BASE_FEATURE_CATEGORIES,
        EXPANSION_CANDIDATES,
        validate_features as validate_base_features,
        get_expansion_candidates,
    )
    HAS_BASE_FEATURES = True
except ImportError:
    HAS_BASE_FEATURES = False
    BASE_FEATURES = []
    BASE_FEATURE_CATEGORIES = {}
    EXPANSION_CANDIDATES = {}
    get_expansion_candidates = lambda flat=False: [] if flat else {}

# =============================================================================
# FEATURE DEFINITIONS - Descriptions and expected behavior
# =============================================================================

# Complete feature descriptions including all BASE_FEATURES V2
# Updated to match the curated 38 features in base_features.py
FEATURE_DESCRIPTIONS = {
    # ==========================================================================
    # BASE_FEATURES V2 - Curated core features (~38 total)
    # Streamlined set for multi-timeframe coverage, diverse signal types,
    # low correlation, and strong feature selection performance
    # ==========================================================================

    # === TREND / MOMENTUM (4 BASE_FEATURES) ===
    "rsi_14": "14-day RSI (0-100, >70 overbought, <30 oversold) [BASE_FEATURE]",
    "w_macd_histogram": "Weekly MACD histogram - momentum trend [BASE_FEATURE]",
    "trend_score_sign": "Multi-MA alignment direction (+1/-1 per MA) [BASE_FEATURE]",
    "trend_score_slope": "Rate of change of trend score [BASE_FEATURE]",

    # === TREND SLOPES (3 BASE_FEATURES) ===
    "pct_slope_ma_20": "20-day MA slope as % of price (short-term trend) [BASE_FEATURE]",
    "pct_slope_ma_100": "100-day MA slope as % of price (medium-term trend) [BASE_FEATURE]",
    "w_pct_slope_ma_50": "Weekly 50-day MA slope [BASE_FEATURE]",

    # === PRICE POSITION / MEAN REVERSION (5 BASE_FEATURES) ===
    "pct_dist_ma_20_z": "Z-scored distance from 20-day MA [BASE_FEATURE]",
    "pct_dist_ma_50_z": "Z-scored distance from 50-day MA [BASE_FEATURE]",
    "relative_dist_20_50_z": "Relative position between 20/50 MAs (z-scored) [BASE_FEATURE]",
    "pos_in_20d_range": "Position in 20-day high-low range (0-1) [BASE_FEATURE]",
    "vwap_dist_20d_zscore": "Z-scored distance from 20d VWAP [BASE_FEATURE]",

    # === VOLATILITY / REGIME (5 BASE_FEATURES) ===
    "atr_percent": "ATR as % of price (REQUIRED for targets) [BASE_FEATURE]",
    "vol_regime_ema10": "10-day EMA smoothed volatility regime [BASE_FEATURE]",
    "rv_z_60": "60-day realized vol z-score [BASE_FEATURE]",
    "vix_zscore_60d": "VIX z-score vs 60-day history (market fear) [BASE_FEATURE]",
    "w_vix_vxn_spread": "Weekly VIX-VXN spread (tech vs broad vol) [BASE_FEATURE]",

    # === RELATIVE PERFORMANCE / CROSS-SECTION (6 BASE_FEATURES) ===
    "alpha_mom_spy_20_ema10": "20-day alpha momentum vs SPY (EMA smoothed) [BASE_FEATURE]",
    "alpha_mom_sector_20_ema10": "20-day alpha momentum vs sector [BASE_FEATURE]",
    "w_alpha_mom_spy_20_ema10": "Weekly alpha vs SPY [BASE_FEATURE]",
    "rel_strength_sector": "Relative strength vs sector ETF [BASE_FEATURE]",
    "xsec_mom_20d_z": "20-day momentum cross-sectional z-score [BASE_FEATURE]",
    "w_xsec_mom_4w_z": "Weekly cross-sectional momentum z-score [BASE_FEATURE]",

    # === MARKET BREADTH (1 BASE_FEATURE) ===
    "w_ad_ratio_universe": "Weekly advance-decline ratio [BASE_FEATURE]",

    # === LIQUIDITY / VOLUME (2 BASE_FEATURES) ===
    "upper_shadow_ratio": "Upper shadow / range (selling pressure) [BASE_FEATURE]",
    "w_volshock_ema": "Weekly volume shock indicator [BASE_FEATURE]",

    # === MACRO / INTERMARKET (6 BASE_FEATURES) ===
    "copper_gold_zscore": "Copper/Gold z-score (growth indicator) [BASE_FEATURE]",
    "gold_spy_ratio_zscore": "Gold/SPY z-score (risk-off indicator) [BASE_FEATURE]",
    "w_equity_bond_corr_60d": "Weekly equity-bond correlation [BASE_FEATURE]",
    "w_fred_bamlh0a0hym2_z60": "Weekly HY spread z-score [BASE_FEATURE]",
    "fred_dgs2_chg20d": "20-day change in 2Y Treasury rate [BASE_FEATURE]",
    "fred_ccsa_z52w": "Continued claims z-score (labor market) [BASE_FEATURE]",

    # ==========================================================================
    # EXPANSION_CANDIDATES - Features for forward selection (~200 total)
    # ==========================================================================

    # === MOMENTUM (EXPANSION) ===
    "rsi_21": "21-day RSI",
    "macd_hist_deriv_ema3": "3-day EMA of MACD histogram derivative",
    "w_rsi_14": "Weekly 14-day RSI",
    "w_rsi_21": "Weekly 21-day RSI",
    "w_macd_hist_deriv_ema3": "Weekly MACD histogram derivative",
    "trend_persist_ema": "EMA-smoothed consecutive up/down days",

    # === TREND SHAPE (EXPANSION) ===
    "trend_score_granular": "Multi-level trend strength (-3 to +3)",
    "w_trend_score_sign": "Weekly trend score sign",
    "w_trend_score_granular": "Weekly granular trend score",
    "w_trend_persist_ema": "Weekly trend persistence",
    "quiet_trend": "Low volatility trend indicator",
    "trend_alignment": "Multi-timeframe trend alignment",

    # === TREND SLOPES (EXPANSION) ===
    "pct_slope_ma_10": "10-day MA slope as % of price",
    "pct_slope_ma_30": "30-day MA slope as % of price",
    "pct_slope_ma_50": "50-day MA slope as % of price",
    "pct_slope_ma_150": "150-day MA slope as % of price",
    "pct_slope_ma_200": "200-day MA slope as % of price",
    "rv60_slope_norm": "60-day realized vol slope (normalized)",
    "w_pct_slope_ma_20": "Weekly 20-day MA slope",
    "w_pct_slope_ma_100": "Weekly 100-day MA slope",
    "w_rv60_slope_norm": "Weekly 60-day vol slope",
    "w_trend_score_slope": "Weekly trend score slope",

    # === DISTANCE TO MA (EXPANSION) ===
    "pct_dist_ma_100": "% distance from 100-day MA",
    "pct_dist_ma_100_z": "Z-score of 100-day MA distance",
    "pct_dist_ma_200": "% distance from 200-day MA",
    "pct_dist_ma_200_z": "Z-score of 200-day MA distance",
    "min_pct_dist_ma": "Distance to nearest MA (support/resistance)",
    "relative_dist_20_50": "Relative position between 20/50 MAs",
    "w_pct_dist_ma_20": "Weekly % distance from 20-day MA",
    "w_pct_dist_ma_20_z": "Weekly z-scored distance from 20d MA",
    "w_pct_dist_ma_50_z": "Weekly z-scored distance from 50d MA",
    "w_pct_dist_ma_100_z": "Weekly z-scored distance from 100d MA",
    "w_min_pct_dist_ma": "Weekly distance to nearest MA",
    "w_relative_dist_20_50_z": "Weekly relative position z-score",

    # === RANGE/BREAKOUT (EXPANSION) ===
    "pos_in_5d_range": "Position in 5-day high-low range (0-1)",
    "pos_in_10d_range": "Position in 10-day high-low range (0-1)",
    "breakout_up_5d": "Binary: broke above 5-day high",
    "breakout_up_10d": "Binary: broke above 10-day high",
    "breakout_up_20d": "Binary: broke above 20-day high",
    "breakout_dn_20d": "Binary: broke below 20-day low",
    "range_expansion_20d": "20-day range expansion ratio",
    "range_z_20d": "20-day range z-score",
    "w_pos_in_5d_range": "Weekly position in 5d range",
    "w_pos_in_10d_range": "Weekly position in 10d range",
    "w_pos_in_20d_range": "Weekly position in 20d range",
    "w_breakout_up_20d": "Weekly breakout up 20d",
    "w_breakout_dn_20d": "Weekly breakout down 20d",
    "w_range_expansion_20d": "Weekly range expansion",
    "w_range_z_20d": "Weekly range z-score",
    "gap_atr_ratio": "Gap / ATR ratio",

    # === VOLATILITY (EXPANSION) ===
    "vol_regime": "Volatility regime (0-1, higher = more volatile)",
    "rv_ratio_10_60": "10d/60d realized vol ratio",
    "rv_ratio_20_100": "20d/100d realized vol ratio",
    "vol_z_20": "20-day volatility z-score",
    "vol_z_60": "60-day volatility z-score",
    "rvol_20": "Relative volume vs 20-day average",
    "vol_regime_cs_median": "Cross-sectional median vol regime",
    "vol_regime_rel": "Relative vol regime vs median",
    "w_rv_z_60": "Weekly 60d vol z-score",
    "w_vol_z_60": "Weekly 60d volatility z-score",
    "w_rvol_20": "Weekly relative volume",
    "w_vol_regime": "Weekly volatility regime",
    "w_vol_regime_ema10": "Weekly smoothed vol regime",
    "w_vol_regime_rel": "Weekly relative vol regime",

    # === VIX / IMPLIED VOL (EXPANSION) ===
    "vix_percentile_252d": "VIX percentile vs 252-day history",
    "vix_ma20_ratio": "VIX / 20-day MA ratio",
    "vix_vxn_spread": "VIX-VXN spread (equity vs tech vol)",
    "vix_change_5d": "5-day VIX change",
    "vix_change_20d": "20-day VIX change",
    "vix_regime": "VIX regime (0=low, 1=elevated, 2=high)",
    "w_vix_percentile_52w": "Weekly VIX percentile (52-week)",
    "w_vix_zscore_12w": "Weekly VIX z-score (12-week)",
    "w_vix_regime": "Weekly VIX regime",
    "w_vix_ma4_ratio": "Weekly VIX vs 4-week MA ratio",
    "w_vix_change_4w": "Weekly 4-week VIX change",
    "w_vxn_percentile_252d": "Weekly VXN percentile",

    # === ALPHA MOMENTUM (EXPANSION) ===
    "alpha_mom_spy_60_ema10": "60-day alpha momentum vs SPY",
    "alpha_mom_spy_120_ema10": "120-day alpha momentum vs SPY",
    "alpha_mom_qqq_20_ema10": "20-day alpha momentum vs QQQ",
    "alpha_mom_qqq_60_ema10": "60-day alpha momentum vs QQQ",
    "alpha_mom_sector_60_ema10": "60-day alpha momentum vs sector",
    "alpha_mom_combo_20_ema10": "20-day combo alpha momentum",
    "alpha_mom_combo_60_ema10": "60-day combo alpha momentum",
    "beta_spy": "Rolling beta vs SPY",
    "beta_qqq": "Rolling beta vs QQQ",
    "beta_sector": "Rolling beta vs sector ETF",
    "w_alpha_mom_spy_60_ema10": "Weekly 60d alpha vs SPY",
    "w_alpha_mom_qqq_60_ema10": "Weekly 60d alpha vs QQQ",
    "w_alpha_mom_sector_60_ema10": "Weekly 60d alpha vs sector",
    "w_alpha_mom_combo_60_ema10": "Weekly 60d combo alpha",
    "w_beta_spy": "Weekly beta vs SPY",
    "w_beta_qqq": "Weekly beta vs QQQ",

    # === FACTOR BETAS (EXPANSION) ===
    "beta_market": "Factor regression: market beta",
    "beta_bestmatch": "Factor regression: best-match ETF beta",
    "beta_breadth": "Factor regression: breadth beta",
    "residual_cumret": "Cumulative residual return",
    "residual_vol": "Residual volatility",
    "w_beta_market": "Weekly market beta",
    "w_beta_bestmatch": "Weekly best-match beta",
    "w_beta_breadth": "Weekly breadth beta",
    "w_residual_cumret": "Weekly residual cumret",
    "w_residual_vol": "Weekly residual vol",

    # === FACTOR SPREADS (EXPANSION) ===
    "qqq_spy_cumret_20": "QQQ-SPY 20d cumulative return spread",
    "qqq_spy_cumret_60": "QQQ-SPY 60d cumulative return spread",
    "qqq_spy_zscore_60": "QQQ-SPY spread z-score",
    "qqq_spy_slope_20": "QQQ-SPY spread slope",
    "rsp_spy_cumret_20": "RSP-SPY 20d cumulative return spread",
    "rsp_spy_cumret_60": "RSP-SPY 60d cumulative return spread",
    "rsp_spy_zscore_60": "RSP-SPY spread z-score",
    "rsp_spy_slope_20": "RSP-SPY spread slope",
    "bestmatch_spy_cumret_60": "Bestmatch-SPY 60d cumret",
    "bestmatch_spy_zscore_60": "Bestmatch-SPY z-score",
    "w_qqq_spy_cumret_12": "Weekly QQQ-SPY 12w cumret",
    "w_qqq_spy_zscore_12": "Weekly QQQ-SPY z-score",
    "w_qqq_spy_slope_4": "Weekly QQQ-SPY slope",
    "w_rsp_spy_cumret_12": "Weekly RSP-SPY 12w cumret",
    "w_rsp_spy_zscore_12": "Weekly RSP-SPY z-score",
    "w_rsp_spy_slope_4": "Weekly RSP-SPY slope",
    "w_bestmatch_spy_cumret_12": "Weekly bestmatch-SPY cumret",
    "w_bestmatch_spy_zscore_12": "Weekly bestmatch-SPY z-score",

    # === RELATIVE STRENGTH (EXPANSION) ===
    "rel_strength_spy": "Relative strength vs SPY",
    "rel_strength_spy_zscore": "Z-score of RS vs SPY",
    "rel_strength_spy_rsi": "RSI of relative strength vs SPY",
    "rel_strength_qqq": "Relative strength vs QQQ",
    "rel_strength_qqq_zscore": "Z-score of RS vs QQQ",
    "rel_strength_sector_zscore": "Z-score of RS vs sector",
    "rel_strength_sector_rsi": "RSI of RS vs sector",
    "rel_strength_sector_vs_market": "Sector RS vs market",
    "w_rel_strength_spy": "Weekly RS vs SPY",
    "w_rel_strength_spy_zscore": "Weekly RS vs SPY z-score",
    "w_rel_strength_qqq": "Weekly RS vs QQQ",
    "w_rel_strength_sector": "Weekly RS vs sector",
    "w_rel_strength_sector_zscore": "Weekly RS vs sector z-score",
    "rel_strength_qqq_spy_spread": "QQQ-SPY relative strength spread",

    # === CROSS-SECTIONAL MOMENTUM (EXPANSION) ===
    "xsec_mom_5d_z": "5-day momentum cross-sectional z-score",
    "xsec_mom_60d_z": "60-day momentum cross-sectional z-score",
    "xsec_mom_5d_sect_neutral_z": "5d sector-neutral momentum z-score",
    "xsec_mom_20d_sect_neutral_z": "20d sector-neutral momentum z-score",
    "xsec_pct_20d": "20-day return percentile (0-100)",
    "xsec_pct_60d": "60-day return percentile",
    "w_xsec_mom_1w_z": "Weekly 1w momentum z-score",
    "w_xsec_mom_13w_z": "Weekly 13w momentum z-score",
    "w_xsec_mom_4w_sect_neutral_z": "Weekly 4w sector-neutral z-score",
    "w_xsec_pct_4w": "Weekly 4w percentile",
    "w_xsec_pct_13w": "Weekly 13w percentile",
    "w_xsec_pct_4w_sect": "Weekly 4w sector percentile",

    # === LIQUIDITY (EXPANSION) ===
    "vwap_dist_5d_zscore": "Z-scored distance from 5d VWAP",
    "vwap_dist_10d_zscore": "Z-scored distance from 10d VWAP",
    "lower_shadow_ratio": "Lower shadow / range (buying pressure)",
    "overnight_ratio": "Overnight vs intraday move ratio",
    "range_efficiency": "Close move / HL range (trend quality)",
    "rel_volume_20d": "Relative volume vs 20d average",
    "volume_direction": "Volume-weighted price direction",
    "pv_divergence_5d": "5-day price-volume divergence",
    "amihud_illiq_ratio": "Amihud illiquidity ratio",
    "illiquidity_score": "Composite illiquidity score",
    "w_vwap_dist_20d_zscore": "Weekly VWAP distance z-score",
    "w_range_efficiency": "Weekly range efficiency",
    "w_rel_volume_20d": "Weekly relative volume",
    "w_illiquidity_score": "Weekly illiquidity score",

    # === MARKET BREADTH (EXPANSION) ===
    "ad_ratio_ema10": "Advance-decline ratio (10-day EMA)",
    "ad_ratio_universe": "Universe-wide A/D ratio",
    "mcclellan_oscillator": "McClellan oscillator (breadth momentum)",
    "w_ad_ratio_ema10": "Weekly A/D ratio EMA",
    "w_mcclellan_oscillator": "Weekly McClellan oscillator",
    "w_ad_thrust_4w": "Weekly 4w A/D thrust",

    # === INTERMARKET RATIOS (EXPANSION) ===
    "copper_gold_ratio": "Copper/Gold ratio - growth indicator",
    "gold_spy_ratio": "Gold/SPY ratio - risk-off indicator",
    "cyclical_defensive_ratio": "Cyclicals vs defensives ratio",
    "tech_spy_ratio": "Tech/SPY ratio (growth preference)",
    "financials_utilities_ratio": "XLF/XLU ratio (rate expectations)",
    "w_copper_gold_ratio": "Weekly copper/gold ratio",
    "w_gold_spy_ratio": "Weekly gold/SPY ratio",
    "w_cyclical_defensive_ratio": "Weekly cyclical/defensive",
    "w_tech_spy_ratio": "Weekly tech/SPY ratio",
    "w_financials_utilities_ratio": "Weekly financials/utilities",

    # === MACRO (FRED) (EXPANSION) ===
    "fred_bamlh0a0hym2_z60": "HY spread 60d z-score",
    "fred_bamlh0a0hym2_chg20d": "HY spread 20d change",
    "fred_dgs10_chg20d": "10Y Treasury 20d change",
    "fred_dgs10_z60": "10Y Treasury 60d z-score",
    "fred_dgs2_chg5d": "2Y Treasury 5d change",
    "fred_t10y2y_z60": "Yield curve (10Y-2Y) z-score",
    "fred_t10y3m_z60": "Yield curve (10Y-3M) z-score",
    "fred_nfci_chg4w": "Financial conditions 4w change",
    "fred_nfci_z52w": "Financial conditions 52w z-score",
    "fred_icsa_chg4w": "Initial claims 4w change",
    "fred_icsa_z52w": "Initial claims 52w z-score",
    "fred_ccsa_chg4w": "Continued claims 4w change",
    "w_fred_bamlh0a0hym2_chg20d": "Weekly HY spread change",
    "w_fred_dgs10_z60": "Weekly 10Y z-score",
    "w_fred_dgs2_chg20d": "Weekly 2Y change",
    "w_fred_t10y2y_z60": "Weekly yield curve z-score",
    "w_fred_nfci_chg4w": "Weekly NFCI change",
    "w_fred_icsa_chg4w": "Weekly claims change",
    "w_fred_icsa_z52w": "Weekly claims z-score",
    "w_fred_ccsa_z52w": "Weekly continued claims z-score",

    # === VOLUME ANALYSIS (EXPANSION) ===
    "obv_z_60": "60-day OBV z-score",
    "rdollar_vol_20": "Relative dollar volume vs 20d avg",
    "volshock_z": "Volume shock z-score",
    "volshock_dir": "Volume shock direction",
    "w_obv_z_60": "Weekly OBV z-score",
    "w_rdollar_vol_20": "Weekly relative dollar volume",
    "w_volshock_z": "Weekly volume shock z-score",
    "w_volshock_dir": "Weekly volume shock direction",

    # === REGIME & CORRELATION (EXPANSION) ===
    "credit_spread_zscore": "Credit spread z-score",
    "yield_curve_zscore": "Yield curve z-score",
    "equity_bond_corr_60d": "60-day equity-bond correlation",
    "w_credit_spread_zscore": "Weekly credit spread z-score",
    "w_yield_curve_zscore": "Weekly yield curve z-score",
    "w_quiet_trend": "Weekly quiet trend",
    "w_trend_alignment": "Weekly trend alignment",
    "dollar_momentum_20d": "Dollar 20-day momentum",
}

# Feature categories with expected NaN ranges
FEATURE_CATEGORIES = {
    "trend": {
        "patterns": ["trend_score", "pct_slope_ma"],
        "expected_nan": (5, 20),  # (min%, max%)
        "description": "Trend direction and strength indicators",
    },
    "momentum": {
        "patterns": ["rsi_", "macd_"],
        "expected_nan": (5, 15),
        "description": "Momentum oscillators (RSI, MACD)",
    },
    "volatility": {
        "patterns": ["vol_regime", "atr_", "rv_z", "rvol"],
        "expected_nan": (5, 15),
        "description": "Volatility regime and realized vol features",
    },
    "price_position": {
        "patterns": ["pct_dist_ma", "min_pct_dist"],
        "expected_nan": (5, 20),
        "description": "Distance to moving averages",
    },
    "range_breakout": {
        "patterns": ["pos_in_", "breakout_", "range_"],
        "expected_nan": (2, 10),
        "description": "Range position and breakout signals",
    },
    "volume": {
        "patterns": ["obv_", "rdollar_vol", "volshock"],
        "expected_nan": (5, 15),
        "description": "Volume-based indicators",
    },
    "liquidity": {
        "patterns": ["spread", "amihud", "vwap_dist", "illiq", "shadow", "overnight", "range_eff"],
        "expected_nan": (5, 15),
        "description": "Liquidity and bid-ask proxies",
    },
    "alpha_beta": {
        "patterns": ["alpha_", "beta_", "residual_"],
        "expected_nan": (5, 30),
        "description": "Alpha momentum and factor betas",
    },
    "relative_strength": {
        "patterns": ["rel_strength"],
        "expected_nan": (5, 20),
        "description": "Relative strength vs benchmarks",
    },
    "breadth": {
        "patterns": ["ad_ratio", "mcclellan", "pct_universe"],
        "expected_nan": (0, 5),
        "description": "Market breadth indicators",
    },
    "cross_sectional": {
        "patterns": ["xsec_", "cs_"],
        "expected_nan": (5, 15),
        "description": "Cross-sectional momentum rankings",
    },
    "vix_macro": {
        "patterns": ["vix_"],
        "expected_nan": (0, 5),
        "description": "VIX regime and term structure",
    },
    "fred_macro": {
        "patterns": ["fred_"],
        "expected_nan": (5, 20),
        "description": "FRED economic data (yields, credit, labor)",
    },
    "intermarket": {
        "patterns": ["copper_", "gold_", "dollar_", "oil_", "cyclical", "financials", "tech_spy", "equity_bond", "credit_spread", "yield_curve"],
        "expected_nan": (5, 20),
        "description": "Cross-asset relationships",
    },
    "weekly": {
        "patterns": ["w_"],
        "expected_nan": (10, 30),
        "description": "Weekly timeframe features (w_ prefix)",
    },
}


def categorize_feature(col: str) -> Tuple[str, str]:
    """Return (category, description) for a feature column."""
    col_lower = col.lower()

    for cat_name, cat_info in FEATURE_CATEGORIES.items():
        for pattern in cat_info["patterns"]:
            if pattern in col_lower:
                desc = FEATURE_DESCRIPTIONS.get(col, f"Part of {cat_name} feature set")
                return cat_name, desc

    return "other", FEATURE_DESCRIPTIONS.get(col, "Unknown feature")


def analyze_features(df: pd.DataFrame, verbose: bool = False) -> Dict:
    """Analyze feature quality and return summary dict."""

    # Identify feature columns (numeric, not metadata)
    meta_cols = {'symbol', 'date', 'index'}
    feature_cols = [c for c in df.columns
                    if c not in meta_cols and df[c].dtype in [np.float32, np.float64]]

    # Compute NaN percentages
    nan_pcts = (df[feature_cols].isna().sum() / len(df) * 100)

    # Organize by category
    category_stats = {}
    for cat_name, cat_info in FEATURE_CATEGORIES.items():
        patterns = cat_info["patterns"]
        matching = [c for c in feature_cols if any(p in c.lower() for p in patterns)]

        if matching:
            cat_nan = nan_pcts[matching]
            category_stats[cat_name] = {
                "features": matching,
                "count": len(matching),
                "nan_mean": cat_nan.mean(),
                "nan_max": cat_nan.max(),
                "nan_min": cat_nan.min(),
                "expected_nan": cat_info["expected_nan"],
                "description": cat_info["description"],
                "healthy": sum(1 for c in matching if nan_pcts[c] < 50),
                "broken": [c for c in matching if nan_pcts[c] >= 90],
            }

    # Find infinite values
    inf_cols = []
    for col in feature_cols:
        n = np.isinf(df[col]).sum()
        if n > 0:
            inf_cols.append((col, n))

    # High NaN features
    high_nan = nan_pcts[nan_pcts > 50].sort_values(ascending=False)

    return {
        "total_features": len(feature_cols),
        "category_stats": category_stats,
        "nan_pcts": nan_pcts,
        "inf_cols": inf_cols,
        "high_nan": high_nan,
    }


def validate_expansion_candidates(df: pd.DataFrame) -> Dict:
    """Validate EXPANSION_CANDIDATES presence and quality.

    Returns dict with:
        - present: list of features present in df
        - missing: list of features not in df
        - coverage: percentage of features present
        - by_category: dict of category -> {present, missing, coverage, missing_list, missing_daily, missing_weekly}
        - high_nan: list of (feature, nan_rate) for features with >80% NaN
    """
    expansion_flat = get_expansion_candidates(flat=True)
    if not expansion_flat:
        return {"error": "EXPANSION_CANDIDATES not available"}

    columns = set(df.columns)
    present = [f for f in expansion_flat if f in columns]
    missing = [f for f in expansion_flat if f not in columns]
    coverage = len(present) / len(expansion_flat) * 100 if expansion_flat else 100

    # By category breakdown with daily/weekly split
    by_category = {}
    for cat_name, cat_features in EXPANSION_CANDIDATES.items():
        cat_present = [f for f in cat_features if f in columns]
        cat_missing = [f for f in cat_features if f not in columns]
        cat_coverage = len(cat_present) / len(cat_features) * 100 if cat_features else 100

        # Split missing into daily vs weekly
        missing_daily = [f for f in cat_missing if not f.startswith('w_')]
        missing_weekly = [f for f in cat_missing if f.startswith('w_')]

        by_category[cat_name] = {
            "present": len(cat_present),
            "missing": len(cat_missing),
            "missing_list": cat_missing,  # Full list for detailed report
            "missing_daily": missing_daily,
            "missing_weekly": missing_weekly,
            "total": len(cat_features),
            "coverage": cat_coverage,
        }

    # Check NaN rates for present features
    high_nan = []
    for feat in present:
        nan_rate = df[feat].isna().mean() * 100
        if nan_rate > 80:
            high_nan.append((feat, nan_rate))
    high_nan.sort(key=lambda x: -x[1])  # Sort by NaN rate descending

    return {
        "present": present,
        "missing": missing,
        "coverage": coverage,
        "by_category": by_category,
        "high_nan": high_nan,
        "total": len(expansion_flat),
    }


def print_expansion_detail_report(expansion_analysis: Dict):
    """Print detailed report of missing EXPANSION_CANDIDATES grouped by category."""
    if not expansion_analysis or "error" in expansion_analysis:
        return

    print("\n" + "=" * 80)
    print("DETAILED EXPANSION_CANDIDATES MISSING REPORT")
    print("=" * 80)

    by_cat = expansion_analysis["by_category"]

    # Sort categories by coverage (lowest first)
    sorted_cats = sorted(by_cat.items(), key=lambda x: x[1]["coverage"])

    for cat_name, info in sorted_cats:
        if info["missing"] == 0:
            continue  # Skip categories with 100% coverage

        coverage = info["coverage"]
        status = "PASS" if coverage >= 90 else ("WARN" if coverage >= 70 else "LOW")

        print(f"\n{'-'*60}")
        print(f"{cat_name.upper()}: {info['present']}/{info['total']} ({coverage:.0f}%) [{status}]")
        print(f"{'-'*60}")

        missing_daily = info.get("missing_daily", [])
        missing_weekly = info.get("missing_weekly", [])

        if missing_daily:
            print(f"  Daily features missing ({len(missing_daily)}):")
            for feat in missing_daily:
                print(f"    - {feat}")

        if missing_weekly:
            print(f"  Weekly features missing ({len(missing_weekly)}):")
            for feat in missing_weekly:
                print(f"    - {feat}")

    # Summary of what to implement
    print("\n" + "=" * 80)
    print("IMPLEMENTATION PRIORITY SUMMARY")
    print("=" * 80)

    # Categorize missing features by type
    missing_by_type = {
        "daily_single_stock": [],
        "daily_cross_sectional": [],
        "weekly_single_stock": [],
        "weekly_cross_sectional": [],
    }

    # Cross-sectional categories
    cs_categories = {"alpha_momentum", "relative_strength", "cross_sectional_momentum",
                     "market_breadth", "joint_factor"}

    for cat_name, info in by_cat.items():
        is_cs = cat_name in cs_categories
        for feat in info.get("missing_daily", []):
            key = "daily_cross_sectional" if is_cs else "daily_single_stock"
            missing_by_type[key].append((cat_name, feat))
        for feat in info.get("missing_weekly", []):
            key = "weekly_cross_sectional" if is_cs else "weekly_single_stock"
            missing_by_type[key].append((cat_name, feat))

    for feat_type, features in missing_by_type.items():
        if not features:
            continue
        print(f"\n{feat_type.upper().replace('_', ' ')} ({len(features)} features):")
        # Group by category
        by_cat_grp = {}
        for cat, feat in features:
            by_cat_grp.setdefault(cat, []).append(feat)
        for cat, feats in sorted(by_cat_grp.items()):
            print(f"  {cat}: {', '.join(feats[:5])}")
            if len(feats) > 5:
                print(f"    ... and {len(feats) - 5} more")


def analyze_targets(targets_path: Path) -> Dict:
    """Analyze targets file quality."""
    if not targets_path.exists():
        return {"error": f"Targets file not found: {targets_path}"}

    df = pd.read_parquet(targets_path)

    analysis = {
        "rows": len(df),
        "columns": list(df.columns),
        "column_stats": {},
    }

    for col in df.columns:
        if df[col].dtype in [np.float32, np.float64, np.int64, np.int32]:
            nan_pct = df[col].isna().sum() / len(df) * 100
            vmin, vmax = df[col].min(), df[col].max()
            analysis["column_stats"][col] = {
                "nan_pct": nan_pct,
                "min": vmin,
                "max": vmax,
            }

    # Check for anomalies
    anomalies = []
    for col, stats in analysis["column_stats"].items():
        if col in ['entry_px', 'top', 'bot', 'price_hit']:
            if stats["max"] > 1e6:
                anomalies.append(f"{col}: max value {stats['max']:.0f} (suspiciously large)")
        if col == 'ret_from_entry':
            if abs(stats["min"]) > 10 or abs(stats["max"]) > 10:
                anomalies.append(f"{col}: range [{stats['min']:.2f}, {stats['max']:.2f}] (>1000% return?)")

    analysis["anomalies"] = anomalies
    return analysis


def print_summary(features_analysis: Dict, targets_analysis: Dict, base_features_analysis: Dict,
                  expansion_analysis: Dict = None, verbose: bool = False):
    """Print actionable summary."""

    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    # === BASE_FEATURES V2 VALIDATION (Golden Reference) ===
    if base_features_analysis:
        print(f"\n{'='*40}")
        print("BASE_FEATURES V2 VALIDATION (~38 curated core features)")
        print(f"{'='*40}")
        valid = base_features_analysis.get("valid", [])
        missing = base_features_analysis.get("missing", [])
        nan_rates = base_features_analysis.get("nan_rates", {})
        total = len(valid) + len(missing)
        coverage = len(valid) / total * 100 if total > 0 else 0

        if coverage >= 95:
            status = "PASS"
        elif coverage >= 80:
            status = "WARN"
        else:
            status = "FAIL"

        print(f"   Coverage: {len(valid)}/{total} ({coverage:.1f}%) [{status}]")

        if missing:
            print(f"\n   Missing BASE_FEATURES ({len(missing)}):")
            for feat in missing[:10]:
                desc = FEATURE_DESCRIPTIONS.get(feat, "No description")
                print(f"   - {feat}")
                if verbose:
                    print(f"     {desc[:70]}")
            if len(missing) > 10:
                print(f"   ... and {len(missing) - 10} more")

        # Show high NaN BASE_FEATURES
        high_nan_base = [(f, r) for f, r in nan_rates.items() if r > 30]
        if high_nan_base:
            print(f"\n   BASE_FEATURES with high NaN (>30%):")
            for feat, rate in sorted(high_nan_base, key=lambda x: -x[1])[:5]:
                print(f"   - {feat}: {rate:.1f}%")

    # === EXPANSION_CANDIDATES V2 VALIDATION ===
    if expansion_analysis and "error" not in expansion_analysis:
        print(f"\n{'='*40}")
        print("EXPANSION_CANDIDATES V2 (~200 features for forward selection)")
        print(f"{'='*40}")

        coverage = expansion_analysis["coverage"]
        present = len(expansion_analysis["present"])
        total = expansion_analysis["total"]

        if coverage >= 90:
            status = "PASS"
        elif coverage >= 70:
            status = "WARN"
        else:
            status = "FAIL"

        print(f"   Coverage: {present}/{total} ({coverage:.1f}%) [{status}]")

        # Show by-category breakdown
        by_cat = expansion_analysis["by_category"]
        low_coverage_cats = [(cat, info) for cat, info in by_cat.items()
                            if info["coverage"] < 70]
        if low_coverage_cats:
            print(f"\n   Categories with low coverage (<70%):")
            for cat, info in sorted(low_coverage_cats, key=lambda x: x[1]["coverage"]):
                print(f"   - {cat}: {info['present']}/{info['total']} ({info['coverage']:.0f}%)")
                if verbose and info["missing_list"]:
                    print(f"     Missing: {', '.join(info['missing_list'][:3])}...")

        # Show high NaN expansion features
        high_nan_exp = expansion_analysis.get("high_nan", [])
        if high_nan_exp:
            print(f"\n   EXPANSION features with >80% NaN ({len(high_nan_exp)} features):")
            for feat, rate in high_nan_exp[:5]:
                print(f"   - {feat}: {rate:.0f}%")
            if len(high_nan_exp) > 5:
                print(f"   ... and {len(high_nan_exp) - 5} more")

    # === FEATURES SUMMARY ===
    print(f"\n{'='*40}")
    print("FEATURES SUMMARY")
    print(f"{'='*40}")
    print(f"Total features: {features_analysis['total_features']}")

    # Category breakdown
    print(f"\n{'Category':<20} {'Count':>6} {'Avg NaN':>8} {'Max NaN':>8} {'Status':<15}")
    print("-" * 60)

    issues = []
    for cat_name, stats in sorted(features_analysis['category_stats'].items()):
        exp_min, exp_max = stats["expected_nan"]
        avg_nan = stats["nan_mean"]

        if stats["broken"]:
            status = f"BROKEN ({len(stats['broken'])})"
            issues.append((cat_name, "broken", stats["broken"]))
        elif avg_nan > exp_max * 1.5:
            status = "HIGH NaN"
            issues.append((cat_name, "high_nan", avg_nan))
        elif stats["healthy"] < stats["count"] * 0.5:
            status = "DEGRADED"
            issues.append((cat_name, "degraded", stats["count"] - stats["healthy"]))
        else:
            status = "OK"

        print(f"{cat_name:<20} {stats['count']:>6} {avg_nan:>7.1f}% {stats['nan_max']:>7.1f}% {status:<15}")

    # === ISSUES ===
    if issues:
        print(f"\n{'='*40}")
        print("ISSUES REQUIRING ATTENTION")
        print(f"{'='*40}")

        for cat, issue_type, detail in issues:
            if issue_type == "broken":
                print(f"\n[CRITICAL] {cat}: {len(detail)} features are 100% NaN")
                for feat in detail[:5]:
                    print(f"   - {feat}")
                if len(detail) > 5:
                    print(f"   ... and {len(detail) - 5} more")
            elif issue_type == "high_nan":
                print(f"\n[WARNING] {cat}: Average NaN rate {detail:.1f}% is above expected")
            elif issue_type == "degraded":
                print(f"\n[WARNING] {cat}: {detail} features have >50% NaN")

    # === INFINITE VALUES ===
    if features_analysis['inf_cols']:
        print(f"\n{'='*40}")
        print("INFINITE VALUES (data corruption)")
        print(f"{'='*40}")
        for col, count in features_analysis['inf_cols'][:10]:
            print(f"   {col}: {count:,} inf values")

    # === HIGH NaN FEATURES ===
    high_nan = features_analysis['high_nan']
    if len(high_nan) > 0:
        print(f"\n{'='*40}")
        print(f"HIGH NaN FEATURES (>50%): {len(high_nan)} total")
        print(f"{'='*40}")
        for col, pct in high_nan.head(15).items():
            cat, desc = categorize_feature(col)
            print(f"   {col}: {pct:.1f}% NaN")
            if verbose:
                print(f"      Category: {cat}, Desc: {desc[:60]}")

    # === TARGETS SUMMARY ===
    print(f"\n{'='*40}")
    print("TARGETS FILE")
    print(f"{'='*40}")

    if "error" in targets_analysis:
        print(f"   ERROR: {targets_analysis['error']}")
    else:
        print(f"   Rows: {targets_analysis['rows']:,}")
        print(f"   Columns: {', '.join(targets_analysis['columns'])}")

        if targets_analysis.get("anomalies"):
            print(f"\n   ANOMALIES:")
            for anomaly in targets_analysis["anomalies"]:
                print(f"   - {anomaly}")

    # === ACTIONABLE RECOMMENDATIONS ===
    print(f"\n{'='*40}")
    print("RECOMMENDATIONS")
    print(f"{'='*40}")

    recommendations = []

    # Check for broken weekly factor features
    if "alpha_beta" in features_analysis['category_stats']:
        broken = features_analysis['category_stats']['alpha_beta'].get('broken', [])
        weekly_broken = [f for f in broken if f.startswith('w_')]
        if weekly_broken:
            recommendations.append(
                "Weekly factor regression features are 100% NaN.\n"
                "   Check: src/features/factor_regression.py weekly computation\n"
                "   Look for: add_joint_factor_features(..., frequency='weekly')"
            )

    # Check for missing FRED features
    if "fred_macro" in features_analysis['category_stats']:
        fred_stats = features_analysis['category_stats']['fred_macro']
        if fred_stats['nan_mean'] > 50:
            recommendations.append(
                "FRED macro features have high NaN rate.\n"
                "   Check: cache/fred_data.parquet exists\n"
                "   If missing: export FRED_API_KEY and re-run pipeline"
            )

    # Check for high NaN in cross-sectional
    if "cross_sectional" in features_analysis['category_stats']:
        xsec_stats = features_analysis['category_stats']['cross_sectional']
        if xsec_stats['nan_mean'] > 30:
            recommendations.append(
                "Cross-sectional features have elevated NaN rate.\n"
                "   Check: Sufficient symbols being processed\n"
                "   Look for: Missing ETF data in cache/etfs/stock_data_etf.parquet"
            )

    # Target anomalies
    if targets_analysis.get("anomalies"):
        recommendations.append(
            "Targets file has suspicious values.\n"
            "   Check: Invalid symbols (non-equity) in input data\n"
            "   Look for: Symbols with extremely high prices (>$10k)"
        )

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\nNo critical issues found. Data quality is acceptable.")

    # === DETAILED EXPANSION CANDIDATES REPORT ===
    if verbose and expansion_analysis and "error" not in expansion_analysis:
        print_expansion_detail_report(expansion_analysis)

    # === VERBOSE FEATURE DETAILS ===
    if verbose:
        print(f"\n{'='*40}")
        print("DETAILED FEATURE LISTING")
        print(f"{'='*40}")

        for cat_name, stats in sorted(features_analysis['category_stats'].items()):
            print(f"\n--- {cat_name.upper()} ({stats['description']}) ---")
            for feat in sorted(stats['features']):
                nan_pct = features_analysis['nan_pcts'][feat]
                desc = FEATURE_DESCRIPTIONS.get(feat, "No description")
                status = "OK" if nan_pct < 20 else ("WARN" if nan_pct < 50 else "BAD")
                print(f"   [{status:4}] {feat}: {nan_pct:.1f}% NaN")
                print(f"         {desc[:70]}")


def main():
    parser = argparse.ArgumentParser(
        description="Data quality check for pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_data_quality.py
    python run_data_quality.py --verbose
    python run_data_quality.py --features artifacts/features_complete.parquet
        """
    )
    parser.add_argument(
        "--features",
        type=str,
        default="artifacts/features_filtered.parquet",
        help="Path to features parquet file (default: features_filtered.parquet, ML-ready curated features)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="artifacts/targets_triple_barrier.parquet",
        help="Path to targets parquet file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed feature descriptions"
    )

    args = parser.parse_args()

    features_path = Path(args.features)
    targets_path = Path(args.targets)

    # Check files exist
    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}")
        print("Run the pipeline first: conda run -n stocks_predictor python -m src.cli.compute")
        sys.exit(1)

    # Load and analyze
    print(f"Loading {features_path}...")
    df = pd.read_parquet(features_path)

    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Get basic info
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        print(f"   Symbols: {len(symbols):,}")
    if 'date' in df.columns:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nAnalyzing features...")
    features_analysis = analyze_features(df, verbose=args.verbose)

    print("Analyzing targets...")
    targets_analysis = analyze_targets(targets_path)

    # Validate BASE_FEATURES (golden reference)
    base_features_analysis = None
    if HAS_BASE_FEATURES and BASE_FEATURES:
        print("Validating BASE_FEATURES (golden reference)...")
        base_features_analysis = validate_base_features(df)

    # Validate EXPANSION_CANDIDATES (feature selection pool)
    expansion_analysis = None
    if HAS_BASE_FEATURES and EXPANSION_CANDIDATES:
        print("Validating EXPANSION_CANDIDATES (feature selection pool)...")
        expansion_analysis = validate_expansion_candidates(df)

    # Print summary
    print_summary(features_analysis, targets_analysis, base_features_analysis,
                  expansion_analysis=expansion_analysis, verbose=args.verbose)


if __name__ == "__main__":
    main()
