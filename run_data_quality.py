#!/usr/bin/env python3
"""
Data Quality Check Script

Analyzes pipeline output files and provides actionable insights:
- BASE_FEATURES validation (golden reference from base_features.py)
- EXPANSION_CANDIDATES validation (feature selection pool coverage)
- Feature coverage and NaN rates by category
- Data quality issues (infinite values, missing features)
- Targets file validation
- Recommendations for fixing issues

Usage:
    conda run -n stocks_predictor python run_data_quality.py
    conda run -n stocks_predictor python run_data_quality.py --verbose
    conda run -n stocks_predictor python run_data_quality.py --features artifacts/features_daily.parquet
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

# Complete feature descriptions including all BASE_FEATURES
FEATURE_DESCRIPTIONS = {
    # ==========================================================================
    # BASE_FEATURES - The 53 features selected by ML model (0.6932 AUC)
    # These are the GOLDEN REFERENCE features that must be present
    # ==========================================================================

    # === TREND DIRECTION (3 BASE_FEATURES) ===
    "rsi_14": "14-day RSI (0-100, >70 overbought, <30 oversold) [BASE_FEATURE]",
    "w_rsi_14": "Weekly 14-day RSI - slower, filters noise [BASE_FEATURE]",
    "w_macd_histogram": "Weekly MACD histogram - momentum trend [BASE_FEATURE]",

    # === TREND STRENGTH (8 BASE_FEATURES) ===
    "trend_score_sign": "Multi-MA alignment direction (+1/-1 per MA) [BASE_FEATURE]",
    "trend_score_slope": "Rate of change of trend score [BASE_FEATURE]",
    "trend_persist_ema": "EMA-smoothed consecutive up/down days [BASE_FEATURE]",
    "pct_slope_ma_20": "20-day MA slope as % of price (short-term trend) [BASE_FEATURE]",
    "pct_slope_ma_100": "100-day MA slope as % of price (medium-term trend) [BASE_FEATURE]",
    "w_pct_slope_ma_20": "Weekly 20-day MA slope [BASE_FEATURE]",
    "w_rv60_slope_norm": "Weekly 60-day realized vol slope (normalized) [BASE_FEATURE]",
    "w_rv100_slope_norm": "Weekly 100-day realized vol slope (normalized) [BASE_FEATURE]",

    # === PRICE POSITION (8 BASE_FEATURES) ===
    "pct_dist_ma_20": "% distance from 20-day MA (mean reversion signal) [BASE_FEATURE]",
    "pct_dist_ma_50": "% distance from 50-day MA (trend distance) [BASE_FEATURE]",
    "w_pct_dist_ma_20": "Weekly % distance from 20-day MA [BASE_FEATURE]",
    "w_pct_dist_ma_100_z": "Weekly distance to 100d MA z-score [BASE_FEATURE]",
    "pos_in_20d_range": "Position in 20-day high-low range (0-1) [BASE_FEATURE]",
    "w_pos_in_5d_range": "Weekly position in 5d range [BASE_FEATURE]",
    "vwap_dist_20d_zscore": "Z-scored distance from 20d VWAP [BASE_FEATURE]",
    "w_vwap_dist_20d_zscore": "Weekly VWAP distance (smoother) [BASE_FEATURE]",

    # === VOLATILITY REGIME (7 BASE_FEATURES) ===
    "vol_regime": "Volatility regime (0-1, higher = more volatile) [BASE_FEATURE]",
    "vol_regime_ema10": "10-day EMA smoothed volatility regime [BASE_FEATURE]",
    "atr_percent": "ATR as % of price (position sizing) [BASE_FEATURE]",
    "vix_regime": "VIX regime (0=low, 1=elevated, 2=high) [BASE_FEATURE]",
    "w_vix_ma4_ratio": "Weekly VIX vs 4-week MA ratio [BASE_FEATURE]",
    "w_vix_vxn_spread": "Weekly VIX-VXN spread (equity vs tech vol) [BASE_FEATURE]",
    "w_alpha_mom_qqq_spread_60_ema10": "Weekly QQQ-SPY alpha spread (growth vs value) [BASE_FEATURE]",

    # === RELATIVE PERFORMANCE (7 BASE_FEATURES) ===
    "alpha_mom_spy_20_ema10": "20-day alpha momentum vs SPY (EMA smoothed) [BASE_FEATURE]",
    "alpha_mom_sector_20_ema10": "20-day alpha momentum vs sector [BASE_FEATURE]",
    "beta_spy": "Rolling beta vs SPY (CAPM) [BASE_FEATURE]",
    "rel_strength_sector": "Relative strength vs sector ETF [BASE_FEATURE]",
    "xsec_mom_20d_z": "20-day momentum cross-sectional z-score [BASE_FEATURE]",
    "w_alpha_mom_spy_20_ema10": "Weekly alpha vs SPY [BASE_FEATURE]",
    "w_beta_spy": "Weekly market beta [BASE_FEATURE]",

    # === MACRO/INTERMARKET (16 BASE_FEATURES) ===
    "copper_gold_ratio": "Copper/Gold ratio - economic growth indicator [BASE_FEATURE]",
    "copper_gold_zscore": "Copper/Gold z-score [BASE_FEATURE]",
    "gold_spy_ratio": "Gold/SPY ratio - risk-off indicator [BASE_FEATURE]",
    "fred_ccsa_z52w": "Continued claims z-score (labor market) [BASE_FEATURE]",
    "fred_dgs2_chg20d": "20-day change in 2Y Treasury rate [BASE_FEATURE]",
    "fred_icsa_chg4w": "4-week change in initial claims [BASE_FEATURE]",
    "w_copper_gold_ratio": "Weekly copper/gold ratio [BASE_FEATURE]",
    "w_gold_spy_ratio": "Weekly gold/SPY ratio [BASE_FEATURE]",
    "w_gold_spy_ratio_zscore": "Weekly gold/SPY z-score [BASE_FEATURE]",
    "w_dollar_momentum_20d": "Weekly dollar momentum [BASE_FEATURE]",
    "w_equity_bond_corr_60d": "Weekly stock-bond correlation [BASE_FEATURE]",
    "w_financials_utilities_ratio": "Weekly risk-on/off sector ratio [BASE_FEATURE]",
    "w_fred_bamlh0a0hym2_pct252": "Weekly HY spread percentile [BASE_FEATURE]",
    "w_fred_bamlh0a0hym2_z60": "Weekly HY spread z-score [BASE_FEATURE]",
    "w_fred_icsa_chg4w": "Weekly initial claims change [BASE_FEATURE]",
    "w_fred_nfci_chg4w": "Weekly financial conditions change [BASE_FEATURE]",

    # === MARKET BREADTH (1 BASE_FEATURE) ===
    "w_ad_ratio_universe": "Weekly advance-decline ratio [BASE_FEATURE]",

    # === VOLUME/LIQUIDITY (2 BASE_FEATURES) ===
    "upper_shadow_ratio": "Upper shadow / range (selling pressure) [BASE_FEATURE]",
    "w_volshock_ema": "Weekly volume shock indicator [BASE_FEATURE]",

    # === BREAKOUT (1 BASE_FEATURE) ===
    "breakout_up_20d": "Binary: broke above 20-day high [BASE_FEATURE]",

    # ==========================================================================
    # ADDITIONAL FEATURES - Not in BASE_FEATURES but useful for expansion
    # ==========================================================================

    # === TREND FEATURES (EXPANSION) ===
    "trend_score_granular": "Multi-level trend strength (-3 to +3)",
    "pct_slope_ma_10": "10-day MA slope as % of price",
    "pct_slope_ma_30": "30-day MA slope as % of price",
    "pct_slope_ma_50": "50-day MA slope as % of price",
    "pct_slope_ma_200": "200-day MA slope as % of price",

    # === MOMENTUM FEATURES (EXPANSION) ===
    "rsi_21": "21-day RSI",
    "rsi_30": "30-day RSI",
    "macd_histogram": "MACD histogram (momentum strength)",
    "macd_hist_deriv_ema3": "3-day EMA of MACD histogram derivative",

    # === VOLATILITY FEATURES (EXPANSION) ===
    "rv_z_20": "20-day realized vol z-score vs 60-day lookback",
    "rv_z_60": "60-day realized vol z-score vs 252-day lookback",
    "rvol_20": "Relative volume vs 20-day average",

    # === PRICE POSITION FEATURES (EXPANSION) ===
    "pct_dist_ma_100": "% distance from 100-day MA",
    "pct_dist_ma_200": "% distance from 200-day MA",
    "pct_dist_ma_100_z": "Z-score of 100-day MA distance",
    "pct_dist_ma_200_z": "Z-score of 200-day MA distance",
    "min_pct_dist_ma": "Distance to nearest MA (support/resistance)",
    "relative_dist_20_50_z": "Z-score of 20-50 MA distance ratio",

    # === RANGE/BREAKOUT FEATURES (EXPANSION) ===
    "pos_in_5d_range": "Position in 5-day high-low range (0-1)",
    "pos_in_10d_range": "Position in 10-day high-low range (0-1)",
    "breakout_up_5d": "Binary: broke above 5-day high",
    "breakout_up_10d": "Binary: broke above 10-day high",
    "breakout_dn_5d": "Binary: broke below 5-day low",
    "breakout_dn_10d": "Binary: broke below 10-day low",
    "breakout_dn_20d": "Binary: broke below 20-day low",
    "range_expansion_5d": "5-day range vs previous 5-day range",
    "range_expansion_10d": "10-day range vs previous 10-day range",
    "range_expansion_20d": "20-day range vs previous 20-day range",

    # === VOLUME FEATURES (EXPANSION) ===
    "obv_z_60": "60-day OBV z-score (accumulation/distribution)",
    "rdollar_vol_20": "Relative dollar volume vs 20-day avg",
    "volshock_z": "Volume shock z-score (unusual volume)",
    "volshock_ema": "EMA-smoothed volume shock",

    # === LIQUIDITY FEATURES (EXPANSION) ===
    "hl_spread_proxy": "High-low spread proxy (intraday volatility)",
    "cs_spread_est": "Corwin-Schultz spread estimator",
    "roll_spread_est": "Roll bid-ask spread estimator",
    "overnight_ratio": "Overnight vs intraday move ratio",
    "range_efficiency": "Close move / HL range (trend quality)",
    "lower_shadow_ratio": "Lower shadow / range (buying pressure)",
    "amihud_illiq": "Amihud illiquidity (price impact per $)",
    "illiquidity_score": "Composite illiquidity score",

    # === ALPHA/BETA FEATURES (EXPANSION) ===
    "beta_qqq": "Rolling beta vs QQQ",
    "beta_sector": "Rolling beta vs sector ETF",
    "beta_market": "Factor regression: market beta",
    "beta_bestmatch": "Factor regression: best-match ETF beta",
    "alpha_mom_spy_60_ema10": "60-day alpha momentum vs SPY (EMA smoothed)",
    "alpha_mom_qqq_20_ema10": "20-day alpha momentum vs QQQ (EMA smoothed)",
    "alpha_resid_spy": "Residual alpha after SPY regression",
    "alpha_qqq_vs_spy": "QQQ-relative alpha vs SPY-relative alpha",
    "residual_cumret": "Cumulative residual return (factor-adjusted momentum)",
    "residual_vol": "Residual volatility (idiosyncratic risk)",
    "residual_mean": "Mean residual return",
    "alpha": "Factor regression alpha (intercept)",

    # === RELATIVE STRENGTH FEATURES (EXPANSION) ===
    "rel_strength_spy": "Relative strength vs SPY (price ratio trend)",
    "rel_strength_spy_zscore": "Z-score of RS vs SPY",
    "rel_strength_qqq": "Relative strength vs QQQ",
    "rel_strength_qqq_zscore": "Z-score of RS vs QQQ",
    "rel_strength_sector_zscore": "Z-score of RS vs sector",
    "rel_strength_sector_ew_norm": "RS vs equal-weight sector (normalized)",
    "rel_strength_sector_ew_macd": "MACD of RS vs EW sector",
    "rel_strength_sector_ew_macd_hist": "MACD histogram of RS vs EW sector",

    # === BREADTH FEATURES (EXPANSION) ===
    "ad_ratio_ema10": "Advance-decline ratio (10-day EMA)",
    "ad_ratio_universe": "Universe-wide A/D ratio",
    "ad_thrust_10d": "10-day A/D thrust (breadth momentum)",
    "mcclellan_oscillator": "McClellan oscillator (breadth momentum)",
    "pct_universe_above_ma20": "% of universe above 20-day MA",
    "pct_universe_above_ma50": "% of universe above 50-day MA",

    # === CROSS-SECTIONAL FEATURES (EXPANSION) ===
    "vol_regime_cs_median": "Cross-sectional median vol regime",
    "vol_regime_rel": "Relative vol regime vs median",
    "xsec_mom_5d_z": "5-day momentum cross-sectional z-score",
    "xsec_mom_60d_z": "60-day momentum cross-sectional z-score",
    "xsec_pct_20d": "20-day return percentile (0-100)",
    "xsec_pct_20d_sect": "20-day return percentile within sector",

    # === VIX/MACRO FEATURES (EXPANSION) ===
    "vix_percentile_252d": "VIX percentile vs 252-day history",
    "vix_zscore_60d": "VIX z-score vs 60-day history",
    "vix_ma20_ratio": "VIX / 20-day MA ratio (term structure)",
    "vix_vxn_spread": "VIX-VXN spread (equity vs nasdaq vol)",

    # === FRED MACRO FEATURES (EXPANSION) ===
    "fred_dgs10": "10-year Treasury yield",
    "fred_dgs10_chg5d": "5-day change in 10Y yield",
    "fred_dgs10_chg20d": "20-day change in 10Y yield",
    "fred_dgs2": "2-year Treasury yield",
    "fred_dgs2_chg5d": "5-day change in 2Y yield",
    "fred_t10y2y": "10Y-2Y yield spread (yield curve)",
    "fred_t10y2y_z60": "Yield curve z-score (60-day)",
    "fred_bamlh0a0hym2": "High yield OAS spread",
    "fred_bamlh0a0hym2_z60": "HY spread z-score (credit stress)",
    "fred_bamlh0a0hym2_pct252": "HY spread percentile (252-day)",
    "fred_icsa": "Initial jobless claims",
    "fred_icsa_z52w": "Jobless claims z-score (52-week)",
    "fred_nfci": "Chicago Fed Financial Conditions Index",
    "fred_nfci_chg4w": "4-week change in NFCI",
    "fred_vixcls": "VIX close (from FRED)",

    # === INTERMARKET FEATURES (EXPANSION) ===
    "gold_spy_ratio_zscore": "Gold/SPY z-score",
    "dollar_momentum_20d": "Dollar index 20-day momentum",
    "dollar_percentile_252d": "Dollar percentile (252-day)",
    "oil_momentum_20d": "Crude oil 20-day momentum",
    "cyclical_defensive_ratio": "Cyclicals vs defensives ratio",
    "financials_utilities_ratio": "XLF/XLU ratio (rate expectations)",
    "tech_spy_ratio": "Tech/SPY ratio (growth preference)",
    "equity_bond_corr_60d": "60-day SPY-TLT correlation",
    "credit_spread_zscore": "Credit spread z-score",
    "yield_curve_zscore": "Yield curve z-score",
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

    # === BASE_FEATURES VALIDATION (Golden Reference) ===
    if base_features_analysis:
        print(f"\n{'='*40}")
        print("BASE_FEATURES VALIDATION (Golden Reference)")
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

    # === EXPANSION_CANDIDATES VALIDATION ===
    if expansion_analysis and "error" not in expansion_analysis:
        print(f"\n{'='*40}")
        print("EXPANSION_CANDIDATES (Feature Selection Pool)")
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
                "   Look for: Missing ETF data in stock_data_etf.parquet"
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
        default="artifacts/features_daily.parquet",
        help="Path to features parquet file"
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
