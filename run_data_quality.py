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
import textwrap
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

# Import provenance validation functions
try:
    from src.features.provenance import (
        load_provenance_metadata,
        validate_provenance as _validate_provenance_internal,
        report_provenance_summary,
        get_missing_provenance,
        FEATURE_PROVENANCE_REGISTRY,
    )
    HAS_PROVENANCE = True
except ImportError:
    HAS_PROVENANCE = False
    FEATURE_PROVENANCE_REGISTRY = {}

# =============================================================================
# DIAGNOSTIC TIER CLASSIFICATION SYSTEM
# =============================================================================
# Tier 1 (FAIL): Deterministic, hard violations that must be fixed
#   - Provenance violations with negative effective lookback
#   - Features using future prices in computation
#   - Adjusted price columns in feature frame
#
# Tier 2 (WARN): Requires review but may be acceptable
#   - Shift test with high AUC but no provenance violations
#   - Target autocorrelation above threshold
#   - High NaN rates in critical features
#
# Tier 3 (INFO): Expected behavior, informational only
#   - Shift test modest AUC with overlapping targets
#   - Target persistence due to barrier width
#   - Feature correlations within expected ranges

DIAGNOSTIC_TIERS = {
    "FAIL": {
        "severity": 1,
        "description": "Deterministic violation - must be fixed before training",
        "action": "Block model training until resolved",
    },
    "WARN": {
        "severity": 2,
        "description": "Requires review - may indicate issue or be acceptable",
        "action": "Review and document decision",
    },
    "INFO": {
        "severity": 3,
        "description": "Informational - expected behavior given target structure",
        "action": "No action required",
    },
}


def get_target_overlap_days(target_col: str) -> int:
    """Estimate target window overlap in days based on target type.

    Triple barrier targets have overlapping windows by design:
    - 'normal' targets: ~20-day horizon
    - 'parabolic' targets: ~15-day horizon

    This affects shift test interpretation since consecutive targets
    share much of the same future price information.

    Args:
        target_col: Target column name (e.g., 'hit_long_normal', 'hit_short_parabolic')

    Returns:
        Estimated overlap in days (conservative estimate)
    """
    if 'parabolic' in target_col.lower():
        return 12  # ~15-day horizon with ~80% overlap
    else:
        return 16  # ~20-day horizon with ~80% overlap


def interpret_shift_test_with_context(
    shift_result: dict,
    autocorr_result: dict = None,
    provenance_result: dict = None,
) -> dict:
    """Interpret shift test results in context of target structure.

    The shift test is a heuristic signal, not proof of leakage. Results should be
    interpreted relative to:
    - Target autocorrelation (regime persistence)
    - Overlap in label windows (consecutive targets share future price info)
    - Provenance check results (deterministic validation)

    Interpretation rules:
    1. If shifted AUC remains very high (>0.65) AND provenance checks fail
       → Strong evidence of leakage (FAIL)
    2. If shifted AUC drops materially (>0.10) AND provenance checks pass
       → Likely regime persistence, not leakage (INFO)
    3. If shifted AUC is modest (0.53-0.60) with overlapping targets
       → Expected behavior (INFO or WARN at most)

    Args:
        shift_result: Output from run_shift_test()
        autocorr_result: Output from check_target_autocorrelation()
        provenance_result: Output from validate_provenance_data()

    Returns:
        Dict with:
        - tier: 'FAIL', 'WARN', or 'INFO'
        - interpretation: Human-readable explanation
        - factors: Dict of contributing factors
        - action: Recommended action
    """
    if "error" in shift_result:
        return {
            "tier": "INFO",
            "interpretation": f"Shift test could not run: {shift_result['error']}",
            "factors": {},
            "action": "Resolve error and rerun shift test",
        }

    auc_normal = shift_result.get("auc_normal", 0.5)
    auc_shifted = shift_result.get("auc_shifted", 0.5)
    auc_drop = shift_result.get("auc_drop", 0)
    target_col = shift_result.get("target_used", "unknown")

    # Gather context
    factors = {
        "auc_normal": auc_normal,
        "auc_shifted": auc_shifted,
        "auc_drop": auc_drop,
    }

    # Check provenance results
    provenance_has_violations = False
    if provenance_result and not provenance_result.get("error"):
        critical_violations = provenance_result.get("critical_violations", [])
        provenance_has_violations = len(critical_violations) > 0
        factors["provenance_violations"] = len(critical_violations)

    # Check autocorrelation
    target_autocorr = 0.0
    target_persistence_pct = 50.0
    if autocorr_result and not autocorr_result.get("error"):
        target_autocorr = autocorr_result.get("lag1_autocorr", 0.0)
        target_persistence_pct = autocorr_result.get("pct_same_as_prev", 50.0)
        factors["target_autocorr"] = target_autocorr
        factors["target_persistence_pct"] = target_persistence_pct

    # Estimate target overlap
    target_overlap_days = get_target_overlap_days(target_col)
    factors["target_overlap_days"] = target_overlap_days

    # Apply interpretation rules
    # Rule 1: High shifted AUC with provenance violations → FAIL
    if auc_shifted > 0.65 and provenance_has_violations:
        return {
            "tier": "FAIL",
            "interpretation": (
                f"STRONG EVIDENCE OF LEAKAGE. Shifted AUC ({auc_shifted:.3f}) remains very high "
                f"AND provenance checks detected {factors['provenance_violations']} violations. "
                "This combination indicates deterministic future data usage."
            ),
            "factors": factors,
            "action": "Review and fix provenance violations before training",
        }

    # Rule 2: Large AUC drop with clean provenance → INFO (regime persistence)
    if auc_drop > 0.10 and not provenance_has_violations:
        return {
            "tier": "INFO",
            "interpretation": (
                f"Shift test shows expected behavior. AUC dropped by {auc_drop:.3f} after shifting, "
                f"and provenance checks pass. The shifted AUC ({auc_shifted:.3f}) reflects "
                f"regime persistence, not leakage. Target autocorrelation is {target_autocorr:.2f}."
            ),
            "factors": factors,
            "action": "No action required - this is expected behavior",
        }

    # Rule 3: Modest shifted AUC (0.53-0.60) with overlapping targets → INFO/WARN
    if 0.53 <= auc_shifted <= 0.60:
        # Check if this is explainable by target overlap
        if target_autocorr > 0.2 or target_persistence_pct > 60:
            return {
                "tier": "INFO",
                "interpretation": (
                    f"Shift test shows modest residual signal (AUC={auc_shifted:.3f}). "
                    f"This is expected given target autocorrelation ({target_autocorr:.2f}) "
                    f"and ~{target_overlap_days}-day target window overlap. "
                    "Features shifted by 1 day still contain information relevant to "
                    "overlapping target windows."
                ),
                "factors": factors,
                "action": "No action required - expected with overlapping targets",
            }
        else:
            return {
                "tier": "WARN",
                "interpretation": (
                    f"Shift test shows modest residual signal (AUC={auc_shifted:.3f}) "
                    f"but target autocorrelation is low ({target_autocorr:.2f}). "
                    "Review feature computation for subtle lookahead patterns."
                ),
                "factors": factors,
                "action": "Review top features for potential lookahead bias",
            }

    # Rule 4: High shifted AUC without provenance violations → WARN
    if auc_shifted > 0.60 and not provenance_has_violations:
        return {
            "tier": "WARN",
            "interpretation": (
                f"Shift test shows elevated residual signal (AUC={auc_shifted:.3f}) "
                f"but provenance checks pass. AUC drop was {auc_drop:.3f}. "
                "This may indicate: (1) strong regime persistence, (2) features with "
                "high autocorrelation, or (3) subtle lookahead patterns not caught by provenance."
            ),
            "factors": factors,
            "action": "Review top features; document if acceptable",
        }

    # Rule 5: Small or negative AUC drop → investigate
    if auc_drop < 0.02:
        tier = "FAIL" if provenance_has_violations else "WARN"
        return {
            "tier": tier,
            "interpretation": (
                f"AUC barely dropped ({auc_drop:+.3f}) after shifting features. "
                f"Normal AUC: {auc_normal:.3f}, Shifted AUC: {auc_shifted:.3f}. "
                "This is unusual - features should lose predictive power when shifted."
            ),
            "factors": factors,
            "action": "Investigate feature computation for lookahead bias",
        }

    # Default: Clean
    return {
        "tier": "INFO",
        "interpretation": (
            f"Shift test is clean. AUC dropped from {auc_normal:.3f} to {auc_shifted:.3f} "
            f"(drop: {auc_drop:.3f}). No evidence of leakage."
        ),
        "factors": factors,
        "action": "No action required",
    }


def create_combined_leakage_assessment(
    shift_result: dict,
    autocorr_result: dict,
    provenance_result: dict,
    raw_price_result: dict = None,
) -> dict:
    """Create combined leakage assessment from all diagnostic checks.

    Combines multiple leakage signals into a single assessment with:
    - Overall tier (FAIL/WARN/INFO)
    - Evidence summary
    - Recommended actions

    Args:
        shift_result: Output from run_shift_test()
        autocorr_result: Output from check_target_autocorrelation()
        provenance_result: Output from validate_provenance_data()
        raw_price_result: Output from validate_raw_price_policy()

    Returns:
        Dict with:
        - overall_tier: Highest severity tier across all checks
        - checks: Dict of check_name -> {tier, summary}
        - evidence_summary: Combined narrative
        - engineering_judgment: Final recommendation
    """
    checks = {}
    highest_severity = 3  # Start with INFO (lowest severity)

    # 1. Provenance Check (Tier 1 capable - deterministic)
    if provenance_result:
        if provenance_result.get("error"):
            checks["provenance"] = {
                "tier": "INFO",
                "summary": f"Provenance check unavailable: {provenance_result.get('error', 'unknown')}",
            }
        elif provenance_result.get("critical_violations"):
            n_violations = len(provenance_result.get("critical_violations", []))
            checks["provenance"] = {
                "tier": "FAIL",
                "summary": f"{n_violations} features have negative effective lookback (use future data)",
            }
            highest_severity = min(highest_severity, 1)
        else:
            checks["provenance"] = {
                "tier": "INFO",
                "summary": "All features have valid lookback (no future data usage)",
            }

    # 2. Raw Price Policy Check (Tier 1 capable - deterministic)
    if raw_price_result:
        if not raw_price_result.get("passed"):
            n_adjusted = len(raw_price_result.get("adjusted_columns", []))
            checks["raw_price_policy"] = {
                "tier": "FAIL",
                "summary": f"{n_adjusted} adjusted price columns in feature frame (policy violation)",
            }
            highest_severity = min(highest_severity, 1)
        else:
            checks["raw_price_policy"] = {
                "tier": "INFO",
                "summary": "Features use raw OHLC only (policy compliant)",
            }

    # 3. Shift Test (Tier 2 capable - heuristic)
    shift_interpretation = interpret_shift_test_with_context(
        shift_result, autocorr_result, provenance_result
    )
    checks["shift_test"] = {
        "tier": shift_interpretation["tier"],
        "summary": shift_interpretation["interpretation"][:200],  # Truncate for summary
    }
    tier_severity = DIAGNOSTIC_TIERS[shift_interpretation["tier"]]["severity"]
    highest_severity = min(highest_severity, tier_severity)

    # 4. Target Autocorrelation (Tier 2/3 - context for interpretation)
    if autocorr_result and not autocorr_result.get("error"):
        autocorr = autocorr_result.get("lag1_autocorr", 0)
        pct_same = autocorr_result.get("pct_same_as_prev", 0)
        if autocorr > 0.5 or pct_same > 85:
            checks["target_autocorr"] = {
                "tier": "WARN",
                "summary": f"High target persistence: autocorr={autocorr:.2f}, {pct_same:.0f}% same as previous",
            }
            highest_severity = min(highest_severity, 2)
        elif autocorr > 0.3 or pct_same > 70:
            checks["target_autocorr"] = {
                "tier": "INFO",
                "summary": f"Moderate target persistence: autocorr={autocorr:.2f}, {pct_same:.0f}% same as previous",
            }
        else:
            checks["target_autocorr"] = {
                "tier": "INFO",
                "summary": f"Acceptable target persistence: autocorr={autocorr:.2f}",
            }

    # Determine overall tier
    tier_map = {1: "FAIL", 2: "WARN", 3: "INFO"}
    overall_tier = tier_map[highest_severity]

    # Build evidence summary
    fail_checks = [k for k, v in checks.items() if v["tier"] == "FAIL"]
    warn_checks = [k for k, v in checks.items() if v["tier"] == "WARN"]

    if fail_checks:
        evidence_summary = (
            f"CRITICAL: {len(fail_checks)} check(s) indicate deterministic leakage or policy violation. "
            f"Failed checks: {', '.join(fail_checks)}. "
            "These issues must be resolved before model training."
        )
    elif warn_checks:
        evidence_summary = (
            f"REVIEW REQUIRED: {len(warn_checks)} check(s) require review. "
            f"Flagged checks: {', '.join(warn_checks)}. "
            "These may indicate issues or may be acceptable given target structure."
        )
    else:
        evidence_summary = (
            "All leakage checks pass or show expected behavior. "
            "No evidence of problematic future data usage."
        )

    # Engineering judgment
    if overall_tier == "FAIL":
        engineering_judgment = (
            "BLOCK: Do not proceed with model training until FAIL issues are resolved. "
            "Deterministic violations indicate features contain future information."
        )
    elif overall_tier == "WARN":
        engineering_judgment = (
            "REVIEW: The shift test and/or other checks show signals that warrant review. "
            "If provenance checks pass and the residual signal is explainable by target "
            "autocorrelation or regime persistence, document the decision and proceed. "
            "Use purged cross-validation to mitigate temporal leakage effects."
        )
    else:
        engineering_judgment = (
            "PROCEED: All checks pass or show expected behavior. "
            "The feature set appears clean for model training. "
            "Standard temporal cross-validation practices are recommended."
        )

    return {
        "overall_tier": overall_tier,
        "checks": checks,
        "evidence_summary": evidence_summary,
        "engineering_judgment": engineering_judgment,
    }


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

# =============================================================================
# FEATURE VALUE RANGE VALIDATION
# Features with known bounded ranges - if outside bounds, indicates data issues
# (e.g., OHLC adjustment errors, indicator computation bugs)
# =============================================================================

BOUNDED_FEATURES = {
    # Format: feature_pattern -> (min_valid, max_valid, description, exclusions)
    # exclusions: list of substrings that should NOT match (to avoid false positives)

    # Choppiness Index: 0-100 by definition (OHLC-sensitive)
    "chop_": (0, 100, "Choppiness Index should be 0-100", []),

    # Position in range: 0-1 by definition (OHLC-sensitive)
    "pos_in_": (0, 1, "Position in range should be 0-1", []),

    # ADX components: typically 0-100 (OHLC-sensitive)
    "di_plus": (0, 100, "DI+ should be 0-100", []),
    "di_minus": (0, 100, "DI- should be 0-100", []),
    "adx_": (0, 100, "ADX should be 0-100", []),

    # Binary features: 0-1
    "breakout_up": (0, 1, "Breakout up flags should be 0 or 1", []),
    "breakout_dn": (0, 1, "Breakout down flags should be 0 or 1", []),

    # Vol regime: typically 0-3 normalized
    "vol_regime": (0, 3, "Vol regime typically 0-1 or 0-2", ["rel"]),

    # VIX regime: 0-2 (low/medium/high)
    "vix_regime": (0, 3, "VIX regime should be 0-2", []),
}


def validate_feature_ranges(df: pd.DataFrame, clip_outliers: bool = True) -> dict:
    """Validate that bounded features are within expected ranges.

    This catches OHLC adjustment issues that cause indicators like
    chop_14, pos_in_range, and DI+/- to produce invalid values.

    Args:
        df: DataFrame with features to validate
        clip_outliers: If True, clip rare outliers (<0.1% violations) to expected bounds

    Returns:
        Dict with validation results:
        - violations: list of (feature, issue_type, details)
        - summary: count of features with range violations
        - ohlc_issue_likely: bool indicating probable OHLC adjustment problem
        - clipped_features: list of features that were clipped
    """
    violations = []
    features_checked = 0
    clipped_features = []

    # Threshold for "rare outlier" vs "systematic issue"
    OUTLIER_THRESHOLD_PCT = 0.1  # <0.1% is considered rare outliers

    for col in df.columns:
        col_lower = col.lower()

        for pattern, (min_val, max_val, description, exclusions) in BOUNDED_FEATURES.items():
            if pattern in col_lower:
                # Check exclusions
                if any(excl in col_lower for excl in exclusions):
                    continue

                features_checked += 1

                # Get non-NaN values
                values = df[col].dropna()
                if len(values) == 0:
                    continue

                # Check for values outside bounds
                below_min = (values < min_val).sum()
                above_max = (values > max_val).sum()
                total = len(values)

                if below_min > 0 or above_max > 0:
                    pct_violations = (below_min + above_max) / total * 100

                    # Get actual range
                    actual_min = values.min()
                    actual_max = values.max()

                    # Determine severity based on percentage of violations
                    if pct_violations >= OUTLIER_THRESHOLD_PCT:
                        severity = "CRITICAL"  # Systemic issue
                    else:
                        severity = "INFO"  # Rare outliers

                    violations.append({
                        "feature": col,
                        "pattern": pattern,
                        "expected_range": (min_val, max_val),
                        "actual_range": (actual_min, actual_max),
                        "below_min_count": below_min,
                        "above_max_count": above_max,
                        "pct_violations": pct_violations,
                        "description": description,
                        "severity": severity,
                    })

                    # Clip rare outliers if requested
                    if clip_outliers and severity == "INFO":
                        df[col] = df[col].clip(lower=min_val, upper=max_val)
                        clipped_features.append(col)

                break  # Only check first matching pattern

    # Determine if this looks like an OHLC adjustment issue
    # Only consider it likely if there are CRITICAL (not INFO) violations
    ohlc_indicators = ["chop_", "pos_in_", "di_plus", "di_minus"]
    ohlc_violations = [v for v in violations
                       if any(ind in v["pattern"] for ind in ohlc_indicators)
                       and v.get("severity") == "CRITICAL"]
    ohlc_issue_likely = len(ohlc_violations) >= 2

    # Count critical vs info violations
    critical_violations = [v for v in violations if v.get("severity") == "CRITICAL"]
    info_violations = [v for v in violations if v.get("severity") == "INFO"]

    return {
        "violations": violations,
        "critical_violations": critical_violations,
        "info_violations": info_violations,
        "clipped_features": clipped_features,
        "features_checked": features_checked,
        "summary": len(violations),
        "ohlc_issue_likely": ohlc_issue_likely,
        "ohlc_violations": ohlc_violations,
    }


# =============================================================================
# RAW PRICE POLICY VALIDATION
# =============================================================================

# Pattern to detect adjusted price column names (prohibited in feature frame)
ADJUSTED_PRICE_PATTERNS = [
    r'^adj_',          # adj_close, adj_high, etc.
    r'^adjusted_',     # adjusted_close, adjusted_open, etc.
    r'_adjusted$',     # close_adjusted, etc.
    r'_adj$',          # close_adj, etc.
    r'^adjclose$',     # The adjclose column itself (only raw close allowed)
    r'split_factor',   # Split adjustment factors
    r'dividend_factor', # Dividend adjustment factors
    r'corp_action',    # Corporate action columns
    r'adjustment_factor', # Generic adjustment factor
]


def validate_raw_price_policy(df: pd.DataFrame) -> dict:
    """Validate that feature frame only contains raw-price-derived features.

    POLICY: Raw prices for features, adjusted prices for targets
    - Feature computation uses RAW OHLC (unadjusted)
    - Adjusted prices are only permitted for target generation and PnL
    - This check detects if adjusted price columns leaked into the feature frame

    Args:
        df: DataFrame with features to validate

    Returns:
        Dict with validation results:
        - passed: bool indicating if policy is satisfied
        - adjusted_columns: list of columns that violate the policy
        - warnings: list of warning messages
        - severity: "PASS", "WARN", or "FAIL"
    """
    import re

    adjusted_columns = []
    warnings = []

    # Check column names against patterns
    for col in df.columns:
        col_lower = col.lower()
        for pattern in ADJUSTED_PRICE_PATTERNS:
            if re.search(pattern, col_lower):
                adjusted_columns.append(col)
                break

    # Determine severity
    if adjusted_columns:
        severity = "FAIL"
        passed = False
        warnings.append(
            f"CRITICAL: {len(adjusted_columns)} adjusted price column(s) found in feature frame. "
            "Features must use raw OHLC only. Adjusted prices are only for targets/PnL."
        )
    else:
        severity = "PASS"
        passed = True

    # Additional check: look for column names with suspicious patterns
    suspicious_patterns = [
        (r'(?:^|_)adj(?:$|_)', "contains 'adj'"),
        (r'split', "contains 'split'"),
        (r'dividend', "contains 'dividend'"),
    ]

    suspicious_columns = []
    for col in df.columns:
        col_lower = col.lower()
        # Skip if already flagged as adjusted
        if col in adjusted_columns:
            continue
        for pattern, desc in suspicious_patterns:
            if re.search(pattern, col_lower):
                suspicious_columns.append((col, desc))
                break

    # Warn about suspicious columns (INFO level, not FAIL)
    if suspicious_columns and not adjusted_columns:
        # Only warn if not already failing
        warnings.append(
            f"INFO: {len(suspicious_columns)} column(s) have names suggesting adjustment usage. "
            "Verify these are not derived from adjusted prices."
        )

    return {
        "passed": passed,
        "adjusted_columns": adjusted_columns,
        "suspicious_columns": suspicious_columns,
        "warnings": warnings,
        "severity": severity,
        "policy_description": "Raw prices for features, adjusted prices for targets only",
    }


def run_shift_test(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    target_col: str = "hit_long_normal",
    max_samples: int = 100000,
) -> dict:
    """Run the shift test to detect statistical leakage.

    The shift test is a robust leakage detection method:
    1. Train a model normally → record AUC
    2. Shift all features forward by 1 bar (features[t] → features[t+1])
    3. Retrain with same settings → record AUC

    Interpretation:
    - AUC collapses (~0.50-0.52) → likely clean
    - AUC barely changes → very suspicious (features may contain future info)
    - AUC improves → almost certainly leaking

    Args:
        features_df: DataFrame with features (must have 'symbol' and 'date' columns)
        targets_df: DataFrame with targets (must have 'symbol' and 't0' columns)
        target_col: Target column to use (default: 'hit_top')
        max_samples: Maximum samples to use for speed (default: 100000)

    Returns:
        Dict with:
        - auc_normal: AUC with unshifted features
        - auc_shifted: AUC with shifted features (features[t] → features[t+1])
        - auc_drop: auc_normal - auc_shifted
        - interpretation: "clean", "suspicious", or "leaking"
        - leakage_likely: bool indicating probable data leakage
        - error: error message if test failed
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"error": "lightgbm or sklearn not installed", "leakage_likely": False}

    # Identify feature columns (numeric, not metadata)
    meta_cols = {'symbol', 'date', 'index', '_row_id'}
    feature_cols = [c for c in features_df.columns
                    if c not in meta_cols and features_df[c].dtype in [np.float32, np.float64]]

    if not feature_cols:
        return {"error": "No feature columns found", "leakage_likely": False}

    # Prepare features
    feat_df = features_df.copy()
    if 'date' not in feat_df.columns and feat_df.index.name:
        feat_df = feat_df.reset_index()

    # Prepare targets
    tgt_df = targets_df.copy()
    if 't0' in tgt_df.columns:
        tgt_df = tgt_df.rename(columns={'t0': 'date'})

    # Check target column exists
    if target_col not in tgt_df.columns:
        # Try common alternatives (triple barrier targets)
        for alt in ['hit_long_normal', 'hit_short_normal', 'hit_long_parabolic',
                    'hit_short_parabolic', 'hit_top', 'hit_bot', 'label', 'target', 'y']:
            if alt in tgt_df.columns:
                target_col = alt
                break
        else:
            return {"error": f"Target column '{target_col}' not found", "leakage_likely": False}

    # Merge features and targets
    try:
        if 'symbol' in feat_df.columns and 'symbol' in tgt_df.columns:
            merged = pd.merge(
                feat_df[['symbol', 'date'] + feature_cols],
                tgt_df[['symbol', 'date', target_col]],
                on=['symbol', 'date'],
                how='inner'
            )
        else:
            merged = pd.merge(
                feat_df[['date'] + feature_cols],
                tgt_df[['date', target_col]],
                on='date',
                how='inner'
            )
    except Exception as e:
        return {"error": f"Merge failed: {e}", "leakage_likely": False}

    if len(merged) < 2000:
        return {"error": f"Insufficient merged rows ({len(merged)})", "leakage_likely": False}

    # Drop rows with NaN target
    merged = merged.dropna(subset=[target_col])
    if len(merged) < 2000:
        return {"error": f"Insufficient rows after dropping NaN targets ({len(merged)})", "leakage_likely": False}

    # Sample if too large
    if len(merged) > max_samples:
        merged = merged.sample(n=max_samples, random_state=42)

    # Sort by symbol and date for proper shifting
    merged = merged.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Prepare X and y
    X = merged[feature_cols].values.astype(np.float32)
    y = merged[target_col].values.astype(np.float32)

    # Handle NaN in features by filling with 0 (LightGBM can handle this)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Check we have both classes
    unique_y = np.unique(y[~np.isnan(y)])
    if len(unique_y) < 2:
        return {"error": "Target has only one class", "leakage_likely": False}

    # Convert to binary if needed (for hit_top, values should already be 0/1)
    if not np.all(np.isin(y, [0, 1])):
        # Treat as binary: positive if > 0.5
        y = (y > 0.5).astype(np.float32)

    # Train/test split (use last 20% as test)
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # LightGBM parameters (fast defaults)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_samples': 20,
        'random_state': 42,
    }

    # Train normal model
    try:
        model_normal = lgb.LGBMClassifier(**params)
        model_normal.fit(X_train, y_train)
        y_pred_normal = model_normal.predict_proba(X_test)[:, 1]
        auc_normal = roc_auc_score(y_test, y_pred_normal)
    except Exception as e:
        return {"error": f"Normal model training failed: {e}", "leakage_likely": False}

    # Shift features forward by 1 bar within each symbol
    # This means features[t] → features[t+1], simulating that we're using
    # features that would only be available at t+1 to predict t
    try:
        merged_shifted = merged.copy()

        # Shift features forward within each symbol group
        # After shift, row t will have features from row t-1 (previous bar)
        for col in feature_cols:
            merged_shifted[col] = merged_shifted.groupby('symbol')[col].shift(1)

        # Drop rows with NaN from shifting (first row of each symbol)
        merged_shifted = merged_shifted.dropna(subset=feature_cols[:1])  # Check first feature col

        if len(merged_shifted) < 1000:
            return {"error": "Insufficient rows after shifting", "leakage_likely": False}

        # Re-sort and prepare shifted data
        merged_shifted = merged_shifted.sort_values(['symbol', 'date']).reset_index(drop=True)

        X_shifted = merged_shifted[feature_cols].values.astype(np.float32)
        y_shifted = merged_shifted[target_col].values.astype(np.float32)

        X_shifted = np.nan_to_num(X_shifted, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to binary if needed
        if not np.all(np.isin(y_shifted, [0, 1])):
            y_shifted = (y_shifted > 0.5).astype(np.float32)

        # Split
        n_shifted = len(X_shifted)
        split_idx_shifted = int(n_shifted * 0.8)
        X_train_s, X_test_s = X_shifted[:split_idx_shifted], X_shifted[split_idx_shifted:]
        y_train_s, y_test_s = y_shifted[:split_idx_shifted], y_shifted[split_idx_shifted:]

        # Train shifted model
        model_shifted = lgb.LGBMClassifier(**params)
        model_shifted.fit(X_train_s, y_train_s)
        y_pred_shifted = model_shifted.predict_proba(X_test_s)[:, 1]
        auc_shifted = roc_auc_score(y_test_s, y_pred_shifted)

    except Exception as e:
        return {"error": f"Shifted model training failed: {e}", "leakage_likely": False}

    # Compute AUC drop
    auc_drop = auc_normal - auc_shifted

    # Interpret results
    # Clean data: shifting should destroy predictive power → AUC drops to ~0.50
    # Leaky data: future info in features → shifting doesn't hurt or helps
    if auc_shifted < 0.52 and auc_drop > 0.02:
        interpretation = "clean"
        leakage_likely = False
    elif auc_drop < 0.01:
        interpretation = "suspicious"
        leakage_likely = True
    elif auc_drop < 0:
        interpretation = "leaking"
        leakage_likely = True
    elif auc_shifted > 0.55:
        # Even after shifting, model still has decent predictive power
        # This suggests features contain information about future targets
        interpretation = "suspicious"
        leakage_likely = True
    else:
        interpretation = "clean"
        leakage_likely = False

    # If leakage detected, identify suspicious features
    suspicious_features = []
    if leakage_likely:
        try:
            # Get feature importances from normal model
            importances = model_normal.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # For top 30 important features, check correlation persistence
            top_features = importance_df.head(30)['feature'].tolist()

            for feat in top_features:
                feat_idx = feature_cols.index(feat)

                # Get feature values and target
                feat_vals = X[:, feat_idx]
                target_vals = y

                # Compute correlation with current target
                valid = ~(np.isnan(feat_vals) | np.isnan(target_vals))
                if valid.sum() < 100:
                    continue

                feat_v = feat_vals[valid]
                tgt_v = target_vals[valid]

                if np.std(feat_v) < 1e-10:
                    continue

                corr_current = np.corrcoef(feat_v, tgt_v)[0, 1]

                # For shifted model: get shifted feature correlation with target
                # If correlation stays high, feature likely contains future info
                shifted_feat_vals = X_shifted[:, feat_idx]
                shifted_target_vals = y_shifted

                valid_s = ~(np.isnan(shifted_feat_vals) | np.isnan(shifted_target_vals))
                if valid_s.sum() < 100:
                    continue

                feat_v_s = shifted_feat_vals[valid_s]
                tgt_v_s = shifted_target_vals[valid_s]

                if np.std(feat_v_s) < 1e-10:
                    continue

                corr_shifted = np.corrcoef(feat_v_s, tgt_v_s)[0, 1]

                # Suspicious if: high importance AND correlation doesn't drop much
                importance = importances[feat_idx]
                corr_drop = abs(corr_current) - abs(corr_shifted)

                if abs(corr_shifted) > 0.1 or corr_drop < 0.02:
                    suspicious_features.append({
                        'feature': feat,
                        'importance': float(importance),
                        'importance_rank': int(importance_df[importance_df['feature'] == feat].index[0]) + 1,
                        'corr_current': float(corr_current),
                        'corr_shifted': float(corr_shifted),
                        'corr_drop': float(corr_drop),
                    })

            # Sort by importance
            suspicious_features.sort(key=lambda x: x['importance'], reverse=True)

        except Exception as e:
            # Don't fail the whole test if feature analysis fails
            pass

    return {
        "auc_normal": float(auc_normal),
        "auc_shifted": float(auc_shifted),
        "auc_drop": float(auc_drop),
        "interpretation": interpretation,
        "leakage_likely": leakage_likely,
        "target_used": target_col,
        "samples_used": len(merged),
        "features_used": len(feature_cols),
        "suspicious_features": suspicious_features,
    }


def check_target_autocorrelation(
    targets_df: pd.DataFrame,
    target_col: str = "hit_long_normal",
) -> dict:
    """Check target autocorrelation to detect regime persistence issues.

    High target autocorrelation can cause misleading model performance:
    - A model that just predicts "same as yesterday" can achieve high accuracy
    - This doesn't indicate real predictive power, just target persistence
    - Cross-validation may overestimate performance due to temporal leakage

    Args:
        targets_df: DataFrame with targets (must have 'symbol' and 't0' or 'date' columns)
        target_col: Target column to analyze (default: 'hit_long_normal')

    Returns:
        Dict with:
        - lag1_autocorr: Lag-1 autocorrelation (pooled across symbols)
        - pct_same_as_prev: Percentage of targets identical to previous
        - class_distribution: Dict of {class: percentage}
        - persistence_issue: bool indicating high autocorrelation
        - interpretation: Description of findings
        - by_symbol_stats: Per-symbol autocorrelation statistics
    """
    tgt_df = targets_df.copy()

    # Normalize date column name
    if 't0' in tgt_df.columns and 'date' not in tgt_df.columns:
        tgt_df = tgt_df.rename(columns={'t0': 'date'})

    # Check target column exists
    if target_col not in tgt_df.columns:
        # Try common alternatives
        for alt in ['hit_long_normal', 'hit_short_normal', 'hit_long_parabolic',
                    'hit_short_parabolic', 'hit_top', 'hit_bot', 'label', 'target', 'y']:
            if alt in tgt_df.columns:
                target_col = alt
                break
        else:
            return {"error": f"Target column '{target_col}' not found"}

    # Check required columns
    if 'symbol' not in tgt_df.columns:
        return {"error": "Column 'symbol' not found in targets"}
    if 'date' not in tgt_df.columns:
        return {"error": "Column 'date' (or 't0') not found in targets"}

    # Drop NaN targets
    tgt_df = tgt_df.dropna(subset=[target_col])
    if len(tgt_df) < 100:
        return {"error": f"Insufficient rows ({len(tgt_df)}) after dropping NaN targets"}

    # Sort by symbol and date for proper lag calculation
    tgt_df = tgt_df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Class distribution
    target_values = tgt_df[target_col].values
    unique_vals, counts = np.unique(target_values[~np.isnan(target_values)], return_counts=True)
    total = counts.sum()
    class_distribution = {float(v): float(c / total * 100) for v, c in zip(unique_vals, counts)}

    # Calculate lag-1 shifted target within each symbol
    tgt_df['_target_lag1'] = tgt_df.groupby('symbol')[target_col].shift(1)

    # Drop rows where lag is NaN (first row of each symbol)
    valid_rows = tgt_df.dropna(subset=['_target_lag1'])

    if len(valid_rows) < 100:
        return {"error": "Insufficient valid rows after computing lag"}

    current = valid_rows[target_col].values
    lagged = valid_rows['_target_lag1'].values

    # Calculate percentage same as previous
    same_as_prev = (current == lagged).sum()
    pct_same_as_prev = same_as_prev / len(current) * 100

    # Calculate lag-1 autocorrelation (pooled)
    # For binary targets, this is equivalent to Pearson correlation
    if np.std(current) > 1e-10 and np.std(lagged) > 1e-10:
        lag1_autocorr = np.corrcoef(current, lagged)[0, 1]
    else:
        lag1_autocorr = 0.0

    # Per-symbol autocorrelation statistics
    by_symbol_stats = {}
    for symbol, group in tgt_df.groupby('symbol'):
        g = group.dropna(subset=['_target_lag1'])
        if len(g) < 20:
            continue
        curr = g[target_col].values
        lag = g['_target_lag1'].values
        if np.std(curr) > 1e-10 and np.std(lag) > 1e-10:
            sym_autocorr = np.corrcoef(curr, lag)[0, 1]
        else:
            sym_autocorr = 0.0
        sym_same = (curr == lag).sum() / len(curr) * 100
        by_symbol_stats[symbol] = {
            'autocorr': float(sym_autocorr),
            'pct_same': float(sym_same),
            'n_rows': len(g),
        }

    # Aggregate per-symbol stats
    if by_symbol_stats:
        autocorrs = [s['autocorr'] for s in by_symbol_stats.values()]
        pct_sames = [s['pct_same'] for s in by_symbol_stats.values()]
        symbol_stats_summary = {
            'mean_autocorr': float(np.mean(autocorrs)),
            'median_autocorr': float(np.median(autocorrs)),
            'std_autocorr': float(np.std(autocorrs)),
            'mean_pct_same': float(np.mean(pct_sames)),
            'n_symbols': len(by_symbol_stats),
        }
    else:
        symbol_stats_summary = {}

    # Determine if there's a persistence issue
    # Thresholds: autocorr > 0.3 or same-as-previous > 70%
    persistence_issue = lag1_autocorr > 0.3 or pct_same_as_prev > 70

    # Interpretation
    if lag1_autocorr > 0.5 or pct_same_as_prev > 85:
        interpretation = "HIGH PERSISTENCE"
        detail = (
            "Targets are highly autocorrelated. A naive 'same as yesterday' model "
            "would achieve good accuracy. This inflates apparent model performance "
            "and may indicate targets are too persistent for reliable prediction."
        )
    elif lag1_autocorr > 0.3 or pct_same_as_prev > 70:
        interpretation = "MODERATE PERSISTENCE"
        detail = (
            "Targets show notable autocorrelation. Model validation should use "
            "purged/embargo cross-validation to avoid temporal leakage. Consider "
            "whether the prediction horizon is appropriate."
        )
    else:
        interpretation = "ACCEPTABLE"
        detail = (
            "Target autocorrelation is within acceptable range. Standard temporal "
            "cross-validation should be reliable."
        )

    return {
        "lag1_autocorr": float(lag1_autocorr),
        "pct_same_as_prev": float(pct_same_as_prev),
        "class_distribution": class_distribution,
        "persistence_issue": persistence_issue,
        "interpretation": interpretation,
        "interpretation_detail": detail,
        "target_used": target_col,
        "n_samples": len(valid_rows),
        "symbol_stats_summary": symbol_stats_summary,
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
                  expansion_analysis: Dict = None, range_analysis: Dict = None,
                  raw_price_analysis: Dict = None,
                  shift_test_analysis: Dict = None, autocorr_analysis: Dict = None,
                  provenance_analysis: Dict = None, verbose: bool = False):
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

    # === FEATURE RANGE VALIDATION ===
    if range_analysis:
        print(f"\n{'='*40}")
        print("FEATURE VALUE RANGE VALIDATION")
        print(f"{'='*40}")
        print(f"   Features checked: {range_analysis['features_checked']}")

        critical_violations = range_analysis.get("critical_violations", [])
        info_violations = range_analysis.get("info_violations", [])
        clipped_features = range_analysis.get("clipped_features", [])

        if critical_violations:
            # Systemic issues (>0.1% of values affected)
            print(f"   Critical range violations: {len(critical_violations)} features [FAIL]")

            # Check for OHLC-related issues
            if range_analysis.get("ohlc_issue_likely"):
                print(f"\n   [CRITICAL] OHLC ADJUSTMENT ISSUE DETECTED!")
                print(f"   Multiple OHLC-dependent indicators have invalid values.")
                print(f"   This typically means high/low/close prices are inconsistent.")
                print(f"   Fix: Ensure adjust_ohlc_to_adjclose() is applied before computing indicators.")

            print(f"\n   Features with systemic out-of-range values (>0.1%):")
            for v in critical_violations[:10]:
                feat = v["feature"]
                exp_min, exp_max = v["expected_range"]
                act_min, act_max = v["actual_range"]
                pct = v["pct_violations"]
                print(f"   - {feat}")
                print(f"     Expected: [{exp_min}, {exp_max}], Actual: [{act_min:.2f}, {act_max:.2f}]")
                print(f"     Violations: {pct:.1f}% of values out of range")

            if len(critical_violations) > 10:
                print(f"   ... and {len(critical_violations) - 10} more features with systemic issues")

        elif info_violations:
            # Only rare outliers - not a systemic issue
            print(f"   Range violations: 0 [PASS]")
            print(f"   All bounded features are within expected ranges.")

        else:
            print(f"   Range violations: 0 [PASS]")
            print(f"   All bounded features are within expected ranges.")

        # Show clipped outliers
        if clipped_features:
            print(f"\n   Rare outliers clipped (<0.1%): {len(clipped_features)} features [INFO]")
            if verbose:
                for feat in clipped_features[:5]:
                    # Find the violation info
                    v = next((x for x in info_violations if x["feature"] == feat), None)
                    if v:
                        exp_min, exp_max = v["expected_range"]
                        act_min, act_max = v["actual_range"]
                        count = v["below_min_count"] + v["above_max_count"]
                        print(f"   - {feat}: {count} values clipped to [{exp_min}, {exp_max}]")
                if len(clipped_features) > 5:
                    print(f"   ... and {len(clipped_features) - 5} more")

    # === RAW PRICE POLICY VALIDATION ===
    if raw_price_analysis:
        print(f"\n{'='*40}")
        print("RAW PRICE POLICY VALIDATION")
        print(f"{'='*40}")
        print(f"   Policy: {raw_price_analysis.get('policy_description', 'N/A')}")

        severity = raw_price_analysis.get("severity", "UNKNOWN")
        adjusted_cols = raw_price_analysis.get("adjusted_columns", [])
        suspicious_cols = raw_price_analysis.get("suspicious_columns", [])
        warnings = raw_price_analysis.get("warnings", [])

        if raw_price_analysis.get("passed"):
            print(f"   Status: PASS - No adjusted price columns in feature frame")
        else:
            print(f"   Status: FAIL - Adjusted price columns detected!")
            print(f"\n   [CRITICAL] Adjusted price column(s) found in feature frame:")
            for col in adjusted_cols[:10]:
                print(f"   - {col}")
            if len(adjusted_cols) > 10:
                print(f"   ... and {len(adjusted_cols) - 10} more")
            print(f"\n   POLICY VIOLATION: Features must use raw OHLC only.")
            print(f"   Adjusted prices are permitted only for target generation and PnL.")
            print(f"   Fix: Remove adjusted price columns from feature output or")
            print(f"        ensure feature computation uses raw close/high/low/open.")

        if suspicious_cols and raw_price_analysis.get("passed"):
            print(f"\n   Suspicious column names (review manually): {len(suspicious_cols)} [INFO]")
            for col, reason in suspicious_cols[:5]:
                print(f"   - {col} ({reason})")
            if len(suspicious_cols) > 5:
                print(f"   ... and {len(suspicious_cols) - 5} more")

        # Note about splits in rolling windows
        if raw_price_analysis.get("passed"):
            print(f"\n   Note: Splits within rolling windows are acceptable and expected")
            print(f"         in raw OHLC features. This is correct behavior.")

    # === FEATURE PROVENANCE VALIDATION ===
    if provenance_analysis:
        print(f"\n{'='*40}")
        print("FEATURE PROVENANCE (Leakage Prevention)")
        print(f"{'='*40}")

        if "error" in provenance_analysis:
            print(f"   Error: {provenance_analysis['error']}")
            if "recommendation" in provenance_analysis:
                print(f"   Recommendation: {provenance_analysis['recommendation']}")
        elif provenance_analysis.get("provenance_loaded"):
            summary = provenance_analysis.get("summary", {})
            total = summary.get("total_features", 0)
            daily = summary.get("daily_features", 0)
            weekly = summary.get("weekly_features", 0)
            max_lookback = summary.get("max_lookback_days", 0)
            min_lookback = summary.get("min_lookback_days", 0)
            with_pub_lag = summary.get("with_publication_lag", 0)

            print(f"   Features tracked: {total}")
            print(f"   Daily: {daily}, Weekly: {weekly}")
            print(f"   Lookback range: {min_lookback}-{max_lookback} days")
            print(f"   With publication lag: {with_pub_lag}")

            # Critical violations (hard leakage)
            critical = provenance_analysis.get("critical_violations", [])
            if critical:
                print(f"\n   [CRITICAL] HARD LEAKAGE DETECTED!")
                print(f"   {len(critical)} features have negative effective lookback (use future data)")
                for issue in critical[:5]:
                    print(f"   - {issue}")
                if len(critical) > 5:
                    print(f"   ... and {len(critical) - 5} more")
            else:
                print(f"\n   Hard leakage check: PASS (no negative lookbacks)")

            # Warnings (zero lookback)
            warnings = provenance_analysis.get("warnings", [])
            if warnings:
                print(f"\n   Zero lookback warnings: {len(warnings)} features")
                if verbose:
                    for warn in warnings[:5]:
                        print(f"   - {warn}")

            # Missing from registry
            missing = provenance_analysis.get("missing_from_registry", [])
            if missing:
                status = "WARN" if len(missing) > 50 else "INFO"
                print(f"\n   Features not in registry: {len(missing)} [{status}]")
                if verbose and len(missing) <= 20:
                    for feat in missing[:10]:
                        print(f"   - {feat}")
                    if len(missing) > 10:
                        print(f"   ... and {len(missing) - 10} more")

            # Optimization opportunities (publication lags that could be reduced)
            optimizations = provenance_analysis.get("optimizations", [])
            if optimizations:
                print(f"\n   Publication lag review: {len(optimizations)} features [INFO]")
                print(f"   These features have publication_lag buffers you added for data availability.")
                print(f"   Verify these are still needed - reducing them gives fresher signals:")
                for opt in optimizations[:5]:
                    print(f"   - {opt}")
                if len(optimizations) > 5:
                    print(f"   ... and {len(optimizations) - 5} more")
        else:
            print(f"   Provenance file not found - run pipeline to generate")

    # === SHIFT TEST FOR LEAKAGE (Context-Aware Interpretation) ===
    if shift_test_analysis:
        print(f"\n{'='*40}")
        print("SHIFT TEST FOR LEAKAGE")
        print(f"{'='*40}")

        if "error" in shift_test_analysis:
            print(f"   Error: {shift_test_analysis['error']}")
        else:
            auc_normal = shift_test_analysis.get("auc_normal", 0)
            auc_shifted = shift_test_analysis.get("auc_shifted", 0)
            auc_drop = shift_test_analysis.get("auc_drop", 0)
            target_used = shift_test_analysis.get("target_used", "unknown")
            samples_used = shift_test_analysis.get("samples_used", 0)
            features_used = shift_test_analysis.get("features_used", 0)

            print(f"   Target: {target_used}")
            print(f"   Samples: {samples_used:,}, Features: {features_used}")
            print(f"\n   AUC (normal):  {auc_normal:.4f}")
            print(f"   AUC (shifted): {auc_shifted:.4f}")
            print(f"   AUC drop:      {auc_drop:+.4f}")

            # Context-aware interpretation
            shift_context = interpret_shift_test_with_context(
                shift_test_analysis, autocorr_analysis, provenance_analysis
            )
            tier = shift_context["tier"]
            interpretation = shift_context["interpretation"]
            factors = shift_context.get("factors", {})
            action = shift_context.get("action", "")

            print(f"\n   Context-Aware Interpretation [{tier}]:")
            print(f"   {'-'*60}")

            # Print interpretation with word wrapping
            wrapped = textwrap.wrap(interpretation, width=60)
            for line in wrapped:
                print(f"   {line}")

            # Show contributing factors
            if factors:
                print(f"\n   Contributing Factors:")
                if "target_autocorr" in factors:
                    print(f"   - Target autocorrelation: {factors['target_autocorr']:.3f}")
                if "target_persistence_pct" in factors:
                    print(f"   - Target persistence: {factors['target_persistence_pct']:.1f}% same as previous")
                if "target_overlap_days" in factors:
                    print(f"   - Target window overlap: ~{factors['target_overlap_days']} days")
                if "provenance_violations" in factors:
                    print(f"   - Provenance violations: {factors['provenance_violations']}")

            print(f"\n   Recommended Action: {action}")

            # Show suspicious features only for WARN/FAIL tiers
            if tier in ["WARN", "FAIL"]:
                suspicious = shift_test_analysis.get("suspicious_features", [])
                if suspicious:
                    print(f"\n   Features with persistent correlation after shift:")
                    print(f"   {'Feature':<35} {'Imp Rank':>8} {'Corr Now':>9} {'Corr Shift':>10} {'Drop':>7}")
                    print(f"   {'-'*70}")
                    for sf in suspicious[:10]:
                        feat = sf['feature'][:34]
                        rank = sf['importance_rank']
                        corr_now = sf['corr_current']
                        corr_shift = sf['corr_shifted']
                        drop = sf['corr_drop']
                        print(f"   {feat:<35} {rank:>8} {corr_now:>+9.3f} {corr_shift:>+10.3f} {drop:>+7.3f}")
                    if len(suspicious) > 10:
                        print(f"   ... and {len(suspicious) - 10} more features")

    # === TARGET AUTOCORRELATION ===
    if autocorr_analysis:
        print(f"\n{'='*40}")
        print("TARGET AUTOCORRELATION CHECK")
        print(f"{'='*40}")

        if "error" in autocorr_analysis:
            print(f"   Error: {autocorr_analysis['error']}")
        else:
            lag1_autocorr = autocorr_analysis.get("lag1_autocorr", 0)
            pct_same = autocorr_analysis.get("pct_same_as_prev", 0)
            interpretation = autocorr_analysis.get("interpretation", "unknown")
            target_used = autocorr_analysis.get("target_used", "unknown")
            n_samples = autocorr_analysis.get("n_samples", 0)
            class_dist = autocorr_analysis.get("class_distribution", {})

            print(f"   Target: {target_used}")
            print(f"   Samples: {n_samples:,}")

            # Class distribution
            if class_dist:
                dist_str = ", ".join([f"{k:.0f}: {v:.1f}%" for k, v in sorted(class_dist.items())])
                print(f"   Class distribution: {dist_str}")

            print(f"\n   Lag-1 autocorrelation: {lag1_autocorr:.3f}")
            print(f"   Same as previous:     {pct_same:.1f}%")

            # Status based on interpretation
            if interpretation == "HIGH PERSISTENCE":
                status = "WARN"
                print(f"\n   Interpretation: {interpretation} [{status}]")
                print(f"   Targets are highly persistent. A naive 'same as yesterday'")
                print(f"   model would achieve {pct_same:.0f}% accuracy.")
            elif interpretation == "MODERATE PERSISTENCE":
                status = "CAUTION"
                print(f"\n   Interpretation: {interpretation} [{status}]")
                print(f"   Targets show notable autocorrelation. Use purged CV.")
            else:
                status = "PASS"
                print(f"\n   Interpretation: {interpretation} [{status}]")
                print(f"   Autocorrelation is within acceptable range.")

            # Per-symbol summary
            sym_stats = autocorr_analysis.get("symbol_stats_summary", {})
            if sym_stats:
                print(f"\n   Per-symbol statistics ({sym_stats.get('n_symbols', 0)} symbols):")
                print(f"   - Mean autocorr:   {sym_stats.get('mean_autocorr', 0):.3f}")
                print(f"   - Median autocorr: {sym_stats.get('median_autocorr', 0):.3f}")
                print(f"   - Std autocorr:    {sym_stats.get('std_autocorr', 0):.3f}")

    # === COMBINED LEAKAGE ASSESSMENT ===
    # Only show if we have at least one leakage-related check
    has_leakage_checks = (
        shift_test_analysis or provenance_analysis or raw_price_analysis
    )
    if has_leakage_checks:
        print(f"\n{'='*80}")
        print("COMBINED LEAKAGE ASSESSMENT")
        print(f"{'='*80}")

        # Create combined assessment
        combined = create_combined_leakage_assessment(
            shift_result=shift_test_analysis or {},
            autocorr_result=autocorr_analysis or {},
            provenance_result=provenance_analysis or {},
            raw_price_result=raw_price_analysis,
        )

        overall_tier = combined["overall_tier"]
        checks = combined["checks"]
        evidence_summary = combined["evidence_summary"]
        engineering_judgment = combined["engineering_judgment"]

        # Tier indicator with visual marker
        tier_markers = {"FAIL": "[X]", "WARN": "[!]", "INFO": "[i]"}
        tier_colors = {"FAIL": "BLOCK", "WARN": "REVIEW", "INFO": "PROCEED"}

        print(f"\n   Overall Assessment: {overall_tier} - {tier_colors[overall_tier]}")
        print(f"   {'-'*70}")

        # Individual check results
        print(f"\n   Check Results:")
        for check_name, check_info in checks.items():
            check_tier = check_info["tier"]
            marker = tier_markers[check_tier]
            summary = check_info["summary"][:65]
            print(f"   {marker} {check_name:<20} [{check_tier}]")
            print(f"       {summary}")

        # Evidence summary
        print(f"\n   Evidence Summary:")
        wrapped = textwrap.wrap(evidence_summary, width=68)
        for line in wrapped:
            print(f"   {line}")

        # Engineering Judgment (final recommendation)
        print(f"\n   " + "=" * 70)
        print(f"   ENGINEERING JUDGMENT")
        print(f"   " + "=" * 70)
        wrapped = textwrap.wrap(engineering_judgment, width=68)
        for line in wrapped:
            print(f"   {line}")

        # Additional guidance for WARN tier
        if overall_tier == "WARN":
            print(f"\n   Guidance for WARN tier:")
            print(f"   - The shift test is a heuristic, not proof of leakage")
            print(f"   - Modest residual AUC (0.53-0.60) is expected with overlapping targets")
            print(f"   - If provenance checks pass, regime persistence is likely cause")
            print(f"   - Document your decision and use purged cross-validation")

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

    # OHLC adjustment issues
    if range_analysis and range_analysis.get("ohlc_issue_likely"):
        recommendations.append(
            "OHLC ADJUSTMENT ISSUE: Multiple indicators have out-of-range values.\n"
            "   Affected indicators: chop_14, pos_in_range, di_plus, di_minus\n"
            "   Root cause: high/low/close price inconsistency (e.g., close > high)\n"
            "   Fix: Re-run pipeline to apply forward OHLC adjustment:\n"
            "        python -m src.cli.compute --timeframes D,W"
        )

    # Shift test leakage issues (now uses context-aware interpretation)
    if shift_test_analysis and not shift_test_analysis.get("error"):
        shift_context = interpret_shift_test_with_context(
            shift_test_analysis, autocorr_analysis, provenance_analysis
        )
        tier = shift_context.get("tier", "INFO")

        if tier == "FAIL":
            auc_drop = shift_test_analysis.get("auc_drop", 0)
            recommendations.append(
                f"SHIFT TEST: CRITICAL LEAKAGE DETECTED\n"
                f"   Tier: FAIL - Deterministic violation\n"
                f"   AUC drop after shifting: {auc_drop:+.4f}\n"
                f"   {shift_context.get('interpretation', '')[:200]}\n"
                f"   Action: {shift_context.get('action', 'Review feature computation')}"
            )
        elif tier == "WARN":
            auc_shifted = shift_test_analysis.get("auc_shifted", 0)
            recommendations.append(
                f"SHIFT TEST: Review Required\n"
                f"   Tier: WARN - Requires investigation\n"
                f"   Shifted AUC: {auc_shifted:.4f}\n"
                "   This may be acceptable if provenance checks pass and can be explained\n"
                "   by target autocorrelation or regime persistence.\n"
                f"   Action: {shift_context.get('action', 'Review top features')}"
        )

    # Target autocorrelation issues
    if autocorr_analysis and autocorr_analysis.get("persistence_issue"):
        lag1 = autocorr_analysis.get("lag1_autocorr", 0)
        pct_same = autocorr_analysis.get("pct_same_as_prev", 0)
        interpretation = autocorr_analysis.get("interpretation", "")
        recommendations.append(
            f"TARGET AUTOCORRELATION: Targets are highly persistent ({interpretation}).\n"
            f"   Lag-1 autocorrelation: {lag1:.3f} (threshold: 0.3)\n"
            f"   Same as previous: {pct_same:.1f}% (threshold: 70%)\n"
            "   This means a naive 'same as yesterday' predictor performs well.\n"
            "   Implications:\n"
            "   - Model performance may be inflated by temporal persistence\n"
            "   - Use purged/embargo cross-validation to get realistic estimates\n"
            "   - Consider shorter or different prediction horizons"
        )

    # Provenance issues
    if provenance_analysis:
        if provenance_analysis.get("hard_leakage_detected"):
            critical = provenance_analysis.get("critical_violations", [])
            recommendations.append(
                f"PROVENANCE LEAKAGE: {len(critical)} features have negative effective lookback.\n"
                "   This indicates features that use future data in their computation.\n"
                "   Fix: Review feature computation code for the flagged features\n"
                "   Fix: Ensure all rolling windows look backward only"
            )
        elif not provenance_analysis.get("provenance_loaded"):
            if "error" in provenance_analysis:
                recommendations.append(
                    "PROVENANCE MISSING: Feature provenance metadata not found.\n"
                    "   Provenance tracking enables deterministic leakage detection.\n"
                    "   Fix: Re-run the feature pipeline to generate provenance metadata\n"
                    "   Command: python -m src.cli.compute"
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


def validate_provenance_data(
    provenance_path: Path,
    features_df: pd.DataFrame,
) -> Dict:
    """Validate feature provenance metadata for leakage detection.

    This function performs the following checks:
    1. Hard leakage check: Verify no feature has negative effective lookback
    2. Target overlap check: Flag features with zero lookback (may use same-day data)
    3. Coverage check: Report features missing from provenance registry
    4. Summary statistics: Report provenance distribution

    Args:
        provenance_path: Path to feature_provenance.json
        features_df: DataFrame with features for cross-referencing

    Returns:
        Dict with:
        - provenance_loaded: bool indicating if provenance file was found
        - total_features: number of features in provenance
        - critical_violations: list of hard leakage violations
        - warnings: list of target overlap warnings
        - missing_from_registry: list of features in df but not in registry
        - summary: provenance summary statistics
        - hard_leakage_detected: bool indicating critical issue
    """
    if not HAS_PROVENANCE:
        return {"error": "Provenance module not available"}

    if not provenance_path.exists():
        return {
            "provenance_loaded": False,
            "error": f"Provenance file not found: {provenance_path}",
            "recommendation": "Re-run the feature pipeline to generate provenance metadata.",
        }

    try:
        # Load provenance metadata
        provenance = load_provenance_metadata(provenance_path)

        # Validate for issues
        critical_violations, warnings, optimizations = _validate_provenance_internal(provenance)

        # Get features from DataFrame
        meta_cols = {'symbol', 'date', '_row_id', 'index'}
        feature_cols = [c for c in features_df.columns if c not in meta_cols]

        # Check for features missing from registry
        missing_from_registry = get_missing_provenance(feature_cols)

        # Get summary statistics
        summary = report_provenance_summary(provenance)

        # Determine if hard leakage was detected
        hard_leakage_detected = len(critical_violations) > 0

        return {
            "provenance_loaded": True,
            "total_features": len(provenance),
            "critical_violations": critical_violations,
            "warnings": warnings,
            "optimizations": optimizations,  # Lags that could potentially be reduced
            "missing_from_registry": missing_from_registry,
            "summary": summary,
            "hard_leakage_detected": hard_leakage_detected,
            "provenance": provenance,  # Full provenance data for detailed analysis
        }

    except Exception as e:
        return {
            "provenance_loaded": False,
            "error": f"Failed to load provenance: {e}",
        }


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

    # Validate feature value ranges (catches OHLC adjustment issues)
    print("Validating feature value ranges...")
    range_analysis = validate_feature_ranges(df)

    # Validate raw price policy (no adjusted price columns in feature frame)
    print("Validating raw price policy...")
    raw_price_analysis = validate_raw_price_policy(df)

    # Validate feature provenance metadata (deterministic leakage detection)
    provenance_analysis = None
    provenance_path = features_path.parent / "feature_provenance.json"
    if HAS_PROVENANCE:
        print("Validating feature provenance...")
        provenance_analysis = validate_provenance_data(provenance_path, df)
    else:
        print("Provenance module not available - skipping provenance validation")

    # Run shift test for leakage detection
    shift_test_analysis = None
    autocorr_analysis = None
    if targets_path.exists():
        print("Running shift test for leakage detection...")
        targets_df = pd.read_parquet(targets_path)
        shift_test_analysis = run_shift_test(df, targets_df)

        print("Checking target autocorrelation...")
        autocorr_analysis = check_target_autocorrelation(targets_df)

    # Print summary
    print_summary(features_analysis, targets_analysis, base_features_analysis,
                  expansion_analysis=expansion_analysis, range_analysis=range_analysis,
                  raw_price_analysis=raw_price_analysis,
                  shift_test_analysis=shift_test_analysis, autocorr_analysis=autocorr_analysis,
                  provenance_analysis=provenance_analysis, verbose=args.verbose)


if __name__ == "__main__":
    main()
