"""
Feature Date Provenance Tracking for Leakage Prevention.

This module provides explicit temporal provenance tracking for all features
in the pipeline. It enables deterministic validation that features don't
leak future information into the training process.

Trading Semantics (Enforced):
- Feature row at timestamp `t` represents end-of-day data for date `t`
- Trade entry occurs at `t+1`
- Targets are defined relative to entry at `t+1`
- Valid: feature_max_source_date <= t
- Hard leakage: feature_max_source_date > t (CRITICAL)
- Target overlap: feature_max_source_date >= t+1 (WARNING)

Usage:
    from src.features.provenance import (
        FEATURE_PROVENANCE_REGISTRY,
        get_provenance_for_features,
        validate_provenance,
        save_provenance_metadata
    )

    # Get provenance for computed features
    feature_cols = [c for c in df.columns if c in FEATURE_PROVENANCE_REGISTRY]
    provenance = get_provenance_for_features(feature_cols)

    # Validate provenance (raises on hard leakage)
    validate_provenance(provenance)

    # Save to artifacts
    save_provenance_metadata(provenance, Path('artifacts/'))
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class LeakageType(str, Enum):
    """Types of temporal leakage."""
    NONE = "none"
    HARD = "hard"  # feature_max_source_date > row_date
    TARGET_OVERLAP = "target_overlap"  # feature_max_source_date >= entry_date (t+1)


@dataclass
class FeatureProvenance:
    """
    Provenance specification for a single feature.

    Attributes:
        feature_name: Name of the feature
        lookback_days: Number of trading days of historical data used
        publication_lag_days: Days between data date and availability (for FRED)
        depends_on: List of parent features (for derived/interaction features)
        timeframe: 'D' for daily, 'W' for weekly
        source_module: Module where feature is computed
        description: Human-readable description
    """
    feature_name: str
    lookback_days: int
    publication_lag_days: int = 0
    depends_on: List[str] = field(default_factory=list)
    timeframe: str = 'D'
    source_module: str = ''
    description: str = ''

    def effective_lookback_days(self) -> int:
        """
        Get total lookback in trading days.

        Weekly features have their lookback multiplied by 5 (trading days per week).
        Publication lag is added on top.
        """
        base = self.lookback_days
        if self.timeframe == 'W':
            base = self.lookback_days * 5
        return base + self.publication_lag_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'lookback_days': self.lookback_days,
            'publication_lag_days': self.publication_lag_days,
            'effective_lookback_days': self.effective_lookback_days(),
            'timeframe': self.timeframe,
            'source_module': self.source_module,
            'depends_on': self.depends_on if self.depends_on else None,
            'description': self.description if self.description else None,
        }


# =============================================================================
# FEATURE PROVENANCE REGISTRY
# =============================================================================
# Comprehensive registry of all features with their lookback windows.
# Organized by source module and feature category.

FEATURE_PROVENANCE_REGISTRY: Dict[str, FeatureProvenance] = {}


def _register(name: str, lookback: int, source: str = '', timeframe: str = 'D',
              pub_lag: int = 0, depends_on: List[str] = None, desc: str = '') -> None:
    """Helper to register a feature."""
    FEATURE_PROVENANCE_REGISTRY[name] = FeatureProvenance(
        feature_name=name,
        lookback_days=lookback,
        publication_lag_days=pub_lag,
        depends_on=depends_on or [],
        timeframe=timeframe,
        source_module=source,
        description=desc,
    )


# -----------------------------------------------------------------------------
# TREND FEATURES (trend.py)
# -----------------------------------------------------------------------------
# RSI: Uses 14-period price changes
_register('rsi_14', 14, 'trend', desc='14-period RSI')
_register('rsi_21', 21, 'trend', desc='21-period RSI')
_register('rsi_30', 30, 'trend', desc='30-period RSI')
_register('w_rsi_14', 14, 'trend', 'W', desc='Weekly 14-period RSI')
_register('w_rsi_21', 21, 'trend', 'W', desc='Weekly 21-period RSI')

# MACD: Uses 26-day slow EMA + 9-day signal = 35 days effective
_register('macd_histogram', 35, 'trend', desc='MACD histogram (12/26/9)')
_register('macd_hist_deriv_ema3', 38, 'trend', desc='MACD histogram derivative')
_register('w_macd_histogram', 35, 'trend', 'W', desc='Weekly MACD histogram')
_register('w_macd_hist_deriv_ema3', 38, 'trend', 'W')

# Trend scores: Based on MA agreements (longest MA is 200)
_register('trend_score_sign', 200, 'trend', desc='Multi-MA trend direction')
_register('trend_score_granular', 200, 'trend', desc='Granular trend score')
_register('trend_score_slope', 220, 'trend', desc='Trend score rate of change')
_register('trend_persist_ema', 20, 'trend', desc='Consecutive up/down days EMA')
_register('w_trend_score_sign', 200, 'trend', 'W')
_register('w_trend_score_granular', 200, 'trend', 'W')
_register('w_trend_score_slope', 220, 'trend', 'W')
_register('w_trend_persist_ema', 20, 'trend', 'W')

# MA slopes: Slope window + MA period
_register('pct_slope_ma_10', 30, 'trend', desc='10-day MA slope (20d window)')
_register('pct_slope_ma_20', 40, 'trend', desc='20-day MA slope')
_register('pct_slope_ma_30', 50, 'trend', desc='30-day MA slope')
_register('pct_slope_ma_50', 70, 'trend', desc='50-day MA slope')
_register('pct_slope_ma_100', 120, 'trend', desc='100-day MA slope')
_register('pct_slope_ma_150', 170, 'trend', desc='150-day MA slope')
_register('pct_slope_ma_200', 220, 'trend', desc='200-day MA slope')
_register('w_pct_slope_ma_20', 40, 'trend', 'W')
_register('w_pct_slope_ma_50', 70, 'trend', 'W')
_register('w_pct_slope_ma_100', 120, 'trend', 'W')

# ADX/DI: 14-period smoothed indicators
_register('adx_14', 28, 'trend', desc='Average Directional Index')
_register('di_plus_14', 14, 'trend', desc='Positive Directional Indicator')
_register('di_minus_14', 14, 'trend', desc='Negative Directional Indicator')

# Choppiness Index: Uses 14-period ATR
_register('chop_14', 14, 'trend', desc='Choppiness Index')

# Quiet trend and alignment: Based on trend + vol
_register('quiet_trend', 100, 'trend', desc='Low volatility trend indicator')
_register('trend_alignment', 200, 'trend', desc='Multi-timeframe trend alignment')
_register('w_quiet_trend', 100, 'trend', 'W')
_register('w_trend_alignment', 200, 'trend', 'W')

# -----------------------------------------------------------------------------
# VOLATILITY FEATURES (volatility.py)
# -----------------------------------------------------------------------------
# Realized volatility windows
_register('rv_10', 10, 'volatility', desc='10-day realized volatility')
_register('rv_20', 20, 'volatility', desc='20-day realized volatility')
_register('rv_60', 60, 'volatility', desc='60-day realized volatility')
_register('rv_100', 100, 'volatility', desc='100-day realized volatility')

# RV ratios and z-scores
_register('rv_ratio_10_60', 60, 'volatility', desc='RV 10/60 ratio')
_register('rv_ratio_20_100', 100, 'volatility', desc='RV 20/100 ratio')
_register('rv_z_60', 120, 'volatility', desc='RV z-score (60d window)')
_register('w_rv_z_60', 120, 'volatility', 'W')

# Vol regime: Uses 100-day RV + 10-day EMA = 110 days
_register('vol_regime', 100, 'volatility', desc='Volatility regime (0-1)')
_register('vol_regime_ema10', 110, 'volatility', desc='Smoothed vol regime')
_register('w_vol_regime', 100, 'volatility', 'W')
_register('w_vol_regime_ema10', 110, 'volatility', 'W')
_register('vol_regime_cs_median', 110, 'volatility', desc='Cross-sectional vol median')
_register('vol_regime_rel', 110, 'volatility', desc='Relative vol regime')
_register('w_vol_regime_rel', 110, 'volatility', 'W')

# ATR features
_register('atr14', 14, 'volatility', desc='14-period ATR')
_register('atr_percent', 14, 'volatility', desc='ATR as % of price')
_register('atr_percent_chg_5', 19, 'volatility', desc='5-day change in ATR%')
_register('w_atr_percent', 14, 'volatility', 'W')

# Bollinger features
_register('bb_width_20_2', 20, 'volatility', desc='Bollinger bandwidth')
_register('bb_width_20_2_z60', 80, 'volatility', desc='BB width z-score (60d)')

# Squeeze features
_register('squeeze_on_20', 20, 'volatility', desc='BB inside KC (narrow)')
_register('squeeze_on_wide_20', 20, 'volatility', desc='BB inside KC (wide)')
_register('squeeze_intensity_20', 20, 'volatility', desc='Squeeze intensity')
_register('squeeze_release_20', 21, 'volatility', desc='Squeeze release signal')
_register('days_in_squeeze_20', 70, 'volatility', desc='Consecutive squeeze days')

# Volatility acceleration (2nd derivatives)
_register('rv_delta_10_60', 60, 'volatility', desc='RV10 - RV60 difference')
_register('rv_delta_10_60_z', 120, 'volatility', desc='Z-scored vol delta')
_register('rv_accel_20', 85, 'volatility', desc='5-day change in RV20, z-scored')
_register('rv_accel_60', 190, 'volatility', desc='10-day change in RV60, z-scored')
_register('rv_impulse_5d_z', 70, 'volatility', desc='5-day vol impulse, z-scored')
_register('w_rv_delta_10_60_z', 120, 'volatility', 'W')
_register('w_rv_accel_20', 85, 'volatility', 'W')

# RV slope
_register('rv60_slope_norm', 80, 'volatility', desc='60-day RV slope')
_register('w_rv60_slope_norm', 80, 'volatility', 'W')
_register('rv100_slope_norm', 120, 'volatility', desc='100-day RV slope')
_register('w_rv100_slope_norm', 120, 'volatility', 'W')

# -----------------------------------------------------------------------------
# DISTANCE TO MA FEATURES (distance.py)
# -----------------------------------------------------------------------------
_register('pct_dist_ma_20', 20, 'distance', desc='% distance from 20-day MA')
_register('pct_dist_ma_50', 50, 'distance', desc='% distance from 50-day MA')
_register('pct_dist_ma_100', 100, 'distance', desc='% distance from 100-day MA')
_register('pct_dist_ma_200', 200, 'distance', desc='% distance from 200-day MA')

# Z-scored distances (add 60d for z-score window)
_register('pct_dist_ma_20_z', 80, 'distance', desc='Z-scored distance from 20d MA')
_register('pct_dist_ma_50_z', 110, 'distance', desc='Z-scored distance from 50d MA')
_register('pct_dist_ma_100_z', 160, 'distance', desc='Z-scored distance from 100d MA')
_register('pct_dist_ma_200_z', 260, 'distance', desc='Z-scored distance from 200d MA')

# Min distance to any MA
_register('min_pct_dist_ma', 200, 'distance', desc='Distance to nearest MA')
_register('w_min_pct_dist_ma', 200, 'distance', 'W')

# Relative distances
_register('relative_dist_20_50', 50, 'distance', desc='Relative MA20 vs MA50 position')
_register('relative_dist_20_50_z', 110, 'distance', desc='Z-scored relative distance')
_register('w_relative_dist_20_50_z', 110, 'distance', 'W')

# Weekly distance features
_register('w_pct_dist_ma_20', 20, 'distance', 'W')
_register('w_pct_dist_ma_50', 50, 'distance', 'W')
_register('w_pct_dist_ma_20_z', 80, 'distance', 'W')
_register('w_pct_dist_ma_50_z', 110, 'distance', 'W')
_register('w_pct_dist_ma_100_z', 160, 'distance', 'W')

# -----------------------------------------------------------------------------
# RANGE/BREAKOUT FEATURES (range_breakout.py)
# -----------------------------------------------------------------------------
# Position in range features
_register('pos_in_5d_range', 5, 'range_breakout', desc='Position in 5-day range')
_register('pos_in_10d_range', 10, 'range_breakout', desc='Position in 10-day range')
_register('pos_in_20d_range', 20, 'range_breakout', desc='Position in 20-day range')
_register('w_pos_in_5d_range', 5, 'range_breakout', 'W')
_register('w_pos_in_10d_range', 10, 'range_breakout', 'W')
_register('w_pos_in_20d_range', 20, 'range_breakout', 'W')

# Breakout signals
_register('breakout_up_5d', 5, 'range_breakout', desc='Broke above 5-day high')
_register('breakout_up_10d', 10, 'range_breakout', desc='Broke above 10-day high')
_register('breakout_up_20d', 20, 'range_breakout', desc='Broke above 20-day high')
_register('breakout_dn_5d', 5, 'range_breakout', desc='Broke below 5-day low')
_register('breakout_dn_10d', 10, 'range_breakout', desc='Broke below 10-day low')
_register('breakout_dn_20d', 20, 'range_breakout', desc='Broke below 20-day low')
_register('w_breakout_up_20d', 20, 'range_breakout', 'W')
_register('w_breakout_dn_20d', 20, 'range_breakout', 'W')

# Range expansion
_register('range_expansion_5d', 10, 'range_breakout', desc='5-day range expansion')
_register('range_expansion_10d', 20, 'range_breakout', desc='10-day range expansion')
_register('range_expansion_20d', 40, 'range_breakout', desc='20-day range expansion')
_register('range_z_20d', 80, 'range_breakout', desc='20-day range z-score')
_register('w_range_expansion_20d', 40, 'range_breakout', 'W')
_register('w_range_z_20d', 80, 'range_breakout', 'W')

# Gap features (use EOD t data: open, high, low, close, prev_close - all known at EOD t)
_register('overnight_ret', 1, 'range_breakout', desc='Overnight return (open/prev_close - 1)')
_register('gap_atr_ratio', 15, 'range_breakout', desc='Daily return / ATR%')
_register('gap_atr_ratio_raw', 15, 'range_breakout', desc='Gap size / ATR')
_register('gap_fill_frac', 2, 'range_breakout', desc='Gap fill fraction')
_register('range_efficiency', 1, 'range_breakout', desc='Close move / HL range')
_register('w_range_efficiency', 1, 'range_breakout', 'W')

# -----------------------------------------------------------------------------
# VOLUME FEATURES (volume.py)
# -----------------------------------------------------------------------------
_register('obv_z_60', 120, 'volume', desc='OBV z-score (60d window)')
_register('w_obv_z_60', 120, 'volume', 'W')
_register('rdollar_vol_20', 20, 'volume', desc='Relative dollar volume')
_register('w_rdollar_vol_20', 20, 'volume', 'W')
_register('volshock_z', 60, 'volume', desc='Volume shock z-score')
_register('volshock_ema', 30, 'volume', desc='Volume shock EMA')
_register('volshock_dir', 30, 'volume', desc='Volume shock direction')
_register('w_volshock_z', 60, 'volume', 'W')
_register('w_volshock_ema', 30, 'volume', 'W')
_register('w_volshock_dir', 30, 'volume', 'W')
_register('pv_divergence_5d', 5, 'volume', desc='Price-volume divergence')
_register('volume_direction', 10, 'volume', desc='Volume-weighted direction')
_register('volume_trend_10d', 10, 'volume', desc='Volume trend (10d)')
_register('rel_volume_5d', 5, 'volume', desc='Relative volume vs 5d avg')
_register('rel_volume_10d', 10, 'volume', desc='Relative volume vs 10d avg')
_register('rel_volume_20d', 20, 'volume', desc='Relative volume vs 20d avg')
_register('w_rel_volume_5d', 5, 'volume', 'W')
_register('w_rel_volume_10d', 10, 'volume', 'W')
_register('w_rel_volume_20d', 20, 'volume', 'W')

# -----------------------------------------------------------------------------
# LIQUIDITY FEATURES (liquidity.py)
# -----------------------------------------------------------------------------
_register('overnight_ratio', 2, 'liquidity', desc='Overnight vs intraday ratio')
_register('upper_shadow_ratio', 1, 'liquidity', desc='Upper shadow / range')
_register('lower_shadow_ratio', 1, 'liquidity', desc='Lower shadow / range')
_register('vwap_dist_5d_zscore', 65, 'liquidity', desc='VWAP distance z-score (5d)')
_register('vwap_dist_10d_zscore', 70, 'liquidity', desc='VWAP distance z-score (10d)')
_register('vwap_dist_20d_zscore', 80, 'liquidity', desc='VWAP distance z-score (20d)')
_register('w_vwap_dist_20d_zscore', 80, 'liquidity', 'W')
_register('amihud_illiq_ratio', 20, 'liquidity', desc='Amihud illiquidity ratio')
_register('illiquidity_score', 60, 'liquidity', desc='Composite illiquidity score')
_register('w_illiquidity_score', 60, 'liquidity', 'W')
_register('hl_spread_proxy', 1, 'liquidity', desc='High-low spread proxy')
_register('cs_spread_est', 1, 'liquidity', desc='Cross-sectional spread estimate')
_register('roll_spread_est', 2, 'liquidity', desc='Roll spread estimate')

# -----------------------------------------------------------------------------
# DRAWDOWN FEATURES (drawdown.py)
# -----------------------------------------------------------------------------
_register('drawdown_20d', 20, 'drawdown', desc='Drawdown from 20d high')
_register('drawdown_60d', 60, 'drawdown', desc='Drawdown from 60d high')
_register('drawdown_120d', 120, 'drawdown', desc='Drawdown from 120d high')
_register('drawdown_expanding', 252, 'drawdown', desc='Expanding drawdown from ATH')
_register('drawdown_20d_z', 80, 'drawdown', desc='Z-scored 20d drawdown')
_register('drawdown_60d_z', 120, 'drawdown', desc='Z-scored 60d drawdown')
_register('drawdown_120d_z', 180, 'drawdown', desc='Z-scored 120d drawdown')
_register('drawdown_regime', 60, 'drawdown', desc='Drawdown regime indicator')
_register('days_since_high_20d_norm', 40, 'drawdown', desc='Normalized days since 20d high')
_register('days_since_high_60d_norm', 120, 'drawdown', desc='Normalized days since 60d high')
_register('days_since_high_120d_norm', 240, 'drawdown', desc='Normalized days since 120d high')
_register('recovery_20d', 20, 'drawdown', desc='Recovery from 20d low')
_register('recovery_60d', 60, 'drawdown', desc='Recovery from 60d low')
_register('recovery_120d', 120, 'drawdown', desc='Recovery from 120d low')
_register('recovery_20d_z', 80, 'drawdown', desc='Z-scored 20d recovery')
_register('recovery_60d_z', 120, 'drawdown', desc='Z-scored 60d recovery')
_register('recovery_120d_z', 180, 'drawdown', desc='Z-scored 120d recovery')
_register('drawdown_velocity_20d', 25, 'drawdown', desc='Drawdown velocity (20d)')
_register('drawdown_velocity_60d', 70, 'drawdown', desc='Drawdown velocity (60d)')
_register('drawdown_velocity_120d', 130, 'drawdown', desc='Drawdown velocity (120d)')
_register('hl_range_position_60d', 60, 'drawdown', desc='Position in 60d H-L range')
_register('w_drawdown_60d', 60, 'drawdown', 'W')
_register('w_drawdown_60d_z', 120, 'drawdown', 'W')
_register('w_days_since_high_60d_norm', 120, 'drawdown', 'W')
_register('w_recovery_60d', 60, 'drawdown', 'W')
_register('w_drawdown_velocity_60d', 70, 'drawdown', 'W')

# -----------------------------------------------------------------------------
# DIVERGENCE FEATURES (divergence.py)
# -----------------------------------------------------------------------------
_register('rsi_price_div_10d', 24, 'divergence', desc='RSI-price divergence (10d)')
_register('rsi_price_div_20d', 34, 'divergence', desc='RSI-price divergence (20d)')
_register('rsi_price_div_cum_10d', 24, 'divergence', desc='Cumulative RSI-price div')
_register('rsi_price_div_cum_20d', 34, 'divergence', desc='Cumulative RSI-price div')
_register('macd_price_div_10d', 45, 'divergence', desc='MACD-price divergence (10d)')
_register('macd_price_div_20d', 55, 'divergence', desc='MACD-price divergence (20d)')
_register('trend_rsi_div_10d', 24, 'divergence', desc='Trend-RSI divergence')
_register('trend_rsi_div_20d', 34, 'divergence', desc='Trend-RSI divergence')
_register('vol_trend_div_10d', 70, 'divergence', desc='Vol-trend divergence')
_register('vol_trend_div_20d', 80, 'divergence', desc='Vol-trend divergence')
_register('w_rsi_price_div_20d', 34, 'divergence', 'W')
_register('w_macd_price_div_20d', 55, 'divergence', 'W')

# -----------------------------------------------------------------------------
# CROSS-SECTIONAL FEATURES (cross_sectional.py)
# -----------------------------------------------------------------------------
_register('xsec_mom_5d_z', 5, 'cross_sectional', desc='5d momentum z-score')
_register('xsec_mom_20d_z', 20, 'cross_sectional', desc='20d momentum z-score')
_register('xsec_mom_60d_z', 60, 'cross_sectional', desc='60d momentum z-score')
_register('xsec_mom_20d_sect_neutral_z', 20, 'cross_sectional', desc='Sector-neutral mom')
_register('xsec_pct_20d', 20, 'cross_sectional', desc='20d return percentile')
_register('xsec_pct_60d', 60, 'cross_sectional', desc='60d return percentile')
_register('w_xsec_mom_1w_z', 1, 'cross_sectional', 'W', desc='Weekly 1w mom z-score')
_register('w_xsec_mom_4w_z', 4, 'cross_sectional', 'W', desc='Weekly 4w mom z-score')
_register('w_xsec_mom_13w_z', 13, 'cross_sectional', 'W', desc='Weekly 13w mom z-score')
_register('w_xsec_pct_4w', 4, 'cross_sectional', 'W')
_register('w_xsec_pct_13w', 13, 'cross_sectional', 'W')

# -----------------------------------------------------------------------------
# ALPHA/BETA FEATURES (factor_regression.py)
# -----------------------------------------------------------------------------
# Simple betas (rolling regression windows)
_register('beta_spy_simple', 60, 'factor_regression', desc='Rolling beta vs SPY')
_register('beta_qqq_simple', 60, 'factor_regression', desc='Rolling beta vs QQQ')
_register('beta_sector', 60, 'factor_regression', desc='Rolling beta vs sector')
_register('w_beta_spy_simple', 60, 'factor_regression', 'W')
_register('w_beta_qqq_simple', 60, 'factor_regression', 'W')
_register('w_beta_qqq', 60, 'factor_regression', 'W')

# Alpha momentum (rolling window + EMA smoothing)
_register('alpha_mom_spy_20_ema10', 30, 'factor_regression', desc='Alpha vs SPY (20d, EMA10)')
_register('alpha_mom_spy_60_ema10', 70, 'factor_regression', desc='Alpha vs SPY (60d, EMA10)')
_register('alpha_mom_spy_120_ema10', 130, 'factor_regression', desc='Alpha vs SPY (120d, EMA10)')
_register('alpha_mom_qqq_20_ema10', 30, 'factor_regression', desc='Alpha vs QQQ (20d)')
_register('alpha_mom_qqq_60_ema10', 70, 'factor_regression', desc='Alpha vs QQQ (60d)')
_register('alpha_mom_sector_20_ema10', 30, 'factor_regression', desc='Alpha vs sector (20d)')
_register('alpha_mom_sector_60_ema10', 70, 'factor_regression', desc='Alpha vs sector (60d)')
_register('alpha_mom_combo_20_ema10', 30, 'factor_regression', desc='Combo alpha (20d)')
_register('alpha_mom_combo_60_ema10', 70, 'factor_regression', desc='Combo alpha (60d)')
_register('w_alpha_mom_spy_20_ema10', 30, 'factor_regression', 'W')
_register('w_alpha_mom_spy_60_ema10', 70, 'factor_regression', 'W')
_register('w_alpha_mom_qqq_60_ema10', 70, 'factor_regression', 'W')
_register('w_alpha_mom_sector_60_ema10', 70, 'factor_regression', 'W')
_register('w_alpha_mom_combo_60_ema10', 70, 'factor_regression', 'W')

# Joint factor model betas
_register('beta_market', 60, 'factor_regression', desc='Joint model: market beta')
_register('beta_qqq', 60, 'factor_regression', desc='Joint model: QQQ beta')
_register('beta_bestmatch', 60, 'factor_regression', desc='Joint model: best-match beta')
_register('beta_breadth', 60, 'factor_regression', desc='Joint model: breadth beta')
_register('w_beta_market', 60, 'factor_regression', 'W')
_register('w_beta_bestmatch', 60, 'factor_regression', 'W')
_register('w_beta_breadth', 60, 'factor_regression', 'W')

# Residual features
_register('residual_cumret', 60, 'factor_regression', desc='Cumulative residual return')
_register('residual_vol', 60, 'factor_regression', desc='Residual volatility')
_register('residual_mean', 60, 'factor_regression', desc='Mean residual')
_register('w_residual_cumret', 60, 'factor_regression', 'W')
_register('w_residual_vol', 60, 'factor_regression', 'W')

# -----------------------------------------------------------------------------
# RELATIVE STRENGTH FEATURES (cross_sectional.py)
# -----------------------------------------------------------------------------
_register('rel_strength_spy', 20, 'cross_sectional', desc='Relative strength vs SPY')
_register('rel_strength_qqq', 20, 'cross_sectional', desc='Relative strength vs QQQ')
_register('rel_strength_sector', 20, 'cross_sectional', desc='Relative strength vs sector')
_register('rel_strength_spy_zscore', 80, 'cross_sectional', desc='RS vs SPY z-scored')
_register('rel_strength_qqq_zscore', 80, 'cross_sectional', desc='RS vs QQQ z-scored')
_register('rel_strength_sector_zscore', 80, 'cross_sectional', desc='RS vs sector z-scored')
_register('w_rel_strength_spy', 20, 'cross_sectional', 'W')
_register('w_rel_strength_qqq', 20, 'cross_sectional', 'W')
_register('w_rel_strength_sector', 20, 'cross_sectional', 'W')
_register('w_rel_strength_spy_zscore', 80, 'cross_sectional', 'W')

# -----------------------------------------------------------------------------
# SPREAD FEATURES (spread_features.py)
# -----------------------------------------------------------------------------
# QQQ spreads
_register('qqq_cumret_20', 20, 'spread_features', desc='QQQ cumulative return 20d')
_register('qqq_cumret_60', 60, 'spread_features', desc='QQQ cumulative return 60d')
_register('qqq_cumret_120', 120, 'spread_features', desc='QQQ cumulative return 120d')
_register('qqq_zscore_60', 120, 'spread_features', desc='QQQ z-score (60d)')
_register('qqq_slope_20', 20, 'spread_features', desc='QQQ slope 20d')
_register('qqq_slope_60', 60, 'spread_features', desc='QQQ slope 60d')

# SPY spreads
_register('spy_cumret_20', 20, 'spread_features', desc='SPY cumulative return 20d')
_register('spy_cumret_60', 60, 'spread_features', desc='SPY cumulative return 60d')
_register('spy_cumret_120', 120, 'spread_features', desc='SPY cumulative return 120d')
_register('spy_zscore_60', 120, 'spread_features', desc='SPY z-score (60d)')

# QQQ-SPY spreads
_register('qqq_spy_cumret_20', 20, 'spread_features', desc='QQQ-SPY cumret 20d')
_register('qqq_spy_cumret_60', 60, 'spread_features', desc='QQQ-SPY cumret 60d')
_register('qqq_spy_cumret_120', 120, 'spread_features', desc='QQQ-SPY cumret 120d')
_register('qqq_spy_zscore_60', 120, 'spread_features', desc='QQQ-SPY z-score')
_register('qqq_spy_slope_20', 20, 'spread_features', desc='QQQ-SPY slope 20d')
_register('qqq_spy_slope_60', 60, 'spread_features', desc='QQQ-SPY slope 60d')
_register('w_qqq_spy_cumret_12', 12, 'spread_features', 'W')
_register('w_qqq_spy_zscore_12', 24, 'spread_features', 'W')
_register('w_qqq_spy_slope_4', 4, 'spread_features', 'W')

# RSP-SPY spreads
_register('rsp_spy_cumret_20', 20, 'spread_features', desc='RSP-SPY cumret 20d')
_register('rsp_spy_cumret_60', 60, 'spread_features', desc='RSP-SPY cumret 60d')
_register('rsp_spy_cumret_120', 120, 'spread_features', desc='RSP-SPY cumret 120d')
_register('rsp_spy_zscore_60', 120, 'spread_features', desc='RSP-SPY z-score')
_register('w_rsp_spy_cumret_12', 12, 'spread_features', 'W')
_register('w_rsp_spy_zscore_12', 24, 'spread_features', 'W')

# Bestmatch spreads
_register('bestmatch_spy_cumret_20', 20, 'spread_features', desc='Bestmatch-SPY cumret 20d')
_register('bestmatch_spy_cumret_60', 60, 'spread_features', desc='Bestmatch-SPY cumret 60d')
_register('bestmatch_spy_cumret_120', 120, 'spread_features', desc='Bestmatch-SPY cumret 120d')
_register('bestmatch_spy_zscore_60', 120, 'spread_features', desc='Bestmatch-SPY z-score')
_register('w_bestmatch_spy_cumret_12', 12, 'spread_features', 'W')
_register('w_bestmatch_spy_zscore_12', 24, 'spread_features', 'W')

# -----------------------------------------------------------------------------
# SECTOR BREADTH FEATURES (sector_breadth.py)
# -----------------------------------------------------------------------------
_register('sector_breadth_net_adv', 1, 'sector_breadth', desc='Net advancing sectors')
_register('sector_breadth_ad_line', 252, 'sector_breadth', desc='Cumulative A/D line')
_register('sector_breadth_pct_above_ma10', 10, 'sector_breadth', desc='% above 10d MA')
_register('sector_breadth_pct_above_ma50', 50, 'sector_breadth', desc='% above 50d MA')
_register('sector_breadth_pct_above_ma200', 200, 'sector_breadth', desc='% above 200d MA')
_register('sector_breadth_mcclellan_osc', 39, 'sector_breadth', desc='McClellan oscillator')
_register('sector_breadth_mcclellan_sum', 252, 'sector_breadth', desc='McClellan summation')
_register('w_sector_breadth_pct_above_ma10', 10, 'sector_breadth', 'W')
_register('w_sector_breadth_pct_above_ma40', 40, 'sector_breadth', 'W')
_register('w_sector_breadth_mcclellan_osc', 39, 'sector_breadth', 'W')

# Breadth motion features
_register('sector_breadth_ad_chg_10d', 262, 'sector_breadth', desc='10d change in A/D line')
_register('sector_breadth_ad_slope_20d', 272, 'sector_breadth', desc='20d slope of A/D line')
_register('sector_breadth_mcclellan_chg_5d', 44, 'sector_breadth', desc='5d change in McClellan')
_register('sector_breadth_mcclellan_slope_10d', 49, 'sector_breadth', desc='10d McClellan slope')
_register('sector_breadth_pct_ma50_chg_10d', 60, 'sector_breadth', desc='10d chg in % above MA50')
_register('w_sector_breadth_ad_slope_8w', 60, 'sector_breadth', 'W')
_register('w_sector_breadth_mcclellan_chg_2w', 41, 'sector_breadth', 'W')

# -----------------------------------------------------------------------------
# MACRO FEATURES - VIX/VXN (vix.py)
# -----------------------------------------------------------------------------
_register('vix_percentile_252d', 252, 'vix', desc='VIX percentile (1 year)')
_register('vix_zscore_60d', 60, 'vix', desc='VIX z-score (60d)')
_register('vix_ma20_ratio', 20, 'vix', desc='VIX / 20d MA ratio')
_register('vix_vxn_spread', 1, 'vix', desc='VIX - VXN spread')
_register('vix_change_5d', 5, 'vix', desc='5-day VIX change')
_register('vix_change_20d', 20, 'vix', desc='20-day VIX change')
_register('vix_regime', 252, 'vix', desc='VIX regime (0-2)')
_register('w_vix_vxn_spread', 1, 'vix', 'W')
_register('w_vix_percentile_52w', 52, 'vix', 'W')
_register('w_vix_zscore_12w', 12, 'vix', 'W')
_register('w_vix_regime', 52, 'vix', 'W')

# -----------------------------------------------------------------------------
# MACRO FEATURES - FRED (macro.py)
# Publication lags per series:
#   DGS10, DGS2, DGS3MO: 1 day
#   BAMLH0A0HYM2 (HY spread): 1 day
#   ICSA (initial claims): 5 days (week + Thu release)
#   CCSA (continued claims): 12 days (extra week lag)
#   T10Y2Y, T10Y3M (yield curves): 1 day
#   NFCI (financial conditions): 7 days (weekly, ~1 week lag)
# -----------------------------------------------------------------------------
# Treasury yields
_register('fred_dgs10', 0, 'macro', pub_lag=1, desc='10Y Treasury yield')
_register('fred_dgs10_chg5d', 5, 'macro', pub_lag=1, desc='10Y yield 5d change')
_register('fred_dgs10_chg20d', 20, 'macro', pub_lag=1, desc='10Y yield 20d change')
_register('fred_dgs10_z60', 60, 'macro', pub_lag=1, desc='10Y yield z-score (60d)')
_register('w_fred_dgs10_z60', 60, 'macro', 'W', pub_lag=1)

_register('fred_dgs2', 0, 'macro', pub_lag=1, desc='2Y Treasury yield')
_register('fred_dgs2_chg5d', 5, 'macro', pub_lag=1, desc='2Y yield 5d change')
_register('fred_dgs2_chg20d', 20, 'macro', pub_lag=1, desc='2Y yield 20d change')
_register('w_fred_dgs2_chg20d', 20, 'macro', 'W', pub_lag=1)

# Yield curve spreads
_register('fred_t10y2y', 0, 'macro', pub_lag=1, desc='10Y-2Y spread')
_register('fred_t10y2y_z60', 60, 'macro', pub_lag=1, desc='10Y-2Y spread z-score')
_register('w_fred_t10y2y_z60', 60, 'macro', 'W', pub_lag=1)
_register('fred_t10y3m', 0, 'macro', pub_lag=1, desc='10Y-3M spread')
_register('fred_t10y3m_z60', 60, 'macro', pub_lag=1, desc='10Y-3M spread z-score')

# Credit spreads (HY)
_register('fred_bamlh0a0hym2', 0, 'macro', pub_lag=1, desc='HY spread')
_register('fred_bamlh0a0hym2_chg20d', 20, 'macro', pub_lag=1, desc='HY spread 20d change')
_register('fred_bamlh0a0hym2_z60', 60, 'macro', pub_lag=1, desc='HY spread z-score')
_register('fred_bamlh0a0hym2_pct252', 252, 'macro', pub_lag=1, desc='HY spread percentile')
_register('w_fred_bamlh0a0hym2_z60', 60, 'macro', 'W', pub_lag=1)
_register('w_fred_bamlh0a0hym2_chg20d', 20, 'macro', 'W', pub_lag=1)

# Initial claims (ICSA) - 5 day publication lag
_register('fred_icsa', 0, 'macro', pub_lag=5, desc='Initial claims')
_register('fred_icsa_chg4w', 28, 'macro', pub_lag=5, desc='Initial claims 4w change')
_register('fred_icsa_z52w', 364, 'macro', pub_lag=5, desc='Initial claims z-score')
_register('w_fred_icsa_chg4w', 28, 'macro', 'W', pub_lag=5)
_register('w_fred_icsa_z52w', 364, 'macro', 'W', pub_lag=5)

# Continued claims (CCSA) - 12 day publication lag
_register('fred_ccsa', 0, 'macro', pub_lag=12, desc='Continued claims')
_register('fred_ccsa_chg4w', 28, 'macro', pub_lag=12, desc='Continued claims 4w change')
_register('fred_ccsa_z52w', 364, 'macro', pub_lag=12, desc='Continued claims z-score')
_register('w_fred_ccsa_z52w', 364, 'macro', 'W', pub_lag=12)

# Financial conditions (NFCI) - 7 day publication lag (weekly)
_register('fred_nfci', 0, 'macro', pub_lag=7, desc='Financial conditions index')
_register('fred_nfci_chg4w', 28, 'macro', pub_lag=7, desc='NFCI 4w change')
_register('fred_nfci_z52w', 364, 'macro', pub_lag=7, desc='NFCI z-score')
_register('w_fred_nfci_chg4w', 28, 'macro', 'W', pub_lag=7)

# -----------------------------------------------------------------------------
# INTERMARKET FEATURES (intermarket.py)
# -----------------------------------------------------------------------------
_register('copper_gold_ratio', 1, 'intermarket', desc='Copper/Gold ratio')
_register('copper_gold_zscore', 60, 'intermarket', desc='Copper/Gold z-score')
_register('gold_spy_ratio', 1, 'intermarket', desc='Gold/SPY ratio')
_register('gold_spy_ratio_zscore', 60, 'intermarket', desc='Gold/SPY ratio z-score')
_register('cyclical_defensive_ratio', 1, 'intermarket', desc='Cyclicals/Defensives ratio')
_register('w_cyclical_defensive_ratio', 1, 'intermarket', 'W')
_register('equity_bond_corr_60d', 60, 'intermarket', desc='Equity-bond correlation')
_register('w_equity_bond_corr_60d', 60, 'intermarket', 'W')
_register('credit_spread_zscore', 60, 'intermarket', desc='Credit spread z-score')
_register('yield_curve_zscore', 60, 'intermarket', desc='Yield curve z-score')
_register('w_credit_spread_zscore', 60, 'intermarket', 'W')
_register('w_yield_curve_zscore', 60, 'intermarket', 'W')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_provenance_for_features(feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get provenance metadata for a list of feature names.

    Args:
        feature_names: List of feature column names

    Returns:
        Dict mapping feature_name -> provenance dict
    """
    result = {}
    for name in feature_names:
        if name in FEATURE_PROVENANCE_REGISTRY:
            prov = FEATURE_PROVENANCE_REGISTRY[name]
            result[name] = prov.to_dict()
    return result


def get_missing_provenance(feature_names: List[str]) -> List[str]:
    """
    Get features that are not in the provenance registry.

    Args:
        feature_names: List of feature column names

    Returns:
        List of feature names missing from registry
    """
    return [name for name in feature_names if name not in FEATURE_PROVENANCE_REGISTRY]


def validate_provenance(
    provenance: Dict[str, Dict]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate provenance data for issues.

    Args:
        provenance: Dict from get_provenance_for_features()

    Returns:
        Tuple of (critical_violations, warnings, optimization_opportunities)
        - critical: Features using future data (must fix)
        - warnings: Zero lookback features to verify
        - optimizations: Features with potentially excessive lag that could be tightened
    """
    critical = []
    warnings = []
    optimizations = []

    for feat, prov in provenance.items():
        effective = prov.get('effective_lookback_days', 0)
        pub_lag = prov.get('publication_lag_days', 0)
        timeframe = prov.get('timeframe', 'D')

        # Critical: negative lookback means using future data
        if effective < 0:
            critical.append(f"{feat}: negative effective lookback ({effective} days) - USING FUTURE DATA")

        # Warning: zero lookback with no publication lag
        if effective == 0 and pub_lag == 0:
            warnings.append(f"{feat}: zero lookback - verify timing semantics")

        # Optimization: features with publication lag that might be reducible
        # These are buffers YOU added for data availability - the inherent lookback
        # (e.g., 200 days for MA-200) is NOT flagged since that's required by definition.
        # Only flag publication_lag > 0 as these are conservative buffers that could
        # potentially be tightened if the actual data availability is faster.
        if pub_lag > 0:
            optimizations.append(
                f"{feat}: publication_lag={pub_lag}d - verify if this buffer is still needed"
            )

    return critical, warnings, optimizations


def save_provenance_metadata(
    provenance: Dict[str, Dict],
    output_dir: Path,
    filename: str = 'feature_provenance.json'
) -> Path:
    """
    Save provenance metadata to JSON file.

    Args:
        provenance: Dict from get_provenance_for_features()
        output_dir: Directory to save to
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_path = output_dir / filename

    # Add metadata
    output_data = {
        '_metadata': {
            'description': 'Feature date provenance tracking for leakage prevention',
            'semantics': {
                'lookback_days': 'Trading days of historical data used',
                'publication_lag_days': 'Days between data date and availability',
                'effective_lookback_days': 'Total lookback in trading days',
                'timeframe': 'D=daily, W=weekly',
            },
            'trading_assumption': 'Feature at row t uses data through t, entry at t+1',
        },
        'features': provenance,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved provenance for {len(provenance)} features to {output_path}")
    return output_path


def load_provenance_metadata(provenance_path: Path) -> Dict[str, Dict]:
    """
    Load provenance metadata from JSON file.

    Args:
        provenance_path: Path to provenance JSON file

    Returns:
        Dict mapping feature_name -> provenance dict
    """
    with open(provenance_path) as f:
        data = json.load(f)

    # Handle both old format (flat dict) and new format (with _metadata)
    if 'features' in data:
        return data['features']
    return data


def get_max_lookback(feature_names: List[str]) -> int:
    """
    Get the maximum lookback across a list of features.

    Args:
        feature_names: List of feature names

    Returns:
        Maximum effective lookback in trading days
    """
    max_lookback = 0
    for name in feature_names:
        if name in FEATURE_PROVENANCE_REGISTRY:
            prov = FEATURE_PROVENANCE_REGISTRY[name]
            max_lookback = max(max_lookback, prov.effective_lookback_days())
    return max_lookback


def report_provenance_summary(provenance: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Generate summary statistics for provenance data.

    Args:
        provenance: Dict from get_provenance_for_features()

    Returns:
        Dict with summary statistics
    """
    if not provenance:
        return {'error': 'No provenance data'}

    lookbacks = [p['effective_lookback_days'] for p in provenance.values()]

    return {
        'total_features': len(provenance),
        'max_lookback_days': max(lookbacks),
        'min_lookback_days': min(lookbacks),
        'mean_lookback_days': sum(lookbacks) / len(lookbacks),
        'weekly_features': sum(1 for p in provenance.values() if p['timeframe'] == 'W'),
        'daily_features': sum(1 for p in provenance.values() if p['timeframe'] == 'D'),
        'with_publication_lag': sum(1 for p in provenance.values() if p.get('publication_lag_days', 0) > 0),
    }
