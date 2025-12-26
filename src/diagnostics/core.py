"""
Core diagnostic types and base classes.

Defines the result structures and severity levels for diagnostic checks.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy scalar, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


class Severity(str, Enum):
    """Severity level for diagnostic flags."""
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value


@dataclass
class DiagnosticFlag:
    """
    A single diagnostic finding with actionable information.

    Attributes:
        severity: INFO, WARN, or CRITICAL
        check_name: Short identifier for the check
        symptom: What was observed
        why_it_matters: Explanation of the impact
        suggested_fix: Concrete change to address the issue
        evidence: Key numbers, fold IDs, dates, etc.
    """
    severity: Severity
    check_name: str
    symptom: str
    why_it_matters: str
    suggested_fix: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'severity': str(self.severity),
            'check_name': self.check_name,
            'symptom': self.symptom,
            'why_it_matters': self.why_it_matters,
            'suggested_fix': self.suggested_fix,
            'evidence': convert_to_json_serializable(self.evidence),
        }


@dataclass
class DiagnosticResult:
    """
    Complete diagnostic result for a model.

    Attributes:
        model_key: The model being diagnosed
        timestamp: When the diagnostic was run
        flags: List of diagnostic findings
        metrics_summary: Summary of key metrics
        passed: Whether all checks passed (no CRITICAL flags)
    """
    model_key: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    flags: List[DiagnosticFlag] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if no CRITICAL flags."""
        return not any(f.severity == Severity.CRITICAL for f in self.flags)

    @property
    def n_critical(self) -> int:
        """Count of CRITICAL flags."""
        return sum(1 for f in self.flags if f.severity == Severity.CRITICAL)

    @property
    def n_warn(self) -> int:
        """Count of WARN flags."""
        return sum(1 for f in self.flags if f.severity == Severity.WARN)

    @property
    def n_info(self) -> int:
        """Count of INFO flags."""
        return sum(1 for f in self.flags if f.severity == Severity.INFO)

    def add_flag(self, flag: DiagnosticFlag) -> None:
        """Add a diagnostic flag."""
        self.flags.append(flag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_key': self.model_key,
            'timestamp': self.timestamp,
            'passed': self.passed,
            'summary': {
                'n_critical': self.n_critical,
                'n_warn': self.n_warn,
                'n_info': self.n_info,
                'total_flags': len(self.flags),
            },
            'metrics_summary': convert_to_json_serializable(self.metrics_summary),
            'flags': [f.to_dict() for f in self.flags],
        }


# =============================================================================
# THRESHOLDS - Centralized configuration for diagnostic thresholds
# =============================================================================

@dataclass
class DiagnosticThresholds:
    """Configurable thresholds for diagnostic checks."""

    # Data quality
    nan_rate_warn: float = 0.01  # 1%
    nan_rate_critical: float = 0.05  # 5%
    constant_variance_threshold: float = 1e-8
    duplicate_correlation_threshold: float = 0.9999

    # Leakage detection
    suspiciously_high_auc: float = 0.85  # AUC above this suggests leakage

    # CV stability
    auc_cv_warn: float = 0.15  # Coefficient of variation
    auc_cv_critical: float = 0.25
    performance_degradation_threshold: float = 0.05  # Drop from first to last fold
    min_train_samples_warn: int = 2000

    # Calibration
    brier_baseline_multiplier_warn: float = 1.1  # Brier worse than baseline * this
    extreme_pred_warn: float = 0.10  # >10% of predictions outside [0.05, 0.95]
    precision_at_k_min_lift: float = 1.5  # P@5% should be at least 1.5x base rate

    # Positive rate drift
    positive_rate_drift_ratio: float = 1.5  # Max/min ratio across folds

    # Weighting
    weight_top_1pct_share_warn: float = 0.10  # Top 1% has >10% of weight
    ess_ratio_warn: float = 0.50  # ESS < 50% of n_samples
    weight_metric_sensitivity: float = 0.03  # AUC diff with/without weights

    # Hyperopt
    pruning_rate_warn: float = 0.50  # >50% trials pruned
    degenerate_num_leaves_threshold: int = 128
    degenerate_min_child_threshold: int = 100
    learning_rate_low: float = 0.01
    learning_rate_high: float = 0.12

    # Feature importance
    top_5_importance_concentration_warn: float = 0.50  # Top 5 features > 50% importance


DEFAULT_THRESHOLDS = DiagnosticThresholds()


# =============================================================================
# FEATURE FAMILIES - For story consistency checks
# =============================================================================

FEATURE_FAMILIES = {
    'alpha_relative_strength': [
        'alpha_', 'rel_strength', 'xsec_mom', 'xsec_pct',
        'w_alpha_', 'w_rel_strength', 'w_xsec_',
    ],
    'macro_intermarket': [
        'fred_', 'vix_', 'copper_gold', 'gold_spy', 'qqq_spy',
        'rsp_spy', 'bestmatch_spy', 'cyclical_defensive',
        'equity_bond_corr', 'credit_spread', 'yield_curve',
        'w_fred_', 'w_vix_', 'w_copper_', 'w_gold_',
    ],
    'volatility_regime': [
        'vol_regime', 'rv_z', 'atr_percent', 'bb_width', 'squeeze',
        'w_rv_', 'w_vol_regime',
    ],
    'breadth': [
        'sector_breadth', 'ad_ratio', 'mcclellan', 'pct_universe',
        'w_sector_breadth', 'w_ad_ratio', 'w_mcclellan',
    ],
    'trend': [
        'pct_slope', 'macd', 'trend_score', 'adx', 'di_plus', 'di_minus',
        'chop', 'w_pct_slope', 'w_macd', 'w_trend',
    ],
    'price_position': [
        'pct_dist_ma', 'pos_in_', 'relative_dist', 'min_pct_dist',
        'w_pct_dist', 'w_pos_in', 'w_relative_dist',
    ],
    'gaps_overnight': [
        'gap_', 'overnight', 'range_efficiency',
    ],
    'volume_liquidity': [
        'vwap_', 'volshock', 'pv_divergence', 'obv_', 'rdollar',
        'amihud', 'illiquidity', 'w_vwap', 'w_volshock',
    ],
    'drawdown_recovery': [
        'drawdown', 'recovery', 'days_since_high', 'hl_range_position',
        'w_drawdown', 'w_recovery', 'w_days_since',
    ],
    'momentum_oscillator': [
        'rsi_', 'w_rsi_',
    ],
    'candle_patterns': [
        'upper_shadow', 'lower_shadow',
    ],
}

# Expected feature families by model type for story consistency
EXPECTED_STORY_EMPHASIS = {
    'long_normal': ['alpha_relative_strength', 'trend', 'breadth', 'momentum_oscillator'],
    'long_parabolic': ['volatility_regime', 'drawdown_recovery', 'trend', 'alpha_relative_strength'],
    'short_normal': ['breadth', 'volatility_regime', 'alpha_relative_strength', 'candle_patterns'],
    'short_parabolic': ['volatility_regime', 'drawdown_recovery', 'macro_intermarket', 'breadth'],
}


def classify_feature_family(feature_name: str) -> Optional[str]:
    """Classify a feature into its family based on name patterns."""
    feature_lower = feature_name.lower()
    for family, patterns in FEATURE_FAMILIES.items():
        for pattern in patterns:
            if pattern.lower() in feature_lower:
                return family
    return None
