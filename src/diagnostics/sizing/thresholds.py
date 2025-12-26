"""Configurable thresholds for sizing diagnostics.

Thresholds define when warnings are triggered and at what severity level.
All thresholds can be customized via config file or constructor.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class SizingDiagnosticThresholds:
    """Configurable thresholds for sizing diagnostic warnings.

    All thresholds define when to trigger INFO, WARN, or ERROR.
    Naming convention: <metric>_<severity> e.g., missing_preds_warn, missing_preds_error.

    Attributes are grouped by diagnostic category.
    """

    # =========================================================================
    # Data Integrity
    # =========================================================================

    # Missing predictions (% of expected predictions absent)
    missing_preds_warn: float = 0.01  # 1%
    missing_preds_error: float = 0.05  # 5%

    # NaN predictions (% of predictions that are NaN)
    nan_preds_warn: float = 0.005  # 0.5%
    nan_preds_error: float = 0.02  # 2%

    # Stale predictions (days since prediction timestamp)
    stale_preds_warn_days: int = 3
    stale_preds_error_days: int = 7

    # Missing weeks threshold (consecutive)
    missing_weeks_warn: int = 1
    missing_weeks_error: int = 2

    # =========================================================================
    # Signal Distribution / Drift
    # =========================================================================

    # PSI (Population Stability Index) vs baseline
    psi_warn: float = 0.10
    psi_error: float = 0.25

    # KS statistic vs baseline
    ks_warn: float = 0.10
    ks_error: float = 0.20

    # Extreme signal rate (fraction with p > extreme_high or p < extreme_low)
    extreme_signal_high: float = 0.90
    extreme_signal_low: float = 0.10
    extreme_signal_rate_warn: float = 0.10  # 10% extreme is warning
    extreme_signal_rate_spike_warn: float = 2.0  # 2x baseline is warning

    # Rank correlation vs previous week
    rank_correlation_warn: float = 0.50  # Spearman < 0.5 is warning
    rank_correlation_error: float = 0.30  # < 0.3 is error

    # =========================================================================
    # Portfolio Construction
    # =========================================================================

    # Gross exposure (vs configured max)
    gross_exposure_tolerance: float = 0.001  # Epsilon for exceeding max
    # Note: Exceeding max_gross is always ERROR

    # Max weight (vs configured max)
    max_weight_tolerance: float = 0.001  # Epsilon for exceeding cap
    # Note: Exceeding max_weight is always ERROR

    # Concentration (HHI)
    hhi_warn: float = 0.10  # HHI > 0.10 is moderately concentrated
    hhi_error: float = 0.25  # HHI > 0.25 is highly concentrated

    # Top5 exposure share
    top5_concentration_warn: float = 0.50  # Top 5 > 50% of gross is warning
    top5_concentration_error: float = 0.70  # > 70% is error

    # Turnover
    turnover_warn: float = 0.50  # 50% turnover is warning
    turnover_error: float = 1.00  # 100% turnover is error
    turnover_spike_ratio: float = 2.0  # 2x recent avg is spike

    # Position counts
    min_positions_warn: int = 3
    max_positions_warn: int = 100  # Optional upper bound

    # Long/short balance
    short_dominance_warn: float = 0.70  # Shorts > 70% of gross is unusual

    # Cash fraction (unexpectedly high/low)
    cash_low_warn: float = 0.10  # Less than 10% cash unexpected
    cash_high_warn: float = 0.80  # More than 80% cash unexpected

    # =========================================================================
    # Gating
    # =========================================================================

    # Consecutive weeks at maximum reduction
    gating_max_consecutive_warn: int = 3  # 3+ weeks at max gating

    # =========================================================================
    # Execution Risk
    # =========================================================================

    # Liquidity (rdollar_vol in millions)
    liquidity_threshold_millions: float = 1.0  # $1M daily volume
    pct_below_liquidity_warn: float = 0.10  # >10% below threshold is warn
    pct_below_liquidity_error: float = 0.30  # >30% is error

    # Gap risk (gap_atr_ratio)
    gap_atr_threshold: float = 1.0  # Gap > 1x ATR is risky
    pct_above_gap_warn: float = 0.20  # >20% above threshold is warn
    pct_above_gap_error: float = 0.40  # >40% is error

    # Slippage budget (bps)
    slippage_budget_bps: float = 20.0  # 20 bps budget
    slippage_utilization_warn: float = 0.80  # >80% utilization is warn
    slippage_utilization_error: float = 1.00  # >100% is error

    # =========================================================================
    # Backtest Outcomes
    # =========================================================================

    # Tail week detection
    tail_week_percentile: float = 5.0  # Bottom 5th percentile
    tail_week_absolute: float = -0.05  # -5% absolute threshold

    # Rolling Sharpe deterioration
    sharpe_warn: float = 0.0  # Sharpe below 0 is warn
    sharpe_error: float = -1.0  # Sharpe below -1 is error

    # Drawdown
    drawdown_warn: float = -0.15  # -15% drawdown is warn
    drawdown_error: float = -0.25  # -25% is error

    # =========================================================================
    # Behavioral Risk (Compound Conditions)
    # =========================================================================

    # High exposure + high volatility regime
    high_exposure_threshold: float = 0.80  # >80% gross exposure
    high_vol_vix_percentile: float = 80.0  # >80th percentile VIX

    # High concentration + high turnover
    concentration_turnover_warn_hhi: float = 0.08
    concentration_turnover_warn_turnover: float = 0.40

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SizingDiagnosticThresholds":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "SizingDiagnosticThresholds":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def merge(self, overrides: Dict[str, Any]) -> "SizingDiagnosticThresholds":
        """Create new thresholds with overrides applied."""
        base = self.to_dict()
        base.update({k: v for k, v in overrides.items() if k in base})
        return SizingDiagnosticThresholds.from_dict(base)


# Default thresholds instance
DEFAULT_SIZING_THRESHOLDS = SizingDiagnosticThresholds()


def load_thresholds(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SizingDiagnosticThresholds:
    """Load thresholds from file and/or apply overrides.

    Args:
        path: Optional path to JSON config file.
        overrides: Optional dict of overrides to apply.

    Returns:
        SizingDiagnosticThresholds instance.
    """
    if path is not None:
        thresholds = SizingDiagnosticThresholds.from_json(path)
    else:
        thresholds = SizingDiagnosticThresholds()

    if overrides:
        thresholds = thresholds.merge(overrides)

    return thresholds
