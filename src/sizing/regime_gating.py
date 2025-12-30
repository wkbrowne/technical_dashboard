"""Direction-aware regime-based exposure gating.

Implements a gating stage that adjusts exposure based on market conditions.
This is a risk-control overlay, not an alpha signal.

Key design principles:
1. Direction-aware: Separate multipliers for longs vs shorts
2. Multiplicative: Rules combine multiplicatively
3. Auditable: Full justification for each decision
4. Strict mode: Can fail loudly on missing features

Gating is applied AFTER raw weights are computed but BEFORE final
portfolio constraints.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .config import (
    ModelType,
    RegimeGatingConfig,
    REGIME_FEATURE_COLS,
    get_tpe_param_range,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeMultipliers:
    """Direction-aware regime multipliers with full justification.

    All multipliers are in [0, 1] where 1.0 means no reduction.

    Attributes:
        vix_mult: Multiplier from VIX rule (affects both directions).
        credit_mult: Multiplier from credit spread rule (affects both directions).
        breadth_long_mult: Multiplier for longs from breadth rule.
        breadth_short_mult: Multiplier for shorts from breadth rule.
        short_regime_mult: Additional multiplier for shorts only.
        final_long_mult: Combined multiplier for long positions.
        final_short_mult: Combined multiplier for short positions.
    """
    vix_mult: float = 1.0
    credit_mult: float = 1.0
    breadth_long_mult: float = 1.0
    breadth_short_mult: float = 1.0
    short_regime_mult: float = 1.0

    # These are computed from the above
    final_long_mult: float = field(init=False)
    final_short_mult: float = field(init=False)

    def __post_init__(self):
        """Compute final multipliers."""
        # Longs: vix * credit * breadth_long
        self.final_long_mult = self.vix_mult * self.credit_mult * self.breadth_long_mult
        # Shorts: vix * credit * breadth_short * short_regime
        self.final_short_mult = (
            self.vix_mult * self.credit_mult *
            self.breadth_short_mult * self.short_regime_mult
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for justification output."""
        return {
            "vix_mult": self.vix_mult,
            "credit_mult": self.credit_mult,
            "breadth_long_mult": self.breadth_long_mult,
            "breadth_short_mult": self.breadth_short_mult,
            "short_regime_mult": self.short_regime_mult,
            "final_long_mult": self.final_long_mult,
            "final_short_mult": self.final_short_mult,
        }


@dataclass
class GatingJustification:
    """Full justification for regime gating decision.

    Used for dashboard display and audit trail.
    """
    # Feature values used
    vix_percentile: Optional[float] = None
    credit_zscore: Optional[float] = None
    breadth_percentile: Optional[float] = None

    # Which rules triggered
    vix_triggered: bool = False
    credit_triggered: bool = False
    breadth_triggered: bool = False

    # Multipliers applied
    multipliers: RegimeMultipliers = field(default_factory=RegimeMultipliers)

    # Missing features
    missing_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for justification output."""
        return {
            "vix_percentile": self.vix_percentile,
            "credit_zscore": self.credit_zscore,
            "breadth_percentile": self.breadth_percentile,
            "vix_triggered": self.vix_triggered,
            "credit_triggered": self.credit_triggered,
            "breadth_triggered": self.breadth_triggered,
            "missing_features": self.missing_features,
            **self.multipliers.to_dict(),
        }

    @property
    def any_triggered(self) -> bool:
        """Check if any rule was triggered."""
        return self.vix_triggered or self.credit_triggered or self.breadth_triggered


def find_feature_column(
    data: Union[pd.Series, pd.DataFrame],
    candidates: List[str],
) -> Tuple[Optional[str], Optional[float]]:
    """Find first available feature column and its value.

    Args:
        data: Row data (Series or single-row DataFrame).
        candidates: List of candidate column names.

    Returns:
        Tuple of (column_name, value) or (None, None) if not found.
    """
    if isinstance(data, pd.DataFrame):
        if len(data) == 0:
            return None, None
        data = data.iloc[0]

    for col in candidates:
        if col in data.index:
            val = data[col]
            if not pd.isna(val):
                return col, float(val)

    return None, None


def compute_regime_multipliers(
    row: Union[pd.Series, pd.DataFrame],
    config: RegimeGatingConfig,
    strict: Optional[bool] = None,
) -> Tuple[RegimeMultipliers, GatingJustification]:
    """Compute direction-aware regime multipliers from feature values.

    This is the core gating function that returns both the multipliers
    and full justification for the decision.

    Args:
        row: Row data with regime feature values.
        config: Regime gating configuration.
        strict: Override config's strict_missing_features setting.

    Returns:
        Tuple of (RegimeMultipliers, GatingJustification).

    Raises:
        ValueError: If strict=True and required features are missing.
    """
    strict = strict if strict is not None else config.strict_missing_features
    justification = GatingJustification()

    if not config.enabled:
        return RegimeMultipliers(), justification

    # Convert DataFrame to Series if needed
    if isinstance(row, pd.DataFrame):
        if len(row) == 0:
            return RegimeMultipliers(), justification
        row = row.iloc[0]

    # Initialize multipliers
    vix_mult = 1.0
    credit_mult = 1.0
    breadth_long_mult = 1.0
    breadth_short_mult = 1.0

    # Rule 1: VIX high
    vix_col, vix_val = find_feature_column(row, REGIME_FEATURE_COLS["vix_percentile"])
    if vix_val is not None:
        justification.vix_percentile = vix_val
        if vix_val > config.vix_high_threshold:
            vix_mult = config.vix_high_exposure_mult
            justification.vix_triggered = True
    else:
        justification.missing_features.append("vix_percentile")
        if strict:
            raise ValueError(f"Missing required regime feature: vix_percentile")

    # Rule 2: Credit spread high
    credit_col, credit_val = find_feature_column(row, REGIME_FEATURE_COLS["credit_spread"])
    if credit_val is not None:
        justification.credit_zscore = credit_val
        if credit_val > config.credit_risk_threshold:
            credit_mult = config.credit_risk_exposure_mult
            justification.credit_triggered = True
    else:
        justification.missing_features.append("credit_spread")
        if strict:
            raise ValueError(f"Missing required regime feature: credit_spread")

    # Rule 3: Breadth poor (direction-aware)
    breadth_col, breadth_val = find_feature_column(row, REGIME_FEATURE_COLS["breadth"])
    if breadth_val is not None:
        justification.breadth_percentile = breadth_val
        if breadth_val < config.breadth_poor_threshold:
            breadth_long_mult = config.breadth_poor_long_mult
            breadth_short_mult = config.breadth_poor_short_mult
            justification.breadth_triggered = True
    else:
        justification.missing_features.append("breadth")
        if strict:
            raise ValueError(f"Missing required regime feature: breadth")

    # Warn about missing features (if not strict)
    if justification.missing_features and not strict:
        logger.debug(
            f"Missing regime features (using mult=1.0): {justification.missing_features}"
        )

    # Create multipliers
    multipliers = RegimeMultipliers(
        vix_mult=vix_mult,
        credit_mult=credit_mult,
        breadth_long_mult=breadth_long_mult,
        breadth_short_mult=breadth_short_mult,
        short_regime_mult=config.short_regime_mult,
    )
    justification.multipliers = multipliers

    return multipliers, justification


def apply_regime_gating(
    signals: pd.DataFrame,
    regime_features: Union[pd.Series, pd.DataFrame],
    config: RegimeGatingConfig,
) -> pd.DataFrame:
    """Apply direction-aware regime gating to signals.

    Applies separate multipliers to long and short positions.

    Args:
        signals: DataFrame with combined_weight column.
        regime_features: Regime features for gating.
        config: Gating configuration.

    Returns:
        DataFrame with gated weights and justification columns.
    """
    result = signals.copy()

    if not config.enabled:
        result["gating_long_mult"] = 1.0
        result["gating_short_mult"] = 1.0
        result["gating_triggered"] = False
        return result

    # Compute multipliers
    multipliers, justification = compute_regime_multipliers(regime_features, config)

    # Apply direction-aware multipliers
    if "combined_weight" in result.columns:
        weights = result["combined_weight"].values.copy()

        # Apply long multiplier to positive weights
        long_mask = weights > 0
        weights[long_mask] *= multipliers.final_long_mult

        # Apply short multiplier to negative weights
        short_mask = weights < 0
        weights[short_mask] *= multipliers.final_short_mult

        result["combined_weight"] = weights

    # Add justification columns
    result["gating_long_mult"] = multipliers.final_long_mult
    result["gating_short_mult"] = multipliers.final_short_mult
    result["gating_triggered"] = justification.any_triggered
    result["gating_vix_triggered"] = justification.vix_triggered
    result["gating_credit_triggered"] = justification.credit_triggered
    result["gating_breadth_triggered"] = justification.breadth_triggered

    # Add feature values used
    if justification.vix_percentile is not None:
        result["regime_vix_percentile"] = justification.vix_percentile
    if justification.credit_zscore is not None:
        result["regime_credit_zscore"] = justification.credit_zscore
    if justification.breadth_percentile is not None:
        result["regime_breadth_percentile"] = justification.breadth_percentile

    return result


def apply_regime_gating_vectorized(
    signals: pd.DataFrame,
    config: RegimeGatingConfig,
) -> pd.DataFrame:
    """Apply regime gating when regime features are already in signals DataFrame.

    More efficient for large DataFrames where regime features are already joined.

    Args:
        signals: DataFrame with combined_weight and regime feature columns.
        config: Gating configuration.

    Returns:
        DataFrame with gated weights.
    """
    result = signals.copy()

    if not config.enabled:
        result["gating_long_mult"] = 1.0
        result["gating_short_mult"] = 1.0
        result["gating_triggered"] = False
        return result

    n_rows = len(result)

    # Initialize multiplier arrays
    vix_mult = np.ones(n_rows)
    credit_mult = np.ones(n_rows)
    breadth_long_mult = np.ones(n_rows)
    breadth_short_mult = np.ones(n_rows)

    # VIX rule
    vix_col, _ = find_feature_column(result.iloc[0:1], REGIME_FEATURE_COLS["vix_percentile"])
    if vix_col and vix_col in result.columns:
        vix_triggered = result[vix_col] > config.vix_high_threshold
        vix_mult[vix_triggered] = config.vix_high_exposure_mult
        result["gating_vix_triggered"] = vix_triggered
    else:
        result["gating_vix_triggered"] = False

    # Credit rule
    credit_col, _ = find_feature_column(result.iloc[0:1], REGIME_FEATURE_COLS["credit_spread"])
    if credit_col and credit_col in result.columns:
        credit_triggered = result[credit_col] > config.credit_risk_threshold
        credit_mult[credit_triggered] = config.credit_risk_exposure_mult
        result["gating_credit_triggered"] = credit_triggered
    else:
        result["gating_credit_triggered"] = False

    # Breadth rule
    breadth_col, _ = find_feature_column(result.iloc[0:1], REGIME_FEATURE_COLS["breadth"])
    if breadth_col and breadth_col in result.columns:
        breadth_triggered = result[breadth_col] < config.breadth_poor_threshold
        breadth_long_mult[breadth_triggered] = config.breadth_poor_long_mult
        breadth_short_mult[breadth_triggered] = config.breadth_poor_short_mult
        result["gating_breadth_triggered"] = breadth_triggered
    else:
        result["gating_breadth_triggered"] = False

    # Compute final multipliers
    final_long_mult = vix_mult * credit_mult * breadth_long_mult
    final_short_mult = vix_mult * credit_mult * breadth_short_mult * config.short_regime_mult

    # Apply to weights
    if "combined_weight" in result.columns:
        weights = result["combined_weight"].values.copy()
        long_mask = weights > 0
        short_mask = weights < 0
        weights[long_mask] *= final_long_mult[long_mask]
        weights[short_mask] *= final_short_mult[short_mask]
        result["combined_weight"] = weights

    result["gating_long_mult"] = final_long_mult
    result["gating_short_mult"] = final_short_mult
    result["gating_triggered"] = (
        result["gating_vix_triggered"] |
        result["gating_credit_triggered"] |
        result["gating_breadth_triggered"]
    )

    return result


# TPE optimization helpers

def suggest_gating_params(trial) -> Dict:
    """Suggest gating parameters for TPE optimization.

    Args:
        trial: Optuna trial object.

    Returns:
        Dict of suggested parameters.
    """
    params = {
        "vix_high_threshold": trial.suggest_float(
            "vix_high_threshold",
            *get_tpe_param_range("vix_high_threshold"),
        ),
        "vix_high_exposure_mult": trial.suggest_float(
            "vix_high_exposure_mult",
            *get_tpe_param_range("vix_high_exposure_mult"),
        ),
        "credit_risk_threshold": trial.suggest_float(
            "credit_risk_threshold",
            *get_tpe_param_range("credit_risk_threshold"),
        ),
        "credit_risk_exposure_mult": trial.suggest_float(
            "credit_risk_exposure_mult",
            *get_tpe_param_range("credit_risk_exposure_mult"),
        ),
        "breadth_poor_threshold": trial.suggest_float(
            "breadth_poor_threshold",
            *get_tpe_param_range("breadth_poor_threshold"),
        ),
        "breadth_poor_long_mult": trial.suggest_float(
            "breadth_poor_long_mult",
            *get_tpe_param_range("breadth_poor_long_mult"),
        ),
        "breadth_poor_short_mult": trial.suggest_float(
            "breadth_poor_short_mult",
            *get_tpe_param_range("breadth_poor_short_mult"),
        ),
        "short_regime_mult": trial.suggest_float(
            "short_regime_mult",
            *get_tpe_param_range("short_regime_mult"),
        ),
    }

    return params


def suggest_short_selectivity_params(trial) -> Dict:
    """Suggest short selectivity parameters for TPE optimization.

    Args:
        trial: Optuna trial object.

    Returns:
        Dict of suggested parameters.
    """
    params = {
        "short_threshold_offset": trial.suggest_float(
            "short_threshold_offset",
            *get_tpe_param_range("short_threshold_offset"),
        ),
        "short_max_weight_mult": trial.suggest_float(
            "short_max_weight_mult",
            *get_tpe_param_range("short_max_weight_mult"),
        ),
        "short_exposure_mult": trial.suggest_float(
            "short_exposure_mult",
            *get_tpe_param_range("short_exposure_mult"),
        ),
    }

    return params


def create_gating_config_from_params(
    params: Dict,
    enabled: bool = True,
    strict_missing_features: bool = False,
) -> RegimeGatingConfig:
    """Create gating config from optimization parameters.

    Args:
        params: Dict of parameter values.
        enabled: Whether gating is enabled.
        strict_missing_features: Whether to fail on missing features.

    Returns:
        RegimeGatingConfig object.
    """
    return RegimeGatingConfig(
        enabled=enabled,
        strict_missing_features=strict_missing_features,
        vix_high_threshold=params.get("vix_high_threshold", 80.0),
        vix_high_exposure_mult=params.get("vix_high_exposure_mult", 0.7),
        credit_risk_threshold=params.get("credit_risk_threshold", 1.5),
        credit_risk_exposure_mult=params.get("credit_risk_exposure_mult", 0.8),
        breadth_poor_threshold=params.get("breadth_poor_threshold", 30.0),
        breadth_poor_long_mult=params.get("breadth_poor_long_mult", 0.8),
        breadth_poor_short_mult=params.get("breadth_poor_short_mult", 1.0),
        short_regime_mult=params.get("short_regime_mult", 1.0),
    )


def get_gating_diagnostics(
    signals: pd.DataFrame,
) -> Dict:
    """Get summary diagnostics from gated signals.

    Args:
        signals: DataFrame with gating columns.

    Returns:
        Dict with gating statistics.
    """
    diag = {}

    if "gating_triggered" in signals.columns:
        n_total = len(signals)
        n_triggered = signals["gating_triggered"].sum()
        diag["n_total"] = n_total
        diag["n_gating_triggered"] = int(n_triggered)
        diag["pct_gating_triggered"] = 100 * n_triggered / n_total if n_total > 0 else 0

    if "gating_vix_triggered" in signals.columns:
        diag["n_vix_triggered"] = int(signals["gating_vix_triggered"].sum())

    if "gating_credit_triggered" in signals.columns:
        diag["n_credit_triggered"] = int(signals["gating_credit_triggered"].sum())

    if "gating_breadth_triggered" in signals.columns:
        diag["n_breadth_triggered"] = int(signals["gating_breadth_triggered"].sum())

    if "gating_long_mult" in signals.columns:
        diag["avg_long_mult"] = signals["gating_long_mult"].mean()

    if "gating_short_mult" in signals.columns:
        diag["avg_short_mult"] = signals["gating_short_mult"].mean()

    return diag
