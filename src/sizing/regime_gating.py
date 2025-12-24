"""Regime-based exposure gating.

Implements a gating stage that adjusts exposure based on market conditions.
This is a risk-control overlay, not an alpha signal.

Why gating even if regime is in the model?
- Models include regime features for alpha generation (predicting returns)
- Gating provides hard risk-control limits independent of alpha
- Separation of concerns: models predict, gating controls risk
- Gating can force exposure to zero even when model sees opportunity
- Gating parameters can be tuned with TPE for risk-adjusted returns

Gating is applied AFTER raw weights are computed but BEFORE final
portfolio normalization.

Available regime features (examples from feature pipeline):
- vix_percentile_252d: VIX relative to 252-day history
- vix_zscore_60d: VIX z-score over 60 days
- fred_bamlh0a0hym2_z60: Credit spread z-score
- sector_breadth_pct_above_ma200: Breadth indicator
- w_equity_bond_corr_60d: Stock-bond correlation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from .config import ModelType, RegimeGatingConfig


@dataclass
class GatingResult:
    """Result of regime gating computation.

    Attributes:
        exposure_multiplier: Final exposure multiplier in [0, 1].
        allowed_models: Set of models allowed in current regime.
        triggered_rules: List of rules that triggered gating.
        rule_multipliers: Multiplier from each triggered rule.
        regime_features: Values of regime features used.
    """
    exposure_multiplier: float
    allowed_models: Set[ModelType]
    triggered_rules: List[str]
    rule_multipliers: Dict[str, float]
    regime_features: Dict[str, float]


class RegimeGate:
    """Regime-based gating rules.

    Implements simple rule-based gating that can be turned off by
    setting thresholds wide and multipliers to 1.

    The gate computes an exposure multiplier in [0, 1] that scales
    the raw weights before portfolio constraints are applied.
    """

    def __init__(self, config: RegimeGatingConfig):
        """Initialize gate with configuration.

        Args:
            config: RegimeGatingConfig with thresholds and multipliers.
        """
        self.config = config

    def compute_multiplier(
        self,
        regime_features: pd.Series,
    ) -> GatingResult:
        """Compute exposure multiplier from regime features.

        Applies rules in order, multiplicatively combining multipliers.

        Args:
            regime_features: Series with regime feature values.

        Returns:
            GatingResult with multiplier and diagnostics.
        """
        if not self.config.enabled:
            return GatingResult(
                exposure_multiplier=1.0,
                allowed_models=set(ModelType),
                triggered_rules=[],
                rule_multipliers={},
                regime_features=dict(regime_features) if not regime_features.empty else {},
            )

        multiplier = 1.0
        triggered_rules = []
        rule_multipliers = {}
        features_used = {}

        # Rule 1: VIX high
        vix_col = self._find_feature(regime_features, [
            "vix_percentile_252d",
            "d_vix_percentile_252d",
            "vix_pct_252",
        ])
        if vix_col:
            vix_pct = regime_features[vix_col]
            features_used["vix_percentile"] = vix_pct
            if vix_pct > self.config.vix_high_threshold:
                mult = self.config.vix_high_exposure_mult
                multiplier *= mult
                triggered_rules.append("vix_high")
                rule_multipliers["vix_high"] = mult

        # Rule 2: Credit spread high
        credit_col = self._find_feature(regime_features, [
            "fred_bamlh0a0hym2_z60",
            "d_fred_bamlh0a0hym2_z60",
            "credit_spread_zscore",
            "hy_spread_z60",
        ])
        if credit_col:
            credit_z = regime_features[credit_col]
            features_used["credit_zscore"] = credit_z
            if credit_z > self.config.credit_risk_threshold:
                mult = self.config.credit_risk_exposure_mult
                multiplier *= mult
                triggered_rules.append("credit_risk")
                rule_multipliers["credit_risk"] = mult

        # Rule 3: Breadth poor (applies to longs only via allowed_models)
        breadth_col = self._find_feature(regime_features, [
            "sector_breadth_pct_above_ma200",
            "d_sector_breadth_pct_above_ma200",
            "breadth_pct_above_ma200",
            "pct_above_ma200",
        ])
        breadth_poor = False
        if breadth_col:
            breadth_pct = regime_features[breadth_col]
            features_used["breadth_pct"] = breadth_pct
            if breadth_pct < self.config.breadth_poor_threshold:
                breadth_poor = True
                mult = self.config.breadth_poor_long_mult
                # Note: This multiplier is applied to longs specifically
                triggered_rules.append("breadth_poor")
                rule_multipliers["breadth_poor_long"] = mult

        # Determine allowed models
        allowed_models = set(ModelType)
        if self.config.allowed_models is not None:
            allowed_models = set(
                ModelType.from_string(m) for m in self.config.allowed_models
            )

        # Clip multiplier to [0, 1]
        multiplier = np.clip(multiplier, 0.0, 1.0)

        return GatingResult(
            exposure_multiplier=multiplier,
            allowed_models=allowed_models,
            triggered_rules=triggered_rules,
            rule_multipliers=rule_multipliers,
            regime_features=features_used,
        )

    def _find_feature(
        self,
        features: pd.Series,
        candidates: List[str],
    ) -> Optional[str]:
        """Find first available feature from candidates.

        Args:
            features: Feature series.
            candidates: List of candidate column names.

        Returns:
            First matching column name, or None.
        """
        for col in candidates:
            if col in features.index:
                val = features[col]
                if not pd.isna(val):
                    return col
        return None


def compute_regime_exposure_multiplier(
    regime_features: Union[pd.Series, pd.DataFrame],
    config: RegimeGatingConfig,
) -> Tuple[float, Dict]:
    """Compute exposure multiplier from regime features.

    Convenience function for single-date regime gating.

    Args:
        regime_features: Regime features (Series or single-row DataFrame).
        config: Gating configuration.

    Returns:
        Tuple of (multiplier, diagnostics_dict).
    """
    if isinstance(regime_features, pd.DataFrame):
        if len(regime_features) == 0:
            return 1.0, {"triggered_rules": [], "enabled": config.enabled}
        regime_features = regime_features.iloc[0]

    gate = RegimeGate(config)
    result = gate.compute_multiplier(regime_features)

    diagnostics = {
        "exposure_multiplier": result.exposure_multiplier,
        "triggered_rules": result.triggered_rules,
        "rule_multipliers": result.rule_multipliers,
        "regime_features": result.regime_features,
        "enabled": config.enabled,
    }

    return result.exposure_multiplier, diagnostics


def apply_regime_gating(
    signals: pd.DataFrame,
    regime_features: Union[pd.Series, pd.DataFrame],
    config: RegimeGatingConfig,
) -> pd.DataFrame:
    """Apply regime gating to signals.

    Scales combined_weight by exposure multiplier and filters
    to allowed models only.

    Args:
        signals: DataFrame with combined_weight column.
        regime_features: Regime features for gating.
        config: Gating configuration.

    Returns:
        DataFrame with gated weights and diagnostics.
    """
    if not config.enabled:
        signals["gating_multiplier"] = 1.0
        signals["gating_triggered"] = False
        return signals

    result = signals.copy()

    # Compute multiplier
    multiplier, diagnostics = compute_regime_exposure_multiplier(
        regime_features, config
    )

    # Apply to weights
    if "combined_weight" in result.columns:
        # Apply overall multiplier
        result["combined_weight"] = result["combined_weight"] * multiplier

        # Apply breadth-poor multiplier to longs specifically
        if "breadth_poor_long" in diagnostics.get("rule_multipliers", {}):
            long_mult = diagnostics["rule_multipliers"]["breadth_poor_long"]
            long_mask = result["combined_weight"] > 0
            result.loc[long_mask, "combined_weight"] *= long_mult

    # Filter to allowed models
    if "contributing_model" in result.columns and config.allowed_models is not None:
        allowed = set(config.allowed_models)
        model_ok = result["contributing_model"].isin(allowed) | result["contributing_model"].isna()
        result.loc[~model_ok, "combined_weight"] = 0.0

    # Add diagnostics
    result["gating_multiplier"] = multiplier
    result["gating_triggered"] = len(diagnostics.get("triggered_rules", [])) > 0
    result["gating_rules"] = ",".join(diagnostics.get("triggered_rules", []))

    return result


def get_weekly_gating_diagnostics(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Get per-week gating diagnostics.

    Args:
        signals: DataFrame with gating columns.

    Returns:
        DataFrame with weekly gating summary.
    """
    if "week_monday" not in signals.columns:
        return pd.DataFrame()

    weekly = signals.groupby("week_monday").agg({
        "gating_multiplier": "first",
        "gating_triggered": "first",
        "gating_rules": "first",
    }).reset_index()

    # Compute fraction of weeks with gating
    n_weeks = len(weekly)
    n_gated = weekly["gating_triggered"].sum()
    weekly["cumulative_gated_frac"] = (
        weekly["gating_triggered"].cumsum() / (weekly.index + 1)
    )

    return weekly


# TPE optimization helpers

def suggest_gating_params(trial) -> Dict:
    """Suggest gating parameters for TPE optimization.

    Args:
        trial: Optuna trial object.

    Returns:
        Dict of suggested parameters.
    """
    from .config import get_tpe_param_range

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
    }

    return params


def create_gating_config_from_params(
    params: Dict,
    enabled: bool = True,
) -> RegimeGatingConfig:
    """Create gating config from optimization parameters.

    Args:
        params: Dict of parameter values.
        enabled: Whether gating is enabled.

    Returns:
        RegimeGatingConfig object.
    """
    return RegimeGatingConfig(
        enabled=enabled,
        vix_high_threshold=params.get("vix_high_threshold", 80.0),
        vix_high_exposure_mult=params.get("vix_high_exposure_mult", 0.7),
        credit_risk_threshold=params.get("credit_risk_threshold", 1.5),
        credit_risk_exposure_mult=params.get("credit_risk_exposure_mult", 0.8),
        breadth_poor_threshold=params.get("breadth_poor_threshold", 30.0),
        breadth_poor_long_mult=params.get("breadth_poor_long_mult", 0.8),
    )
