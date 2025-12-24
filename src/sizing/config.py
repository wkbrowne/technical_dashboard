"""Configuration for multi-model position sizing.

This module defines configuration dataclasses and JSON schema for:
- Multi-model sizing with 4 models (LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC)
- Combining policies (mode_priority, blend)
- Regime gating overlay
- TPE-optimizable parameters
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import json


class ModelType(Enum):
    """Model types for multi-model sizing."""
    LONG_NORMAL = "long_normal"
    LONG_PARABOLIC = "long_parabolic"
    SHORT_NORMAL = "short_normal"
    SHORT_PARABOLIC = "short_parabolic"

    @classmethod
    def long_models(cls) -> Set["ModelType"]:
        """Get set of long models."""
        return {cls.LONG_NORMAL, cls.LONG_PARABOLIC}

    @classmethod
    def short_models(cls) -> Set["ModelType"]:
        """Get set of short models."""
        return {cls.SHORT_NORMAL, cls.SHORT_PARABOLIC}

    @classmethod
    def normal_models(cls) -> Set["ModelType"]:
        """Get set of normal models."""
        return {cls.LONG_NORMAL, cls.SHORT_NORMAL}

    @classmethod
    def parabolic_models(cls) -> Set["ModelType"]:
        """Get set of parabolic models."""
        return {cls.LONG_PARABOLIC, cls.SHORT_PARABOLIC}

    @classmethod
    def from_string(cls, s: str) -> "ModelType":
        """Convert string to ModelType."""
        return cls(s.lower())

    @property
    def is_long(self) -> bool:
        """Check if this is a long model."""
        return self in self.long_models()

    @property
    def is_short(self) -> bool:
        """Check if this is a short model."""
        return self in self.short_models()

    @property
    def direction_sign(self) -> int:
        """Get direction sign (+1 for long, -1 for short)."""
        return 1 if self.is_long else -1


class CombinePolicy(Enum):
    """Policy for combining multiple model signals."""
    MODE_PRIORITY = "mode_priority"  # Pick model with highest edge score
    BLEND = "blend"  # Weighted blend of all models


class NettingPolicy(Enum):
    """Policy for handling conflicting long/short signals."""
    STRONGEST = "strongest"  # Pick direction with strongest absolute edge
    NET = "net"  # Net long and short weights


@dataclass
class MonotoneSizingParams:
    """Parameters for monotone probability sizing.

    Formula: raw_weight = exposure_mult * clip(sigmoid(slope * (p - intercept)), 0, 1)

    Attributes:
        slope: Sensitivity to probability (higher = steeper).
        intercept: Probability threshold (signals below this get near-zero weight).
        exposure_mult: Overall exposure multiplier.
        max_weight: Maximum weight per position.
        min_weight: Minimum weight to include.
    """
    slope: float = 2.0
    intercept: float = 0.5
    exposure_mult: float = 1.0
    max_weight: float = 0.10
    min_weight: float = 0.01

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MonotoneSizingParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RegimeGatingConfig:
    """Configuration for regime-based exposure gating.

    Gating acts as a hard risk-control overlay that adjusts exposure
    based on market conditions, independent of model predictions.

    Why gating even if regime is in the model?
    - Models include regime features for alpha (predicting returns)
    - Gating is for risk control (hard exposure limits regardless of alpha)
    - Separation of concerns: models predict, gating controls risk

    Attributes:
        enabled: Whether regime gating is active.
        vix_high_threshold: VIX percentile above which to reduce exposure.
        vix_high_exposure_mult: Exposure multiplier when VIX is high.
        credit_risk_threshold: Credit spread z-score threshold.
        credit_risk_exposure_mult: Exposure multiplier when credit risk is high.
        breadth_poor_threshold: Breadth percentile below which to reduce longs.
        breadth_poor_long_mult: Long exposure multiplier when breadth is poor.
        allowed_models: Set of models allowed in current regime (None = all).
    """
    enabled: bool = False

    # VIX gating
    vix_high_threshold: float = 80.0  # 80th percentile
    vix_high_exposure_mult: float = 0.7

    # Credit spread gating
    credit_risk_threshold: float = 1.5  # Z-score
    credit_risk_exposure_mult: float = 0.8

    # Breadth gating
    breadth_poor_threshold: float = 30.0  # 30th percentile
    breadth_poor_long_mult: float = 0.8

    # Model filtering (optional)
    allowed_models: Optional[List[str]] = None  # None = all models allowed

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Handle None properly
        if self.allowed_models is None:
            d["allowed_models"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "RegimeGatingConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def disabled(cls) -> "RegimeGatingConfig":
        """Create a disabled (neutral) gating config."""
        return cls(
            enabled=False,
            vix_high_threshold=100.0,  # Never triggers
            vix_high_exposure_mult=1.0,
            credit_risk_threshold=100.0,
            credit_risk_exposure_mult=1.0,
            breadth_poor_threshold=0.0,
            breadth_poor_long_mult=1.0,
        )


@dataclass
class MultiModelSizingConfig:
    """Configuration for multi-model position sizing.

    Supports four models with configurable combining and gating.

    Attributes:
        models: List of model types to use.
        combine_policy: How to combine model signals.
        netting_policy: How to handle long/short conflicts.
        sizing_params: Per-model sizing parameters (or shared).
        regime_gating: Regime gating configuration.
        max_gross_exposure: Maximum gross exposure as NAV fraction.
        max_net_exposure: Maximum net exposure (long - short).
        max_weight_per_name: Maximum weight per position.
        turnover_penalty: Penalty per unit turnover.
        parabolic_threshold_offset: Extra threshold for parabolic models.
    """
    # Models to use
    models: List[str] = field(
        default_factory=lambda: [
            "long_normal",
            "long_parabolic",
            "short_normal",
            "short_parabolic",
        ]
    )

    # Combining
    combine_policy: str = "mode_priority"
    netting_policy: str = "strongest"

    # Sizing parameters (shared by default)
    sizing_params: MonotoneSizingParams = field(default_factory=MonotoneSizingParams)

    # Per-model overrides (optional)
    model_params: Optional[Dict[str, Dict]] = None

    # Regime gating
    regime_gating: RegimeGatingConfig = field(default_factory=RegimeGatingConfig)

    # Portfolio constraints
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_weight_per_name: float = 0.10
    min_weight: float = 0.01
    max_positions: Optional[int] = None
    sector_caps: Optional[Dict[str, float]] = None

    # Turnover
    turnover_penalty: float = 0.0

    # Parabolic-specific
    parabolic_threshold_offset: float = 0.05  # Extra threshold for parabolic signals

    def get_model_types(self) -> List[ModelType]:
        """Get list of ModelType enums."""
        return [ModelType.from_string(m) for m in self.models]

    def get_combine_policy(self) -> CombinePolicy:
        """Get CombinePolicy enum."""
        return CombinePolicy(self.combine_policy)

    def get_netting_policy(self) -> NettingPolicy:
        """Get NettingPolicy enum."""
        return NettingPolicy(self.netting_policy)

    def get_sizing_params_for_model(self, model: ModelType) -> MonotoneSizingParams:
        """Get sizing params for a specific model.

        Falls back to shared params if no per-model override.
        """
        if self.model_params and model.value in self.model_params:
            base = asdict(self.sizing_params)
            base.update(self.model_params[model.value])
            return MonotoneSizingParams.from_dict(base)

        # Apply parabolic offset if applicable
        if model in ModelType.parabolic_models():
            params = MonotoneSizingParams.from_dict(asdict(self.sizing_params))
            params.intercept += self.parabolic_threshold_offset
            return params

        return self.sizing_params

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "models": self.models,
            "combine_policy": self.combine_policy,
            "netting_policy": self.netting_policy,
            "sizing_params": self.sizing_params.to_dict(),
            "model_params": self.model_params,
            "regime_gating": self.regime_gating.to_dict(),
            "max_gross_exposure": self.max_gross_exposure,
            "max_net_exposure": self.max_net_exposure,
            "max_weight_per_name": self.max_weight_per_name,
            "min_weight": self.min_weight,
            "max_positions": self.max_positions,
            "sector_caps": self.sector_caps,
            "turnover_penalty": self.turnover_penalty,
            "parabolic_threshold_offset": self.parabolic_threshold_offset,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MultiModelSizingConfig":
        """Create from dictionary."""
        sizing_params = MonotoneSizingParams.from_dict(d.get("sizing_params", {}))
        regime_gating = RegimeGatingConfig.from_dict(d.get("regime_gating", {}))

        return cls(
            models=d.get("models", cls.__dataclass_fields__["models"].default_factory()),
            combine_policy=d.get("combine_policy", "mode_priority"),
            netting_policy=d.get("netting_policy", "strongest"),
            sizing_params=sizing_params,
            model_params=d.get("model_params"),
            regime_gating=regime_gating,
            max_gross_exposure=d.get("max_gross_exposure", 1.0),
            max_net_exposure=d.get("max_net_exposure", 1.0),
            max_weight_per_name=d.get("max_weight_per_name", 0.10),
            min_weight=d.get("min_weight", 0.01),
            max_positions=d.get("max_positions"),
            sector_caps=d.get("sector_caps"),
            turnover_penalty=d.get("turnover_penalty", 0.0),
            parabolic_threshold_offset=d.get("parabolic_threshold_offset", 0.05),
        )

    @classmethod
    def single_model_compat(
        cls,
        slope: float = 2.0,
        intercept: float = 0.5,
        exposure_mult: float = 1.0,
        max_gross_exposure: float = 1.0,
        max_weight: float = 0.10,
        turnover_penalty: float = 0.0,
    ) -> "MultiModelSizingConfig":
        """Create config compatible with single-model legacy mode.

        For backward compatibility with existing single-model configs.
        """
        return cls(
            models=["long_normal"],  # Single model
            combine_policy="mode_priority",
            sizing_params=MonotoneSizingParams(
                slope=slope,
                intercept=intercept,
                exposure_mult=exposure_mult,
                max_weight=max_weight,
            ),
            max_gross_exposure=max_gross_exposure,
            turnover_penalty=turnover_penalty,
            regime_gating=RegimeGatingConfig.disabled(),
        )


def load_multi_model_config(path: Union[str, Path]) -> MultiModelSizingConfig:
    """Load multi-model sizing config from JSON file.

    Args:
        path: Path to JSON config file.

    Returns:
        MultiModelSizingConfig object.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    # Handle legacy single-model format
    if "models" not in data and "slope" in data:
        return MultiModelSizingConfig.single_model_compat(
            slope=data.get("slope", 2.0),
            intercept=data.get("intercept", 0.5),
            exposure_mult=data.get("exposure_mult", 1.0),
            max_gross_exposure=data.get("max_gross_exposure", 1.0),
            max_weight=data.get("max_weight", 0.10),
            turnover_penalty=data.get("turnover_penalty", 0.0),
        )

    return MultiModelSizingConfig.from_dict(data)


def save_multi_model_config(
    config: MultiModelSizingConfig,
    path: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    """Save multi-model sizing config to JSON file.

    Args:
        config: MultiModelSizingConfig object.
        path: Output path.
        metadata: Optional metadata to include.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()
    if metadata:
        data["_metadata"] = metadata

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# TPE parameter ranges for optimization
TPE_PARAM_RANGES = {
    # Sizing params
    "slope": (1.0, 5.0),
    "intercept": (0.3, 0.7),
    "exposure_mult": (0.5, 1.5),
    "turnover_penalty": (0.0, 0.02),
    "parabolic_threshold_offset": (0.0, 0.15),

    # Regime gating params
    "vix_high_threshold": (60.0, 95.0),
    "vix_high_exposure_mult": (0.3, 1.0),
    "credit_risk_threshold": (1.0, 3.0),
    "credit_risk_exposure_mult": (0.5, 1.0),
    "breadth_poor_threshold": (20.0, 50.0),
    "breadth_poor_long_mult": (0.5, 1.0),
}


def get_tpe_param_range(param_name: str) -> tuple:
    """Get TPE optimization range for a parameter.

    Args:
        param_name: Parameter name.

    Returns:
        Tuple of (min, max) values.
    """
    return TPE_PARAM_RANGES.get(param_name, (0.0, 1.0))
