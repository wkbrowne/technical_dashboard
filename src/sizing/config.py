"""Configuration for multi-model position sizing.

This module defines configuration dataclasses and JSON schema for:
- Multi-model sizing with 4 models (LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC)
- Per-model sizing parameters with short selectivity controls
- Combining policies (mode_priority, blend)
- Direction-aware regime gating overlay
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
    def is_parabolic(self) -> bool:
        """Check if this is a parabolic model."""
        return self in self.parabolic_models()

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

    Formula: raw_weight = exposure_mult * clip(slope * (p - intercept), 0, max_weight)

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
class ShortSelectivityConfig:
    """Configuration for making shorts more selective than longs.

    Short selectivity is achieved through multiple mechanisms that compound:
    1. Intercept offset: shorts require higher probability to trigger
    2. Max weight multiplier: shorts get smaller position sizes
    3. Regime gating: shorts can have additional gating multiplier

    All defaults are set to make shorts more selective.

    Attributes:
        short_threshold_offset: Added to intercept for short models (default 0.05).
            e.g., if base intercept=0.5, shorts need p > 0.55.
        short_max_weight_mult: Multiplier for max_weight on shorts (default 0.8).
            e.g., if base max_weight=0.10, shorts max at 0.08.
        short_exposure_mult: Additional exposure multiplier for shorts (default 1.0).
            Applied after base exposure_mult.
    """
    short_threshold_offset: float = 0.05
    short_max_weight_mult: float = 0.8
    short_exposure_mult: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ShortSelectivityConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def neutral(cls) -> "ShortSelectivityConfig":
        """Create neutral config (no short bias)."""
        return cls(
            short_threshold_offset=0.0,
            short_max_weight_mult=1.0,
            short_exposure_mult=1.0,
        )


@dataclass
class RegimeGatingConfig:
    """Configuration for regime-based exposure gating.

    Gating acts as a hard risk-control overlay that adjusts exposure
    based on market conditions, independent of model predictions.

    Direction-aware gating:
    - VIX and credit rules affect both directions equally by default
    - Breadth rules affect longs and shorts separately

    Attributes:
        enabled: Whether regime gating is active.
        strict_missing_features: If True, raise error when regime features missing.
            If False, warn and use multiplier=1.0 for missing rules.

        vix_high_threshold: VIX percentile above which to reduce exposure.
        vix_high_exposure_mult: Exposure multiplier when VIX is high (both directions).

        credit_risk_threshold: Credit spread z-score threshold.
        credit_risk_exposure_mult: Exposure multiplier when credit risk is high.

        breadth_poor_threshold: Breadth percentile below which to reduce exposure.
        breadth_poor_long_mult: Long exposure multiplier when breadth is poor.
        breadth_poor_short_mult: Short exposure multiplier when breadth is poor.
            Default 1.0 means shorts are not reduced by poor breadth.
            Set < 1.0 if poor breadth should also reduce shorts (unusual).

        short_regime_mult: Additional regime multiplier applied only to shorts.
            This is a "short gating" mechanism that makes shorts more conservative.
    """
    enabled: bool = False
    strict_missing_features: bool = False

    # VIX gating (affects both directions)
    vix_high_threshold: float = 80.0  # 80th percentile
    vix_high_exposure_mult: float = 0.7

    # Credit spread gating (affects both directions)
    credit_risk_threshold: float = 1.5  # Z-score
    credit_risk_exposure_mult: float = 0.8

    # Breadth gating (direction-aware)
    breadth_poor_threshold: float = 30.0  # 30th percentile
    breadth_poor_long_mult: float = 0.8
    breadth_poor_short_mult: float = 1.0  # Shorts not reduced by poor breadth by default

    # Additional short gating multiplier
    short_regime_mult: float = 1.0  # Set < 1.0 to always reduce shorts

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "RegimeGatingConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def disabled(cls) -> "RegimeGatingConfig":
        """Create a disabled (neutral) gating config."""
        return cls(
            enabled=False,
            strict_missing_features=False,
            vix_high_threshold=100.0,  # Never triggers
            vix_high_exposure_mult=1.0,
            credit_risk_threshold=100.0,
            credit_risk_exposure_mult=1.0,
            breadth_poor_threshold=0.0,
            breadth_poor_long_mult=1.0,
            breadth_poor_short_mult=1.0,
            short_regime_mult=1.0,
        )


@dataclass
class MultiModelSizingConfig:
    """Configuration for multi-model position sizing.

    Supports four models with configurable combining, gating, and short selectivity.

    Sizing parameter precedence (highest to lowest):
    1. Per-model overrides in model_params[model_name]
    2. Short selectivity adjustments (for short models)
    3. Parabolic threshold offset (for parabolic models)
    4. Base sizing_params

    Attributes:
        models: List of model types to use.
        combine_policy: How to combine model signals ("mode_priority" or "blend").
        netting_policy: How to handle long/short conflicts ("strongest" or "net").
        sizing_params: Base sizing parameters (shared across models).
        model_params: Per-model parameter overrides (optional).
        short_selectivity: Short selectivity configuration.
        regime_gating: Regime gating configuration.
        max_gross_exposure: Maximum gross exposure as NAV fraction.
        max_net_exposure: Maximum net exposure (long - short).
        max_weight_per_name: Maximum weight per position.
        min_weight: Minimum weight threshold.
        max_positions: Maximum number of positions (optional).
        turnover_penalty: Penalty per unit turnover (optimizer proxy for costs).
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

    # Short selectivity
    short_selectivity: ShortSelectivityConfig = field(default_factory=ShortSelectivityConfig)

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

        Applies adjustments in order:
        1. Start with base sizing_params
        2. Apply per-model overrides from model_params
        3. Apply short selectivity (for short models)
        4. Apply parabolic offset (for parabolic models)
        """
        # Start with base params
        base = asdict(self.sizing_params)

        # Apply per-model overrides if present
        if self.model_params and model.value in self.model_params:
            base.update(self.model_params[model.value])

        params = MonotoneSizingParams.from_dict(base)

        # Apply short selectivity for short models
        if model.is_short:
            params.intercept += self.short_selectivity.short_threshold_offset
            params.max_weight *= self.short_selectivity.short_max_weight_mult
            params.exposure_mult *= self.short_selectivity.short_exposure_mult

        # Apply parabolic offset for parabolic models
        if model.is_parabolic:
            params.intercept += self.parabolic_threshold_offset

        return params

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "models": self.models,
            "combine_policy": self.combine_policy,
            "netting_policy": self.netting_policy,
            "sizing_params": self.sizing_params.to_dict(),
            "model_params": self.model_params,
            "short_selectivity": self.short_selectivity.to_dict(),
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
        short_selectivity = ShortSelectivityConfig.from_dict(d.get("short_selectivity", {}))
        regime_gating = RegimeGatingConfig.from_dict(d.get("regime_gating", {}))

        return cls(
            models=d.get("models", cls.__dataclass_fields__["models"].default_factory()),
            combine_policy=d.get("combine_policy", "mode_priority"),
            netting_policy=d.get("netting_policy", "strongest"),
            sizing_params=sizing_params,
            model_params=d.get("model_params"),
            short_selectivity=short_selectivity,
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
            short_selectivity=ShortSelectivityConfig.neutral(),
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
    # Base sizing params
    "slope": (1.0, 5.0),
    "intercept": (0.3, 0.7),
    "exposure_mult": (0.5, 1.5),
    "turnover_penalty": (0.0, 0.02),
    "parabolic_threshold_offset": (0.0, 0.15),

    # Short selectivity params
    "short_threshold_offset": (0.0, 0.15),
    "short_max_weight_mult": (0.5, 1.0),
    "short_exposure_mult": (0.5, 1.0),

    # Regime gating params
    "vix_high_threshold": (60.0, 95.0),
    "vix_high_exposure_mult": (0.3, 1.0),
    "credit_risk_threshold": (1.0, 3.0),
    "credit_risk_exposure_mult": (0.5, 1.0),
    "breadth_poor_threshold": (20.0, 50.0),
    "breadth_poor_long_mult": (0.5, 1.0),
    "breadth_poor_short_mult": (0.7, 1.0),  # Typically less aggressive reduction
    "short_regime_mult": (0.7, 1.0),
}


def get_tpe_param_range(param_name: str) -> tuple:
    """Get TPE optimization range for a parameter.

    Args:
        param_name: Parameter name.

    Returns:
        Tuple of (min, max) values.
    """
    return TPE_PARAM_RANGES.get(param_name, (0.0, 1.0))


# Regime feature column names (used by regime_gating and regime_data modules)
REGIME_FEATURE_COLS = {
    "vix_percentile": [
        "vix_percentile_252d",
        "vix_pct_rank_252d",
        "w_vix_percentile_252d",
    ],
    "credit_spread": [
        "fred_bamlh0a0hym2_z60",
        "credit_spread_zscore",
        "hy_spread_z60",
        "w_fred_bamlh0a0hym2_z60",
    ],
    "breadth": [
        "sector_breadth_pct_above_ma200",
        "breadth_pct_above_200dma",
        "w_sector_breadth_pct_above_ma200",
    ],
}
