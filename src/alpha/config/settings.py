"""Configuration dataclasses and loading utilities for alpha module.

This module defines all configuration objects used throughout the position
sizing and backtesting pipeline.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml


class SizingMethod(Enum):
    """Position sizing methods."""
    MONOTONE_PROBABILITY = "monotone_probability"
    RANK_BUCKET = "rank_bucket"
    THRESHOLD_CONFIDENCE = "threshold_confidence"
    VOLATILITY_TARGETED = "volatility_targeted"


class ExposureMode(Enum):
    """Exposure control modes."""
    NAV_FRACTION = "nav_fraction"  # Fixed fraction of NAV
    VOL_TARGETED = "vol_targeted"  # Volatility-targeted, capped by NAV


class WeightScheme(Enum):
    """Sample weighting schemes for training."""
    UNIFORM = "uniform"
    LIQUIDITY = "liquidity"  # Weight by ADV
    INVERSE_VOLATILITY = "inverse_volatility"  # 1/ATR%
    ATR_PERCENT = "atr_percent"  # Weight by ATR%
    OVERLAP_INVERSE = "overlap_inverse"  # Pre-computed in targets


@dataclass
class CVConfig:
    """Configuration for purged walk-forward cross-validation.

    Attributes:
        n_splits: Number of CV folds.
        min_train_weeks: Minimum training period in weeks.
        test_weeks: Test period size in weeks.
        purge_days: Days to purge from training end (overlap prevention).
        embargo_days: Days to skip after test period before next train.
        horizon_days: Label horizon in trading days (used for purge calculation).
    """
    n_splits: int = 5
    min_train_weeks: int = 52  # 1 year minimum
    test_weeks: int = 13  # ~1 quarter test
    purge_days: int = 25  # >= horizon to prevent leakage
    embargo_days: int = 5  # Gap after test
    horizon_days: int = 20  # Triple barrier horizon

    def __post_init__(self):
        if self.purge_days < self.horizon_days:
            raise ValueError(
                f"purge_days ({self.purge_days}) must be >= horizon_days ({self.horizon_days}) "
                "to prevent label leakage"
            )


@dataclass
class CostConfig:
    """Transaction cost configuration.

    Attributes:
        spread_bps: Half-spread in basis points (each way).
        commission_bps: Commission in basis points (each way).
        slippage_bps: Market impact in basis points (each way).
        min_trade_value: Minimum trade value (USD) to execute.
    """
    spread_bps: float = 5.0  # 5 bps each way
    commission_bps: float = 0.0  # Assume zero commission
    slippage_bps: float = 5.0  # 5 bps market impact
    min_trade_value: float = 100.0

    @property
    def one_way_cost_bps(self) -> float:
        """Total one-way cost in basis points."""
        return self.spread_bps + self.commission_bps + self.slippage_bps

    @property
    def round_trip_cost_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return 2 * self.one_way_cost_bps


@dataclass
class SizingConfig:
    """Position sizing configuration.

    Attributes:
        method: Sizing method to use.
        exposure_mode: How to control gross exposure.
        max_gross_exposure: Maximum gross exposure as NAV fraction.
        max_name_weight: Maximum weight per position.
        min_name_weight: Minimum weight to include position.
        cash_buffer: Minimum cash reserve fraction.
        sector_caps: Optional dict of sector -> max weight.
        turnover_penalty: Penalty per unit turnover (for optimizer).
        vol_target_annual: Annual volatility target (for vol-targeted mode).
        vol_lookback_days: Lookback for volatility estimation.

        # Monotone probability params
        prob_slope: Slope parameter 'a' in w = a * (p - b).
        prob_intercept: Intercept parameter 'b'.

        # Rank bucket params
        n_buckets: Number of rank buckets.
        bucket_weights: Weights per bucket (descending order).

        # Threshold params
        entry_threshold: Minimum probability for entry.
        confidence_scale: Scaling factor for confidence-based sizing.

        # Trade selection
        top_n: Maximum number of positions.
        top_pct: Alternative: top percentile of scores.
    """
    method: SizingMethod = SizingMethod.MONOTONE_PROBABILITY
    exposure_mode: ExposureMode = ExposureMode.NAV_FRACTION

    # Exposure controls
    max_gross_exposure: float = 1.0  # 100% NAV
    max_name_weight: float = 0.10  # 10% max per name
    min_name_weight: float = 0.01  # 1% minimum
    cash_buffer: float = 0.0  # No cash buffer by default
    sector_caps: Optional[Dict[str, float]] = None
    turnover_penalty: float = 0.0  # No penalty by default

    # Volatility targeting (when exposure_mode = VOL_TARGETED)
    vol_target_annual: float = 0.15  # 15% annual vol target
    vol_lookback_days: int = 63  # ~3 months

    # Monotone probability sizing
    prob_slope: float = 2.0
    prob_intercept: float = 0.5

    # Rank bucket sizing
    n_buckets: int = 5
    bucket_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2]
    )

    # Threshold-based sizing
    entry_threshold: float = 0.5
    confidence_scale: float = 1.0

    # Trade selection
    top_n: Optional[int] = 20
    top_pct: Optional[float] = None  # Alternative to top_n

    def __post_init__(self):
        if self.max_gross_exposure > 1.0 and self.exposure_mode == ExposureMode.NAV_FRACTION:
            pass  # Allow leverage if explicitly set
        if self.max_name_weight > self.max_gross_exposure:
            raise ValueError("max_name_weight cannot exceed max_gross_exposure")
        if self.top_n is not None and self.top_pct is not None:
            raise ValueError("Specify either top_n or top_pct, not both")


@dataclass
class BacktestConfig:
    """Backtest engine configuration.

    Attributes:
        initial_capital: Starting capital (USD).
        entry_day: Day of week for entries (0=Monday).
        rebalance_day: Day of week for rebalance decisions (4=Friday).
        use_barrier_exits: Use actual barrier exit timestamps vs horizon returns.
        reinvest_profits: Whether to reinvest profits.
        track_positions: Whether to track full position history.
    """
    initial_capital: float = 1_000_000.0
    entry_day: int = 0  # Monday
    rebalance_day: int = 4  # Friday
    use_barrier_exits: bool = True
    reinvest_profits: bool = True
    track_positions: bool = True


@dataclass
class AlphaConfig:
    """Master configuration for the alpha module.

    Combines all sub-configurations into a single object.
    """
    cv: CVConfig = field(default_factory=CVConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    # Paths
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    sizing_dir: str = "artifacts/sizing"
    backtests_dir: str = "artifacts/backtests"
    reports_dir: str = "artifacts/reports"

    # Data files
    features_file: str = "artifacts/features_complete.parquet"
    targets_file: str = "artifacts/targets_triple_barrier.parquet"
    predictions_file: str = "artifacts/predictions/predictions_latest.parquet"

    # Random seed
    random_state: int = 42


def load_config(path: Union[str, Path]) -> AlphaConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        AlphaConfig object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse nested configs
    cv_data = data.get("cv", {})
    sizing_data = data.get("sizing", {})
    backtest_data = data.get("backtest", {})
    costs_data = data.get("costs", {})

    # Handle enums
    if "method" in sizing_data:
        sizing_data["method"] = SizingMethod(sizing_data["method"])
    if "exposure_mode" in sizing_data:
        sizing_data["exposure_mode"] = ExposureMode(sizing_data["exposure_mode"])

    config = AlphaConfig(
        cv=CVConfig(**cv_data),
        sizing=SizingConfig(**sizing_data),
        backtest=BacktestConfig(**backtest_data),
        costs=CostConfig(**costs_data),
    )

    # Override top-level fields
    for key in ["artifacts_dir", "models_dir", "sizing_dir", "backtests_dir",
                "reports_dir", "features_file", "targets_file", "predictions_file",
                "random_state"]:
        if key in data:
            setattr(config, key, data[key])

    return config


def save_config(config: AlphaConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: AlphaConfig object to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = {
        "cv": asdict(config.cv),
        "sizing": asdict(config.sizing),
        "backtest": asdict(config.backtest),
        "costs": asdict(config.costs),
        "artifacts_dir": config.artifacts_dir,
        "models_dir": config.models_dir,
        "sizing_dir": config.sizing_dir,
        "backtests_dir": config.backtests_dir,
        "reports_dir": config.reports_dir,
        "features_file": config.features_file,
        "targets_file": config.targets_file,
        "predictions_file": config.predictions_file,
        "random_state": config.random_state,
    }

    # Convert enums to strings
    data["sizing"]["method"] = data["sizing"]["method"].value
    data["sizing"]["exposure_mode"] = data["sizing"]["exposure_mode"].value

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Preset configurations
def conservative_config() -> AlphaConfig:
    """Conservative configuration with low exposure and strict controls."""
    return AlphaConfig(
        sizing=SizingConfig(
            max_gross_exposure=0.3,
            max_name_weight=0.05,
            entry_threshold=0.6,
            top_n=10,
            turnover_penalty=0.001,
        ),
        costs=CostConfig(
            spread_bps=7.5,
            slippage_bps=7.5,
        ),
    )


def moderate_config() -> AlphaConfig:
    """Moderate configuration with balanced exposure."""
    return AlphaConfig(
        sizing=SizingConfig(
            max_gross_exposure=0.6,
            max_name_weight=0.08,
            entry_threshold=0.55,
            top_n=15,
        ),
    )


def aggressive_config() -> AlphaConfig:
    """Aggressive configuration with high exposure."""
    return AlphaConfig(
        sizing=SizingConfig(
            max_gross_exposure=1.0,
            max_name_weight=0.10,
            entry_threshold=0.5,
            top_n=20,
        ),
    )
