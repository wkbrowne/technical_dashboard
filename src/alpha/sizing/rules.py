"""Position sizing rules.

This module implements various position sizing strategies:
    A) Monotone probability mapping
    B) Rank-bucket sizing
    C) Two-stage threshold + confidence sizing
    D) Volatility-targeted overlay

All rules support the same interface and can be combined with
portfolio constraints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ..config import SizingConfig, SizingMethod


class BaseSizingRule(ABC):
    """Abstract base class for sizing rules."""

    @abstractmethod
    def compute_weights(
        self,
        signals: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """Compute raw position weights.

        Args:
            signals: DataFrame with probability and other features.
            **kwargs: Additional parameters.

        Returns:
            Series of raw weights indexed by signal index.
        """
        pass

    def get_params(self) -> Dict[str, float]:
        """Get sizing parameters."""
        return {}

    def set_params(self, **params) -> None:
        """Set sizing parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class MonotoneProbabilitySizing(BaseSizingRule):
    """Monotone probability mapping sizing.

    Computes weights as:
        w_raw = slope * (p_cal - intercept)
        w = clip(w_raw, 0, max_weight)

    Higher probability -> higher weight, with linear mapping.

    Attributes:
        slope: Slope parameter 'a'.
        intercept: Intercept parameter 'b'.
        max_weight: Maximum weight per position.
        min_weight: Minimum weight to include.
    """
    slope: float = 2.0
    intercept: float = 0.5
    max_weight: float = 0.10
    min_weight: float = 0.01

    def compute_weights(
        self,
        signals: pd.DataFrame,
        prob_col: str = "probability",
        **kwargs,
    ) -> pd.Series:
        """Compute weights from probability scores."""
        probs = signals[prob_col].values

        # Linear mapping: w = slope * (p - intercept)
        raw_weights = self.slope * (probs - self.intercept)

        # Clip to valid range
        weights = np.clip(raw_weights, 0, self.max_weight)

        # Zero out weights below minimum
        weights[weights < self.min_weight] = 0

        return pd.Series(weights, index=signals.index)

    def get_params(self) -> Dict[str, float]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
        }


@dataclass
class RankBucketSizing(BaseSizingRule):
    """Rank-bucket sizing.

    Buckets signals by rank percentile and assigns monotonic
    weights per bucket.

    Attributes:
        n_buckets: Number of rank buckets.
        bucket_weights: Weights per bucket (descending by rank).
        max_weight: Maximum weight per position.
    """
    n_buckets: int = 5
    bucket_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2]
    )
    max_weight: float = 0.10

    def compute_weights(
        self,
        signals: pd.DataFrame,
        prob_col: str = "probability",
        **kwargs,
    ) -> pd.Series:
        """Compute weights from rank buckets."""
        n = len(signals)
        if n == 0:
            return pd.Series(dtype=float)

        # Rank by probability (higher = better rank)
        ranks = signals[prob_col].rank(ascending=False, method="first")

        # Assign to buckets
        bucket_size = n / self.n_buckets
        bucket_idx = ((ranks - 1) / bucket_size).astype(int)
        bucket_idx = np.clip(bucket_idx, 0, self.n_buckets - 1)

        # Get weights from bucket
        weights = np.zeros(n)
        for i, bi in enumerate(bucket_idx):
            if bi < len(self.bucket_weights):
                weights[i] = self.bucket_weights[bi]

        # Scale to max weight
        if weights.max() > 0:
            weights = weights * (self.max_weight / weights.max())

        return pd.Series(weights, index=signals.index)

    def get_params(self) -> Dict[str, float]:
        params = {"n_buckets": self.n_buckets, "max_weight": self.max_weight}
        for i, w in enumerate(self.bucket_weights):
            params[f"bucket_weight_{i}"] = w
        return params


@dataclass
class ThresholdConfidenceSizing(BaseSizingRule):
    """Two-stage threshold + confidence sizing.

    Stage 1: Entry if p_cal >= threshold
    Stage 2: Size by confidence × regime multiplier

    Attributes:
        entry_threshold: Minimum probability for entry.
        confidence_scale: Scaling factor for confidence sizing.
        regime_multipliers: Dict of regime -> multiplier.
        max_weight: Maximum weight per position.
        min_weight: Minimum weight to include.
    """
    entry_threshold: float = 0.5
    confidence_scale: float = 1.0
    regime_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "low_vix": 1.2,
            "normal_vix": 1.0,
            "high_vix": 0.8,
        }
    )
    max_weight: float = 0.10
    min_weight: float = 0.01

    def compute_weights(
        self,
        signals: pd.DataFrame,
        prob_col: str = "probability",
        regime_col: Optional[str] = "vix_regime",
        **kwargs,
    ) -> pd.Series:
        """Compute weights with threshold entry and confidence sizing."""
        probs = signals[prob_col].values
        n = len(signals)

        # Stage 1: Entry filter
        entry_mask = probs >= self.entry_threshold

        # Stage 2: Confidence sizing
        # Confidence = how far above threshold
        confidence = (probs - self.entry_threshold) / (1 - self.entry_threshold)
        confidence = np.clip(confidence, 0, 1)

        # Base weights from confidence
        weights = confidence * self.confidence_scale * self.max_weight

        # Apply regime multipliers
        if regime_col and regime_col in signals.columns:
            for regime, mult in self.regime_multipliers.items():
                regime_mask = signals[regime_col] == regime
                weights[regime_mask] *= mult

        # Apply entry filter
        weights[~entry_mask] = 0

        # Clip and filter
        weights = np.clip(weights, 0, self.max_weight)
        weights[weights < self.min_weight] = 0

        return pd.Series(weights, index=signals.index)

    def get_params(self) -> Dict[str, float]:
        params = {
            "entry_threshold": self.entry_threshold,
            "confidence_scale": self.confidence_scale,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
        }
        for regime, mult in self.regime_multipliers.items():
            params[f"regime_{regime}"] = mult
        return params


@dataclass
class VolatilityTargetedSizing(BaseSizingRule):
    """Volatility-targeted portfolio sizing.

    Targets a portfolio-level volatility, capped by max exposure.

    weight_i = (target_vol / portfolio_vol) * base_weight_i

    Base weights come from another sizing rule (e.g., monotone).

    Attributes:
        base_rule: Underlying sizing rule.
        vol_target_annual: Annual volatility target (e.g., 0.15 = 15%).
        max_gross_exposure: Maximum gross exposure cap.
        vol_lookback_days: Days for vol estimation.
        annualization_factor: Trading days per year.
    """
    base_rule: Optional[BaseSizingRule] = None
    vol_target_annual: float = 0.15
    max_gross_exposure: float = 1.0
    vol_lookback_days: int = 63
    annualization_factor: float = 252.0

    def __post_init__(self):
        if self.base_rule is None:
            self.base_rule = MonotoneProbabilitySizing()

    def compute_weights(
        self,
        signals: pd.DataFrame,
        prob_col: str = "probability",
        vol_col: str = "atr_percent",
        historical_returns: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.Series:
        """Compute volatility-targeted weights."""
        # Get base weights
        base_weights = self.base_rule.compute_weights(signals, prob_col=prob_col)

        # Estimate portfolio volatility
        if vol_col in signals.columns:
            # Use individual asset volatilities
            asset_vols = signals[vol_col].values
            asset_vols = np.maximum(asset_vols, 0.01)  # Floor at 1%

            # Simple approximation: portfolio vol as weighted average of asset vols
            # (Assumes zero correlation for simplicity; more sophisticated would use covariance)
            weighted_vols = base_weights.values * asset_vols
            portfolio_vol_daily = np.sqrt((weighted_vols ** 2).sum())

            # Annualize
            portfolio_vol_annual = portfolio_vol_daily * np.sqrt(self.annualization_factor)

        else:
            # Default assumption
            portfolio_vol_annual = 0.20  # 20% vol

        # Scale factor to hit target vol
        if portfolio_vol_annual > 0:
            vol_scale = self.vol_target_annual / portfolio_vol_annual
        else:
            vol_scale = 1.0

        # Apply scaling
        scaled_weights = base_weights * vol_scale

        # Cap at max exposure
        gross_exposure = scaled_weights.sum()
        if gross_exposure > self.max_gross_exposure:
            scaled_weights = scaled_weights * (self.max_gross_exposure / gross_exposure)

        return scaled_weights

    def get_params(self) -> Dict[str, float]:
        params = {
            "vol_target_annual": self.vol_target_annual,
            "max_gross_exposure": self.max_gross_exposure,
        }
        if self.base_rule:
            base_params = self.base_rule.get_params()
            params.update({f"base_{k}": v for k, v in base_params.items()})
        return params


def create_sizing_rule(
    config: SizingConfig,
) -> BaseSizingRule:
    """Factory function to create sizing rule from config.

    Args:
        config: SizingConfig object.

    Returns:
        Configured sizing rule.
    """
    method = config.method

    if method == SizingMethod.MONOTONE_PROBABILITY:
        return MonotoneProbabilitySizing(
            slope=config.prob_slope,
            intercept=config.prob_intercept,
            max_weight=config.max_name_weight,
            min_weight=config.min_name_weight,
        )

    elif method == SizingMethod.RANK_BUCKET:
        return RankBucketSizing(
            n_buckets=config.n_buckets,
            bucket_weights=config.bucket_weights,
            max_weight=config.max_name_weight,
        )

    elif method == SizingMethod.THRESHOLD_CONFIDENCE:
        return ThresholdConfidenceSizing(
            entry_threshold=config.entry_threshold,
            confidence_scale=config.confidence_scale,
            max_weight=config.max_name_weight,
            min_weight=config.min_name_weight,
        )

    elif method == SizingMethod.VOLATILITY_TARGETED:
        base_rule = MonotoneProbabilitySizing(
            slope=config.prob_slope,
            intercept=config.prob_intercept,
            max_weight=config.max_name_weight,
            min_weight=config.min_name_weight,
        )
        return VolatilityTargetedSizing(
            base_rule=base_rule,
            vol_target_annual=config.vol_target_annual,
            max_gross_exposure=config.max_gross_exposure,
        )

    else:
        raise ValueError(f"Unknown sizing method: {method}")


def compute_equal_weights(
    signals: pd.DataFrame,
    top_n: Optional[int] = None,
    max_weight: float = 0.10,
    prob_col: str = "probability",
) -> pd.Series:
    """Compute equal weights for baseline comparison.

    Args:
        signals: Signals DataFrame.
        top_n: Number of top signals to include.
        max_weight: Maximum weight per position.
        prob_col: Column for ranking.

    Returns:
        Equal weight series.
    """
    if top_n is not None:
        # Take top N by probability
        top_signals = signals.nlargest(top_n, prob_col)
        n = len(top_signals)
        weight = min(1.0 / n, max_weight) if n > 0 else 0
        weights = pd.Series(0.0, index=signals.index)
        weights[top_signals.index] = weight
    else:
        n = len(signals)
        weight = min(1.0 / n, max_weight) if n > 0 else 0
        weights = pd.Series(weight, index=signals.index)

    return weights


def compute_probability_weights(
    signals: pd.DataFrame,
    top_n: Optional[int] = None,
    max_weight: float = 0.10,
    prob_col: str = "probability",
) -> pd.Series:
    """Compute probability-proportional weights for baseline.

    Args:
        signals: Signals DataFrame.
        top_n: Number of top signals to include.
        max_weight: Maximum weight per position.
        prob_col: Column for ranking.

    Returns:
        Probability-proportional weight series.
    """
    if top_n is not None:
        top_signals = signals.nlargest(top_n, prob_col)
        probs = top_signals[prob_col].values
    else:
        top_signals = signals
        probs = signals[prob_col].values

    # Normalize to sum to 1
    if probs.sum() > 0:
        weights_raw = probs / probs.sum()
    else:
        weights_raw = np.ones(len(probs)) / len(probs)

    # Clip individual weights
    weights_raw = np.clip(weights_raw, 0, max_weight)

    # Renormalize after clipping
    if weights_raw.sum() > 0:
        weights_raw = weights_raw / weights_raw.sum()

    weights = pd.Series(0.0, index=signals.index)
    weights[top_signals.index] = weights_raw

    return weights
