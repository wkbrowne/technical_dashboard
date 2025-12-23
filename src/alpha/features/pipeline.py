"""Feature pipeline for position sizing.

This module provides feature preparation for sizing models,
ensuring all transforms are fit only on training data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .transforms import FeatureTransformer, create_default_transformer


# Default features useful for position sizing
SIZING_FEATURES = [
    # Volatility (for vol-targeted sizing)
    "atr_percent",
    "volatility_21d",
    "volatility_63d",

    # Trend strength (for confidence sizing)
    "trend_score",
    "trend_score_sign",
    "rsi_14",

    # Volume/Liquidity (for liquidity-weighted sizing)
    "volume_ratio_20d",
    "adv_20d",
    "dollar_volume",

    # Momentum (for regime-aware sizing)
    "alpha_mom_spy_20_ema10",
    "relative_strength_20d",

    # Market regime
    "vix_regime",
    "vix_percentile_252d",

    # Beta (for risk adjustment)
    "beta_market",
    "beta_spy_21d",
]


@dataclass
class SizingFeaturePipeline:
    """Pipeline for preparing sizing features.

    Ensures all feature transforms are fit only on training data
    and applied consistently to validation/test data.

    Attributes:
        sizing_features: List of features to use for sizing.
        transformer: Feature transformer instance.
        volatility_col: Column to use for volatility estimation.
        liquidity_col: Column to use for liquidity.
    """
    sizing_features: List[str] = field(default_factory=lambda: SIZING_FEATURES.copy())
    transformer: Optional[FeatureTransformer] = None
    volatility_col: str = "atr_percent"
    liquidity_col: str = "adv_20d"
    _fitted: bool = False

    def fit(
        self,
        signals: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> "SizingFeaturePipeline":
        """Fit the pipeline on training data.

        Args:
            signals: Training signals DataFrame.
            features: Optional separate features DataFrame.

        Returns:
            Fitted pipeline.
        """
        # Merge features if provided
        if features is not None:
            data = self._merge_features(signals, features)
        else:
            data = signals.copy()

        # Initialize transformer if not provided
        if self.transformer is None:
            self.transformer = create_default_transformer()

        # Filter to available sizing features
        available_features = [f for f in self.sizing_features if f in data.columns]
        if available_features:
            self.transformer.feature_columns = available_features
            self.transformer.fit(data)

        self._fitted = True
        return self

    def transform(
        self,
        signals: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Apply fitted transforms to data.

        Args:
            signals: Signals DataFrame.
            features: Optional separate features DataFrame.

        Returns:
            Transformed DataFrame with sizing features.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fit before transform")

        # Merge features if provided
        if features is not None:
            data = self._merge_features(signals, features)
        else:
            data = signals.copy()

        # Apply transforms
        if self.transformer is not None and self.transformer._fitted:
            data = self.transformer.transform(data)

        return data

    def fit_transform(
        self,
        signals: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(signals, features).transform(signals, features)

    def _merge_features(
        self,
        signals: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge features with signals."""
        # Determine merge keys
        signal_date_col = "week_monday" if "week_monday" in signals.columns else "signal_date"

        # Get relevant feature columns
        feature_cols = ["symbol", "date"] + [
            f for f in self.sizing_features if f in features.columns
        ]
        features_subset = features[feature_cols].copy()

        # Merge
        merged = signals.merge(
            features_subset,
            left_on=["symbol", signal_date_col],
            right_on=["symbol", "date"],
            how="left",
            suffixes=("", "_feat"),
        )

        # Drop duplicate date column
        if "date" in merged.columns and signal_date_col != "date":
            merged = merged.drop(columns=["date"])

        return merged

    def get_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Get volatility series from data."""
        if self.volatility_col in data.columns:
            return data[self.volatility_col]
        elif "atr_pct" in data.columns:
            return data["atr_pct"]
        else:
            # Default estimate
            return pd.Series(0.02, index=data.index)

    def get_liquidity(self, data: pd.DataFrame) -> pd.Series:
        """Get liquidity series from data."""
        if self.liquidity_col in data.columns:
            return data[self.liquidity_col]
        elif "dollar_volume" in data.columns:
            return data["dollar_volume"]
        else:
            # Default (no liquidity adjustment)
            return pd.Series(1.0, index=data.index)


def get_sizing_features(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience function to get sizing features.

    Args:
        signals: Signals DataFrame.
        features: Features DataFrame.
        feature_list: Optional custom feature list.

    Returns:
        DataFrame with sizing features merged.
    """
    if feature_list is None:
        feature_list = SIZING_FEATURES

    pipeline = SizingFeaturePipeline(sizing_features=feature_list)
    return pipeline.fit_transform(signals, features)


def compute_regime_features(
    signals: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute regime features for adaptive sizing.

    Args:
        signals: Signals DataFrame with probability scores.
        vix_data: Optional VIX data for regime detection.

    Returns:
        Signals with regime features added.
    """
    signals = signals.copy()

    # VIX regime (if available)
    if "vix_regime" in signals.columns:
        pass  # Already have it
    elif vix_data is not None:
        # Merge VIX regime
        signals = signals.merge(
            vix_data[["date", "vix_regime"]],
            left_on="week_monday",
            right_on="date",
            how="left",
        )

    # Model confidence regime
    if "probability" in signals.columns:
        signals["confidence_regime"] = pd.cut(
            signals["probability"],
            bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
            labels=["very_low", "low", "medium", "high", "very_high"],
        )

    # Trend regime (if available)
    if "trend_score_sign" in signals.columns:
        signals["trend_regime"] = signals["trend_score_sign"].map({
            -1: "bearish",
            0: "neutral",
            1: "bullish",
        })

    return signals
