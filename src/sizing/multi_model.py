"""Multi-model position sizing engine.

Implements a sizing engine that:
1. Computes candidate weights from 4 models (LONG_NORMAL, LONG_PARABOLIC,
   SHORT_NORMAL, SHORT_PARABOLIC)
2. Combines signals using configurable policy (mode_priority or blend)
3. Handles long/short direction and netting
4. Applies portfolio constraints

The engine supports TPE optimization of sizing parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import (
    ModelType,
    CombinePolicy,
    NettingPolicy,
    MultiModelSizingConfig,
    MonotoneSizingParams,
)
from .predictions import get_prob_column


@dataclass
class WeightResult:
    """Result of weight computation for a single symbol.

    Attributes:
        symbol: Stock symbol.
        final_weight: Final weight after combining and netting.
        direction: +1 for long, -1 for short.
        contributing_model: Model that contributed to final weight.
        model_weights: Raw weights from each model before combining.
        edge_scores: Edge scores from each model.
    """
    symbol: str
    final_weight: float
    direction: int
    contributing_model: Optional[ModelType]
    model_weights: Dict[ModelType, float]
    edge_scores: Dict[ModelType, float]


def compute_edge_score(
    probability: float,
    intercept: float,
) -> float:
    """Compute edge score for a probability.

    Edge score measures how far the probability is above the threshold.
    Higher edge = stronger signal.

    Args:
        probability: Model probability in [0, 1].
        intercept: Probability threshold.

    Returns:
        Edge score (can be negative if below threshold).
    """
    return probability - intercept


def compute_raw_weight(
    probability: float,
    params: MonotoneSizingParams,
) -> float:
    """Compute raw weight from probability using monotone sizing.

    Formula: raw = exposure_mult * clip(slope * (p - intercept), 0, max_weight)

    Args:
        probability: Model probability in [0, 1].
        params: Sizing parameters.

    Returns:
        Raw weight (always non-negative).
    """
    if np.isnan(probability):
        return 0.0

    # Linear mapping with threshold
    raw = params.slope * (probability - params.intercept)

    # Clip to valid range
    raw = np.clip(raw, 0, params.max_weight)

    # Apply exposure multiplier
    raw = raw * params.exposure_mult

    # Zero out if below minimum
    if raw < params.min_weight:
        raw = 0.0

    return raw


def compute_model_weights(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Compute raw weights for each model.

    Args:
        signals: DataFrame with probability columns for each model.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with weight columns (w_<model>) for each model.
    """
    result = signals.copy()
    models = config.get_model_types()

    for model in models:
        prob_col = get_prob_column(model)
        if prob_col not in signals.columns:
            continue

        # Get sizing params for this model
        params = config.get_sizing_params_for_model(model)

        # Compute weights (vectorized)
        probs = signals[prob_col].fillna(0.5)
        raw_weights = params.slope * (probs - params.intercept)
        raw_weights = np.clip(raw_weights, 0, params.max_weight)
        raw_weights = raw_weights * params.exposure_mult
        raw_weights[raw_weights < params.min_weight] = 0.0

        # Apply direction sign (short models get negative weights)
        raw_weights = raw_weights * model.direction_sign

        result[f"w_{model.value}"] = raw_weights

    return result


def compute_edge_scores(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Compute edge scores for each model.

    Args:
        signals: DataFrame with probability columns.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with edge score columns (edge_<model>) for each model.
    """
    result = signals.copy()
    models = config.get_model_types()

    for model in models:
        prob_col = get_prob_column(model)
        if prob_col not in signals.columns:
            continue

        params = config.get_sizing_params_for_model(model)
        probs = signals[prob_col].fillna(0.5)
        edge = probs - params.intercept

        result[f"edge_{model.value}"] = edge

    return result


def combine_model_weights(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Combine weights from multiple models into final weights.

    Implements two combination policies:
    - mode_priority: Pick model with highest edge score per symbol
    - blend: Weighted blend of all models

    Args:
        signals: DataFrame with model weights and edge scores.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with combined_weight and contributing_model columns.
    """
    result = signals.copy()
    models = config.get_model_types()
    policy = config.get_combine_policy()
    netting = config.get_netting_policy()

    # Get weight and edge column names
    weight_cols = {m: f"w_{m.value}" for m in models if f"w_{m.value}" in signals.columns}
    edge_cols = {m: f"edge_{m.value}" for m in models if f"edge_{m.value}" in signals.columns}

    if policy == CombinePolicy.MODE_PRIORITY:
        # For each row, pick the model with highest absolute edge
        combined_weights = []
        contributing_models = []

        for idx in signals.index:
            # Get edges for all models
            edges = {}
            weights = {}
            for model in models:
                if model in edge_cols:
                    edges[model] = abs(signals.loc[idx, edge_cols[model]])
                    weights[model] = signals.loc[idx, weight_cols[model]]

            if not edges:
                combined_weights.append(0.0)
                contributing_models.append(None)
                continue

            # Find best model
            best_model = max(edges, key=edges.get)
            best_weight = weights[best_model]

            combined_weights.append(best_weight)
            contributing_models.append(best_model.value)

        result["combined_weight"] = combined_weights
        result["contributing_model"] = contributing_models

    elif policy == CombinePolicy.BLEND:
        # Blend all model weights (with edge-based weighting)
        combined_weights = np.zeros(len(signals))

        for model in models:
            if model not in weight_cols:
                continue

            w = signals[weight_cols[model]].values
            combined_weights += w

        # Normalize by number of models
        n_models = len([m for m in models if m in weight_cols])
        if n_models > 0:
            combined_weights = combined_weights / n_models

        result["combined_weight"] = combined_weights
        result["contributing_model"] = "blend"

    # Apply netting for conflicting long/short signals
    result = _apply_netting(result, config)

    return result


def _apply_netting(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Apply netting policy for conflicting long/short signals.

    For each symbol, if both long and short weights exist:
    - strongest: Pick direction with larger absolute weight
    - net: Subtract short from long weight

    Args:
        signals: DataFrame with combined weights.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with netted weights.
    """
    result = signals.copy()
    netting = config.get_netting_policy()
    models = config.get_model_types()

    # Get long and short weight columns
    long_cols = [f"w_{m.value}" for m in models if m.is_long and f"w_{m.value}" in signals.columns]
    short_cols = [f"w_{m.value}" for m in models if m.is_short and f"w_{m.value}" in signals.columns]

    if not long_cols or not short_cols:
        return result  # Only one direction, no netting needed

    # Compute aggregate long and short weights
    long_weight = signals[long_cols].sum(axis=1).abs() if long_cols else 0
    short_weight = signals[short_cols].sum(axis=1).abs() if short_cols else 0

    if netting == NettingPolicy.STRONGEST:
        # Pick direction with stronger signal
        use_long = long_weight >= short_weight
        netted = result["combined_weight"].copy()

        # Where short is stronger, negate
        for idx in signals.index:
            if not use_long.loc[idx]:
                netted.loc[idx] = -abs(netted.loc[idx])
            else:
                netted.loc[idx] = abs(netted.loc[idx])

        result["combined_weight"] = netted

    elif netting == NettingPolicy.NET:
        # Net long and short weights
        netted = long_weight - short_weight
        result["combined_weight"] = netted

    # Set direction based on sign
    result["direction"] = np.sign(result["combined_weight"]).fillna(0).astype(int)

    return result


def apply_portfolio_constraints(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
    previous_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Apply portfolio constraints to combined weights.

    Constraints applied in order:
    1. Individual weight clipping (max_weight_per_name)
    2. Min weight filtering
    3. Max positions
    4. Gross exposure scaling
    5. Net exposure constraint

    Args:
        signals: DataFrame with combined_weight column.
        config: Multi-model sizing configuration.
        previous_weights: Previous weights for turnover penalty.

    Returns:
        DataFrame with final_weight column.
    """
    result = signals.copy()
    weights = result["combined_weight"].copy()

    # 1. Clip individual weights
    max_abs = config.max_weight_per_name
    weights = weights.clip(-max_abs, max_abs)

    # 2. Filter by minimum weight
    weights[weights.abs() < config.min_weight] = 0

    # 3. Apply max positions
    if config.max_positions is not None:
        n_positions = (weights != 0).sum()
        if n_positions > config.max_positions:
            # Keep top positions by absolute weight
            threshold = weights.abs().nlargest(config.max_positions).min()
            weights[weights.abs() < threshold] = 0

    # 4. Scale to max gross exposure
    gross = weights.abs().sum()
    if gross > config.max_gross_exposure:
        scale = config.max_gross_exposure / gross
        weights = weights * scale

    # 5. Enforce net exposure constraint
    net = weights.sum()
    if abs(net) > config.max_net_exposure:
        # Scale down to meet net exposure
        if net > 0:
            # Too long: scale down longs
            long_mask = weights > 0
            excess = net - config.max_net_exposure
            long_sum = weights[long_mask].sum()
            if long_sum > 0:
                scale = (long_sum - excess) / long_sum
                weights[long_mask] = weights[long_mask] * scale
        else:
            # Too short: scale down shorts
            short_mask = weights < 0
            excess = abs(net) - config.max_net_exposure
            short_sum = weights[short_mask].abs().sum()
            if short_sum > 0:
                scale = (short_sum - excess) / short_sum
                weights[short_mask] = weights[short_mask] * scale

    # Re-clip after adjustments
    weights = weights.clip(-max_abs, max_abs)

    result["final_weight"] = weights
    result["gross_exposure"] = weights.abs().sum()
    result["net_exposure"] = weights.sum()
    result["n_longs"] = (weights > 0).sum()
    result["n_shorts"] = (weights < 0).sum()

    return result


class MultiModelSizingEngine:
    """Engine for multi-model position sizing.

    Coordinates the full sizing workflow:
    1. Load predictions for all models
    2. Compute raw weights per model
    3. Compute edge scores
    4. Combine using configured policy
    5. Apply netting
    6. Apply regime gating (if enabled)
    7. Apply portfolio constraints

    Attributes:
        config: Multi-model sizing configuration.
    """

    def __init__(self, config: MultiModelSizingConfig):
        """Initialize engine with configuration.

        Args:
            config: Multi-model sizing configuration.
        """
        self.config = config
        self.models = config.get_model_types()

    def compute_weights(
        self,
        signals: pd.DataFrame,
        regime_features: Optional[pd.DataFrame] = None,
        previous_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute final position weights from signals.

        Main entry point for weight computation.

        Args:
            signals: DataFrame with probability columns for each model.
            regime_features: Optional regime features for gating.
            previous_weights: Previous weights (symbol-indexed) for turnover.

        Returns:
            DataFrame with final weights and diagnostics.
        """
        # Step 1: Compute raw weights for each model
        result = compute_model_weights(signals, self.config)

        # Step 2: Compute edge scores
        result = compute_edge_scores(result, self.config)

        # Step 3: Combine model weights
        result = combine_model_weights(result, self.config)

        # Step 4: Apply regime gating (imported lazily to avoid circular import)
        if self.config.regime_gating.enabled and regime_features is not None:
            from .regime_gating import apply_regime_gating
            result = apply_regime_gating(result, regime_features, self.config.regime_gating)

        # Step 5: Apply portfolio constraints
        result = apply_portfolio_constraints(result, self.config, previous_weights)

        return result

    def compute_weekly_weights(
        self,
        signals: pd.DataFrame,
        regime_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute weights for each rebalance week.

        Groups signals by week_monday and computes weights per week.

        Args:
            signals: DataFrame with week_monday and probability columns.
            regime_features: Optional regime features (date-indexed).

        Returns:
            DataFrame with weekly weights and diagnostics.
        """
        if "week_monday" not in signals.columns:
            raise ValueError("signals must have week_monday column")

        weeks = signals["week_monday"].unique()
        results = []
        previous_weights = None

        for week in sorted(weeks):
            week_signals = signals[signals["week_monday"] == week].copy()

            # Get regime features for this week if available
            week_regime = None
            if regime_features is not None:
                if "date" in regime_features.columns:
                    week_regime = regime_features[
                        regime_features["date"] == week
                    ].iloc[0:1] if len(regime_features[regime_features["date"] == week]) > 0 else None
                elif week in regime_features.index:
                    week_regime = regime_features.loc[[week]]

            # Compute weights
            week_result = self.compute_weights(
                week_signals,
                regime_features=week_regime,
                previous_weights=previous_weights,
            )

            # Update previous weights for next week
            if "symbol" in week_result.columns and "final_weight" in week_result.columns:
                previous_weights = week_result.set_index("symbol")["final_weight"]

            results.append(week_result)

        return pd.concat(results, ignore_index=True)

    def get_diagnostics(
        self,
        weighted_signals: pd.DataFrame,
    ) -> Dict:
        """Get diagnostics from weighted signals.

        Args:
            weighted_signals: Output from compute_weights.

        Returns:
            Dict with diagnostic metrics.
        """
        diag = {
            "n_signals": len(weighted_signals),
            "n_positions": (weighted_signals["final_weight"] != 0).sum(),
            "n_longs": (weighted_signals["final_weight"] > 0).sum(),
            "n_shorts": (weighted_signals["final_weight"] < 0).sum(),
        }

        if "gross_exposure" in weighted_signals.columns:
            diag["gross_exposure"] = weighted_signals["gross_exposure"].iloc[0]
        if "net_exposure" in weighted_signals.columns:
            diag["net_exposure"] = weighted_signals["net_exposure"].iloc[0]

        # Model contribution counts
        if "contributing_model" in weighted_signals.columns:
            contrib = weighted_signals["contributing_model"].value_counts()
            diag["model_contributions"] = contrib.to_dict()

        return diag


def create_multi_model_engine(
    config: Optional[MultiModelSizingConfig] = None,
    **kwargs,
) -> MultiModelSizingEngine:
    """Factory function to create sizing engine.

    Args:
        config: Pre-built configuration (optional).
        **kwargs: Parameters to override in default config.

    Returns:
        Configured MultiModelSizingEngine.
    """
    if config is None:
        config = MultiModelSizingConfig(**kwargs)
    return MultiModelSizingEngine(config)
