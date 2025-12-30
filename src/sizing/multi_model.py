"""Multi-model position sizing engine with full justification.

Implements a sizing engine that:
1. Computes candidate weights from 4 models (LONG_NORMAL, LONG_PARABOLIC,
   SHORT_NORMAL, SHORT_PARABOLIC)
2. Combines signals using configurable policy (mode_priority or blend)
3. Handles long/short direction and netting
4. Applies direction-aware regime gating
5. Applies portfolio constraints
6. Produces full justification for every decision (dashboard-ready)

The engine is deterministic and auditable.
"""

from dataclasses import dataclass, field
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


# Justification column prefixes
JUSTIFY_PREFIX = "j_"


def compute_model_weights_with_justification(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Compute raw weights for each model with full justification.

    Adds justification columns for:
    - Per-model intercepts used (includes short/parabolic offsets)
    - Per-model slopes used
    - Per-model raw weights before direction sign

    Args:
        signals: DataFrame with probability columns for each model.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with weight and justification columns.
    """
    result = signals.copy()
    models = config.get_model_types()

    for model in models:
        prob_col = get_prob_column(model)
        if prob_col not in signals.columns:
            continue

        # Get sizing params for this model (includes short selectivity and parabolic offset)
        params = config.get_sizing_params_for_model(model)

        # Store params used for justification
        result[f"{JUSTIFY_PREFIX}intercept_{model.value}"] = params.intercept
        result[f"{JUSTIFY_PREFIX}slope_{model.value}"] = params.slope
        result[f"{JUSTIFY_PREFIX}max_weight_{model.value}"] = params.max_weight

        # Compute weights (vectorized)
        probs = signals[prob_col].fillna(0.5)
        raw_weights = params.slope * (probs - params.intercept)
        raw_weights = np.clip(raw_weights, 0, params.max_weight)
        raw_weights = raw_weights * params.exposure_mult

        # Store pre-direction weight for justification
        result[f"{JUSTIFY_PREFIX}raw_w_{model.value}"] = raw_weights.copy()

        # Apply min weight filter
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

    Edge = probability - intercept (model-specific intercept).

    Args:
        signals: DataFrame with probability columns.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with edge score columns.
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


def combine_model_weights_with_justification(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Combine weights from multiple models with justification.

    Adds justification columns for:
    - All edge scores for ranking
    - Winning model and its edge
    - Blend weights if using blend policy

    Args:
        signals: DataFrame with model weights and edge scores.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with combined_weight, contributing_model, and justification.
    """
    result = signals.copy()
    models = config.get_model_types()
    policy = config.get_combine_policy()

    # Get weight and edge column names
    weight_cols = {m: f"w_{m.value}" for m in models if f"w_{m.value}" in signals.columns}
    edge_cols = {m: f"edge_{m.value}" for m in models if f"edge_{m.value}" in signals.columns}

    if policy == CombinePolicy.MODE_PRIORITY:
        # For each row, pick the model with highest absolute edge
        combined_weights = []
        contributing_models = []
        winning_edges = []
        edge_rankings = []

        for idx in signals.index:
            # Get edges for all models
            edges = {}
            weights = {}
            for model in models:
                if model in edge_cols and model in weight_cols:
                    edges[model] = signals.loc[idx, edge_cols[model]]
                    weights[model] = signals.loc[idx, weight_cols[model]]

            if not edges:
                combined_weights.append(0.0)
                contributing_models.append(None)
                winning_edges.append(0.0)
                edge_rankings.append("")
                continue

            # Sort by absolute edge
            sorted_models = sorted(edges.keys(), key=lambda m: abs(edges[m]), reverse=True)
            best_model = sorted_models[0]
            best_weight = weights[best_model]
            best_edge = edges[best_model]

            combined_weights.append(best_weight)
            contributing_models.append(best_model.value)
            winning_edges.append(best_edge)

            # Create ranking string for justification
            ranking = ";".join([f"{m.value}:{edges[m]:.3f}" for m in sorted_models])
            edge_rankings.append(ranking)

        result["combined_weight"] = combined_weights
        result["contributing_model"] = contributing_models
        result[f"{JUSTIFY_PREFIX}winning_edge"] = winning_edges
        result[f"{JUSTIFY_PREFIX}edge_ranking"] = edge_rankings

    elif policy == CombinePolicy.BLEND:
        # Average all model weights
        combined_weights = np.zeros(len(signals))
        n_models = 0

        for model in models:
            if model not in weight_cols:
                continue
            w = signals[weight_cols[model]].values
            combined_weights += w
            n_models += 1

        if n_models > 0:
            combined_weights = combined_weights / n_models

        result["combined_weight"] = combined_weights
        result["contributing_model"] = "blend"
        result[f"{JUSTIFY_PREFIX}blend_n_models"] = n_models

    # Apply netting for conflicting long/short signals
    result = _apply_netting_with_justification(result, config)

    return result


def _apply_netting_with_justification(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
) -> pd.DataFrame:
    """Apply netting policy with full justification.

    Adds justification columns for:
    - Aggregate long and short weights
    - Chosen direction
    - Netting result

    Args:
        signals: DataFrame with combined weights.
        config: Multi-model sizing configuration.

    Returns:
        DataFrame with netted weights and justification.
    """
    result = signals.copy()
    netting = config.get_netting_policy()
    models = config.get_model_types()

    # Get long and short weight columns
    long_cols = [f"w_{m.value}" for m in models if m.is_long and f"w_{m.value}" in signals.columns]
    short_cols = [f"w_{m.value}" for m in models if m.is_short and f"w_{m.value}" in signals.columns]

    # Compute aggregate weights for justification
    long_sum = signals[long_cols].sum(axis=1) if long_cols else pd.Series(0, index=signals.index)
    short_sum = signals[short_cols].sum(axis=1).abs() if short_cols else pd.Series(0, index=signals.index)

    result[f"{JUSTIFY_PREFIX}long_sum"] = long_sum
    result[f"{JUSTIFY_PREFIX}short_sum"] = short_sum

    if not long_cols or not short_cols:
        # Only one direction, no netting needed
        result["direction"] = np.sign(result["combined_weight"]).fillna(0).astype(int)
        result[f"{JUSTIFY_PREFIX}netting_action"] = "single_direction"
        return result

    if netting == NettingPolicy.STRONGEST:
        # Pick direction with stronger signal
        use_long = long_sum >= short_sum
        netted = result["combined_weight"].copy()

        # Adjust direction based on which is stronger
        netted = np.where(use_long, np.abs(netted), -np.abs(netted))
        result["combined_weight"] = netted
        result[f"{JUSTIFY_PREFIX}netting_action"] = np.where(use_long, "chose_long", "chose_short")

    elif netting == NettingPolicy.NET:
        # Net long and short weights
        netted = long_sum - short_sum
        result["combined_weight"] = netted
        result[f"{JUSTIFY_PREFIX}netting_action"] = "netted"

    # Set direction based on sign
    result["direction"] = np.sign(result["combined_weight"]).fillna(0).astype(int)

    return result


def apply_portfolio_constraints_with_justification(
    signals: pd.DataFrame,
    config: MultiModelSizingConfig,
    previous_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Apply portfolio constraints with full justification.

    Tracks which constraints fired and by how much.

    Args:
        signals: DataFrame with combined_weight column.
        config: Multi-model sizing configuration.
        previous_weights: Previous weights for turnover calculation.

    Returns:
        DataFrame with final_weight and constraint justification.
    """
    result = signals.copy()
    weights = result["combined_weight"].copy()
    original_weights = weights.copy()

    # Track constraint applications
    constraint_log = []

    # 1. Clip individual weights
    max_abs = config.max_weight_per_name
    pre_clip = weights.copy()
    weights = weights.clip(-max_abs, max_abs)
    n_clipped = (pre_clip != weights).sum()
    if n_clipped > 0:
        constraint_log.append(f"clip:{n_clipped}")
    result[f"{JUSTIFY_PREFIX}weight_clipped"] = (pre_clip != weights)

    # 2. Filter by minimum weight
    pre_min = weights.copy()
    weights[weights.abs() < config.min_weight] = 0
    n_filtered = ((pre_min != 0) & (weights == 0)).sum()
    if n_filtered > 0:
        constraint_log.append(f"min_filter:{n_filtered}")
    result[f"{JUSTIFY_PREFIX}min_weight_filtered"] = ((pre_min != 0) & (weights == 0))

    # 3. Apply max positions
    if config.max_positions is not None:
        n_positions = (weights != 0).sum()
        if n_positions > config.max_positions:
            pre_pos = weights.copy()
            threshold = weights.abs().nlargest(config.max_positions).min()
            weights[weights.abs() < threshold] = 0
            n_dropped = ((pre_pos != 0) & (weights == 0)).sum()
            constraint_log.append(f"max_pos:{n_dropped}")
            result[f"{JUSTIFY_PREFIX}max_pos_filtered"] = ((pre_pos != 0) & (weights == 0))
        else:
            result[f"{JUSTIFY_PREFIX}max_pos_filtered"] = False
    else:
        result[f"{JUSTIFY_PREFIX}max_pos_filtered"] = False

    # 4. Scale to max gross exposure
    gross = weights.abs().sum()
    if gross > config.max_gross_exposure:
        scale = config.max_gross_exposure / gross
        weights = weights * scale
        constraint_log.append(f"gross_scale:{scale:.3f}")
        result[f"{JUSTIFY_PREFIX}gross_scale"] = scale
    else:
        result[f"{JUSTIFY_PREFIX}gross_scale"] = 1.0

    # 5. Enforce net exposure constraint
    net = weights.sum()
    if abs(net) > config.max_net_exposure:
        pre_net = weights.copy()
        if net > 0:
            # Too long: scale down longs
            long_mask = weights > 0
            excess = net - config.max_net_exposure
            long_sum = weights[long_mask].sum()
            if long_sum > 0:
                scale = (long_sum - excess) / long_sum
                weights[long_mask] = weights[long_mask] * scale
                constraint_log.append(f"net_long_scale:{scale:.3f}")
                result[f"{JUSTIFY_PREFIX}net_scale"] = scale
        else:
            # Too short: scale down shorts
            short_mask = weights < 0
            excess = abs(net) - config.max_net_exposure
            short_sum = weights[short_mask].abs().sum()
            if short_sum > 0:
                scale = (short_sum - excess) / short_sum
                weights[short_mask] = weights[short_mask] * scale
                constraint_log.append(f"net_short_scale:{scale:.3f}")
                result[f"{JUSTIFY_PREFIX}net_scale"] = scale
    else:
        result[f"{JUSTIFY_PREFIX}net_scale"] = 1.0

    # Re-clip after adjustments
    weights = weights.clip(-max_abs, max_abs)

    # Store results
    result["final_weight"] = weights
    result["gross_exposure"] = weights.abs().sum()
    result["net_exposure"] = weights.sum()
    result["n_longs"] = (weights > 0).sum()
    result["n_shorts"] = (weights < 0).sum()
    result[f"{JUSTIFY_PREFIX}constraints_applied"] = ";".join(constraint_log) if constraint_log else "none"

    # Compute turnover if previous weights available
    if previous_weights is not None and "symbol" in result.columns:
        # Align by symbol
        turnover = 0.0
        for idx, row in result.iterrows():
            sym = row["symbol"]
            new_w = row["final_weight"]
            old_w = previous_weights.get(sym, 0.0)
            turnover += abs(new_w - old_w)
        result["turnover"] = turnover
    else:
        result["turnover"] = weights.abs().sum()  # Assume full rebalance

    return result


class MultiModelSizingEngine:
    """Engine for multi-model position sizing with full justification.

    Coordinates the full sizing workflow:
    1. Compute raw weights per model (with per-model params)
    2. Compute edge scores
    3. Combine using configured policy
    4. Apply netting
    5. Apply direction-aware regime gating
    6. Apply portfolio constraints

    All steps produce justification columns for dashboard display.

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
        include_justification: bool = True,
    ) -> pd.DataFrame:
        """Compute final position weights with full justification.

        Main entry point for weight computation.

        Args:
            signals: DataFrame with probability columns for each model.
            regime_features: Optional regime features for gating.
            previous_weights: Previous weights for turnover calculation.
            include_justification: Whether to include justification columns.

        Returns:
            DataFrame with final weights and justification columns.
        """
        # Step 1: Compute raw weights for each model
        result = compute_model_weights_with_justification(signals, self.config)

        # Step 2: Compute edge scores
        result = compute_edge_scores(result, self.config)

        # Step 3: Combine model weights
        result = combine_model_weights_with_justification(result, self.config)

        # Step 4: Apply regime gating
        if self.config.regime_gating.enabled:
            if regime_features is not None:
                from .regime_gating import apply_regime_gating
                result = apply_regime_gating(result, regime_features, self.config.regime_gating)
            else:
                # Regime features in signals (already joined)
                from .regime_gating import apply_regime_gating_vectorized
                result = apply_regime_gating_vectorized(result, self.config.regime_gating)
        else:
            result["gating_long_mult"] = 1.0
            result["gating_short_mult"] = 1.0
            result["gating_triggered"] = False

        # Step 5: Apply portfolio constraints
        result = apply_portfolio_constraints_with_justification(
            result, self.config, previous_weights
        )

        # Optionally remove justification columns
        if not include_justification:
            justify_cols = [c for c in result.columns if c.startswith(JUSTIFY_PREFIX)]
            result = result.drop(columns=justify_cols)

        return result

    def compute_weights_for_date(
        self,
        signals: pd.DataFrame,
        regime_row: Optional[pd.Series] = None,
        previous_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute weights for a single rebalance date.

        Convenience method for single-date processing.

        Args:
            signals: DataFrame with signals for one date.
            regime_row: Single row of regime features.
            previous_weights: Previous weights.

        Returns:
            DataFrame with weights and justification.
        """
        regime_df = None
        if regime_row is not None:
            regime_df = pd.DataFrame([regime_row])

        return self.compute_weights(
            signals,
            regime_features=regime_df,
            previous_weights=previous_weights,
        )

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
            DataFrame with weekly weights and justification.
        """
        if "week_monday" not in signals.columns:
            raise ValueError("signals must have week_monday column")

        weeks = sorted(signals["week_monday"].unique())
        results = []
        previous_weights = None

        for week in weeks:
            week_signals = signals[signals["week_monday"] == week].copy()

            # Get regime features for this week
            week_regime = None
            if regime_features is not None:
                if "date" in regime_features.columns:
                    mask = regime_features["date"] == week
                    if mask.sum() > 0:
                        week_regime = regime_features[mask].iloc[0:1]
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
        """Get summary diagnostics from weighted signals.

        Args:
            weighted_signals: Output from compute_weights.

        Returns:
            Dict with diagnostic metrics.
        """
        diag = {
            "n_signals": len(weighted_signals),
            "n_positions": int((weighted_signals["final_weight"] != 0).sum()),
            "n_longs": int((weighted_signals["final_weight"] > 0).sum()),
            "n_shorts": int((weighted_signals["final_weight"] < 0).sum()),
        }

        if "gross_exposure" in weighted_signals.columns:
            diag["gross_exposure"] = float(weighted_signals["gross_exposure"].iloc[0])
        if "net_exposure" in weighted_signals.columns:
            diag["net_exposure"] = float(weighted_signals["net_exposure"].iloc[0])

        # Model contribution counts
        if "contributing_model" in weighted_signals.columns:
            contrib = weighted_signals["contributing_model"].value_counts()
            diag["model_contributions"] = contrib.to_dict()

        # Gating stats
        if "gating_triggered" in weighted_signals.columns:
            diag["gating_triggered"] = bool(weighted_signals["gating_triggered"].any())

        # Short stats
        short_mask = weighted_signals["final_weight"] < 0
        if short_mask.sum() > 0:
            diag["short_avg_weight"] = float(weighted_signals.loc[short_mask, "final_weight"].abs().mean())
            diag["short_total_weight"] = float(weighted_signals.loc[short_mask, "final_weight"].abs().sum())

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


def get_justification_columns(df: pd.DataFrame) -> List[str]:
    """Get list of justification columns in a DataFrame.

    Args:
        df: DataFrame with justification columns.

    Returns:
        List of justification column names.
    """
    return [c for c in df.columns if c.startswith(JUSTIFY_PREFIX)]


def create_decision_log(
    weighted_signals: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    """Create a decision log DataFrame for artifact storage.

    Selects the most important columns for dashboard display
    and debugging.

    Args:
        weighted_signals: Output from compute_weights.
        run_id: Run identifier for this computation.

    Returns:
        DataFrame with essential decision log columns.
    """
    # Essential columns for decision log
    essential_cols = [
        "date", "symbol", "final_weight", "direction",
        "contributing_model", "gross_exposure", "net_exposure",
        "gating_triggered", "gating_long_mult", "gating_short_mult",
    ]

    # Probability columns
    prob_cols = [c for c in weighted_signals.columns if c.startswith("p_")]

    # Weight columns
    weight_cols = [c for c in weighted_signals.columns if c.startswith("w_")]

    # Edge columns
    edge_cols = [c for c in weighted_signals.columns if c.startswith("edge_")]

    # Justification columns
    justify_cols = get_justification_columns(weighted_signals)

    # Regime columns
    regime_cols = [c for c in weighted_signals.columns if c.startswith("regime_")]

    # Gating columns
    gating_cols = [c for c in weighted_signals.columns if c.startswith("gating_")]

    # Combine all
    all_cols = (
        essential_cols +
        prob_cols +
        weight_cols +
        edge_cols +
        justify_cols +
        regime_cols +
        gating_cols
    )

    # Filter to columns that exist
    available_cols = [c for c in all_cols if c in weighted_signals.columns]

    log = weighted_signals[available_cols].copy()
    log["run_id"] = run_id

    return log
