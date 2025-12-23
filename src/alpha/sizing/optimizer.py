"""Sizing optimizer using Optuna.

Optimizes sizing parameters using purged walk-forward CV,
ensuring no leakage between training and validation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

from ..config import SizingConfig, SizingMethod, AlphaConfig
from ..cv import WeeklySignalCV
from .rules import (
    BaseSizingRule,
    MonotoneProbabilitySizing,
    RankBucketSizing,
    ThresholdConfidenceSizing,
    create_sizing_rule,
)
from .constraints import PortfolioConstraints, apply_constraints


@dataclass
class OptimizationResult:
    """Result of sizing optimization."""
    best_params: Dict[str, float]
    best_value: float
    study: Any  # optuna.Study
    param_importance: Dict[str, float]
    trials_df: pd.DataFrame
    cv_metrics: List[Dict[str, float]]


def create_sizing_rule_from_trial(
    trial: Trial,
    method: SizingMethod,
    constraints: PortfolioConstraints,
) -> BaseSizingRule:
    """Create sizing rule with parameters from Optuna trial.

    Args:
        trial: Optuna trial object.
        method: Sizing method to use.
        constraints: Portfolio constraints.

    Returns:
        Configured sizing rule.
    """
    if method == SizingMethod.MONOTONE_PROBABILITY:
        return MonotoneProbabilitySizing(
            slope=trial.suggest_float("slope", 1.0, 5.0),
            intercept=trial.suggest_float("intercept", 0.3, 0.6),
            max_weight=constraints.max_name_weight,
            min_weight=constraints.min_name_weight,
        )

    elif method == SizingMethod.RANK_BUCKET:
        n_buckets = trial.suggest_int("n_buckets", 3, 7)
        # Monotonically decreasing weights
        bucket_weights = []
        prev_weight = 1.0
        for i in range(n_buckets):
            w = trial.suggest_float(f"bucket_weight_{i}", 0.1, prev_weight)
            bucket_weights.append(w)
            prev_weight = w

        return RankBucketSizing(
            n_buckets=n_buckets,
            bucket_weights=bucket_weights,
            max_weight=constraints.max_name_weight,
        )

    elif method == SizingMethod.THRESHOLD_CONFIDENCE:
        return ThresholdConfidenceSizing(
            entry_threshold=trial.suggest_float("entry_threshold", 0.45, 0.7),
            confidence_scale=trial.suggest_float("confidence_scale", 0.5, 2.0),
            max_weight=constraints.max_name_weight,
            min_weight=constraints.min_name_weight,
            regime_multipliers={
                "low_vix": trial.suggest_float("regime_low_vix", 0.8, 1.5),
                "normal_vix": 1.0,
                "high_vix": trial.suggest_float("regime_high_vix", 0.5, 1.2),
            },
        )

    else:
        raise ValueError(f"Unsupported method for optimization: {method}")


def compute_backtest_objective(
    signals: pd.DataFrame,
    weights: pd.Series,
    turnover_penalty: float = 0.0,
    previous_weights: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute objective metrics from weights and outcomes.

    Args:
        signals: Signals with actual returns.
        weights: Position weights.
        turnover_penalty: Penalty per unit turnover.
        previous_weights: Previous period weights for turnover.

    Returns:
        Dict of metrics.
    """
    # Get returns
    returns = signals["actual_return"].values
    w = weights.values

    # Portfolio return (weighted average)
    portfolio_return = np.sum(w * returns)

    # Win rate (weighted)
    wins = (returns > 0).astype(float)
    if w.sum() > 0:
        hit_rate = np.sum(w * wins) / w.sum()
    else:
        hit_rate = 0.0

    # Gross exposure
    gross_exposure = np.abs(w).sum()

    # Number of positions
    n_positions = (w > 0).sum()

    # Turnover (if previous weights provided)
    turnover = 0.0
    if previous_weights is not None:
        # Map by symbol
        for i, idx in enumerate(signals.index):
            symbol = signals.loc[idx, "symbol"]
            old_w = previous_weights.get(symbol, 0.0)
            new_w = w[i]
            turnover += abs(new_w - old_w)
        turnover /= 2  # Two-sided

    # Penalized return
    penalized_return = portfolio_return - turnover_penalty * turnover

    return {
        "portfolio_return": portfolio_return,
        "penalized_return": penalized_return,
        "hit_rate": hit_rate,
        "gross_exposure": gross_exposure,
        "n_positions": n_positions,
        "turnover": turnover,
    }


class SizingOptimizer:
    """Optimizer for position sizing parameters.

    Uses Optuna to optimize sizing parameters with purged CV,
    ensuring objective is computed only on validation folds.

    Attributes:
        config: Alpha configuration.
        method: Sizing method to optimize.
        n_trials: Number of optimization trials.
        metric: Metric to optimize ('penalized_return', 'sharpe', etc.).
    """

    def __init__(
        self,
        config: AlphaConfig,
        method: Optional[SizingMethod] = None,
        n_trials: int = 100,
        metric: str = "penalized_return",
        random_state: int = 42,
    ):
        """Initialize optimizer.

        Args:
            config: Alpha configuration.
            method: Sizing method (defaults to config.sizing.method).
            n_trials: Number of optimization trials.
            metric: Metric to optimize.
            random_state: Random seed.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for SizingOptimizer")

        self.config = config
        self.method = method or config.sizing.method
        self.n_trials = n_trials
        self.metric = metric
        self.random_state = random_state

        self.constraints = PortfolioConstraints(
            max_gross_exposure=config.sizing.max_gross_exposure,
            max_name_weight=config.sizing.max_name_weight,
            min_name_weight=config.sizing.min_name_weight,
            cash_buffer=config.sizing.cash_buffer,
            sector_caps=config.sizing.sector_caps,
            max_positions=config.sizing.top_n,
        )

        self.cv = WeeklySignalCV(
            n_splits=config.cv.n_splits,
            min_train_weeks=config.cv.min_train_weeks,
            test_weeks=config.cv.test_weeks,
            purge_days=config.cv.purge_days,
            embargo_days=config.cv.embargo_days,
        )

    def _objective(
        self,
        trial: Trial,
        signals: pd.DataFrame,
    ) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial.
            signals: Full signals DataFrame.

        Returns:
            Objective value (higher is better).
        """
        # Suggest turnover penalty
        turnover_penalty = trial.suggest_float("turnover_penalty", 0.0, 0.01)

        # Create sizing rule
        sizing_rule = create_sizing_rule_from_trial(
            trial, self.method, self.constraints
        )

        # Suggest exposure multiplier
        exposure_mult = trial.suggest_float("exposure_mult", 0.5, 1.5)

        # Evaluate on CV folds
        fold_metrics = []
        previous_weights = None

        for train_idx, val_idx, fold_info in self.cv.split(signals):
            # Get validation signals
            val_signals = signals.iloc[val_idx].copy()

            # Compute raw weights
            raw_weights = sizing_rule.compute_weights(val_signals)

            # Apply exposure multiplier
            raw_weights = raw_weights * exposure_mult

            # Apply constraints
            constrained_weights = apply_constraints(
                raw_weights, val_signals, self.constraints
            )

            # Compute metrics
            metrics = compute_backtest_objective(
                val_signals,
                constrained_weights,
                turnover_penalty=turnover_penalty,
                previous_weights=previous_weights,
            )
            fold_metrics.append(metrics)

            # Update previous weights for next fold
            previous_weights = pd.Series({
                val_signals.loc[idx, "symbol"]: constrained_weights.loc[idx]
                for idx in constrained_weights.index if constrained_weights.loc[idx] > 0
            })

        # Aggregate across folds
        metric_values = [m[self.metric] for m in fold_metrics]
        mean_metric = np.mean(metric_values)

        # Penalize high variance across folds
        std_metric = np.std(metric_values)
        stability_penalty = 0.5 * std_metric

        return mean_metric - stability_penalty

    def optimize(
        self,
        signals: pd.DataFrame,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            signals: Prepared signals DataFrame with outcomes.
            show_progress: Whether to show progress bar.

        Returns:
            OptimizationResult with best parameters.
        """
        # Create study
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, signals),
            n_trials=self.n_trials,
            show_progress_bar=show_progress,
        )

        # Get parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
        except Exception:
            importance = {}

        # Create trials DataFrame
        trials_data = []
        for trial in study.trials:
            trial_data = trial.params.copy()
            trial_data["value"] = trial.value
            trial_data["state"] = trial.state.name
            trials_data.append(trial_data)
        trials_df = pd.DataFrame(trials_data)

        # Compute CV metrics for best params
        cv_metrics = self._get_cv_metrics(signals, study.best_params)

        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            study=study,
            param_importance=importance,
            trials_df=trials_df,
            cv_metrics=cv_metrics,
        )

    def _get_cv_metrics(
        self,
        signals: pd.DataFrame,
        params: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """Get detailed CV metrics for given parameters.

        Args:
            signals: Signals DataFrame.
            params: Sizing parameters.

        Returns:
            List of metrics per fold.
        """
        # Reconstruct sizing rule
        if self.method == SizingMethod.MONOTONE_PROBABILITY:
            sizing_rule = MonotoneProbabilitySizing(
                slope=params.get("slope", 2.0),
                intercept=params.get("intercept", 0.5),
                max_weight=self.constraints.max_name_weight,
                min_weight=self.constraints.min_name_weight,
            )
        elif self.method == SizingMethod.THRESHOLD_CONFIDENCE:
            sizing_rule = ThresholdConfidenceSizing(
                entry_threshold=params.get("entry_threshold", 0.5),
                confidence_scale=params.get("confidence_scale", 1.0),
                max_weight=self.constraints.max_name_weight,
                min_weight=self.constraints.min_name_weight,
            )
        else:
            sizing_rule = create_sizing_rule(self.config.sizing)

        exposure_mult = params.get("exposure_mult", 1.0)
        turnover_penalty = params.get("turnover_penalty", 0.0)

        fold_metrics = []
        previous_weights = None

        for train_idx, val_idx, fold_info in self.cv.split(signals):
            val_signals = signals.iloc[val_idx].copy()

            raw_weights = sizing_rule.compute_weights(val_signals)
            raw_weights = raw_weights * exposure_mult
            constrained_weights = apply_constraints(
                raw_weights, val_signals, self.constraints
            )

            metrics = compute_backtest_objective(
                val_signals,
                constrained_weights,
                turnover_penalty=turnover_penalty,
                previous_weights=previous_weights,
            )
            metrics["fold"] = fold_info.fold_idx
            metrics["test_start"] = str(pd.Timestamp(fold_info.test_start).date())
            metrics["test_end"] = str(pd.Timestamp(fold_info.test_end).date())
            fold_metrics.append(metrics)

            previous_weights = pd.Series({
                val_signals.loc[idx, "symbol"]: constrained_weights.loc[idx]
                for idx in constrained_weights.index if constrained_weights.loc[idx] > 0
            })

        return fold_metrics


def print_optimization_summary(result: OptimizationResult) -> None:
    """Print optimization results summary."""
    print("\n" + "=" * 60)
    print("SIZING OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest objective value: {result.best_value:.4f}")

    print("\nBest parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value:.4f}")

    if result.param_importance:
        print("\nParameter importance:")
        sorted_imp = sorted(
            result.param_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for param, imp in sorted_imp:
            print(f"  {param}: {imp:.4f}")

    print("\nCV fold metrics:")
    for metrics in result.cv_metrics:
        print(f"  Fold {metrics.get('fold', '?')}: "
              f"return={metrics['portfolio_return']:.4f}, "
              f"hit_rate={metrics['hit_rate']:.2%}, "
              f"n_pos={metrics['n_positions']}")

    print("=" * 60)
