"""Multi-model sizing optimizer using Optuna TPE.

Extends the existing sizing optimizer to support:
- Four models with per-model parameters
- Regime gating parameters
- Combined optimization of sizing + gating

The optimizer uses purged walk-forward CV to ensure no leakage.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

from .config import (
    ModelType,
    MultiModelSizingConfig,
    MonotoneSizingParams,
    RegimeGatingConfig,
    get_tpe_param_range,
)
from .multi_model import MultiModelSizingEngine
from .regime_gating import suggest_gating_params, create_gating_config_from_params


@dataclass
class MultiModelOptimizationResult:
    """Result of multi-model sizing optimization.

    Attributes:
        best_params: Best parameter values found.
        best_config: Best MultiModelSizingConfig.
        best_value: Best objective value.
        study: Optuna study object.
        param_importance: Parameter importance dict.
        trials_df: DataFrame of all trials.
        cv_metrics: List of CV fold metrics.
    """
    best_params: Dict[str, float]
    best_config: MultiModelSizingConfig
    best_value: float
    study: Any
    param_importance: Dict[str, float]
    trials_df: pd.DataFrame
    cv_metrics: List[Dict[str, float]]


def create_config_from_trial(
    trial: Trial,
    models: List[ModelType],
    optimize_gating: bool = False,
    base_config: Optional[MultiModelSizingConfig] = None,
) -> MultiModelSizingConfig:
    """Create sizing config from Optuna trial parameters.

    Args:
        trial: Optuna trial object.
        models: List of models to configure.
        optimize_gating: Whether to optimize gating parameters.
        base_config: Base config for defaults.

    Returns:
        MultiModelSizingConfig with suggested parameters.
    """
    # Suggest shared sizing parameters
    slope = trial.suggest_float("slope", *get_tpe_param_range("slope"))
    intercept = trial.suggest_float("intercept", *get_tpe_param_range("intercept"))
    exposure_mult = trial.suggest_float("exposure_mult", *get_tpe_param_range("exposure_mult"))
    turnover_penalty = trial.suggest_float(
        "turnover_penalty", *get_tpe_param_range("turnover_penalty")
    )

    # Parabolic offset
    parabolic_offset = trial.suggest_float(
        "parabolic_threshold_offset",
        *get_tpe_param_range("parabolic_threshold_offset"),
    )

    # Build sizing params
    sizing_params = MonotoneSizingParams(
        slope=slope,
        intercept=intercept,
        exposure_mult=exposure_mult,
        max_weight=base_config.max_weight_per_name if base_config else 0.10,
        min_weight=base_config.min_weight if base_config else 0.01,
    )

    # Build gating config
    if optimize_gating:
        gating_params = suggest_gating_params(trial)
        gating_config = create_gating_config_from_params(gating_params, enabled=True)
    else:
        gating_config = RegimeGatingConfig.disabled()

    # Create full config
    config = MultiModelSizingConfig(
        models=[m.value for m in models],
        combine_policy=base_config.combine_policy if base_config else "mode_priority",
        netting_policy=base_config.netting_policy if base_config else "strongest",
        sizing_params=sizing_params,
        regime_gating=gating_config,
        max_gross_exposure=base_config.max_gross_exposure if base_config else 1.0,
        max_net_exposure=base_config.max_net_exposure if base_config else 1.0,
        max_weight_per_name=base_config.max_weight_per_name if base_config else 0.10,
        min_weight=base_config.min_weight if base_config else 0.01,
        max_positions=base_config.max_positions if base_config else None,
        turnover_penalty=turnover_penalty,
        parabolic_threshold_offset=parabolic_offset,
    )

    return config


def compute_backtest_objective(
    signals: pd.DataFrame,
    turnover_penalty: float = 0.0,
) -> Dict[str, float]:
    """Compute objective metrics from weighted signals.

    Args:
        signals: DataFrame with final_weight and actual_return columns.
        turnover_penalty: Penalty per unit turnover.

    Returns:
        Dict of metrics.
    """
    if "final_weight" not in signals.columns or "actual_return" not in signals.columns:
        return {"portfolio_return": 0.0, "penalized_return": 0.0}

    weights = signals["final_weight"].values
    returns = signals["actual_return"].values

    # Handle NaN returns
    valid_mask = ~np.isnan(returns)
    weights = weights[valid_mask]
    returns = returns[valid_mask]

    # Portfolio return (weighted average)
    if len(weights) == 0 or np.abs(weights).sum() == 0:
        portfolio_return = 0.0
    else:
        portfolio_return = np.sum(weights * returns)

    # Hit rate
    if np.abs(weights).sum() > 0:
        wins = (returns > 0).astype(float)
        hit_rate = np.sum(np.abs(weights) * wins) / np.abs(weights).sum()
    else:
        hit_rate = 0.0

    # Gross exposure
    gross_exposure = np.abs(weights).sum()

    # Net exposure
    net_exposure = weights.sum()

    # Position counts
    n_longs = (weights > 0).sum()
    n_shorts = (weights < 0).sum()
    n_positions = n_longs + n_shorts

    # Penalized return (turnover computed separately)
    penalized_return = portfolio_return

    return {
        "portfolio_return": portfolio_return,
        "penalized_return": penalized_return,
        "hit_rate": hit_rate,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "n_positions": n_positions,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
    }


class MultiModelSizingOptimizer:
    """Optimizer for multi-model sizing parameters.

    Uses Optuna TPE to optimize:
    - Shared sizing parameters (slope, intercept, exposure_mult)
    - Parabolic threshold offset
    - Turnover penalty
    - Regime gating parameters (optional)

    Evaluation uses purged walk-forward CV.

    Attributes:
        models: List of models to optimize.
        n_trials: Number of optimization trials.
        metric: Metric to optimize.
        optimize_gating: Whether to optimize gating parameters.
        base_config: Base configuration for defaults.
        cv_splitter: CV splitter for evaluation.
    """

    def __init__(
        self,
        models: Optional[List[ModelType]] = None,
        n_trials: int = 100,
        metric: str = "penalized_return",
        optimize_gating: bool = False,
        base_config: Optional[MultiModelSizingConfig] = None,
        cv_splitter=None,
        random_state: int = 42,
    ):
        """Initialize optimizer.

        Args:
            models: Models to optimize (default: all four).
            n_trials: Number of optimization trials.
            metric: Metric to optimize.
            optimize_gating: Whether to optimize gating.
            base_config: Base configuration.
            cv_splitter: CV splitter (uses WeeklySignalCV if not provided).
            random_state: Random seed.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for MultiModelSizingOptimizer")

        self.models = models or list(ModelType)
        self.n_trials = n_trials
        self.metric = metric
        self.optimize_gating = optimize_gating
        self.base_config = base_config or MultiModelSizingConfig()
        self.cv_splitter = cv_splitter
        self.random_state = random_state

    def _objective(
        self,
        trial: Trial,
        signals: pd.DataFrame,
        regime_features: Optional[pd.DataFrame] = None,
    ) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial.
            signals: Prepared signals DataFrame with outcomes.
            regime_features: Optional regime features.

        Returns:
            Objective value (higher is better).
        """
        # Create config from trial
        config = create_config_from_trial(
            trial,
            self.models,
            optimize_gating=self.optimize_gating,
            base_config=self.base_config,
        )

        # Create engine
        engine = MultiModelSizingEngine(config)

        # Evaluate on CV folds
        fold_metrics = []

        if self.cv_splitter is not None:
            for train_idx, val_idx, fold_info in self.cv_splitter.split(signals):
                val_signals = signals.iloc[val_idx].copy()

                # Get regime features for validation period if available
                val_regime = None
                if regime_features is not None and "date" in regime_features.columns:
                    val_dates = val_signals["date"].unique() if "date" in val_signals.columns else []
                    val_regime = regime_features[regime_features["date"].isin(val_dates)]

                # Compute weights
                weighted = engine.compute_weights(val_signals, regime_features=val_regime)

                # Compute metrics
                metrics = compute_backtest_objective(
                    weighted, turnover_penalty=config.turnover_penalty
                )
                fold_metrics.append(metrics)
        else:
            # No CV - evaluate on full dataset
            weighted = engine.compute_weights(signals, regime_features=regime_features)
            metrics = compute_backtest_objective(
                weighted, turnover_penalty=config.turnover_penalty
            )
            fold_metrics.append(metrics)

        # Aggregate across folds
        metric_values = [m[self.metric] for m in fold_metrics]
        mean_metric = np.mean(metric_values)

        # Penalize high variance
        std_metric = np.std(metric_values) if len(metric_values) > 1 else 0
        stability_penalty = 0.5 * std_metric

        return mean_metric - stability_penalty

    def optimize(
        self,
        signals: pd.DataFrame,
        regime_features: Optional[pd.DataFrame] = None,
        show_progress: bool = True,
    ) -> MultiModelOptimizationResult:
        """Run optimization.

        Args:
            signals: Prepared signals DataFrame with outcomes.
            regime_features: Optional regime features for gating.
            show_progress: Whether to show progress bar.

        Returns:
            MultiModelOptimizationResult.
        """
        # Create study
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, signals, regime_features),
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

        # Create best config
        best_config = self._create_best_config(study.best_params)

        # Compute CV metrics for best params
        cv_metrics = self._get_cv_metrics(signals, best_config, regime_features)

        return MultiModelOptimizationResult(
            best_params=study.best_params,
            best_config=best_config,
            best_value=study.best_value,
            study=study,
            param_importance=importance,
            trials_df=trials_df,
            cv_metrics=cv_metrics,
        )

    def _create_best_config(self, params: Dict) -> MultiModelSizingConfig:
        """Create config from best parameters.

        Args:
            params: Best parameter dict.

        Returns:
            MultiModelSizingConfig.
        """
        sizing_params = MonotoneSizingParams(
            slope=params.get("slope", 2.0),
            intercept=params.get("intercept", 0.5),
            exposure_mult=params.get("exposure_mult", 1.0),
            max_weight=self.base_config.max_weight_per_name,
            min_weight=self.base_config.min_weight,
        )

        if self.optimize_gating:
            gating_config = create_gating_config_from_params(params, enabled=True)
        else:
            gating_config = RegimeGatingConfig.disabled()

        return MultiModelSizingConfig(
            models=[m.value for m in self.models],
            combine_policy=self.base_config.combine_policy,
            netting_policy=self.base_config.netting_policy,
            sizing_params=sizing_params,
            regime_gating=gating_config,
            max_gross_exposure=self.base_config.max_gross_exposure,
            max_net_exposure=self.base_config.max_net_exposure,
            max_weight_per_name=self.base_config.max_weight_per_name,
            min_weight=self.base_config.min_weight,
            max_positions=self.base_config.max_positions,
            turnover_penalty=params.get("turnover_penalty", 0.0),
            parabolic_threshold_offset=params.get("parabolic_threshold_offset", 0.05),
        )

    def _get_cv_metrics(
        self,
        signals: pd.DataFrame,
        config: MultiModelSizingConfig,
        regime_features: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, float]]:
        """Get detailed CV metrics for given config.

        Args:
            signals: Signals DataFrame.
            config: Sizing configuration.
            regime_features: Optional regime features.

        Returns:
            List of metrics per fold.
        """
        engine = MultiModelSizingEngine(config)
        fold_metrics = []

        if self.cv_splitter is not None:
            for train_idx, val_idx, fold_info in self.cv_splitter.split(signals):
                val_signals = signals.iloc[val_idx].copy()

                val_regime = None
                if regime_features is not None and "date" in regime_features.columns:
                    val_dates = val_signals["date"].unique() if "date" in val_signals.columns else []
                    val_regime = regime_features[regime_features["date"].isin(val_dates)]

                weighted = engine.compute_weights(val_signals, regime_features=val_regime)
                metrics = compute_backtest_objective(weighted, config.turnover_penalty)

                if hasattr(fold_info, 'fold_idx'):
                    metrics["fold"] = fold_info.fold_idx
                if hasattr(fold_info, 'test_start'):
                    metrics["test_start"] = str(fold_info.test_start)
                if hasattr(fold_info, 'test_end'):
                    metrics["test_end"] = str(fold_info.test_end)

                fold_metrics.append(metrics)
        else:
            weighted = engine.compute_weights(signals, regime_features=regime_features)
            metrics = compute_backtest_objective(weighted, config.turnover_penalty)
            fold_metrics.append(metrics)

        return fold_metrics


def print_optimization_summary(result: MultiModelOptimizationResult) -> None:
    """Print optimization results summary."""
    print("\n" + "=" * 60)
    print("MULTI-MODEL SIZING OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest objective value: {result.best_value:.4f}")

    print("\nBest parameters:")
    for param, value in sorted(result.best_params.items()):
        print(f"  {param}: {value:.4f}")

    if result.param_importance:
        print("\nParameter importance:")
        sorted_imp = sorted(
            result.param_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for param, imp in sorted_imp[:10]:
            print(f"  {param}: {imp:.4f}")

    print("\nCV fold metrics:")
    for metrics in result.cv_metrics:
        fold = metrics.get("fold", "?")
        ret = metrics.get("portfolio_return", 0)
        hit = metrics.get("hit_rate", 0)
        n_pos = metrics.get("n_positions", 0)
        print(f"  Fold {fold}: return={ret:.4f}, hit_rate={hit:.2%}, n_pos={n_pos}")

    print("=" * 60)


def save_optimization_result(
    result: MultiModelOptimizationResult,
    output_dir: Path,
    name: str = "multi_model",
) -> Dict[str, Path]:
    """Save optimization results to files.

    Args:
        result: Optimization result.
        output_dir: Output directory.
        name: Base name for files.

    Returns:
        Dict of file type -> path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save best config
    config_path = output_dir / f"best_config_{name}.json"
    from .config import save_multi_model_config
    save_multi_model_config(
        result.best_config,
        config_path,
        metadata={
            "best_value": result.best_value,
            "n_trials": len(result.trials_df),
            "cv_metrics": result.cv_metrics,
        },
    )
    paths["config"] = config_path

    # Save best params (legacy format)
    params_path = output_dir / f"best_params_{name}.json"
    with open(params_path, "w") as f:
        json.dump({
            **result.best_params,
            "_metadata": {
                "best_value": result.best_value,
                "n_trials": len(result.trials_df),
            },
        }, f, indent=2)
    paths["params"] = params_path

    # Save trials
    trials_path = output_dir / f"trials_{name}.csv"
    result.trials_df.to_csv(trials_path, index=False)
    paths["trials"] = trials_path

    # Save parameter importance
    if result.param_importance:
        importance_path = output_dir / f"param_importance_{name}.json"
        with open(importance_path, "w") as f:
            json.dump(result.param_importance, f, indent=2)
        paths["importance"] = importance_path

    return paths
