"""Core evaluation primitive for feature subsets.

This module provides the evaluate_subset function, which is the fundamental
building block for all feature selection algorithms. It handles training,
cross-validation, and metric computation in a memory-efficient manner.
"""

import gc
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from .config import (
    CVConfig, MetricConfig, MetricType, ModelConfig, SearchConfig,
    SubsetResult, TaskType
)
from .cv import TimeSeriesCV
from .metrics import (
    MetricComputer, get_metric_function,
    compute_auc, compute_aupr, compute_brier, compute_log_loss
)
from .models import GBMWrapper


class SubsetEvaluator:
    """Evaluates feature subsets using time-series cross-validation.

    This class holds references to the data and configuration, providing
    an efficient interface for evaluating many feature subsets without
    repeatedly copying data.

    Attributes:
        X: Feature DataFrame (stored once, never copied).
        y: Target Series.
        sample_weight: Optional sample weights (e.g., from overlap inverse weighting).
        model_config: Model configuration.
        cv_config: Cross-validation configuration.
        metric_config: Metrics configuration.
        search_config: Search configuration.
        regime: Optional regime labels for stratified evaluation.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: ModelConfig,
        cv_config: CVConfig,
        metric_config: MetricConfig,
        search_config: SearchConfig,
        regime: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ):
        """Initialize the evaluator.

        Args:
            X: Feature DataFrame with DateTimeIndex.
            y: Target Series aligned with X.
            model_config: Model settings.
            cv_config: CV settings.
            metric_config: Metrics settings.
            search_config: Search settings.
            regime: Optional regime labels for stratified metrics.
            sample_weight: Optional sample weights for training (e.g., from overlap inverse).
                          These weights are applied during model training to down-weight
                          overlapping trajectories from triple barrier targets.
        """
        # Store references (not copies)
        self._X = X
        self._y = y
        self._regime = regime
        self._sample_weight = sample_weight

        # Store configurations
        self.model_config = model_config
        self.cv_config = cv_config
        self.metric_config = metric_config
        self.search_config = search_config

        # Pre-compute feature indices for efficiency
        self._feature_to_idx = {f: i for i, f in enumerate(X.columns)}
        self._all_features = list(X.columns)

        # Initialize CV splitter
        self._cv = TimeSeriesCV(cv_config)

        # Initialize metric computer
        self._metric_computer = MetricComputer(
            primary_metric=metric_config.primary_metric,
            secondary_metrics=metric_config.secondary_metrics,
            task_type=model_config.task_type,
            tail_quantile=metric_config.tail_quantile
        )

        # Cache for fold indices (computed once)
        self._fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def _get_fold_indices(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cached CV fold indices."""
        if self._fold_indices is None:
            self._fold_indices = list(self._cv.split(self._X))
        return self._fold_indices

    def evaluate(
        self,
        features: Union[List[str], Set[str]],
        return_model: bool = False,
        n_jobs: int = 1
    ) -> SubsetResult:
        """Evaluate a feature subset.

        Args:
            features: List or set of feature names to evaluate.
            return_model: If True, also return the trained model (last fold).
            n_jobs: Number of parallel jobs for fold evaluation.

        Returns:
            SubsetResult with metrics and optional model.
        """
        feature_list = list(features)

        if not feature_list:
            return SubsetResult(
                features=[],
                metric_main=0.0,
                metric_std=0.0,
                fold_metrics=[],
                n_base_features=0,
                n_interaction_features=0
            )

        # Get column indices for efficient slicing
        col_indices = [self._feature_to_idx[f] for f in feature_list if f in self._feature_to_idx]

        if not col_indices:
            return SubsetResult(
                features=feature_list,
                metric_main=0.0,
                metric_std=0.0,
                fold_metrics=[],
                n_base_features=len(feature_list),
                n_interaction_features=0
            )

        # Get fold indices
        fold_indices = self._get_fold_indices()

        # Evaluate each fold
        if n_jobs > 1 and len(fold_indices) > 1:
            fold_results = self._evaluate_folds_parallel(
                col_indices, feature_list, fold_indices, n_jobs
            )
        else:
            fold_results = self._evaluate_folds_sequential(
                col_indices, feature_list, fold_indices
            )

        # Aggregate results
        primary_values = [r['primary'] for r in fold_results]
        metric_main = np.mean(primary_values)
        metric_std = np.std(primary_values)

        # Compute secondary metrics
        aggregated = self._metric_computer.aggregate_fold_metrics(fold_results)
        secondary_metrics = {
            k: v for k, v in aggregated.items()
            if k != 'primary' and not k.startswith('tail_')
        }

        # Aggregate extended metrics (AUC, AUPR, Brier, LogLoss)
        extended_keys = ['auc', 'aupr', 'brier', 'log_loss']
        for ext_key in extended_keys:
            ext_values = [r.get('extended', {}).get(ext_key, np.nan) for r in fold_results]
            valid_values = [v for v in ext_values if not np.isnan(v)]
            if valid_values:
                secondary_metrics[ext_key] = (np.mean(valid_values), np.std(valid_values))

        # Regime metrics (average across folds)
        regime_metrics = {}
        if self._regime is not None:
            for r in fold_results:
                if 'regime' in r:
                    for regime_label, value in r['regime'].items():
                        if regime_label not in regime_metrics:
                            regime_metrics[regime_label] = []
                        regime_metrics[regime_label].append(value)
            regime_metrics = {k: np.mean(v) for k, v in regime_metrics.items()}

        # Count interaction features
        n_interaction = sum(1 for f in feature_list if '_x_' in f or '*' in f)

        return SubsetResult(
            features=feature_list,
            metric_main=metric_main,
            metric_std=metric_std,
            fold_metrics=primary_values,
            secondary_metrics=secondary_metrics,
            regime_metrics=regime_metrics,
            n_base_features=len(feature_list) - n_interaction,
            n_interaction_features=n_interaction
        )

    def _evaluate_folds_sequential(
        self,
        col_indices: List[int],
        feature_names: List[str],
        fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """Evaluate folds sequentially.

        Args:
            col_indices: Column indices for the feature subset.
            feature_names: Feature names.
            fold_indices: List of (train_idx, test_idx) tuples.

        Returns:
            List of metric dicts per fold.
        """
        results = []

        for train_idx, test_idx in fold_indices:
            result = self._evaluate_single_fold(
                col_indices, feature_names, train_idx, test_idx
            )
            results.append(result)
            gc.collect()  # Clean up after each fold

        return results

    def _evaluate_folds_parallel(
        self,
        col_indices: List[int],
        feature_names: List[str],
        fold_indices: List[Tuple[np.ndarray, np.ndarray]],
        n_jobs: int
    ) -> List[Dict[str, Any]]:
        """Evaluate folds in parallel using joblib.

        Args:
            col_indices: Column indices for the feature subset.
            feature_names: Feature names.
            fold_indices: List of (train_idx, test_idx) tuples.
            n_jobs: Number of parallel workers.

        Returns:
            List of metric dicts per fold (in order).
        """
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(self._evaluate_single_fold)(
                col_indices, feature_names, train_idx, test_idx
            )
            for train_idx, test_idx in fold_indices
        )

        gc.collect()
        return results

    def _evaluate_single_fold(
        self,
        col_indices: List[int],
        feature_names: List[str],
        train_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate a single CV fold.

        Args:
            col_indices: Column indices for features.
            feature_names: Feature names.
            train_idx: Training sample indices.
            test_idx: Test sample indices.

        Returns:
            Dict with computed metrics.
        """
        # Extract data using iloc for efficiency (view-based when possible)
        X_train = self._X.iloc[train_idx, col_indices]
        X_test = self._X.iloc[test_idx, col_indices]
        y_train = self._y.iloc[train_idx]
        y_test = self._y.iloc[test_idx]

        # Extract sample weights if available
        w_train = None
        if self._sample_weight is not None:
            w_train = self._sample_weight.iloc[train_idx]

        # Handle NaN values
        train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())

        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]
        X_test = X_test.loc[test_mask]
        y_test = y_test.loc[test_mask]

        # Apply mask to weights
        if w_train is not None:
            w_train = w_train.loc[train_mask]

        if len(X_train) < 50 or len(X_test) < 10:
            return {'primary': 0.0, 'tail': {}}

        # Create and train model
        model = GBMWrapper(self.model_config)

        # Split training for early stopping
        n_train = len(X_train)
        split_point = int(n_train * 0.85)

        if self.model_config.early_stopping_rounds and split_point > 50:
            X_tr = X_train.iloc[:split_point]
            X_val = X_train.iloc[split_point:]
            y_tr = y_train.iloc[:split_point]
            y_val = y_train.iloc[split_point:]

            # Split weights for early stopping
            w_tr = None
            w_val = None
            if w_train is not None:
                w_tr = w_train.iloc[:split_point]
                w_val = w_train.iloc[split_point:]

            model.train(X_tr, y_tr, X_val, y_val, feature_names,
                       sample_weight=w_tr, sample_weight_val=w_val)
        else:
            model.train(X_train, y_train, feature_names=feature_names,
                       sample_weight=w_train)

        # Generate predictions
        y_pred = model.predict(X_test)

        # Compute metrics using the MetricComputer (includes primary and configured secondaries)
        regime_arr = None
        if self._regime is not None:
            regime_arr = self._regime.iloc[test_idx].loc[test_mask].values

        metrics = self._metric_computer.compute_all(
            y_test.values, y_pred, regime_arr
        )

        # Additionally compute extended metrics for comprehensive tracking
        # These are always computed regardless of configuration
        y_true_arr = y_test.values
        metrics['extended'] = {
            'auc': compute_auc(y_true_arr, y_pred),
            'aupr': compute_aupr(y_true_arr, y_pred),
            'brier': compute_brier(y_true_arr, y_pred),
            'log_loss': compute_log_loss(y_true_arr, y_pred),
        }

        # Cleanup
        model.cleanup()

        return metrics

    def get_feature_importance(
        self,
        features: List[str],
        importance_type: str = 'shap',
        max_samples: int = 5000
    ) -> Dict[str, float]:
        """Get feature importance for a subset.

        Args:
            features: Feature names to analyze.
            importance_type: 'shap', 'gain', or 'permutation'.
            max_samples: Max samples for SHAP computation.

        Returns:
            Dict mapping feature name to importance score.
        """
        if not features:
            return {}

        col_indices = [self._feature_to_idx[f] for f in features if f in self._feature_to_idx]

        # Use last fold for importance
        fold_indices = self._get_fold_indices()
        train_idx, test_idx = fold_indices[-1]

        X_train = self._X.iloc[train_idx, col_indices]
        y_train = self._y.iloc[train_idx]

        # Handle NaN
        mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]

        # Train model
        model = GBMWrapper(self.model_config)
        model.train(X_train, y_train, feature_names=features)

        if importance_type == 'shap':
            # Subsample for SHAP
            if len(X_train) > max_samples:
                sample_idx = np.random.choice(len(X_train), max_samples, replace=False)
                X_sample = X_train.iloc[sample_idx]
            else:
                X_sample = X_train

            try:
                shap_values = model.get_shap_values(X_sample, check_additivity=False)
                importance = np.abs(shap_values).mean(axis=0)
                importance_dict = dict(zip(features, importance))
            except Exception:
                # Fall back to gain importance
                importance_dict = model.get_feature_importance('gain')
        else:
            importance_dict = model.get_feature_importance(importance_type)

        model.cleanup()
        return importance_dict


def evaluate_subset(
    features: Union[List[str], Set[str]],
    evaluator: SubsetEvaluator
) -> SubsetResult:
    """Convenience function to evaluate a feature subset.

    Args:
        features: Feature names to evaluate.
        evaluator: Configured SubsetEvaluator instance.

    Returns:
        SubsetResult with evaluation metrics.
    """
    return evaluator.evaluate(features)


def _evaluate_subset_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    model_config,
    cv_config,
    metric_config,
    idx: int,
    sample_weight: Optional[pd.Series] = None
) -> Tuple[int, SubsetResult]:
    """Joblib-compatible worker for parallel subset evaluation.

    Creates a fresh evaluator in each worker process to avoid pickling issues.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        features: Feature names to evaluate.
        model_config: Model configuration.
        cv_config: CV configuration.
        metric_config: Metrics configuration.
        idx: Index to preserve ordering in results.
        sample_weight: Optional sample weights for training.

    Returns:
        Tuple of (index, SubsetResult).
    """
    try:
        from .config import SearchConfig
        evaluator = SubsetEvaluator(
            X=X, y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=SearchConfig(),
            sample_weight=sample_weight,
        )
        result = evaluator.evaluate(features)
        return (idx, result)
    except Exception as e:
        return (idx, SubsetResult(
            features=features,
            metric_main=0.0,
            metric_std=0.0
        ))


def evaluate_subsets_parallel(
    feature_sets: List[List[str]],
    evaluator: SubsetEvaluator,
    n_jobs: int = 4
) -> List[SubsetResult]:
    """Evaluate multiple feature subsets in parallel using joblib.

    Args:
        feature_sets: List of feature lists to evaluate.
        evaluator: Configured SubsetEvaluator.
        n_jobs: Number of parallel workers.

    Returns:
        List of SubsetResult objects.
    """
    from joblib import Parallel, delayed

    # Extract data and configs for passing to worker processes
    X = evaluator._X
    y = evaluator._y
    model_config = evaluator.model_config
    cv_config = evaluator.cv_config
    metric_config = evaluator.metric_config

    # Run parallel evaluations with joblib (loky backend for true parallelism)
    results_with_idx = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_evaluate_subset_joblib)(
            X, y, features, model_config, cv_config, metric_config, idx
        )
        for idx, features in enumerate(feature_sets)
    )

    # Sort by index to preserve original order
    results_with_idx.sort(key=lambda x: x[0])
    results = [r[1] for r in results_with_idx]

    return results


class EvaluationCache:
    """Cache for subset evaluation results.

    Stores results keyed by frozen feature sets to avoid
    re-evaluating the same subset multiple times.

    Attributes:
        max_size: Maximum number of cached results.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the cache.

        Args:
            max_size: Maximum entries to cache.
        """
        self.max_size = max_size
        self._cache: Dict[frozenset, SubsetResult] = {}
        self._access_order: List[frozenset] = []

    def get(self, features: Union[List[str], Set[str]]) -> Optional[SubsetResult]:
        """Get cached result for a feature set.

        Args:
            features: Feature names.

        Returns:
            Cached SubsetResult or None if not cached.
        """
        key = frozenset(features)
        return self._cache.get(key)

    def put(self, features: Union[List[str], Set[str]], result: SubsetResult):
        """Cache a result.

        Args:
            features: Feature names.
            result: Evaluation result.
        """
        key = frozenset(features)

        if key in self._cache:
            return

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = result
        self._access_order.append(key)

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
