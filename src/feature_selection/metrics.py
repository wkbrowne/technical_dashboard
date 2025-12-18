"""Evaluation metrics for feature selection.

This module provides functions for computing classification and regression
metrics used in the feature selection pipeline, including tail metrics
for quantitative finance applications.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from .config import MetricType, TaskType


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities or scores.

    Returns:
        AUC score (0-1, higher is better).
    """
    from sklearn.metrics import roc_auc_score

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute logarithmic loss.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities (clipped to avoid log(0)).

    Returns:
        Log loss (lower is better).
    """
    from sklearn.metrics import log_loss

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    try:
        return log_loss(y_true, y_pred)
    except ValueError:
        return np.inf


def compute_aupr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve.

    AUPR is more informative than AUC for imbalanced datasets, as it focuses
    on the positive class performance.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities or scores.

    Returns:
        AUPR score (0-1, higher is better).
    """
    from sklearn.metrics import average_precision_score

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.0

    try:
        return average_precision_score(y_true, y_pred)
    except ValueError:
        return 0.0


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score (calibration metric).

    Brier score measures the mean squared error between predicted probabilities
    and actual outcomes. Lower is better (perfect calibration = 0).

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities.

    Returns:
        Brier score (0-1, lower is better).
    """
    from sklearn.metrics import brier_score_loss

    # Clip predictions to valid probability range
    y_pred = np.clip(y_pred, 0, 1)

    try:
        return brier_score_loss(y_true, y_pred)
    except ValueError:
        return 1.0


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Information Coefficient (Spearman rank correlation).

    Args:
        y_true: True values (can be continuous).
        y_pred: Predicted values.

    Returns:
        IC (Spearman correlation, -1 to 1).
    """
    if len(y_true) < 3:
        return 0.0

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return 0.0

    correlation, _ = stats.spearmanr(y_true[mask], y_pred[mask])

    if np.isnan(correlation):
        return 0.0

    return correlation


def compute_precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: float = 0.1,
    direction: str = 'top'
) -> float:
    """Compute precision in the top/bottom k percentile of predictions.

    Args:
        y_true: True binary labels (for classification) or continuous values.
        y_pred: Predicted scores.
        k: Fraction of samples to consider (e.g., 0.1 for top 10%).
        direction: 'top' for highest predictions, 'bottom' for lowest.

    Returns:
        Precision in the selected quantile.
    """
    n_samples = len(y_pred)
    n_select = max(1, int(n_samples * k))

    if direction == 'top':
        # Select samples with highest predictions
        selected_idx = np.argsort(y_pred)[-n_select:]
    else:
        # Select samples with lowest predictions
        selected_idx = np.argsort(y_pred)[:n_select]

    # For binary classification: precision = fraction of positives
    if np.array_equal(np.unique(y_true), [0, 1]) or np.array_equal(np.unique(y_true), [0., 1.]):
        return np.mean(y_true[selected_idx])

    # For regression: return mean actual value in selected samples
    return np.mean(y_true[selected_idx])


def compute_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute hit rate (accuracy) at a given threshold.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Hit rate (accuracy).
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return np.mean(y_true == y_pred_binary)


def compute_tail_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.1
) -> Dict[str, float]:
    """Compute metrics for tail predictions (top and bottom quantiles).

    Args:
        y_true: True labels/values.
        y_pred: Predicted scores.
        quantile: Fraction to use for tails (e.g., 0.1 for top/bottom 10%).

    Returns:
        Dict with tail metrics.
    """
    return {
        'precision_top': compute_precision_at_k(y_true, y_pred, quantile, 'top'),
        'precision_bottom': compute_precision_at_k(y_true, y_pred, quantile, 'bottom'),
        'spread': (
            compute_precision_at_k(y_true, y_pred, quantile, 'top') -
            compute_precision_at_k(y_true, y_pred, quantile, 'bottom')
        )
    }


def compute_regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime: np.ndarray,
    metric_fn: callable
) -> Dict[str, float]:
    """Compute metrics stratified by regime.

    Args:
        y_true: True labels/values.
        y_pred: Predicted scores.
        regime: Regime labels (same length as y_true).
        metric_fn: Function to compute metric (takes y_true, y_pred).

    Returns:
        Dict mapping regime label to metric value.
    """
    regime_metrics = {}

    for regime_label in np.unique(regime):
        mask = regime == regime_label
        if mask.sum() > 10:  # Minimum samples for reliable metric
            regime_metrics[str(regime_label)] = metric_fn(
                y_true[mask], y_pred[mask]
            )

    return regime_metrics


def get_metric_function(metric_type: MetricType, task_type: TaskType) -> callable:
    """Get the metric computation function for a given metric type.

    Args:
        metric_type: Type of metric to compute.
        task_type: Classification or regression task.

    Returns:
        Function that takes (y_true, y_pred) and returns metric value.
    """
    if metric_type == MetricType.AUC:
        return compute_auc
    elif metric_type == MetricType.AUPR:
        return compute_aupr
    elif metric_type == MetricType.BRIER:
        return compute_brier
    elif metric_type == MetricType.LOG_LOSS:
        return compute_log_loss
    elif metric_type == MetricType.IC:
        return compute_ic
    elif metric_type == MetricType.PRECISION_AT_K:
        return lambda y, p: compute_precision_at_k(y, p, k=0.1, direction='top')
    elif metric_type == MetricType.HIT_RATE:
        return compute_hit_rate
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def is_higher_better(metric_type: MetricType) -> bool:
    """Check if higher values are better for this metric.

    Args:
        metric_type: The metric type.

    Returns:
        True if higher is better, False if lower is better.
    """
    # Lower is better for these metrics
    lower_is_better = {MetricType.LOG_LOSS, MetricType.BRIER}
    return metric_type not in lower_is_better


class MetricComputer:
    """Handles computation of multiple metrics for model evaluation.

    Attributes:
        primary_metric: The main metric for optimization.
        secondary_metrics: Additional metrics to track.
        task_type: Classification or regression.
        tail_quantile: Quantile for tail metrics.
    """

    def __init__(
        self,
        primary_metric: MetricType = MetricType.AUC,
        secondary_metrics: Optional[List[MetricType]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        tail_quantile: float = 0.1
    ):
        """Initialize the metric computer.

        Args:
            primary_metric: Main metric for optimization.
            secondary_metrics: Additional metrics to compute.
            task_type: Type of ML task.
            tail_quantile: Quantile for tail metrics.
        """
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or []
        self.task_type = task_type
        self.tail_quantile = tail_quantile

        # Get metric functions
        self._primary_fn = get_metric_function(primary_metric, task_type)
        self._secondary_fns = {
            m: get_metric_function(m, task_type)
            for m in self.secondary_metrics
        }

    def compute_primary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Compute the primary metric.

        Args:
            y_true: True labels/values.
            y_pred: Predictions.

        Returns:
            Primary metric value.
        """
        return self._primary_fn(y_true, y_pred)

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regime: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Compute all configured metrics.

        Args:
            y_true: True labels/values.
            y_pred: Predictions.
            regime: Optional regime labels for stratified metrics.

        Returns:
            Dict with all computed metrics.
        """
        results = {
            'primary': self._primary_fn(y_true, y_pred)
        }

        # Secondary metrics
        for metric_type, metric_fn in self._secondary_fns.items():
            results[metric_type.value] = metric_fn(y_true, y_pred)

        # Tail metrics
        results['tail'] = compute_tail_metrics(
            y_true, y_pred, self.tail_quantile
        )

        # Regime metrics
        if regime is not None:
            results['regime'] = compute_regime_metrics(
                y_true, y_pred, regime, self._primary_fn
            )

        return results

    def aggregate_fold_metrics(
        self,
        fold_results: List[Dict[str, Union[float, Dict]]]
    ) -> Dict[str, Tuple[float, float]]:
        """Aggregate metrics across CV folds.

        Args:
            fold_results: List of metric dicts from each fold.

        Returns:
            Dict mapping metric name to (mean, std) tuple.
        """
        aggregated = {}

        # Primary metric
        primary_values = [r['primary'] for r in fold_results]
        aggregated['primary'] = (np.mean(primary_values), np.std(primary_values))

        # Secondary metrics
        for metric_type in self._secondary_fns:
            values = [r.get(metric_type.value, np.nan) for r in fold_results]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                aggregated[metric_type.value] = (
                    np.mean(valid_values), np.std(valid_values)
                )

        # Tail metrics
        for tail_key in ['precision_top', 'precision_bottom', 'spread']:
            values = [r['tail'].get(tail_key, np.nan) for r in fold_results]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                aggregated[f'tail_{tail_key}'] = (
                    np.mean(valid_values), np.std(valid_values)
                )

        return aggregated
