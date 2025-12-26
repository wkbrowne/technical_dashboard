"""
Future-biased pruning for HPO.

Provides pruning logic that evaluates trials on recent CV folds
to avoid optimizing for stale regimes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
import numpy as np


@dataclass
class PruneFoldResult:
    """Result from evaluating prune folds."""
    fold_aucs: Dict[int, float]  # fold_idx -> AUC
    prune_metric_value: float  # Aggregated prune metric
    should_prune: bool
    prune_reason: Optional[str] = None

    @property
    def mean_auc(self) -> float:
        """Get mean AUC across prune folds."""
        if not self.fold_aucs:
            return 0.0
        return np.mean(list(self.fold_aucs.values()))

    @property
    def min_auc(self) -> float:
        """Get minimum AUC across prune folds."""
        if not self.fold_aucs:
            return 0.0
        return min(self.fold_aucs.values())


@dataclass
class FoldEvaluator:
    """
    Evaluates CV folds for pruning decisions.

    Supports future-biased pruning where only recent folds are evaluated
    for initial pruning decisions.
    """
    # Configuration
    prune_folds: List[int] = field(default_factory=lambda: [3, 4])
    prune_metric: str = "mean"  # "mean" or "min"
    prune_margin: float = 0.002
    min_completed_trials: int = 30
    min_auc: float = 0.54
    max_brier: float = 0.26
    max_cv_coef: float = 0.20
    eval_all_folds: bool = True

    # State
    best_prune_metric: float = field(default=0.0, init=False)
    completed_trials: int = field(default=0, init=False)
    all_prune_metrics: List[float] = field(default_factory=list, init=False)

    def reset(self):
        """Reset state for new HPO round."""
        self.best_prune_metric = 0.0
        self.completed_trials = 0
        self.all_prune_metrics = []

    def compute_prune_metric(self, fold_aucs: Dict[int, float]) -> float:
        """Compute the prune metric from fold AUCs."""
        prune_aucs = [fold_aucs[f] for f in self.prune_folds if f in fold_aucs]
        if not prune_aucs:
            return 0.0
        if self.prune_metric == "min":
            return min(prune_aucs)
        return np.mean(prune_aucs)

    def check_absolute_thresholds(
        self,
        fold_metrics: Dict[int, Dict[str, float]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check absolute pruning thresholds.

        Args:
            fold_metrics: Dict mapping fold_idx to metrics dict
                         (must contain 'auc', 'brier')

        Returns:
            Tuple of (should_prune, reason)
        """
        for fold_idx, metrics in fold_metrics.items():
            auc = metrics.get('auc', 0.0)
            brier = metrics.get('brier', 1.0)

            if auc < self.min_auc:
                return True, f"low_auc_fold_{fold_idx} ({auc:.4f} < {self.min_auc})"

            if brier > self.max_brier:
                return True, f"high_brier_fold_{fold_idx} ({brier:.4f} > {self.max_brier})"

        return False, None

    def check_relative_pruning(
        self,
        prune_metric_value: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check relative pruning against best seen so far.

        Only applies after min_completed_trials.

        Args:
            prune_metric_value: The computed prune metric for this trial

        Returns:
            Tuple of (should_prune, reason)
        """
        if self.completed_trials < self.min_completed_trials:
            return False, None

        threshold = self.best_prune_metric - self.prune_margin
        if prune_metric_value < threshold:
            return True, (
                f"relative_pruning (prune_metric={prune_metric_value:.4f} < "
                f"best-margin={threshold:.4f})"
            )

        return False, None

    def check_cv_variance(
        self,
        all_fold_aucs: List[float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check CV coefficient (variance penalty).

        Only applies when we have results from multiple folds.

        Args:
            all_fold_aucs: List of AUC values from all evaluated folds

        Returns:
            Tuple of (should_prune, reason)
        """
        if len(all_fold_aucs) < 3:
            return False, None

        auc_mean = np.mean(all_fold_aucs)
        auc_std = np.std(all_fold_aucs)
        cv_coef = auc_std / (auc_mean + 1e-8)

        if cv_coef > self.max_cv_coef:
            return True, f"high_variance (cv_coef={cv_coef:.4f} > {self.max_cv_coef})"

        return False, None

    def evaluate_prune_folds(
        self,
        fold_metrics: Dict[int, Dict[str, float]],
    ) -> PruneFoldResult:
        """
        Evaluate prune folds and decide whether to prune.

        This is the main entry point for pruning decisions during trial
        evaluation. Call this after evaluating the prune_folds.

        Args:
            fold_metrics: Dict mapping fold_idx to metrics dict

        Returns:
            PruneFoldResult with pruning decision
        """
        # Extract AUCs for prune folds
        fold_aucs = {
            f: fold_metrics[f]['auc']
            for f in self.prune_folds
            if f in fold_metrics
        }

        if not fold_aucs:
            # No prune folds evaluated yet
            return PruneFoldResult(
                fold_aucs={},
                prune_metric_value=0.0,
                should_prune=False,
            )

        prune_metric_value = self.compute_prune_metric(fold_aucs)

        # Check absolute thresholds on prune folds only
        prune_fold_metrics = {f: fold_metrics[f] for f in self.prune_folds if f in fold_metrics}
        should_prune, reason = self.check_absolute_thresholds(prune_fold_metrics)
        if should_prune:
            return PruneFoldResult(
                fold_aucs=fold_aucs,
                prune_metric_value=prune_metric_value,
                should_prune=True,
                prune_reason=reason,
            )

        # Check relative pruning
        should_prune, reason = self.check_relative_pruning(prune_metric_value)
        if should_prune:
            return PruneFoldResult(
                fold_aucs=fold_aucs,
                prune_metric_value=prune_metric_value,
                should_prune=True,
                prune_reason=reason,
            )

        return PruneFoldResult(
            fold_aucs=fold_aucs,
            prune_metric_value=prune_metric_value,
            should_prune=False,
        )

    def evaluate_all_folds(
        self,
        all_fold_metrics: Dict[int, Dict[str, float]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate all folds for post-prune-fold checks.

        Call this after the trial has passed prune fold evaluation
        and remaining folds have been evaluated.

        NOTE: Absolute thresholds (min_auc, max_brier) are only checked on
        prune folds to maintain future-biased pruning. We already checked
        prune folds in evaluate_prune_folds(), so here we only check CV
        variance across all folds.

        Args:
            all_fold_metrics: Dict mapping all fold indices to metrics

        Returns:
            Tuple of (should_prune, reason)
        """
        # NOTE: We intentionally do NOT check absolute thresholds on all folds.
        # The prune folds (e.g., 3, 4) represent the "future" we care about.
        # Old folds (e.g., 0, 1, 2) may have different market regimes and
        # we don't want to prune trials that perform well on recent data
        # just because they struggle on older data.

        # Only check CV variance across all folds
        all_aucs = [m['auc'] for m in all_fold_metrics.values()]
        should_prune, reason = self.check_cv_variance(all_aucs)
        if should_prune:
            return True, reason

        return False, None

    def record_completed_trial(
        self,
        prune_metric_value: float,
    ) -> None:
        """
        Record a completed trial for tracking best metrics.

        Call this when a trial completes successfully.

        Args:
            prune_metric_value: The prune metric value for this trial
        """
        self.completed_trials += 1
        self.all_prune_metrics.append(prune_metric_value)

        if prune_metric_value > self.best_prune_metric:
            self.best_prune_metric = prune_metric_value

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get summary statistics about pruning behavior."""
        return {
            'completed_trials': self.completed_trials,
            'best_prune_metric': self.best_prune_metric,
            'prune_metric_mean': float(np.mean(self.all_prune_metrics)) if self.all_prune_metrics else 0.0,
            'prune_metric_std': float(np.std(self.all_prune_metrics)) if self.all_prune_metrics else 0.0,
            'prune_folds': self.prune_folds,
            'prune_metric_type': self.prune_metric,
            'prune_margin': self.prune_margin,
            'min_completed_trials': self.min_completed_trials,
        }


def should_prune_trial(
    fold_metrics: Dict[int, Dict[str, float]],
    prune_folds: List[int] = None,
    prune_metric: str = "mean",
    min_auc: float = 0.54,
    max_brier: float = 0.26,
    best_prune_metric: float = 0.0,
    prune_margin: float = 0.002,
    completed_trials: int = 0,
    min_completed_trials: int = 30,
) -> Tuple[bool, Optional[str], float]:
    """
    Simplified function interface for pruning decisions.

    Args:
        fold_metrics: Dict mapping fold_idx to metrics dict
        prune_folds: Which folds to use for pruning (default [3, 4])
        prune_metric: "mean" or "min"
        min_auc: Minimum acceptable AUC
        max_brier: Maximum acceptable Brier score
        best_prune_metric: Best prune metric seen so far
        prune_margin: Margin for relative pruning
        completed_trials: Number of completed trials so far
        min_completed_trials: Minimum trials before relative pruning

    Returns:
        Tuple of (should_prune, reason, prune_metric_value)
    """
    if prune_folds is None:
        prune_folds = [3, 4]

    # Compute prune metric
    prune_aucs = [
        fold_metrics[f]['auc']
        for f in prune_folds
        if f in fold_metrics
    ]
    if not prune_aucs:
        return False, None, 0.0

    if prune_metric == "min":
        prune_metric_value = min(prune_aucs)
    else:
        prune_metric_value = np.mean(prune_aucs)

    # Check absolute thresholds on prune folds
    for f in prune_folds:
        if f not in fold_metrics:
            continue
        metrics = fold_metrics[f]
        if metrics['auc'] < min_auc:
            return True, f"low_auc_fold_{f}", prune_metric_value
        if metrics.get('brier', 0) > max_brier:
            return True, f"high_brier_fold_{f}", prune_metric_value

    # Check relative pruning
    if completed_trials >= min_completed_trials:
        threshold = best_prune_metric - prune_margin
        if prune_metric_value < threshold:
            return True, "relative_pruning", prune_metric_value

    return False, None, prune_metric_value


def get_folds_to_evaluate(
    n_folds: int,
    prune_folds: List[int],
    trial_passed_prune: bool,
    eval_all_folds: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Determine which folds to evaluate in which order.

    For efficiency, we evaluate prune folds first. If the trial passes,
    we optionally evaluate remaining folds.

    Args:
        n_folds: Total number of CV folds
        prune_folds: Folds to evaluate first for pruning
        trial_passed_prune: Whether trial passed prune fold evaluation
        eval_all_folds: Whether to evaluate all folds for surviving trials

    Returns:
        Tuple of (prune_fold_indices, remaining_fold_indices)
    """
    all_folds = set(range(n_folds))
    prune_set = set(prune_folds)

    # Validate prune folds
    valid_prune = [f for f in prune_folds if 0 <= f < n_folds]

    if not trial_passed_prune:
        # Only evaluate prune folds
        return valid_prune, []

    if eval_all_folds:
        # Evaluate remaining folds
        remaining = sorted(all_folds - prune_set)
        return valid_prune, remaining

    return valid_prune, []
