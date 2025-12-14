"""Feature selection algorithms.

This module implements the core feature selection algorithms:
- Initial feature ranking
- Forward selection (SFS)
- Backward elimination (SBS)
- Pairwise swapping local search
- Parallel interaction forward selection (enhanced)
- Late interaction refinement
"""

import gc
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .config import (
    FeatureRanking, ModelConfig, SearchConfig, SubsetResult
)
from .evaluation import EvaluationCache, SubsetEvaluator
from .progress import ProgressTracker
from .interactions import (
    InteractionCandidate,
    PairwiseInteractionSearch,
    compute_interaction_lazily,
    generate_interaction_candidates
)


def compute_initial_ranking(
    evaluator: SubsetEvaluator,
    features: List[str],
    search_config: SearchConfig,
    progress: Optional[ProgressTracker] = None
) -> FeatureRanking:
    """Compute initial feature ranking using a full model.

    Trains a model on all features and computes importance scores
    using SHAP values (or gain if SHAP fails). Features are then
    partitioned into core, borderline, and low-value tiers.

    Args:
        evaluator: Configured SubsetEvaluator.
        features: List of all candidate features.
        search_config: Search configuration.
        progress: Optional progress tracker.

    Returns:
        FeatureRanking with sorted features and tier assignments.
    """
    if progress:
        progress.start_phase("initial_ranking", total_evals=1)

    start_time = time.time()

    # Get importance scores
    importance = evaluator.get_feature_importance(
        features,
        importance_type='shap',
        max_samples=5000
    )

    # Sort by importance
    sorted_features = sorted(
        importance.keys(),
        key=lambda f: importance.get(f, 0),
        reverse=True
    )

    # Normalize importance scores
    max_imp = max(importance.values()) if importance else 1.0
    if max_imp > 0:
        importance = {k: v / max_imp for k, v in importance.items()}

    # Partition into tiers
    n_core = min(search_config.n_core_features, len(sorted_features))
    n_borderline = min(
        search_config.n_borderline_features,
        len(sorted_features) - n_core
    )

    core_features = set(sorted_features[:n_core])
    borderline_features = set(sorted_features[n_core:n_core + n_borderline])
    low_value_features = set(sorted_features[n_core + n_borderline:])

    # Filter low-value features by importance threshold
    low_value_features = {
        f for f in low_value_features
        if importance.get(f, 0) >= search_config.importance_threshold
    }

    ranking = FeatureRanking(
        feature_names=sorted_features,
        importance_scores=importance,
        core_features=core_features,
        borderline_features=borderline_features,
        low_value_features=low_value_features
    )

    if progress:
        eval_time = time.time() - start_time
        progress.update(
            completed=1,
            best_metric=0.0,
            best_subset_size=len(features),
            eval_time=eval_time
        )
        progress.end_phase()

    return ranking


def _evaluate_candidate_worker(
    evaluator: SubsetEvaluator,
    current_set: Set[str],
    candidate: str,
    cache: Optional[EvaluationCache]
) -> Tuple[str, SubsetResult]:
    """Worker function to evaluate a single candidate (for parallel execution).

    Args:
        evaluator: SubsetEvaluator instance.
        current_set: Current feature set.
        candidate: Candidate feature to add.
        cache: Optional evaluation cache.

    Returns:
        Tuple of (candidate_name, SubsetResult).
    """
    test_set = current_set | {candidate}
    result = _cached_evaluate(evaluator, test_set, cache)
    return (candidate, result)


def _evaluate_candidate_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    current_features: List[str],
    candidate: str,
    model_config,
    cv_config,
    metric_config,
    search_config,
) -> Tuple[str, Optional[SubsetResult]]:
    """Joblib-compatible worker for parallel candidate evaluation.

    Creates a fresh evaluator in each worker process to avoid pickling issues.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        current_features: List of current feature names.
        candidate: Candidate feature to add.
        model_config: Model configuration.
        cv_config: CV configuration.
        metric_config: Metric configuration.
        search_config: Search configuration.

    Returns:
        Tuple of (candidate_name, SubsetResult or None on error).
    """
    try:
        # Create fresh evaluator in worker process
        from .evaluation import SubsetEvaluator

        evaluator = SubsetEvaluator(
            X=X,
            y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
        )

        test_features = list(current_features) + [candidate]
        result = evaluator.evaluate(test_features)
        return (candidate, result)
    except Exception as e:
        # Return None on error so we can skip this candidate
        return (candidate, None)


def _evaluate_swap_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    current_features: List[str],
    f_in: str,
    f_out: str,
    model_config,
    cv_config,
    metric_config,
    search_config,
) -> Tuple[str, str, Optional[SubsetResult]]:
    """Joblib-compatible worker for parallel swap evaluation.

    Creates a fresh evaluator in each worker process to avoid pickling issues.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        current_features: List of current feature names.
        f_in: Feature to remove from current set.
        f_out: Feature to add from outside candidates.
        model_config: Model configuration.
        cv_config: CV configuration.
        metric_config: Metric configuration.
        search_config: Search configuration.

    Returns:
        Tuple of (f_in, f_out, SubsetResult or None on error).
    """
    try:
        # Create fresh evaluator in worker process
        from .evaluation import SubsetEvaluator

        evaluator = SubsetEvaluator(
            X=X,
            y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
        )

        # Create swapped feature set
        test_features = [f for f in current_features if f != f_in] + [f_out]
        result = evaluator.evaluate(test_features)
        return (f_in, f_out, result)
    except Exception as e:
        # Return None on error so we can skip this swap
        return (f_in, f_out, None)


def _check_fold_improvement(
    baseline_folds: List[float],
    candidate_folds: List[float],
    min_ratio: float = 0.5
) -> bool:
    """Check if candidate shows improvement in sufficient folds.

    Args:
        baseline_folds: Per-fold metrics for baseline (current set).
        candidate_folds: Per-fold metrics for candidate (current set + feature).
        min_ratio: Minimum fraction of folds that must show non-negative improvement.

    Returns:
        True if candidate passes fold-level check.
    """
    if not baseline_folds or not candidate_folds:
        return True  # Skip check if fold metrics unavailable

    if len(baseline_folds) != len(candidate_folds):
        return True  # Skip check if mismatch

    n_improved = sum(
        1 for base, cand in zip(baseline_folds, candidate_folds)
        if cand >= base - 1e-6  # Allow tiny tolerance
    )
    ratio = n_improved / len(baseline_folds)
    return ratio >= min_ratio


def _is_harmful(
    baseline_score: float,
    candidate_score: float,
    baseline_folds: List[float],
    candidate_folds: List[float],
    epsilon_harm: float = 0.002,
    min_fold_harm_ratio: float = 0.6
) -> bool:
    """Check if adding a feature is clearly harmful.

    Used in lenient mode: we ADD unless this returns True.

    Args:
        baseline_score: Mean metric for baseline (current set).
        candidate_score: Mean metric for candidate (current set + feature).
        baseline_folds: Per-fold metrics for baseline.
        candidate_folds: Per-fold metrics for candidate.
        epsilon_harm: Minimum degradation to consider harmful.
        min_fold_harm_ratio: Fraction of folds that must show harm.

    Returns:
        True if candidate is clearly harmful and should be rejected.
    """
    # Check if mean performance dropped significantly
    mean_degradation = baseline_score - candidate_score
    if mean_degradation < epsilon_harm:
        return False  # Not harmful - mean didn't drop enough

    # Check if majority of folds show harm
    if not baseline_folds or not candidate_folds:
        return mean_degradation >= epsilon_harm  # Fall back to mean-only check

    if len(baseline_folds) != len(candidate_folds):
        return mean_degradation >= epsilon_harm

    n_harmed = sum(
        1 for base, cand in zip(baseline_folds, candidate_folds)
        if base > cand + 1e-6  # Fold got worse
    )
    fold_harm_ratio = n_harmed / len(baseline_folds)

    # Harmful only if BOTH mean dropped significantly AND majority of folds got worse
    return fold_harm_ratio >= min_fold_harm_ratio


def forward_selection(
    evaluator: SubsetEvaluator,
    initial_features: Set[str],
    candidate_features: List[str],
    search_config: SearchConfig,
    cache: Optional[EvaluationCache] = None,
    progress: Optional[ProgressTracker] = None,
    max_features: Optional[int] = None,
    parallel: bool = True
) -> Tuple[Set[str], SubsetResult]:
    """Forward feature selection (SFS) with optional parallelization.

    Two modes:
    - Strict mode (lenient_add=False): Add only if improvement >= epsilon_add
    - Lenient mode (lenient_add=True): Add unless clearly harmful

    When parallel=True, evaluates all candidates within an iteration in parallel.

    Args:
        evaluator: Configured SubsetEvaluator.
        initial_features: Starting feature set (e.g., core features).
        candidate_features: Pool of features to consider adding.
        search_config: Search configuration.
        cache: Optional evaluation cache.
        progress: Optional progress tracker.
        max_features: Maximum features to select (overrides search_config).
        parallel: Whether to parallelize candidate evaluation within iterations.

    Returns:
        Tuple of (selected_features, best_result).
    """
    max_feats = max_features or search_config.max_features
    epsilon = search_config.epsilon_add
    use_fold_check = search_config.use_fold_level_check
    min_fold_ratio = search_config.min_fold_improvement_ratio

    # Lenient mode: add unless harmful
    lenient_mode = search_config.lenient_add
    epsilon_harm = search_config.epsilon_harm
    min_fold_harm_ratio = search_config.min_fold_harm_ratio

    # Forced exploration: add best feature until min_features reached
    min_features = search_config.min_features
    force_best = search_config.force_best_per_iteration

    # Use dedicated forward selection parallelism config
    n_jobs = search_config.forward_selection_n_jobs if parallel else 0
    if n_jobs <= 0:
        n_jobs = 1  # Sequential if disabled

    # Initialize
    current_set = set(initial_features)
    candidates = [f for f in candidate_features if f not in current_set]

    # Evaluate initial set
    best_result = _cached_evaluate(evaluator, current_set, cache)
    best_score = best_result.metric_main

    # Estimate total evaluations
    max_iterations = min(len(candidates), max_feats - len(current_set))
    total_evals_estimate = sum(range(len(candidates), len(candidates) - max_iterations, -1))

    if progress:
        progress.start_phase(
            "forward_selection",
            total_evals=total_evals_estimate,
            is_estimate=True
        )

    completed_evals = 0
    iteration = 0

    while len(current_set) < max_feats and candidates:
        iteration += 1
        best_candidate = None
        best_candidate_score = best_score
        best_candidate_result = None
        iteration_start = time.time()

        if parallel and n_jobs > 1 and len(candidates) > 1:
            # Parallel evaluation using joblib with loky backend (process-based)
            from joblib import Parallel, delayed

            # Get data and configs from evaluator for worker processes
            X = evaluator._X
            y = evaluator._y
            model_config = evaluator.model_config
            cv_config = evaluator.cv_config
            metric_config = evaluator.metric_config
            current_features_list = list(current_set)

            # Run parallel evaluations
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(_evaluate_candidate_joblib)(
                    X, y, current_features_list, cand,
                    model_config, cv_config, metric_config, search_config
                )
                for cand in candidates
            )

            # Process results - collect all acceptable candidates and track best overall
            acceptable_candidates = []
            best_overall_candidate = None
            best_overall_score = float('-inf')
            best_overall_result = None

            for cand_name, result in results:
                completed_evals += 1

                if result is None:
                    continue

                # Track best overall (for forced exploration)
                if result.metric_main > best_overall_score:
                    best_overall_candidate = cand_name
                    best_overall_score = result.metric_main
                    best_overall_result = result

                if lenient_mode:
                    # LENIENT: Accept unless clearly harmful
                    is_harmful = _is_harmful(
                        best_score, result.metric_main,
                        best_result.fold_metrics, result.fold_metrics,
                        epsilon_harm, min_fold_harm_ratio
                    )
                    if not is_harmful:
                        acceptable_candidates.append((cand_name, result))
                else:
                    # STRICT: Require improvement >= epsilon
                    if result.metric_main <= best_candidate_score + epsilon:
                        continue

                    # Check fold-level improvement if enabled
                    if use_fold_check and best_result.fold_metrics and result.fold_metrics:
                        if not _check_fold_improvement(
                            best_result.fold_metrics, result.fold_metrics, min_fold_ratio
                        ):
                            continue

                    acceptable_candidates.append((cand_name, result))

            # Pick the best acceptable candidate
            for cand_name, result in acceptable_candidates:
                if result.metric_main > best_candidate_score:
                    best_candidate = cand_name
                    best_candidate_score = result.metric_main
                    best_candidate_result = result

            # Forced exploration: if no acceptable and under min_features, take best overall
            if best_candidate is None and force_best and len(current_set) < min_features:
                if best_overall_candidate is not None:
                    best_candidate = best_overall_candidate
                    best_candidate_score = best_overall_score
                    best_candidate_result = best_overall_result

            # Update progress after batch
            if progress:
                elapsed = time.time() - iteration_start
                avg_time = elapsed / len(candidates) if candidates else 1.0
                progress.update(
                    completed=completed_evals,
                    best_metric=best_score,
                    best_subset_size=len(current_set),
                    eval_time=avg_time
                )
        else:
            # Sequential evaluation (original behavior)
            best_overall_candidate = None
            best_overall_score = float('-inf')
            best_overall_result = None

            for candidate in candidates:
                start_time = time.time()

                test_set = current_set | {candidate}
                result = _cached_evaluate(evaluator, test_set, cache)

                completed_evals += 1
                eval_time = time.time() - start_time

                # Track best overall (for forced exploration)
                if result.metric_main > best_overall_score:
                    best_overall_candidate = candidate
                    best_overall_score = result.metric_main
                    best_overall_result = result

                is_acceptable = False
                if lenient_mode:
                    # LENIENT: Accept unless clearly harmful
                    is_harmful = _is_harmful(
                        best_score, result.metric_main,
                        best_result.fold_metrics, result.fold_metrics,
                        epsilon_harm, min_fold_harm_ratio
                    )
                    is_acceptable = not is_harmful
                else:
                    # STRICT: Require improvement >= epsilon
                    if result.metric_main > best_candidate_score + epsilon:
                        # Check fold-level improvement if enabled
                        passes_fold_check = True
                        if use_fold_check and best_result.fold_metrics and result.fold_metrics:
                            passes_fold_check = _check_fold_improvement(
                                best_result.fold_metrics, result.fold_metrics, min_fold_ratio
                            )
                        is_acceptable = passes_fold_check

                # Pick the best acceptable candidate
                if is_acceptable and result.metric_main > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = result.metric_main
                    best_candidate_result = result

                if progress:
                    progress.update(
                        completed=completed_evals,
                        best_metric=best_score,
                        best_subset_size=len(current_set),
                        eval_time=eval_time
                    )

            # Forced exploration: if no acceptable and under min_features, take best overall
            if best_candidate is None and force_best and len(current_set) < min_features:
                if best_overall_candidate is not None:
                    best_candidate = best_overall_candidate
                    best_candidate_score = best_overall_score
                    best_candidate_result = best_overall_result

        # Add best candidate if found
        if best_candidate is not None:
            current_set.add(best_candidate)
            candidates.remove(best_candidate)
            best_score = best_candidate_score
            best_result = best_candidate_result
        else:
            # No improvement found, stop
            break

        gc.collect()

    if progress:
        progress.end_phase(final_metric=best_score)

    return current_set, best_result


def backward_elimination(
    evaluator: SubsetEvaluator,
    features: Set[str],
    search_config: SearchConfig,
    protected_features: Optional[Set[str]] = None,
    cache: Optional[EvaluationCache] = None,
    progress: Optional[ProgressTracker] = None
) -> Tuple[Set[str], SubsetResult]:
    """Backward feature elimination (SBS) with strict removal criteria.

    Iteratively removes features only if removal improves or barely changes performance.
    This stricter version keeps features unless removing them is clearly beneficial.

    Removal criteria:
    - Performance must improve or stay within epsilon_remove (very strict)
    - If use_fold_level_check=True, most folds must also show maintained/improved performance

    Args:
        evaluator: Configured SubsetEvaluator.
        features: Starting feature set.
        search_config: Search configuration.
        protected_features: Features that cannot be removed.
        cache: Optional evaluation cache.
        progress: Optional progress tracker.

    Returns:
        Tuple of (remaining_features, best_result).
    """
    epsilon = search_config.epsilon_remove
    protected = protected_features or set()
    use_fold_check = search_config.use_fold_level_check
    # Stricter: require more folds to maintain performance for removal
    min_fold_ratio = max(search_config.min_fold_improvement_ratio, 0.6)

    # Initialize
    current_set = set(features)
    removable = [f for f in current_set if f not in protected]

    # Evaluate initial set
    best_result = _cached_evaluate(evaluator, current_set, cache)
    best_score = best_result.metric_main

    # Get importance to order removal attempts
    importance = evaluator.get_feature_importance(
        list(current_set), importance_type='gain'
    )

    # Sort removable features by importance (ascending - try removing least important first)
    removable = sorted(removable, key=lambda f: importance.get(f, 0))

    # Estimate total evaluations
    total_evals_estimate = len(removable)

    if progress:
        progress.start_phase(
            "backward_elimination",
            total_evals=total_evals_estimate,
            is_estimate=True
        )

    completed_evals = 0
    features_removed = []

    for feature in removable:
        if feature not in current_set:
            continue

        start_time = time.time()

        # Try removing this feature
        test_set = current_set - {feature}
        result = _cached_evaluate(evaluator, test_set, cache)

        completed_evals += 1
        eval_time = time.time() - start_time

        # STRICT removal criteria:
        # 1. Performance must improve or barely change (within epsilon_remove)
        should_remove = result.metric_main >= best_score - epsilon

        # 2. If fold-level check enabled, most folds must maintain performance
        if should_remove and use_fold_check and best_result.fold_metrics and result.fold_metrics:
            # For removal, we check if performance is MAINTAINED (not degraded)
            # so we flip the comparison - check that removal doesn't hurt most folds
            n_maintained = sum(
                1 for base, new in zip(best_result.fold_metrics, result.fold_metrics)
                if new >= base - epsilon  # Fold performance maintained
            )
            fold_ratio = n_maintained / len(best_result.fold_metrics)
            should_remove = fold_ratio >= min_fold_ratio

        if should_remove:
            current_set = test_set
            best_score = result.metric_main
            best_result = result
            features_removed.append(feature)

        if progress:
            progress.update(
                completed=completed_evals,
                best_metric=best_score,
                best_subset_size=len(current_set),
                eval_time=eval_time
            )

    if progress:
        progress.end_phase(final_metric=best_score)

    return current_set, best_result


def pairwise_swapping(
    evaluator: SubsetEvaluator,
    features: Set[str],
    outside_candidates: List[str],
    search_config: SearchConfig,
    protected_features: Optional[Set[str]] = None,
    cache: Optional[EvaluationCache] = None,
    progress: Optional[ProgressTracker] = None
) -> Tuple[Set[str], SubsetResult]:
    """Pairwise swapping local search with parallel evaluation.

    Iteratively swaps features in the current set with outside candidates
    to escape local optima. Evaluates all swap candidates in parallel.

    Args:
        evaluator: Configured SubsetEvaluator.
        features: Current feature set.
        outside_candidates: Pool of features outside the current set.
        search_config: Search configuration.
        protected_features: Features that cannot be swapped out.
        cache: Optional evaluation cache.
        progress: Optional progress tracker.

    Returns:
        Tuple of (final_features, best_result).
    """
    from joblib import Parallel, delayed

    epsilon = search_config.epsilon_swap
    max_swaps = search_config.max_swaps
    protected = protected_features or set()
    n_jobs = search_config.forward_selection_n_jobs or 16

    # Initialize
    current_set = set(features)
    outside = [f for f in outside_candidates if f not in current_set]

    # Evaluate initial set
    best_result = _cached_evaluate(evaluator, current_set, cache)
    best_score = best_result.metric_main

    # Get importance to identify swappable (borderline) features
    importance = evaluator.get_feature_importance(
        list(current_set), importance_type='gain'
    )

    # Identify borderline features (not in top protected, not explicitly protected)
    sorted_inside = sorted(
        current_set,
        key=lambda f: importance.get(f, 0),
        reverse=True
    )

    # Top features are protected (core inside)
    n_protect = min(search_config.n_core_features, len(sorted_inside) // 2)
    core_inside = set(sorted_inside[:n_protect])
    swappable_inside = [f for f in sorted_inside if f not in core_inside and f not in protected]

    # Estimate total evaluations (rough upper bound)
    total_evals_estimate = min(
        max_swaps * len(swappable_inside) * len(outside),
        max_swaps * 50  # Cap estimate
    )

    if progress:
        progress.start_phase(
            "pairwise_swapping",
            total_evals=total_evals_estimate,
            is_estimate=True
        )

    completed_evals = 0
    total_swaps = 0

    # Extract data and configs for joblib workers
    X = evaluator._X
    y = evaluator._y
    model_config = evaluator.model_config
    cv_config = evaluator.cv_config
    metric_config = evaluator.metric_config

    while total_swaps < max_swaps:
        # Generate all swap candidates for this iteration
        swap_candidates = []
        for f_in in swappable_inside:
            if f_in not in current_set:
                continue
            for f_out in outside:
                if f_out in current_set:
                    continue
                swap_candidates.append((f_in, f_out))

        if not swap_candidates:
            break

        start_time = time.time()
        current_features_list = list(current_set)

        # Evaluate all swap candidates in parallel
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(_evaluate_swap_joblib)(
                X, y, current_features_list, f_in, f_out,
                model_config, cv_config, metric_config, search_config
            )
            for f_in, f_out in swap_candidates
        )

        eval_time = time.time() - start_time
        completed_evals += len(swap_candidates)

        if progress:
            progress.update(
                completed=completed_evals,
                best_metric=best_score,
                best_subset_size=len(current_set),
                eval_time=eval_time / max(1, len(swap_candidates))
            )

        # Find best swap that improves over current score
        best_swap = None
        best_swap_score = best_score + epsilon  # Must beat this to be accepted

        for f_in, f_out, result in results:
            if result is None:
                continue
            if result.metric_main > best_swap_score:
                best_swap_score = result.metric_main
                best_swap = (f_in, f_out, result)

        if best_swap is None:
            # No improvement found
            break

        # Apply best swap
        f_in, f_out, best_swap_result = best_swap
        current_set = (current_set - {f_in}) | {f_out}
        best_score = best_swap_score
        best_result = best_swap_result
        total_swaps += 1

        # Update swappable features after swap
        importance = evaluator.get_feature_importance(
            list(current_set), importance_type='gain'
        )
        sorted_inside = sorted(
            current_set,
            key=lambda f: importance.get(f, 0),
            reverse=True
        )
        n_protect = min(search_config.n_core_features, len(sorted_inside) // 2)
        core_inside = set(sorted_inside[:n_protect])
        swappable_inside = [
            f for f in sorted_inside
            if f not in core_inside and f not in protected
        ]

        # Update outside candidates
        outside = [f for f in outside_candidates if f not in current_set]

        gc.collect()

    if progress:
        progress.end_phase(final_metric=best_score)

    return current_set, best_result


def _cached_evaluate(
    evaluator: SubsetEvaluator,
    features: Set[str],
    cache: Optional[EvaluationCache]
) -> SubsetResult:
    """Evaluate with optional caching.

    Args:
        evaluator: Subset evaluator.
        features: Features to evaluate.
        cache: Optional cache.

    Returns:
        Evaluation result.
    """
    if cache is not None:
        cached = cache.get(features)
        if cached is not None:
            return cached

    result = evaluator.evaluate(features)

    if cache is not None:
        cache.put(features, result)

    return result


class TopKTracker:
    """Tracks the top K best feature subsets.

    Maintains a sorted list of the best subsets encountered during
    the search process.

    Attributes:
        k: Maximum number of subsets to track.
    """

    def __init__(self, k: int = 10):
        """Initialize the tracker.

        Args:
            k: Number of top subsets to maintain.
        """
        self.k = k
        self._subsets: List[SubsetResult] = []

    def update(self, result: SubsetResult):
        """Potentially add a new result to the top-K.

        Args:
            result: Evaluation result to consider.
        """
        # Check if this exact feature set is already tracked
        feature_set = frozenset(result.features)
        for existing in self._subsets:
            if frozenset(existing.features) == feature_set:
                # Update if better
                if result.metric_main > existing.metric_main:
                    self._subsets.remove(existing)
                    break
                else:
                    return

        self._subsets.append(result)
        self._subsets.sort(key=lambda r: r.metric_main, reverse=True)

        # Trim to K
        if len(self._subsets) > self.k:
            self._subsets = self._subsets[:self.k]

    def get_best(self) -> Optional[SubsetResult]:
        """Get the best subset.

        Returns:
            Best SubsetResult or None if empty.
        """
        return self._subsets[0] if self._subsets else None

    def get_top_k(self, k: Optional[int] = None) -> List[SubsetResult]:
        """Get the top K subsets.

        Args:
            k: Number to return (defaults to all tracked).

        Returns:
            List of top SubsetResult objects.
        """
        n = k or self.k
        return self._subsets[:n]

    def __len__(self) -> int:
        return len(self._subsets)


def add_interaction_features_forward(
    evaluator: SubsetEvaluator,
    base_features: Set[str],
    interaction_features: List[str],
    unused_base_features: List[str],
    search_config: SearchConfig,
    cache: Optional[EvaluationCache] = None,
    progress: Optional[ProgressTracker] = None
) -> Tuple[Set[str], SubsetResult]:
    """Forward selection to add interaction features.

    Args:
        evaluator: Configured SubsetEvaluator.
        base_features: Current base feature set.
        interaction_features: Available interaction features.
        unused_base_features: Unused base features to also consider.
        search_config: Search configuration.
        cache: Optional evaluation cache.
        progress: Optional progress tracker.

    Returns:
        Tuple of (extended_features, best_result).
    """
    # Combine candidates: unused base features + interaction features
    candidates = unused_base_features + interaction_features

    # Track interaction count
    max_interactions = search_config.max_interaction_features

    # Use forward selection with interaction tracking
    current_set = set(base_features)
    best_result = _cached_evaluate(evaluator, current_set, cache)
    best_score = best_result.metric_main

    n_interactions_added = 0

    if progress:
        progress.start_phase(
            "interaction_forward_selection",
            total_evals=len(candidates) * (len(candidates) + 1) // 2,
            is_estimate=True
        )

    completed_evals = 0

    while len(current_set) < search_config.max_features:
        best_candidate = None
        best_candidate_score = best_score
        best_candidate_result = None
        is_interaction = False

        for candidate in candidates:
            if candidate in current_set:
                continue

            # Check interaction limit
            candidate_is_interaction = '_x_' in candidate or '*' in candidate
            if candidate_is_interaction and n_interactions_added >= max_interactions:
                continue

            start_time = time.time()

            test_set = current_set | {candidate}
            result = _cached_evaluate(evaluator, test_set, cache)

            completed_evals += 1
            eval_time = time.time() - start_time

            if result.metric_main > best_candidate_score + search_config.epsilon_add:
                best_candidate = candidate
                best_candidate_score = result.metric_main
                best_candidate_result = result
                is_interaction = candidate_is_interaction

            if progress:
                progress.update(
                    completed=completed_evals,
                    best_metric=best_score,
                    best_subset_size=len(current_set),
                    eval_time=eval_time,
                    extra_info={'n_interactions': n_interactions_added}
                )

        if best_candidate is not None:
            current_set.add(best_candidate)
            best_score = best_candidate_score
            best_result = best_candidate_result
            if is_interaction:
                n_interactions_added += 1
        else:
            break

        gc.collect()

    if progress:
        progress.end_phase(final_metric=best_score)

    return current_set, best_result


# =============================================================================
# Parallel Interaction Search (Enhanced)
# =============================================================================

def _evaluate_interaction_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    base_features: Set[str],
    candidate: InteractionCandidate,
    evaluator: SubsetEvaluator,
    model_config: ModelConfig,
    baseline_score: float
) -> Tuple[InteractionCandidate, float, Optional[SubsetResult]]:
    """Evaluate a single interaction candidate (worker function for parallel execution).

    Args:
        X: Feature DataFrame.
        y: Target Series.
        base_features: Current base feature set.
        candidate: InteractionCandidate to evaluate.
        evaluator: SubsetEvaluator instance.
        model_config: Model configuration.
        baseline_score: Baseline metric to compare against.

    Returns:
        Tuple of (candidate, delta_improvement, SubsetResult or None).
    """
    try:
        # Lazily compute the interaction feature
        name, values = compute_interaction_lazily(X, candidate)

        # Check for excessive NaNs
        if values.isna().mean() > 0.5:
            return (candidate, -999.0, None)

        # Create temporary DataFrame with interaction feature
        X_temp = X[list(base_features)].copy()
        X_temp[name] = values

        # Create temporary evaluator for this evaluation
        from .evaluation import SubsetEvaluator as TempEvaluator
        temp_evaluator = TempEvaluator(
            X=X_temp,
            y=y,
            model_config=model_config,
            cv_config=evaluator.cv_config,
            metric_config=evaluator.metric_config,
            search_config=evaluator.search_config,
            regime=evaluator._regime
        )

        # Evaluate base + interaction
        test_features = set(base_features) | {name}
        result = temp_evaluator.evaluate(test_features)

        delta = result.metric_main - baseline_score

        # Cleanup
        del X_temp
        gc.collect()

        return (candidate, delta, result)

    except Exception as e:
        return (candidate, -999.0, None)


def _evaluate_interaction_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    base_features_list: List[str],
    candidate: InteractionCandidate,
    model_config,
    cv_config,
    metric_config,
    search_config,
    regime,
    baseline_score: float
) -> Tuple[InteractionCandidate, float, Optional[SubsetResult]]:
    """Joblib-compatible worker for parallel interaction evaluation.

    Creates a fresh evaluator in each worker process to avoid pickling issues.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        base_features_list: List of current base feature names.
        candidate: InteractionCandidate to evaluate.
        model_config: Model configuration.
        cv_config: CV configuration.
        metric_config: Metric configuration.
        search_config: Search configuration.
        regime: Regime labels (or None).
        baseline_score: Baseline metric to compare against.

    Returns:
        Tuple of (candidate, delta_improvement, SubsetResult or None).
    """
    try:
        # Lazily compute the interaction feature
        name, values = compute_interaction_lazily(X, candidate)

        # Check for excessive NaNs
        if values.isna().mean() > 0.5:
            return (candidate, -999.0, None)

        # Create temporary DataFrame with interaction feature
        X_temp = X[base_features_list].copy()
        X_temp[name] = values

        # Create fresh evaluator in worker process
        from .evaluation import SubsetEvaluator

        temp_evaluator = SubsetEvaluator(
            X=X_temp,
            y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
            regime=regime
        )

        # Evaluate base + interaction
        test_features = set(base_features_list) | {name}
        result = temp_evaluator.evaluate(test_features)

        delta = result.metric_main - baseline_score

        # Cleanup
        del X_temp
        gc.collect()

        return (candidate, delta, result)

    except Exception:
        return (candidate, -999.0, None)


def parallel_interaction_forward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    evaluator: SubsetEvaluator,
    base_features: Set[str],
    interaction_candidates: List[InteractionCandidate],
    search_config: SearchConfig,
    model_config: ModelConfig,
    cache: Optional[EvaluationCache] = None,
    progress: Optional[ProgressTracker] = None
) -> Tuple[Set[str], SubsetResult, List[str], PairwiseInteractionSearch]:
    """Parallel forward selection for interaction features.

    Evaluates interaction candidates in parallel batches, selecting those
    that improve the metric above the threshold.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        evaluator: Configured SubsetEvaluator.
        base_features: Current base feature set.
        interaction_candidates: List of InteractionCandidate objects.
        search_config: Search configuration.
        model_config: Model configuration.
        cache: Optional evaluation cache.
        progress: Optional progress tracker.

    Returns:
        Tuple of (extended_features, best_result, selected_interaction_names, pairwise_search).
    """
    n_jobs = search_config.get_effective_interaction_n_jobs()
    batch_size = search_config.interaction_batch_size
    epsilon = search_config.epsilon_add
    max_interactions = search_config.max_interaction_features

    # Initialize tracking
    pairwise_search = PairwiseInteractionSearch(search_config)
    pairwise_search.candidates = interaction_candidates

    current_features = set(base_features)
    selected_interactions: List[str] = []

    # Evaluate baseline
    baseline_result = _cached_evaluate(evaluator, current_features, cache)
    baseline_score = baseline_result.metric_main
    best_result = baseline_result
    best_score = baseline_score

    total_candidates = len(interaction_candidates)

    if progress:
        progress.start_phase(
            "parallel_interaction_search",
            total_evals=total_candidates,
            is_estimate=False
        )

    completed_evals = 0
    best_delta = 0.0
    best_pair_name = ""
    phase_start_time = time.time()

    # Process candidates in batches
    candidates_to_process = list(interaction_candidates)

    while len(selected_interactions) < max_interactions and candidates_to_process:
        # Take next batch
        batch = candidates_to_process[:batch_size]
        candidates_to_process = candidates_to_process[batch_size:]

        # Parallel evaluation of batch
        batch_results = []

        if search_config.parallel_over_interactions and n_jobs > 1:
            # Use joblib with loky backend for true process-based parallelism
            from joblib import Parallel, delayed

            # Get configs from evaluator for worker processes
            cv_config = evaluator.cv_config
            metric_config = evaluator.metric_config
            search_cfg = evaluator.search_config
            regime = evaluator._regime
            base_features_list = list(current_features)

            # Run parallel evaluations
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(_evaluate_interaction_joblib)(
                    X, y, base_features_list, cand,
                    model_config, cv_config, metric_config, search_cfg,
                    regime, best_score
                )
                for cand in batch
            )

            # Process results
            for result in results:
                if result is not None:
                    batch_results.append(result)
                completed_evals += 1

            # Update progress after batch
            if progress:
                elapsed = time.time() - phase_start_time
                avg_time = elapsed / completed_evals if completed_evals > 0 else 1.0
                progress.update(
                    completed=completed_evals,
                    best_metric=best_score,
                    best_subset_size=len(current_features),
                    eval_time=avg_time,
                    extra_info={
                        'best_delta': best_delta,
                        'best_pair': best_pair_name,
                        'n_interactions': len(selected_interactions)
                    }
                )
        else:
            # Sequential fallback
            for cand in batch:
                start_time = time.time()
                result = _evaluate_interaction_candidate(
                    X, y, current_features, cand, evaluator, model_config, best_score
                )
                batch_results.append(result)
                completed_evals += 1
                eval_time = time.time() - start_time

                if progress:
                    progress.update(
                        completed=completed_evals,
                        best_metric=best_score,
                        best_subset_size=len(current_features),
                        eval_time=eval_time,
                        extra_info={
                            'best_delta': best_delta,
                            'best_pair': best_pair_name,
                            'n_interactions': len(selected_interactions)
                        }
                    )

        # Process batch results
        for cand, delta, result in batch_results:
            pairwise_search.record_evaluation(cand, delta, result)

            if delta > best_delta:
                best_delta = delta
                best_pair_name = cand.feature_name

            # Select if improvement exceeds threshold
            if delta > epsilon and result is not None:
                # Re-compute lazily and add to current features
                try:
                    name, values = compute_interaction_lazily(X, cand)
                    X[name] = values  # Add to DataFrame

                    current_features.add(name)
                    selected_interactions.append(name)
                    pairwise_search.select_interaction(cand)

                    best_score = result.metric_main
                    best_result = result

                    # Update evaluator with new feature
                    # (evaluator already sees updated X since it holds a reference)
                except Exception:
                    pass

        # Cleanup after batch
        gc.collect()

    if progress:
        progress.end_phase(final_metric=best_score)

    return current_features, best_result, selected_interactions, pairwise_search


def late_interaction_refinement(
    X: pd.DataFrame,
    y: pd.Series,
    evaluator: SubsetEvaluator,
    current_features: Set[str],
    feature_ranking: FeatureRanking,
    search_config: SearchConfig,
    model_config: ModelConfig,
    cache: Optional[EvaluationCache] = None,
    progress: Optional[ProgressTracker] = None
) -> Tuple[Set[str], SubsetResult, List[str]]:
    """Run late-stage interaction refinement after swapping.

    This is a lighter-weight interaction search using stricter thresholds,
    run after the pairwise swapping phase to discover any remaining
    high-value interactions.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        evaluator: Configured SubsetEvaluator.
        current_features: Current feature set (post-swapping).
        feature_ranking: Feature ranking for prioritization.
        search_config: Search configuration.
        model_config: Model configuration.
        cache: Optional evaluation cache.
        progress: Optional progress tracker.

    Returns:
        Tuple of (final_features, best_result, new_interaction_names).
    """
    epsilon = search_config.epsilon_add_interaction_late
    max_late = search_config.max_late_interactions
    n_top_global = min(search_config.n_top_global_for_interactions, 15)

    # Generate fresh candidates based on current feature set
    base_list = list(current_features)
    top_global = feature_ranking.feature_names[:n_top_global]

    candidates = generate_interaction_candidates(
        base_features=base_list,
        top_global_features=top_global,
        feature_importance=feature_ranking.importance_scores,
        interaction_types=search_config.interaction_types,
        use_domain_filter=True,
        shap_interactions_df=None
    )

    # Limit to top candidates
    candidates = candidates[:search_config.n_top_interactions // 2]

    # Evaluate baseline
    baseline_result = _cached_evaluate(evaluator, current_features, cache)
    baseline_score = baseline_result.metric_main
    best_result = baseline_result
    best_score = baseline_score

    new_interactions: List[str] = []

    if progress:
        progress.start_phase(
            "late_interaction_refinement",
            total_evals=len(candidates),
            is_estimate=False
        )

    completed_evals = 0

    for cand in candidates:
        if len(new_interactions) >= max_late:
            break

        start_time = time.time()

        try:
            # Lazily compute interaction
            name, values = compute_interaction_lazily(X, cand)

            if values.isna().mean() > 0.5:
                continue

            # Create temp DataFrame
            X_temp = X[list(current_features)].copy()
            X_temp[name] = values

            # Evaluate
            from .evaluation import SubsetEvaluator as TempEvaluator
            temp_evaluator = TempEvaluator(
                X=X_temp,
                y=y,
                model_config=model_config,
                cv_config=evaluator.cv_config,
                metric_config=evaluator.metric_config,
                search_config=evaluator.search_config,
                regime=evaluator._regime
            )

            test_features = set(current_features) | {name}
            result = temp_evaluator.evaluate(test_features)

            delta = result.metric_main - best_score

            if delta > epsilon:
                # Add interaction
                X[name] = values
                current_features.add(name)
                new_interactions.append(name)
                best_score = result.metric_main
                best_result = result

            del X_temp
            gc.collect()

        except Exception:
            pass

        completed_evals += 1
        eval_time = time.time() - start_time

        if progress:
            progress.update(
                completed=completed_evals,
                best_metric=best_score,
                best_subset_size=len(current_features),
                eval_time=eval_time,
                extra_info={'n_late_interactions': len(new_interactions)}
            )

    if progress:
        progress.end_phase(final_metric=best_score)

    return current_features, best_result, new_interactions
