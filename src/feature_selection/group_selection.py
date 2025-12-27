"""Group-first feature selection algorithms.

This module implements group-first feature selection where entire hypothesis
groups are the atomic units of selection rather than individual features.

Key algorithms:
- grouped_forward_selection: Add entire groups if improvement > epsilon
- grouped_swap_selection: Deterministic group swaps (no randomness)
- grouped_backward_elimination: Remove entire groups if loss < epsilon
- interaction_group_selection: Select from curated INTERACTION_GROUPS

Design principles:
- Groups are atomic: all features in a group are added/removed together
- Deterministic: no randomness, reproducible results
- Baseline demotion: optionally allow core/head groups to be dropped
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import SearchConfig, SubsetResult
from .evaluation import SubsetEvaluator
from .progress import ProgressTracker
from .base_features import (
    CORE_GROUPS, HEAD_GROUPS, CANDIDATE_GROUPS, INTERACTION_GROUPS,
    INTERACTION_TEMPLATES,
    get_core_groups, get_head_groups, get_candidate_groups,
    get_baseline_groups, get_all_groups, get_interaction_groups,
    get_eligible_templates, generate_interaction_feature_names,
)
from .evaluation import EvaluationCache

try:
    from ..config.model_keys import ModelKey
except ImportError:
    from src.config.model_keys import ModelKey


# =============================================================================
# DATA CLASSES FOR GROUP SELECTION RESULTS
# =============================================================================

@dataclass
class GroupResult:
    """Result of evaluating a single group."""
    group_name: str
    features: List[str]
    metric_before: float
    metric_after: float
    delta: float
    accepted: bool
    reason: str = ""


@dataclass
class GroupSelectionResult:
    """Complete result of group-first selection."""
    selected_groups: Dict[str, List[str]]
    selected_features: List[str]
    final_metric: float
    baseline_metric: float

    # Secondary metrics (mean, std) tuples
    final_secondary_metrics: Dict[str, tuple] = field(default_factory=dict)
    baseline_secondary_metrics: Dict[str, tuple] = field(default_factory=dict)

    # Per-phase results
    forward_results: List[GroupResult] = field(default_factory=list)
    swap_results: List[GroupResult] = field(default_factory=list)
    backward_results: List[GroupResult] = field(default_factory=list)
    interaction_results: List[GroupResult] = field(default_factory=list)

    # Tracking
    total_groups_evaluated: int = 0
    total_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML output."""
        return {
            "selected_groups": self.selected_groups,
            "selected_features": self.selected_features,
            "final_metric": self.final_metric,
            "baseline_metric": self.baseline_metric,
            "improvement": self.final_metric - self.baseline_metric,
            "final_secondary_metrics": {k: {"mean": v[0], "std": v[1]} for k, v in self.final_secondary_metrics.items()},
            "baseline_secondary_metrics": {k: {"mean": v[0], "std": v[1]} for k, v in self.baseline_secondary_metrics.items()},
            "n_groups": len(self.selected_groups),
            "n_features": len(self.selected_features),
            "forward_additions": [r.__dict__ for r in self.forward_results if r.accepted],
            "swaps_made": [r.__dict__ for r in self.swap_results if r.accepted],
            "backward_removals": [r.__dict__ for r in self.backward_results if r.accepted],
            "interactions_added": [r.__dict__ for r in self.interaction_results if r.accepted],
            "total_groups_evaluated": self.total_groups_evaluated,
            "total_time_seconds": self.total_time_seconds,
        }


@dataclass
class GroupSelectionConfig:
    """Configuration for group-first selection."""
    # Thresholds for selection decisions
    epsilon_add: float = 0.002       # Min improvement to add a group
    epsilon_swap: float = 0.001      # Min improvement to accept a swap
    epsilon_remove: float = 0.001    # Max loss allowed when removing a group
    epsilon_drop: float = 0.0005     # Min improvement to drop a group (for add/drop moves)
    epsilon_add_interaction: float = 0.0015  # Min improvement for interaction groups

    # Group constraints
    allow_baseline_demotions: bool = False  # Allow dropping core/head groups
    max_groups: int = 20             # Maximum total groups to select
    max_interaction_groups: int = 3  # Maximum interaction groups

    # Enhanced local search options
    enable_add_drop_moves: bool = True  # Enable add/drop moves in swap phase
    max_search_iterations: int = 50     # Max iterations for local search

    # Tabu mechanism (optional)
    enable_tabu: bool = False        # Enable tabu list to avoid cycling
    tabu_tenure: int = 5             # How many iterations to keep moves tabu
    tabu_aspiration_delta: float = 0.005  # Override tabu if improvement > this

    # K-of-N feature selection within groups
    enable_k_of_n: bool = True       # Enable K-of-N selection (False = use all features)
    group_k_default: int = 2         # Default K for all groups
    group_k: Dict[str, int] = field(default_factory=dict)  # Per-group K overrides
    epsilon_add_feature: float = 0.0005  # Min improvement to add feature within group

    # Caching
    enable_caching: bool = True      # Cache evaluation results
    cache_max_size: int = 10000      # Max cached evaluations

    # Parallelization
    n_jobs: int = 1                  # Parallelization (-1 for all cores)
    verbose: bool = True             # Print progress

    # Move evaluation parallelization (for local search)
    parallelize_moves: bool = True   # Parallelize candidate move evaluation
    n_move_workers: int = -1         # Workers for move eval (-1 = use n_jobs)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _flatten_groups(groups: Dict[str, List[str]]) -> List[str]:
    """Flatten a dict of groups to a deduplicated list of features."""
    seen = set()
    result = []
    for group_features in groups.values():
        for feat in group_features:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
    return result


def _variance_adjusted_acceptance(
    result_before: 'SubsetResult',
    result_after: 'SubsetResult',
    epsilon: float,
    c: float = 0.5,
    verbose: bool = False,
    context: str = "",
) -> Tuple[float, float, float, bool]:
    """
    Compute variance-adjusted acceptance for a move.

    A move is accepted if and only if:
        delta_mean > max(epsilon, c * delta_std)

    Where:
        - delta_i = metric_after_fold_i - metric_before_fold_i
        - delta_mean = mean(delta_i)
        - delta_std = std(delta_i)  (sample std, ddof=1)

    This reduces false positives from selection by requiring the improvement
    to exceed the noise floor estimated from fold-level variance.

    Args:
        result_before: SubsetResult from evaluation before the move.
        result_after: SubsetResult from evaluation after the move.
        epsilon: Base epsilon threshold (e.g., config.epsilon_add).
        c: Noise floor multiplier (default 0.5).
        verbose: Whether to print acceptance details.
        context: Context string for verbose logging.

    Returns:
        Tuple of (delta_mean, delta_std, threshold, accepted).
    """
    # Compute fold-level deltas
    n_folds = len(result_before.fold_metrics)
    if n_folds != len(result_after.fold_metrics):
        raise ValueError(
            f"Fold count mismatch: before={n_folds}, after={len(result_after.fold_metrics)}"
        )

    if n_folds == 0:
        # Fallback: use metric_main if no fold_metrics available
        delta_mean = result_after.metric_main - result_before.metric_main
        delta_std = 0.0
        threshold = epsilon
        accepted = delta_mean > threshold
        if verbose:
            print(f"    {context}: delta_mean={delta_mean:.5f} (no fold data), "
                  f"threshold={threshold:.5f}, accepted={accepted}")
        return delta_mean, delta_std, threshold, accepted

    deltas = [
        result_after.fold_metrics[i] - result_before.fold_metrics[i]
        for i in range(n_folds)
    ]

    delta_mean = float(np.mean(deltas))
    # Use sample std (ddof=1) for unbiased estimate
    delta_std = float(np.std(deltas, ddof=1)) if n_folds > 1 else 0.0

    threshold = max(epsilon, c * delta_std)
    accepted = delta_mean > threshold

    if verbose:
        print(f"    {context}: Δ_mean={delta_mean:.5f}, Δ_std={delta_std:.5f}, "
              f"thresh=max({epsilon:.4f}, {c}×{delta_std:.5f})={threshold:.5f}, "
              f"accepted={accepted}")

    return delta_mean, delta_std, threshold, accepted


def _groups_to_features(groups: Dict[str, List[str]]) -> List[str]:
    """Convert groups dict to flat feature list."""
    return _flatten_groups(groups)


def _get_k_for_group(group_name: str, config: GroupSelectionConfig) -> int:
    """Get the K value for a specific group (per-group override or default)."""
    return config.group_k.get(group_name, config.group_k_default)


def _select_k_features_for_group(
    evaluator: SubsetEvaluator,
    current_features: List[str],
    group_name: str,
    group_features: List[str],
    config: GroupSelectionConfig,
    cache: Optional[EvaluationCache] = None,
) -> Tuple[List[str], float]:
    """
    Select up to K best features from a group using greedy forward selection.

    This implements K-of-N selection within a single group. Starting from
    the current feature set (without any features from this group), we
    greedily add features from the group one at a time until either:
    - We reach K features
    - Adding more features doesn't improve by epsilon_add_feature

    The selection is fully deterministic: features are evaluated in sorted
    order, and we always pick the feature with the highest delta.

    Args:
        evaluator: Configured SubsetEvaluator
        current_features: Current feature set (excluding this group's features)
        group_name: Name of the group (for logging)
        group_features: All features available in this group
        config: Selection configuration
        cache: Optional evaluation cache

    Returns:
        Tuple of (selected_features, final_metric)
    """
    if not config.enable_k_of_n:
        # K-of-N disabled: return all features
        all_features = current_features + group_features
        result = _cached_evaluate(evaluator, all_features, cache, n_jobs=config.n_jobs)
        return group_features, result.metric_main

    k = _get_k_for_group(group_name, config)

    # Filter to features that exist in evaluator's data
    available = [f for f in sorted(group_features) if f in evaluator._all_features]
    if not available:
        # No features available - return empty
        result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
        return [], result.metric_main

    # Start with current features (no group features)
    baseline_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
    current_metric = baseline_result.metric_main

    selected_from_group = []
    remaining = list(available)

    # Greedy forward selection within the group
    while len(selected_from_group) < k and remaining:
        best_feature = None
        best_delta = -float('inf')
        best_metric = current_metric

        # Evaluate each remaining feature (deterministic order: sorted)
        for feat in remaining:
            test_features = current_features + selected_from_group + [feat]
            result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)
            delta = result.metric_main - current_metric

            if delta > best_delta:
                best_delta = delta
                best_feature = feat
                best_metric = result.metric_main

        # Accept if improvement >= epsilon_add_feature, or if it's the first feature
        # (we always want at least one feature from the group if any improve)
        if best_feature is not None:
            if len(selected_from_group) == 0 or best_delta >= config.epsilon_add_feature:
                selected_from_group.append(best_feature)
                remaining.remove(best_feature)
                current_metric = best_metric
            else:
                # No more improving features
                break

    return selected_from_group, current_metric


def _validate_no_duplicate_groups(
    *group_dicts: Dict[str, List[str]]
) -> None:
    """
    Validate that there are no duplicate group names across multiple group dicts.

    Raises ValueError if duplicates are found.
    """
    all_names = []
    for gd in group_dicts:
        all_names.extend(gd.keys())

    seen = set()
    duplicates = []
    for name in all_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)

    if duplicates:
        raise ValueError(f"Duplicate group names found: {duplicates}")


# =============================================================================
# GROUPED FORWARD SELECTION
# =============================================================================

def grouped_forward_selection(
    evaluator: SubsetEvaluator,
    baseline_groups: Dict[str, List[str]],
    candidate_groups: Dict[str, List[str]],
    config: GroupSelectionConfig,
    progress: Optional[ProgressTracker] = None,
    cache: Optional[EvaluationCache] = None,
) -> Tuple[Dict[str, List[str]], List[GroupResult], float]:
    """
    Forward selection at the group level.

    For each candidate group, add it to the current set and evaluate.
    Accept if improvement >= epsilon_add.

    When K-of-N is enabled, only the best K features from each group
    are selected using greedy forward selection within the group.

    Args:
        evaluator: Configured SubsetEvaluator
        baseline_groups: Starting groups (CORE + HEAD)
        candidate_groups: Groups available for selection
        config: Selection configuration
        progress: Optional progress tracker
        cache: Optional evaluation cache

    Returns:
        Tuple of (selected_groups, group_results, final_metric)
    """
    # Validate no duplicate group names
    _validate_no_duplicate_groups(baseline_groups, candidate_groups)

    selected = dict(baseline_groups)
    results = []

    # Initialize cache if not provided and caching is enabled
    if cache is None and config.enable_caching:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # If K-of-N is enabled, we need to re-select features for baseline groups too
    if config.enable_k_of_n:
        # Re-select features for each baseline group using K-of-N
        new_selected = {}
        current_features = []
        for group_name, group_features in sorted(baseline_groups.items()):
            selected_features, _ = _select_k_features_for_group(
                evaluator, current_features, group_name, group_features, config, cache
            )
            if selected_features:
                new_selected[group_name] = selected_features
                current_features = current_features + selected_features
        selected = new_selected

    # Evaluate baseline
    baseline_features = _groups_to_features(selected)
    baseline_result = _cached_evaluate(evaluator, baseline_features, cache, n_jobs=config.n_jobs)
    current_metric = baseline_result.metric_main
    current_result = baseline_result  # Track full result for variance-adjusted acceptance

    if config.verbose:
        k_info = ""
        if config.enable_k_of_n:
            k_info = f", K-of-N enabled (K={config.group_k_default})"
        print(f"Forward selection: baseline metric = {current_metric:.4f} "
              f"({len(selected)} groups, {len(baseline_features)} features{k_info})")

    # Try adding each candidate group (sorted for determinism)
    for group_name, group_features in sorted(candidate_groups.items()):
        if group_name in selected:
            continue  # Already in selected set

        if len(selected) >= config.max_groups:
            if config.verbose:
                print(f"  Max groups ({config.max_groups}) reached, stopping forward selection")
            break

        # Get current features (excluding this group)
        current_features = _groups_to_features(selected)

        if config.enable_k_of_n:
            # Use K-of-N selection to pick best features from this group
            selected_group_features, new_metric = _select_k_features_for_group(
                evaluator, current_features, group_name, group_features, config, cache
            )
            if not selected_group_features:
                # No features selected from this group
                if config.verbose:
                    print(f"  - Skipped '{group_name}': no features available or selected")
                continue
            features_to_add = selected_group_features
            # Re-evaluate to get full SubsetResult with fold_metrics for variance-adjusted check
            test_features = current_features + selected_group_features
            test_result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)
            new_metric = test_result.metric_main
        else:
            # Original behavior: add all features from the group
            test_groups = dict(selected)
            test_groups[group_name] = group_features
            test_features = _groups_to_features(test_groups)

            # Evaluate
            test_result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)
            new_metric = test_result.metric_main
            features_to_add = group_features

        # Variance-adjusted acceptance: delta_mean > max(epsilon, 0.5 * delta_std)
        delta_mean, delta_std, threshold, accepted = _variance_adjusted_acceptance(
            current_result, test_result, config.epsilon_add,
            verbose=config.verbose, context=f"Forward '{group_name}'"
        )

        result = GroupResult(
            group_name=group_name,
            features=features_to_add,
            metric_before=current_metric,
            metric_after=new_metric,
            delta=delta_mean,
            accepted=accepted,
            reason=f"Δ_mean={delta_mean:.4f}, Δ_std={delta_std:.4f}, thresh={threshold:.4f}"
        )
        results.append(result)

        if accepted:
            selected[group_name] = features_to_add
            current_metric = new_metric
            current_result = test_result  # Update current result for next iteration
            if config.verbose:
                k_info = ""
                if config.enable_k_of_n:
                    k_info = f" [{len(features_to_add)}/{len(group_features)} selected]"
                print(f"  + Added '{group_name}' ({len(features_to_add)} features){k_info}: "
                      f"Δ = {delta_mean:+.4f}, new metric = {current_metric:.4f}")
        elif config.verbose:
            print(f"  - Rejected '{group_name}': Δ_mean={delta_mean:+.4f} <= thresh={threshold:.4f}")

    return selected, results, current_metric


# =============================================================================
# GROUPED SWAP SELECTION (Enhanced with add/drop moves, caching, tabu)
# =============================================================================

@dataclass
class LocalSearchMove:
    """Represents a local search move."""
    move_type: str  # "swap", "add", "drop"
    group_out: Optional[str]  # Group removed (for swap/drop)
    group_in: Optional[str]   # Group added (for swap/add)
    features_in: Optional[List[str]]  # Features of added group
    delta: float
    new_metric: float

    def __str__(self):
        if self.move_type == "swap":
            return f"swap({self.group_out} -> {self.group_in})"
        elif self.move_type == "add":
            return f"add({self.group_in})"
        elif self.move_type == "drop":
            return f"drop({self.group_out})"
        return f"{self.move_type}(?)"


@dataclass
class MoveSpec:
    """Specification for a candidate move (before evaluation)."""
    move_type: str  # "swap", "add", "drop"
    group_out: Optional[str]
    group_in: Optional[str]
    group_in_features: Optional[List[str]]  # Full feature list for group_in
    base_features: List[str]  # Features after removal (for swap/drop) or current (for add)
    threshold: float  # Acceptance threshold (epsilon)
    is_tabu: bool  # Whether this move is tabu

    def sort_key(self) -> tuple:
        """Key for deterministic ordering."""
        return (self.move_type, self.group_out or "", self.group_in or "")


@dataclass
class MoveResult:
    """Result of evaluating a move spec."""
    spec: MoveSpec
    delta: float
    new_metric: float
    features_in: Optional[List[str]]  # Selected features (may be K-of-N subset)
    result: Optional[SubsetResult]  # Full result with fold_metrics for variance-adjusted acceptance

    def sort_key(self) -> tuple:
        """Key for deterministic tie-breaking: best delta first, then by spec."""
        # Negative delta for descending sort, then by spec for tie-breaking
        return (-self.delta, -self.new_metric, self.spec.sort_key())


def _evaluate_move_spec(
    spec: MoveSpec,
    evaluator: SubsetEvaluator,
    current_metric: float,
    config: GroupSelectionConfig,
    cache: Optional[EvaluationCache] = None,
) -> MoveResult:
    """
    Evaluate a single move specification.

    This is the worker function for parallel move evaluation. It takes a
    MoveSpec and returns a MoveResult with the evaluated delta/metric.

    For swap/add moves with K-of-N enabled, this function performs the
    K-of-N selection within the incoming group.

    Note: This function returns the raw SubsetResult for variance-adjusted
    acceptance to be computed by the caller (who has the current_result).

    Args:
        spec: Move specification to evaluate
        evaluator: SubsetEvaluator (shared via threading backend)
        current_metric: Current metric value before the move
        config: Selection configuration
        cache: Optional evaluation cache (shared via threading backend)

    Returns:
        MoveResult with evaluation results and full SubsetResult
    """
    if spec.move_type == "drop":
        # Drop move: just evaluate the base features (without the dropped group)
        test_features = spec.base_features
        result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)
        delta = result.metric_main - current_metric
        new_metric = result.metric_main

        return MoveResult(
            spec=spec,
            delta=delta,
            new_metric=new_metric,
            features_in=None,
            result=result,
        )

    # Swap or Add move: need to evaluate adding group_in
    if config.enable_k_of_n and spec.group_in_features:
        # Use K-of-N selection to pick best features from the incoming group
        selected_features, new_metric = _select_k_features_for_group(
            evaluator, spec.base_features, spec.group_in, spec.group_in_features,
            config, cache
        )
        if not selected_features:
            # No features could be selected - return non-accepted result
            return MoveResult(
                spec=spec,
                delta=-float('inf'),
                new_metric=current_metric,
                features_in=None,
                result=None,
            )
        # Re-evaluate to get full SubsetResult with fold_metrics
        test_features = spec.base_features + selected_features
        result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)
        delta = result.metric_main - current_metric
        new_metric = result.metric_main
        features_for_move = selected_features
    else:
        # Original behavior: use all features from the group
        test_features = spec.base_features + (spec.group_in_features or [])
        result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)
        delta = result.metric_main - current_metric
        new_metric = result.metric_main
        features_for_move = spec.group_in_features

    return MoveResult(
        spec=spec,
        delta=delta,
        new_metric=new_metric,
        features_in=features_for_move,
        result=result,
    )


def _cached_evaluate(
    evaluator: SubsetEvaluator,
    features: List[str],
    cache: Optional[EvaluationCache] = None,
    n_jobs: int = 1
) -> SubsetResult:
    """Evaluate with optional caching."""
    if cache is not None:
        cached = cache.get(features)
        if cached is not None:
            return cached
        result = evaluator.evaluate(features, n_jobs=n_jobs)
        cache.put(features, result)
        return result
    return evaluator.evaluate(features, n_jobs=n_jobs)


def _generate_move_specs(
    selected: Dict[str, List[str]],
    candidate_groups: Dict[str, List[str]],
    baseline_group_names: Set[str],
    config: GroupSelectionConfig,
    tabu_set: Optional[Set[str]] = None,
) -> List[MoveSpec]:
    """
    Generate all candidate move specifications (pure Python, no evaluation).

    This separates move generation from evaluation to enable parallel evaluation.

    Args:
        selected: Currently selected groups
        candidate_groups: Groups available for swapping/adding
        baseline_group_names: Names of baseline groups (core + head)
        config: Selection configuration
        tabu_set: Set of group names that are currently tabu

    Returns:
        List of MoveSpec objects to evaluate
    """
    specs = []

    # Get groups that can be removed
    removable_groups = list(selected.keys())
    if not config.allow_baseline_demotions:
        removable_groups = [g for g in removable_groups if g not in baseline_group_names]

    def is_tabu(group_name: str) -> bool:
        if tabu_set is None:
            return False
        return group_name in tabu_set

    # Current features for add moves
    current_features = _groups_to_features(selected)

    # ---------------------------
    # SWAP MOVES: remove g, add h
    # ---------------------------
    for group_out in sorted(removable_groups):  # sorted for determinism
        # Base features after removing group_out
        test_groups = {k: v for k, v in selected.items() if k != group_out}
        base_features = _groups_to_features(test_groups)

        for group_in, group_in_features in sorted(candidate_groups.items()):
            if group_in in selected:
                continue
            if group_in == group_out:
                continue

            move_is_tabu = is_tabu(group_out) or is_tabu(group_in)
            threshold = config.tabu_aspiration_delta if move_is_tabu else config.epsilon_swap

            specs.append(MoveSpec(
                move_type="swap",
                group_out=group_out,
                group_in=group_in,
                group_in_features=group_in_features,
                base_features=base_features,
                threshold=threshold,
                is_tabu=move_is_tabu,
            ))

    # ---------------------------
    # ADD MOVES: add h (if enabled and room)
    # ---------------------------
    if config.enable_add_drop_moves and len(selected) < config.max_groups:
        for group_in, group_in_features in sorted(candidate_groups.items()):
            if group_in in selected:
                continue

            move_is_tabu = is_tabu(group_in)
            threshold = config.tabu_aspiration_delta if move_is_tabu else config.epsilon_add

            specs.append(MoveSpec(
                move_type="add",
                group_out=None,
                group_in=group_in,
                group_in_features=group_in_features,
                base_features=current_features,
                threshold=threshold,
                is_tabu=move_is_tabu,
            ))

    # ---------------------------
    # DROP MOVES: remove g (if enabled and more than 1 group)
    # ---------------------------
    if config.enable_add_drop_moves and len(selected) > 1:
        for group_out in sorted(removable_groups):
            # Base features after removing group_out
            test_groups = {k: v for k, v in selected.items() if k != group_out}
            base_features = _groups_to_features(test_groups)

            move_is_tabu = is_tabu(group_out)
            threshold = config.tabu_aspiration_delta if move_is_tabu else config.epsilon_drop

            specs.append(MoveSpec(
                move_type="drop",
                group_out=group_out,
                group_in=None,
                group_in_features=None,
                base_features=base_features,
                threshold=threshold,
                is_tabu=move_is_tabu,
            ))

    return specs


def _find_best_move(
    evaluator: SubsetEvaluator,
    selected: Dict[str, List[str]],
    candidate_groups: Dict[str, List[str]],
    baseline_group_names: Set[str],
    current_metric: float,
    current_result: SubsetResult,
    config: GroupSelectionConfig,
    cache: Optional[EvaluationCache] = None,
    tabu_list: Optional[Set[str]] = None,
) -> Optional[LocalSearchMove]:
    """
    Find the best improving move among swap, add, and drop.

    This function generates all candidate moves, evaluates them (in parallel
    if configured), and returns the best move with deterministic tie-breaking.

    Uses variance-adjusted acceptance: delta_mean > max(epsilon, 0.5 * delta_std)

    Args:
        evaluator: SubsetEvaluator for feature evaluation
        selected: Currently selected groups
        candidate_groups: Groups available for swapping/adding
        baseline_group_names: Names of baseline groups (core + head)
        current_metric: Current metric value
        current_result: Current SubsetResult (for variance-adjusted acceptance)
        config: Selection configuration
        cache: Optional evaluation cache
        tabu_list: Set of group names that are currently tabu

    Returns:
        The best improving move, or None if no improving move exists.
    """
    # Step 1: Generate all candidate move specs
    specs = _generate_move_specs(
        selected, candidate_groups, baseline_group_names, config, tabu_list
    )

    if not specs:
        return None

    # Step 2: Evaluate moves (parallel or sequential)
    n_workers = config.n_move_workers if config.n_move_workers != -1 else config.n_jobs

    if config.parallelize_moves and n_workers != 1 and len(specs) > 1:
        # Parallel evaluation using threading backend (shared memory for evaluator/cache)
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_evaluate_move_spec)(spec, evaluator, current_metric, config, cache)
            for spec in specs
        )
    else:
        # Sequential evaluation
        results = [
            _evaluate_move_spec(spec, evaluator, current_metric, config, cache)
            for spec in specs
        ]

    # Step 3: Apply variance-adjusted acceptance to each result
    # Filter to moves that pass variance-adjusted threshold
    accepted_results = []
    for r in results:
        if r.result is None:
            # No valid result (e.g., K-of-N found no features)
            continue

        # Determine the appropriate epsilon for this move type
        if r.spec.move_type == "swap":
            epsilon = config.epsilon_swap
        elif r.spec.move_type == "add":
            epsilon = config.epsilon_add
        elif r.spec.move_type == "drop":
            epsilon = config.epsilon_drop
        else:
            epsilon = config.epsilon_swap

        # Use tabu aspiration threshold if move is tabu
        if r.spec.is_tabu:
            epsilon = config.tabu_aspiration_delta

        # Variance-adjusted acceptance
        delta_mean, delta_std, threshold, accepted = _variance_adjusted_acceptance(
            current_result, r.result, epsilon,
            verbose=False,  # Avoid noisy output during move search
        )

        if accepted and delta_mean > 0:
            # Update delta with the properly computed delta_mean
            accepted_results.append((r, delta_mean, delta_std, threshold))

    if not accepted_results:
        return None

    # Sort by delta_mean (descending), then by spec for deterministic tie-breaking
    accepted_results.sort(key=lambda x: (-x[1], -x[0].new_metric, x[0].spec.sort_key()))
    best_result, best_delta, best_std, best_thresh = accepted_results[0]

    if config.verbose:
        move_str = str(LocalSearchMove(
            move_type=best_result.spec.move_type,
            group_out=best_result.spec.group_out,
            group_in=best_result.spec.group_in,
            features_in=best_result.features_in,
            delta=best_delta,
            new_metric=best_result.new_metric,
        ))
        print(f"    Best move: {move_str}, Δ_mean={best_delta:.5f}, Δ_std={best_std:.5f}, "
              f"thresh={best_thresh:.5f}")

    # Convert MoveResult back to LocalSearchMove
    return LocalSearchMove(
        move_type=best_result.spec.move_type,
        group_out=best_result.spec.group_out,
        group_in=best_result.spec.group_in,
        features_in=best_result.features_in,
        delta=best_delta,  # Use variance-adjusted delta
        new_metric=best_result.new_metric,
    )


def _update_tabu_list(
    tabu_list: Dict[str, int],
    move: LocalSearchMove,
    current_iteration: int,
    tabu_tenure: int
) -> Dict[str, int]:
    """Update tabu list after applying a move."""
    # Add moved groups to tabu
    if move.group_out is not None:
        tabu_list[move.group_out] = current_iteration + tabu_tenure
    if move.group_in is not None:
        tabu_list[move.group_in] = current_iteration + tabu_tenure

    # Remove expired entries
    expired = [g for g, exp in tabu_list.items() if exp <= current_iteration]
    for g in expired:
        del tabu_list[g]

    return tabu_list


def grouped_swap_selection(
    evaluator: SubsetEvaluator,
    current_groups: Dict[str, List[str]],
    candidate_groups: Dict[str, List[str]],
    baseline_group_names: Set[str],
    config: GroupSelectionConfig,
    progress: Optional[ProgressTracker] = None,
    cache: Optional[EvaluationCache] = None,
) -> Tuple[Dict[str, List[str]], List[GroupResult], float]:
    """
    Enhanced deterministic local search over group sets.

    Supports multiple move types:
    - swap: remove selected group g, add unselected group h
    - add: add unselected group h (if delta >= epsilon_add)
    - drop: remove selected group g (if delta >= epsilon_drop)

    Features:
    - Deterministic: no randomness, reproducible results
    - Caching: evaluation results are cached to avoid redundant CV
    - Optional tabu: prevents cycling in local minima
    - Logging: detailed move-by-move logging

    Args:
        evaluator: Configured SubsetEvaluator
        current_groups: Current selected groups
        candidate_groups: Groups available for swapping in
        baseline_group_names: Names of baseline groups (core + head)
        config: Selection configuration
        progress: Optional progress tracker
        cache: Optional evaluation cache

    Returns:
        Tuple of (final_groups, move_results, final_metric)
    """
    selected = dict(current_groups)
    results = []

    # Initialize cache if not provided and caching is enabled
    if cache is None and config.enable_caching:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # Initialize tabu list (maps group_name -> expiration iteration)
    tabu_dict: Dict[str, int] = {}

    # Evaluate current
    current_features = _groups_to_features(selected)
    current_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
    current_metric = current_result.metric_main
    start_metric = current_metric

    if config.verbose:
        move_types = "swap"
        if config.enable_add_drop_moves:
            move_types = "swap/add/drop"
        k_info = ""
        if config.enable_k_of_n:
            k_info = f", K-of-N=True (K={config.group_k_default})"
        print(f"Local search: starting metric = {current_metric:.4f} "
              f"(moves: {move_types}, tabu: {config.enable_tabu}{k_info})")

    iteration = 0

    while iteration < config.max_search_iterations:
        iteration += 1

        # Build active tabu set
        tabu_set = None
        if config.enable_tabu:
            tabu_set = {g for g, exp in tabu_dict.items() if exp > iteration}

        # Find best move (using variance-adjusted acceptance)
        best_move = _find_best_move(
            evaluator, selected, candidate_groups, baseline_group_names,
            current_metric, current_result, config, cache, tabu_set
        )

        if best_move is None:
            if config.verbose:
                print(f"  Iteration {iteration}: no improving move found, stopping")
            break

        # Apply the move
        if best_move.move_type == "swap":
            del selected[best_move.group_out]
            selected[best_move.group_in] = best_move.features_in
        elif best_move.move_type == "add":
            selected[best_move.group_in] = best_move.features_in
        elif best_move.move_type == "drop":
            del selected[best_move.group_out]

        # Update tabu list
        if config.enable_tabu:
            tabu_dict = _update_tabu_list(
                tabu_dict, best_move, iteration, config.tabu_tenure
            )

        # Record result
        result = GroupResult(
            group_name=str(best_move),
            features=best_move.features_in or [],
            metric_before=current_metric,
            metric_after=best_move.new_metric,
            delta=best_move.delta,
            accepted=True,
            reason=f"iter {iteration}: {best_move.move_type}"
        )
        results.append(result)

        # Update current state for next iteration
        current_metric = best_move.new_metric
        # Re-evaluate to get full SubsetResult with fold_metrics for next iteration
        current_features = _groups_to_features(selected)
        current_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)

        if config.verbose:
            print(f"  Iteration {iteration}: {best_move} "
                  f"Δ={best_move.delta:+.4f} -> metric={current_metric:.4f}")

    if config.verbose:
        total_improvement = current_metric - start_metric
        cache_info = ""
        if cache is not None:
            cache_info = f", cache_size={len(cache)}"
        print(f"  Local search complete: {len(results)} moves, "
              f"improvement={total_improvement:+.4f}{cache_info}")

    return selected, results, current_metric


# =============================================================================
# GROUPED BACKWARD ELIMINATION
# =============================================================================

def grouped_backward_elimination(
    evaluator: SubsetEvaluator,
    current_groups: Dict[str, List[str]],
    baseline_group_names: Set[str],
    config: GroupSelectionConfig,
    progress: Optional[ProgressTracker] = None,
    cache: Optional[EvaluationCache] = None,
) -> Tuple[Dict[str, List[str]], List[GroupResult], float]:
    """
    Backward elimination at the group level.

    For each group in current set, try removing it.
    Remove if loss <= epsilon_remove (i.e., minimal harm).

    Args:
        evaluator: Configured SubsetEvaluator
        current_groups: Current selected groups
        baseline_group_names: Names of baseline groups (core + head)
        config: Selection configuration
        progress: Optional progress tracker
        cache: Optional evaluation cache

    Returns:
        Tuple of (final_groups, removal_results, final_metric)
    """
    selected = dict(current_groups)
    results = []

    # Initialize cache if not provided and caching is enabled
    if cache is None and config.enable_caching:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # Evaluate current
    current_features = _groups_to_features(selected)
    current_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
    current_metric = current_result.metric_main

    if config.verbose:
        print(f"Backward elimination: starting metric = {current_metric:.4f}")

    improved = True

    while improved:
        improved = False

        # Get groups that can be removed
        removable_groups = list(selected.keys())
        if not config.allow_baseline_demotions:
            removable_groups = [g for g in removable_groups if g not in baseline_group_names]

        # Find group whose removal causes least harm (or even improvement)
        # Uses variance-adjusted acceptance: accept if loss <= max(epsilon, 0.5 * delta_std)
        best_removal = None
        best_delta_mean = -float('inf')

        for group_name in sorted(removable_groups):  # sorted for determinism
            if len(selected) <= 1:
                break  # Don't remove last group

            # Create test set without this group
            test_groups = {k: v for k, v in selected.items() if k != group_name}
            test_features = _groups_to_features(test_groups)

            # Evaluate
            test_result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)

            # Variance-adjusted acceptance for removal:
            # Accept if delta_mean >= -max(epsilon_remove, 0.5 * delta_std)
            # i.e., the loss is within the acceptable threshold
            delta_mean, delta_std, threshold, _ = _variance_adjusted_acceptance(
                current_result, test_result, config.epsilon_remove,
                verbose=False,
            )

            # For removal: accept if loss <= threshold (i.e., delta_mean >= -threshold)
            acceptable = delta_mean >= -threshold

            if acceptable and delta_mean > best_delta_mean:
                best_removal = (group_name, selected[group_name], test_result.metric_main,
                                delta_mean, delta_std, threshold, test_result)
                best_delta_mean = delta_mean

        if best_removal is not None:
            group_name, group_features, new_metric, delta_mean, delta_std, threshold, new_result = best_removal

            result = GroupResult(
                group_name=group_name,
                features=group_features,
                metric_before=current_metric,
                metric_after=new_metric,
                delta=delta_mean,
                accepted=True,
                reason=f"Δ_mean={delta_mean:.4f} >= -{threshold:.4f} (loss acceptable)"
            )
            results.append(result)

            # Apply removal
            del selected[group_name]
            current_metric = new_metric
            current_result = new_result  # Update for next iteration
            improved = True

            if config.verbose:
                print(f"  - Removed '{group_name}' ({len(group_features)} features): "
                      f"Δ_mean={delta_mean:+.4f}, Δ_std={delta_std:.4f}, "
                      f"thresh={threshold:.4f}, new metric={current_metric:.4f}")

    if config.verbose:
        print(f"  Backward elimination complete: {len(results)} groups removed")

    return selected, results, current_metric


# =============================================================================
# INTERACTION GROUP SELECTION
# =============================================================================

def interaction_group_selection(
    evaluator: SubsetEvaluator,
    current_groups: Dict[str, List[str]],
    interaction_groups: Dict[str, List[str]],
    config: GroupSelectionConfig,
    progress: Optional[ProgressTracker] = None,
    cache: Optional[EvaluationCache] = None,
) -> Tuple[Dict[str, List[str]], List[GroupResult], float]:
    """
    Select from curated interaction groups (legacy interface).

    Unlike pairwise generation, this simply evaluates pre-curated
    interaction groups and adds them if they improve performance.

    Args:
        evaluator: Configured SubsetEvaluator
        current_groups: Current selected groups
        interaction_groups: Curated interaction groups to consider
        config: Selection configuration
        progress: Optional progress tracker
        cache: Optional evaluation cache

    Returns:
        Tuple of (final_groups, interaction_results, final_metric)
    """
    selected = dict(current_groups)
    results = []

    # Initialize cache if not provided and caching is enabled
    if cache is None and config.enable_caching:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # Evaluate current
    current_features = _groups_to_features(selected)
    current_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
    current_metric = current_result.metric_main

    if config.verbose:
        print(f"Interaction selection: starting metric = {current_metric:.4f}, "
              f"{len(interaction_groups)} candidate interaction groups")

    n_interactions_added = 0

    for group_name, group_features in sorted(interaction_groups.items()):  # sorted for determinism
        if n_interactions_added >= config.max_interaction_groups:
            if config.verbose:
                print(f"  Max interaction groups ({config.max_interaction_groups}) reached")
            break

        if group_name in selected:
            continue

        # Check if all features exist in evaluator's data
        available_features = [f for f in group_features if f in evaluator._all_features]
        if len(available_features) == 0:
            if config.verbose:
                print(f"  - Skipped '{group_name}': no features available in data")
            continue

        # Create test set with interaction group
        test_groups = dict(selected)
        test_groups[group_name] = available_features
        test_features = _groups_to_features(test_groups)

        # Evaluate
        test_result = _cached_evaluate(evaluator, test_features, cache, n_jobs=config.n_jobs)

        # Variance-adjusted acceptance: delta_mean > max(epsilon, 0.5 * delta_std)
        delta_mean, delta_std, threshold, accepted = _variance_adjusted_acceptance(
            current_result, test_result, config.epsilon_add,
            verbose=config.verbose, context=f"Interaction '{group_name}'"
        )

        result = GroupResult(
            group_name=group_name,
            features=available_features,
            metric_before=current_metric,
            metric_after=test_result.metric_main,
            delta=delta_mean,
            accepted=accepted,
            reason=f"Δ_mean={delta_mean:.4f}, Δ_std={delta_std:.4f}, thresh={threshold:.4f}"
        )
        results.append(result)

        if accepted:
            selected[group_name] = available_features
            current_metric = test_result.metric_main
            current_result = test_result  # Update for next iteration
            n_interactions_added += 1
            if config.verbose:
                print(f"  + Added interaction '{group_name}': "
                      f"Δ={delta_mean:+.4f}, new metric = {current_metric:.4f}")
        elif config.verbose:
            print(f"  - Rejected interaction '{group_name}': "
                  f"Δ_mean={delta_mean:+.4f} <= thresh={threshold:.4f}")

    return selected, results, current_metric


def template_interaction_selection(
    evaluator: SubsetEvaluator,
    X: pd.DataFrame,
    current_groups: Dict[str, List[str]],
    config: GroupSelectionConfig,
    progress: Optional[ProgressTracker] = None,
    cache: Optional[EvaluationCache] = None
) -> Tuple[Dict[str, List[str]], List[GroupResult], float]:
    """
    Template-based interaction selection.

    Determines eligible templates based on selected parent groups, computes
    interaction features on-the-fly, and selects the best interaction groups.

    This is the new template-driven approach that:
    1. Checks which parent groups are present in current_groups
    2. For eligible templates, generates interaction feature names
    3. Computes the interaction features dynamically
    4. Evaluates adding each template as an interaction "group atom"
    5. Greedily adds up to max_interaction_groups

    Args:
        evaluator: Configured SubsetEvaluator
        X: Feature DataFrame (will be modified to add interaction features)
        current_groups: Current selected groups (keys = group names)
        config: Selection configuration
        progress: Optional progress tracker
        cache: Optional evaluation cache

    Returns:
        Tuple of (final_groups, interaction_results, final_metric)
    """
    from src.features.interaction_production import compute_template_interactions

    selected = dict(current_groups)
    results = []

    # Get names of currently selected groups
    selected_group_names = set(selected.keys())

    # Evaluate current baseline (need full SubsetResult for variance-adjusted acceptance)
    current_features = _groups_to_features(selected)
    if cache is not None:
        cached = cache.get(current_features)
        if cached:
            current_result = cached
            current_metric = cached.metric_main
        else:
            current_result = evaluator.evaluate(current_features, n_jobs=config.n_jobs)
            cache.put(current_features, current_result)
            current_metric = current_result.metric_main
    else:
        current_result = evaluator.evaluate(current_features, n_jobs=config.n_jobs)
        current_metric = current_result.metric_main

    # Find eligible templates (both parent groups must be present)
    eligible_templates = get_eligible_templates(selected_group_names)

    if config.verbose:
        print(f"Template interaction selection: starting metric = {current_metric:.4f}")
        print(f"  Selected groups: {len(selected_group_names)}")
        print(f"  Eligible templates: {len(eligible_templates)}")

    if not eligible_templates:
        if config.verbose:
            print("  No eligible templates (parent groups not selected)")
        return selected, results, current_metric

    n_interactions_added = 0

    # Evaluate each eligible template
    for template_name in eligible_templates:
        if n_interactions_added >= config.max_interaction_groups:
            if config.verbose:
                print(f"  Max interaction groups ({config.max_interaction_groups}) reached")
            break

        if template_name in selected:
            continue

        template = INTERACTION_TEMPLATES[template_name]
        parent_a, parent_b = template["parents"]

        # Generate feature names for this template
        template_feature_names = generate_interaction_feature_names(template_name, template)

        # Compute interaction features if not already present in X
        missing_features = [f for f in template_feature_names if f not in X.columns]
        if missing_features:
            X, computed = compute_template_interactions(
                X, template_name, template, skip_missing=True, inplace=True
            )
            if config.verbose and computed:
                print(f"    Computed {len(computed)} interaction features for {template_name}")

        # Check which features are actually available
        available_features = [f for f in template_feature_names if f in X.columns]
        if len(available_features) == 0:
            if config.verbose:
                print(f"  - Skipped template '{template_name}': no features could be computed")
            continue

        # Update evaluator's X with new interaction features
        # We need to update the evaluator's internal data
        for feat in available_features:
            if feat not in evaluator._X.columns:
                evaluator._X[feat] = X[feat]
                evaluator._feature_to_idx[feat] = len(evaluator._feature_to_idx)
                evaluator._all_features.append(feat)

        # Create test set with interaction group
        test_groups = dict(selected)
        test_groups[template_name] = available_features
        test_features = _groups_to_features(test_groups)

        # Evaluate
        if cache is not None:
            cached = cache.get(test_features)
            if cached:
                test_result = cached
            else:
                test_result = evaluator.evaluate(test_features, n_jobs=config.n_jobs)
                cache.put(test_features, test_result)
        else:
            test_result = evaluator.evaluate(test_features, n_jobs=config.n_jobs)

        # Variance-adjusted acceptance: delta_mean > max(epsilon, 0.5 * delta_std)
        # Use interaction-specific epsilon threshold
        delta_mean, delta_std, threshold, accepted = _variance_adjusted_acceptance(
            current_result, test_result, config.epsilon_add_interaction,
            verbose=config.verbose, context=f"Template '{template_name}'"
        )

        result = GroupResult(
            group_name=template_name,
            features=available_features,
            metric_before=current_metric,
            metric_after=test_result.metric_main,
            delta=delta_mean,
            accepted=accepted,
            reason=f"Δ_mean={delta_mean:.4f}, Δ_std={delta_std:.4f}, thresh={threshold:.4f} "
                   f"(parents: {parent_a}, {parent_b})"
        )
        results.append(result)

        if accepted:
            selected[template_name] = available_features
            current_metric = test_result.metric_main
            current_result = test_result  # Update for next iteration
            n_interactions_added += 1
            if config.verbose:
                print(f"  + Added template '{template_name}': "
                      f"{len(available_features)} features, "
                      f"Δ={delta_mean:+.4f}, new metric = {current_metric:.4f}")
                print(f"      Parents: {parent_a}, {parent_b}")
        elif config.verbose:
            print(f"  - Rejected template '{template_name}': "
                  f"Δ_mean={delta_mean:+.4f} <= thresh={threshold:.4f}")

    if config.verbose:
        print(f"  Template selection complete: added {n_interactions_added} interaction groups")

    return selected, results, current_metric


# =============================================================================
# MAIN GROUP SELECTION PIPELINE
# =============================================================================

def run_group_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    model_config,
    cv_config,
    metric_config,
    search_config: SearchConfig,
    config: Optional[GroupSelectionConfig] = None,
    progress: Optional[ProgressTracker] = None
) -> GroupSelectionResult:
    """
    Run the complete group-first selection pipeline.

    Pipeline stages:
    1. Start with baseline groups (CORE + HEAD)
    2. Forward selection from candidate groups
    3. Deterministic swaps between groups
    4. Backward elimination of non-essential groups
    5. Add curated interaction groups

    Args:
        X: Feature DataFrame
        y: Target Series
        model_key: Model key (LONG_NORMAL, etc.)
        model_config: Model configuration
        cv_config: CV configuration
        metric_config: Metric configuration
        search_config: Search configuration
        config: Group selection configuration
        progress: Optional progress tracker

    Returns:
        GroupSelectionResult with selected groups and metrics
    """
    if config is None:
        config = GroupSelectionConfig(
            epsilon_add=search_config.epsilon_add,
            epsilon_swap=search_config.epsilon_swap,
        )

    start_time = time.time()

    # Create evaluator
    from .evaluation import SubsetEvaluator
    evaluator = SubsetEvaluator(
        X=X,
        y=y,
        model_config=model_config,
        cv_config=cv_config,
        metric_config=metric_config,
        search_config=search_config,
    )

    # Get groups
    baseline_groups = get_baseline_groups(model_key)
    candidate_groups = get_candidate_groups()
    interaction_groups = get_interaction_groups()

    baseline_group_names = set(baseline_groups.keys())

    # Create shared cache for all stages
    cache = None
    if config.enable_caching:
        cache = EvaluationCache(max_size=config.cache_max_size)

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"GROUP-FIRST SELECTION for {model_key.value}")
        print(f"{'='*60}")
        print(f"Baseline: {len(baseline_groups)} groups, "
              f"{len(_groups_to_features(baseline_groups))} features")
        print(f"Candidates: {len(candidate_groups)} groups")
        print(f"Interactions: {len(interaction_groups)} groups")
        if config.enable_k_of_n:
            print(f"K-of-N: ENABLED (default K={config.group_k_default})")
            if config.group_k:
                print(f"  Per-group K overrides: {config.group_k}")
        print()

    # Evaluate baseline
    baseline_features = _groups_to_features(baseline_groups)
    baseline_result = _cached_evaluate(evaluator, baseline_features, cache, n_jobs=config.n_jobs)
    baseline_metric = baseline_result.metric_main
    baseline_secondary = baseline_result.secondary_metrics.copy() if baseline_result.secondary_metrics else {}

    if config.verbose:
        print(f"Baseline metrics:")
        print(f"  AUC:        {baseline_metric:.4f} ± {baseline_result.metric_std:.4f}")
        for metric_name, (mean, std) in baseline_secondary.items():
            print(f"  {metric_name.upper():10s} {mean:.4f} ± {std:.4f}")

    # Stage 1: Forward selection
    if config.verbose:
        print(f"\n--- Stage 1: Forward Selection ---")
    selected, forward_results, metric = grouped_forward_selection(
        evaluator, baseline_groups, candidate_groups, config, progress, cache
    )

    # Stage 2: Swap selection
    if config.verbose:
        print(f"\n--- Stage 2: Swap Selection ---")
    selected, swap_results, metric = grouped_swap_selection(
        evaluator, selected, candidate_groups, baseline_group_names, config, progress, cache
    )

    # Stage 3: Backward elimination
    if config.verbose:
        print(f"\n--- Stage 3: Backward Elimination ---")
    selected, backward_results, metric = grouped_backward_elimination(
        evaluator, selected, baseline_group_names, config, progress, cache
    )

    # Stage 4: Interaction groups
    if config.verbose:
        print(f"\n--- Stage 4: Interaction Selection ---")
    selected, interaction_results, metric = interaction_group_selection(
        evaluator, selected, interaction_groups, config, progress, cache
    )

    total_time = time.time() - start_time

    # Evaluate final set to get secondary metrics
    final_features = _groups_to_features(selected)
    final_result = _cached_evaluate(evaluator, final_features, cache, n_jobs=config.n_jobs)
    final_metric = final_result.metric_main
    final_secondary = final_result.secondary_metrics.copy() if final_result.secondary_metrics else {}

    # Build result
    result = GroupSelectionResult(
        selected_groups=selected,
        selected_features=final_features,
        final_metric=final_metric,
        baseline_metric=baseline_metric,
        final_secondary_metrics=final_secondary,
        baseline_secondary_metrics=baseline_secondary,
        forward_results=forward_results,
        swap_results=swap_results,
        backward_results=backward_results,
        interaction_results=interaction_results,
        total_groups_evaluated=(
            len(forward_results) + len(swap_results) +
            len(backward_results) + len(interaction_results)
        ),
        total_time_seconds=total_time,
    )

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"SELECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Final: {len(selected)} groups, {len(result.selected_features)} features")
        print()

        # Display metrics comparison
        print(f"{'Metric':<15} {'Baseline':>15} {'Final':>15} {'Improvement':>15}")
        print(f"{'-'*60}")
        print(f"{'AUC':<15} {baseline_metric:>15.4f} {final_metric:>15.4f} {final_metric - baseline_metric:>+15.4f}")

        # Display secondary metrics with improvement
        all_metrics = set(baseline_secondary.keys()) | set(final_secondary.keys())
        for metric_name in sorted(all_metrics):
            base_val = baseline_secondary.get(metric_name, (0, 0))[0]
            final_val = final_secondary.get(metric_name, (0, 0))[0]
            # For brier/log_loss, lower is better so show negative improvement as positive
            improvement = final_val - base_val
            print(f"{metric_name:<15} {base_val:>15.4f} {final_val:>15.4f} {improvement:>+15.4f}")

        print()
        print(f"Time: {total_time:.1f}s")
        print()

    return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def select_groups_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    model_config,
    cv_config,
    metric_config,
    search_config: SearchConfig,
    allow_baseline_demotions: bool = False,
    max_groups: int = 20,
    verbose: bool = True
) -> GroupSelectionResult:
    """
    Convenience function to run group selection for a single model.

    Args:
        X: Feature DataFrame
        y: Target Series
        model_key: Model key (LONG_NORMAL, etc.)
        model_config: Model configuration
        cv_config: CV configuration
        metric_config: Metric configuration
        search_config: Search configuration
        allow_baseline_demotions: Allow dropping core/head groups
        max_groups: Maximum total groups
        verbose: Print progress

    Returns:
        GroupSelectionResult
    """
    config = GroupSelectionConfig(
        epsilon_add=search_config.epsilon_add,
        epsilon_swap=search_config.epsilon_swap,
        epsilon_remove=search_config.epsilon_swap,  # Use swap epsilon for removals
        allow_baseline_demotions=allow_baseline_demotions,
        max_groups=max_groups,
        verbose=verbose,
    )

    return run_group_selection(
        X=X,
        y=y,
        model_key=model_key,
        model_config=model_config,
        cv_config=cv_config,
        metric_config=metric_config,
        search_config=search_config,
        config=config,
    )
