"""Group-first feature selection algorithms.

This module implements group-first feature selection where entire hypothesis
groups are the atomic units of selection rather than individual features.

Key algorithms:
- grouped_forward_selection: Add entire groups if improvement > epsilon
- grouped_swap_selection: Deterministic group swaps (no randomness)
- grouped_backward_elimination: Remove entire groups if loss < epsilon
- interaction_group_selection: Select from curated INTERACTION_GROUPS
- run_outer_cv: Walk-forward outer CV wrapper for robustness estimation

Design principles:
- Groups are atomic: all features in a group are added/removed together
- Deterministic: no randomness, reproducible results
- Baseline demotion: optionally allow core/head groups to be dropped

Outer CV Note:
- Outer CV reduces feature-selection bias by treating selection as a learned procedure
- 3 outer folds is sufficient to detect overfitting without excessive computation
- This is a robustness check, not a production training loop
"""

import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import SearchConfig, SubsetResult
from .evaluation import SubsetEvaluator
from .progress import ProgressTracker
from .parallel_config import get_joblib_kwargs, get_loky_kwargs
from .executor import parallel_map, shutdown_executor, get_worker_stats, reset_worker_stats
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
    # Thresholds for selection decisions (epsilon = minimum improvement in metric units)
    epsilon_add: float = 0.0001      # Min improvement to add a group
    epsilon_swap: float = 0.0005     # Min improvement to accept a swap
    epsilon_remove: float = 0.001    # Max loss allowed when removing a group
    epsilon_drop: float = 0.0005     # Min improvement to drop a group (for add/drop moves)
    epsilon_add_interaction: float = 0.0015  # Min improvement for interaction groups

    # Signal-to-noise thresholds (t = delta_mean / SE, measures improvement reliability)
    # A move is accepted if: delta_mean > epsilon AND t > t_threshold
    # Higher t means more confident the improvement is real, not noise
    t_add: float = 0.5               # t-threshold for add moves (lenient: want exploration)
    t_swap: float = 1.0              # t-threshold for swap moves (moderate)
    t_drop: float = 1.0              # t-threshold for drop moves (moderate)
    t_add_interaction: float = 0.5   # t-threshold for interaction group adds

    # Debug flag for acceptance diagnostics
    debug_acceptance: bool = False   # Print detailed acceptance diagnostics per move

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
    k_of_n_seed: Optional[int] = 42  # Seed for randomizing group order in K-of-N (None=alphabetical)

    # Caching
    enable_caching: bool = True      # Cache evaluation results
    cache_max_size: int = 10000      # Max cached evaluations

    # Parallelization
    n_jobs: int = 1                  # Parallelization (-1 for all cores)
    verbose: bool = True             # Print progress

    # Move evaluation parallelization (for local search)
    parallelize_moves: bool = True   # Parallelize candidate move evaluation
    n_move_workers: int = -1         # Workers for move eval (-1 = use n_jobs)

    # Process-based parallelism for CPU-bound move evaluation
    # Threading backend is ineffective here due to Python GIL - LightGBM training
    # is CPU-bound and threads cannot run in parallel. Loky (process-based)
    # backend spawns separate processes that bypass GIL.
    use_loky_for_moves: bool = True  # Use loky (processes) instead of threading

    # Debug flag for parallelism verification
    debug_parallelism: bool = False  # Print PID from worker processes


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


@dataclass
class AcceptanceResult:
    """Result of signal-to-noise acceptance check."""
    delta_mean: float      # Mean improvement across folds
    delta_std: float       # Std of per-fold improvements
    delta_se: float        # Standard error of mean improvement
    t_stat: float          # t-statistic = delta_mean / (delta_se + tiny)
    epsilon: float         # Epsilon threshold used
    t_threshold: float     # t-threshold used
    accepted: bool         # Whether move was accepted
    reason: str            # Human-readable reason


def _snr_acceptance(
    result_before: 'SubsetResult',
    result_after: 'SubsetResult',
    epsilon: float,
    t_threshold: float,
    debug: bool = False,
    context: str = "",
) -> AcceptanceResult:
    """
    Signal-to-noise ratio acceptance for a move using per-fold improvement deltas.

    WHY PER-FOLD DELTAS?
    --------------------
    We compute improvement RELATIVE TO THE BASELINE per fold:
        d_i = metric_after_fold_i - metric_before_fold_i

    This measures the IMPROVEMENT signal, not absolute metric variability.
    A feature that improves every fold by 0.001 has low noise in improvement,
    even if absolute AUC varies a lot (0.58, 0.62, 0.60...).

    ACCEPTANCE RULE (two-gate):
        accepted = (delta_mean > epsilon) AND (t_stat > t_threshold)

    Where:
        - delta_mean = mean(d_i)
        - delta_std = std(d_i, ddof=1)
        - delta_se = delta_std / sqrt(n_folds)
        - t_stat = delta_mean / (delta_se + 1e-12)

    The epsilon gate ensures minimum practical improvement.
    The t-stat gate ensures the improvement is reliable (high signal-to-noise).

    Args:
        result_before: SubsetResult from evaluation before the move (baseline).
        result_after: SubsetResult from evaluation after the move (candidate).
        epsilon: Minimum improvement in metric units (e.g., 0.0001 AUC points).
        t_threshold: Minimum t-statistic for acceptance (e.g., 0.5 for adds).
        debug: Whether to print detailed diagnostics.
        context: Context string for debug logging.

    Returns:
        AcceptanceResult with all computed statistics and decision.
    """
    TINY = 1e-12

    n_folds_before = len(result_before.fold_metrics)
    n_folds_after = len(result_after.fold_metrics)

    # Handle edge cases: fold count mismatch or no fold data
    if n_folds_before != n_folds_after or n_folds_before == 0:
        delta_mean = result_after.metric_main - result_before.metric_main
        reason = (f"fold mismatch {n_folds_before}→{n_folds_after}"
                  if n_folds_before != n_folds_after else "no fold data")
        # Fallback: only epsilon gate, no t-stat gate
        accepted = bool(delta_mean > epsilon)  # Convert numpy bool to Python bool
        result = AcceptanceResult(
            delta_mean=delta_mean,
            delta_std=0.0,
            delta_se=0.0,
            t_stat=float('inf') if delta_mean > 0 else float('-inf'),
            epsilon=epsilon,
            t_threshold=t_threshold,
            accepted=accepted,
            reason=f"FALLBACK ({reason}): Δ={delta_mean:.5f} {'>' if accepted else '≤'} ε={epsilon:.5f}",
        )
        if debug:
            print(f"    {context}: {result.reason}")
        return result

    # Compute per-fold improvement deltas (this is the key insight!)
    deltas = [
        result_after.fold_metrics[i] - result_before.fold_metrics[i]
        for i in range(n_folds_before)
    ]

    n_folds = n_folds_before
    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas, ddof=1)) if n_folds > 1 else 0.0
    delta_se = delta_std / np.sqrt(n_folds) if n_folds > 1 else 0.0
    t_stat = delta_mean / (delta_se + TINY)

    # Two-gate acceptance:
    # Gate 1: Improvement exceeds minimum practical threshold (epsilon)
    # Gate 2: Improvement is reliable (t-stat exceeds threshold)
    passes_epsilon = delta_mean > epsilon
    passes_t = t_stat > t_threshold
    accepted = bool(passes_epsilon and passes_t)  # Convert numpy bool to Python bool

    # Build human-readable reason
    if accepted:
        reason = f"ACCEPT: Δ={delta_mean:.5f}>ε={epsilon:.5f}, t={t_stat:.2f}>{t_threshold:.2f}"
    elif not passes_epsilon:
        reason = f"REJECT (epsilon): Δ={delta_mean:.5f}≤ε={epsilon:.5f}"
    else:
        reason = f"REJECT (t-stat): t={t_stat:.2f}≤{t_threshold:.2f}"

    result = AcceptanceResult(
        delta_mean=delta_mean,
        delta_std=delta_std,
        delta_se=delta_se,
        t_stat=t_stat,
        epsilon=epsilon,
        t_threshold=t_threshold,
        accepted=accepted,
        reason=reason,
    )

    if debug:
        print(f"    {context}: Δ_mean={delta_mean:.5f}, Δ_std={delta_std:.5f}, "
              f"SE={delta_se:.5f}, t={t_stat:.2f}, ε={epsilon:.5f}, "
              f"t_thresh={t_threshold:.2f} → {reason}")

    return result


def _get_acceptance_thresholds(
    move_type: str,
    config: GroupSelectionConfig,
    is_tabu: bool = False,
) -> Tuple[float, float]:
    """
    Get epsilon and t-threshold for a given move type.

    Args:
        move_type: One of "add", "swap", "drop", "add_interaction"
        config: GroupSelectionConfig with thresholds
        is_tabu: If True, use tabu_aspiration_delta as epsilon override

    Returns:
        Tuple of (epsilon, t_threshold)
    """
    if move_type == "add":
        epsilon = config.epsilon_add
        t_threshold = config.t_add
    elif move_type == "swap":
        epsilon = config.epsilon_swap
        t_threshold = config.t_swap
    elif move_type == "drop":
        epsilon = config.epsilon_drop
        t_threshold = config.t_drop
    elif move_type == "add_interaction":
        epsilon = config.epsilon_add_interaction
        t_threshold = config.t_add_interaction
    else:
        # Fallback
        epsilon = config.epsilon_add
        t_threshold = config.t_add

    # Tabu aspiration: if move is tabu, require higher epsilon to override
    if is_tabu:
        epsilon = config.tabu_aspiration_delta

    return epsilon, t_threshold


def _variance_adjusted_acceptance(
    result_before: 'SubsetResult',
    result_after: 'SubsetResult',
    epsilon: float,
    t_threshold: float = 0.5,
    verbose: bool = False,
    context: str = "",
) -> Tuple[float, float, float, bool]:
    """
    Legacy wrapper for backward compatibility. Use _snr_acceptance directly for new code.

    Returns:
        Tuple of (delta_mean, delta_std, threshold, accepted).
        Note: threshold is now epsilon (the t-gate is separate).
    """
    result = _snr_acceptance(
        result_before=result_before,
        result_after=result_after,
        epsilon=epsilon,
        t_threshold=t_threshold,
        debug=verbose,
        context=context,
    )
    return result.delta_mean, result.delta_std, epsilon, result.accepted


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
    force_sequential: bool = False,
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
        force_sequential: If True, disable parallelism (use when already in worker)

    Returns:
        Tuple of (selected_features, final_metric)
    """
    # When force_sequential=True (in worker process), CV should run sequentially
    # LightGBM uses model_config.num_threads for its internal threading
    eval_n_jobs = 1 if force_sequential else config.n_jobs

    if not config.enable_k_of_n:
        # K-of-N disabled: return all features
        all_features = current_features + group_features
        result = _cached_evaluate(evaluator, all_features, cache, n_jobs=eval_n_jobs)
        return group_features, result.metric_main

    k = _get_k_for_group(group_name, config)

    # Filter to features that exist in evaluator's data
    available = [f for f in sorted(group_features) if f in evaluator._all_features]
    if not available:
        # No features available - return empty
        result = _cached_evaluate(evaluator, current_features, cache, n_jobs=eval_n_jobs)
        return [], result.metric_main

    # Start with current features (no group features)
    baseline_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=eval_n_jobs)
    current_metric = baseline_result.metric_main

    selected_from_group = []
    remaining = list(available)

    # Determine parallelization
    # force_sequential=True when called from a worker process to avoid nested parallelism
    n_workers = config.n_move_workers if config.n_move_workers != -1 else config.n_jobs
    use_parallel = config.parallelize_moves and n_workers > 1 and not force_sequential

    # Greedy forward selection within the group
    while len(selected_from_group) < k and remaining:
        best_feature = None
        best_delta = -float('inf')
        best_metric = current_metric

        # Build list of features to evaluate
        base_features = current_features + selected_from_group

        if use_parallel and len(remaining) > 1:
            # Parallel evaluation of all remaining features
            # Always use threading for K-of-N (quick per-feature tests)
            # Loky is only beneficial for longer-running group evaluations
            joblib_kwargs = get_joblib_kwargs(n_workers)
            cache_for_workers = cache  # Threading shares cache

            def eval_feature(feat):
                """Evaluate adding a single feature."""
                test_features = base_features + [feat]
                result = _cached_evaluate(evaluator, test_features, cache_for_workers, n_jobs=1)
                return feat, result.metric_main

            from joblib import Parallel, delayed
            results = Parallel(**joblib_kwargs)(
                delayed(eval_feature)(feat) for feat in remaining
            )

            # Find best
            for feat, metric in results:
                delta = metric - current_metric
                if delta > best_delta:
                    best_delta = delta
                    best_feature = feat
                    best_metric = metric
        else:
            # Sequential evaluation
            for feat in remaining:
                test_features = base_features + [feat]
                result = _cached_evaluate(evaluator, test_features, cache, n_jobs=eval_n_jobs)
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
# MOVE DATA CLASSES (used by forward and swap selection)
# =============================================================================

@dataclass
class LocalSearchMove:
    """Represents a local search move with SNR acceptance stats."""
    move_type: str  # "swap", "add", "drop"
    group_out: Optional[str]  # Group removed (for swap/drop)
    group_in: Optional[str]   # Group added (for swap/add)
    features_in: Optional[List[str]]  # Features of added group
    delta: float              # delta_mean from SNR acceptance
    new_metric: float         # Metric value after the move
    # SNR acceptance stats (optional, populated when available)
    t_stat: Optional[float] = None    # t-statistic = delta_mean / SE
    delta_std: Optional[float] = None  # Std of per-fold deltas

    def __str__(self):
        if self.move_type == "swap":
            return f"swap({self.group_out} -> {self.group_in})"
        elif self.move_type == "add":
            return f"add({self.group_in})"
        elif self.move_type == "drop":
            return f"drop({self.group_out})"
        return f"{self.move_type}(?)"

    def format_log(self, metric_before: Optional[float] = None) -> str:
        """Format a log entry showing both metric and SNR stats."""
        parts = [str(self)]
        if metric_before is not None:
            parts.append(f"metric: {metric_before:.4f} → {self.new_metric:.4f}")
        else:
            parts.append(f"metric: {self.new_metric:.4f}")
        parts.append(f"Δ={self.delta:+.5f}")
        if self.t_stat is not None:
            parts.append(f"t={self.t_stat:.2f}")
        if self.delta_std is not None:
            parts.append(f"σ={self.delta_std:.5f}")
        return " | ".join(parts)


@dataclass
class MoveSpec:
    """Specification for a candidate move (before evaluation)."""
    move_type: str  # "swap", "add", "drop"
    group_out: Optional[str]
    group_in: Optional[str]
    group_in_features: Optional[List[str]]  # Full feature list for group_in
    base_features: List[str]  # Features after removal (for swap/drop) or current (for add)
    is_tabu: bool = False  # Whether this move is tabu

    def sort_key(self) -> tuple:
        """Key for deterministic ordering."""
        return (self.move_type, self.group_out or "", self.group_in or "")


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
    Forward selection at the group level with parallel candidate evaluation.

    Each round, all remaining candidate groups are evaluated in parallel.
    The best group that passes the variance-adjusted threshold is added.
    Repeat until no group passes or max_groups reached.

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
    # Note: cache is disabled when using loky (processes don't share memory)
    if cache is None and config.enable_caching and not config.use_loky_for_moves:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # If K-of-N is enabled, we need to re-select features for baseline groups too
    if config.enable_k_of_n:
        # Re-select features for each baseline group using K-of-N
        new_selected = {}
        current_features = []
        n_baseline = len(baseline_groups)

        # Randomize group order to avoid bias from alphabetical ordering
        # (each group's K-of-N sees previously selected features as context)
        group_items = list(baseline_groups.items())
        if config.k_of_n_seed is not None:
            import random
            rng = random.Random(config.k_of_n_seed)
            rng.shuffle(group_items)
        else:
            # Fallback to alphabetical if no seed
            group_items = sorted(group_items)

        if config.verbose:
            order_info = f"seed={config.k_of_n_seed}" if config.k_of_n_seed is not None else "alphabetical"
            print(f"K-of-N baseline selection: {n_baseline} groups ({order_info})...", flush=True)

        for i, (group_name, group_features) in enumerate(group_items, 1):
            if config.verbose:
                print(f"  [{i}/{n_baseline}] Selecting K-of-{len(group_features)} for '{group_name}'...",
                      end='', flush=True)
            try:
                selected_features, _ = _select_k_features_for_group(
                    evaluator, current_features, group_name, group_features, config, cache
                )
            except Exception as e:
                # Log error but continue with empty selection for this group
                import traceback
                print(f"\n  ERROR in K-of-N for '{group_name}': {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                selected_features = []

            if selected_features:
                new_selected[group_name] = selected_features
                current_features = current_features + selected_features
                if config.verbose:
                    print(f" selected {len(selected_features)}/{len(group_features)}", flush=True)
            elif config.verbose:
                print(" (no features available)", flush=True)
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

    # Remaining candidates to try
    remaining_candidates = {k: v for k, v in candidate_groups.items() if k not in selected}

    # Iteratively add best group until none pass threshold or max reached
    round_num = 0
    while remaining_candidates and len(selected) < config.max_groups:
        round_num += 1
        current_features = _groups_to_features(selected)

        # Find best add move among all remaining candidates (parallel)
        best_move = _find_best_add_move(
            evaluator=evaluator,
            current_features=current_features,
            candidate_groups=remaining_candidates,
            current_metric=current_metric,
            current_result=current_result,
            config=config,
            cache=cache,
        )

        if best_move is None:
            if config.verbose:
                print(f"  Round {round_num}: No improving group found, stopping")
            break

        # Record result
        result = GroupResult(
            group_name=best_move.group_in,
            features=best_move.features_in,
            metric_before=current_metric,
            metric_after=best_move.new_metric,
            delta=best_move.delta,
            accepted=True,
            reason=f"Best of {len(remaining_candidates)} candidates"
        )
        results.append(result)

        # Accept the move
        selected[best_move.group_in] = best_move.features_in
        current_metric = best_move.new_metric
        # Re-evaluate to get full SubsetResult for next round's variance check
        new_features = _groups_to_features(selected)
        current_result = _cached_evaluate(evaluator, new_features, cache, n_jobs=config.n_jobs)

        # Remove from remaining
        del remaining_candidates[best_move.group_in]

        if config.verbose:
            k_info = ""
            orig_features = candidate_groups.get(best_move.group_in, [])
            if config.enable_k_of_n and len(best_move.features_in) != len(orig_features):
                k_info = f" [{len(best_move.features_in)}/{len(orig_features)} selected]"
            # Consolidated log: metric and SNR stats
            t_info = f", t={best_move.t_stat:.2f}" if best_move.t_stat is not None else ""
            print(f"  Round {round_num}: + Added '{best_move.group_in}' "
                  f"({len(best_move.features_in)} features){k_info}")
            print(f"      metric: {result.metric_before:.4f} → {current_metric:.4f} | "
                  f"Δ={best_move.delta:+.5f}{t_info} | "
                  f"{len(remaining_candidates)} candidates left")

    if len(selected) >= config.max_groups and config.verbose:
        print(f"  Max groups ({config.max_groups}) reached, stopping forward selection")

    return selected, results, current_metric


def _find_best_add_move(
    evaluator: SubsetEvaluator,
    current_features: List[str],
    candidate_groups: Dict[str, List[str]],
    current_metric: float,
    current_result: SubsetResult,
    config: GroupSelectionConfig,
    cache: Optional[EvaluationCache] = None,
) -> Optional[LocalSearchMove]:
    """
    Find the best add move among all candidate groups (parallel evaluation).

    Evaluates all candidate groups in parallel using loky backend,
    then returns the best one that passes variance-adjusted threshold.

    Args:
        evaluator: SubsetEvaluator for feature evaluation
        current_features: Currently selected features
        candidate_groups: Groups available for adding
        current_metric: Current metric value
        current_result: Current SubsetResult (for variance-adjusted acceptance)
        config: Selection configuration
        cache: Optional evaluation cache (None when using loky)

    Returns:
        Best add move, or None if no group passes threshold.
    """
    if not candidate_groups:
        return None

    # Generate add specs for all candidates
    specs = []
    for group_name, group_features in sorted(candidate_groups.items()):
        spec = MoveSpec(
            move_type="add",
            group_out=None,
            group_in=group_name,
            base_features=list(current_features),
            group_in_features=list(group_features),
            is_tabu=False,
        )
        specs.append(spec)

    if not specs:
        return None

    # Evaluate in parallel
    n_workers = config.n_move_workers if config.n_move_workers != -1 else config.n_jobs

    if config.parallelize_moves and n_workers != 1 and len(specs) > 1:
        if config.use_loky_for_moves:
            # Use reusable executor for loky (avoids pool churn and resource leaks)
            if config.debug_parallelism:
                print(f"[DEBUG forward] Evaluating {len(specs)} candidates with {n_workers} workers "
                      f"(reusable executor)", flush=True)

            # Create evaluation function that captures fixed args
            def eval_spec(spec):
                return _evaluate_move_spec(spec, evaluator, current_metric, config, None)

            move_results = parallel_map(
                eval_spec, specs, n_workers, debug=config.debug_parallelism
            )
            # Filter out None results from failed tasks
            move_results = [r for r in move_results if r is not None]
        else:
            # Threading backend (for non-loky)
            joblib_kwargs = get_joblib_kwargs(n_workers)
            if config.debug_parallelism:
                print(f"[DEBUG forward] Evaluating {len(specs)} candidates with {n_workers} workers "
                      f"(threading)", flush=True)

            from joblib import Parallel, delayed
            move_results = Parallel(**joblib_kwargs)(
                delayed(_evaluate_move_spec)(spec, evaluator, current_metric, config, cache)
                for spec in specs
            )
    else:
        # Sequential evaluation
        move_results = [
            _evaluate_move_spec(spec, evaluator, current_metric, config, cache)
            for spec in specs
        ]

    # Apply SNR-based acceptance and find best
    accepted_moves = []
    for r in move_results:
        if r.result is None:
            continue

        # SNR-based acceptance for add moves
        epsilon, t_threshold = _get_acceptance_thresholds("add", config)
        acceptance = _snr_acceptance(
            current_result, r.result, epsilon, t_threshold,
            debug=config.debug_acceptance,
            context=f"add {r.spec.group_in}" if r.spec.group_in else "add",
        )

        if acceptance.accepted:
            move = LocalSearchMove(
                move_type="add",
                group_out=None,
                group_in=r.spec.group_in,
                features_in=r.features_in or r.spec.group_in_features,
                delta=acceptance.delta_mean,
                new_metric=r.new_metric,
                t_stat=acceptance.t_stat,
                delta_std=acceptance.delta_std,
            )
            accepted_moves.append(move)

    if not accepted_moves:
        return None

    # Deterministic tie-breaking: best delta, then group name
    accepted_moves.sort(key=lambda m: (-m.delta, m.group_in))
    return accepted_moves[0]


# =============================================================================
# GROUPED SWAP SELECTION (Enhanced with add/drop moves, caching, tabu)
# =============================================================================

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

    When using loky backend (config.use_loky_for_moves=True), this function
    runs in a separate process. The evaluator must be picklable and cache
    is not shared across processes (should be None when using loky).

    Note: This function returns the raw SubsetResult for variance-adjusted
    acceptance to be computed by the caller (who has the current_result).

    Args:
        spec: Move specification to evaluate
        evaluator: SubsetEvaluator (pickled to worker process with loky)
        current_metric: Current metric value before the move
        config: Selection configuration
        cache: Optional evaluation cache (None when using loky, shared when threading)

    Returns:
        MoveResult with evaluation results and full SubsetResult.
        On exception, returns MoveResult with result=None and delta=-inf.
    """
    try:
        return _evaluate_move_spec_impl(spec, evaluator, current_metric, config, cache)
    except Exception as e:
        # Capture exception but don't let it bubble up to crash the pool
        import traceback
        tb = traceback.format_exc()
        if config.debug_parallelism:
            spec_key = f"{spec.move_type}:{spec.group_out or 'None'}->{spec.group_in or 'None'}"
            print(f"[eval] pid={os.getpid()} EXCEPTION in {spec_key}: {e}\n{tb}", flush=True)
        # Return a safe result that will be filtered out
        return MoveResult(
            spec=spec,
            delta=-float('inf'),
            new_metric=current_metric,
            features_in=None,
            result=None,
        )


def _evaluate_move_spec_impl(
    spec: MoveSpec,
    evaluator: SubsetEvaluator,
    current_metric: float,
    config: GroupSelectionConfig,
    cache: Optional[EvaluationCache] = None,
) -> MoveResult:
    """Implementation of move evaluation (called by wrapper with exception handling)."""
    # Force single-threaded execution inside worker to avoid oversubscription
    # Each loky worker runs one evaluation; parallelism comes from workers not threads
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Debug logging: print PID to verify parallel processes are running
    if config.debug_parallelism:
        spec_key = f"{spec.move_type}:{spec.group_out or 'None'}->{spec.group_in or 'None'}"
        print(f"[eval] pid={os.getpid()} spec={spec_key}", flush=True)

    # When running in a loky worker, CV folds should run sequentially (n_jobs=1)
    # LightGBM uses model_config.num_threads for its internal threading
    eval_n_jobs = 1  # Always sequential in worker process

    if spec.move_type == "drop":
        # Drop move: just evaluate the base features (without the dropped group)
        test_features = spec.base_features
        result = _cached_evaluate(evaluator, test_features, cache, n_jobs=eval_n_jobs)
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
    # First, filter out any features from group_in that are already in base_features
    # to avoid duplicate feature errors in LightGBM
    base_set = set(spec.base_features)
    group_in_features = [f for f in (spec.group_in_features or []) if f not in base_set]

    if not group_in_features:
        # All features from this group are already present - nothing to add
        return MoveResult(
            spec=spec,
            delta=-float('inf'),
            new_metric=current_metric,
            features_in=None,
            result=None,
        )

    if config.enable_k_of_n and group_in_features:
        # Use K-of-N selection to pick best features from the incoming group
        # force_sequential=True because we're already in a worker process
        # (called from parallel move evaluation) - avoid nested parallelism
        selected_features, new_metric = _select_k_features_for_group(
            evaluator, spec.base_features, spec.group_in, group_in_features,
            config, cache, force_sequential=True
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
        result = _cached_evaluate(evaluator, test_features, cache, n_jobs=eval_n_jobs)
        delta = result.metric_main - current_metric
        new_metric = result.metric_main
        features_for_move = selected_features
    else:
        # Original behavior: use all features from the group
        test_features = spec.base_features + group_in_features
        result = _cached_evaluate(evaluator, test_features, cache, n_jobs=eval_n_jobs)
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

            specs.append(MoveSpec(
                move_type="swap",
                group_out=group_out,
                group_in=group_in,
                group_in_features=group_in_features,
                base_features=base_features,
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

            specs.append(MoveSpec(
                move_type="add",
                group_out=None,
                group_in=group_in,
                group_in_features=group_in_features,
                base_features=current_features,
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

            specs.append(MoveSpec(
                move_type="drop",
                group_out=group_out,
                group_in=None,
                group_in_features=None,
                base_features=base_features,
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

    Uses variance-adjusted acceptance: delta_mean > max(epsilon, t_critical * SE)

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
        if config.use_loky_for_moves:
            # Use reusable executor for loky (avoids pool churn and resource leaks)
            # Why loky? Threading is ineffective for CPU-bound LightGBM training
            # because Python GIL prevents threads from running in parallel.
            if config.debug_parallelism:
                print(f"[DEBUG _find_best_move] Evaluating {len(specs)} moves with {n_workers} workers "
                      f"(reusable executor)", flush=True)

            def eval_spec(spec):
                return _evaluate_move_spec(spec, evaluator, current_metric, config, None)

            results = parallel_map(
                eval_spec, specs, n_workers, debug=config.debug_parallelism
            )
            # Filter out None results from failed tasks
            results = [r for r in results if r is not None]
        else:
            # Threading backend: shared memory, but GIL-limited for CPU-bound work
            joblib_kwargs = get_joblib_kwargs(n_workers)
            if config.debug_parallelism:
                print(f"[DEBUG _find_best_move] Evaluating {len(specs)} moves with {n_workers} workers "
                      f"(threading)", flush=True)

            from joblib import Parallel, delayed
            results = Parallel(**joblib_kwargs)(
                delayed(_evaluate_move_spec)(spec, evaluator, current_metric, config, cache)
                for spec in specs
            )
    else:
        # Sequential evaluation
        results = [
            _evaluate_move_spec(spec, evaluator, current_metric, config, cache)
            for spec in specs
        ]

    # Step 3: Apply SNR-based acceptance to each result
    # Filter to moves that pass epsilon + t-stat thresholds
    accepted_results = []
    for r in results:
        if r.result is None:
            # No valid result (e.g., K-of-N found no features)
            continue

        # Get move-specific thresholds (handles tabu aspiration internally)
        epsilon, t_threshold = _get_acceptance_thresholds(
            r.spec.move_type, config, is_tabu=r.spec.is_tabu
        )

        # SNR-based acceptance
        context = f"{r.spec.move_type}"
        if r.spec.group_in:
            context += f" +{r.spec.group_in}"
        if r.spec.group_out:
            context += f" -{r.spec.group_out}"
        acceptance = _snr_acceptance(
            current_result, r.result, epsilon, t_threshold,
            debug=config.debug_acceptance,
            context=context,
        )

        if acceptance.accepted and acceptance.delta_mean > 0:
            # Store acceptance result for later use
            accepted_results.append((r, acceptance))

    if not accepted_results:
        return None

    # Sort by delta_mean (descending), then by spec for deterministic tie-breaking
    accepted_results.sort(key=lambda x: (-x[1].delta_mean, -x[0].new_metric, x[0].spec.sort_key()))
    best_result, best_acceptance = accepted_results[0]

    # Return move with SNR stats populated (caller handles logging)
    return LocalSearchMove(
        move_type=best_result.spec.move_type,
        group_out=best_result.spec.group_out,
        group_in=best_result.spec.group_in,
        features_in=best_result.features_in,
        delta=best_acceptance.delta_mean,
        new_metric=best_result.new_metric,
        t_stat=best_acceptance.t_stat,
        delta_std=best_acceptance.delta_std,
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
            # Consolidated log: metric and SNR stats
            t_info = f", t={best_move.t_stat:.2f}" if best_move.t_stat is not None else ""
            print(f"  Iteration {iteration}: {best_move}")
            print(f"      metric: {result.metric_before:.4f} → {current_metric:.4f} | "
                  f"Δ={best_move.delta:+.5f}{t_info}")

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
    Backward elimination at the group level with parallel evaluation.

    Each round, all removable groups are evaluated in parallel.
    The group whose removal causes least harm (and passes threshold) is removed.
    Repeat until no group can be safely removed.

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
    # Note: cache is disabled when using loky (processes don't share memory)
    if cache is None and config.enable_caching and not config.use_loky_for_moves:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # Evaluate current
    current_features = _groups_to_features(selected)
    current_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
    current_metric = current_result.metric_main

    if config.verbose:
        print(f"Backward elimination: starting metric = {current_metric:.4f}")

    round_num = 0
    while True:
        round_num += 1

        if len(selected) <= 1:
            if config.verbose:
                print(f"  Only 1 group remaining, stopping backward elimination")
            break

        # Find best drop move (parallel evaluation)
        best_move = _find_best_drop_move(
            evaluator=evaluator,
            selected=selected,
            baseline_group_names=baseline_group_names,
            current_metric=current_metric,
            current_result=current_result,
            config=config,
            cache=cache,
        )

        if best_move is None:
            if config.verbose:
                print(f"  Round {round_num}: No safe removal found, stopping")
            break

        # Record result
        result = GroupResult(
            group_name=best_move.group_out,
            features=selected[best_move.group_out],
            metric_before=current_metric,
            metric_after=best_move.new_metric,
            delta=best_move.delta,
            accepted=True,
            reason=f"Best removal (loss acceptable)"
        )
        results.append(result)

        # Apply removal
        removed_features = selected[best_move.group_out]
        del selected[best_move.group_out]
        current_metric = best_move.new_metric
        # Re-evaluate to get full SubsetResult for next round
        new_features = _groups_to_features(selected)
        current_result = _cached_evaluate(evaluator, new_features, cache, n_jobs=config.n_jobs)

        if config.verbose:
            # Consolidated log: metric and SNR stats
            t_info = f", t={best_move.t_stat:.2f}" if best_move.t_stat is not None else ""
            print(f"  Round {round_num}: - Removed '{best_move.group_out}' "
                  f"({len(removed_features)} features)")
            print(f"      metric: {result.metric_before:.4f} → {current_metric:.4f} | "
                  f"Δ={best_move.delta:+.5f}{t_info}")

    if config.verbose:
        print(f"  Backward elimination complete: {len(results)} groups removed")

    return selected, results, current_metric


def _find_best_drop_move(
    evaluator: SubsetEvaluator,
    selected: Dict[str, List[str]],
    baseline_group_names: Set[str],
    current_metric: float,
    current_result: SubsetResult,
    config: GroupSelectionConfig,
    cache: Optional[EvaluationCache] = None,
) -> Optional[LocalSearchMove]:
    """
    Find the best drop move among all removable groups (parallel evaluation).

    For backward elimination: finds the group whose removal causes least harm
    and passes the variance-adjusted threshold.

    Args:
        evaluator: SubsetEvaluator for feature evaluation
        selected: Currently selected groups
        baseline_group_names: Groups that cannot be removed (unless demotions allowed)
        current_metric: Current metric value
        current_result: Current SubsetResult (for variance-adjusted acceptance)
        config: Selection configuration
        cache: Optional evaluation cache (None when using loky)

    Returns:
        Best drop move, or None if no group can be safely removed.
    """
    # Get groups that can be removed
    removable_groups = list(selected.keys())
    if not config.allow_baseline_demotions:
        removable_groups = [g for g in removable_groups if g not in baseline_group_names]

    if not removable_groups or len(selected) <= 1:
        return None

    # Generate drop specs for all removable groups
    specs = []
    for group_name in sorted(removable_groups):
        test_groups = {k: v for k, v in selected.items() if k != group_name}
        base_features = _groups_to_features(test_groups)

        spec = MoveSpec(
            move_type="drop",
            group_out=group_name,
            group_in=None,
            group_in_features=None,
            base_features=base_features,
            is_tabu=False,
        )
        specs.append(spec)

    if not specs:
        return None

    # Evaluate in parallel
    n_workers = config.n_move_workers if config.n_move_workers != -1 else config.n_jobs

    if config.parallelize_moves and n_workers != 1 and len(specs) > 1:
        if config.use_loky_for_moves:
            # Use reusable executor for loky (avoids pool churn and resource leaks)
            if config.debug_parallelism:
                print(f"[DEBUG backward] Evaluating {len(specs)} removals with {n_workers} workers "
                      f"(reusable executor)", flush=True)

            def eval_spec(spec):
                return _evaluate_move_spec(spec, evaluator, current_metric, config, None)

            move_results = parallel_map(
                eval_spec, specs, n_workers, debug=config.debug_parallelism
            )
            # Filter out None results from failed tasks
            move_results = [r for r in move_results if r is not None]
        else:
            # Threading backend
            joblib_kwargs = get_joblib_kwargs(n_workers)
            if config.debug_parallelism:
                print(f"[DEBUG backward] Evaluating {len(specs)} removals with {n_workers} workers "
                      f"(threading)", flush=True)

            from joblib import Parallel, delayed
            move_results = Parallel(**joblib_kwargs)(
                delayed(_evaluate_move_spec)(spec, evaluator, current_metric, config, cache)
                for spec in specs
            )
    else:
        move_results = [
            _evaluate_move_spec(spec, evaluator, current_metric, config, cache)
            for spec in specs
        ]

    # Apply acceptance for removal: accept if loss is within tolerance
    # For removal: accept if delta_mean >= -epsilon_remove (loss <= epsilon_remove)
    # Note: We compute per-fold deltas for accurate statistics, but don't use t-stat gate
    # (backward elimination tests "no significant harm", not "significant improvement")
    accepted_moves = []
    for r in move_results:
        if r.result is None:
            continue

        # Compute per-fold deltas for accurate delta_mean (same as SNR approach)
        n_folds_before = len(current_result.fold_metrics)
        n_folds_after = len(r.result.fold_metrics)

        if n_folds_before == n_folds_after and n_folds_before > 0:
            deltas = [
                r.result.fold_metrics[i] - current_result.fold_metrics[i]
                for i in range(n_folds_before)
            ]
            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas, ddof=1)) if n_folds_before > 1 else 0.0
        else:
            delta_mean = r.result.metric_main - current_result.metric_main
            delta_std = 0.0

        # For removal: accept if loss <= epsilon_remove
        acceptable = delta_mean >= -config.epsilon_remove

        if config.debug_acceptance:
            status = "ACCEPT" if acceptable else "REJECT"
            print(f"    remove {r.spec.group_out}: Δ_mean={delta_mean:.5f}, "
                  f"Δ_std={delta_std:.5f}, thresh=-{config.epsilon_remove:.5f} → {status}")

        if acceptable:
            # Compute t_stat for logging (even though not used for acceptance in backward elim)
            n_folds = len(current_result.fold_metrics)
            delta_se = delta_std / np.sqrt(n_folds) if n_folds > 1 else 0.0
            t_stat = delta_mean / (delta_se + 1e-12) if delta_se > 0 else float('inf')

            move = LocalSearchMove(
                move_type="drop",
                group_out=r.spec.group_out,
                group_in=None,
                features_in=None,
                delta=delta_mean,
                new_metric=r.new_metric,
                t_stat=t_stat,
                delta_std=delta_std,
            )
            accepted_moves.append(move)

    if not accepted_moves:
        return None

    # Deterministic tie-breaking: best delta (least harm), then group name
    accepted_moves.sort(key=lambda m: (-m.delta, m.group_out))
    return accepted_moves[0]


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
    Select from curated interaction groups with parallel evaluation.

    Each round, all remaining interaction groups are evaluated in parallel.
    The best one that passes variance-adjusted threshold is added.
    Repeat until max_interaction_groups or no improvement.

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
    # Note: cache is disabled when using loky (processes don't share memory)
    if cache is None and config.enable_caching and not config.use_loky_for_moves:
        cache = EvaluationCache(max_size=config.cache_max_size)

    # Evaluate current
    current_features = _groups_to_features(selected)
    current_result = _cached_evaluate(evaluator, current_features, cache, n_jobs=config.n_jobs)
    current_metric = current_result.metric_main

    if config.verbose:
        print(f"Interaction selection: starting metric = {current_metric:.4f}, "
              f"{len(interaction_groups)} candidate interaction groups")

    # Filter interaction groups to those with available features
    available_interactions = {}
    for group_name, group_features in interaction_groups.items():
        if group_name in selected:
            continue
        available_features = [f for f in group_features if f in evaluator._all_features]
        if len(available_features) > 0:
            available_interactions[group_name] = available_features
        elif config.verbose:
            print(f"  - Skipped '{group_name}': no features available in data")

    # Iteratively add best interaction until max reached or none pass
    n_interactions_added = 0
    round_num = 0

    while available_interactions and n_interactions_added < config.max_interaction_groups:
        round_num += 1
        current_features = _groups_to_features(selected)

        # Find best add move among remaining interactions (parallel)
        best_move = _find_best_add_move(
            evaluator=evaluator,
            current_features=current_features,
            candidate_groups=available_interactions,
            current_metric=current_metric,
            current_result=current_result,
            config=config,
            cache=cache,
        )

        if best_move is None:
            if config.verbose:
                print(f"  Round {round_num}: No improving interaction found, stopping")
            break

        # Record result
        result = GroupResult(
            group_name=best_move.group_in,
            features=best_move.features_in,
            metric_before=current_metric,
            metric_after=best_move.new_metric,
            delta=best_move.delta,
            accepted=True,
            reason=f"Best of {len(available_interactions)} interactions"
        )
        results.append(result)

        # Accept the move
        selected[best_move.group_in] = best_move.features_in
        current_metric = best_move.new_metric
        # Re-evaluate to get full SubsetResult for next round
        new_features = _groups_to_features(selected)
        current_result = _cached_evaluate(evaluator, new_features, cache, n_jobs=config.n_jobs)

        # Remove from remaining
        del available_interactions[best_move.group_in]
        n_interactions_added += 1

        if config.verbose:
            print(f"  Round {round_num}: + Added '{best_move.group_in}' "
                  f"({len(best_move.features_in)} features): "
                  f"Δ = {best_move.delta:+.4f}, metric = {current_metric:.4f}")

    if n_interactions_added >= config.max_interaction_groups and config.verbose:
        print(f"  Max interaction groups ({config.max_interaction_groups}) reached")

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

        # SNR-based acceptance for interaction group adds
        epsilon, t_threshold = _get_acceptance_thresholds("add_interaction", config)
        acceptance = _snr_acceptance(
            current_result, test_result, epsilon, t_threshold,
            debug=config.debug_acceptance,
            context=f"interaction '{template_name}'",
        )

        result = GroupResult(
            group_name=template_name,
            features=available_features,
            metric_before=current_metric,
            metric_after=test_result.metric_main,
            delta=acceptance.delta_mean,
            accepted=acceptance.accepted,
            reason=f"Δ_mean={acceptance.delta_mean:.4f}, Δ_std={acceptance.delta_std:.4f}, "
                   f"t={acceptance.t_stat:.2f} (parents: {parent_a}, {parent_b})"
        )
        results.append(result)

        if acceptance.accepted:
            selected[template_name] = available_features
            prev_metric = current_metric
            current_metric = test_result.metric_main
            current_result = test_result  # Update for next iteration
            n_interactions_added += 1
            if config.verbose:
                # Consolidated log: metric and SNR stats
                print(f"  + Added template '{template_name}' ({len(available_features)} features)")
                print(f"      metric: {prev_metric:.4f} → {current_metric:.4f} | "
                      f"Δ={acceptance.delta_mean:+.5f}, t={acceptance.t_stat:.2f}")
                print(f"      parents: {parent_a}, {parent_b}")
        elif config.verbose:
            print(f"  - Rejected template '{template_name}': {acceptance.reason}")

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
            if metric_name.lower() == 'auc':
                continue  # Already printed as primary metric
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
            if metric_name.lower() == 'auc':
                continue  # Already printed as primary metric
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


# =============================================================================
# OUTER CROSS-VALIDATION FOR ROBUSTNESS ESTIMATION
# =============================================================================

@dataclass
class OuterFoldResult:
    """Result from a single outer CV fold.

    Attributes:
        fold: Fold index (1-indexed for display).
        train_start: Start date of training period.
        train_end: End date of training period.
        test_start: Start date of test period.
        test_end: End date of test period.
        n_train_rows: Number of training rows.
        n_test_rows: Number of test rows.
        selected_groups: Groups selected in this fold.
        n_groups: Number of groups selected.
        n_features: Number of features selected.
        inner_cv_metric: Best metric from inner CV (feature selection).
        outer_auc: AUC on outer test set.
        selection_result: Full GroupSelectionResult from inner selection.
    """
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_rows: int
    n_test_rows: int
    selected_groups: List[str]
    n_groups: int
    n_features: int
    inner_cv_metric: float
    outer_auc: float
    selection_result: Optional[GroupSelectionResult] = None


@dataclass
class OuterCVResult:
    """Complete result from outer cross-validation.

    Outer CV reduces feature-selection bias by treating the entire
    selection pipeline as a learned procedure. 3 folds is sufficient
    to detect overfitting without excessive computation.

    This is a robustness check, not a production training loop.

    Attributes:
        outer_auc_mean: Mean AUC across outer folds.
        outer_auc_std: Standard deviation of AUC across outer folds.
        outer_fold_results: Per-fold detailed results.
        group_selection_frequency: How often each group was selected.
        total_time_seconds: Total time for outer CV.
    """
    outer_auc_mean: float
    outer_auc_std: float
    outer_fold_results: List[OuterFoldResult]
    group_selection_frequency: Dict[str, int]
    total_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML output."""
        return {
            "outer_auc_mean": self.outer_auc_mean,
            "outer_auc_std": self.outer_auc_std,
            "outer_fold_results": [
                {
                    "fold": r.fold,
                    "auc": r.outer_auc,
                    "n_groups": r.n_groups,
                    "n_features": r.n_features,
                    "selected_groups": r.selected_groups,
                    "train_period": f"{r.train_start} → {r.train_end}",
                    "test_period": f"{r.test_start} → {r.test_end}",
                    "inner_cv_metric": r.inner_cv_metric,
                }
                for r in self.outer_fold_results
            ],
            "group_selection_frequency": self.group_selection_frequency,
            "total_time_seconds": self.total_time_seconds,
        }


def generate_outer_splits(
    timestamps: pd.Series,
    n_splits: int = 3,
    test_frac: float = 0.10,
    final_holdout_frac: float = 0.05,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (expanding window) splits for outer CV.

    This function creates contiguous, time-ordered splits where:
    - Training expands over time (expanding window)
    - Test windows are non-overlapping and ~10% of usable data each
    - Final 5% of data is excluded entirely (reserved for true holdout)

    Example timeline (conceptual):
    |====================== TRAIN 1 ======================|==== TEST 1 ====|==|
    |========================== TRAIN 2 ===========================|==== TEST 2 ====|==|
    |================================ TRAIN 3 ==================================|==== TEST 3 ====|==|
                                                                                  ^ final 5% untouched

    Args:
        timestamps: Series of timestamps (must be sortable, typically DateTimeIndex).
        n_splits: Number of outer folds (default 3).
        test_frac: Fraction of total time per test fold (default 0.10).
        final_holdout_frac: Fraction reserved as final holdout (default 0.05).

    Returns:
        List of (train_idx, test_idx) tuples. Indices are row positions (not labels).
    """
    # Get sorted unique dates
    unique_dates = np.sort(timestamps.unique())
    n_dates = len(unique_dates)

    # Reserve final holdout
    holdout_size = int(n_dates * final_holdout_frac)
    usable_n_dates = n_dates - holdout_size

    if usable_n_dates < n_splits * 2:
        raise ValueError(
            f"Not enough dates for {n_splits} outer folds. "
            f"Have {usable_n_dates} usable dates after reserving {holdout_size} for holdout."
        )

    # Test size per fold (in dates)
    test_size_dates = int(usable_n_dates * test_frac)
    if test_size_dates < 1:
        test_size_dates = 1

    # Build date -> row indices mapping
    date_to_rows = {}
    for row_idx, date_val in enumerate(timestamps):
        if date_val not in date_to_rows:
            date_to_rows[date_val] = []
        date_to_rows[date_val].append(row_idx)

    splits = []

    for fold_idx in range(n_splits):
        # Test period: work backwards from the end of usable data
        # Fold 0 gets the earliest test window, fold n_splits-1 gets the latest
        test_end_idx = usable_n_dates - (n_splits - fold_idx - 1) * test_size_dates
        test_start_idx = test_end_idx - test_size_dates

        # Training: all data before test start (expanding window)
        train_start_idx = 0
        train_end_idx = test_start_idx

        # Validate
        if train_end_idx <= train_start_idx:
            continue
        if test_start_idx >= usable_n_dates or test_end_idx > usable_n_dates:
            continue

        # Get date ranges
        train_dates = unique_dates[train_start_idx:train_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        # Map dates to row indices
        train_rows = []
        for d in train_dates:
            train_rows.extend(date_to_rows[d])

        test_rows = []
        for d in test_dates:
            test_rows.extend(date_to_rows[d])

        if len(train_rows) > 0 and len(test_rows) > 0:
            splits.append((np.array(train_rows), np.array(test_rows)))

    return splits


def _train_final_model_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: List[str],
    model_config,
    sample_weight: Optional[pd.Series] = None,
) -> float:
    """
    Train a final model on outer-train data and evaluate on outer-test.

    This is a single evaluation with no CV, no hyperparameter tuning,
    and no feature selection. Just train and predict once.

    Args:
        X_train: Training features (outer-train).
        y_train: Training labels.
        X_test: Test features (outer-test).
        y_test: Test labels.
        selected_features: Features selected during inner CV.
        model_config: Model configuration.
        sample_weight: Optional sample weights for training.

    Returns:
        AUC on the outer test set.
    """
    from .models import GBMWrapper
    from .metrics import compute_auc

    # Subset to selected features
    available_features = [f for f in selected_features if f in X_train.columns and f in X_test.columns]
    if not available_features:
        return 0.5  # Random baseline

    X_train_sub = X_train[available_features]
    X_test_sub = X_test[available_features]

    # Handle NaN values
    train_mask = ~(X_train_sub.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test_sub.isna().any(axis=1) | y_test.isna())

    X_train_clean = X_train_sub.loc[train_mask]
    y_train_clean = y_train.loc[train_mask]
    X_test_clean = X_test_sub.loc[test_mask]
    y_test_clean = y_test.loc[test_mask]

    if len(X_train_clean) < 100 or len(X_test_clean) < 10:
        return 0.5  # Insufficient data

    # Handle sample weights
    w_train = None
    if sample_weight is not None:
        w_train = sample_weight.loc[train_mask]

    # Train model with early stopping using train/val split
    model = GBMWrapper(model_config)

    n_train = len(X_train_clean)
    split_point = int(n_train * 0.85)

    if model_config.early_stopping_rounds and split_point > 50:
        X_tr = X_train_clean.iloc[:split_point]
        X_val = X_train_clean.iloc[split_point:]
        y_tr = y_train_clean.iloc[:split_point]
        y_val = y_train_clean.iloc[split_point:]

        w_tr = None
        w_val = None
        if w_train is not None:
            w_tr = w_train.iloc[:split_point]
            w_val = w_train.iloc[split_point:]

        model.train(X_tr, y_tr, X_val, y_val, available_features,
                   sample_weight=w_tr, sample_weight_val=w_val)
    else:
        model.train(X_train_clean, y_train_clean, feature_names=available_features,
                   sample_weight=w_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_clean)
    outer_auc = compute_auc(y_test_clean.values, y_pred)

    model.cleanup()

    return outer_auc


def run_outer_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    model_config,
    cv_config,
    metric_config,
    search_config: SearchConfig,
    group_config: Optional[GroupSelectionConfig] = None,
    n_outer_splits: int = 3,
    test_frac: float = 0.10,
    final_holdout_frac: float = 0.05,
    sample_weight: Optional[pd.Series] = None,
    verbose: bool = True,
) -> OuterCVResult:
    """
    Run outer cross-validation to estimate generalization when feature selection
    is treated as a learned procedure.

    Outer CV reduces feature-selection bias:
    - 3 folds is sufficient to detect overfitting without excessive computation
    - This is a robustness check, not a production training loop

    Design:
    - Walk-forward (expanding window) splits
    - Final 5% of data excluded entirely (reserved holdout)
    - Feature selection runs independently on each outer-train fold
    - Selected features may differ across folds (expected and desired)
    - Outer test is evaluated ONCE with no tuning

    Critical constraints:
    - Inner CV stays exactly the same (5 folds, purge=2, embargo=20)
    - Feature selection happens inside each outer-train fold
    - No hyperparameter tuning on outer test
    - Deterministic given same data and config

    Args:
        X: Feature DataFrame with DateTimeIndex.
        y: Target Series aligned with X.
        model_key: Model key (LONG_NORMAL, etc.).
        model_config: Model configuration.
        cv_config: CV configuration (used for inner CV).
        metric_config: Metric configuration.
        search_config: Search configuration.
        group_config: Optional GroupSelectionConfig (uses defaults if None).
        n_outer_splits: Number of outer folds (default 3).
        test_frac: Fraction of data per outer test fold (default 0.10).
        final_holdout_frac: Fraction reserved as final holdout (default 0.05).
        sample_weight: Optional sample weights for training.
        verbose: Print progress.

    Returns:
        OuterCVResult with per-fold results and aggregated statistics.
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*70}")
        print(f"OUTER CROSS-VALIDATION for {model_key.value}")
        print(f"{'='*70}")
        print(f"Outer folds: {n_outer_splits}")
        print(f"Test fraction per fold: {test_frac:.1%}")
        print(f"Final holdout reserved: {final_holdout_frac:.1%}")
        print()

    # Generate outer splits
    timestamps = pd.Series(X.index)
    outer_splits = generate_outer_splits(
        timestamps,
        n_splits=n_outer_splits,
        test_frac=test_frac,
        final_holdout_frac=final_holdout_frac,
    )

    if len(outer_splits) == 0:
        raise ValueError("No valid outer splits could be generated")

    if verbose:
        print(f"Generated {len(outer_splits)} outer splits")
        print()

    # Track results
    fold_results = []
    all_selected_groups = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        fold_num = fold_idx + 1  # 1-indexed for display

        # Slice data for this outer fold
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_test_outer = y.iloc[test_idx]

        # Get date ranges for logging
        train_dates = X_train_outer.index
        test_dates = X_test_outer.index
        train_start = str(train_dates.min().date()) if hasattr(train_dates.min(), 'date') else str(train_dates.min())
        train_end = str(train_dates.max().date()) if hasattr(train_dates.max(), 'date') else str(train_dates.max())
        test_start = str(test_dates.min().date()) if hasattr(test_dates.min(), 'date') else str(test_dates.min())
        test_end = str(test_dates.max().date()) if hasattr(test_dates.max(), 'date') else str(test_dates.max())

        if verbose:
            print(f"OUTER FOLD {fold_num}")
            print(f"  Train: {train_start} → {train_end} ({len(X_train_outer)} rows)")
            print(f"  Test:  {test_start} → {test_end} ({len(X_test_outer)} rows)")

        # Slice sample weights if provided
        w_train_outer = None
        if sample_weight is not None:
            w_train_outer = sample_weight.iloc[train_idx]

        # Run full feature selection on outer-train
        # This uses the existing run_group_selection with inner CV unchanged
        selection_config = group_config
        if selection_config is None:
            selection_config = GroupSelectionConfig(
                epsilon_add=search_config.epsilon_add,
                epsilon_swap=search_config.epsilon_swap,
                epsilon_remove=search_config.epsilon_swap,
                verbose=False,  # Suppress inner verbosity during outer CV
            )
        else:
            # Make a copy with verbose disabled for inner selection
            selection_config = GroupSelectionConfig(
                epsilon_add=selection_config.epsilon_add,
                epsilon_swap=selection_config.epsilon_swap,
                epsilon_remove=selection_config.epsilon_remove,
                epsilon_drop=selection_config.epsilon_drop,
                epsilon_add_interaction=selection_config.epsilon_add_interaction,
                allow_baseline_demotions=selection_config.allow_baseline_demotions,
                max_groups=selection_config.max_groups,
                max_interaction_groups=selection_config.max_interaction_groups,
                enable_add_drop_moves=selection_config.enable_add_drop_moves,
                max_search_iterations=selection_config.max_search_iterations,
                enable_tabu=selection_config.enable_tabu,
                tabu_tenure=selection_config.tabu_tenure,
                tabu_aspiration_delta=selection_config.tabu_aspiration_delta,
                enable_k_of_n=selection_config.enable_k_of_n,
                group_k_default=selection_config.group_k_default,
                group_k=selection_config.group_k,
                epsilon_add_feature=selection_config.epsilon_add_feature,
                enable_caching=selection_config.enable_caching,
                cache_max_size=selection_config.cache_max_size,
                n_jobs=selection_config.n_jobs,
                parallelize_moves=selection_config.parallelize_moves,
                n_move_workers=selection_config.n_move_workers,
                verbose=False,  # Suppress for outer CV
            )

        # Run group selection on outer-train data
        selection_result = run_group_selection(
            X=X_train_outer,
            y=y_train_outer,
            model_key=model_key,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
            config=selection_config,
        )

        # Freeze selected groups and features
        selected_groups = list(selection_result.selected_groups.keys())
        selected_features = selection_result.selected_features
        n_groups = len(selected_groups)
        n_features = len(selected_features)
        inner_cv_metric = selection_result.final_metric

        # Track groups for frequency analysis
        all_selected_groups.extend(selected_groups)

        if verbose:
            print(f"  Selected: {n_groups} groups, {n_features} features")
            print(f"  Inner CV AUC: {inner_cv_metric:.4f}")

        # Train final model on entire outer-train using selected features
        # Evaluate once on outer-test (no selection, no tuning)
        outer_auc = _train_final_model_and_evaluate(
            X_train=X_train_outer,
            y_train=y_train_outer,
            X_test=X_test_outer,
            y_test=y_test_outer,
            selected_features=selected_features,
            model_config=model_config,
            sample_weight=w_train_outer,
        )

        if verbose:
            print(f"  Outer AUC: {outer_auc:.4f}")
            print()

        # Record fold result
        fold_result = OuterFoldResult(
            fold=fold_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_train_rows=len(X_train_outer),
            n_test_rows=len(X_test_outer),
            selected_groups=selected_groups,
            n_groups=n_groups,
            n_features=n_features,
            inner_cv_metric=inner_cv_metric,
            outer_auc=outer_auc,
            selection_result=selection_result,
        )
        fold_results.append(fold_result)

    # Aggregate results
    outer_aucs = [r.outer_auc for r in fold_results]
    outer_auc_mean = float(np.mean(outer_aucs))
    outer_auc_std = float(np.std(outer_aucs, ddof=1)) if len(outer_aucs) > 1 else 0.0

    # Compute group selection frequency
    group_frequency = dict(Counter(all_selected_groups))
    # Sort by frequency (descending)
    group_frequency = dict(sorted(group_frequency.items(), key=lambda x: -x[1]))

    total_time = time.time() - start_time

    if verbose:
        print(f"{'='*70}")
        print(f"OUTER CV SUMMARY")
        print(f"{'='*70}")
        print(f"  Mean AUC: {outer_auc_mean:.4f}")
        print(f"  Std AUC:  {outer_auc_std:.4f}")
        print()
        print(f"  Per-fold AUCs: {[f'{auc:.4f}' for auc in outer_aucs]}")
        print()
        print(f"  Group selection frequency (across {n_outer_splits} folds):")
        for group_name, freq in list(group_frequency.items())[:10]:
            print(f"    {group_name}: {freq}/{n_outer_splits}")
        if len(group_frequency) > 10:
            print(f"    ... and {len(group_frequency) - 10} more groups")
        print()
        print(f"  Total time: {total_time:.1f}s")
        print()

    return OuterCVResult(
        outer_auc_mean=outer_auc_mean,
        outer_auc_std=outer_auc_std,
        outer_fold_results=fold_results,
        group_selection_frequency=group_frequency,
        total_time_seconds=total_time,
    )


# =============================================================================
# OUTER CV STABILITY AGGREGATION AND FINALIZATION
# =============================================================================

@dataclass
class CandidateSetResult:
    """Result from evaluating a candidate group set on holdout.

    Attributes:
        name: Candidate set name ("core" or "stable").
        groups: List of group names in this candidate set.
        n_groups: Number of groups.
        n_features: Number of features.
        holdout_auc: AUC on the final holdout set.
    """
    name: str
    groups: List[str]
    n_groups: int
    n_features: int
    holdout_auc: float


@dataclass
class StabilityAggregationResult:
    """Result from stability-based group aggregation and finalization.

    This step uses outer CV stability counts to select the final feature set,
    then validates on the true holdout (final 5%) that was never used.

    Selection logic:
    - groups_core: Selected in ALL outer folds
    - groups_stable: Selected in >= ceil(n_folds * 2/3) outer folds
    - Candidate A: baseline ∪ groups_core
    - Candidate B: baseline ∪ groups_stable
    - Choose B only if holdout_auc_B >= holdout_auc_A + 0.01 AND n_groups_B <= n_groups_A + 5

    Attributes:
        chosen_set_name: Name of chosen candidate ("core" or "stable").
        chosen_groups: Final selected groups.
        chosen_features: Final selected features.
        n_groups: Number of groups in final set.
        n_features: Number of features in final set.
        holdout_auc: Holdout AUC for chosen set.
        candidate_core: Results for core candidate (A).
        candidate_stable: Results for stable candidate (B).
        groups_core: Groups selected in ALL folds.
        groups_stable: Groups selected in >= 2/3 folds.
        group_frequency: Full frequency table.
        decision_reason: Explanation for the choice.
    """
    chosen_set_name: str
    chosen_groups: List[str]
    chosen_features: List[str]
    n_groups: int
    n_features: int
    holdout_auc: float
    candidate_core: CandidateSetResult
    candidate_stable: CandidateSetResult
    groups_core: List[str]
    groups_stable: List[str]
    group_frequency: Dict[str, int]
    decision_reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML output."""
        return {
            "chosen_set_name": self.chosen_set_name,
            "chosen_groups": self.chosen_groups,
            "n_groups": self.n_groups,
            "n_features": self.n_features,
            "holdout_auc": self.holdout_auc,
            "decision_reason": self.decision_reason,
            "candidate_core": {
                "groups": self.candidate_core.groups,
                "n_groups": self.candidate_core.n_groups,
                "n_features": self.candidate_core.n_features,
                "holdout_auc": self.candidate_core.holdout_auc,
            },
            "candidate_stable": {
                "groups": self.candidate_stable.groups,
                "n_groups": self.candidate_stable.n_groups,
                "n_features": self.candidate_stable.n_features,
                "holdout_auc": self.candidate_stable.holdout_auc,
            },
            "groups_core": self.groups_core,
            "groups_stable": self.groups_stable,
            "group_frequency": self.group_frequency,
        }


def _get_features_for_groups(
    group_names: List[str],
    model_key: ModelKey,
    available_columns: List[str],
) -> List[str]:
    """Get all features for a list of groups that exist in the data.

    Args:
        group_names: List of group names.
        model_key: Model key for group lookup.
        available_columns: Columns available in the data.

    Returns:
        List of feature names.
    """
    from .base_features import get_all_groups, get_interaction_groups

    # Get all group definitions
    all_groups = get_all_groups(model_key)
    interaction_groups = get_interaction_groups(model_key)

    # Merge group definitions
    group_defs = {**all_groups, **interaction_groups}

    features = []
    available_set = set(available_columns)

    for group_name in group_names:
        if group_name in group_defs:
            group_features = group_defs[group_name]
            # Only include features that exist in the data
            for f in group_features:
                if f in available_set and f not in features:
                    features.append(f)

    return features


def _evaluate_candidate_on_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    candidate_groups: List[str],
    model_key: ModelKey,
    model_config,
    sample_weight: Optional[pd.Series] = None,
) -> CandidateSetResult:
    """Evaluate a candidate group set on the holdout.

    No feature selection, no hyperparameter tuning - just train and evaluate once.

    Args:
        X_train: Training features (all data except holdout).
        y_train: Training labels.
        X_holdout: Holdout features (final 5%).
        y_holdout: Holdout labels.
        candidate_groups: List of group names to use.
        model_key: Model key for feature lookup.
        model_config: Model configuration.
        sample_weight: Optional sample weights.

    Returns:
        CandidateSetResult with holdout AUC.
    """
    # Get features for these groups
    features = _get_features_for_groups(
        candidate_groups,
        model_key,
        list(X_train.columns),
    )

    if not features:
        return CandidateSetResult(
            name="",
            groups=candidate_groups,
            n_groups=len(candidate_groups),
            n_features=0,
            holdout_auc=0.5,
        )

    # Evaluate on holdout
    holdout_auc = _train_final_model_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        X_test=X_holdout,
        y_test=y_holdout,
        selected_features=features,
        model_config=model_config,
        sample_weight=sample_weight,
    )

    return CandidateSetResult(
        name="",
        groups=candidate_groups,
        n_groups=len(candidate_groups),
        n_features=len(features),
        holdout_auc=holdout_auc,
    )


def run_stability_aggregation(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv_result: OuterCVResult,
    model_key: ModelKey,
    model_config,
    final_holdout_frac: float = 0.05,
    allow_baseline_demotions: bool = False,
    sample_weight: Optional[pd.Series] = None,
    verbose: bool = True,
) -> StabilityAggregationResult:
    """
    Aggregate outer CV results and select final feature set based on stability.

    This function:
    1. Computes group selection frequency from outer CV
    2. Identifies core groups (selected in ALL folds) and stable groups (>= 2/3)
    3. Builds two candidate sets: baseline ∪ core (A) and baseline ∪ stable (B)
    4. Evaluates BOTH on the reserved final 5% holdout
    5. Deterministically chooses winner based on holdout performance

    Selection rule:
    - Choose B (stable) only if:
      - holdout_auc_B >= holdout_auc_A + 0.01 (meaningful improvement)
      - AND n_groups_B <= n_groups_A + 5 (prevent bloat)
    - Otherwise choose A (core)

    This step uses ONLY the holdout for validation. Outer CV test AUC is NOT
    used for selection - it's only used for robustness estimation.

    Args:
        X: Full feature DataFrame with DateTimeIndex.
        y: Full target Series.
        outer_cv_result: Result from run_outer_cv.
        model_key: Model key (LONG_NORMAL, etc.).
        model_config: Model configuration.
        final_holdout_frac: Fraction of data reserved as holdout (default 0.05).
        allow_baseline_demotions: If True, don't force baseline group inclusion.
        sample_weight: Optional sample weights.
        verbose: Print progress.

    Returns:
        StabilityAggregationResult with chosen groups and holdout performance.
    """
    import math
    from .base_features import get_baseline_groups

    if verbose:
        print(f"\n{'='*70}")
        print("STABILITY AGGREGATION")
        print(f"{'='*70}")

    # Get group frequency from outer CV
    group_freq = outer_cv_result.group_selection_frequency
    n_outer_folds = len(outer_cv_result.outer_fold_results)

    # Compute stability thresholds
    # groups_core: selected in ALL folds
    # groups_stable: selected in >= ceil(n_folds * 2/3) folds
    stable_threshold = math.ceil(n_outer_folds * 2 / 3)

    groups_core = [g for g, freq in group_freq.items() if freq == n_outer_folds]
    groups_stable = [g for g, freq in group_freq.items() if freq >= stable_threshold]

    if verbose:
        print(f"Outer folds: {n_outer_folds}")
        print(f"Stable threshold: >= {stable_threshold}/{n_outer_folds} folds")
        print(f"Groups selected in ALL folds (core): {len(groups_core)}")
        print(f"Groups selected in >= {stable_threshold} folds (stable): {len(groups_stable)}")
        print()

    # Get baseline groups for this model
    baseline_groups = list(get_baseline_groups(model_key).keys())

    if verbose:
        print(f"Baseline groups: {len(baseline_groups)}")
        if allow_baseline_demotions:
            print("  (baseline demotions ALLOWED - not forcing baseline inclusion)")
        else:
            print("  (baseline demotions NOT allowed - forcing baseline inclusion)")
        print()

    # Build candidate sets
    if allow_baseline_demotions:
        # Don't force baseline inclusion
        candidate_a_groups = list(set(groups_core))
        candidate_b_groups = list(set(groups_stable))
    else:
        # Force baseline inclusion
        candidate_a_groups = list(set(baseline_groups) | set(groups_core))
        candidate_b_groups = list(set(baseline_groups) | set(groups_stable))

    # Sort for determinism
    candidate_a_groups.sort()
    candidate_b_groups.sort()

    if verbose:
        print(f"Candidate A (baseline ∪ core): {len(candidate_a_groups)} groups")
        print(f"Candidate B (baseline ∪ stable): {len(candidate_b_groups)} groups")
        print()

    # Split data: reserve final holdout
    timestamps = pd.Series(X.index)
    unique_dates = np.sort(timestamps.unique())
    n_dates = len(unique_dates)
    holdout_size = int(n_dates * final_holdout_frac)

    if holdout_size < 1:
        holdout_size = 1

    holdout_dates = set(unique_dates[-holdout_size:])
    train_dates = set(unique_dates[:-holdout_size])

    # Build masks
    train_mask = timestamps.isin(train_dates).values
    holdout_mask = timestamps.isin(holdout_dates).values

    X_train = X.iloc[train_mask]
    y_train = y.iloc[train_mask]
    X_holdout = X.iloc[holdout_mask]
    y_holdout = y.iloc[holdout_mask]

    w_train = None
    if sample_weight is not None:
        w_train = sample_weight.iloc[train_mask]

    if verbose:
        train_start = str(X_train.index.min().date()) if hasattr(X_train.index.min(), 'date') else str(X_train.index.min())
        train_end = str(X_train.index.max().date()) if hasattr(X_train.index.max(), 'date') else str(X_train.index.max())
        holdout_start = str(X_holdout.index.min().date()) if hasattr(X_holdout.index.min(), 'date') else str(X_holdout.index.min())
        holdout_end = str(X_holdout.index.max().date()) if hasattr(X_holdout.index.max(), 'date') else str(X_holdout.index.max())
        print(f"Train:   {train_start} → {train_end} ({len(X_train)} rows)")
        print(f"Holdout: {holdout_start} → {holdout_end} ({len(X_holdout)} rows)")
        print()

    # Evaluate candidate A (core)
    if verbose:
        print("Evaluating Candidate A (core) on holdout...")

    result_a = _evaluate_candidate_on_holdout(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        candidate_groups=candidate_a_groups,
        model_key=model_key,
        model_config=model_config,
        sample_weight=w_train,
    )
    result_a = CandidateSetResult(
        name="core",
        groups=result_a.groups,
        n_groups=result_a.n_groups,
        n_features=result_a.n_features,
        holdout_auc=result_a.holdout_auc,
    )

    if verbose:
        print(f"  Candidate A: {result_a.n_groups} groups, {result_a.n_features} features, AUC={result_a.holdout_auc:.4f}")

    # Evaluate candidate B (stable)
    if verbose:
        print("Evaluating Candidate B (stable) on holdout...")

    result_b = _evaluate_candidate_on_holdout(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        candidate_groups=candidate_b_groups,
        model_key=model_key,
        model_config=model_config,
        sample_weight=w_train,
    )
    result_b = CandidateSetResult(
        name="stable",
        groups=result_b.groups,
        n_groups=result_b.n_groups,
        n_features=result_b.n_features,
        holdout_auc=result_b.holdout_auc,
    )

    if verbose:
        print(f"  Candidate B: {result_b.n_groups} groups, {result_b.n_features} features, AUC={result_b.holdout_auc:.4f}")
        print()

    # Decision logic:
    # Choose B only if:
    #   holdout_auc_B >= holdout_auc_A + 0.01 (meaningful improvement)
    #   AND n_groups_B <= n_groups_A + 5 (prevent bloat)
    # Otherwise choose A

    auc_improvement = result_b.holdout_auc - result_a.holdout_auc
    group_increase = result_b.n_groups - result_a.n_groups

    choose_b = (auc_improvement >= 0.01) and (group_increase <= 5)

    if choose_b:
        chosen = result_b
        decision_reason = (
            f"Chose STABLE: AUC improvement {auc_improvement:.4f} >= 0.01 "
            f"and group increase {group_increase} <= 5"
        )
    else:
        chosen = result_a
        if auc_improvement < 0.01:
            decision_reason = (
                f"Chose CORE: AUC improvement {auc_improvement:.4f} < 0.01 threshold"
            )
        else:
            decision_reason = (
                f"Chose CORE: group increase {group_increase} > 5 (prevents bloat)"
            )

    # Get final features for chosen groups
    chosen_features = _get_features_for_groups(
        chosen.groups,
        model_key,
        list(X.columns),
    )

    if verbose:
        print(f"{'='*70}")
        print("DECISION")
        print(f"{'='*70}")
        print(f"  {decision_reason}")
        print()
        print(f"  Chosen: {chosen.name.upper()}")
        print(f"  Groups: {chosen.n_groups}")
        print(f"  Features: {len(chosen_features)}")
        print(f"  Holdout AUC: {chosen.holdout_auc:.4f}")
        print()

    return StabilityAggregationResult(
        chosen_set_name=chosen.name,
        chosen_groups=chosen.groups,
        chosen_features=chosen_features,
        n_groups=chosen.n_groups,
        n_features=len(chosen_features),
        holdout_auc=chosen.holdout_auc,
        candidate_core=result_a,
        candidate_stable=result_b,
        groups_core=groups_core,
        groups_stable=groups_stable,
        group_frequency=group_freq,
        decision_reason=decision_reason,
    )


def run_outer_cv_with_finalization(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    model_config,
    cv_config,
    metric_config,
    search_config: SearchConfig,
    group_config: Optional[GroupSelectionConfig] = None,
    n_outer_splits: int = 3,
    test_frac: float = 0.10,
    final_holdout_frac: float = 0.05,
    allow_baseline_demotions: bool = False,
    sample_weight: Optional[pd.Series] = None,
    verbose: bool = True,
) -> Tuple[OuterCVResult, StabilityAggregationResult]:
    """
    Run outer CV followed by stability aggregation to get final feature set.

    This is a convenience function that combines:
    1. run_outer_cv: Walk-forward CV for robustness estimation
    2. run_stability_aggregation: Select final groups based on stability

    The workflow:
    1. Run outer CV with N folds (default 3)
    2. Compute group selection frequency across folds
    3. Build core (ALL folds) and stable (>= 2/3 folds) group sets
    4. Evaluate both on final 5% holdout
    5. Deterministically choose winner

    Args:
        X: Feature DataFrame with DateTimeIndex.
        y: Target Series aligned with X.
        model_key: Model key (LONG_NORMAL, etc.).
        model_config: Model configuration.
        cv_config: CV configuration (used for inner CV).
        metric_config: Metric configuration.
        search_config: Search configuration.
        group_config: Optional GroupSelectionConfig.
        n_outer_splits: Number of outer folds (default 3).
        test_frac: Fraction of data per outer test fold (default 0.10).
        final_holdout_frac: Fraction reserved as final holdout (default 0.05).
        allow_baseline_demotions: If True, don't force baseline inclusion.
        sample_weight: Optional sample weights.
        verbose: Print progress.

    Returns:
        Tuple of (OuterCVResult, StabilityAggregationResult).
    """
    # Step 1: Run outer CV
    outer_cv_result = run_outer_cv(
        X=X,
        y=y,
        model_key=model_key,
        model_config=model_config,
        cv_config=cv_config,
        metric_config=metric_config,
        search_config=search_config,
        group_config=group_config,
        n_outer_splits=n_outer_splits,
        test_frac=test_frac,
        final_holdout_frac=final_holdout_frac,
        sample_weight=sample_weight,
        verbose=verbose,
    )

    # Step 2: Run stability aggregation
    aggregation_result = run_stability_aggregation(
        X=X,
        y=y,
        outer_cv_result=outer_cv_result,
        model_key=model_key,
        model_config=model_config,
        final_holdout_frac=final_holdout_frac,
        allow_baseline_demotions=allow_baseline_demotions,
        sample_weight=sample_weight,
        verbose=verbose,
    )

    return outer_cv_result, aggregation_result
