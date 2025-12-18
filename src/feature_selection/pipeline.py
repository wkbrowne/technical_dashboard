"""
Loose-then-Tight Feature Selection Pipeline.

This pipeline implements a "high recall early, high precision later" approach
optimized for noisy financial labels (triple-barrier targets).

Pipeline Structure:
1. Base feature set (from base_features.py) - seed, not forced
2. Loose forward selection (high recall)
3. Strict backward elimination (high precision)
4. Light pairwise interaction pass (optional)
5. Hill-climbing / pair swapping (local improvement)
6. Final cleanup pass
7. Select best subset seen anywhere

Checkpointing:
- Automatically saves state after each stage to artifacts/feature_selection/checkpoint.pkl
- Resume with: pipeline.resume_from_checkpoint(X, y, checkpoint_path)
"""

import gc
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base_features import (
    BASE_FEATURES,
    FEATURE_CATEGORIES,
    EXPANSION_CANDIDATES,
    get_base_features,
    get_expansion_candidates,
)
from .config import (
    CVConfig, MetricConfig, MetricType, ModelConfig, SearchConfig, SubsetResult
)
from .evaluation import SubsetEvaluator
from .display import StatsDisplay

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LooseTightConfig:
    """Configuration for the loose-then-tight pipeline.

    Attributes:
        # Base feature elimination
        run_base_elimination: Whether to run reverse elimination on base features
        epsilon_remove_base: Tolerance for removal during base elimination

        # Loose forward selection
        epsilon_add_loose: Minimum mean improvement to add (very small for high recall)
        min_fold_improvement_ratio_loose: Fraction of folds that must improve
        max_features_loose: Hard cap on features after loose forward selection

        # Strict backward elimination
        epsilon_remove_strict: Tolerance for removal (0 = remove if doesn't hurt)

        # Interactions
        run_interactions: Whether to run interaction discovery
        max_interactions: Maximum interaction features to add
        epsilon_add_interaction: Threshold for adding interactions

        # Swapping
        epsilon_swap: Minimum improvement to accept a swap
        max_swap_iterations: Maximum swap iterations

        # Parallelization
        n_jobs: Number of parallel workers
    """
    # Base feature elimination (quick pruning of BASE_FEATURES)
    run_base_elimination: bool = False  # Disabled by default
    epsilon_remove_base: float = 0.0005  # Slightly more lenient than strict elimination

    # Loose forward selection (high recall)
    epsilon_add_loose: float = 0.0002  # Very small - allow weak features
    min_fold_improvement_ratio_loose: float = 0.6  # 60% of folds must improve
    max_features_loose: int = 80  # Hard cap after loose FS

    # Strict backward elimination (high precision)
    epsilon_remove_strict: float = 0.0  # Remove if doesn't hurt at all

    # Light interaction pass
    run_interactions: bool = True
    max_interactions: int = 8
    epsilon_add_interaction: float = 0.001  # Stricter than loose FS
    n_top_features_for_interactions: int = 20

    # Hill climbing / swapping
    epsilon_swap: float = 0.0005
    max_swap_iterations: int = 50
    n_borderline_features: int = 10  # Features to consider swapping out

    # Parallelization (reuse existing infrastructure)
    n_jobs: int = 8


@dataclass
class SubsetSnapshot:
    """Snapshot of a feature subset at a pipeline stage."""
    stage: str
    features: List[str]
    metric_mean: float
    metric_std: float
    fold_metrics: List[float] = field(default_factory=list)

    def __repr__(self):
        return f"SubsetSnapshot({self.stage}: {len(self.features)} features, {self.metric_mean:.4f}±{self.metric_std:.4f})"


# =============================================================================
# Joblib Worker Functions (for parallel evaluation)
# =============================================================================

def _evaluate_subset_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    model_config: ModelConfig,
    cv_config: CVConfig,
    metric_config: MetricConfig,
    search_config: SearchConfig,
    sample_weight: Optional[pd.Series] = None,
) -> Optional[SubsetResult]:
    """Evaluate a feature subset (joblib worker)."""
    try:
        from .evaluation import SubsetEvaluator
        evaluator = SubsetEvaluator(
            X=X, y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
            sample_weight=sample_weight,
        )
        return evaluator.evaluate(features)
    except Exception as e:
        logger.debug(f"Evaluation failed: {e}")
        return None


def _evaluate_addition_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    current_features: List[str],
    candidate: str,
    model_config: ModelConfig,
    cv_config: CVConfig,
    metric_config: MetricConfig,
    search_config: SearchConfig,
    sample_weight: Optional[pd.Series] = None,
) -> Tuple[str, Optional[SubsetResult]]:
    """Evaluate adding a candidate feature (joblib worker)."""
    try:
        from .evaluation import SubsetEvaluator
        evaluator = SubsetEvaluator(
            X=X, y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
            sample_weight=sample_weight,
        )
        test_features = list(current_features) + [candidate]
        result = evaluator.evaluate(test_features)
        return (candidate, result)
    except Exception as e:
        logger.debug(f"Addition evaluation failed for {candidate}: {e}")
        return (candidate, None)


def _evaluate_removal_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    current_features: List[str],
    feature_to_remove: str,
    model_config: ModelConfig,
    cv_config: CVConfig,
    metric_config: MetricConfig,
    search_config: SearchConfig,
    sample_weight: Optional[pd.Series] = None,
) -> Tuple[str, Optional[SubsetResult]]:
    """Evaluate removing a feature (joblib worker)."""
    try:
        from .evaluation import SubsetEvaluator
        evaluator = SubsetEvaluator(
            X=X, y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
            sample_weight=sample_weight,
        )
        test_features = [f for f in current_features if f != feature_to_remove]
        result = evaluator.evaluate(test_features)
        return (feature_to_remove, result)
    except Exception as e:
        logger.debug(f"Removal evaluation failed for {feature_to_remove}: {e}")
        return (feature_to_remove, None)


def _evaluate_swap_joblib(
    X: pd.DataFrame,
    y: pd.Series,
    current_features: List[str],
    f_out: str,
    f_in: str,
    model_config: ModelConfig,
    cv_config: CVConfig,
    metric_config: MetricConfig,
    search_config: SearchConfig,
    sample_weight: Optional[pd.Series] = None,
) -> Tuple[str, str, Optional[SubsetResult]]:
    """Evaluate swapping features (joblib worker)."""
    try:
        from .evaluation import SubsetEvaluator
        evaluator = SubsetEvaluator(
            X=X, y=y,
            model_config=model_config,
            cv_config=cv_config,
            metric_config=metric_config,
            search_config=search_config,
            sample_weight=sample_weight,
        )
        test_features = [f for f in current_features if f != f_out] + [f_in]
        result = evaluator.evaluate(test_features)
        return (f_out, f_in, result)
    except Exception as e:
        logger.debug(f"Swap evaluation failed {f_out}->{f_in}: {e}")
        return (f_out, f_in, None)


# =============================================================================
# Helper Functions
# =============================================================================

def _check_loose_acceptance(
    baseline_result: SubsetResult,
    candidate_result: SubsetResult,
    epsilon_add_loose: float,
    min_fold_improvement_ratio: float,
) -> Tuple[bool, str]:
    """Check if a candidate should be accepted under loose criteria.

    Accept if ANY of:
    - Mean CV metric improves by at least epsilon_add_loose
    - Median CV metric improves
    - Feature improves in at least min_fold_improvement_ratio of folds

    Returns:
        Tuple of (accepted, reason)
    """
    mean_improvement = candidate_result.metric_main - baseline_result.metric_main

    # Check 1: Mean improvement
    if mean_improvement >= epsilon_add_loose:
        return True, f"mean +{mean_improvement:.4f}"

    # Check 2: Median improvement
    if baseline_result.fold_metrics and candidate_result.fold_metrics:
        baseline_median = np.median(baseline_result.fold_metrics)
        candidate_median = np.median(candidate_result.fold_metrics)
        if candidate_median > baseline_median:
            return True, f"median +{candidate_median - baseline_median:.4f}"

        # Check 3: Fold-level improvement
        n_improved = sum(
            1 for b, c in zip(baseline_result.fold_metrics, candidate_result.fold_metrics)
            if c > b - 1e-6  # Small tolerance
        )
        fold_ratio = n_improved / len(baseline_result.fold_metrics)
        if fold_ratio >= min_fold_improvement_ratio:
            return True, f"{fold_ratio*100:.0f}% folds improved"

    return False, "no improvement"


def _get_feature_importance_from_evaluator(
    evaluator: SubsetEvaluator,
    features: List[str],
) -> Dict[str, float]:
    """Get feature importance scores."""
    try:
        return evaluator.get_feature_importance(features, importance_type='gain')
    except Exception:
        # Fallback to uniform importance
        return {f: 1.0 for f in features}


# =============================================================================
# Main Pipeline Class
# =============================================================================

class LooseTightPipeline:
    """
    Loose-then-Tight Feature Selection Pipeline.

    Implements "high recall early, high precision later" approach:
    1. Start with base features (seed, not forced)
    2. Loose forward selection (over-inclusive)
    3. Strict backward elimination (aggressive pruning)
    4. Light interaction pass (targeted)
    5. Hill climbing / swapping (local optimization)
    6. Final cleanup pass (stability)
    7. Return best subset seen anywhere
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        cv_config: Optional[CVConfig] = None,
        search_config: Optional[SearchConfig] = None,
        metric_config: Optional[MetricConfig] = None,
        pipeline_config: Optional[LooseTightConfig] = None,
    ):
        """Initialize the pipeline.

        Args:
            model_config: Model configuration (LightGBM/XGBoost settings)
            cv_config: Cross-validation configuration
            search_config: Search configuration (used for evaluator)
            metric_config: Metrics configuration
            pipeline_config: Pipeline-specific configuration
        """
        self.model_config = model_config or ModelConfig()
        self.cv_config = cv_config or CVConfig()
        self.search_config = search_config or SearchConfig()
        self.metric_config = metric_config or MetricConfig()
        self.config = pipeline_config or LooseTightConfig()

        # Results tracking
        self.snapshots: List[SubsetSnapshot] = []
        self.best_snapshot: Optional[SubsetSnapshot] = None
        self.stage_log: List[str] = []

        # Data references (set during run)
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._sample_weight: Optional[pd.Series] = None

        # Statistics display
        self._stats_display: Optional[StatsDisplay] = None

        # Checkpointing
        self._checkpoint_dir = Path('artifacts/feature_selection')
        self._checkpoint_file = self._checkpoint_dir / 'checkpoint.pkl'
        self._current_features: Optional[Set[str]] = None
        self._completed_stages: List[str] = []

    def _save_checkpoint(self, stage: str, current_features: Set[str], result: 'SubsetResult'):
        """Save checkpoint after completing a stage."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'stage': stage,
            'current_features': list(current_features),
            'completed_stages': self._completed_stages.copy(),
            'snapshots': self.snapshots.copy(),
            'best_snapshot': self.best_snapshot,
            'result_metric_main': result.metric_main,
            'result_metric_std': result.metric_std,
            'result_fold_metrics': result.fold_metrics.copy() if result.fold_metrics else [],
            'config': self.config,
            'model_config': self.model_config,
            'cv_config': self.cv_config,
            'metric_config': self.metric_config,
            'search_config': self.search_config,
            'timestamp': time.time(),
        }

        # Save to temp file first, then rename (atomic)
        temp_file = self._checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        temp_file.rename(self._checkpoint_file)

        logger.info(f"Checkpoint saved: stage={stage}, features={len(current_features)}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str = None) -> Optional[dict]:
        """Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses default location.

        Returns:
            Checkpoint dict or None if not found.
        """
        if checkpoint_path is None:
            checkpoint_path = Path('artifacts/feature_selection/checkpoint.pkl')
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def has_checkpoint(cls, checkpoint_path: str = None) -> bool:
        """Check if a checkpoint exists."""
        if checkpoint_path is None:
            checkpoint_path = Path('artifacts/feature_selection/checkpoint.pkl')
        else:
            checkpoint_path = Path(checkpoint_path)
        return checkpoint_path.exists()

    @classmethod
    def checkpoint_info(cls, checkpoint_path: str = None) -> Optional[str]:
        """Get human-readable info about a checkpoint."""
        checkpoint = cls.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            return None

        from datetime import datetime
        ts = datetime.fromtimestamp(checkpoint['timestamp'])
        return (
            f"Stage: {checkpoint['stage']}\n"
            f"Features: {len(checkpoint['current_features'])}\n"
            f"Metric: {checkpoint['result_metric_main']:.4f} ± {checkpoint['result_metric_std']:.4f}\n"
            f"Completed: {', '.join(checkpoint['completed_stages'])}\n"
            f"Saved: {ts.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def resume_from_checkpoint(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        checkpoint_path: str = None,
        verbose: bool = True,
    ) -> 'LooseTightPipeline':
        """Resume pipeline from a saved checkpoint.

        Args:
            X: Feature DataFrame (must have same columns as original)
            y: Target Series
            checkpoint_path: Path to checkpoint. If None, uses default.
            verbose: Print progress

        Returns:
            self for chaining
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path or 'default location'}")

        # Restore state
        self._completed_stages = checkpoint['completed_stages']
        self.snapshots = checkpoint['snapshots']
        self.best_snapshot = checkpoint['best_snapshot']
        self._current_features = set(checkpoint['current_features'])

        # Restore configs
        self.config = checkpoint['config']
        self.model_config = checkpoint['model_config']
        self.cv_config = checkpoint['cv_config']
        self.metric_config = checkpoint['metric_config']
        self.search_config = checkpoint['search_config']

        # Set data references
        self._X = X
        self._y = y
        self._verbose = verbose

        last_stage = checkpoint['stage']

        if verbose:
            print(f"\n{'='*70}")
            print("RESUMING FROM CHECKPOINT")
            print(f"{'='*70}")
            print(f"Last completed stage: {last_stage}")
            print(f"Features at checkpoint: {len(self._current_features)}")
            print(f"Metric at checkpoint: {checkpoint['result_metric_main']:.4f}")
            print(f"Completed stages: {', '.join(self._completed_stages)}")
            print()

        # Determine which stages to run
        all_stages = [
            '1_base_features',
            '1b_base_elimination',
            '2_loose_forward',
            '3_strict_backward',
            '4_interactions',
            '5_swapping',
            '6_final_cleanup',
        ]

        # Find where to resume from
        try:
            resume_idx = all_stages.index(last_stage) + 1
        except ValueError:
            resume_idx = 0

        # Get expansion candidates
        all_expansion = [f for f in get_expansion_candidates(flat=True) if f in X.columns]

        # Create a dummy result for continuation
        from .config import SubsetResult
        last_result = SubsetResult(
            features=list(self._current_features),
            metric_main=checkpoint['result_metric_main'],
            metric_std=checkpoint['result_metric_std'],
            fold_metrics=checkpoint['result_fold_metrics'],
        )

        current_features = self._current_features

        # Run remaining stages
        for stage_name in all_stages[resume_idx:]:
            if stage_name == '1b_base_elimination':
                if self.config.run_base_elimination:
                    current_features, last_result = self._base_feature_elimination(
                        current_features
                    )
                    self._completed_stages.append(stage_name)
                    self._record_snapshot(stage_name, current_features, last_result)
                    self._save_checkpoint(stage_name, current_features, last_result)

            elif stage_name == '2_loose_forward':
                current_features, last_result = self._loose_forward_selection(
                    current_features, all_expansion
                )
                self._completed_stages.append(stage_name)
                self._record_snapshot(stage_name, current_features, last_result)
                self._save_checkpoint(stage_name, current_features, last_result)

            elif stage_name == '3_strict_backward':
                current_features, last_result = self._strict_backward_elimination(current_features)
                self._completed_stages.append(stage_name)
                self._record_snapshot(stage_name, current_features, last_result)
                self._save_checkpoint(stage_name, current_features, last_result)

            elif stage_name == '4_interactions':
                if self.config.run_interactions:
                    current_features, last_result = self._light_interaction_pass(
                        current_features, all_expansion
                    )
                    self._completed_stages.append(stage_name)
                    self._record_snapshot(stage_name, current_features, last_result)
                    self._save_checkpoint(stage_name, current_features, last_result)

            elif stage_name == '5_swapping':
                current_features, last_result = self._hill_climbing_swapping(
                    current_features, all_expansion
                )
                self._completed_stages.append(stage_name)
                self._record_snapshot(stage_name, current_features, last_result)
                self._save_checkpoint(stage_name, current_features, last_result)

            elif stage_name == '6_final_cleanup':
                current_features, last_result = self._final_cleanup(current_features)
                self._completed_stages.append(stage_name)
                self._record_snapshot(stage_name, current_features, last_result)
                self._save_checkpoint(stage_name, current_features, last_result)

        # Final best selection
        self._select_best_subset()

        if verbose:
            print(f"\n{'='*70}")
            print("PIPELINE COMPLETE (resumed)")
            print(f"{'='*70}")
            print(f"\nBest subset found at stage: {self.best_snapshot.stage}")
            print(f"Features: {len(self.best_snapshot.features)}")
            print(f"Metric: {self.best_snapshot.metric_mean:.4f} ± {self.best_snapshot.metric_std:.4f}")

        return self

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True,
        sample_weight: Optional[pd.Series] = None,
    ) -> 'LooseTightPipeline':
        """Run the complete pipeline.

        Args:
            X: Feature DataFrame
            y: Target Series
            verbose: Whether to print progress
            sample_weight: Optional sample weights for training (e.g., from overlap inverse).
                          These weights down-weight overlapping trajectories from triple
                          barrier targets to reduce the influence of correlated samples.

        Returns:
            self for chaining
        """
        self._X = X
        self._y = y
        self._verbose = verbose
        self._sample_weight = sample_weight

        # Get base features that exist in the data
        base_features = [f for f in get_base_features() if f in X.columns]
        all_expansion = [f for f in get_expansion_candidates(flat=True) if f in X.columns]

        if verbose:
            print(f"\n{'='*70}")
            print("LOOSE-THEN-TIGHT FEATURE SELECTION PIPELINE")
            print(f"{'='*70}")
            print(f"Base features available: {len(base_features)}")
            print(f"Expansion candidates available: {len(all_expansion)}")
            print()

        # STEP 1: Start with base features (use all cores since no parallelism here)
        current_features = set(base_features)
        baseline_result = self._evaluate_subset(list(current_features), use_all_cores=True)
        self._completed_stages.append("1_base_features")
        self._record_snapshot("1_base_features", current_features, baseline_result)
        self._save_checkpoint("1_base_features", current_features, baseline_result)

        if verbose:
            print(f"STEP 1: Base Features")
            print(f"  Features: {len(current_features)}")
            print(f"  Metric: {baseline_result.metric_main:.4f} ± {baseline_result.metric_std:.4f}")
            self._print_extended_metrics("1_base_features", baseline_result, len(current_features))
            print(f"  [Checkpoint saved]")
            print()

        # Store baseline for comparison
        initial_baseline = baseline_result

        # STEP 1b: Optional base feature elimination (quick pruning)
        if self.config.run_base_elimination:
            current_features, result = self._base_feature_elimination(current_features)
            self._completed_stages.append("1b_base_elimination")
            self._record_snapshot("1b_base_elimination", current_features, result)
            self._save_checkpoint("1b_base_elimination", current_features, result)
            if verbose:
                self._print_comparison_table(initial_baseline, result, "Base Elimination")
                print(f"  [Checkpoint saved]")

        # STEP 2: Loose forward selection
        pre_forward_result = self._evaluate_subset(list(current_features), use_all_cores=True)
        current_features, result = self._loose_forward_selection(
            current_features, all_expansion
        )
        self._completed_stages.append("2_loose_forward")
        self._record_snapshot("2_loose_forward", current_features, result)
        self._save_checkpoint("2_loose_forward", current_features, result)
        if verbose:
            self._print_extended_metrics("2_loose_forward", result, len(current_features))
            self._print_comparison_table(pre_forward_result, result, "Loose Forward Selection")
            print(f"  [Checkpoint saved]")

        # STEP 3: Strict backward elimination
        pre_backward_result = result
        current_features, result = self._strict_backward_elimination(current_features)
        self._completed_stages.append("3_strict_backward")
        self._record_snapshot("3_strict_backward", current_features, result)
        self._save_checkpoint("3_strict_backward", current_features, result)
        if verbose:
            self._print_extended_metrics("3_strict_backward", result, len(current_features))
            self._print_comparison_table(pre_backward_result, result, "Strict Backward Elimination")
            print(f"  [Checkpoint saved]")

        # STEP 4: Light interaction pass (optional)
        if self.config.run_interactions:
            pre_interaction_result = result
            current_features, result = self._light_interaction_pass(
                current_features, all_expansion
            )
            self._completed_stages.append("4_interactions")
            self._record_snapshot("4_interactions", current_features, result)
            self._save_checkpoint("4_interactions", current_features, result)
            if verbose:
                self._print_extended_metrics("4_interactions", result, len(current_features))
                self._print_comparison_table(pre_interaction_result, result, "Interaction Pass")
                print(f"  [Checkpoint saved]")

        # STEP 5: Hill climbing / swapping
        pre_swap_result = result
        current_features, result = self._hill_climbing_swapping(
            current_features, all_expansion
        )
        self._completed_stages.append("5_swapping")
        self._record_snapshot("5_swapping", current_features, result)
        self._save_checkpoint("5_swapping", current_features, result)
        if verbose:
            self._print_extended_metrics("5_swapping", result, len(current_features))
            self._print_comparison_table(pre_swap_result, result, "Hill Climbing / Swapping")
            print(f"  [Checkpoint saved]")

        # STEP 6: Final cleanup pass
        pre_cleanup_result = result
        current_features, result = self._final_cleanup(current_features)
        self._completed_stages.append("6_final_cleanup")
        self._record_snapshot("6_final_cleanup", current_features, result)
        self._save_checkpoint("6_final_cleanup", current_features, result)
        if verbose:
            self._print_extended_metrics("6_final_cleanup", result, len(current_features))
            self._print_comparison_table(pre_cleanup_result, result, "Final Cleanup")
            print(f"  [Checkpoint saved]")

        # STEP 7: Select best subset seen anywhere
        self._select_best_subset()

        if verbose:
            self._print_final_summary(initial_baseline, result)

        return self

    # =========================================================================
    # STEP 1b: Base Feature Elimination (Optional)
    # =========================================================================

    def _base_feature_elimination(
        self,
        features: Set[str],
    ) -> Tuple[Set[str], SubsetResult]:
        """
        STEP 1b: Quick reverse elimination on base features.

        This is a single-pass backward elimination that runs once over all base
        features to quickly prune any that don't contribute. Uses a slightly
        more lenient epsilon than strict elimination to avoid being too aggressive.

        Args:
            features: Set of base features to evaluate.

        Returns:
            Tuple of (pruned feature set, result after pruning)
        """
        if self._verbose:
            print(f"STEP 1b: Base Feature Elimination")
            print(f"  epsilon_remove_base={self.config.epsilon_remove_base}")
            print(f"  (single-pass quick pruning)")
            print("-" * 50)

        current_set = set(features)
        best_result = self._evaluate_subset(list(current_set))
        best_score = best_result.metric_main

        # Parallel evaluation of all removals
        features_list = list(current_set)
        results = Parallel(n_jobs=self.config.n_jobs, backend='loky', verbose=0)(
            delayed(_evaluate_removal_joblib)(
                self._X, self._y, features_list, f,
                self.model_config, self.cv_config,
                self.metric_config, self.search_config,
                self._sample_weight
            )
            for f in features_list
        )

        # Find features that can be removed (don't hurt beyond epsilon)
        removable = []
        for f_name, result in results:
            if result is None:
                continue
            # Remove if: improves OR doesn't hurt beyond epsilon
            if result.metric_main >= best_score - self.config.epsilon_remove_base:
                improvement = result.metric_main - best_score
                removable.append((f_name, result, improvement))

        if not removable:
            if self._verbose:
                print(f"  No removable features found")
                print(f"  Final: {len(current_set)} features, {best_score:.4f}")
                print()
            return current_set, best_result

        # Sort by improvement (best removals first)
        removable.sort(key=lambda x: x[2], reverse=True)

        if self._verbose:
            print(f"  Found {len(removable)} potentially removable features:")

        # Remove features one at a time, re-evaluating after each
        removed_features = []
        for f_name, _, _ in removable:
            if f_name not in current_set:
                continue
            if len(current_set) <= 5:  # Keep at least 5 base features
                if self._verbose:
                    print(f"    Stopping: minimum feature count reached")
                break

            # Re-evaluate removal
            test_set = current_set - {f_name}
            result = self._evaluate_subset(list(test_set))

            if result.metric_main >= best_score - self.config.epsilon_remove_base:
                current_set = test_set
                improvement = result.metric_main - best_score
                best_score = result.metric_main
                best_result = result
                removed_features.append(f_name)

                if self._verbose:
                    sign = "+" if improvement >= 0 else ""
                    print(f"    -{f_name}: {best_score:.4f} ({sign}{improvement:.4f})")

        if self._verbose:
            print(f"\n  Removed {len(removed_features)} base features")
            print(f"  Final: {len(current_set)} features, {best_score:.4f}")
            print()

        gc.collect()
        return current_set, best_result

    # =========================================================================
    # STEP 2: Loose Forward Selection
    # =========================================================================

    def _loose_forward_selection(
        self,
        initial_features: Set[str],
        candidates: List[str],
    ) -> Tuple[Set[str], SubsetResult]:
        """
        STEP 2: Loose forward selection with high recall.

        Accept features under ANY of these criteria:
        - Mean CV metric improves by epsilon_add_loose
        - Median CV metric improves
        - Feature improves in min_fold_improvement_ratio of folds
        """
        if self._verbose:
            print(f"STEP 2: Loose Forward Selection")
            print(f"  epsilon_add_loose={self.config.epsilon_add_loose}")
            print(f"  min_fold_ratio={self.config.min_fold_improvement_ratio_loose}")
            print(f"  max_features={self.config.max_features_loose}")
            print("-" * 50)

        current_set = set(initial_features)
        remaining = [c for c in candidates if c not in current_set]

        # Evaluate baseline
        baseline_result = self._evaluate_subset(list(current_set))
        best_result = baseline_result

        added_features = []
        iteration = 0

        while len(current_set) < self.config.max_features_loose and remaining:
            iteration += 1
            start_time = time.time()

            # Parallel evaluation of all candidates
            results = Parallel(n_jobs=self.config.n_jobs, backend='loky', verbose=0)(
                delayed(_evaluate_addition_joblib)(
                    self._X, self._y, list(current_set), cand,
                    self.model_config, self.cv_config,
                    self.metric_config, self.search_config,
                    self._sample_weight
                )
                for cand in remaining
            )

            # Find candidates that pass loose acceptance criteria
            acceptable = []
            for cand_name, result in results:
                if result is None:
                    continue
                accepted, reason = _check_loose_acceptance(
                    best_result, result,
                    self.config.epsilon_add_loose,
                    self.config.min_fold_improvement_ratio_loose
                )
                if accepted:
                    acceptable.append((cand_name, result, reason))

            if not acceptable:
                if self._verbose:
                    print(f"  No more acceptable features found")
                break

            # Sort by metric and take best
            acceptable.sort(key=lambda x: x[1].metric_main, reverse=True)
            best_cand, best_cand_result, reason = acceptable[0]

            # Add feature
            current_set.add(best_cand)
            remaining.remove(best_cand)
            best_result = best_cand_result
            added_features.append(best_cand)

            elapsed = time.time() - start_time
            if self._verbose:
                print(f"  +{best_cand}: {best_result.metric_main:.4f} ({reason}) [{elapsed:.1f}s]")

            gc.collect()

        if self._verbose:
            print(f"\n  Added {len(added_features)} features")
            print(f"  Final: {len(current_set)} features, {best_result.metric_main:.4f}")
            print()

        return current_set, best_result

    # =========================================================================
    # STEP 3: Strict Backward Elimination
    # =========================================================================

    def _strict_backward_elimination(
        self,
        features: Set[str],
    ) -> Tuple[Set[str], SubsetResult]:
        """
        STEP 3: Strict backward elimination with high precision.

        Remove any feature whose removal:
        - Improves the main metric, OR
        - Does not worsen the metric beyond epsilon_remove_strict

        Base features are NOT protected - they can be removed.
        """
        if self._verbose:
            print(f"STEP 3: Strict Backward Elimination")
            print(f"  epsilon_remove_strict={self.config.epsilon_remove_strict}")
            print(f"  (base features NOT protected)")
            print("-" * 50)

        current_set = set(features)
        best_result = self._evaluate_subset(list(current_set))
        best_score = best_result.metric_main

        removed_features = []
        pass_num = 0

        while True:
            pass_num += 1
            if self._verbose:
                print(f"  Pass {pass_num}: {len(current_set)} features")

            # Parallel evaluation of all removals
            features_list = list(current_set)
            results = Parallel(n_jobs=self.config.n_jobs, backend='loky', verbose=0)(
                delayed(_evaluate_removal_joblib)(
                    self._X, self._y, features_list, f,
                    self.model_config, self.cv_config,
                    self.metric_config, self.search_config,
                    self._sample_weight
                )
                for f in features_list
            )

            # Find features that can be removed
            removable = []
            for f_name, result in results:
                if result is None:
                    continue
                # Remove if: improves OR doesn't hurt beyond epsilon
                if result.metric_main >= best_score - self.config.epsilon_remove_strict:
                    improvement = result.metric_main - best_score
                    removable.append((f_name, result, improvement))

            if not removable:
                if self._verbose:
                    print(f"    No removable features found")
                break

            # Sort by improvement (best removals first)
            removable.sort(key=lambda x: x[2], reverse=True)

            # Remove features one at a time, re-evaluating after each
            removed_this_pass = []
            for f_name, _, _ in removable:
                if f_name not in current_set:
                    continue
                if len(current_set) <= 1:
                    break

                # Re-evaluate removal
                test_set = current_set - {f_name}
                result = self._evaluate_subset(list(test_set))

                if result.metric_main >= best_score - self.config.epsilon_remove_strict:
                    current_set = test_set
                    improvement = result.metric_main - best_score
                    best_score = result.metric_main
                    best_result = result
                    removed_features.append(f_name)
                    removed_this_pass.append(f_name)

                    if self._verbose:
                        sign = "+" if improvement >= 0 else ""
                        print(f"    -{f_name}: {best_score:.4f} ({sign}{improvement:.4f})")

            if not removed_this_pass:
                break

            gc.collect()

        if self._verbose:
            print(f"\n  Removed {len(removed_features)} features")
            print(f"  Final: {len(current_set)} features, {best_score:.4f}")
            print()

        return current_set, best_result

    # =========================================================================
    # STEP 4: Light Interaction Pass
    # =========================================================================

    def _light_interaction_pass(
        self,
        features: Set[str],
        all_candidates: List[str],
    ) -> Tuple[Set[str], SubsetResult]:
        """
        STEP 4: Light pairwise interaction discovery.

        Generate interactions only from:
        - Current selected subset
        - Top globally ranked features

        Use stricter threshold than loose FS.
        """
        if self._verbose:
            print(f"STEP 4: Light Interaction Pass")
            print(f"  max_interactions={self.config.max_interactions}")
            print(f"  epsilon_add_interaction={self.config.epsilon_add_interaction}")
            print("-" * 50)

        current_set = set(features)
        best_result = self._evaluate_subset(list(current_set))
        best_score = best_result.metric_main

        # Get feature importance to identify top features
        evaluator = SubsetEvaluator(
            X=self._X, y=self._y,
            model_config=self.model_config,
            cv_config=self.cv_config,
            metric_config=self.metric_config,
            search_config=self.search_config,
        )
        importance = _get_feature_importance_from_evaluator(evaluator, list(current_set))

        # Sort by importance
        sorted_features = sorted(current_set, key=lambda f: importance.get(f, 0), reverse=True)
        top_features = sorted_features[:self.config.n_top_features_for_interactions]

        # Generate interaction candidates (products of top features)
        interaction_candidates = []
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i+1:]:
                interaction_candidates.append((f1, f2))

        if self._verbose:
            print(f"  Generated {len(interaction_candidates)} interaction candidates")

        added_interactions = []

        for f1, f2 in interaction_candidates:
            if len(added_interactions) >= self.config.max_interactions:
                break

            # Create interaction feature
            interaction_name = f"{f1}_x_{f2}"
            if interaction_name in self._X.columns:
                continue

            try:
                # Compute interaction
                interaction_values = self._X[f1] * self._X[f2]

                # Check for excessive NaNs
                if interaction_values.isna().mean() > 0.5:
                    continue

                # Add to DataFrame temporarily
                self._X[interaction_name] = interaction_values

                # Evaluate
                test_features = list(current_set) + [interaction_name]
                result = self._evaluate_subset(test_features)

                improvement = result.metric_main - best_score

                if improvement >= self.config.epsilon_add_interaction:
                    current_set.add(interaction_name)
                    best_score = result.metric_main
                    best_result = result
                    added_interactions.append(interaction_name)

                    if self._verbose:
                        print(f"  +{interaction_name}: {best_score:.4f} (+{improvement:.4f})")
                else:
                    # Remove from DataFrame if not accepted
                    del self._X[interaction_name]

            except Exception as e:
                logger.debug(f"Interaction {interaction_name} failed: {e}")
                if interaction_name in self._X.columns:
                    del self._X[interaction_name]

        if self._verbose:
            print(f"\n  Added {len(added_interactions)} interactions")
            print(f"  Final: {len(current_set)} features, {best_score:.4f}")
            print()

        return current_set, best_result

    # =========================================================================
    # STEP 5: Hill Climbing / Swapping
    # =========================================================================

    def _hill_climbing_swapping(
        self,
        features: Set[str],
        all_candidates: List[str],
    ) -> Tuple[Set[str], SubsetResult]:
        """
        STEP 5: Hill climbing via pairwise swapping.

        Identify borderline features (weakest in current set) and try
        swapping with promising outside features.
        """
        if self._verbose:
            print(f"STEP 5: Hill Climbing / Swapping")
            print(f"  epsilon_swap={self.config.epsilon_swap}")
            print(f"  max_iterations={self.config.max_swap_iterations}")
            print("-" * 50)

        current_set = set(features)
        outside = [c for c in all_candidates if c not in current_set and c in self._X.columns]

        best_result = self._evaluate_subset(list(current_set))
        best_score = best_result.metric_main

        total_swaps = 0

        for iteration in range(self.config.max_swap_iterations):
            # Get feature importance to identify borderline features
            evaluator = SubsetEvaluator(
                X=self._X, y=self._y,
                model_config=self.model_config,
                cv_config=self.cv_config,
                metric_config=self.metric_config,
                search_config=self.search_config,
            )
            importance = _get_feature_importance_from_evaluator(evaluator, list(current_set))

            # Sort by importance (ascending) - borderline are weakest
            sorted_inside = sorted(current_set, key=lambda f: importance.get(f, 0))
            borderline = sorted_inside[:self.config.n_borderline_features]

            if not borderline or not outside:
                if self._verbose:
                    print(f"  No swap candidates available")
                break

            # Generate swap candidates
            swap_candidates = [(f_out, f_in) for f_out in borderline for f_in in outside[:30]]

            if not swap_candidates:
                break

            # Parallel evaluation of swaps
            results = Parallel(n_jobs=self.config.n_jobs, backend='loky', verbose=0)(
                delayed(_evaluate_swap_joblib)(
                    self._X, self._y, list(current_set), f_out, f_in,
                    self.model_config, self.cv_config,
                    self.metric_config, self.search_config,
                    self._sample_weight
                )
                for f_out, f_in in swap_candidates
            )

            # Find best swap
            best_swap = None
            best_swap_score = best_score + self.config.epsilon_swap
            best_swap_result = None

            for f_out, f_in, result in results:
                if result is None:
                    continue
                if result.metric_main > best_swap_score:
                    best_swap = (f_out, f_in)
                    best_swap_score = result.metric_main
                    best_swap_result = result

            if best_swap is None:
                if self._verbose:
                    print(f"  Iteration {iteration+1}: no improving swap found")
                break

            # Apply swap
            f_out, f_in = best_swap
            current_set = (current_set - {f_out}) | {f_in}
            outside = [c for c in outside if c != f_in] + [f_out]
            improvement = best_swap_score - best_score
            best_score = best_swap_score
            best_result = best_swap_result
            total_swaps += 1

            if self._verbose:
                print(f"  Swap: {f_out} -> {f_in}: {best_score:.4f} (+{improvement:.4f})")

            gc.collect()

        if self._verbose:
            print(f"\n  Total swaps: {total_swaps}")
            print(f"  Final: {len(current_set)} features, {best_score:.4f}")
            print()

        return current_set, best_result

    # =========================================================================
    # STEP 6: Final Cleanup
    # =========================================================================

    def _final_cleanup(
        self,
        features: Set[str],
    ) -> Tuple[Set[str], SubsetResult]:
        """
        STEP 6: Final cleanup pass.

        One last backward elimination with strict criteria.
        Goal is stability and simplicity.
        """
        if self._verbose:
            print(f"STEP 6: Final Cleanup Pass")
            print("-" * 50)

        # Reuse strict backward elimination
        return self._strict_backward_elimination(features)

    # =========================================================================
    # STEP 7: Best Subset Selection
    # =========================================================================

    def _select_best_subset(self):
        """
        STEP 7: Select the best subset seen at any stage.
        """
        if not self.snapshots:
            return

        # Find best snapshot by metric
        self.best_snapshot = max(self.snapshots, key=lambda s: s.metric_mean)

        if self._verbose:
            print(f"\nSTEP 7: Best Subset Selection")
            print("-" * 50)
            print(f"  Evaluating {len(self.snapshots)} snapshots:")
            for s in self.snapshots:
                marker = " <-- BEST" if s == self.best_snapshot else ""
                print(f"    {s.stage}: {len(s.features)} features, {s.metric_mean:.4f}{marker}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _evaluate_subset(self, features: List[str], use_all_cores: bool = False) -> SubsetResult:
        """Evaluate a feature subset.

        Args:
            features: List of feature names to evaluate.
            use_all_cores: If True, use all available cores for this evaluation.
                          Used for base features evaluation where there's no parallelism.
        """
        model_config = self.model_config
        if use_all_cores:
            # Create a copy with all cores for single evaluations (e.g., base features)
            model_config = ModelConfig(
                model_type=self.model_config.model_type,
                task_type=self.model_config.task_type,
                params=self.model_config.params.copy(),
                num_threads=-1,  # Use all cores
                early_stopping_rounds=self.model_config.early_stopping_rounds,
                num_boost_round=self.model_config.num_boost_round,
            )
        evaluator = SubsetEvaluator(
            X=self._X, y=self._y,
            model_config=model_config,
            cv_config=self.cv_config,
            metric_config=self.metric_config,
            search_config=self.search_config,
            sample_weight=self._sample_weight,
        )
        return evaluator.evaluate(features)

    def _record_snapshot(
        self,
        stage: str,
        features: Set[str],
        result: SubsetResult,
    ):
        """Record a snapshot of the current state."""
        snapshot = SubsetSnapshot(
            stage=stage,
            features=list(features),
            metric_mean=result.metric_main,
            metric_std=result.metric_std,
            fold_metrics=result.fold_metrics.copy() if result.fold_metrics else [],
        )
        self.snapshots.append(snapshot)

        # Update best if this is better
        if self.best_snapshot is None or snapshot.metric_mean > self.best_snapshot.metric_mean:
            self.best_snapshot = snapshot

    def _print_final_summary(self, initial_baseline: SubsetResult, final_result: SubsetResult):
        """Print final summary with all stages and overall improvement.

        Args:
            initial_baseline: Initial baseline result (base features).
            final_result: Final result after all stages.
        """
        if not self._verbose:
            return

        print()
        print(f"\n{'=' * 80}")
        print(f"{'PIPELINE COMPLETE - FINAL SUMMARY':^80}")
        print(f"{'=' * 80}")

        # Stage progression table
        print()
        print("  STAGE PROGRESSION")
        print(f"  ┌{'─' * 25}┬{'─' * 10}┬{'─' * 12}┬{'─' * 12}┬{'─' * 12}┬{'─' * 5}┐")
        print(f"  │ {'Stage':<23} │ {'Features':^8} │ {'AUC':^10} │ {'AUPR':^10} │ {'Brier':^10} │ {'Best':^3} │")
        print(f"  ├{'─' * 25}┼{'─' * 10}┼{'─' * 12}┼{'─' * 12}┼{'─' * 12}┼{'─' * 5}┤")

        for snapshot in self.snapshots:
            is_best = snapshot == self.best_snapshot
            best_marker = '★' if is_best else ' '

            # Get metrics from the snapshot (we need to re-evaluate for extended metrics)
            # For now, show primary metric
            print(f"  │ {snapshot.stage:<23} │ {len(snapshot.features):^8} │ {snapshot.metric_mean:^10.4f} │ {'─':^10} │ {'─':^10} │ {best_marker:^3} │")

        print(f"  └{'─' * 25}┴{'─' * 10}┴{'─' * 12}┴{'─' * 12}┴{'─' * 12}┴{'─' * 5}┘")

        # Best result details
        print()
        print(f"  BEST RESULT: {self.best_snapshot.stage}")
        print(f"  {'─' * 40}")
        print(f"  Features: {len(self.best_snapshot.features)}")
        print(f"  Primary Metric (AUC): {self.best_snapshot.metric_mean:.4f} ± {self.best_snapshot.metric_std:.4f}")

        # Overall improvement
        if initial_baseline.metric_main > 0:
            improvement = final_result.metric_main - initial_baseline.metric_main
            pct_improvement = 100 * improvement / initial_baseline.metric_main
            sign = '+' if improvement > 0 else ''
            print()
            print(f"  OVERALL IMPROVEMENT (vs Base Features)")
            print(f"  {'─' * 40}")
            print(f"  AUC: {initial_baseline.metric_main:.4f} → {final_result.metric_main:.4f} ({sign}{improvement:.4f}, {sign}{pct_improvement:.1f}%)")

            # Extended metrics improvement
            for metric_key in ['aupr', 'brier', 'log_loss']:
                if metric_key in initial_baseline.secondary_metrics and metric_key in final_result.secondary_metrics:
                    base_val = initial_baseline.secondary_metrics[metric_key][0]
                    final_val = final_result.secondary_metrics[metric_key][0]
                    delta = final_val - base_val
                    sign = '+' if delta > 0 else ''
                    print(f"  {metric_key.upper()}: {base_val:.4f} → {final_val:.4f} ({sign}{delta:.4f})")

        # Feature list
        print()
        print(f"  SELECTED FEATURES ({len(self.best_snapshot.features)} total)")
        print(f"  {'─' * 40}")
        for i, f in enumerate(sorted(self.best_snapshot.features), 1):
            print(f"  {i:3d}. {f}")

        print()
        print(f"{'=' * 80}")

    def _print_extended_metrics(self, stage: str, result: SubsetResult, n_features: int):
        """Print extended metrics table for a stage result.

        Args:
            stage: Stage name.
            result: SubsetResult with secondary_metrics containing extended metrics.
            n_features: Number of features in subset.
        """
        if not self._verbose:
            return

        # Header
        print()
        print(f"  ┌{'─' * 66}┐")
        print(f"  │ {'METRICS SUMMARY':^64} │")
        print(f"  ├{'─' * 66}┤")
        print(f"  │ Stage: {stage:20s}  Features: {n_features:3d}                   │")
        print(f"  ├{'─' * 10}┬{'─' * 12}┬{'─' * 12}┬{'─' * 10}┬{'─' * 17}┤")
        print(f"  │ {'Metric':^8} │ {'Mean':^10} │ {'Std':^10} │ {'Better':^8} │ {'Interpretation':^15} │")
        print(f"  ├{'─' * 10}┼{'─' * 12}┼{'─' * 12}┼{'─' * 10}┼{'─' * 17}┤")

        # Metric definitions with interpretations
        metrics_info = [
            ('auc', '↑ higher', 'Discrimination'),
            ('aupr', '↑ higher', 'Prec-Recall'),
            ('brier', '↓ lower', 'Calibration'),
            ('log_loss', '↓ lower', 'Likelihood'),
        ]

        for metric_key, direction, interpretation in metrics_info:
            if metric_key in result.secondary_metrics:
                mean, std = result.secondary_metrics[metric_key]
                # Format with appropriate precision
                mean_str = f"{mean:.4f}"
                std_str = f"{std:.4f}"
                print(f"  │ {metric_key.upper():^8} │ {mean_str:^10} │ {std_str:^10} │ {direction:^8} │ {interpretation:^15} │")
            else:
                print(f"  │ {metric_key.upper():^8} │ {'N/A':^10} │ {'N/A':^10} │ {direction:^8} │ {interpretation:^15} │")

        print(f"  └{'─' * 10}┴{'─' * 12}┴{'─' * 12}┴{'─' * 10}┴{'─' * 17}┘")
        print()

    def _print_comparison_table(self, baseline: SubsetResult, current: SubsetResult, stage: str):
        """Print comparison table between baseline and current results.

        Args:
            baseline: Baseline result for comparison.
            current: Current result.
            stage: Stage name.
        """
        if not self._verbose:
            return

        print()
        print(f"  ┌{'─' * 72}┐")
        print(f"  │ {'COMPARISON: ' + stage:^70} │")
        print(f"  ├{'─' * 10}┬{'─' * 14}┬{'─' * 14}┬{'─' * 14}┬{'─' * 15}┤")
        print(f"  │ {'Metric':^8} │ {'Baseline':^12} │ {'Current':^12} │ {'Delta':^12} │ {'Status':^13} │")
        print(f"  ├{'─' * 10}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 15}┤")

        # Metric definitions
        metrics_info = [
            ('auc', True),      # higher is better
            ('aupr', True),     # higher is better
            ('brier', False),   # lower is better
            ('log_loss', False), # lower is better
        ]

        for metric_key, higher_better in metrics_info:
            base_val = baseline.secondary_metrics.get(metric_key, (np.nan, np.nan))[0]
            curr_val = current.secondary_metrics.get(metric_key, (np.nan, np.nan))[0]

            if np.isnan(base_val) or np.isnan(curr_val):
                print(f"  │ {metric_key.upper():^8} │ {'N/A':^12} │ {'N/A':^12} │ {'N/A':^12} │ {'─':^13} │")
                continue

            delta = curr_val - base_val
            # Determine if improved
            if higher_better:
                improved = delta > 0.0001
                worsened = delta < -0.0001
            else:
                improved = delta < -0.0001
                worsened = delta > 0.0001

            status = "✓ Improved" if improved else ("✗ Worsened" if worsened else "─ Same")

            sign = '+' if delta > 0 else ''
            delta_str = f"{sign}{delta:.4f}"

            print(f"  │ {metric_key.upper():^8} │ {base_val:^12.4f} │ {curr_val:^12.4f} │ {delta_str:^12} │ {status:^13} │")

        print(f"  └{'─' * 10}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┴{'─' * 15}┘")
        print()

    # =========================================================================
    # Results Access
    # =========================================================================

    def get_best_features(self) -> List[str]:
        """Get the best feature subset."""
        if self.best_snapshot is None:
            return []
        return self.best_snapshot.features.copy()

    def get_best_metric(self) -> Tuple[float, float]:
        """Get the best metric (mean, std)."""
        if self.best_snapshot is None:
            return (0.0, 0.0)
        return (self.best_snapshot.metric_mean, self.best_snapshot.metric_std)

    def get_stage_summary(self) -> pd.DataFrame:
        """Get summary of all pipeline stages."""
        if not self.snapshots:
            return pd.DataFrame()

        data = []
        for s in self.snapshots:
            data.append({
                'stage': s.stage,
                'n_features': len(s.features),
                'metric_mean': s.metric_mean,
                'metric_std': s.metric_std,
                'is_best': s == self.best_snapshot,
            })
        return pd.DataFrame(data)


# =============================================================================
# Convenience Function
# =============================================================================

def run_loose_tight_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_jobs: int = 8,
    verbose: bool = True,
    sample_weight: Optional[pd.Series] = None,
    **config_kwargs,
) -> LooseTightPipeline:
    """
    Convenience function to run the loose-then-tight pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        n_jobs: Number of parallel workers
        verbose: Whether to print progress
        sample_weight: Optional sample weights for training (e.g., from overlap inverse)
        **config_kwargs: Additional config parameters

    Returns:
        Fitted LooseTightPipeline
    """
    config = LooseTightConfig(n_jobs=n_jobs, **config_kwargs)
    pipeline = LooseTightPipeline(pipeline_config=config)
    pipeline.run(X, y, verbose=verbose, sample_weight=sample_weight)
    return pipeline
