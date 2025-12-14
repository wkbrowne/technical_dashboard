"""Configuration dataclasses for feature selection framework.

This module defines all configuration objects used throughout the feature
selection pipeline, including model, CV, search, and progress configs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class ModelType(Enum):
    """Supported model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"


class TaskType(Enum):
    """ML task type."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class MetricType(Enum):
    """Supported evaluation metrics."""
    AUC = "auc"
    LOG_LOSS = "log_loss"
    IC = "ic"  # Information coefficient (Spearman correlation)
    PRECISION_AT_K = "precision_at_k"
    HIT_RATE = "hit_rate"


class CVScheme(Enum):
    """Cross-validation schemes."""
    ROLLING = "rolling"
    EXPANDING = "expanding"


@dataclass
class ModelConfig:
    """Configuration for the gradient boosting model.

    Attributes:
        model_type: Which model to use (LightGBM or XGBoost).
        task_type: Classification or regression task.
        params: Model hyperparameters passed to the underlying library.
        num_threads: Number of threads for model training (controls nested parallelism).
        early_stopping_rounds: Early stopping patience (None to disable).
        num_boost_round: Maximum number of boosting rounds.
    """
    model_type: ModelType = ModelType.LIGHTGBM
    task_type: TaskType = TaskType.CLASSIFICATION
    params: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1,
    })
    num_threads: int = 1  # Keep low to avoid nested parallelism issues
    early_stopping_rounds: Optional[int] = 50
    num_boost_round: int = 500

    def get_model_params(self) -> Dict[str, Any]:
        """Get full model parameters including thread settings."""
        params = self.params.copy()
        if self.model_type == ModelType.LIGHTGBM:
            params['num_threads'] = self.num_threads
            params['verbose'] = -1
            if self.task_type == TaskType.CLASSIFICATION:
                params.setdefault('objective', 'binary')
                params.setdefault('metric', 'auc')
            else:
                params.setdefault('objective', 'regression')
                params.setdefault('metric', 'rmse')
        elif self.model_type == ModelType.XGBOOST:
            params['nthread'] = self.num_threads
            params['verbosity'] = 0
            if self.task_type == TaskType.CLASSIFICATION:
                params.setdefault('objective', 'binary:logistic')
                params.setdefault('eval_metric', 'auc')
            else:
                params.setdefault('objective', 'reg:squarederror')
        return params


@dataclass
class CVConfig:
    """Configuration for time-series cross-validation.

    Attributes:
        n_splits: Number of CV folds.
        scheme: Rolling or expanding window scheme.
        train_size: Size of training window (for rolling) or minimum train size (for expanding).
            Can be int (number of samples) or float (fraction of data).
        test_size: Size of test window. Can be int or float.
        gap: Number of samples to skip between train and test (embargo period).
        purge_window: Number of samples to purge from end of training set
            to prevent leakage from overlapping events.
        min_train_samples: Minimum number of training samples required.
    """
    n_splits: int = 5
    scheme: CVScheme = CVScheme.EXPANDING
    train_size: Optional[Union[int, float]] = None  # None = determined by n_splits
    test_size: Optional[Union[int, float]] = None  # None = determined by n_splits
    gap: int = 0  # Embargo gap between train and test
    purge_window: int = 0  # Purge window at end of training
    min_train_samples: int = 100

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")
        if self.purge_window < 0:
            raise ValueError("purge_window must be non-negative")


class ParallelOver(Enum):
    """Parallelization strategy for interaction search."""
    INTERACTIONS = "interactions"  # Parallelize over interaction candidates
    FOLDS = "folds"  # Parallelize over CV folds (default for other phases)


@dataclass
class SearchConfig:
    """Configuration for feature search algorithms.

    Attributes:
        epsilon_add: Minimum improvement required to add a feature in forward selection.
        epsilon_remove: Minimum degradation tolerance to remove a feature in backward elimination.
        epsilon_swap: Minimum improvement required to accept a swap.
        max_features: Maximum total features in final subset.
        max_interaction_features: Maximum number of interaction features to include.
        max_swaps: Maximum number of swaps in local search.
        n_core_features: Number of top features to treat as "core" (always keep).
        n_borderline_features: Number of borderline features to consider.
        importance_threshold: Minimum importance to consider a feature.
        n_top_interactions: Number of top interactions to generate.
        interaction_types: Types of interactions to generate ('product', 'threshold').
        n_jobs: Number of parallel jobs for subset evaluation.
        max_concurrent_evals: Maximum concurrent subset evaluations.
        memory_budget_gb: Approximate memory budget in GB (used to tune parallelism).
        random_state: Random seed for reproducibility.

        # Enhanced interaction search parameters
        parallel_over_interactions: Whether to parallelize over interaction candidates.
        interaction_batch_size: Number of interactions to evaluate per batch (memory control).
        interaction_n_jobs: Number of parallel workers for interaction evaluation.
        interaction_num_threads_model: Threads per model during interaction evaluation.
        interaction_pass_after_swapping: Whether to run late interaction refinement.
        epsilon_add_interaction_late: Stricter threshold for late-stage interactions.
        max_late_interactions: Maximum interactions to add in late refinement.
        include_interactions_in_swapping: Whether interactions can participate in swapping.
        n_top_global_for_interactions: Number of top global features for interaction pairs.
        use_domain_filter_interactions: Whether to apply domain-aware filtering.
    """
    # Thresholds for selection decisions
    epsilon_add: float = 0.002  # Minimum improvement to add (ignored if lenient_add=True)
    epsilon_remove: float = 0.001  # Tolerance for removal
    epsilon_swap: float = 0.001  # Minimum improvement for swap

    # Lenient forward selection: add unless clearly harmful
    # When True, adds feature UNLESS it causes significant degradation
    lenient_add: bool = False  # Enable lenient "add unless harmful" mode
    epsilon_harm: float = 0.002  # Degradation threshold to reject (only used if lenient_add=True)
    min_fold_harm_ratio: float = 0.6  # Fraction of folds that must show harm to reject

    # Fold-level acceptance criteria (used when lenient_add=False)
    min_fold_improvement_ratio: float = 0.5  # 50% of folds must show improvement
    use_fold_level_check: bool = True  # Enable fold-level acceptance check

    # Forced exploration: ensure minimum feature count before strict criteria
    min_features: int = 0  # Minimum features to select before applying strict criteria (0 = disabled)
    force_best_per_iteration: bool = False  # If True, always add best feature per iteration until min_features
    warm_start_n: int = 0  # Number of top SHAP features to start with (0 = start empty)

    # Size constraints
    max_features: int = 50
    max_interaction_features: int = 10
    max_swaps: int = 100

    # Feature tiering
    n_core_features: int = 10
    n_borderline_features: int = 50
    importance_threshold: float = 0.001

    # Interaction discovery (original)
    n_top_interactions: int = 20
    interaction_types: List[str] = field(default_factory=lambda: ['product'])

    # Compute resources
    n_jobs: int = -1  # -1 = use all cores
    max_concurrent_evals: int = 4
    memory_budget_gb: float = 32.0

    # Forward selection parallelism (controls nested parallelism)
    # Total threads = forward_n_jobs × folds × model_threads
    # Recommend: forward_n_jobs * model_threads ≈ CPU_count
    forward_selection_n_jobs: int = 8  # Parallel candidate evaluations (0 = sequential)
    forward_selection_model_threads: int = 1  # Threads per model during parallel forward selection

    # Reproducibility
    random_state: Optional[int] = 42

    # Enhanced interaction search parameters
    parallel_over_interactions: bool = True  # Parallelize over interaction candidates
    interaction_batch_size: int = 20  # Evaluate in batches for memory safety
    interaction_n_jobs: int = -1  # Workers for interaction evaluation (-1 = all cores)
    interaction_num_threads_model: int = 1  # Threads per model (keep low to avoid nested parallelism)
    interaction_pass_after_swapping: bool = False  # Run late interaction refinement
    epsilon_add_interaction_late: float = 0.005  # Stricter threshold for late interactions
    max_late_interactions: int = 2  # Maximum interactions in late refinement
    include_interactions_in_swapping: bool = True  # Allow interactions in swap candidates
    n_top_global_for_interactions: int = 30  # Top global features for interaction pairs
    use_domain_filter_interactions: bool = True  # Apply domain-aware filtering

    def get_effective_n_jobs(self) -> int:
        """Get effective number of jobs, resolving -1 to actual core count."""
        import os
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return min(self.n_jobs, os.cpu_count() or 1)

    def get_effective_interaction_n_jobs(self) -> int:
        """Get effective number of jobs for interaction evaluation."""
        import os
        if self.interaction_n_jobs == -1:
            return os.cpu_count() or 1
        return min(self.interaction_n_jobs, os.cpu_count() or 1)


@dataclass
class ProgressConfig:
    """Configuration for progress reporting.

    Attributes:
        enable_console_logging: Whether to print progress to console.
        update_frequency_evals: Print update every N evaluations.
        update_frequency_seconds: Print update at least every N seconds.
        progress_callback: Optional callback function for custom progress handling.
        verbose: Verbosity level (0=silent, 1=phases, 2=detailed).
    """
    enable_console_logging: bool = True
    update_frequency_evals: int = 10
    update_frequency_seconds: float = 30.0
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    verbose: int = 1


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics.

    Attributes:
        primary_metric: Main metric for optimization.
        secondary_metrics: Additional metrics to track.
        tail_quantile: Quantile for tail metrics (e.g., 0.1 for top/bottom 10%).
        regime_column: Optional column name for regime-stratified evaluation.
    """
    primary_metric: MetricType = MetricType.AUC
    secondary_metrics: List[MetricType] = field(default_factory=list)
    tail_quantile: float = 0.1
    regime_column: Optional[str] = None


@dataclass
class SubsetResult:
    """Result from evaluating a feature subset.

    Attributes:
        features: List of feature names in the subset.
        metric_main: Mean of primary metric across folds.
        metric_std: Standard deviation of primary metric across folds.
        fold_metrics: Per-fold primary metric values.
        secondary_metrics: Dict of secondary metric name -> (mean, std).
        regime_metrics: Dict of regime -> primary metric value.
        n_base_features: Count of base (non-interaction) features.
        n_interaction_features: Count of interaction features.
    """
    features: List[str]
    metric_main: float
    metric_std: float
    fold_metrics: List[float] = field(default_factory=list)
    secondary_metrics: Dict[str, tuple] = field(default_factory=dict)
    regime_metrics: Dict[str, float] = field(default_factory=dict)
    n_base_features: int = 0
    n_interaction_features: int = 0

    def __lt__(self, other: 'SubsetResult') -> bool:
        """Compare by primary metric (higher is better)."""
        return self.metric_main < other.metric_main

    def __repr__(self) -> str:
        return (f"SubsetResult(n_features={len(self.features)}, "
                f"metric={self.metric_main:.4f}±{self.metric_std:.4f})")


@dataclass
class FeatureRanking:
    """Ranked features with importance scores and tiers.

    Attributes:
        feature_names: List of all feature names, sorted by importance (descending).
        importance_scores: Dict of feature name -> importance score.
        core_features: Set of core (top tier) feature names.
        borderline_features: Set of borderline (middle tier) feature names.
        low_value_features: Set of low value (bottom tier) feature names.
    """
    feature_names: List[str]
    importance_scores: Dict[str, float]
    core_features: set = field(default_factory=set)
    borderline_features: set = field(default_factory=set)
    low_value_features: set = field(default_factory=set)

    def get_tier(self, feature: str) -> str:
        """Get the tier of a feature."""
        if feature in self.core_features:
            return "core"
        elif feature in self.borderline_features:
            return "borderline"
        else:
            return "low_value"
