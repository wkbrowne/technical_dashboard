"""Feature selection framework for ML-based trading models.

This module provides a comprehensive feature selection framework designed
for quantitative finance applications, with support for:

- Time-series aware cross-validation (no lookahead, purging, embargo)
- Forward selection, backward elimination, and pairwise swapping
- SHAP-based interaction discovery
- Memory-efficient evaluation of many feature subsets
- Progress tracking with ETA estimation

Quick Start
-----------
```python
from src.feature_selection import FeatureSubsetSearch, create_default_search

# Create search with defaults
search = create_default_search(task='classification', n_folds=5)

# Run the full pipeline
search.fit(X, y)

# Get results
best = search.get_best_subset()
print(f"Best features: {best.features}")
print(f"Best AUC: {best.metric_main:.4f}")

# Train final model
model = search.train_final_model()
```

Advanced Usage
--------------
```python
from src.feature_selection import (
    FeatureSubsetSearch,
    ModelConfig, ModelType, TaskType,
    CVConfig, CVScheme,
    SearchConfig,
    MetricConfig, MetricType,
    ProgressConfig
)

# Configure model
model_config = ModelConfig(
    model_type=ModelType.LIGHTGBM,
    task_type=TaskType.CLASSIFICATION,
    params={'learning_rate': 0.03, 'max_depth': 5}
)

# Configure CV with purging
cv_config = CVConfig(
    n_splits=5,
    scheme=CVScheme.EXPANDING,
    gap=5,  # 5-sample embargo
    purge_window=2
)

# Configure search
search_config = SearchConfig(
    epsilon_add=0.002,
    epsilon_swap=0.001,
    max_features=30,
    max_interaction_features=5,
    n_jobs=-1  # Use all cores
)

# Configure metrics
metric_config = MetricConfig(
    primary_metric=MetricType.AUC,
    secondary_metrics=[MetricType.LOG_LOSS],
    tail_quantile=0.1
)

# Configure progress (with custom callback)
def my_callback(info):
    print(f"Phase: {info['phase']}, ETA: {info['eta_seconds']:.0f}s")

progress_config = ProgressConfig(
    enable_console_logging=True,
    progress_callback=my_callback
)

# Create and run search
search = FeatureSubsetSearch(
    model_config=model_config,
    cv_config=cv_config,
    search_config=search_config,
    metric_config=metric_config,
    progress_config=progress_config
)

search.fit(X, y, regime=regime_labels)
```

Running Individual Steps
------------------------
```python
from src.feature_selection.evaluation import SubsetEvaluator
from src.feature_selection.algorithms import (
    compute_initial_ranking,
    forward_selection,
    backward_elimination,
    pairwise_swapping
)

# Create evaluator
evaluator = SubsetEvaluator(X, y, model_config, cv_config, metric_config, search_config)

# Run individual steps
ranking = compute_initial_ranking(evaluator, list(X.columns), search_config)
features, result = forward_selection(evaluator, ranking.core_features, candidates, search_config)
```
"""

# Configuration classes
from .config import (
    # Enums
    ModelType,
    TaskType,
    MetricType,
    CVScheme,
    ParallelOver,  # New: parallelization strategy for interactions
    # Config dataclasses
    ModelConfig,
    CVConfig,
    SearchConfig,
    MetricConfig,
    ProgressConfig,
    # Result dataclasses
    SubsetResult,
    FeatureRanking,
)

# Main search class
from .search import (
    FeatureSubsetSearch,
    create_default_search,
)

# Loose-then-tight pipeline
from .pipeline import (
    LooseTightPipeline,
    LooseTightConfig,
    run_loose_tight_selection,
)

# Base features configuration
from .base_features import (
    # Core data structures
    BASE_FEATURES,
    CORE_FEATURES,
    HEAD_FEATURES,
    FEATURE_CATEGORIES,
    EXPANSION_CANDIDATES,
    EXCLUDED_FEATURES,
    # Model-aware feature retrieval
    get_core_features,
    get_head_features,
    get_featureset,
    get_all_selectable_features,
    get_all_head_features,
    get_features_required_for_model,
    get_retired_features_safe_to_skip,
    # Legacy functions
    get_base_features,
    get_expansion_candidates,
    get_excluded_features,
    # Validation
    validate_features,
    validate_model_featuresets,
    report_head_features_status,
)

# CV utilities
from .cv import (
    TimeSeriesCV,
    PurgedKFold,
    create_cv_splitter,
    get_fold_info,
)

# Evaluation
from .evaluation import (
    SubsetEvaluator,
    EvaluationCache,
    evaluate_subset,
    evaluate_subsets_parallel,
)

# Algorithms
from .algorithms import (
    compute_initial_ranking,
    forward_selection,
    backward_elimination,
    pairwise_swapping,
    add_interaction_features_forward,
    TopKTracker,
    # Enhanced interaction search
    parallel_interaction_forward_selection,
    late_interaction_refinement,
)

# Interactions
from .interactions import (
    InteractionDiscoverer,
    compute_shap_interactions,
    generate_interaction_features,
    filter_interactions,
    # Enhanced pairwise interaction search
    InteractionCandidate,
    PairwiseInteractionSearch,
    generate_interaction_candidates,
    compute_interaction_lazily,
    batch_compute_interactions,
    filter_candidates_by_domain,
    filter_candidates_by_features,
)

# Models
from .models import (
    GBMWrapper,
    create_model,
    train_and_predict,
)

# Progress
from .progress import (
    ProgressTracker,
    create_progress_tracker,
    format_duration,
)

# Metrics
from .metrics import (
    MetricComputer,
    compute_auc,
    compute_ic,
    compute_log_loss,
    compute_precision_at_k,
    compute_tail_metrics,
    compute_regime_metrics,
)

# Utilities
from .utils import (
    validate_data,
    filter_features_by_variance,
    filter_features_by_nan_rate,
    filter_correlated_features,
    categorize_features,
    print_feature_summary,
    estimate_compute_time,
    cleanup_memory,
)

# Multi-model selection support
from .multimodel import (
    FeatureSelectionResult,
    MultiModelSelectionSummary,
    run_single_model_selection,
    compute_overlap_analysis,
    compute_run_signature,
    compute_cv_config_hash,
    compute_universe_hash,
    write_per_model_artifacts,
    write_global_summary,
    validate_selected_features,
    # Auto-update base_features.py
    generate_base_features_update,
    update_base_features_file,
    compute_feature_diff,
    print_feature_diff,
)

__all__ = [
    # Enums
    'ModelType',
    'TaskType',
    'MetricType',
    'CVScheme',
    'ParallelOver',
    # Configs
    'ModelConfig',
    'CVConfig',
    'SearchConfig',
    'MetricConfig',
    'ProgressConfig',
    'LooseTightConfig',
    # Results
    'SubsetResult',
    'FeatureRanking',
    # Main class
    'FeatureSubsetSearch',
    'create_default_search',
    # Loose-then-tight pipeline
    'LooseTightPipeline',
    'run_loose_tight_selection',
    # Base features - core data structures
    'BASE_FEATURES',
    'CORE_FEATURES',
    'HEAD_FEATURES',
    'FEATURE_CATEGORIES',
    'EXPANSION_CANDIDATES',
    'EXCLUDED_FEATURES',
    # Model-aware feature retrieval
    'get_core_features',
    'get_head_features',
    'get_featureset',
    'get_all_selectable_features',
    'get_all_head_features',
    'get_features_required_for_model',
    'get_retired_features_safe_to_skip',
    # Legacy functions
    'get_base_features',
    'get_expansion_candidates',
    'get_excluded_features',
    # Validation
    'validate_features',
    'validate_model_featuresets',
    'report_head_features_status',
    # CV
    'TimeSeriesCV',
    'PurgedKFold',
    'create_cv_splitter',
    'get_fold_info',
    # Evaluation
    'SubsetEvaluator',
    'EvaluationCache',
    'evaluate_subset',
    'evaluate_subsets_parallel',
    # Algorithms
    'compute_initial_ranking',
    'forward_selection',
    'backward_elimination',
    'pairwise_swapping',
    'add_interaction_features_forward',
    'TopKTracker',
    'parallel_interaction_forward_selection',
    'late_interaction_refinement',
    # Interactions
    'InteractionDiscoverer',
    'compute_shap_interactions',
    'generate_interaction_features',
    'filter_interactions',
    'InteractionCandidate',
    'PairwiseInteractionSearch',
    'generate_interaction_candidates',
    'compute_interaction_lazily',
    'batch_compute_interactions',
    'filter_candidates_by_domain',
    'filter_candidates_by_features',
    # Models
    'GBMWrapper',
    'create_model',
    'train_and_predict',
    # Progress
    'ProgressTracker',
    'create_progress_tracker',
    'format_duration',
    # Metrics
    'MetricComputer',
    'compute_auc',
    'compute_ic',
    'compute_log_loss',
    'compute_precision_at_k',
    'compute_tail_metrics',
    'compute_regime_metrics',
    # Utils
    'validate_data',
    'filter_features_by_variance',
    'filter_features_by_nan_rate',
    'filter_correlated_features',
    'categorize_features',
    'print_feature_summary',
    'estimate_compute_time',
    'cleanup_memory',
    # Multi-model selection
    'FeatureSelectionResult',
    'MultiModelSelectionSummary',
    'run_single_model_selection',
    'compute_overlap_analysis',
    'compute_run_signature',
    'compute_cv_config_hash',
    'compute_universe_hash',
    'write_per_model_artifacts',
    'write_global_summary',
    'validate_selected_features',
    # Auto-update base_features.py
    'generate_base_features_update',
    'update_base_features_file',
    'compute_feature_diff',
    'print_feature_diff',
]

__version__ = '1.0.0'
