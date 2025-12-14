"""Main feature subset search class.

This module provides the FeatureSubsetSearch class, which orchestrates
the complete feature selection pipeline from initial ranking through
local search optimization.
"""

import gc
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from .config import (
    CVConfig, CVScheme, FeatureRanking, MetricConfig, MetricType,
    ModelConfig, ModelType, ProgressConfig, SearchConfig, SubsetResult, TaskType
)
from .cv import TimeSeriesCV
from .evaluation import EvaluationCache, SubsetEvaluator
from .algorithms import (
    TopKTracker, add_interaction_features_forward, backward_elimination,
    compute_initial_ranking, forward_selection, pairwise_swapping,
    parallel_interaction_forward_selection, late_interaction_refinement
)
from .interactions import (
    InteractionDiscoverer,
    InteractionCandidate,
    PairwiseInteractionSearch,
    generate_interaction_candidates
)
from .models import GBMWrapper
from .progress import ProgressTracker, create_progress_tracker


class FeatureSubsetSearch:
    """Main class for feature subset search and selection.

    Orchestrates the complete feature selection pipeline:
    1. Initial feature ranking
    2. Forward selection from core features
    3. Backward elimination to prune redundancy
    4. Interaction discovery and forward selection
    5. Pairwise swapping local search
    6. Tracking and returning top-K subsets

    Attributes:
        model_config: Model configuration.
        cv_config: Cross-validation configuration.
        search_config: Search algorithm configuration.
        metric_config: Metrics configuration.
        progress_config: Progress reporting configuration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        cv_config: Optional[CVConfig] = None,
        search_config: Optional[SearchConfig] = None,
        metric_config: Optional[MetricConfig] = None,
        progress_config: Optional[ProgressConfig] = None
    ):
        """Initialize the feature search.

        Args:
            model_config: Model settings (defaults to LightGBM classification).
            cv_config: CV settings (defaults to 5-fold expanding window).
            search_config: Search settings.
            metric_config: Metrics settings (defaults to AUC).
            progress_config: Progress reporting settings.
        """
        self.model_config = model_config or ModelConfig()
        self.cv_config = cv_config or CVConfig()
        self.search_config = search_config or SearchConfig()
        self.metric_config = metric_config or MetricConfig()
        self.progress_config = progress_config or ProgressConfig()

        # Internal state
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._regime: Optional[pd.Series] = None
        self._evaluator: Optional[SubsetEvaluator] = None
        self._cache: Optional[EvaluationCache] = None
        self._progress: Optional[ProgressTracker] = None
        self._top_k_tracker: Optional[TopKTracker] = None

        # Results
        self._ranking: Optional[FeatureRanking] = None
        self._best_result: Optional[SubsetResult] = None
        self._interaction_features: List[str] = []
        self._pairwise_search: Optional[PairwiseInteractionSearch] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime: Optional[pd.Series] = None,
        run_interactions: bool = True,
        run_swapping: bool = True
    ) -> 'FeatureSubsetSearch':
        """Run the complete feature selection pipeline.

        Args:
            X: Feature DataFrame with DateTimeIndex.
            y: Target Series (classification labels or regression values).
            regime: Optional regime labels for stratified evaluation.
            run_interactions: Whether to discover and add interaction features.
            run_swapping: Whether to run pairwise swapping local search.

        Returns:
            self for chaining.
        """
        # Store data references
        self._X = X
        self._y = y
        self._regime = regime

        # Initialize evaluator
        self._evaluator = SubsetEvaluator(
            X=X,
            y=y,
            model_config=self.model_config,
            cv_config=self.cv_config,
            metric_config=self.metric_config,
            search_config=self.search_config,
            regime=regime
        )

        # Initialize cache and trackers
        self._cache = EvaluationCache(max_size=10000)
        self._top_k_tracker = TopKTracker(k=20)

        # Initialize progress tracker
        self._progress = create_progress_tracker(
            enable_console=self.progress_config.enable_console_logging,
            update_freq_evals=self.progress_config.update_frequency_evals,
            update_freq_seconds=self.progress_config.update_frequency_seconds,
            callback=self.progress_config.progress_callback,
            verbose=self.progress_config.verbose
        )

        all_features = list(X.columns)

        # Step 1: Initial feature ranking
        self._ranking = compute_initial_ranking(
            self._evaluator,
            all_features,
            self.search_config,
            self._progress
        )

        # Step 2: Forward selection from core features
        initial_set = self._ranking.core_features.copy()
        candidate_pool = list(self._ranking.borderline_features | self._ranking.low_value_features)

        current_features, result = forward_selection(
            self._evaluator,
            initial_features=initial_set,
            candidate_features=candidate_pool,
            search_config=self.search_config,
            cache=self._cache,
            progress=self._progress
        )
        self._top_k_tracker.update(result)

        # Step 3: Backward elimination
        current_features, result = backward_elimination(
            self._evaluator,
            features=current_features,
            search_config=self.search_config,
            protected_features=self._ranking.core_features,
            cache=self._cache,
            progress=self._progress
        )
        self._top_k_tracker.update(result)
        base_features = current_features.copy()

        # Step 4 & 5: Interaction discovery and forward selection
        if run_interactions:
            current_features, result = self._run_interaction_phase(
                base_features, all_features
            )
            self._top_k_tracker.update(result)

        # Step 6: Pairwise swapping local search
        if run_swapping:
            # Build outside candidates: features not in current set but with reasonable importance
            outside_candidates = [
                f for f in all_features
                if f not in current_features
                and f not in self._interaction_features
                and self._ranking.importance_scores.get(f, 0) >= self.search_config.importance_threshold
            ]

            # Include promising interactions in swap candidates if enabled
            # NOTE: Only include interactions that are actually in the DataFrame
            if (self.search_config.include_interactions_in_swapping and
                self._pairwise_search is not None):
                # Add top evaluated interactions that weren't selected
                # BUT only if they're actually in the DataFrame
                top_interactions = self._pairwise_search.get_top_evaluated(
                    n=10,
                    min_improvement=-0.01  # Include even slightly negative ones
                )
                for cand, delta in top_interactions:
                    # Only add if: not already in current features AND exists in DataFrame
                    if (cand.feature_name not in current_features and
                        cand.feature_name in self._X.columns):
                        outside_candidates.append(cand.feature_name)

            current_features, result = pairwise_swapping(
                self._evaluator,
                features=current_features,
                outside_candidates=outside_candidates,
                search_config=self.search_config,
                protected_features=self._ranking.core_features,
                cache=self._cache,
                progress=self._progress
            )
            self._top_k_tracker.update(result)

        # Step 7: Late interaction refinement (optional)
        if (run_interactions and
            self.search_config.interaction_pass_after_swapping):
            current_features, result, late_interactions = late_interaction_refinement(
                X=self._X,
                y=self._y,
                evaluator=self._evaluator,
                current_features=current_features,
                feature_ranking=self._ranking,
                search_config=self.search_config,
                model_config=self.model_config,
                cache=self._cache,
                progress=self._progress
            )
            self._interaction_features.extend(late_interactions)
            self._top_k_tracker.update(result)

        # Final result
        self._best_result = self._top_k_tracker.get_best()

        # Cleanup
        self._progress.finish()
        if self._pairwise_search:
            self._pairwise_search.cleanup()
        gc.collect()

        return self

    def _run_interaction_phase(
        self,
        base_features: Set[str],
        all_features: List[str]
    ) -> Tuple[Set[str], SubsetResult]:
        """Run interaction discovery and selection using parallel pairwise search.

        Uses the enhanced pairwise candidate search with lazy evaluation
        and parallel processing for memory-efficient interaction discovery.

        Args:
            base_features: Current base feature set.
            all_features: All available features.

        Returns:
            Tuple of (extended_features, best_result).
        """
        # Get top global features for interaction pairs
        n_top_global = self.search_config.n_top_global_for_interactions
        top_global = list(self._ranking.feature_names[:n_top_global])

        # Phase 1: Generate interaction candidates
        if self._progress:
            self._progress.start_phase("interaction_candidate_generation", total_evals=1)

        # Generate candidates from base subset + top global features
        candidates = generate_interaction_candidates(
            base_features=list(base_features),
            top_global_features=top_global,
            feature_importance=self._ranking.importance_scores,
            interaction_types=self.search_config.interaction_types,
            use_domain_filter=self.search_config.use_domain_filter_interactions,
            shap_interactions_df=None  # Skip SHAP for candidate generation
        )

        if self._progress:
            self._progress.update(1, 0.0, len(base_features))
            self._progress.end_phase()

        if not candidates:
            # No candidates generated, return base
            return base_features, self._evaluator.evaluate(base_features)

        # Limit candidates to top N for evaluation
        max_candidates = self.search_config.n_top_interactions * 2
        candidates = candidates[:max_candidates]

        # Phase 2: Parallel forward selection of interactions
        current_features, result, selected_interactions, pairwise_search = \
            parallel_interaction_forward_selection(
                X=self._X,
                y=self._y,
                evaluator=self._evaluator,
                base_features=base_features,
                interaction_candidates=candidates,
                search_config=self.search_config,
                model_config=self.model_config,
                cache=self._cache,
                progress=self._progress
            )

        # Store interaction features and pairwise search for swapping phase
        self._interaction_features = selected_interactions
        self._pairwise_search = pairwise_search

        # If interactions were added, update evaluator with extended DataFrame
        if selected_interactions:
            self._evaluator = SubsetEvaluator(
                X=self._X,
                y=self._y,
                model_config=self.model_config,
                cv_config=self.cv_config,
                metric_config=self.metric_config,
                search_config=self.search_config,
                regime=self._regime
            )

        return current_features, result

    def get_best_subset(self) -> Optional[SubsetResult]:
        """Get the best feature subset found.

        Returns:
            SubsetResult with the best features and metrics.
        """
        return self._best_result

    def get_top_subsets(self, k: int = 5) -> List[SubsetResult]:
        """Get the top K feature subsets.

        Args:
            k: Number of subsets to return.

        Returns:
            List of SubsetResult objects sorted by metric (best first).
        """
        if self._top_k_tracker is None:
            return []
        return self._top_k_tracker.get_top_k(k)

    def get_feature_ranking(self) -> Optional[FeatureRanking]:
        """Get the initial feature ranking.

        Returns:
            FeatureRanking with importance scores and tiers.
        """
        return self._ranking

    def get_interaction_features(self) -> List[str]:
        """Get the generated interaction feature names.

        Returns:
            List of interaction feature names.
        """
        return self._interaction_features

    def train_final_model(
        self,
        features: Optional[List[str]] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None
    ) -> GBMWrapper:
        """Train a final model on the selected features.

        Args:
            features: Features to use (defaults to best subset).
            X: Training data (defaults to fit data).
            y: Training targets (defaults to fit targets).

        Returns:
            Trained GBMWrapper model.
        """
        features = features or (self._best_result.features if self._best_result else [])
        X = X if X is not None else self._X
        y = y if y is not None else self._y

        if not features or X is None or y is None:
            raise ValueError("No features or data available for training")

        # Filter to valid features
        valid_features = [f for f in features if f in X.columns]

        X_train = X[valid_features]
        mask = ~(X_train.isna().any(axis=1) | y.isna())
        X_train = X_train.loc[mask]
        y_train = y.loc[mask]

        model = GBMWrapper(self.model_config)
        model.train(X_train, y_train, feature_names=valid_features)

        return model

    def save(self, path: Union[str, Path]):
        """Save the search results to disk.

        Args:
            path: Path to save directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        results = {
            'best_subset': {
                'features': self._best_result.features if self._best_result else [],
                'metric_main': self._best_result.metric_main if self._best_result else 0,
                'metric_std': self._best_result.metric_std if self._best_result else 0,
            },
            'top_subsets': [
                {
                    'features': r.features,
                    'metric_main': r.metric_main,
                    'metric_std': r.metric_std,
                }
                for r in self.get_top_subsets(10)
            ],
            'interaction_features': self._interaction_features,
            'ranking': {
                'feature_names': self._ranking.feature_names if self._ranking else [],
                'importance_scores': self._ranking.importance_scores if self._ranking else {},
                'core_features': list(self._ranking.core_features) if self._ranking else [],
                'borderline_features': list(self._ranking.borderline_features) if self._ranking else [],
            }
        }

        with open(path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save configs
        configs = {
            'model_config': {
                'model_type': self.model_config.model_type.value,
                'task_type': self.model_config.task_type.value,
                'params': self.model_config.params,
                'num_threads': self.model_config.num_threads,
                'early_stopping_rounds': self.model_config.early_stopping_rounds,
                'num_boost_round': self.model_config.num_boost_round,
            },
            'cv_config': {
                'n_splits': self.cv_config.n_splits,
                'scheme': self.cv_config.scheme.value,
                'gap': self.cv_config.gap,
                'purge_window': self.cv_config.purge_window,
                'min_train_samples': self.cv_config.min_train_samples,
            },
            'search_config': {
                'epsilon_add': self.search_config.epsilon_add,
                'epsilon_remove': self.search_config.epsilon_remove,
                'epsilon_swap': self.search_config.epsilon_swap,
                'max_features': self.search_config.max_features,
                'max_interaction_features': self.search_config.max_interaction_features,
                'max_swaps': self.search_config.max_swaps,
                'n_core_features': self.search_config.n_core_features,
                'n_borderline_features': self.search_config.n_borderline_features,
            }
        }

        with open(path / 'config.json', 'w') as f:
            json.dump(configs, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureSubsetSearch':
        """Load search results from disk.

        Args:
            path: Path to saved results directory.

        Returns:
            FeatureSubsetSearch instance with loaded results.
        """
        path = Path(path)

        # Load configs
        with open(path / 'config.json', 'r') as f:
            configs = json.load(f)

        model_config = ModelConfig(
            model_type=ModelType(configs['model_config']['model_type']),
            task_type=TaskType(configs['model_config']['task_type']),
            params=configs['model_config']['params'],
            num_threads=configs['model_config']['num_threads'],
            early_stopping_rounds=configs['model_config']['early_stopping_rounds'],
            num_boost_round=configs['model_config']['num_boost_round'],
        )

        cv_config = CVConfig(
            n_splits=configs['cv_config']['n_splits'],
            scheme=CVScheme(configs['cv_config']['scheme']),
            gap=configs['cv_config']['gap'],
            purge_window=configs['cv_config']['purge_window'],
            min_train_samples=configs['cv_config']['min_train_samples'],
        )

        search_config = SearchConfig(
            epsilon_add=configs['search_config']['epsilon_add'],
            epsilon_remove=configs['search_config']['epsilon_remove'],
            epsilon_swap=configs['search_config']['epsilon_swap'],
            max_features=configs['search_config']['max_features'],
            max_interaction_features=configs['search_config']['max_interaction_features'],
            max_swaps=configs['search_config']['max_swaps'],
            n_core_features=configs['search_config']['n_core_features'],
            n_borderline_features=configs['search_config']['n_borderline_features'],
        )

        instance = cls(
            model_config=model_config,
            cv_config=cv_config,
            search_config=search_config
        )

        # Load results
        with open(path / 'results.json', 'r') as f:
            results = json.load(f)

        if results['best_subset']['features']:
            instance._best_result = SubsetResult(
                features=results['best_subset']['features'],
                metric_main=results['best_subset']['metric_main'],
                metric_std=results['best_subset']['metric_std'],
            )

        instance._interaction_features = results.get('interaction_features', [])

        if results['ranking']['feature_names']:
            instance._ranking = FeatureRanking(
                feature_names=results['ranking']['feature_names'],
                importance_scores=results['ranking']['importance_scores'],
                core_features=set(results['ranking']['core_features']),
                borderline_features=set(results['ranking']['borderline_features']),
            )

        # Rebuild top-k tracker
        instance._top_k_tracker = TopKTracker(k=20)
        for subset_data in results.get('top_subsets', []):
            result = SubsetResult(
                features=subset_data['features'],
                metric_main=subset_data['metric_main'],
                metric_std=subset_data['metric_std'],
            )
            instance._top_k_tracker.update(result)

        return instance


def create_default_search(
    task: str = 'classification',
    n_folds: int = 5,
    n_jobs: int = -1,
    verbose: int = 1
) -> FeatureSubsetSearch:
    """Create a FeatureSubsetSearch with sensible defaults.

    Args:
        task: 'classification' or 'regression'.
        n_folds: Number of CV folds.
        n_jobs: Number of parallel jobs (-1 for all cores).
        verbose: Verbosity level (0, 1, or 2).

    Returns:
        Configured FeatureSubsetSearch instance.
    """
    task_type = TaskType.CLASSIFICATION if task == 'classification' else TaskType.REGRESSION
    primary_metric = MetricType.AUC if task == 'classification' else MetricType.IC

    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        task_type=task_type,
    )

    cv_config = CVConfig(
        n_splits=n_folds,
        scheme=CVScheme.EXPANDING,
        gap=1,  # 1 week embargo
        purge_window=0,
    )

    search_config = SearchConfig(
        n_jobs=n_jobs,
    )

    metric_config = MetricConfig(
        primary_metric=primary_metric,
    )

    progress_config = ProgressConfig(
        enable_console_logging=verbose > 0,
        verbose=verbose,
    )

    return FeatureSubsetSearch(
        model_config=model_config,
        cv_config=cv_config,
        search_config=search_config,
        metric_config=metric_config,
        progress_config=progress_config,
    )
