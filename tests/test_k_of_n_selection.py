"""Tests for K-of-N feature selection within groups.

These tests validate that the K-of-N baseline selection loop completes
for all groups without silent exit or crash. This addresses an issue where
the pipeline would randomly exit after processing 11/13 baseline groups.

Tests cover:
1. K-of-N selection for all baseline groups completes
2. Parallel evaluation doesn't cause worker crashes
3. Sequential fallback works correctly
4. Edge cases like empty groups, single features
"""

import gc
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List

from src.feature_selection.config import (
    CVConfig, CVScheme, MetricConfig, MetricType,
    ModelConfig, ModelType, SearchConfig, TaskType,
)
from src.feature_selection.evaluation import SubsetEvaluator, EvaluationCache
from src.feature_selection.group_selection import (
    GroupSelectionConfig,
    grouped_forward_selection,
    _select_k_features_for_group,
)


def make_search_config():
    """Create a default SearchConfig for tests."""
    return SearchConfig()


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def synthetic_data_13_groups():
    """Create synthetic data with 13 groups (matching production baseline count).

    This fixture creates data that mimics the production scenario where
    K-of-N baseline selection was silently exiting after 11/13 groups.
    """
    np.random.seed(42)
    n_samples = 200  # Small for fast tests
    n_groups = 13
    features_per_group = 3  # Reduced for speed

    # Create date index for time-series CV
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')

    # Hidden signal for target
    hidden_signal = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 0.3
    target_prob = 1 / (1 + np.exp(-hidden_signal))
    y = pd.Series((np.random.rand(n_samples) < target_prob).astype(int), index=dates)

    # Create feature groups
    feature_data = {'date': dates}
    groups = {}

    group_names = [
        'price_position', 'trend_strength', 'macro_intermarket', 'gap_dynamics',
        'range_breakout', 'microstructure_position', 'momentum_quality',
        'volatility_state', 'volatility_regime', 'volume_shock', 'alpha_momentum',
        'sector_relative', 'weekly_trend',
    ]

    for i, group_name in enumerate(group_names):
        group_features = []
        for j in range(features_per_group):
            feat_name = f"{group_name}_f{j}"
            # Features have varying correlation with target
            correlation = 0.1 + 0.05 * (i % 3) + np.random.rand() * 0.1
            feature_data[feat_name] = hidden_signal * correlation + np.random.randn(n_samples) * 0.5
            group_features.append(feat_name)
        groups[group_name] = group_features

    X = pd.DataFrame(feature_data).set_index('date')

    return X, y, groups


@pytest.fixture
def model_config():
    """LightGBM model configuration for fast tests."""
    return ModelConfig(
        model_type=ModelType.LIGHTGBM,
        task_type=TaskType.CLASSIFICATION,
        params={
            'learning_rate': 0.1,
            'n_estimators': 10,
            'max_depth': 2,
            'min_child_samples': 5,
            'verbosity': -1,
        },
        num_threads=1,
        early_stopping_rounds=5,
        num_boost_round=10,
    )


@pytest.fixture
def cv_config():
    """CV configuration with minimal folds for speed."""
    return CVConfig(
        n_splits=2,  # Minimum for CV
        scheme=CVScheme.EXPANDING,
        gap=2,
        purge_window=0,
        min_train_samples=20,
    )


@pytest.fixture
def metric_config():
    """Metric configuration."""
    return MetricConfig(
        primary_metric=MetricType.AUC,
        secondary_metrics=[],
    )


@pytest.fixture
def search_config():
    """Search configuration."""
    return SearchConfig()


@pytest.fixture
def evaluator(synthetic_data_13_groups, model_config, cv_config, metric_config, search_config):
    """Create SubsetEvaluator with synthetic data."""
    X, y, _ = synthetic_data_13_groups
    return SubsetEvaluator(
        X=X,
        y=y,
        model_config=model_config,
        cv_config=cv_config,
        metric_config=metric_config,
        search_config=search_config,
    )


# =============================================================================
# Tests for K-of-N Baseline Selection Completion
# =============================================================================

class TestKOfNBaselineCompletion:
    """Tests that K-of-N baseline selection completes for ALL groups."""

    def test_all_13_baseline_groups_processed(self, evaluator, synthetic_data_13_groups):
        """K-of-N selection should complete for all 13 baseline groups without exit."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            epsilon_add=0.0001,
            epsilon_add_feature=0.0001,
            n_jobs=1,  # Sequential to avoid parallel issues
            parallelize_moves=False,
            verbose=False,
            max_groups=20,
            k_of_n_seed=42,
        )

        # Run forward selection with all 13 groups as baseline
        selected, results, final_metric = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},  # No additional candidates
            config=config,
            cache=None,
        )

        # Verify all groups were processed
        assert len(selected) == 13, f"Expected 13 groups, got {len(selected)}"
        assert set(selected.keys()) == set(groups.keys())

        # Verify each group has at least one feature selected
        for group_name, features in selected.items():
            assert len(features) >= 1, f"Group {group_name} has no features"
            assert len(features) <= config.group_k_default + 1, \
                f"Group {group_name} has too many features: {len(features)}"

    def test_parallel_k_of_n_completes(self, evaluator, synthetic_data_13_groups):
        """K-of-N with parallel evaluation should complete without worker crashes."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            epsilon_add=0.0001,
            epsilon_add_feature=0.0001,
            n_jobs=2,  # Parallel
            parallelize_moves=True,
            n_move_workers=2,
            verbose=False,
            max_groups=20,
            k_of_n_seed=42,
        )

        # Run forward selection with parallel K-of-N
        selected, results, final_metric = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
            cache=None,
        )

        # Verify all groups were processed
        assert len(selected) == 13, f"Expected 13 groups, got {len(selected)}"

    def test_k_of_n_with_cache_enabled(self, evaluator, synthetic_data_13_groups):
        """K-of-N with caching should complete for all groups."""
        X, y, groups = synthetic_data_13_groups

        cache = EvaluationCache(max_size=1000, thread_safe=True)

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            epsilon_add=0.0001,
            epsilon_add_feature=0.0001,
            enable_caching=True,
            n_jobs=1,
            parallelize_moves=False,
            verbose=False,
            max_groups=20,
        )

        selected, results, final_metric = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
            cache=cache,
        )

        assert len(selected) == 13
        # Check cache was populated (internal _cache dict has entries)
        assert len(cache._cache) > 0, "Cache should have entries"

    def test_k_of_n_disabled_uses_all_features(self, evaluator, synthetic_data_13_groups):
        """With K-of-N disabled, all features in each group should be used."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=False,  # Disabled
            epsilon_add=0.0001,
            n_jobs=1,
            parallelize_moves=False,
            verbose=False,
            max_groups=20,
        )

        selected, results, final_metric = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
            cache=None,
        )

        # All features should be included when K-of-N is disabled
        for group_name, original_features in groups.items():
            selected_features = selected.get(group_name, [])
            assert len(selected_features) == len(original_features), \
                f"Group {group_name}: expected {len(original_features)}, got {len(selected_features)}"


# =============================================================================
# Tests for Individual K-of-N Selection
# =============================================================================

class TestSelectKFeaturesForGroup:
    """Tests for the _select_k_features_for_group function."""

    def test_selects_up_to_k_features(self, evaluator, synthetic_data_13_groups):
        """Should select at most K features from a group."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            epsilon_add_feature=0.0001,
            n_jobs=1,
            parallelize_moves=False,
        )

        group_name = 'price_position'
        group_features = groups[group_name]

        selected, metric = _select_k_features_for_group(
            evaluator=evaluator,
            current_features=[],
            group_name=group_name,
            group_features=group_features,
            config=config,
            cache=None,
            force_sequential=True,
        )

        assert len(selected) <= config.group_k_default
        assert len(selected) >= 1  # At least one feature
        assert all(f in group_features for f in selected)

    def test_handles_empty_group(self, evaluator, synthetic_data_13_groups):
        """Should handle empty feature group gracefully."""
        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            n_jobs=1,
        )

        selected, metric = _select_k_features_for_group(
            evaluator=evaluator,
            current_features=[],
            group_name='empty_group',
            group_features=[],
            config=config,
            cache=None,
            force_sequential=True,
        )

        assert selected == []

    def test_handles_nonexistent_features(self, evaluator, synthetic_data_13_groups):
        """Should filter out features not in evaluator's data."""
        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            n_jobs=1,
        )

        selected, metric = _select_k_features_for_group(
            evaluator=evaluator,
            current_features=[],
            group_name='fake_group',
            group_features=['nonexistent_feature_1', 'nonexistent_feature_2'],
            config=config,
            cache=None,
            force_sequential=True,
        )

        assert selected == []

    def test_single_feature_group(self, evaluator, synthetic_data_13_groups):
        """Group with single feature should return that feature."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            epsilon_add_feature=0.0001,
            n_jobs=1,
        )

        # Use first feature from first group
        single_feature = groups['price_position'][:1]

        selected, metric = _select_k_features_for_group(
            evaluator=evaluator,
            current_features=[],
            group_name='single_feature_group',
            group_features=single_feature,
            config=config,
            cache=None,
            force_sequential=True,
        )

        assert selected == single_feature


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestKOfNEdgeCases:
    """Edge cases and boundary conditions for K-of-N selection."""

    def test_k_greater_than_group_size(self, evaluator, synthetic_data_13_groups):
        """K > group size should return all available features."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=10,  # K > features_per_group (4)
            epsilon_add_feature=0.0001,
            n_jobs=1,
        )

        group_name = 'price_position'
        group_features = groups[group_name]

        selected, metric = _select_k_features_for_group(
            evaluator=evaluator,
            current_features=[],
            group_name=group_name,
            group_features=group_features,
            config=config,
            cache=None,
            force_sequential=True,
        )

        # Should not exceed available features
        assert len(selected) <= len(group_features)

    def test_group_specific_k_override(self, evaluator, synthetic_data_13_groups):
        """Per-group K overrides should be respected."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            group_k={'price_position': 3},  # Override for this group
            epsilon_add_feature=0.0001,
            n_jobs=1,
        )

        selected, metric = _select_k_features_for_group(
            evaluator=evaluator,
            current_features=[],
            group_name='price_position',
            group_features=groups['price_position'],
            config=config,
            cache=None,
            force_sequential=True,
        )

        # Should respect the override K=3
        assert len(selected) <= 3

    def test_deterministic_with_same_seed(self, evaluator, synthetic_data_13_groups):
        """K-of-N selection should be deterministic with same seed."""
        X, y, groups = synthetic_data_13_groups

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            epsilon_add=0.0001,
            epsilon_add_feature=0.0001,
            n_jobs=1,
            parallelize_moves=False,
            verbose=False,
            max_groups=20,
            k_of_n_seed=42,
        )

        # Run twice
        selected1, _, _ = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
            cache=None,
        )

        selected2, _, _ = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
            cache=None,
        )

        # Results should be identical
        assert selected1.keys() == selected2.keys()
        for group_name in selected1:
            assert selected1[group_name] == selected2[group_name], \
                f"Group {group_name} differs between runs"


# =============================================================================
# Stress Tests
# =============================================================================

class TestKOfNStress:
    """Stress tests to catch silent exits and crashes."""

    def test_large_number_of_groups(self):
        """Test with 20+ groups to catch scaling issues."""
        np.random.seed(42)
        n_samples = 300
        n_groups = 25

        dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')
        hidden_signal = np.random.randn(n_samples)
        y = pd.Series((hidden_signal > 0).astype(int), index=dates)

        feature_data = {'date': dates}
        groups = {}

        for i in range(n_groups):
            group_name = f"group_{i:02d}"
            features = []
            for j in range(3):
                feat_name = f"{group_name}_f{j}"
                feature_data[feat_name] = np.random.randn(n_samples)
                features.append(feat_name)
            groups[group_name] = features

        X = pd.DataFrame(feature_data).set_index('date')

        model_config = ModelConfig(
            model_type=ModelType.LIGHTGBM,
            task_type=TaskType.CLASSIFICATION,
            params={'n_estimators': 10, 'max_depth': 2, 'verbosity': -1},
            num_threads=1,
        )
        cv_config = CVConfig(n_splits=2, gap=2, min_train_samples=30)
        metric_config = MetricConfig(primary_metric=MetricType.AUC)
        search_config = make_search_config()

        evaluator = SubsetEvaluator(X, y, model_config, cv_config, metric_config, search_config)

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            n_jobs=1,
            parallelize_moves=False,
            verbose=False,
            max_groups=30,
        )

        selected, _, _ = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
        )

        assert len(selected) == n_groups, f"Expected {n_groups} groups, got {len(selected)}"

    def test_groups_with_varying_sizes(self):
        """Test groups with 1 to 10 features each."""
        np.random.seed(42)
        n_samples = 300

        dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')
        hidden_signal = np.random.randn(n_samples)
        y = pd.Series((hidden_signal > 0).astype(int), index=dates)

        feature_data = {'date': dates}
        groups = {}

        # Create groups with varying sizes
        for i, size in enumerate([1, 2, 3, 4, 5, 7, 10]):
            group_name = f"group_size_{size}"
            features = []
            for j in range(size):
                feat_name = f"{group_name}_f{j}"
                feature_data[feat_name] = np.random.randn(n_samples)
                features.append(feat_name)
            groups[group_name] = features

        X = pd.DataFrame(feature_data).set_index('date')

        model_config = ModelConfig(
            model_type=ModelType.LIGHTGBM,
            task_type=TaskType.CLASSIFICATION,
            params={'n_estimators': 10, 'max_depth': 2, 'verbosity': -1},
            num_threads=1,
        )
        cv_config = CVConfig(n_splits=2, gap=2, min_train_samples=30)
        metric_config = MetricConfig(primary_metric=MetricType.AUC)
        search_config = make_search_config()

        evaluator = SubsetEvaluator(X, y, model_config, cv_config, metric_config, search_config)

        config = GroupSelectionConfig(
            enable_k_of_n=True,
            group_k_default=2,
            n_jobs=1,
            verbose=False,
        )

        selected, _, _ = grouped_forward_selection(
            evaluator=evaluator,
            baseline_groups=groups,
            candidate_groups={},
            config=config,
        )

        # All groups should be processed
        assert len(selected) == len(groups)

        # Each group should have appropriate number of features
        for group_name, selected_features in selected.items():
            original_size = len(groups[group_name])
            expected_max = min(config.group_k_default, original_size)
            assert 1 <= len(selected_features) <= expected_max + 1, \
                f"Group {group_name} ({original_size} features) selected {len(selected_features)}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
