"""
Tests for multi-model feature selection.

These tests validate:
1. Data loading and alignment with panel data (multiple symbols per date)
2. Model-specific target extraction (hit_long_normal, etc.)
3. Row alignment using _row_id (not date index)
4. Sample weight alignment
5. run_single_model_selection function
"""
import gc
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config.model_keys import ModelKey
from src.feature_selection.config import (
    CVConfig, CVScheme, MetricConfig, MetricType,
    ModelConfig, ModelType, SearchConfig, TaskType,
)
from src.feature_selection.pipeline import LooseTightConfig
from src.feature_selection.multimodel import (
    run_single_model_selection,
    compute_overlap_analysis,
    FeatureSelectionResult,
)


# =============================================================================
# Mock Panel Data Generation (multiple symbols per date)
# =============================================================================

def generate_panel_data(
    n_symbols: int = 10,
    n_days: int = 200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate panel data with multiple symbols per date.

    This simulates real data where each date has multiple rows (one per symbol).

    Returns:
        Tuple of (X features, targets_df, sample_weight)
    """
    np.random.seed(seed)

    # Create date range
    dates = pd.bdate_range('2020-01-01', periods=n_days, freq='B')
    symbols = [f'SYM{i:02d}' for i in range(n_symbols)]

    # Create panel index (date x symbol)
    rows = []
    for date in dates:
        for sym in symbols:
            rows.append({'date': date, 'symbol': sym})

    panel = pd.DataFrame(rows)
    n_rows = len(panel)

    # Add _row_id for unique identification
    panel['_row_id'] = np.arange(n_rows)

    # Generate features
    # f_informative: correlated with target
    hidden_signal = np.sin(np.linspace(0, 8*np.pi, n_rows)) + np.random.randn(n_rows) * 0.3
    panel['f_informative'] = hidden_signal * 0.7 + np.random.randn(n_rows) * 0.3

    # f_noise: pure noise
    panel['f_noise_1'] = np.random.randn(n_rows)
    panel['f_noise_2'] = np.random.randn(n_rows)

    # f_symbol_specific: varies by symbol
    symbol_effects = {sym: np.random.randn() for sym in symbols}
    panel['f_symbol_effect'] = panel['symbol'].map(symbol_effects)

    # Generate model-specific hit columns (like triple barrier output)
    # hit_long_normal: 1 (upper barrier), -1 (lower barrier), 0 (timeout)
    target_prob = 1 / (1 + np.exp(-hidden_signal))
    hit_values = np.random.choice([-1, 0, 1], size=n_rows, p=[0.3, 0.3, 0.4])
    # Bias toward 1 when hidden signal is high
    hit_values = np.where(
        (np.random.rand(n_rows) < target_prob) & (hit_values == 0),
        1, hit_values
    )

    panel['hit_long_normal'] = hit_values
    panel['hit_long_parabolic'] = np.random.choice([-1, 0, 1], size=n_rows, p=[0.35, 0.35, 0.3])
    panel['hit_short_normal'] = np.random.choice([-1, 0, 1], size=n_rows, p=[0.35, 0.3, 0.35])
    panel['hit_short_parabolic'] = np.random.choice([-1, 0, 1], size=n_rows, p=[0.3, 0.35, 0.35])

    # Sample weights (simulate overlap weights)
    panel['weight_final'] = np.random.uniform(0.5, 1.0, n_rows)

    # Build X with date index (like real pipeline)
    feature_cols = ['f_informative', 'f_noise_1', 'f_noise_2', 'f_symbol_effect']
    X = panel[feature_cols + ['_row_id']].copy()
    X.index = pd.Index(panel['date'], name='date')

    # Build targets_df
    target_cols = ['symbol', 'date', '_row_id',
                   'hit_long_normal', 'hit_long_parabolic',
                   'hit_short_normal', 'hit_short_parabolic',
                   'weight_final']
    targets_df = panel[target_cols].copy()

    # Build sample_weight indexed by _row_id
    sample_weight = panel['weight_final'].copy()
    sample_weight.index = panel['_row_id']

    return X, targets_df, sample_weight


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope='module')
def panel_data():
    """Generate panel data once for all tests."""
    return generate_panel_data(n_symbols=10, n_days=150, seed=42)


@pytest.fixture(scope='module')
def test_configs():
    """Get test configurations for fast execution."""
    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        task_type=TaskType.CLASSIFICATION,
        params={
            'learning_rate': 0.1,
            'max_depth': 4,
            'num_leaves': 15,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': -1,
        },
        num_threads=1,
        early_stopping_rounds=10,
        num_boost_round=30,
    )

    cv_config = CVConfig(
        n_splits=3,
        scheme=CVScheme.EXPANDING,
        gap=0,
        purge_window=0,
        min_train_samples=50,
    )

    pipeline_config = LooseTightConfig(
        run_base_elimination=False,
        run_interactions=False,
        epsilon_add_loose=0.001,
        max_features_loose=4,
        epsilon_remove_strict=0.0,
        epsilon_swap=0.001,
        max_swap_iterations=2,
        n_jobs=2,
    )

    metric_config = MetricConfig(
        primary_metric=MetricType.AUC,
        secondary_metrics=[],
    )

    return model_config, cv_config, pipeline_config, metric_config


# =============================================================================
# Test Classes
# =============================================================================

class TestPanelDataGeneration:
    """Tests for panel data structure."""

    def test_panel_has_duplicate_dates(self, panel_data):
        """Panel data should have multiple rows per date (non-unique index)."""
        X, targets_df, sample_weight = panel_data

        # X has date index
        assert X.index.name == 'date'

        # Dates are NOT unique (multiple symbols per date)
        assert not X.index.is_unique, "Panel data should have duplicate dates"

        # Check we have multiple symbols per date
        date_counts = X.index.value_counts()
        assert date_counts.max() > 1, "Should have multiple rows per date"

    def test_row_id_is_unique(self, panel_data):
        """_row_id should be unique for each row."""
        X, targets_df, sample_weight = panel_data

        assert '_row_id' in X.columns
        assert X['_row_id'].is_unique, "_row_id must be unique"

        # targets_df also has _row_id
        assert '_row_id' in targets_df.columns
        assert targets_df['_row_id'].is_unique

    def test_sample_weight_indexed_by_row_id(self, panel_data):
        """sample_weight should be indexed by _row_id."""
        X, targets_df, sample_weight = panel_data

        # sample_weight index matches _row_id values
        assert set(sample_weight.index) == set(X['_row_id'].values)

    def test_model_specific_hit_columns_exist(self, panel_data):
        """targets_df should have model-specific hit columns."""
        X, targets_df, sample_weight = panel_data

        expected_cols = [
            'hit_long_normal', 'hit_long_parabolic',
            'hit_short_normal', 'hit_short_parabolic',
        ]
        for col in expected_cols:
            assert col in targets_df.columns, f"Missing column: {col}"

        # Each hit column should have -1, 0, 1 values
        for col in expected_cols:
            unique_vals = set(targets_df[col].unique())
            assert unique_vals <= {-1, 0, 1}, f"{col} has unexpected values: {unique_vals}"


class TestModelLabelExtraction:
    """Tests for extracting model-specific labels."""

    def test_get_model_labels_long_normal(self, panel_data):
        """Should extract correct labels for LONG_NORMAL model."""
        X, targets_df, sample_weight = panel_data

        # Import from the script (simulating what it does)
        # We'll inline the logic here for testing
        model_key = ModelKey.LONG_NORMAL
        label_col = 'hit_long_normal'

        df = targets_df.copy()
        # Binary: exclude timeout (0)
        df = df[df[label_col] != 0].copy()
        # Long: upper barrier hit (1) = success
        df['target'] = (df[label_col] == 1).astype(int)

        y = df['target']
        row_ids = df['_row_id'].values

        # Verify we filtered correctly
        assert len(y) < len(targets_df), "Should exclude timeout samples"
        assert set(y.unique()) == {0, 1}, "Binary labels should be 0 or 1"

        # row_ids should be a subset
        assert set(row_ids) <= set(targets_df['_row_id'].values)

    def test_get_model_labels_short_normal(self, panel_data):
        """Should extract correct labels for SHORT_NORMAL model."""
        X, targets_df, sample_weight = panel_data

        model_key = ModelKey.SHORT_NORMAL
        label_col = 'hit_short_normal'

        df = targets_df.copy()
        df = df[df[label_col] != 0].copy()
        # Short: lower barrier hit (-1) = success
        df['target'] = (df[label_col] == -1).astype(int)

        y = df['target']

        assert len(y) < len(targets_df)
        assert set(y.unique()) == {0, 1}


class TestRowAlignment:
    """Tests for row alignment with non-unique indices."""

    def test_alignment_using_row_id_not_loc(self, panel_data):
        """Should align using _row_id.isin(), not .loc[date_index]."""
        X, targets_df, sample_weight = panel_data

        # Simulate what the script does: filter by model
        label_col = 'hit_long_normal'
        df = targets_df[targets_df[label_col] != 0].copy()
        row_ids = df['_row_id'].values

        # CORRECT: Use _row_id.isin()
        X_aligned = X[X['_row_id'].isin(row_ids)].drop(columns=['_row_id']).copy()

        # Verify alignment worked
        assert len(X_aligned) == len(df), (
            f"X_aligned ({len(X_aligned)}) should match filtered df ({len(df)})"
        )

        # WRONG approach would use .loc[date_index] - this causes explosion
        # We verify this would NOT work correctly
        y_index = pd.Index(df['date'], name='date')
        try:
            X_wrong = X.loc[y_index]
            # If this succeeds, it means cross-join happened (too many rows)
            if len(X_wrong) > len(df) * 1.5:
                pytest.skip("loc[] caused expected cross-join explosion")
        except KeyError:
            # Expected: .loc[] may fail with misaligned indices
            pass

    def test_sample_weight_alignment_by_row_id(self, panel_data):
        """Sample weight should be aligned using row_id, not date index."""
        X, targets_df, sample_weight = panel_data

        # Filter for a model
        label_col = 'hit_long_normal'
        df = targets_df[targets_df[label_col] != 0].copy()
        row_ids = df['_row_id'].values

        # Align sample weight using row_ids
        sample_weight_aligned = sample_weight.loc[row_ids]

        # Should have same length as filtered data
        assert len(sample_weight_aligned) == len(df)

        # Values should match
        expected_weights = df['weight_final'].values
        np.testing.assert_array_almost_equal(
            sample_weight_aligned.values, expected_weights
        )


class TestRunSingleModelSelection:
    """Tests for run_single_model_selection function."""

    def test_accepts_prealigned_data(self, panel_data, test_configs):
        """run_single_model_selection should work with pre-aligned data."""
        import src.feature_selection.base_features as bf
        original_base = bf.BASE_FEATURES
        original_expansion = bf.EXPANSION_CANDIDATES

        try:
            # Mock base features for the pipeline
            bf.BASE_FEATURES = ['f_informative']
            bf.EXPANSION_CANDIDATES = {'test': ['f_noise_1', 'f_noise_2', 'f_symbol_effect']}

            X, targets_df, sample_weight = panel_data
            model_config, cv_config, pipeline_config, metric_config = test_configs

            # Pre-align data (like the script does)
            label_col = 'hit_long_normal'
            df = targets_df[targets_df[label_col] != 0].copy()
            df['target'] = (df[label_col] == 1).astype(int)
            row_ids = df['_row_id'].values

            # Create aligned X (drop _row_id)
            X_aligned = X[X['_row_id'].isin(row_ids)].drop(columns=['_row_id']).copy()

            # Create aligned y with date index (for CV)
            y = df['target'].copy()
            y.index = pd.Index(df['date'], name='date')

            # Pre-aligned sample weight (same length as X_aligned)
            sw_aligned = sample_weight.loc[row_ids].reset_index(drop=True)

            # Run selection
            feature_cols = ['f_informative', 'f_noise_1', 'f_noise_2']
            result = run_single_model_selection(
                X=X_aligned,
                y=y,
                model_key=ModelKey.LONG_NORMAL,
                features=feature_cols,
                sample_weight=sw_aligned,
                cv_config=cv_config,
                model_config=model_config,
                metric_config=metric_config,
                pipeline_config=pipeline_config,
                verbose=False,
            )

            # Verify result
            assert result is not None
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) > 0
            # AUC > 0 is sufficient - we're testing the function runs, not model quality
            assert result.cv_auc_mean > 0, "AUC should be computed"

        finally:
            bf.BASE_FEATURES = original_base
            bf.EXPANSION_CANDIDATES = original_expansion

    def test_handles_sample_weight_same_length(self, panel_data, test_configs):
        """Should handle sample_weight with same length as X (pre-aligned)."""
        import src.feature_selection.base_features as bf
        original_base = bf.BASE_FEATURES
        original_expansion = bf.EXPANSION_CANDIDATES

        try:
            bf.BASE_FEATURES = ['f_informative']
            bf.EXPANSION_CANDIDATES = {'test': ['f_noise_1']}

            X, targets_df, sample_weight = panel_data
            model_config, cv_config, pipeline_config, metric_config = test_configs

            # Use a small subset for speed
            X_small = X.head(500).drop(columns=['_row_id']).copy()
            y = pd.Series(
                np.random.randint(0, 2, len(X_small)),
                index=X_small.index,
                name='target'
            )

            # Sample weight with same length, but different index
            sw = pd.Series(np.ones(len(X_small)))  # Simple index 0, 1, 2...

            feature_cols = ['f_informative', 'f_noise_1']
            result = run_single_model_selection(
                X=X_small,
                y=y,
                model_key=ModelKey.LONG_NORMAL,
                features=feature_cols,
                sample_weight=sw,
                cv_config=cv_config,
                model_config=model_config,
                metric_config=metric_config,
                pipeline_config=pipeline_config,
                verbose=False,
            )

            # Should complete without error
            assert result is not None
            assert result.cv_auc_mean > 0

        finally:
            bf.BASE_FEATURES = original_base
            bf.EXPANSION_CANDIDATES = original_expansion


class TestOverlapAnalysis:
    """Tests for compute_overlap_analysis."""

    def test_overlap_with_identical_results(self):
        """Identical feature sets should have overlap of 1.0."""
        features = ['a', 'b', 'c']
        results = {
            'model1': FeatureSelectionResult(
                model_key='model1',
                selected_features=features,
                n_features=3,
                cv_auc_mean=0.7,
                cv_auc_std=0.01,
                fold_metrics=[0.7, 0.7, 0.7],
            ),
            'model2': FeatureSelectionResult(
                model_key='model2',
                selected_features=features,  # Same features
                n_features=3,
                cv_auc_mean=0.7,
                cv_auc_std=0.01,
                fold_metrics=[0.7, 0.7, 0.7],
            ),
        }

        summary = compute_overlap_analysis(results)

        # Jaccard similarity should be 1.0
        assert summary.overlap_matrix['model1']['model2'] == 1.0
        assert summary.overlap_matrix['model2']['model1'] == 1.0

    def test_overlap_with_disjoint_results(self):
        """Disjoint feature sets should have overlap of 0.0."""
        results = {
            'model1': FeatureSelectionResult(
                model_key='model1',
                selected_features=['a', 'b'],
                n_features=2,
                cv_auc_mean=0.7,
                cv_auc_std=0.01,
                fold_metrics=[0.7, 0.7, 0.7],
            ),
            'model2': FeatureSelectionResult(
                model_key='model2',
                selected_features=['c', 'd'],  # Different features
                n_features=2,
                cv_auc_mean=0.7,
                cv_auc_std=0.01,
                fold_metrics=[0.7, 0.7, 0.7],
            ),
        }

        summary = compute_overlap_analysis(results)

        # Jaccard similarity should be 0.0
        assert summary.overlap_matrix['model1']['model2'] == 0.0

    def test_core_features_intersection(self):
        """Core features should be intersection of all models."""
        results = {
            'model1': FeatureSelectionResult(
                model_key='model1',
                selected_features=['a', 'b', 'c'],
                n_features=3,
                cv_auc_mean=0.7,
                cv_auc_std=0.01,
                fold_metrics=[0.7, 0.7, 0.7],
            ),
            'model2': FeatureSelectionResult(
                model_key='model2',
                selected_features=['a', 'b', 'd'],
                n_features=3,
                cv_auc_mean=0.7,
                cv_auc_std=0.01,
                fold_metrics=[0.7, 0.7, 0.7],
            ),
        }

        summary = compute_overlap_analysis(results)

        # Core should be intersection: {'a', 'b'}
        assert set(summary.core_features) == {'a', 'b'}

        # Head features should be model-specific
        assert set(summary.head_features['model1']) == {'c'}
        assert set(summary.head_features['model2']) == {'d'}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sample_weight(self, panel_data, test_configs):
        """Should handle None sample_weight."""
        import src.feature_selection.base_features as bf
        original_base = bf.BASE_FEATURES
        original_expansion = bf.EXPANSION_CANDIDATES

        try:
            bf.BASE_FEATURES = ['f_informative']
            bf.EXPANSION_CANDIDATES = {'test': ['f_noise_1']}

            X, targets_df, _ = panel_data
            model_config, cv_config, pipeline_config, metric_config = test_configs

            # Small subset
            X_small = X.head(500).drop(columns=['_row_id']).copy()
            y = pd.Series(
                np.random.randint(0, 2, len(X_small)),
                index=X_small.index
            )

            feature_cols = ['f_informative', 'f_noise_1']
            result = run_single_model_selection(
                X=X_small,
                y=y,
                model_key=ModelKey.LONG_NORMAL,
                features=feature_cols,
                sample_weight=None,  # No weights
                cv_config=cv_config,
                model_config=model_config,
                metric_config=metric_config,
                pipeline_config=pipeline_config,
                verbose=False,
            )

            assert result is not None
            assert result.cv_auc_mean > 0

        finally:
            bf.BASE_FEATURES = original_base
            bf.EXPANSION_CANDIDATES = original_expansion

    def test_misaligned_sample_weight_length(self, panel_data, test_configs):
        """Should handle sample_weight with different length gracefully."""
        import src.feature_selection.base_features as bf
        original_base = bf.BASE_FEATURES
        original_expansion = bf.EXPANSION_CANDIDATES

        try:
            bf.BASE_FEATURES = ['f_informative']
            bf.EXPANSION_CANDIDATES = {'test': ['f_noise_1']}

            X, targets_df, _ = panel_data
            model_config, cv_config, pipeline_config, metric_config = test_configs

            # Small subset
            X_small = X.head(500).drop(columns=['_row_id']).copy()
            y = pd.Series(
                np.random.randint(0, 2, len(X_small)),
                index=X_small.index
            )

            # Sample weight with WRONG length (shorter)
            sw_wrong = pd.Series(np.ones(len(X_small) - 100))

            feature_cols = ['f_informative', 'f_noise_1']

            # This should not crash - the function handles misalignment
            result = run_single_model_selection(
                X=X_small,
                y=y,
                model_key=ModelKey.LONG_NORMAL,
                features=feature_cols,
                sample_weight=sw_wrong,
                cv_config=cv_config,
                model_config=model_config,
                metric_config=metric_config,
                pipeline_config=pipeline_config,
                verbose=False,
            )

            # Should still produce a result (fallback handling)
            assert result is not None

        finally:
            bf.BASE_FEATURES = original_base
            bf.EXPANSION_CANDIDATES = original_expansion


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
