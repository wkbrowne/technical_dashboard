"""
Tests for the trend module.

Tests trend feature computation including moving averages, slopes,
trend alignment, and trend persistence metrics.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.trend import add_trend_features, _ensure_ma


class TestEnsureMa:
    """Tests for the _ensure_ma helper function."""
    
    def test_ensure_ma_existing_column(self, sample_ohlcv_df):
        """Test that existing MA columns are returned as-is."""
        df = sample_ohlcv_df.copy()
        df['ma_20'] = df['adjclose'].rolling(20).mean()
        
        result = _ensure_ma(df, src='adjclose', p=20)
        pd.testing.assert_series_equal(result, df['ma_20'])
    
    def test_ensure_ma_compute_new(self, sample_ohlcv_df):
        """Test computation of new MA when column doesn't exist."""
        result = _ensure_ma(sample_ohlcv_df, src='adjclose', p=20)
        expected = sample_ohlcv_df['adjclose'].rolling(20, min_periods=10).mean()
        
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_ensure_ma_missing_source(self, business_day_index):
        """Test behavior when source column is missing."""
        df = pd.DataFrame(index=business_day_index)
        result = _ensure_ma(df, src='nonexistent', p=20)
        
        # Should return empty series with same index
        assert len(result) == len(business_day_index)
        assert result.isna().all()
    
    def test_ensure_ma_custom_minp(self, sample_ohlcv_df):
        """Test custom minimum periods parameter."""
        result = _ensure_ma(sample_ohlcv_df, src='adjclose', p=20, minp=5)
        expected = sample_ohlcv_df['adjclose'].rolling(20, min_periods=5).mean()
        
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestAddTrendFeatures:
    """Tests for the add_trend_features function."""
    
    def test_basic_trend_features(self, sample_ohlcv_df):
        """Test that basic trend features are added correctly."""
        ma_periods = (10, 20, 50)
        result = add_trend_features(sample_ohlcv_df.copy(), ma_periods=ma_periods)
        
        # Check that MA columns are added
        for p in ma_periods:
            assert f'ma_{p}' in result.columns
            assert f'pct_slope_ma_{p}' in result.columns
            assert f'sign_ma_{p}' in result.columns
        
        # Check aggregate trend features
        expected_agg_features = [
            'trend_score_granular',
            'trend_score_sign', 
            'trend_score_slope',
            'trend_persist_ema',
            'trend_alignment'
        ]
        for feature in expected_agg_features:
            assert feature in result.columns
    
    def test_ma_values_correct(self, sample_ohlcv_df):
        """Test that moving average values are computed correctly."""
        ma_periods = (10, 20)
        result = add_trend_features(sample_ohlcv_df.copy(), ma_periods=ma_periods)
        
        for p in ma_periods:
            expected_ma = sample_ohlcv_df['adjclose'].rolling(p, min_periods=max(5, p//2)).mean()
            pd.testing.assert_series_equal(result[f'ma_{p}'], expected_ma, check_names=False)
    
    def test_slope_calculation(self, sample_ohlcv_df):
        """Test that MA slopes are calculated correctly."""
        ma_periods = (20,)
        slope_window = 10
        result = add_trend_features(
            sample_ohlcv_df.copy(), 
            ma_periods=ma_periods,
            slope_window=slope_window
        )
        
        # Manual slope calculation
        ma_20 = result['ma_20']
        expected_slope = (ma_20 / ma_20.shift(slope_window) - 1.0)
        
        pd.testing.assert_series_equal(
            result['pct_slope_ma_20'], 
            expected_slope.astype('float32'), 
            check_names=False
        )
    
    def test_slope_sign_classification(self, business_day_index):
        """Test slope sign classification with known data."""
        # Create DataFrame with predictable trend
        df = pd.DataFrame(index=business_day_index)
        # Create upward trending price
        df['adjclose'] = pd.Series(range(len(business_day_index)), index=business_day_index, dtype=float)
        
        result = add_trend_features(df, ma_periods=(10,), slope_window=5, eps=1e-5)
        
        # With strong upward trend, most slope signs should be positive
        # (after sufficient warmup period)
        slope_signs = result['sign_ma_10'].dropna()
        positive_fraction = (slope_signs > 0).mean()
        assert positive_fraction > 0.7  # Most should be positive
    
    def test_trend_score_calculation(self, business_day_index):
        """Test trend score calculation with controlled data."""
        df = pd.DataFrame(index=business_day_index)
        # Create price that trends up then down
        n = len(business_day_index)
        prices = list(range(n//2)) + list(range(n//2, 0, -1))
        df['adjclose'] = pd.Series(prices, index=business_day_index, dtype=float)
        
        result = add_trend_features(df, ma_periods=(5, 10), slope_window=3)
        
        # Should have trend score values
        assert 'trend_score_granular' in result.columns
        assert not result['trend_score_granular'].dropna().empty
        
        # Values should be in [-1, 1] range
        trend_scores = result['trend_score_granular'].dropna()
        assert trend_scores.between(-1, 1).all()
    
    def test_trend_alignment_calculation(self, sample_ohlcv_df):
        """Test trend alignment calculation."""
        result = add_trend_features(sample_ohlcv_df.copy(), ma_periods=(10, 20, 50))
        
        # Trend alignment should be in [0, 1] range
        alignment = result['trend_alignment'].dropna()
        assert alignment.between(0, 1).all()
    
    def test_float32_dtypes(self, sample_ohlcv_df):
        """Test that trend features have float32 dtype."""
        ma_periods = (10, 20)
        result = add_trend_features(sample_ohlcv_df.copy(), ma_periods=ma_periods)
        
        float32_features = [
            'pct_slope_ma_10', 'pct_slope_ma_20',
            'sign_ma_10', 'sign_ma_20',
            'trend_score_granular', 'trend_score_sign', 'trend_score_slope',
            'trend_persist_ema', 'trend_alignment'
        ]
        
        for feature in float32_features:
            assert result[feature].dtype == 'float32', f"{feature} should be float32"
    
    def test_empty_ma_periods(self, sample_ohlcv_df):
        """Test behavior with empty ma_periods tuple."""
        result = add_trend_features(sample_ohlcv_df.copy(), ma_periods=())
        
        # Should not crash, and should not add aggregate features
        aggregate_features = [
            'trend_score_granular', 'trend_score_sign', 'trend_score_slope',
            'trend_persist_ema', 'trend_alignment'
        ]
        for feature in aggregate_features:
            assert feature not in result.columns
    
    def test_missing_source_column(self, business_day_index):
        """Test behavior when source column is missing."""
        df = pd.DataFrame({'volume': [1000] * len(business_day_index)}, index=business_day_index)
        
        # Should not crash, but MA columns should be NaN
        result = add_trend_features(df, src_col='nonexistent', ma_periods=(10, 20))
        
        for p in (10, 20):
            assert f'ma_{p}' in result.columns
            assert result[f'ma_{p}'].isna().all()
    
    def test_insufficient_data_handling(self, business_day_index):
        """Test behavior with insufficient data for MA calculation."""
        # Use only first 5 days
        short_index = business_day_index[:5]
        df = pd.DataFrame({
            'adjclose': pd.Series([100, 101, 102, 103, 104], index=short_index)
        })
        
        result = add_trend_features(df, ma_periods=(20,))  # Period longer than data
        
        # Should not crash
        assert 'ma_20' in result.columns
        # Most values should be NaN due to insufficient data
        assert result['ma_20'].isna().sum() >= 3
    
    def test_eps_parameter_effect(self, business_day_index):
        """Test effect of eps parameter on slope sign classification."""
        df = pd.DataFrame(index=business_day_index)
        # Create very small positive trend
        df['adjclose'] = pd.Series([100 + 0.0001*i for i in range(len(business_day_index))], index=business_day_index)
        
        # With large eps, slopes should be classified as zero
        result_large_eps = add_trend_features(df.copy(), ma_periods=(10,), eps=1.0)
        zero_count_large = (result_large_eps['sign_ma_10'] == 0).sum()
        
        # With small eps, slopes should be classified as positive
        result_small_eps = add_trend_features(df.copy(), ma_periods=(10,), eps=1e-10)
        pos_count_small = (result_small_eps['sign_ma_10'] > 0).sum()
        
        assert zero_count_large > pos_count_small