"""
Tests for the hurst module.

Tests Hurst exponent computation for measuring long-term memory in time series.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.hurst import add_hurst_features, _safe_hurst_rs


class TestSafeHurstRs:
    """Tests for the _safe_hurst_rs helper function."""
    
    def test_safe_hurst_basic_calculation(self):
        """Test basic Hurst calculation with sufficient data."""
        np.random.seed(42)
        # Generate random walk (should have H ≈ 0.5)
        data = np.random.normal(0, 1, 100).cumsum()
        
        result = _safe_hurst_rs(data)
        
        # Should return a float value
        assert isinstance(result, float)
        assert not np.isnan(result)
        
        # For random walk, Hurst should be around 0.5
        assert 0.2 < result < 0.8  # Reasonable range allowing for sampling variation
    
    def test_safe_hurst_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Very short series
        short_data = np.array([1, 2, 3])
        result = _safe_hurst_rs(short_data)
        
        # Should return NaN
        assert np.isnan(result)
    
    def test_safe_hurst_with_nans(self):
        """Test behavior with NaN values in input."""
        data = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        result = _safe_hurst_rs(data)
        
        # Should handle NaNs gracefully (either return valid result or NaN)
        assert isinstance(result, float)  # Should not crash
    
    def test_safe_hurst_all_nans(self):
        """Test behavior with all NaN input."""
        data = np.array([np.nan, np.nan, np.nan])
        result = _safe_hurst_rs(data)
        
        # Should return NaN
        assert np.isnan(result)
    
    def test_safe_hurst_constant_series(self):
        """Test behavior with constant values."""
        data = np.array([5.0] * 50)
        result = _safe_hurst_rs(data)
        
        # Should handle constant series (may return NaN or a specific value)
        assert isinstance(result, float)
    
    def test_safe_hurst_trending_series(self):
        """Test with clearly trending data."""
        # Strong linear trend should give H > 0.5
        data = np.arange(100, dtype=float)
        result = _safe_hurst_rs(data)
        
        if not np.isnan(result):
            # Trending series should have H > 0.5
            assert result > 0.5
    
    def test_safe_hurst_exception_handling(self, monkeypatch):
        """Test that exceptions are caught and return NaN."""
        def mock_hurst_rs(*args, **kwargs):
            raise ValueError("Mock error")
        
        # Mock the nolds.hurst_rs function to raise an exception
        import src.features.hurst
        monkeypatch.setattr(src.features.hurst.nolds, 'hurst_rs', mock_hurst_rs)
        
        data = np.random.normal(0, 1, 50)
        result = _safe_hurst_rs(data)
        
        # Should return NaN when calculation fails
        assert np.isnan(result)


class TestAddHurstFeatures:
    """Tests for the add_hurst_features function."""
    
    def test_missing_return_column(self, sample_ohlcv_df):
        """Test behavior when return column is missing."""
        df = sample_ohlcv_df.copy()
        del df['ret']  # Remove returns column
        
        result = add_hurst_features(df, ret_col='ret')
        
        # Should return DataFrame unchanged (no new columns added)
        pd.testing.assert_frame_equal(result, df)
    
    def test_basic_hurst_features(self, sample_ohlcv_df):
        """Test that basic Hurst features are added."""
        windows = (64, 128)
        result = add_hurst_features(
            sample_ohlcv_df.copy(),
            ret_col='ret',
            windows=windows,
            ema_halflife=5
        )
        
        # Check that Hurst columns are added
        for w in windows:
            assert f'hurst_ret_{w}' in result.columns
        
        # Check EMA smoothed version for first window
        assert f'hurst_ret_{windows[0]}_emaHL5' in result.columns
    
    def test_hurst_calculation_with_sufficient_data(self, business_day_index):
        """Test Hurst calculation with known data patterns."""
        # Create DataFrame with enough data for Hurst calculation
        n_days = 150  # More than minimum required
        df = pd.DataFrame(index=business_day_index[:n_days])
        
        # Create trending return series
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, n_days)  # Slight positive drift
        df['ret'] = pd.Series(returns, index=business_day_index[:n_days])
        
        result = add_hurst_features(df, windows=(64,))
        
        # Should have valid Hurst values for some observations
        hurst_values = result['hurst_ret_64'].dropna()
        assert len(hurst_values) > 0  # Should have some non-NaN values
        
        # Hurst values should be in reasonable range
        valid_hurst = hurst_values[~np.isinf(hurst_values)]
        if len(valid_hurst) > 0:
            assert (valid_hurst >= 0).all() and (valid_hurst <= 1).all()
    
    def test_insufficient_data_for_hurst(self, business_day_index):
        """Test behavior with insufficient data for Hurst calculation."""
        # Very short time series
        short_df = pd.DataFrame({
            'ret': pd.Series([0.01, -0.02, 0.01], index=business_day_index[:3])
        })
        
        result = add_hurst_features(short_df, windows=(64,))
        
        # Should add column but values should be NaN due to insufficient data
        assert 'hurst_ret_64' in result.columns
        assert result['hurst_ret_64'].isna().all()
    
    def test_multiple_windows(self, sample_ohlcv_df):
        """Test Hurst calculation with multiple windows."""
        windows = (32, 64, 128)
        result = add_hurst_features(sample_ohlcv_df.copy(), windows=windows)
        
        # Should have columns for all windows
        for w in windows:
            assert f'hurst_ret_{w}' in result.columns
    
    def test_ema_smoothing(self, sample_ohlcv_df):
        """Test EMA smoothing of Hurst values."""
        result = add_hurst_features(
            sample_ohlcv_df.copy(),
            windows=(64,),
            ema_halflife=3
        )
        
        # Should have EMA smoothed column
        assert 'hurst_ret_64_emaHL3' in result.columns
        
        # EMA values should be related to original values
        original = result['hurst_ret_64']
        smoothed = result['hurst_ret_64_emaHL3']
        
        # Where both are valid, smoothed should exist
        both_valid = ~(original.isna() | smoothed.isna())
        if both_valid.any():
            # Smoothed series should not be identical (unless all values are the same)
            assert not original[both_valid].equals(smoothed[both_valid]) or len(original[both_valid].unique()) == 1
    
    def test_no_ema_smoothing(self, sample_ohlcv_df):
        """Test behavior when EMA smoothing is disabled."""
        result = add_hurst_features(
            sample_ohlcv_df.copy(),
            windows=(64,),
            ema_halflife=0  # Disable EMA
        )
        
        # Should not have EMA smoothed column
        assert 'hurst_ret_64_emaHL0' not in result.columns
    
    def test_custom_parameters(self, sample_ohlcv_df):
        """Test function with custom parameters."""
        result = add_hurst_features(
            sample_ohlcv_df.copy(),
            ret_col='ret',
            windows=(48, 96),
            ema_halflife=7,
            prefix='custom_hurst'
        )
        
        # Should use custom prefix
        assert 'custom_hurst_48' in result.columns
        assert 'custom_hurst_96' in result.columns
        assert 'custom_hurst_48_emaHL7' in result.columns
    
    def test_min_periods_calculation(self, business_day_index):
        """Test that min_periods is calculated correctly."""
        # Create DataFrame where we can control exactly how much data is available
        df = pd.DataFrame(index=business_day_index[:100])
        df['ret'] = pd.Series([0.01] * 100, index=business_day_index[:100])
        
        result = add_hurst_features(df, windows=(80,))
        
        # With window=80, min_periods should be max(50, 80//2) = 50
        # So we should start getting values around day 50
        hurst_col = result['hurst_ret_80']
        first_valid_idx = hurst_col.first_valid_index()
        
        if first_valid_idx is not None:
            # Should start getting values after sufficient warmup
            first_valid_pos = df.index.get_loc(first_valid_idx)
            assert first_valid_pos >= 40  # Should be at least around min_periods
    
    def test_rolling_window_behavior(self, business_day_index):
        """Test that Hurst calculation respects rolling window behavior."""
        # Create known pattern: first half random, second half trending
        n_days = 120
        df = pd.DataFrame(index=business_day_index[:n_days])
        
        np.random.seed(42)
        first_half = np.random.normal(0, 0.01, n_days//2)
        second_half = np.linspace(0, 0.05, n_days//2)  # Strong trend
        
        returns = np.concatenate([first_half, second_half])
        df['ret'] = pd.Series(returns, index=business_day_index[:n_days])
        
        result = add_hurst_features(df, windows=(40,))
        
        # Should have calculated Hurst values
        hurst_values = result['hurst_ret_40'].dropna()
        assert len(hurst_values) > 0
    
    def test_empty_windows_tuple(self, sample_ohlcv_df):
        """Test behavior with empty windows tuple."""
        result = add_hurst_features(sample_ohlcv_df.copy(), windows=())
        
        # Should not add any columns
        original_cols = set(sample_ohlcv_df.columns)
        result_cols = set(result.columns)
        assert result_cols == original_cols