"""
Tests for the volatility module.

Tests volatility regime features including realized volatility calculations,
volatility ratios, cross-sectional context, and volatility-of-volatility metrics.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.volatility import add_multiscale_vol_regime, add_vol_regime_cs_context, _rolling_z


class TestRollingZ:
    """Tests for the _rolling_z helper function."""
    
    def test_rolling_z_calculation(self, business_day_index):
        """Test rolling z-score calculation."""
        # Create series with known properties
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=business_day_index[:10])
        
        result = _rolling_z(data, win=5)
        
        # Should have NaN for first few values due to min_periods
        assert result.isna().sum() >= 2
        
        # Non-NaN values should have reasonable z-score properties
        valid_z = result.dropna()
        assert not valid_z.empty
    
    def test_rolling_z_with_constant_series(self, business_day_index):
        """Test rolling z-score with constant values."""
        data = pd.Series([5.0] * 20, index=business_day_index[:20])
        
        result = _rolling_z(data, win=10)
        
        # Z-scores should be 0 or NaN (due to zero std dev)
        valid_values = result.dropna()
        if not valid_values.empty:
            assert np.allclose(valid_values, 0, equal_nan=True)


class TestAddMultiscaleVolRegime:
    """Tests for the add_multiscale_vol_regime function."""
    
    def test_missing_return_column(self, sample_ohlcv_df):
        """Test behavior when return column is missing."""
        df = sample_ohlcv_df.copy()
        del df['ret']  # Remove returns
        
        # Should return DataFrame unchanged
        result = add_multiscale_vol_regime(df, ret_col='ret')
        pd.testing.assert_frame_equal(result, df)
    
    def test_basic_volatility_features(self, sample_ohlcv_df):
        """Test that basic volatility features are added."""
        result = add_multiscale_vol_regime(
            sample_ohlcv_df.copy(),
            short_windows=(10, 20),
            long_windows=(60, 100)
        )
        
        # Check realized volatility columns
        expected_rv_cols = ['rv_10', 'rv_20', 'rv_60', 'rv_100']
        for col in expected_rv_cols:
            assert col in result.columns
            assert result[col].dtype == 'float32'
        
        # Check ratio columns
        assert 'rv_ratio_10_60' in result.columns
        assert 'rv_ratio_20_100' in result.columns
        
        # Check regime features
        assert 'vol_regime' in result.columns
        assert 'vol_regime_ema10' in result.columns
    
    def test_realized_volatility_calculation(self, sample_ohlcv_df):
        """Test that realized volatility is calculated correctly."""
        result = add_multiscale_vol_regime(sample_ohlcv_df.copy())
        
        # Manually calculate 20-day realized vol
        returns = sample_ohlcv_df['ret']
        expected_rv_20 = returns.rolling(20, min_periods=7).std(ddof=0)
        
        pd.testing.assert_series_equal(
            result['rv_20'], 
            expected_rv_20.astype('float32'), 
            check_names=False
        )
    
    def test_volatility_ratios(self, sample_ohlcv_df):
        """Test volatility ratio calculations."""
        result = add_multiscale_vol_regime(
            sample_ohlcv_df.copy(),
            short_windows=(10, 20),
            long_windows=(60, 100)
        )
        
        # Manually calculate rv_ratio_20_100
        rv_20 = result['rv_20']
        rv_100 = result['rv_100']
        expected_ratio = (rv_20 / rv_100.replace(0, np.nan)).astype('float32')
        
        pd.testing.assert_series_equal(
            result['rv_ratio_20_100'], 
            expected_ratio, 
            check_names=False
        )
    
    def test_vol_regime_calculation(self, sample_ohlcv_df):
        """Test vol_regime calculation from ratio."""
        result = add_multiscale_vol_regime(sample_ohlcv_df.copy())
        
        if 'rv_ratio_20_100' in result.columns:
            # vol_regime should be log1p of the ratio
            expected_regime = np.log1p(result['rv_ratio_20_100']).astype('float32')
            pd.testing.assert_series_equal(
                result['vol_regime'],
                expected_regime,
                check_names=False
            )
    
    def test_z_score_features(self, sample_ohlcv_df):
        """Test z-score volatility features."""
        result = add_multiscale_vol_regime(sample_ohlcv_df.copy(), z_window=30)
        
        # Should have rv_z_30 column
        assert 'rv_z_30' in result.columns
        
        # Z-scores should have reasonable distribution after warmup
        z_scores = result['rv_z_30'].dropna()
        if len(z_scores) > 10:  # Need sufficient data
            # Most z-scores should be within ±3
            assert (z_scores.abs() < 5).mean() > 0.9
    
    def test_vol_of_vol_calculation(self, sample_ohlcv_df):
        """Test volatility-of-volatility calculation."""
        result = add_multiscale_vol_regime(sample_ohlcv_df.copy())
        
        # Should have vol_of_vol_20d column
        assert 'vol_of_vol_20d' in result.columns
        assert result['vol_of_vol_20d'].dtype == 'float32'
        
        # Values should be non-negative where defined
        vol_of_vol = result['vol_of_vol_20d'].dropna()
        if not vol_of_vol.empty:
            assert (vol_of_vol >= 0).all()
    
    def test_slope_features(self, sample_ohlcv_df):
        """Test volatility slope features."""
        result = add_multiscale_vol_regime(
            sample_ohlcv_df.copy(),
            long_windows=(60, 100),
            slope_win=15
        )
        
        # Should have slope columns
        expected_slope_cols = ['rv60_slope_norm', 'rv100_slope_norm']
        for col in expected_slope_cols:
            assert col in result.columns
            assert result[col].dtype == 'float32'
    
    def test_cross_sectional_context(self, sample_ohlcv_df):
        """Test cross-sectional volatility regime context."""
        # Create mock cross-sectional median
        cs_median = pd.Series(
            [1.0] * len(sample_ohlcv_df), 
            index=sample_ohlcv_df.index
        )
        
        result = add_multiscale_vol_regime(
            sample_ohlcv_df.copy(),
            cs_ratio_median=cs_median
        )
        
        # Should have cross-sectional features
        assert 'vol_regime_cs_median' in result.columns
        assert 'vol_regime_rel' in result.columns
        
        # CS median should be log1p of input
        expected_cs = np.log1p(cs_median).astype('float32')
        pd.testing.assert_series_equal(
            result['vol_regime_cs_median'],
            expected_cs,
            check_names=False
        )
    
    def test_quiet_trend_interaction(self, sample_ohlcv_df):
        """Test quiet trend interaction feature."""
        df = sample_ohlcv_df.copy()
        
        # Add mock trend feature
        df['trend_score_granular'] = pd.Series(
            [0.5] * len(df), 
            index=df.index, 
            dtype='float32'
        )
        
        result = add_multiscale_vol_regime(df)
        
        # Should have quiet_trend column
        assert 'quiet_trend' in result.columns
        assert result['quiet_trend'].dtype == 'float32'
    
    def test_custom_parameters(self, sample_ohlcv_df):
        """Test function with custom parameters."""
        result = add_multiscale_vol_regime(
            sample_ohlcv_df.copy(),
            ret_col='ret',
            short_windows=(5, 15),
            long_windows=(45, 90),
            z_window=45,
            ema_span=5,
            slope_win=10,
            prefix='custom_rv'
        )
        
        # Should use custom windows
        expected_cols = ['custom_rv_5', 'custom_rv_15', 'custom_rv_45', 'custom_rv_90']
        for col in expected_cols:
            assert col in result.columns
        
        # Should use custom z_window
        assert 'rv_z_45' in result.columns
    
    def test_float32_dtypes(self, sample_ohlcv_df):
        """Test that volatility features have float32 dtype."""
        result = add_multiscale_vol_regime(sample_ohlcv_df.copy())
        
        float32_features = [
            'rv_10', 'rv_20', 'rv_60', 'rv_100',
            'rv_ratio_10_60', 'rv_ratio_20_100',
            'vol_regime', 'vol_regime_ema10',
            'rv_z_60', 'vol_of_vol_20d'
        ]
        
        for feature in float32_features:
            if feature in result.columns:
                assert result[feature].dtype == 'float32', f"{feature} should be float32"


class TestAddVolRegimeCsContext:
    """Tests for the add_vol_regime_cs_context function."""
    
    def test_cross_sectional_context_calculation(self, indicators_dict):
        """Test cross-sectional volatility context calculation."""
        # First add volatility features to create rv_ratio_20_100
        for sym, df in indicators_dict.items():
            indicators_dict[sym] = add_multiscale_vol_regime(df)
        
        # Apply cross-sectional context
        add_vol_regime_cs_context(indicators_dict)
        
        # All symbols should have CS context features
        for sym, df in indicators_dict.items():
            assert 'vol_regime_cs_median' in df.columns
            if 'vol_regime' in df.columns:
                assert 'vol_regime_rel' in df.columns
    
    def test_missing_ratio_column(self, indicators_dict):
        """Test behavior when ratio column is missing."""
        # Don't add volatility features, so rv_ratio_20_100 won't exist
        add_vol_regime_cs_context(indicators_dict, ratio_col='nonexistent')
        
        # Should not crash and should not add features
        for sym, df in indicators_dict.items():
            assert 'vol_regime_cs_median' not in df.columns
    
    def test_empty_dataframes(self):
        """Test behavior with empty DataFrames."""
        empty_dict = {'EMPTY': pd.DataFrame()}
        add_vol_regime_cs_context(empty_dict)
        
        # Should not crash
        assert 'EMPTY' in empty_dict
    
    def test_custom_column_names(self, indicators_dict):
        """Test with custom column names."""
        # Add custom ratio column
        for sym, df in indicators_dict.items():
            df['custom_ratio'] = pd.Series([1.0] * len(df), index=df.index)
        
        add_vol_regime_cs_context(
            indicators_dict,
            ratio_col='custom_ratio',
            out_cs_col='custom_cs',
            out_rel_col='custom_rel'
        )
        
        # Should use custom column names
        for sym, df in indicators_dict.items():
            assert 'custom_cs' in df.columns
    
    def test_panel_construction_and_median(self, indicators_dict):
        """Test that cross-sectional panel is constructed correctly."""
        # Add known ratio values
        for i, (sym, df) in enumerate(indicators_dict.items()):
            # Each symbol gets different but consistent ratio values
            df['rv_ratio_20_100'] = pd.Series(
                [1.0 + i * 0.1] * len(df), 
                index=df.index
            )
        
        add_vol_regime_cs_context(indicators_dict)
        
        # Extract the CS median that should have been computed
        # (approximately the median of our values: 1.0, 1.1, 1.2 -> ~1.1)
        sample_df = list(indicators_dict.values())[0]
        cs_values = sample_df['vol_regime_cs_median'].dropna()
        
        if not cs_values.empty:
            # Should be approximately log1p(1.1) for our test data
            expected_approx = np.log1p(1.1)
            assert abs(cs_values.iloc[0] - expected_approx) < 0.1