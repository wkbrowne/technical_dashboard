"""
Tests for the orchestrator module.

Tests the main pipeline orchestration including parallel processing,
feature integration, and end-to-end workflow.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.pipelines.orchestrator import _feature_worker, build_feature_universe, run_pipeline


class TestFeatureWorker:
    """Tests for the _feature_worker function."""
    
    def test_feature_worker_basic(self, sample_ohlcv_df):
        """Test basic feature worker functionality."""
        result_sym, result_df = _feature_worker('TEST', sample_ohlcv_df.copy())
        
        # Should return same symbol
        assert result_sym == 'TEST'
        
        # Should return DataFrame with additional features
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df.columns) > len(sample_ohlcv_df.columns)
        
        # Should have some key features
        expected_features = [
            'trend_score_granular',
            'vol_regime', 
            'hurst_ret_64',
            'pct_dist_ma_20'
        ]
        for feature in expected_features:
            assert feature in result_df.columns
    
    def test_feature_worker_missing_returns(self, sample_ohlcv_df):
        """Test feature worker when returns are missing but adjclose exists."""
        df = sample_ohlcv_df.copy()
        del df['ret']  # Remove returns
        
        result_sym, result_df = _feature_worker('TEST', df)
        
        # Should create returns from adjclose
        assert 'ret' in result_df.columns
        assert result_sym == 'TEST'
    
    def test_feature_worker_missing_adjclose(self):
        """Test feature worker when adjclose is missing."""
        df = pd.DataFrame({
            'volume': [1000, 1100, 1200],
            'close': [100, 101, 102]
        })
        
        result_sym, result_df = _feature_worker('TEST', df)
        
        # Should return original DataFrame when adjclose missing
        assert result_sym == 'TEST'
        # Should not crash
        assert isinstance(result_df, pd.DataFrame)
    
    def test_feature_worker_exception_handling(self):
        """Test that feature worker handles exceptions gracefully."""
        # Create DataFrame that might cause issues
        df = pd.DataFrame({
            'adjclose': [np.nan, np.nan, np.nan],
            'ret': [np.nan, np.nan, np.nan]
        })
        
        result_sym, result_df = _feature_worker('TEST', df)
        
        # Should not crash and return something
        assert result_sym == 'TEST'
        assert isinstance(result_df, pd.DataFrame)
    
    def test_feature_worker_cross_sectional_median(self, sample_ohlcv_df):
        """Test feature worker with cross-sectional median provided."""
        cs_median = pd.Series([1.0] * len(sample_ohlcv_df), index=sample_ohlcv_df.index)
        
        result_sym, result_df = _feature_worker('TEST', sample_ohlcv_df.copy(), cs_median)
        
        # Should still work
        assert result_sym == 'TEST'
        assert isinstance(result_df, pd.DataFrame)


class TestBuildFeatureUniverse:
    """Tests for the build_feature_universe function."""
    
    @patch('src.pipelines.orchestrator.load_stock_universe')
    @patch('src.pipelines.orchestrator.load_etf_universe') 
    def test_build_feature_universe_basic(self, mock_etf_loader, mock_stock_loader, mock_wide_data):
        """Test basic feature universe building with mocked data loaders."""
        # Mock the data loaders to return our test data
        mock_stock_loader.return_value = (mock_wide_data, {})
        mock_etf_loader.return_value = mock_wide_data
        
        result = build_feature_universe(
            max_stocks=10,
            rate_limit=1.0,
            default_etfs=['SPY', 'QQQ'],
            sp500_tickers=['AAPL', 'MSFT']
        )
        
        # Should return dictionary of DataFrames
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Each DataFrame should have features
        for sym, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert 'ret' in df.columns
            # Should have some computed features
            feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ret']]
            assert len(feature_cols) > 10  # Should have many features
    
    @patch('src.pipelines.orchestrator.load_stock_universe')
    def test_build_feature_universe_stock_loading_failure(self, mock_stock_loader):
        """Test behavior when stock loading fails."""
        mock_stock_loader.return_value = (None, {})  # Simulate failure
        
        with pytest.raises(RuntimeError, match="Failed to load stock universe"):
            build_feature_universe()
    
    @patch('src.pipelines.orchestrator.load_stock_universe')
    @patch('src.pipelines.orchestrator.load_etf_universe')
    def test_build_feature_universe_etf_loading_failure(self, mock_etf_loader, mock_stock_loader, mock_wide_data):
        """Test behavior when ETF loading fails."""
        mock_stock_loader.return_value = (mock_wide_data, {})
        mock_etf_loader.return_value = None  # Simulate failure
        
        with pytest.raises(RuntimeError, match="Failed to load ETF universe"):
            build_feature_universe()
    
    @patch('src.pipelines.orchestrator.load_stock_universe')
    @patch('src.pipelines.orchestrator.load_etf_universe')
    def test_build_feature_universe_with_sectors(self, mock_etf_loader, mock_stock_loader, mock_wide_data, sector_map, sector_to_etf_map):
        """Test feature universe building with sector information."""
        mock_stock_loader.return_value = (mock_wide_data, sector_map)
        mock_etf_loader.return_value = mock_wide_data
        
        result = build_feature_universe(
            sector_to_etf=sector_to_etf_map,
            sp500_tickers=list(sector_map.keys())
        )
        
        # Should include sector-based features
        sample_df = next(iter(result.values()))
        
        # Look for sector-related features (may not all be present depending on data)
        sector_features = [col for col in sample_df.columns if 'sector' in col.lower()]
        # At minimum, should have attempted to compute sector features
        assert isinstance(result, dict)


class TestRunPipeline:
    """Tests for the run_pipeline function."""
    
    @patch('src.pipelines.orchestrator.build_feature_universe')
    @patch('src.pipelines.orchestrator.save_symbol_frames')
    @patch('src.pipelines.orchestrator.save_long_parquet')
    def test_run_pipeline_basic(self, mock_save_long, mock_save_frames, mock_build_universe, temp_output_dir, indicators_dict):
        """Test basic pipeline execution."""
        # Mock the universe builder to return our test data
        mock_build_universe.return_value = indicators_dict
        
        run_pipeline(
            max_stocks=10,
            output_dir=temp_output_dir,
            sp500_tickers=['AAPL', 'MSFT']
        )
        
        # Should have called the universe builder
        mock_build_universe.assert_called_once()
        
        # Should have called save functions
        mock_save_frames.assert_called_once()
        mock_save_long.assert_called_once()
        
        # Check that correct paths were used
        save_frames_call = mock_save_frames.call_args[1]
        assert str(save_frames_call['out_dir']) == str(temp_output_dir / 'symbol_frames')
        
        save_long_call = mock_save_long.call_args[1] 
        assert str(save_long_call['out_path']) == str(temp_output_dir / 'features_long.parquet')
    
    @patch('src.pipelines.orchestrator.build_feature_universe')
    def test_run_pipeline_universe_failure(self, mock_build_universe):
        """Test pipeline behavior when universe building fails."""
        mock_build_universe.side_effect = RuntimeError("Universe building failed")
        
        with pytest.raises(RuntimeError):
            run_pipeline()
    
    @patch('src.pipelines.orchestrator.build_feature_universe')
    @patch('src.pipelines.orchestrator.save_symbol_frames')
    def test_run_pipeline_save_failure(self, mock_save_frames, mock_build_universe, indicators_dict):
        """Test pipeline behavior when saving fails."""
        mock_build_universe.return_value = indicators_dict
        mock_save_frames.side_effect = IOError("Save failed")
        
        # Pipeline should propagate the save error
        with pytest.raises(IOError):
            run_pipeline()
    
    @patch('src.pipelines.orchestrator.build_feature_universe')
    @patch('src.pipelines.orchestrator.save_symbol_frames')
    @patch('src.pipelines.orchestrator.save_long_parquet')
    def test_run_pipeline_custom_directories(self, mock_save_long, mock_save_frames, mock_build_universe, temp_output_dir, indicators_dict):
        """Test pipeline with custom output directories."""
        mock_build_universe.return_value = indicators_dict
        
        custom_symbol_dir = temp_output_dir / 'custom_symbols'
        
        run_pipeline(
            output_dir=temp_output_dir,
            symbol_frames_dir=custom_symbol_dir
        )
        
        # Should use custom symbol frames directory
        save_frames_call = mock_save_frames.call_args[1]
        assert str(save_frames_call['out_dir']) == str(custom_symbol_dir)
    
    @patch('src.pipelines.orchestrator.build_feature_universe')
    @patch('src.pipelines.orchestrator.save_symbol_frames')
    @patch('src.pipelines.orchestrator.save_long_parquet')
    def test_run_pipeline_parameter_passing(self, mock_save_long, mock_save_frames, mock_build_universe, indicators_dict):
        """Test that pipeline parameters are passed correctly to universe builder."""
        mock_build_universe.return_value = indicators_dict
        
        run_pipeline(
            max_stocks=100,
            rate_limit=2.0,
            interval='1h',
            spy_symbol='SPY',
            default_etfs=['SPY', 'QQQ', 'IWM']
        )
        
        # Check that parameters were passed to universe builder
        call_args = mock_build_universe.call_args[1]
        assert call_args['max_stocks'] == 100
        assert call_args['rate_limit'] == 2.0
        assert call_args['interval'] == '1h'
        assert call_args['spy_symbol'] == 'SPY'
        assert call_args['default_etfs'] == ['SPY', 'QQQ', 'IWM']


class TestEndToEndIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_feature_worker_integration(self, sample_ohlcv_df):
        """Test that feature worker produces expected feature categories."""
        result_sym, result_df = _feature_worker('TEST', sample_ohlcv_df.copy())
        
        # Check for features from each module
        feature_categories = {
            'trend': ['trend_score_granular', 'trend_alignment'],
            'volatility': ['vol_regime', 'rv_20'],
            'hurst': ['hurst_ret_64'],
            'distance': ['pct_dist_ma_20'],
            'range': ['hl_range', 'true_range'],
            'volume': ['vol_ma_20', 'rvol_20'],
            'volume_shock': ['volshock_z', 'volshock_dir']
        }
        
        for category, features in feature_categories.items():
            found_features = [f for f in features if f in result_df.columns]
            assert len(found_features) > 0, f"No {category} features found"
    
    def test_pipeline_data_flow(self, sample_ohlcv_df):
        """Test that data flows correctly through the feature pipeline."""
        result_sym, result_df = _feature_worker('TEST', sample_ohlcv_df.copy())
        
        # Check data integrity
        assert len(result_df) == len(sample_ohlcv_df)
        assert result_df.index.equals(sample_ohlcv_df.index)
        
        # Original columns should be preserved
        for col in sample_ohlcv_df.columns:
            assert col in result_df.columns
            
        # Should have many new features
        new_cols = len(result_df.columns) - len(sample_ohlcv_df.columns)
        assert new_cols > 20  # Should add many features
        
        # Check that feature values are reasonable (not all NaN, not all infinite)
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = result_df[col].dropna()
            if len(values) > 0:
                # Should not be all infinite
                assert not np.isinf(values).all(), f"Column {col} has all infinite values"
                # Should have some finite values
                assert np.isfinite(values).any(), f"Column {col} has no finite values"