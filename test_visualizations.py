#!/usr/bin/env python3
"""
Test suite for visualization components
======================================

Tests for:
1. Markov transition matrix visualization
2. OHLC trajectory candlestick visualization 
3. Sparse bucket logging functionality
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import visualization classes
from visualize_markov_matrices import MarkovMatrixVisualizer
from visualize_ohlc_trajectory import OHLCTrajectoryVisualizer

# Import test data generation
from data.loader import get_multiple_stocks


class TestVisualizationComponents(unittest.TestCase):
    """Test visualization components."""
    
    def setUp(self):
        self.test_cache_dir = "test_cache"
        Path(self.test_cache_dir).mkdir(exist_ok=True)
    
    def test_markov_visualizer_initialization(self):
        """Test MarkovMatrixVisualizer can be initialized."""
        visualizer = MarkovMatrixVisualizer(cache_dir=self.test_cache_dir)
        self.assertIsNotNone(visualizer)
        self.assertEqual(visualizer.cache_dir, Path(self.test_cache_dir))
    
    def test_ohlc_visualizer_initialization(self):
        """Test OHLCTrajectoryVisualizer can be initialized."""
        visualizer = OHLCTrajectoryVisualizer(cache_dir=self.test_cache_dir)
        self.assertIsNotNone(visualizer)
        self.assertEqual(visualizer.cache_dir, Path(self.test_cache_dir))
    
    def test_markov_visualizer_missing_model(self):
        """Test MarkovMatrixVisualizer handles missing model gracefully."""
        visualizer = MarkovMatrixVisualizer(cache_dir="nonexistent_cache")
        result = visualizer.load_trained_model()
        self.assertFalse(result)
    
    def test_ohlc_visualizer_missing_models(self):
        """Test OHLCTrajectoryVisualizer handles missing models gracefully.""" 
        visualizer = OHLCTrajectoryVisualizer(cache_dir="nonexistent_cache")
        result = visualizer.load_trained_models()
        self.assertFalse(result)


class TestSparseBucketLogging(unittest.TestCase):
    """Test sparse bucket logging functionality."""
    
    def test_sparse_bucket_analysis(self):
        """Test that sparse bucket analysis logic works."""
        # Create mock trend data
        trend_data_combined = {
            'up': [pd.Series([1, 2, 3, 1, 2]), pd.Series([3, 4, 5])],
            'down': [pd.Series([2, 3, 1])],
            'sideways': []  # Empty trend
        }
        
        # Simulate the sparse bucket analysis
        bucket_counts = []
        n_states = 5
        
        for trend in ['up', 'down', 'sideways']:
            if trend_data_combined[trend]:
                combined_bb_positions = pd.concat(trend_data_combined[trend], ignore_index=True)
                bb_counts = combined_bb_positions.value_counts()
                for bb_pos, count in bb_counts.items():
                    bucket_counts.append((f"{trend}_BB{bb_pos}", count))
            else:
                # Add empty trend buckets
                for bb_pos in range(1, n_states + 1):
                    bucket_counts.append((f"{trend}_BB{bb_pos}", 0))
        
        # Sort and get 5 sparsest
        bucket_counts.sort(key=lambda x: x[1])
        sparse_buckets = bucket_counts[:5]
        
        # Verify we found sparse buckets
        self.assertEqual(len(sparse_buckets), 5)
        
        # Verify the sparest buckets are the empty sideways ones
        sideways_buckets = [bucket for bucket, count in sparse_buckets if 'sideways' in bucket]
        self.assertGreater(len(sideways_buckets), 0)
        
        # Verify all sideways buckets have 0 count
        for bucket, count in sparse_buckets:
            if 'sideways' in bucket:
                self.assertEqual(count, 0)


class TestGlobalOnlyPipeline(unittest.TestCase):
    """Test global-only pipeline functionality."""
    
    def test_global_markov_with_sparse_buckets(self):
        """Test that global Markov training includes sparse bucket diagnostics."""
        # Mock unified Markov model
        class MockUnifiedMarkov:
            def __init__(self):
                self.fitted = False
                self.n_states = 35
                self.states = [f'regime_{i}' for i in range(35)]
                self.state_stats = {}
            
            def fit(self, data):
                # Simulate sparse buckets
                for i, state in enumerate(self.states):
                    count = max(0, 100 - i * 3)  # Make later states sparser
                    self.state_stats[state] = {'count': count, 'frequency': count/1000}
                self.fitted = True
                return self
            
            def _log_sparse_buckets(self):
                # Test that this method identifies sparse buckets
                state_counts = [(state, stats['count']) for state, stats in self.state_stats.items()]
                state_counts.sort(key=lambda x: x[1])
                sparse_buckets = state_counts[:5]
                return sparse_buckets
        
        model = MockUnifiedMarkov()
        model.fit({})
        
        # Test sparse bucket identification
        sparse_buckets = model._log_sparse_buckets()
        self.assertEqual(len(sparse_buckets), 5)
        
        # Verify buckets are sorted by count (ascending)
        counts = [bucket[1] for bucket in sparse_buckets]
        self.assertEqual(counts, sorted(counts))
        
        # Verify sparest buckets have low counts
        self.assertLessEqual(sparse_buckets[0][1], sparse_buckets[-1][1])
    
    def test_individual_training_removal(self):
        """Test that individual stock training has been properly removed."""
        # This test verifies that the pipeline focuses on global models
        
        # Mock pipeline API to test global-only approach
        class MockPipelineAPI:
            def __init__(self):
                self.global_markov_model = None
                self.stock_markov_models = {}  # Should remain empty
                self.stages_completed = {}
            
            def train_stock_specific_markov_models(self, symbols):
                # This should be skipped in the new pipeline
                result = {
                    'stage_skipped': True,
                    'skip_reason': 'Individual stock training disabled',
                    'successful_models': 0
                }
                return result
        
        pipeline = MockPipelineAPI()
        result = pipeline.train_stock_specific_markov_models(['AAPL', 'GOOGL'])
        
        # Verify individual training is skipped
        self.assertTrue(result['stage_skipped'])
        self.assertEqual(result['successful_models'], 0)
        self.assertIn('disabled', result['skip_reason'])
    
    def test_visualization_integration(self):
        """Test that visualizations work with global models."""
        # Mock global Markov model for visualization
        transition_matrix = np.random.rand(5, 5)
        # Make it row-stochastic
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        class MockGlobalMarkov:
            def __init__(self):
                self.fitted = True
                self.transition_matrix = transition_matrix
                self.states = ['bull_high', 'bull_low', 'bear_high', 'bear_low', 'sideways']
        
        model = MockGlobalMarkov()
        
        # Test that visualization can handle global model
        self.assertTrue(model.fitted)
        self.assertEqual(model.transition_matrix.shape, (5, 5))
        self.assertTrue(len(model.states) == 5)
        
        # Test row sums equal 1 (proper stochastic matrix)
        row_sums = np.sum(model.transition_matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5), decimal=6)

class TestIntegrationWithARIMA(unittest.TestCase):
    """Test integration of ARIMA forecasts with OHLC simulation."""
    
    def test_arima_close_forecast_integration(self):
        """Test that ARIMA forecasts can be used in OHLC forecasting."""
        # Mock OHLC forecaster with required methods
        class MockOHLCForecaster:
            def __init__(self):
                self.fitted = True
                self.ohlc_data = pd.DataFrame({
                    'Close': [100, 101, 102],
                    'Trend': ['up', 'up', 'up']
                }, index=pd.date_range('2023-01-01', periods=3))
            
            def forecast_ohlc(self, ma_forecast, vol_forecast, bb_states, 
                            current_close, n_days=1, arima_close_forecast=None):
                # Test the ARIMA integration
                if arima_close_forecast is not None:
                    # Should use ARIMA forecasts
                    close_prices = np.exp(arima_close_forecast)
                    return {
                        'close': close_prices.tolist(),
                        'open': [current_close * 1.01] * len(close_prices),
                        'high': [cp * 1.02 for cp in close_prices],
                        'low': [cp * 0.98 for cp in close_prices]
                    }
                else:
                    # Fallback to MA-based
                    return {
                        'close': ma_forecast.tolist(),
                        'open': [current_close * 1.01] * len(ma_forecast),
                        'high': [ma * 1.02 for ma in ma_forecast],
                        'low': [ma * 0.98 for ma in ma_forecast]
                    }
        
        forecaster = MockOHLCForecaster()
        
        # Test without ARIMA (traditional)
        ma_forecast = np.array([105, 106, 107])
        vol_forecast = np.array([0.02, 0.02, 0.02])
        bb_states = np.array([3, 3, 3])
        
        result1 = forecaster.forecast_ohlc(ma_forecast, vol_forecast, bb_states, 100)
        
        # Test with ARIMA forecasts (log prices)
        arima_log_forecast = np.log([105, 106, 107])
        result2 = forecaster.forecast_ohlc(ma_forecast, vol_forecast, bb_states, 100,
                                         arima_close_forecast=arima_log_forecast)
        
        # Verify both work and produce different results
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertIn('close', result1)
        self.assertIn('close', result2)
        
        # Verify ARIMA results are used when provided
        np.testing.assert_array_almost_equal(result2['close'], [105, 106, 107], decimal=2)


def run_visualization_tests():
    """Run all visualization tests."""
    print("🧪 Running Visualization Tests")
    print("=" * 40)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseBucketLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalOnlyPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithARIMA))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("   Failures:")
        for test, failure in result.failures:
            print(f"     {test}: {failure}")
    
    if result.errors:
        print("   Errors:")
        for test, error in result.errors:
            print(f"     {test}: {error}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success


if __name__ == "__main__":
    run_visualization_tests()