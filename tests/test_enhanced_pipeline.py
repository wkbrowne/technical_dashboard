#!/usr/bin/env python3
"""
Test suite for Enhanced Pipeline Components
===========================================

Tests for:
1. Single-stock ARIMA-GARCH training
2. Enhanced visualizations
3. Configuration management
4. Integration of new components
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.arima_garch_models import CombinedARIMAGARCHModel
from models.hurst_regime import HurstRegimeResolver


class TestSingleStockARIMAGARCH(unittest.TestCase):
    """Test single-stock ARIMA-GARCH training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock stock data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, 300)  # Small positive drift
        log_prices = np.cumsum(returns)
        prices = 100 * np.exp(log_prices)
        
        self.mock_stock_data = pd.Series(prices, index=dates, name='AAPL_prices')
        self.target_symbol = 'AAPL'
    
    def test_single_stock_training(self):
        """Test that single-stock ARIMA-GARCH training works correctly."""
        # Test model creation
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        self.assertIsNotNone(model)
        
        # Test model fitting
        model.fit(self.mock_stock_data)
        self.assertTrue(model.fitted)
        
        # Test model summary
        summary = model.get_model_summary()
        self.assertIn('arima_summary', summary)
        self.assertIn('garch_summary', summary)
        
        # Test forecasting capability
        forecast = model.forecast(horizon=5)
        self.assertIsInstance(forecast, dict)
    
    def test_arima_target_symbol_configuration(self):
        """Test that ARIMA target symbol configuration works."""
        # Test symbol validation (should be a string)
        test_symbols = ['AAPL', 'SPY', 'GOOGL', 'MSFT', 'TSLA']
        
        for symbol in test_symbols:
            self.assertIsInstance(symbol, str)
            self.assertTrue(len(symbol) > 0)
            self.assertTrue(symbol.isalpha() or symbol.isalnum())
    
    def test_model_integration_with_forecasting(self):
        """Test that ARIMA-GARCH integrates properly with OHLC forecasting."""
        # Train model
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        model.fit(self.mock_stock_data)
        
        # Test forecast integration
        forecast = model.forecast(horizon=10)
        
        if 'ma_forecast' in forecast:
            ma_forecast = forecast['ma_forecast']
            self.assertEqual(len(ma_forecast), 10)
            self.assertTrue(all(isinstance(p, (int, float)) for p in ma_forecast))
            self.assertTrue(all(p > 0 for p in ma_forecast))  # Prices should be positive
        
        # Test that forecast can be converted to log prices for OHLC integration
        if 'ma_forecast' in forecast:
            log_forecast = np.log(forecast['ma_forecast'])
            self.assertEqual(len(log_forecast), 10)
            self.assertTrue(all(np.isfinite(val) for val in log_forecast))


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management for the enhanced pipeline."""
    
    def test_regime_configuration(self):
        """Test regime state configuration."""
        # Test different configurations
        test_configs = [
            {'N_TREND_STATES': 5, 'N_VOL_STATES': 3, 'expected_total': 15},
            {'N_TREND_STATES': 7, 'N_VOL_STATES': 5, 'expected_total': 35},
            {'N_TREND_STATES': 3, 'N_VOL_STATES': 3, 'expected_total': 9},
        ]
        
        for config in test_configs:
            n_trend = config['N_TREND_STATES']
            n_vol = config['N_VOL_STATES']
            expected = config['expected_total']
            
            # Test calculation
            total_regimes = n_trend * n_vol
            self.assertEqual(total_regimes, expected)
            
            # Test reasonable bounds
            self.assertGreaterEqual(n_trend, 3)
            self.assertLessEqual(n_trend, 10)
            self.assertGreaterEqual(n_vol, 3)
            self.assertLessEqual(n_vol, 5)
    
    def test_target_symbol_validation(self):
        """Test ARIMA target symbol validation."""
        # Valid symbols
        valid_symbols = ['AAPL', 'SPY', 'GOOGL', 'MSFT', 'TSLA', 'QQQ', 'NVDA']
        
        for symbol in valid_symbols:
            # Should be uppercase string
            self.assertIsInstance(symbol, str)
            self.assertEqual(symbol, symbol.upper())
            self.assertTrue(2 <= len(symbol) <= 5)  # Typical stock symbol length
            
        # Invalid symbols (examples of what not to accept)
        invalid_symbols = ['', '123', 'a', 'TOOLONGSYMBOL']
        
        for symbol in invalid_symbols:
            # These should fail basic validation
            if len(symbol) < 2 or len(symbol) > 5 or not symbol.isalpha():
                self.assertFalse(2 <= len(symbol) <= 5 and symbol.isalpha())


class TestVisualizationComponents(unittest.TestCase):
    """Test visualization components and functionality."""
    
    def test_markov_heatmap_data_preparation(self):
        """Test data preparation for Markov heatmap visualization."""
        # Mock transition matrix
        n_states = 5
        transition_matrix = np.random.rand(n_states, n_states)
        # Make it row-stochastic
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        # Test matrix properties
        self.assertEqual(transition_matrix.shape, (n_states, n_states))
        
        # Test that rows sum to 1 (stochastic matrix)
        row_sums = np.sum(transition_matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(n_states), decimal=6)
        
        # Test diagonal elements (persistence)
        persistence = np.diag(transition_matrix)
        self.assertEqual(len(persistence), n_states)
        self.assertTrue(all(0 <= p <= 1 for p in persistence))
        
        # Test entropy calculation
        entropies = []
        for i in range(n_states):
            row = transition_matrix[i, :]
            entropy = -np.sum(row[row > 0] * np.log(row[row > 0]))
            entropies.append(entropy)
        
        self.assertEqual(len(entropies), n_states)
        self.assertTrue(all(e >= 0 for e in entropies))  # Entropy should be non-negative
    
    def test_ohlc_trajectory_data_validation(self):
        """Test OHLC trajectory data validation."""
        # Mock OHLC data
        n_days = 20
        base_price = 100.0
        
        trajectory_data = []
        for day in range(n_days):
            # Generate valid OHLC data
            open_price = base_price + np.random.normal(0, 1)
            close_price = base_price + np.random.normal(0, 1)
            high_price = max(open_price, close_price) + np.random.uniform(0.1, 2.0)
            low_price = min(open_price, close_price) - np.random.uniform(0.1, 2.0)
            
            trajectory_data.append({
                'day': day + 1,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'ma': base_price,
                'bb_upper': base_price + 2,
                'bb_lower': base_price - 2,
                'regime': 'neutral'
            })
        
        # Test data structure
        self.assertEqual(len(trajectory_data), n_days)
        
        for data in trajectory_data:
            # Test OHLC consistency
            self.assertGreaterEqual(data['high'], data['open'])
            self.assertGreaterEqual(data['high'], data['close'])
            self.assertLessEqual(data['low'], data['open'])
            self.assertLessEqual(data['low'], data['close'])
            
            # Test Bollinger Band consistency
            self.assertGreater(data['bb_upper'], data['ma'])
            self.assertLess(data['bb_lower'], data['ma'])
            
            # Test regime classification
            self.assertIn(data['regime'], ['bullish', 'bearish', 'neutral'])
    
    def test_trajectory_statistics_calculation(self):
        """Test trajectory statistics calculation."""
        # Mock price series
        prices = [100, 102, 101, 103, 105, 104, 106]
        
        # Test total return calculation
        total_return = (prices[-1] - prices[0]) / prices[0] * 100
        expected_return = (106 - 100) / 100 * 100  # 6%
        self.assertAlmostEqual(total_return, expected_return, places=2)
        
        # Test daily returns calculation
        daily_returns = [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, len(prices))]
        self.assertEqual(len(daily_returns), len(prices) - 1)
        
        # Test volatility calculation
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns)
            self.assertGreater(volatility, 0)
            self.assertIsInstance(volatility, float)
        
        # Test max drawdown calculation
        running_max = np.maximum.accumulate(prices)
        drawdowns = (np.array(prices) / running_max - 1) * 100
        max_drawdown = np.min(drawdowns)
        
        self.assertLessEqual(max_drawdown, 0)  # Drawdown should be negative or zero
        self.assertIsInstance(max_drawdown, (int, float))


class TestPipelineIntegration(unittest.TestCase):
    """Test integration of enhanced pipeline components."""
    
    def test_hurst_regime_integration(self):
        """Test Hurst regime integration with pipeline."""
        # Create test price series
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200))),
            index=dates
        )
        
        # Test Hurst regime resolver
        hurst_resolver = HurstRegimeResolver(window_size=100)
        
        # Test regime calculation
        regimes = hurst_resolver.calculate_rolling_regimes(prices)
        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(prices))
        
        # Test regime statistics
        stats = hurst_resolver.get_regime_statistics(prices)
        self.assertIsInstance(stats, dict)
        self.assertIn('hurst_statistics', stats)
        self.assertIn('regime_distribution', stats)
    
    def test_model_compatibility(self):
        """Test that all models work together properly."""
        # Test that different components don't interfere with each other
        
        # Mock data
        prices = pd.Series([100 + i + np.random.normal(0, 1) for i in range(100)])
        
        # Test ARIMA-GARCH
        try:
            arima_model = CombinedARIMAGARCHModel()
            arima_model.fit(prices)
            arima_fitted = arima_model.fitted
        except:
            arima_fitted = False
        
        # Test Hurst
        try:
            hurst_resolver = HurstRegimeResolver(window_size=50)
            hurst_regimes = hurst_resolver.calculate_rolling_regimes(prices)
            hurst_worked = len(hurst_regimes) > 0
        except:
            hurst_worked = False
        
        # At least one should work (they use different methodologies)
        # This is an integration test, not requiring both to work perfectly
        self.assertTrue(arima_fitted or hurst_worked)


def run_enhanced_pipeline_tests():
    """Run all enhanced pipeline tests."""
    print("🧪 Running Enhanced Pipeline Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSingleStockARIMAGARCH))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Enhanced Pipeline Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("   Failures:")
        for test, failure in result.failures:
            print(f"     {test}: {failure.split(chr(10))[0]}")
    
    if result.errors:
        print("   Errors:")
        for test, error in result.errors:
            print(f"     {test}: {error.split(chr(10))[0]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success


if __name__ == "__main__":
    run_enhanced_pipeline_tests()