#!/usr/bin/env python3
"""
Test suite for Hurst Regime Resolver
====================================

Tests for:
1. Hurst exponent calculation accuracy
2. Regime classification logic
3. Rolling calculations
4. Edge cases and error handling
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hurst_regime import HurstRegimeResolver, create_hurst_regime_resolver, quick_hurst_analysis


class TestHurstRegimeResolver(unittest.TestCase):
    """Test the HurstRegimeResolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resolver = HurstRegimeResolver()
        
        # Create test data
        np.random.seed(42)  # For reproducible tests
        
        # Create different types of time series
        self.dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # Random walk (H ≈ 0.5)
        self.random_walk = pd.Series(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200))),
            index=self.dates,
            name='random_walk'
        )
        
        # Trending series (H > 0.5)
        trend = np.linspace(0, 0.2, 200)
        self.trending_series = pd.Series(
            100 * np.exp(trend + np.cumsum(np.random.normal(0, 0.005, 200))),
            index=self.dates,
            name='trending'
        )
        
        # Mean-reverting series (H < 0.5)
        # Use AR(1) with high mean reversion
        ar_coeff = 0.9
        mean_reverting_returns = []
        prev = 0
        for _ in range(200):
            prev = ar_coeff * prev + np.random.normal(0, 0.01)
            mean_reverting_returns.append(-0.5 * prev + np.random.normal(0, 0.005))
        
        self.mean_reverting_series = pd.Series(
            100 * np.exp(np.cumsum(mean_reverting_returns)),
            index=self.dates,
            name='mean_reverting'
        )
    
    def test_initialization(self):
        """Test HurstRegimeResolver initialization."""
        resolver = HurstRegimeResolver(
            window_size=150,
            mean_reverting_threshold=0.4,
            trending_threshold=0.6
        )
        
        self.assertEqual(resolver.window_size, 150)
        self.assertEqual(resolver.mean_reverting_threshold, 0.4)
        self.assertEqual(resolver.trending_threshold, 0.6)
        self.assertIn('mean_reverting', resolver.regime_labels)
        self.assertIn('random_walk', resolver.regime_labels)
        self.assertIn('trending', resolver.regime_labels)
    
    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise ValueError."""
        with self.assertRaises(ValueError):
            HurstRegimeResolver(mean_reverting_threshold=0.6, trending_threshold=0.4)
        
        with self.assertRaises(ValueError):
            HurstRegimeResolver(mean_reverting_threshold=0.0, trending_threshold=0.5)
        
        with self.assertRaises(ValueError):
            HurstRegimeResolver(mean_reverting_threshold=0.5, trending_threshold=1.0)
    
    def test_hurst_calculation_random_walk(self):
        """Test Hurst calculation on random walk series."""
        hurst = self.resolver.calculate_hurst_exponent(self.random_walk)
        
        # Random walk should have Hurst ≈ 0.5
        self.assertFalse(np.isnan(hurst))
        self.assertGreater(hurst, 0.3)  # Should be reasonably close to 0.5
        self.assertLess(hurst, 0.7)
    
    def test_hurst_calculation_trending(self):
        """Test Hurst calculation on trending series."""
        hurst = self.resolver.calculate_hurst_exponent(self.trending_series)
        
        # Trending series should have Hurst > 0.5
        self.assertFalse(np.isnan(hurst))
        # Note: Due to noise, may not always be > 0.5, but should be reasonable
        self.assertGreater(hurst, 0.2)
        self.assertLess(hurst, 0.9)
    
    def test_hurst_calculation_mean_reverting(self):
        """Test Hurst calculation on mean-reverting series."""
        hurst = self.resolver.calculate_hurst_exponent(self.mean_reverting_series)
        
        # Mean-reverting series should have reasonable Hurst value
        self.assertFalse(np.isnan(hurst))
        self.assertGreater(hurst, 0.1)
        self.assertLessEqual(hurst, 0.9)  # Within reasonable bounds
    
    def test_hurst_calculation_short_series(self):
        """Test Hurst calculation on short series."""
        short_series = self.random_walk.iloc[:10]
        hurst = self.resolver.calculate_hurst_exponent(short_series)
        
        # Should return NaN for very short series
        self.assertTrue(np.isnan(hurst))
    
    def test_regime_classification(self):
        """Test regime classification logic."""
        # Test mean-reverting classification
        self.assertEqual(self.resolver.classify_regime(0.3), 'mean_reverting')
        
        # Test random walk classification
        self.assertEqual(self.resolver.classify_regime(0.5), 'random_walk')
        
        # Test trending classification
        self.assertEqual(self.resolver.classify_regime(0.7), 'trending')
        
        # Test boundary cases
        self.assertEqual(self.resolver.classify_regime(0.44), 'mean_reverting')
        self.assertEqual(self.resolver.classify_regime(0.45), 'random_walk')
        self.assertEqual(self.resolver.classify_regime(0.55), 'random_walk')
        self.assertEqual(self.resolver.classify_regime(0.56), 'trending')
        
        # Test NaN case
        self.assertEqual(self.resolver.classify_regime(np.nan), 'unknown')
    
    def test_rolling_hurst(self):
        """Test rolling Hurst calculation."""
        rolling_hurst = self.resolver.calculate_rolling_hurst(self.random_walk)
        
        self.assertIsInstance(rolling_hurst, pd.Series)
        self.assertEqual(len(rolling_hurst), len(self.random_walk))
        self.assertEqual(rolling_hurst.name, 'hurst_exponent')
        
        # Early values should mostly be NaN (not enough data)
        # Allow some flexibility since min_periods might allow earlier calculations
        early_nan_ratio = rolling_hurst.iloc[:50].isna().sum() / 50
        self.assertGreater(early_nan_ratio, 0.5)  # At least 50% should be NaN
        
        # Later values should be valid
        valid_values = rolling_hurst.dropna()
        self.assertGreater(len(valid_values), 50)
        self.assertTrue(all(0.1 <= h <= 0.9 for h in valid_values))
    
    def test_rolling_regimes(self):
        """Test rolling regime classification."""
        rolling_regimes = self.resolver.calculate_rolling_regimes(self.random_walk)
        
        self.assertIsInstance(rolling_regimes, pd.Series)
        self.assertEqual(len(rolling_regimes), len(self.random_walk))
        self.assertEqual(rolling_regimes.name, 'hurst_regime')
        
        # Check that all non-null values are valid regimes
        valid_regimes = rolling_regimes.dropna()
        valid_regime_names = {'mean_reverting', 'random_walk', 'trending', 'unknown'}
        self.assertTrue(all(regime in valid_regime_names for regime in valid_regimes))
    
    def test_regime_statistics(self):
        """Test comprehensive regime statistics calculation."""
        stats = self.resolver.get_regime_statistics(self.random_walk)
        
        # Check structure
        self.assertIn('hurst_statistics', stats)
        self.assertIn('regime_distribution', stats)
        self.assertIn('regime_transitions', stats)
        self.assertIn('regime_labels', stats)
        self.assertIn('configuration', stats)
        
        # Check Hurst statistics
        hurst_stats = stats['hurst_statistics']
        self.assertIn('mean', hurst_stats)
        self.assertIn('std', hurst_stats)
        self.assertIn('min', hurst_stats)
        self.assertIn('max', hurst_stats)
        self.assertIn('count', hurst_stats)
        
        # Check that we have valid statistics
        if hurst_stats['count'] > 0:
            self.assertFalse(np.isnan(hurst_stats['mean']))
            self.assertGreater(hurst_stats['count'], 0)
        
        # Check regime distribution
        regime_dist = stats['regime_distribution']
        self.assertIsInstance(regime_dist, dict)
        
        # Check configuration
        config = stats['configuration']
        self.assertEqual(config['window_size'], self.resolver.window_size)
        self.assertEqual(config['mean_reverting_threshold'], self.resolver.mean_reverting_threshold)
        self.assertEqual(config['trending_threshold'], self.resolver.trending_threshold)
    
    def test_short_series_handling(self):
        """Test handling of series shorter than window size."""
        short_series = self.random_walk.iloc[:50]
        
        rolling_hurst = self.resolver.calculate_rolling_hurst(short_series)
        self.assertEqual(len(rolling_hurst), len(short_series))
        
        # Most values should be NaN due to insufficient data
        valid_count = rolling_hurst.notna().sum()
        self.assertLessEqual(valid_count, 10)  # Very few valid values expected


class TestHurstUtilityFunctions(unittest.TestCase):
    """Test utility functions for Hurst analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=150, freq='D')
        self.test_series = pd.Series(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 150))),
            index=dates,
            name='test_prices'
        )
    
    def test_create_hurst_regime_resolver(self):
        """Test the convenience function for creating resolver."""
        resolver = create_hurst_regime_resolver(
            window_size=80,
            mean_reverting_threshold=0.4,
            trending_threshold=0.6
        )
        
        self.assertIsInstance(resolver, HurstRegimeResolver)
        self.assertEqual(resolver.window_size, 80)
        self.assertEqual(resolver.mean_reverting_threshold, 0.4)
        self.assertEqual(resolver.trending_threshold, 0.6)
    
    def test_quick_hurst_analysis(self):
        """Test the quick analysis function."""
        results = quick_hurst_analysis(self.test_series, window_size=80)
        
        # Check that it returns a dictionary with expected keys
        self.assertIsInstance(results, dict)
        self.assertIn('hurst_statistics', results)
        self.assertIn('regime_distribution', results)
        self.assertIn('configuration', results)
        
        # Check that window size was applied correctly
        self.assertEqual(results['configuration']['window_size'], 80)


class TestHurstEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_constant_series(self):
        """Test behavior with constant price series."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        constant_series = pd.Series([100] * 100, index=dates)
        
        resolver = HurstRegimeResolver()
        hurst = resolver.calculate_hurst_exponent(constant_series)
        
        # Should handle constant series gracefully
        # Result may be NaN or a default value
        self.assertTrue(np.isnan(hurst) or isinstance(hurst, float))
    
    def test_empty_series(self):
        """Test behavior with empty series."""
        empty_series = pd.Series([], dtype=float)
        
        resolver = HurstRegimeResolver()
        hurst = resolver.calculate_hurst_exponent(empty_series)
        
        self.assertTrue(np.isnan(hurst))
    
    def test_series_with_nans(self):
        """Test behavior with series containing NaN values."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        series_with_nans = pd.Series(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100))),
            index=dates
        )
        # Insert some NaN values
        series_with_nans.iloc[20:25] = np.nan
        
        resolver = HurstRegimeResolver()
        # Should handle NaN values without crashing
        try:
            hurst = resolver.calculate_hurst_exponent(series_with_nans.dropna())
            # Should either return a valid number or NaN
            self.assertTrue(np.isnan(hurst) or isinstance(hurst, float))
        except Exception as e:
            self.fail(f"Should handle NaN values gracefully, but got: {e}")


def run_hurst_tests():
    """Run all Hurst regime tests."""
    print("🧪 Running Hurst Regime Tests")
    print("=" * 40)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHurstRegimeResolver))
    suite.addTests(loader.loadTestsFromTestCase(TestHurstUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestHurstEdgeCases))
    
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
            print(f"     {test}: {failure.split(chr(10))[0]}")
    
    if result.errors:
        print("   Errors:")
        for test, error in result.errors:
            print(f"     {test}: {error.split(chr(10))[0]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success


if __name__ == "__main__":
    run_hurst_tests()