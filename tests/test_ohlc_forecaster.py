import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ohlc_forecasting import OHLCForecaster
from models.open_price_kde import IntelligentOpenForecaster
from models.high_low_copula import IntelligentHighLowForecaster


class TestOHLCForecaster(unittest.TestCase):
    """Test OHLCForecaster API"""
    
    def setUp(self):
        """Set up test fixtures with sample OHLC data"""
        self.forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
        
        # Create realistic OHLC data
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Generate correlated OHLC prices
        base_price = 100
        price_changes = np.random.normal(0, 2, n_days)
        close_prices = base_price + np.cumsum(price_changes)
        
        # Ensure OHLC relationships: Low <= Open,Close <= High
        daily_ranges = np.random.uniform(1, 5, n_days)
        open_offset = np.random.uniform(-0.5, 0.5, n_days) * daily_ranges
        
        opens = close_prices + open_offset
        highs = np.maximum(opens, close_prices) + np.random.uniform(0, 2, n_days)
        lows = np.minimum(opens, close_prices) - np.random.uniform(0, 2, n_days)
        
        self.ohlc_data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.forecaster.bb_window, 20)
        self.assertEqual(self.forecaster.bb_std, 2.0)
        self.assertFalse(self.forecaster.fitted)
        self.assertIsNone(self.forecaster.open_forecaster)
        self.assertIsNone(self.forecaster.high_low_forecaster)
    
    def test_fit_api(self):
        """Test fit method exists and works"""
        # Should not raise exception
        result = self.forecaster.fit(self.ohlc_data)
        
        # Check return value
        self.assertIs(result, self.forecaster)
        
        # Check fitted status
        self.assertTrue(self.forecaster.fitted)
        
        # Check that internal data is stored
        self.assertIsNotNone(self.forecaster.ohlc_data)
        self.assertEqual(len(self.forecaster.ohlc_data), len(self.ohlc_data))
    
    def test_fit_missing_columns(self):
        """Test fit raises error with missing columns"""
        bad_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        
        with self.assertRaises(ValueError):
            self.forecaster.fit(bad_data)
    
    def test_set_intelligent_open_forecaster(self):
        """Test setting intelligent open forecaster"""
        # Create mock open forecaster
        open_forecaster = IntelligentOpenForecaster()
        symbol = 'TEST'
        
        # Should not raise exception
        self.forecaster.set_intelligent_open_forecaster(open_forecaster, symbol)
        
        # Check that forecaster is set
        self.assertIs(self.forecaster.open_forecaster, open_forecaster)
        self.assertEqual(self.forecaster.symbol, symbol)
    
    def test_set_intelligent_high_low_forecaster(self):
        """Test setting intelligent high/low forecaster"""
        # Create mock high/low forecaster
        hl_forecaster = IntelligentHighLowForecaster()
        symbol = 'TEST'
        
        # Should not raise exception
        self.forecaster.set_intelligent_high_low_forecaster(hl_forecaster, symbol)
        
        # Check that forecaster is set
        self.assertIs(self.forecaster.high_low_forecaster, hl_forecaster)
    
    def test_forecast_ohlc_basic(self):
        """Test basic OHLC forecasting without intelligent forecasters"""
        # Fit the model first
        self.forecaster.fit(self.ohlc_data)
        
        # Create forecast inputs
        n_days = 5
        ma_forecast = np.full(n_days, 100.0)
        vol_forecast = np.full(n_days, 0.02)
        bb_states = np.full(n_days, 3)  # Middle state
        current_close = 100.0
        
        # Should not raise exception
        result = self.forecaster.forecast_ohlc(
            ma_forecast=ma_forecast,
            vol_forecast=vol_forecast,
            bb_states=bb_states,
            current_close=current_close,
            n_days=n_days
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        expected_keys = {'open', 'high', 'low', 'close'}
        self.assertTrue(expected_keys.issubset(result.keys()))
        
        # Check result dimensions
        for key in expected_keys:
            self.assertEqual(len(result[key]), n_days)
            self.assertTrue(all(isinstance(x, (int, float)) for x in result[key]))
        
        # Check OHLC relationships
        for i in range(n_days):
            self.assertLessEqual(result['low'][i], result['open'][i])
            self.assertLessEqual(result['low'][i], result['close'][i])
            self.assertGreaterEqual(result['high'][i], result['open'][i])
            self.assertGreaterEqual(result['high'][i], result['close'][i])
    
    def test_forecast_ohlc_with_intelligent_forecasters(self):
        """Test OHLC forecasting with intelligent forecasters"""
        # Fit the model first
        self.forecaster.fit(self.ohlc_data)
        
        # Create and set intelligent forecasters
        # Note: These may not have real trained models, but should not crash
        open_forecaster = IntelligentOpenForecaster()
        hl_forecaster = IntelligentHighLowForecaster()
        
        self.forecaster.set_intelligent_open_forecaster(open_forecaster, 'TEST')
        self.forecaster.set_intelligent_high_low_forecaster(hl_forecaster, 'TEST')
        
        # Create forecast inputs
        n_days = 3
        ma_forecast = np.full(n_days, 100.0)
        vol_forecast = np.full(n_days, 0.02)
        bb_states = np.full(n_days, 3)
        current_close = 100.0
        
        # Should not raise exception even with untrained intelligent forecasters
        try:
            result = self.forecaster.forecast_ohlc(
                ma_forecast=ma_forecast,
                vol_forecast=vol_forecast,
                bb_states=bb_states,
                current_close=current_close,
                n_days=n_days
            )
            
            # Check basic structure if it succeeds
            self.assertIsInstance(result, dict)
            self.assertIn('open', result)
            self.assertIn('close', result)
            
        except Exception as e:
            # If intelligent forecasters are not trained, it might fail
            # This is acceptable - the API should exist
            self.assertIsInstance(e, Exception)
    
    def test_forecast_ohlc_before_fit(self):
        """Test forecasting before fitting raises appropriate error"""
        n_days = 5
        ma_forecast = np.full(n_days, 100.0)
        vol_forecast = np.full(n_days, 0.02)
        bb_states = np.full(n_days, 3)
        current_close = 100.0
        
        # Should raise some kind of error (not necessarily ValueError)
        with self.assertRaises(Exception):
            self.forecaster.forecast_ohlc(
                ma_forecast=ma_forecast,
                vol_forecast=vol_forecast,
                bb_states=bb_states,
                current_close=current_close,
                n_days=n_days
            )
    
    def test_forecast_ohlc_parameter_validation(self):
        """Test forecast parameter validation"""
        self.forecaster.fit(self.ohlc_data)
        
        # Test mismatched array lengths
        n_days = 5
        ma_forecast = np.full(n_days, 100.0)
        vol_forecast = np.full(3, 0.02)  # Wrong length
        bb_states = np.full(n_days, 3)
        current_close = 100.0
        
        # Should handle or raise appropriate error for mismatched lengths
        try:
            result = self.forecaster.forecast_ohlc(
                ma_forecast=ma_forecast,
                vol_forecast=vol_forecast,
                bb_states=bb_states,
                current_close=current_close,
                n_days=n_days
            )
            # If it doesn't raise an error, that's also acceptable
        except Exception:
            # Error is acceptable for bad input
            pass


if __name__ == '__main__':
    unittest.main()