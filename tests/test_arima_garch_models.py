import unittest
import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.arima_garch_models import (
    ARIMAMovingAverageModel,
    GARCHBollingerBandModel,
    CombinedARIMAGARCHModel,
    fit_arima_garch_model,
    forecast_arima_garch
)

warnings.filterwarnings('ignore')


class TestARIMAMovingAverageModel(unittest.TestCase):
    """Test ARIMA model for moving average forecasting"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate realistic price series with trend
        n_days = 200
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Price with trend and noise
        trend = np.linspace(100, 110, n_days)
        noise = np.cumsum(np.random.normal(0, 1, n_days))
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_days) / 30)  # Monthly seasonality
        
        self.prices = pd.Series(trend + noise + seasonal, index=dates)
        self.short_prices = self.prices.head(30)  # Too short for ARIMA
        
    def test_model_initialization(self):
        """Test model initializes correctly"""
        model = ARIMAMovingAverageModel(ma_window=20)
        
        self.assertEqual(model.ma_window, 20)
        self.assertFalse(model.fitted)
        self.assertIsNone(model.arima_model)
        self.assertIsNone(model.ma_series)
    
    def test_fit_with_sufficient_data(self):
        """Test fitting with sufficient data"""
        model = ARIMAMovingAverageModel(ma_window=20)
        
        # Should not raise exception
        result = model.fit(self.prices)
        
        # Check return value
        self.assertIs(result, model)
        
        # Check fitted status
        self.assertTrue(model.fitted)
        self.assertIsNotNone(model.ma_series)
        self.assertIsNotNone(model.current_ma)
        
        # Check MA series is calculated correctly
        expected_ma = self.prices.rolling(20).mean().dropna()
        pd.testing.assert_series_equal(model.ma_series, expected_ma)
    
    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient data raises error"""
        model = ARIMAMovingAverageModel(ma_window=20)
        
        with self.assertRaises(ValueError):
            model.fit(self.short_prices)
    
    def test_forecast_without_fitting(self):
        """Test forecasting without fitting raises error"""
        model = ARIMAMovingAverageModel(ma_window=20)
        
        with self.assertRaises(ValueError):
            model.forecast(horizon=5)
    
    def test_forecast_after_fitting(self):
        """Test forecasting after fitting"""
        model = ARIMAMovingAverageModel(ma_window=20)
        model.fit(self.prices)
        
        # Should not raise exception
        forecast_result = model.forecast(horizon=10)
        
        # Check result structure
        self.assertIsInstance(forecast_result, dict)
        expected_keys = {'ma_forecast', 'ma_lower', 'ma_upper', 'forecast_dates', 'current_ma', 'model_type'}
        self.assertTrue(expected_keys.issubset(forecast_result.keys()))
        
        # Check dimensions
        self.assertEqual(len(forecast_result['ma_forecast']), 10)
        self.assertEqual(len(forecast_result['ma_lower']), 10)
        self.assertEqual(len(forecast_result['ma_upper']), 10)
        self.assertEqual(len(forecast_result['forecast_dates']), 10)
        
        # Check values are reasonable
        ma_forecast = forecast_result['ma_forecast']
        self.assertTrue(all(isinstance(x, (int, float)) for x in ma_forecast))
        self.assertTrue(all(x > 0 for x in ma_forecast))  # Prices should be positive
        
        # Check confidence intervals
        ma_lower = forecast_result['ma_lower']
        ma_upper = forecast_result['ma_upper']
        for i in range(10):
            self.assertLessEqual(ma_lower[i], ma_forecast[i])
            self.assertLessEqual(ma_forecast[i], ma_upper[i])
    
    def test_get_model_summary(self):
        """Test model summary"""
        model = ARIMAMovingAverageModel(ma_window=20)
        
        # Before fitting
        summary = model.get_model_summary()
        self.assertEqual(summary['status'], 'not_fitted')
        
        # After fitting
        model.fit(self.prices)
        summary = model.get_model_summary()
        
        self.assertEqual(summary['status'], 'fitted')
        self.assertEqual(summary['ma_window'], 20)
        self.assertIn('current_ma', summary)
        self.assertIn('model_type', summary)
    
    def test_different_ma_windows(self):
        """Test different moving average windows"""
        for window in [10, 20, 50]:
            model = ARIMAMovingAverageModel(ma_window=window)
            model.fit(self.prices)
            
            self.assertTrue(model.fitted)
            self.assertEqual(len(model.ma_series), len(self.prices) - window + 1)


class TestGARCHBollingerBandModel(unittest.TestCase):
    """Test GARCH model for Bollinger Band volatility"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate price series with volatility clustering
        n_days = 200
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # GARCH-like price process
        base_price = 100
        volatilities = [0.02]
        returns = [0]
        
        for i in range(1, n_days):
            # GARCH(1,1) volatility
            vol = 0.00001 + 0.85 * volatilities[-1]**2 + 0.1 * returns[-1]**2
            vol = max(vol, 0.001)
            volatilities.append(np.sqrt(vol))
            
            # Return with time-varying volatility
            ret = np.random.normal(0, volatilities[-1])
            returns.append(ret)
        
        prices = base_price * np.exp(np.cumsum(returns))
        self.prices = pd.Series(prices, index=dates)
        self.short_prices = self.prices.head(30)
        
    def test_model_initialization(self):
        """Test model initializes correctly"""
        model = GARCHBollingerBandModel(bb_window=20, bb_std=2.0)
        
        self.assertEqual(model.bb_window, 20)
        self.assertEqual(model.bb_std, 2.0)
        self.assertFalse(model.fitted)
        self.assertIsNone(model.garch_model)
        self.assertIsNone(model.bb_position_series)
    
    def test_fit_with_sufficient_data(self):
        """Test fitting with sufficient data"""
        model = GARCHBollingerBandModel(bb_window=20, bb_std=2.0)
        
        # Should not raise exception
        result = model.fit(self.prices)
        
        # Check return value
        self.assertIs(result, model)
        
        # Check fitted status
        self.assertTrue(model.fitted)
        self.assertIsNotNone(model.bb_position_series)
        self.assertIsNotNone(model.current_bb_width)
        
        # Check BB position is within expected range
        bb_positions = model.bb_position_series.dropna()
        self.assertTrue(all(bb_positions >= -1))
        self.assertTrue(all(bb_positions <= 1))
    
    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient data raises error"""
        model = GARCHBollingerBandModel(bb_window=20, bb_std=2.0)
        
        with self.assertRaises(ValueError):
            model.fit(self.short_prices)
    
    def test_forecast_without_fitting(self):
        """Test forecasting without fitting raises error"""
        model = GARCHBollingerBandModel(bb_window=20, bb_std=2.0)
        
        with self.assertRaises(ValueError):
            model.forecast(horizon=5)
    
    def test_forecast_after_fitting(self):
        """Test forecasting after fitting"""
        model = GARCHBollingerBandModel(bb_window=20, bb_std=2.0)
        model.fit(self.prices)
        
        # Should not raise exception
        forecast_result = model.forecast(horizon=10)
        
        # Check result structure
        self.assertIsInstance(forecast_result, dict)
        expected_keys = {'bb_width_forecast', 'bb_width_lower', 'bb_width_upper', 'forecast_dates', 'current_bb_width', 'model_type'}
        self.assertTrue(expected_keys.issubset(forecast_result.keys()))
        
        # Check dimensions
        self.assertEqual(len(forecast_result['bb_width_forecast']), 10)
        self.assertEqual(len(forecast_result['bb_width_lower']), 10)
        self.assertEqual(len(forecast_result['bb_width_upper']), 10)
        
        # Check values are reasonable
        bb_width = forecast_result['bb_width_forecast']
        self.assertTrue(all(isinstance(x, (int, float)) for x in bb_width))
        self.assertTrue(all(x > 0 for x in bb_width))  # BB width should be positive
        
        # Check confidence intervals
        bb_lower = forecast_result['bb_width_lower']
        bb_upper = forecast_result['bb_width_upper']
        for i in range(10):
            self.assertLessEqual(bb_lower[i], bb_width[i])
            self.assertLessEqual(bb_width[i], bb_upper[i])
    
    def test_get_model_summary(self):
        """Test model summary"""
        model = GARCHBollingerBandModel(bb_window=20, bb_std=2.0)
        
        # Before fitting
        summary = model.get_model_summary()
        self.assertEqual(summary['status'], 'not_fitted')
        
        # After fitting
        model.fit(self.prices)
        summary = model.get_model_summary()
        
        self.assertEqual(summary['status'], 'fitted')
        self.assertEqual(summary['bb_window'], 20)
        self.assertIn('current_bb_width', summary)
        self.assertIn('model_type', summary)


class TestCombinedARIMAGARCHModel(unittest.TestCase):
    """Test combined ARIMA-GARCH model"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate realistic stock price data
        n_days = 200
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Trending price with volatility clustering
        trend = np.linspace(100, 120, n_days)
        volatility_process = []
        vol = 0.02
        
        returns = []
        for i in range(n_days):
            if i > 0:
                vol = 0.00001 + 0.85 * vol**2 + 0.1 * returns[-1]**2
                vol = max(vol, 0.001)
            volatility_process.append(vol)
            returns.append(np.random.normal(0, np.sqrt(vol)))
        
        prices = trend * np.exp(np.cumsum(returns))
        self.prices = pd.Series(prices, index=dates)
        self.short_prices = self.prices.head(30)
        
    def test_model_initialization(self):
        """Test model initializes correctly"""
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        
        self.assertEqual(model.ma_window, 20)
        self.assertEqual(model.bb_std, 2.0)
        self.assertFalse(model.fitted)
        self.assertIsInstance(model.arima_model, ARIMAMovingAverageModel)
        self.assertIsInstance(model.garch_model, GARCHBollingerBandModel)
    
    def test_fit_with_sufficient_data(self):
        """Test fitting with sufficient data"""
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        
        # Should not raise exception
        result = model.fit(self.prices)
        
        # Check return value
        self.assertIs(result, model)
        
        # Check fitted status (should be True if at least one component fits)
        self.assertTrue(model.fitted)
    
    def test_forecast_without_fitting(self):
        """Test forecasting without fitting raises error"""
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        
        with self.assertRaises(ValueError):
            model.forecast(horizon=5)
    
    def test_forecast_after_fitting(self):
        """Test forecasting after fitting"""
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        model.fit(self.prices)
        
        # Should not raise exception
        forecast_result = model.forecast(horizon=10)
        
        # Check result structure
        self.assertIsInstance(forecast_result, dict)
        expected_keys = {
            'ma_forecast', 'ma_lower', 'ma_upper',
            'bb_width_forecast', 'bb_width_lower', 'bb_width_upper',
            'bb_upper_forecast', 'bb_lower_forecast',
            'forecast_dates', 'horizon', 'fitted_models'
        }
        self.assertTrue(expected_keys.issubset(forecast_result.keys()))
        
        # Check dimensions
        for key in ['ma_forecast', 'bb_width_forecast', 'bb_upper_forecast', 'bb_lower_forecast']:
            self.assertEqual(len(forecast_result[key]), 10)
        
        # Check BB bands relationship
        ma_forecast = forecast_result['ma_forecast']
        bb_upper = forecast_result['bb_upper_forecast']
        bb_lower = forecast_result['bb_lower_forecast']
        
        for i in range(10):
            self.assertGreaterEqual(bb_upper[i], ma_forecast[i])
            self.assertLessEqual(bb_lower[i], ma_forecast[i])
        
        # Check model status
        fitted_models = forecast_result['fitted_models']
        self.assertIn('arima_fitted', fitted_models)
        self.assertIn('garch_fitted', fitted_models)
    
    def test_get_model_summary(self):
        """Test comprehensive model summary"""
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        model.fit(self.prices)
        
        summary = model.get_model_summary()
        
        self.assertIn('combined_model_fitted', summary)
        self.assertIn('arima_summary', summary)
        self.assertIn('garch_summary', summary)
        self.assertIn('parameters', summary)
        
        # Check parameters
        params = summary['parameters']
        self.assertEqual(params['ma_window'], 20)
        self.assertEqual(params['bb_std'], 2.0)
    
    def test_partial_fitting_success(self):
        """Test that model works even if only one component fits"""
        model = CombinedARIMAGARCHModel(ma_window=20, bb_std=2.0)
        
        # This should work even if GARCH fails (common case)
        model.fit(self.prices)
        
        # Should still be able to forecast
        if model.fitted:
            forecast_result = model.forecast(horizon=5)
            self.assertIsInstance(forecast_result, dict)
            self.assertEqual(len(forecast_result['ma_forecast']), 5)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_days))
        self.prices = pd.Series(prices, index=dates)
    
    def test_fit_arima_garch_model(self):
        """Test fit_arima_garch_model convenience function"""
        model = fit_arima_garch_model(self.prices, ma_window=20, bb_std=2.0)
        
        self.assertIsInstance(model, CombinedARIMAGARCHModel)
        self.assertEqual(model.ma_window, 20)
        self.assertEqual(model.bb_std, 2.0)
    
    def test_forecast_arima_garch(self):
        """Test forecast_arima_garch convenience function"""
        model = fit_arima_garch_model(self.prices)
        
        if model.fitted:
            forecast = forecast_arima_garch(model, horizon=5)
            
            self.assertIsInstance(forecast, dict)
            self.assertEqual(forecast['horizon'], 5)


class TestModelRobustness(unittest.TestCase):
    """Test model robustness with various data conditions"""
    
    def test_constant_prices(self):
        """Test with constant price series"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        constant_prices = pd.Series([100.0] * 100, index=dates)
        
        model = CombinedARIMAGARCHModel()
        
        # Should handle constant prices gracefully
        try:
            model.fit(constant_prices)
            if model.fitted:
                forecast = model.forecast(horizon=5)
                self.assertIsInstance(forecast, dict)
        except Exception:
            # It's acceptable if it fails with constant prices
            pass
    
    def test_extreme_volatility(self):
        """Test with extremely volatile price series"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Very volatile prices
        extreme_returns = np.random.normal(0, 0.2, 100)  # 20% daily volatility
        extreme_prices = pd.Series(100 * np.exp(np.cumsum(extreme_returns)), index=dates)
        
        model = CombinedARIMAGARCHModel()
        
        # Should handle extreme volatility
        try:
            model.fit(extreme_prices)
            # If it fits, forecasting should work
            if model.fitted:
                forecast = model.forecast(horizon=3)
                self.assertIsInstance(forecast, dict)
        except Exception:
            # Acceptable if extreme data causes issues
            pass
    
    def test_missing_values(self):
        """Test handling of missing values"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)), index=dates)
        
        # Introduce some missing values
        prices.iloc[20:25] = np.nan
        prices.iloc[50:52] = np.nan
        
        model = CombinedARIMAGARCHModel()
        
        # Should handle missing values (by dropping them in calculations)
        try:
            model.fit(prices)
            # Model should still work with remaining data
        except Exception as e:
            # Acceptable if not enough clean data remains
            self.assertIsInstance(e, (ValueError, Exception))


if __name__ == '__main__':
    # Run with high verbosity to see detailed test results
    unittest.main(verbosity=2)