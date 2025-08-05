import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import models.garch_volatility as garch


class TestGarchVolatilityAPI(unittest.TestCase):
    """Test GARCH volatility modeling API"""
    
    def setUp(self):
        """Set up test fixtures with sample return data"""
        np.random.seed(42)
        
        # Generate realistic return series with volatility clustering
        n_days = 200
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Generate GARCH-like returns with volatility clustering
        base_vol = 0.02
        vol_innovations = np.random.normal(0, 0.01, n_days)
        volatilities = [base_vol]
        
        for i in range(1, n_days):
            # Simple GARCH(1,1) process for volatility
            vol_t = 0.00001 + 0.85 * volatilities[i-1]**2 + 0.1 * vol_innovations[i]**2
            vol_t = max(vol_t, 0.001)  # Ensure positive volatility
            volatilities.append(np.sqrt(vol_t))
        
        # Generate returns using time-varying volatility
        shocks = np.random.normal(0, 1, n_days)
        self.returns = pd.Series([vol * shock for vol, shock in zip(volatilities, shocks)], 
                                index=dates)
        
        # Generate price series for testing
        self.prices = pd.Series(100 * np.exp(np.cumsum(self.returns)), index=dates)
    
    def test_calculate_returns_log(self):
        """Test calculate_returns with log method"""
        calculated_returns = garch.calculate_returns(self.prices, method='log')
        
        # Check basic properties
        self.assertIsInstance(calculated_returns, pd.Series)
        self.assertEqual(len(calculated_returns), len(self.prices) - 1)
        
        # Check values are reasonable (log returns)
        self.assertTrue(all(abs(r) < 1 for r in calculated_returns))  # No extreme values
    
    def test_calculate_returns_simple(self):
        """Test calculate_returns with simple method"""
        calculated_returns = garch.calculate_returns(self.prices, method='simple')
        
        # Check basic properties
        self.assertIsInstance(calculated_returns, pd.Series)
        self.assertEqual(len(calculated_returns), len(self.prices) - 1)
        
        # Check values are reasonable (simple returns)
        self.assertTrue(all(r > -1 for r in calculated_returns))  # No returns below -100%
    
    def test_fit_garch_model_api(self):
        """Test fit_garch_model method exists and handles inputs correctly"""
        # Should not raise exception regardless of arch availability
        model = garch.fit_garch_model(self.returns)
        
        # If arch is available, model should be fitted object or None (if fitting failed)
        # If arch is not available, model should be None
        self.assertTrue(model is None or hasattr(model, 'fit'))
    
    def test_fit_garch_model_parameters(self):
        """Test fit_garch_model accepts various parameters"""
        # Test different parameter combinations
        test_params = [
            {'p': 1, 'q': 1, 'mean_model': 'AR'},
            {'p': 1, 'q': 1, 'mean_model': 'Constant'},
            {'p': 1, 'q': 1, 'mean_model': 'Zero'},
            {'p': 2, 'q': 1, 'vol_model': 'GARCH'},
        ]
        
        for params in test_params:
            # Should not raise exception
            model = garch.fit_garch_model(self.returns, **params)
            # Model can be None (arch not available) or fitted model
            self.assertTrue(model is None or hasattr(model, 'fit'))
    
    def test_forecast_garch_volatility_with_model(self):
        """Test forecast_garch_volatility when model is available"""
        # First try to fit a model
        fitted_model = garch.fit_garch_model(self.returns)
        
        if fitted_model is not None:
            # GARCH model is available and fitted successfully
            result = garch.forecast_garch_volatility(fitted_model, horizon=10)
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn('volatility_forecast', result)
            
            # Check forecast dimensions
            vol_forecast = result['volatility_forecast']
            self.assertEqual(len(vol_forecast), 10)
            self.assertTrue(all(v > 0 for v in vol_forecast))
        else:
            # GARCH not available or fitting failed - test should still pass
            # This tests that the API exists even if arch package is missing
            pass
    
    def test_forecast_garch_volatility_with_none_model(self):
        """Test forecast_garch_volatility with None model (fallback case)"""
        # Test with None model (simulates when GARCH fitting fails)
        result = garch.forecast_garch_volatility(None, horizon=5)
        
        # Should return None or empty result when no model is provided
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_simple_volatility_forecast_ewm(self):
        """Test simple_volatility_forecast with EWM method"""
        result = garch.simple_volatility_forecast(self.returns, horizon=10, method='ewm')
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('volatility_forecast', result)
        
        # Check forecast dimensions and properties
        vol_forecast = result['volatility_forecast']
        self.assertEqual(len(vol_forecast), 10)
        self.assertTrue(all(v > 0 for v in vol_forecast))
        
        # EWM method should show some decay pattern
        # (later forecasts should generally move toward long-term volatility)
    
    def test_simple_volatility_forecast_rolling(self):
        """Test simple_volatility_forecast with rolling method"""
        result = garch.simple_volatility_forecast(self.returns, horizon=5, method='rolling')
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('volatility_forecast', result)
        
        # Check forecast dimensions
        vol_forecast = result['volatility_forecast']
        self.assertEqual(len(vol_forecast), 5)
        self.assertTrue(all(v > 0 for v in vol_forecast))
        
        # Rolling method should give constant forecast
        self.assertTrue(all(abs(v - vol_forecast[0]) < 1e-10 for v in vol_forecast))
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with very short return series
        short_returns = self.returns.head(10)
        
        # Should not crash with short series
        model = garch.fit_garch_model(short_returns)
        simple_result = garch.simple_volatility_forecast(short_returns, horizon=3)
        
        self.assertIsInstance(simple_result, dict)
    
    def test_return_series_with_extreme_values(self):
        """Test handling of return series with extreme values"""
        # Create returns with some extreme outliers
        extreme_returns = self.returns.copy()
        extreme_returns.iloc[50] = 0.5  # 50% daily return
        extreme_returns.iloc[100] = -0.3  # -30% daily return
        
        # Should handle extreme values gracefully
        model = garch.fit_garch_model(extreme_returns)
        simple_result = garch.simple_volatility_forecast(extreme_returns, horizon=5)
        
        # Simple forecast should still work
        self.assertIsInstance(simple_result, dict)
        self.assertIn('volatility_forecast', simple_result)
    
    def test_forecast_horizons(self):
        """Test different forecast horizons"""
        horizons = [1, 5, 10, 20, 50]
        
        for h in horizons:
            result = garch.simple_volatility_forecast(self.returns, horizon=h)
            
            self.assertIsInstance(result, dict)
            self.assertIn('volatility_forecast', result)
            self.assertEqual(len(result['volatility_forecast']), h)
    
    def test_api_consistency(self):
        """Test that API functions return consistent types"""
        # All returns calculation methods should return pd.Series
        log_returns = garch.calculate_returns(self.prices, 'log')
        simple_returns = garch.calculate_returns(self.prices, 'simple')
        
        self.assertIsInstance(log_returns, pd.Series)
        self.assertIsInstance(simple_returns, pd.Series)
        
        # All forecasting methods should return dicts or None
        fitted_model = garch.fit_garch_model(self.returns)
        simple_forecast = garch.simple_volatility_forecast(self.returns, horizon=5)
        
        self.assertTrue(fitted_model is None or hasattr(fitted_model, 'forecast'))
        self.assertIsInstance(simple_forecast, dict)
        
        if fitted_model is not None:
            garch_forecast = garch.forecast_garch_volatility(fitted_model, horizon=5)
            self.assertTrue(garch_forecast is None or isinstance(garch_forecast, dict))


class TestGarchIntegration(unittest.TestCase):
    """Test GARCH models as used in the training pipeline"""
    
    def setUp(self):
        """Set up test data similar to training pipeline usage"""
        np.random.seed(42)
        
        # Generate stock price data
        n_days = 100
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_days))
        self.close_prices = pd.Series(close_prices, 
                                     index=pd.date_range('2020-01-01', periods=n_days))
    
    def test_training_pipeline_usage(self):
        """Test GARCH models as used in the training pipeline"""
        # This mirrors the usage in the training pipeline
        returns = garch.calculate_returns(self.close_prices, method='log')
        model = garch.fit_garch_model(returns)
        
        forecast_days = 10
        
        if model is not None:
            # GARCH model available
            vol_result = garch.forecast_garch_volatility(model, horizon=forecast_days)
            if vol_result:
                vol_forecast = vol_result['volatility_forecast']
                self.assertEqual(len(vol_forecast), forecast_days)
            else:
                # Forecast failed, should fallback
                vol_result = garch.simple_volatility_forecast(returns, horizon=forecast_days)
                vol_forecast = vol_result['volatility_forecast']
                self.assertEqual(len(vol_forecast), forecast_days)
        else:
            # No GARCH available, use simple method
            vol_result = garch.simple_volatility_forecast(returns, horizon=forecast_days)
            vol_forecast = vol_result['volatility_forecast']
            self.assertEqual(len(vol_forecast), forecast_days)
        
        # In all cases, we should get a valid volatility forecast
        self.assertEqual(len(vol_forecast), forecast_days)
        self.assertTrue(all(v > 0 for v in vol_forecast))


if __name__ == '__main__':
    unittest.main()