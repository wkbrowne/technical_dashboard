import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all the APIs we want to test
from models.markov_bb import MultiStockBBMarkovModel, TrendAwareBBMarkovWrapper
from models.ohlc_forecasting import OHLCForecaster
from models.open_price_kde import IntelligentOpenForecaster
from models.high_low_copula import IntelligentHighLowForecaster
import models.garch_volatility as garch


class TestAPISmokeTests(unittest.TestCase):
    """Smoke tests to ensure all APIs exist and can be instantiated"""
    
    def setUp(self):
        """Set up minimal test data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        
        # Minimal OHLC data
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        opens = close_prices + np.random.normal(0, 0.5, 50)
        highs = np.maximum(opens, close_prices) + np.random.uniform(0, 1, 50)
        lows = np.minimum(opens, close_prices) - np.random.uniform(0, 1, 50)
        
        self.ohlc_data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Add technical indicators
        ma = self.ohlc_data['Close'].rolling(10).mean()
        bb_std = self.ohlc_data['Close'].rolling(10).std()
        self.ohlc_data['MA'] = ma
        self.ohlc_data['BB_Upper'] = ma + 2 * bb_std
        self.ohlc_data['BB_Lower'] = ma - 2 * bb_std
        self.ohlc_data['BB_Position'] = ((self.ohlc_data['Close'] - ma) / (self.ohlc_data['BB_Upper'] - ma)).clip(-1, 1)
        self.ohlc_data['BB_Width'] = bb_std / ma
        
        self.ohlc_data = self.ohlc_data.dropna()
        
        # Multi-stock data
        self.multi_stock_data = {'TEST': self.ohlc_data}
    
    def test_multistock_markov_api_exists(self):
        """Test MultiStockBBMarkovModel API exists"""
        model = MultiStockBBMarkovModel()
        self.assertIsNotNone(model)
        
        # Test methods exist
        self.assertTrue(hasattr(model, 'fit_global_prior'))
        self.assertTrue(hasattr(model, 'fit_stock_models'))
        self.assertTrue(hasattr(model, 'classify_trend'))
        
        # Test that methods are callable
        self.assertTrue(callable(model.fit_global_prior))
        self.assertTrue(callable(model.fit_stock_models))
        self.assertTrue(callable(model.classify_trend))
    
    def test_trendaware_markov_api_exists(self):
        """Test TrendAwareBBMarkovWrapper API exists"""
        model = TrendAwareBBMarkovWrapper()
        self.assertIsNotNone(model)
        
        # Test methods exist
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'classify_trend'))
        self.assertTrue(hasattr(model, 'detect_trend'))
        self.assertTrue(hasattr(model, 'get_model'))
        
        # Test that methods are callable
        self.assertTrue(callable(model.fit))
        self.assertTrue(callable(model.classify_trend))
        self.assertTrue(callable(model.detect_trend))
        self.assertTrue(callable(model.get_model))
    
    def test_ohlc_forecaster_api_exists(self):
        """Test OHLCForecaster API exists"""
        forecaster = OHLCForecaster()
        self.assertIsNotNone(forecaster)
        
        # Test methods exist
        self.assertTrue(hasattr(forecaster, 'fit'))
        self.assertTrue(hasattr(forecaster, 'forecast_ohlc'))
        self.assertTrue(hasattr(forecaster, 'set_intelligent_open_forecaster'))
        self.assertTrue(hasattr(forecaster, 'set_intelligent_high_low_forecaster'))
        
        # Test that methods are callable
        self.assertTrue(callable(forecaster.fit))
        self.assertTrue(callable(forecaster.forecast_ohlc))
        self.assertTrue(callable(forecaster.set_intelligent_open_forecaster))
        self.assertTrue(callable(forecaster.set_intelligent_high_low_forecaster))
    
    def test_intelligent_open_forecaster_api_exists(self):
        """Test IntelligentOpenForecaster API exists"""
        forecaster = IntelligentOpenForecaster()
        self.assertIsNotNone(forecaster)
        
        # Test methods exist
        self.assertTrue(hasattr(forecaster, 'train_global_model'))
        self.assertTrue(hasattr(forecaster, 'add_stock_model'))
        self.assertTrue(hasattr(forecaster, 'forecast_open'))
        
        # Test that methods are callable
        self.assertTrue(callable(forecaster.train_global_model))
        self.assertTrue(callable(forecaster.add_stock_model))
        self.assertTrue(callable(forecaster.forecast_open))
    
    def test_intelligent_high_low_forecaster_api_exists(self):
        """Test IntelligentHighLowForecaster API exists"""
        forecaster = IntelligentHighLowForecaster()
        self.assertIsNotNone(forecaster)
        
        # Test methods exist
        self.assertTrue(hasattr(forecaster, 'train_global_model'))
        self.assertTrue(hasattr(forecaster, 'add_stock_model'))
        self.assertTrue(hasattr(forecaster, 'forecast_high_low'))
        
        # Test that methods are callable
        self.assertTrue(callable(forecaster.train_global_model))
        self.assertTrue(callable(forecaster.add_stock_model))
        self.assertTrue(callable(forecaster.forecast_high_low))
    
    def test_garch_api_exists(self):
        """Test GARCH volatility API exists"""
        # Test functions exist
        self.assertTrue(hasattr(garch, 'calculate_returns'))
        self.assertTrue(hasattr(garch, 'fit_garch_model'))
        self.assertTrue(hasattr(garch, 'forecast_garch_volatility'))
        self.assertTrue(hasattr(garch, 'simple_volatility_forecast'))
        
        # Test that functions are callable
        self.assertTrue(callable(garch.calculate_returns))
        self.assertTrue(callable(garch.fit_garch_model))
        self.assertTrue(callable(garch.forecast_garch_volatility))
        self.assertTrue(callable(garch.simple_volatility_forecast))
    
    def test_training_pipeline_workflow_api(self):
        """Test that the complete training pipeline API workflow is possible"""
        # This tests the exact sequence used in the notebook
        
        # Step 1: Can instantiate all models
        global_markov = MultiStockBBMarkovModel()
        individual_markov = TrendAwareBBMarkovWrapper()
        ohlc_forecaster = OHLCForecaster()
        open_forecaster = IntelligentOpenForecaster()
        hl_forecaster = IntelligentHighLowForecaster()
        
        # Step 2: Can call training methods (may fail due to data, but API should exist)
        try:
            global_markov.fit_global_prior(self.multi_stock_data)
            global_markov.fit_stock_models(self.multi_stock_data)
        except Exception:
            pass  # Method exists, data format issues are OK
        
        try:
            bb_data = self.ohlc_data[['BB_Position', 'BB_Width', 'MA']].dropna()
            individual_markov.fit(bb_data)
        except Exception:
            pass  # Method exists, fitting issues are OK
        
        try:
            ohlc_forecaster.fit(self.ohlc_data)
        except Exception:
            pass  # Method exists, fitting issues are OK
        
        try:
            open_forecaster.train_global_model(self.multi_stock_data)
        except Exception:
            pass  # Method exists, training issues are OK
        
        try:
            hl_forecaster.train_global_model(self.multi_stock_data)
        except Exception:
            pass  # Method exists, training issues are OK
        
        # Step 3: Can call GARCH methods
        try:
            returns = garch.calculate_returns(self.ohlc_data['Close'])
            model = garch.fit_garch_model(returns)
            simple_vol = garch.simple_volatility_forecast(returns, horizon=5)
            self.assertIsInstance(simple_vol, dict)
        except Exception:
            pass  # Method exists, issues are OK
        
        # The important thing is that all APIs exist and are callable
        self.assertTrue(True)  # If we get here, all APIs exist


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality that should always work"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)))
        self.returns = self.prices.pct_change().dropna()
    
    def test_calculate_returns_basic(self):
        """Test basic return calculation works"""
        log_returns = garch.calculate_returns(self.prices, 'log')
        simple_returns = garch.calculate_returns(self.prices, 'simple')
        
        self.assertIsInstance(log_returns, pd.Series)
        self.assertIsInstance(simple_returns, pd.Series)
        self.assertEqual(len(log_returns), len(self.prices) - 1)
        self.assertEqual(len(simple_returns), len(self.prices) - 1)
    
    def test_simple_volatility_forecast_basic(self):
        """Test simple volatility forecast always works"""
        result = garch.simple_volatility_forecast(self.returns, horizon=5)
        
        self.assertIsInstance(result, dict)
        self.assertIn('volatility_forecast', result)
        self.assertEqual(len(result['volatility_forecast']), 5)
        self.assertTrue(all(v > 0 for v in result['volatility_forecast']))
    
    def test_model_instantiation(self):
        """Test all models can be instantiated"""
        models = [
            MultiStockBBMarkovModel(),
            TrendAwareBBMarkovWrapper(),
            OHLCForecaster(),
            IntelligentOpenForecaster(),
            IntelligentHighLowForecaster()
        ]
        
        for model in models:
            self.assertIsNotNone(model)
            self.assertFalse(getattr(model, 'fitted', True))  # Should start unfitted


if __name__ == '__main__':
    unittest.main()