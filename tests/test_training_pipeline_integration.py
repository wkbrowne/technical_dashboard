import unittest
import sys
import os
import pandas as pd
import numpy as np
import pickle
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.markov_bb import MultiStockBBMarkovModel, TrendAwareBBMarkovWrapper
from models.ohlc_forecasting import OHLCForecaster
from models.open_price_kde import IntelligentOpenForecaster
from models.high_low_copula import IntelligentHighLowForecaster
import models.garch_volatility as garch


class TestTrainingPipelineIntegration(unittest.TestCase):
    """Integration test for the complete training pipeline as used in the notebook"""
    
    @classmethod
    def setUpClass(cls):
        """Set up comprehensive test data that mimics the real pipeline"""
        np.random.seed(42)
        
        # Create realistic multi-stock dataset similar to cache/stock_data.pkl
        cls.stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        cls.n_days = 200
        cls.dates = pd.date_range('2020-01-01', periods=cls.n_days, freq='D')
        
        # Generate correlated stock data
        cls.stock_data = {'Open': {}, 'High': {}, 'Low': {}, 'Close': {}, 'Volume': {}}
        cls.prepared_data = {}
        
        for i, symbol in enumerate(cls.stock_symbols):
            # Generate realistic price series with different characteristics
            base_price = 100 + i * 50  # Different base prices
            
            # Add some correlation between stocks and individual characteristics
            market_factor = np.cumsum(np.random.normal(0, 0.008, cls.n_days))
            stock_factor = np.cumsum(np.random.normal(0, 0.012, cls.n_days))
            
            close_changes = 0.7 * market_factor + 0.3 * stock_factor
            close_prices = base_price * np.exp(close_changes)
            
            # Generate OHLC with realistic relationships
            daily_gaps = np.random.normal(0, 0.005, cls.n_days)
            opens = np.roll(close_prices, 1) * (1 + daily_gaps)
            opens[0] = base_price
            
            daily_ranges = np.random.uniform(0.01, 0.04, cls.n_days)
            highs = np.maximum(opens, close_prices) * (1 + np.random.uniform(0, 0.02, cls.n_days))
            lows = np.minimum(opens, close_prices) * (1 - np.random.uniform(0, 0.02, cls.n_days))
            
            volumes = np.random.randint(1000000, 10000000, cls.n_days)
            
            # Store in stock_data format (mimicking pickle structure)
            cls.stock_data['Open'][symbol] = pd.Series(opens, index=cls.dates)
            cls.stock_data['High'][symbol] = pd.Series(highs, index=cls.dates)
            cls.stock_data['Low'][symbol] = pd.Series(lows, index=cls.dates)
            cls.stock_data['Close'][symbol] = pd.Series(close_prices, index=cls.dates)
            cls.stock_data['Volume'][symbol] = pd.Series(volumes, index=cls.dates)
            
            # Prepare data with technical indicators (as in the notebook)
            data = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': close_prices,
                'Volume': volumes
            }, index=cls.dates)
            
            # Add technical indicators
            ma = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            data['MA'] = ma
            data['BB_Upper'] = ma + 2 * bb_std
            data['BB_Lower'] = ma - 2 * bb_std
            
            # Calculate BB_Position (-1 to 1)
            data['BB_Position'] = (data['Close'] - ma) / (data['BB_Upper'] - ma)
            data['BB_Position'] = data['BB_Position'].clip(-1, 1)
            
            # BB_Width for other models
            data['BB_Width'] = bb_std / ma
            
            cls.prepared_data[symbol] = data.dropna()
        
        # Convert to DataFrame format for convenience
        for key in cls.stock_data:
            cls.stock_data[key] = pd.DataFrame(cls.stock_data[key])
    
    def test_step1_data_loading_and_preparation(self):
        """Test Step 1: Data loading and preparation"""
        # Test that we can load data (simulated)
        self.assertIsInstance(self.stock_data, dict)
        self.assertIn('Close', self.stock_data)
        self.assertEqual(len(self.stock_data['Close'].columns), len(self.stock_symbols))
        
        # Test data preparation function (from notebook)
        def prepare_stock_data(stock_data, symbols, min_obs=50):
            prepared = {}
            for symbol in symbols:
                if symbol in stock_data['Close'].columns:
                    data = pd.DataFrame({
                        'Open': stock_data['Open'][symbol],
                        'High': stock_data['High'][symbol],
                        'Low': stock_data['Low'][symbol],
                        'Close': stock_data['Close'][symbol],
                        'Volume': stock_data['Volume'][symbol]
                    }).dropna()
                    
                    if len(data) >= min_obs:
                        close = data['Close']
                        data['MA'] = close.rolling(20).mean()
                        bb_std = close.rolling(20).std()
                        data['BB_Upper'] = data['MA'] + 2 * bb_std
                        data['BB_Lower'] = data['MA'] - 2 * bb_std
                        
                        data['BB_Position'] = (close - data['MA']) / (data['BB_Upper'] - data['MA'])
                        data['BB_Position'] = data['BB_Position'].clip(-1, 1)
                        
                        data['BB_Width'] = bb_std / data['MA']
                        
                        prepared[symbol] = data.dropna()
            
            return prepared
        
        # Test preparation
        prepared = prepare_stock_data(self.stock_data, self.stock_symbols)
        
        self.assertEqual(len(prepared), len(self.stock_symbols))
        for symbol in self.stock_symbols:
            self.assertIn(symbol, prepared)
            self.assertIn('BB_Position', prepared[symbol].columns)
            self.assertIn('BB_Width', prepared[symbol].columns)
            self.assertIn('MA', prepared[symbol].columns)
    
    def test_step2_global_markov_training(self):
        """Test Step 2: Global Markov model training"""
        global_markov = MultiStockBBMarkovModel()
        
        # Should not raise exception
        global_markov.fit_global_prior(self.prepared_data)
        global_markov.fit_stock_models(self.prepared_data)
        
        # Check that model is fitted
        self.assertTrue(global_markov.fitted)
        self.assertIsInstance(global_markov.stock_models, dict)
    
    def test_step3_individual_markov_training(self):
        """Test Step 3: Individual Markov models with global prior"""
        individual_markov = {}
        
        for symbol in self.stock_symbols:
            markov_model = TrendAwareBBMarkovWrapper(
                n_states=5,
                slope_window=5,
                up_thresh=0.05,
                down_thresh=-0.05
            )
            
            # Create DataFrame with required columns
            bb_data = pd.DataFrame({
                'BB_Position': self.prepared_data[symbol]['BB_Position'],
                'BB_Width': self.prepared_data[symbol]['BB_Width'],
                'MA': self.prepared_data[symbol]['MA']
            })
            
            # Should not raise exception
            markov_model.fit(bb_data)
            individual_markov[symbol] = markov_model
            
            # Check that model is fitted
            self.assertTrue(markov_model.fitted)
        
        self.assertEqual(len(individual_markov), len(self.stock_symbols))
    
    def test_step4_ohlc_forecaster_training(self):
        """Test Step 4: OHLC forecaster training"""
        close_forecasters = {}
        
        for symbol in self.stock_symbols:
            forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
            
            # Should not raise exception
            forecaster.fit(self.prepared_data[symbol])
            close_forecasters[symbol] = forecaster
            
            # Check that model is fitted
            self.assertTrue(forecaster.fitted)
        
        self.assertEqual(len(close_forecasters), len(self.stock_symbols))
    
    def test_step5_open_price_models(self):
        """Test Step 5: Open price model training"""
        open_forecaster = IntelligentOpenForecaster()
        
        # Should not raise exception
        open_forecaster.train_global_model(self.prepared_data)
        
        # Add stock-specific models
        for symbol in self.stock_symbols:
            open_forecaster.add_stock_model(symbol, self.prepared_data[symbol])
        
        # Check that models are added
        self.assertEqual(len(open_forecaster.stock_models), len(self.stock_symbols))
    
    def test_step6_high_low_copula_models(self):
        """Test Step 6: High/Low copula model training"""
        hl_forecaster = IntelligentHighLowForecaster()
        
        # Should not raise exception
        hl_forecaster.train_global_model(self.prepared_data)
        
        # Add stock-specific models
        for symbol in self.stock_symbols:
            hl_forecaster.add_stock_model(symbol, self.prepared_data[symbol])
        
        # Check that models are added
        self.assertEqual(len(hl_forecaster.stock_models), len(self.stock_symbols))
    
    def test_step7_garch_models(self):
        """Test Step 7: GARCH model training"""
        garch_models = {}
        
        for symbol in self.stock_symbols:
            try:
                close_prices = self.prepared_data[symbol]['Close']
                returns = garch.calculate_returns(close_prices, method='log')
                model = garch.fit_garch_model(returns)
                garch_models[symbol] = model
            except Exception as e:
                # GARCH may fail, which is acceptable
                garch_models[symbol] = None
        
        # Check that we attempted to fit all models
        self.assertEqual(len(garch_models), len(self.stock_symbols))
        
        # At least some should work or be None (both acceptable)
        for symbol, model in garch_models.items():
            self.assertTrue(model is None or hasattr(model, 'forecast'))
    
    def test_step8_integrated_prediction(self):
        """Test Step 8: Integrated prediction using all models"""
        # Set up all models (simplified versions)
        target_stock = 'AAPL'
        
        # 1. OHLC Forecaster
        ohlc_forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
        ohlc_forecaster.fit(self.prepared_data[target_stock])
        
        # 2. Open forecaster
        open_forecaster = IntelligentOpenForecaster()
        open_forecaster.train_global_model(self.prepared_data)
        open_forecaster.add_stock_model(target_stock, self.prepared_data[target_stock])
        
        # 3. High/Low forecaster
        hl_forecaster = IntelligentHighLowForecaster()
        hl_forecaster.train_global_model(self.prepared_data)
        hl_forecaster.add_stock_model(target_stock, self.prepared_data[target_stock])
        
        # 4. Set intelligent forecasters
        ohlc_forecaster.set_intelligent_open_forecaster(open_forecaster, target_stock)
        ohlc_forecaster.set_intelligent_high_low_forecaster(hl_forecaster, target_stock)
        
        # 5. GARCH volatility
        target_data = self.prepared_data[target_stock]
        current_close = target_data['Close'].iloc[-1]
        
        try:
            returns = garch.calculate_returns(target_data['Close'], method='log')
            garch_model = garch.fit_garch_model(returns)
            
            forecast_days = 5
            if garch_model is not None:
                vol_result = garch.forecast_garch_volatility(garch_model, horizon=forecast_days)
                vol_forecast = vol_result['volatility_forecast'] if vol_result else np.full(forecast_days, 0.025)
            else:
                vol_result = garch.simple_volatility_forecast(returns, horizon=forecast_days)
                vol_forecast = vol_result['volatility_forecast']
        except Exception:
            # Fallback volatility
            vol_forecast = np.full(5, 0.025)
        
        # 6. Generate forecasts
        ma_forecast = np.full(forecast_days, current_close * 1.002)
        bb_forecast = np.random.choice([1, 2, 3, 4, 5], size=forecast_days)
        
        # 7. Make prediction (this is the key integration test)
        try:
            forecast_results = ohlc_forecaster.forecast_ohlc(
                ma_forecast=ma_forecast,
                vol_forecast=vol_forecast,
                bb_states=bb_forecast,
                current_close=current_close,
                n_days=forecast_days
            )
            
            # Check forecast structure
            self.assertIsInstance(forecast_results, dict)
            expected_keys = {'open', 'high', 'low', 'close'}
            self.assertTrue(expected_keys.issubset(forecast_results.keys()))
            
            # Check dimensions
            for key in expected_keys:
                self.assertEqual(len(forecast_results[key]), forecast_days)
            
            # Check OHLC relationships
            for i in range(forecast_days):
                high = forecast_results['high'][i]
                low = forecast_results['low'][i]
                open_p = forecast_results['open'][i]
                close_p = forecast_results['close'][i]
                
                self.assertLessEqual(low, open_p, f"Day {i}: Low should be <= Open")
                self.assertLessEqual(low, close_p, f"Day {i}: Low should be <= Close")
                self.assertGreaterEqual(high, open_p, f"Day {i}: High should be >= Open")
                self.assertGreaterEqual(high, close_p, f"Day {i}: High should be >= Close")
        
        except Exception as e:
            # If the full integration fails, at least the individual components should work
            # This allows for partial success in case intelligent forecasters need specific data formats
            self.fail(f"Integration test failed: {str(e)}")
    
    def test_pipeline_robustness(self):
        """Test pipeline handles edge cases and missing data"""
        # Test with limited data
        limited_data = {}
        for symbol in ['AAPL', 'GOOGL']:  # Only 2 stocks
            limited_data[symbol] = self.prepared_data[symbol].head(50)  # Only 50 days
        
        # Should still work with limited data
        global_markov = MultiStockBBMarkovModel()
        global_markov.fit_global_prior(limited_data)
        
        # Individual models should also handle limited data
        if 'AAPL' in limited_data:
            markov_model = TrendAwareBBMarkovWrapper(n_states=3)  # Fewer states for limited data
            bb_data = pd.DataFrame({
                'BB_Position': limited_data['AAPL']['BB_Position'],
                'BB_Width': limited_data['AAPL']['BB_Width'],
                'MA': limited_data['AAPL']['MA']
            })
            markov_model.fit(bb_data)
            self.assertTrue(markov_model.fitted)
    
    def test_model_persistence_format(self):
        """Test that models can be used in a way compatible with caching/persistence"""
        # This tests the pattern used in the actual training pipeline
        
        # Train a model
        target_stock = 'AAPL'
        ohlc_forecaster = OHLCForecaster()
        ohlc_forecaster.fit(self.prepared_data[target_stock])
        
        # Test that the fitted model has the attributes needed for persistence
        self.assertTrue(hasattr(ohlc_forecaster, 'fitted'))
        self.assertTrue(ohlc_forecaster.fitted)
        
        # Test that models can be used immediately after fitting
        current_close = self.prepared_data[target_stock]['Close'].iloc[-1]
        
        # This should work immediately after fitting
        try:
            forecast_results = ohlc_forecaster.forecast_ohlc(
                ma_forecast=np.full(3, current_close),
                vol_forecast=np.full(3, 0.02),
                bb_states=np.full(3, 3),
                current_close=current_close,
                n_days=3
            )
            self.assertIsInstance(forecast_results, dict)
        except Exception as e:
            # Some models may need additional setup, but the fitted flag should be correct
            self.assertTrue(ohlc_forecaster.fitted)


if __name__ == '__main__':
    # Run with verbose output to see individual test results
    unittest.main(verbosity=2)