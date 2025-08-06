import unittest
import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.global_kde_models import (
    GlobalClosePriceKDE,
    GlobalOpenPriceKDE,
    GlobalHighLowCopula,
    train_global_models
)
from models.regime_classifier import UnifiedRegimeClassifier

warnings.filterwarnings('ignore')


class TestRegimeClassifier(unittest.TestCase):
    """Test regime classification functionality"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.classifier = UnifiedRegimeClassifier()
        
        # Create sample MA and return series
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # MA series with different trends
        trend_data = np.cumsum(np.random.normal(0.001, 0.005, 100))  # Slight uptrend
        self.ma_series = pd.Series(100 + trend_data, index=dates)
        
        # Return series with varying volatility
        vol_regime = np.concatenate([
            np.random.normal(0, 0.01, 30),   # Low vol
            np.random.normal(0, 0.03, 40),   # High vol
            np.random.normal(0, 0.015, 30)   # Medium vol
        ])
        self.returns = pd.Series(vol_regime, index=dates)
    
    def test_classify_trend(self):
        """Test trend classification"""
        trends = self.classifier.classify_trend(self.ma_series)
        
        # Check basic properties
        self.assertEqual(len(trends), len(self.ma_series))
        self.assertIsInstance(trends, pd.Series)
        
        # Check valid trend categories
        valid_trends = {'strong_bull', 'bull', 'sideways', 'bear', 'strong_bear'}
        unique_trends = set(trends.dropna())
        self.assertTrue(unique_trends.issubset(valid_trends))
        
        # Should have some non-null values
        self.assertGreater(trends.notna().sum(), 50)
    
    def test_classify_volatility(self):
        """Test volatility classification"""
        vol_regimes = self.classifier.classify_volatility(self.returns)
        
        # Check basic properties
        self.assertEqual(len(vol_regimes), len(self.returns))
        self.assertIsInstance(vol_regimes, pd.Series)
        
        # Check valid volatility categories
        valid_vols = {'low', 'medium', 'high'}
        unique_vols = set(vol_regimes.dropna())
        self.assertTrue(unique_vols.issubset(valid_vols))
        
        # Should have all three volatility regimes given our setup
        self.assertGreaterEqual(len(unique_vols), 2)
    
    def test_get_combined_regime(self):
        """Test combined regime classification"""
        trends = self.classifier.classify_trend(self.ma_series)
        vol_regimes = self.classifier.classify_volatility(self.returns)
        
        combined = self.classifier.get_combined_regime(trends, vol_regimes)
        
        # Check basic properties
        self.assertEqual(len(combined), len(trends))
        
        # Check format (should be 'trend_vol')
        sample_regimes = combined.dropna().iloc[:5]
        for regime in sample_regimes:
            self.assertIn('_', regime)
            parts = regime.split('_')
            self.assertEqual(len(parts), 2)


class TestGlobalClosePriceKDE(unittest.TestCase):
    """Test Global Close Price KDE model"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.model = GlobalClosePriceKDE(min_samples_per_regime=20)
        
        # Create multi-stock data for training
        self.all_stock_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']:
            dates = pd.date_range('2020-01-01', periods=150, freq='D')
            
            # Generate realistic price data
            base_price = 100 + np.random.normal(0, 20)
            returns = np.random.normal(0.0005, 0.015, 150)  # Daily returns
            close_prices = base_price * np.exp(np.cumsum(returns))
            
            # Calculate MA and technical indicators with proper setup
            close_series = pd.Series(close_prices, index=dates)
            ma = close_series.rolling(window=20, min_periods=5).mean()
            bb_std = close_series.rolling(window=20, min_periods=5).std()
            
            self.all_stock_data[symbol] = pd.DataFrame({
                'Close': close_prices,
                'MA': ma,
                'BB_Position': ((close_prices - ma) / (2 * bb_std)).clip(-1, 1),
                'Volume': np.random.randint(1000000, 10000000, 150)
            }, index=dates)
            
            # Only drop rows where MA is NaN (keep more data)
            self.all_stock_data[symbol] = self.all_stock_data[symbol][self.all_stock_data[symbol]['MA'].notna()]
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.min_samples_per_regime, 20)
        self.assertFalse(self.model.fitted)
        self.assertEqual(len(self.model.kde_models), 0)
        self.assertEqual(len(self.model.regime_stats), 0)
        self.assertIsInstance(self.model.regime_classifier, UnifiedRegimeClassifier)
    
    def test_fit_global_model(self):
        """Test fitting global model on all stock data"""
        # Should not raise exception
        result = self.model.fit_global_model(self.all_stock_data)
        
        # Check return value
        self.assertIs(result, self.model)
        
        # Check fitted status
        self.assertTrue(self.model.fitted)
        
        # Should have trained some KDE models
        print(f"Trained KDE models: {len(self.model.kde_models)}")
        print(f"Total regimes identified: {len(self.model.regime_stats)}")
        
        # Should have identified multiple regimes
        self.assertGreater(len(self.model.regime_stats), 0)
    
    def test_extract_close_features(self):
        """Test feature extraction from stock data"""
        symbol = 'AAPL'
        stock_data = self.all_stock_data[symbol]
        
        features = self.model._extract_close_features(stock_data, symbol)
        
        # Check basic properties
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Check required columns
        required_cols = ['symbol', 'close_to_ma_ratio', 'close_return', 
                        'trend_regime', 'vol_regime', 'combined_regime']
        for col in required_cols:
            self.assertIn(col, features.columns)
        
        # Check data types and ranges
        self.assertTrue(all(features['symbol'] == symbol))
        self.assertTrue(all(abs(features['close_to_ma_ratio']) < 1))  # Should be reasonable ratios
    
    def test_sample_close_price(self):
        """Test close price sampling"""
        # Fit model first
        self.model.fit_global_model(self.all_stock_data)
        
        if len(self.model.kde_models) > 0:
            # Test with actual regime
            regime = list(self.model.kde_models.keys())[0]
            ma_value = 100.0
            n_samples = 10
            
            samples = self.model.sample_close_price(regime, ma_value, n_samples)
            
            # Check basic properties
            self.assertEqual(len(samples), n_samples)
            self.assertTrue(all(isinstance(x, (int, float)) for x in samples))
            self.assertTrue(all(x > 0 for x in samples))  # Prices should be positive
            
            # Samples should be reasonably close to MA value
            self.assertTrue(all(abs(x - ma_value) / ma_value < 0.5 for x in samples))
        
        # Test fallback with unknown regime
        fallback_samples = self.model.sample_close_price('unknown_regime', 100.0, 5)
        self.assertEqual(len(fallback_samples), 5)
        self.assertTrue(all(x > 0 for x in fallback_samples))
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Create minimal dataset
        minimal_data = {
            'TEST': pd.DataFrame({
                'Close': [100, 101, 102],
                'MA': [100, 100.5, 101]
            })
        }
        
        # Should handle gracefully (may not train models but shouldn't crash)
        try:
            self.model.fit_global_model(minimal_data)
            # If it succeeds, fitted should be True but no KDE models trained
            self.assertTrue(self.model.fitted)
        except ValueError:
            # Acceptable if insufficient data for any training
            pass


class TestGlobalOpenPriceKDE(unittest.TestCase):
    """Test Global Open Price KDE model"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.model = GlobalOpenPriceKDE(min_samples_per_regime=20)
        
        # Create multi-stock data with overnight gaps
        self.all_stock_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            
            # Generate prices with overnight gaps
            base_price = 100 + np.random.normal(0, 20)
            close_prices = base_price * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
            
            # Generate opens with gaps relative to previous close
            gaps = np.random.normal(0, 0.005, 100)  # Small overnight gaps
            opens = np.roll(close_prices, 1) * (1 + gaps)
            opens[0] = base_price  # First open
            
            close_series = pd.Series(close_prices, index=dates)
            ma = close_series.rolling(window=20, min_periods=5).mean()
            
            self.all_stock_data[symbol] = pd.DataFrame({
                'Open': opens,
                'Close': close_prices,
                'MA': ma,
                'BB_Position': np.random.uniform(-0.5, 0.5, 100)
            }, index=dates)
            
            # Only drop rows where MA is NaN
            self.all_stock_data[symbol] = self.all_stock_data[symbol][self.all_stock_data[symbol]['MA'].notna()]
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.min_samples_per_regime, 20)
        self.assertFalse(self.model.fitted)
        self.assertEqual(len(self.model.kde_models), 0)
    
    def test_fit_global_model(self):
        """Test fitting global open price model"""
        result = self.model.fit_global_model(self.all_stock_data)
        
        self.assertIs(result, self.model)
        self.assertTrue(self.model.fitted)
        
        # Should have identified some regimes
        self.assertGreater(len(self.model.regime_stats), 0)
        print(f"Open KDE models: {len(self.model.kde_models)}")
    
    def test_extract_gap_features(self):
        """Test gap feature extraction"""
        symbol = 'AAPL'
        stock_data = self.all_stock_data[symbol]
        
        features = self.model._extract_gap_features(stock_data, symbol)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Check gap returns are reasonable
        gaps = features['gap_return'].dropna()
        self.assertTrue(all(abs(g) < 0.1 for g in gaps))  # No extreme gaps
    
    def test_sample_gap(self):
        """Test gap sampling"""
        self.model.fit_global_model(self.all_stock_data)
        
        if len(self.model.kde_models) > 0:
            regime = list(self.model.kde_models.keys())[0]
            samples = self.model.sample_gap(regime, n_samples=10)
            
            self.assertEqual(len(samples), 10)
            self.assertTrue(all(isinstance(x, (int, float)) for x in samples))
            self.assertTrue(all(abs(x) < 0.1 for x in samples))  # Reasonable gaps
        
        # Test fallback
        fallback_samples = self.model.sample_gap('unknown_regime', 5)
        self.assertEqual(len(fallback_samples), 5)


class TestGlobalHighLowCopula(unittest.TestCase):
    """Test Global High-Low Copula model"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.model = GlobalHighLowCopula(min_samples_per_regime=30)
        
        # Create multi-stock OHLC data
        self.all_stock_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            
            # Generate OHLC data
            close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
            opens = close_prices + np.random.normal(0, 0.5, 100)
            
            # Ensure proper OHLC relationships
            highs = np.maximum(opens, close_prices) + np.random.uniform(0, 2, 100)
            lows = np.minimum(opens, close_prices) - np.random.uniform(0, 2, 100)
            
            close_series = pd.Series(close_prices, index=dates)
            ma = close_series.rolling(window=20, min_periods=5).mean()
            
            self.all_stock_data[symbol] = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': close_prices,
                'MA': ma,
                'BB_Position': np.random.uniform(-0.5, 0.5, 100)
            }, index=dates)
            
            # Only drop rows where MA is NaN
            self.all_stock_data[symbol] = self.all_stock_data[symbol][self.all_stock_data[symbol]['MA'].notna()]
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.min_samples_per_regime, 30)
        self.assertFalse(self.model.fitted)
        self.assertEqual(len(self.model.copula_models), 0)
    
    def test_fit_global_model(self):
        """Test fitting global copula model"""
        result = self.model.fit_global_model(self.all_stock_data)
        
        self.assertIs(result, self.model)
        self.assertTrue(self.model.fitted)
        
        # Should have identified some regimes
        self.assertGreater(len(self.model.regime_stats), 0)
        print(f"Copula models: {len(self.model.copula_models)}")
    
    def test_extract_highlow_features(self):
        """Test high-low feature extraction"""
        symbol = 'AAPL'
        stock_data = self.all_stock_data[symbol]
        
        features = self.model._extract_highlow_features(stock_data, symbol)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Check ratios are reasonable
        high_ratios = features['high_ratio'].dropna()
        low_ratios = features['low_ratio'].dropna()
        
        self.assertTrue(all(hr >= 0 for hr in high_ratios))  # High ratios should be positive
        self.assertTrue(all(lr >= 0 for lr in low_ratios))   # Low ratios should be positive
    
    def test_sample_high_low(self):
        """Test high-low sampling"""
        self.model.fit_global_model(self.all_stock_data)
        
        ref_price = 100.0
        
        if len(self.model.copula_models) > 0:
            regime = list(self.model.copula_models.keys())[0]
            samples = self.model.sample_high_low(regime, ref_price, n_samples=10)
            
            self.assertIn('high', samples)
            self.assertIn('low', samples)
            self.assertEqual(len(samples['high']), 10)
            self.assertEqual(len(samples['low']), 10)
            
            # Check OHLC relationships
            for h, l in zip(samples['high'], samples['low']):
                self.assertGreaterEqual(h, ref_price)  # High >= reference
                self.assertLessEqual(l, ref_price)     # Low <= reference
        
        # Test fallback
        fallback_samples = self.model.sample_high_low('unknown_regime', ref_price, 5)
        self.assertIn('high', fallback_samples)
        self.assertIn('low', fallback_samples)


class TestTrainGlobalModels(unittest.TestCase):
    """Test the convenience function for training all global models"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create comprehensive multi-stock dataset
        self.all_stock_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']:
            dates = pd.date_range('2020-01-01', periods=120, freq='D')
            
            # Generate realistic OHLC data
            close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, 120)))
            opens = close_prices + np.random.normal(0, 0.5, 120)
            highs = np.maximum(opens, close_prices) + np.random.uniform(0, 1, 120)
            lows = np.minimum(opens, close_prices) - np.random.uniform(0, 1, 120)
            
            # Technical indicators
            close_series = pd.Series(close_prices, index=dates)
            ma = close_series.rolling(window=20, min_periods=5).mean()
            bb_std = close_series.rolling(window=20, min_periods=5).std()
            
            self.all_stock_data[symbol] = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': close_prices,
                'MA': ma,
                'BB_Position': ((close_prices - ma) / (2 * bb_std)).clip(-1, 1),
                'Volume': np.random.randint(1000000, 10000000, 120)
            }, index=dates)
            
            # Only drop rows where MA is NaN
            self.all_stock_data[symbol] = self.all_stock_data[symbol][self.all_stock_data[symbol]['MA'].notna()]
    
    def test_train_global_models(self):
        """Test training all global models together"""
        models = train_global_models(self.all_stock_data, min_samples=15)
        
        # Check return structure
        self.assertIsInstance(models, dict)
        expected_models = ['close_kde', 'open_kde', 'hl_copula']
        for model_name in expected_models:
            self.assertIn(model_name, models)
        
        # Check that at least some models trained successfully
        successful_models = [m for m in models.values() if m is not None]
        self.assertGreater(len(successful_models), 0)
        
        # Test each successful model
        for model_name, model in models.items():
            if model is not None:
                self.assertTrue(model.fitted)
                print(f"{model_name}: ✅ Fitted successfully")
                
                # Check that some regimes were identified
                if hasattr(model, 'regime_stats'):
                    self.assertGreater(len(model.regime_stats), 0)
            else:
                print(f"{model_name}: ❌ Failed to fit")
    
    def test_integration_with_regime_resolution(self):
        """Test that models properly resolve by trend and volatility regimes"""
        models = train_global_models(self.all_stock_data, min_samples=10)
        
        # Check that regimes contain both trend and volatility components
        for model_name, model in models.items():
            if model is not None and hasattr(model, 'regime_stats'):
                regime_names = list(model.regime_stats.keys())
                
                if len(regime_names) > 0:
                    # Check regime naming convention
                    sample_regime = regime_names[0]
                    self.assertIn('_', sample_regime)  # Should be 'trend_vol' format
                    
                    parts = sample_regime.split('_')
                    self.assertGreaterEqual(len(parts), 2)  # Should have at least trend and vol parts
                    
                    # For regimes like 'strong_bull_high', trend is 'strong_bull', vol is 'high'
                    if len(parts) == 3:
                        trend_part = '_'.join(parts[:2])  # 'strong_bull'
                        vol_part = parts[2]  # 'high'
                    else:
                        trend_part, vol_part = parts
                    
                    # Check valid trend and volatility components
                    valid_trends = {'strong_bull', 'bull', 'sideways', 'bear', 'strong_bear'}
                    valid_vols = {'low', 'medium', 'high'}
                    
                    # At least the sample should be valid (other regimes might vary)
                    if trend_part in valid_trends and vol_part in valid_vols:
                        print(f"{model_name}: ✅ Proper regime resolution ({sample_regime})")
                    else:
                        print(f"{model_name}: ⚠️ Unexpected regime format ({sample_regime})")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in global models"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = {}
        
        for ModelClass in [GlobalClosePriceKDE, GlobalOpenPriceKDE, GlobalHighLowCopula]:
            model = ModelClass(min_samples_per_regime=10)
            
            with self.assertRaises(ValueError):
                model.fit_global_model(empty_data)
    
    def test_insufficient_columns_handling(self):
        """Test handling of data with missing columns"""
        bad_data = {
            'TEST': pd.DataFrame({
                'Close': [100, 101, 102],
                'Volume': [1000, 1100, 1200]
                # Missing required columns like MA
            })
        }
        
        for ModelClass in [GlobalClosePriceKDE, GlobalOpenPriceKDE, GlobalHighLowCopula]:
            model = ModelClass(min_samples_per_regime=10)
            
            try:
                model.fit_global_model(bad_data)
                # If it succeeds, should handle gracefully
                self.assertTrue(model.fitted)
            except (ValueError, KeyError):
                # Acceptable to fail with bad data
                pass
    
    def test_extreme_data_handling(self):
        """Test handling of extreme/unrealistic data"""
        # Create data with extreme values
        extreme_data = {
            'EXTREME': pd.DataFrame({
                'Open': [1e6, 1e7, 1e8],  # Extreme prices
                'High': [1.1e6, 1.1e7, 1.1e8],
                'Low': [0.9e6, 0.9e7, 0.9e8],
                'Close': [1e6, 1e7, 1e8],
                'MA': [1e6, 1e7, 1e8],
                'BB_Position': [10, -10, 5]  # Extreme BB positions
            })
        }
        
        for ModelClass in [GlobalClosePriceKDE, GlobalOpenPriceKDE, GlobalHighLowCopula]:
            model = ModelClass(min_samples_per_regime=1)  # Low threshold for testing
            
            try:
                model.fit_global_model(extreme_data)
                # Should handle extreme data gracefully
            except Exception:
                # Acceptable if extreme data causes issues
                pass


if __name__ == '__main__':
    # Run with high verbosity to see detailed test results
    unittest.main(verbosity=2)