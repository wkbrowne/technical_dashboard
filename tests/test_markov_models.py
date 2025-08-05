import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.markov_bb import MultiStockBBMarkovModel, TrendAwareBBMarkovWrapper, BollingerBandMarkovModel


class TestMultiStockBBMarkovModel(unittest.TestCase):
    """Test MultiStockBBMarkovModel API"""
    
    def setUp(self):
        """Set up test fixtures with sample data"""
        self.model = MultiStockBBMarkovModel(n_states=5)
        
        # Create sample multi-stock data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        self.sample_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            # Generate realistic BB_Position data (-1 to 1)
            bb_pos = np.random.normal(0, 0.3, 100)
            bb_pos = np.clip(bb_pos, -1, 1)
            
            # Generate MA with trend
            ma_base = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
            
            self.sample_data[symbol] = pd.DataFrame({
                'BB_Position': bb_pos,
                'MA': ma_base,
                'BB_Width': np.random.uniform(0.01, 0.05, 100),
                'Close': ma_base + np.random.normal(0, 2, 100)
            }, index=dates)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.n_states, 5)
        self.assertFalse(self.model.fitted)
        self.assertEqual(len(self.model.trend_names), 7)
        self.assertIn('parabolic_up', self.model.trend_names)
        self.assertIn('ranging', self.model.trend_names)
    
    def test_fit_global_prior_api(self):
        """Test fit_global_prior method exists and works"""
        # Should not raise exception
        self.model.fit_global_prior(self.sample_data)
        
        # Check that global models are fitted
        for trend in self.model.trend_names:
            if trend in self.model.global_trend_models:
                # Models may or may not be fitted depending on data
                self.assertIsInstance(self.model.global_trend_models[trend], BollingerBandMarkovModel)
    
    def test_fit_stock_models_api(self):
        """Test fit_stock_models method exists and works"""
        # Fit global prior first
        self.model.fit_global_prior(self.sample_data)
        
        # Should not raise exception
        self.model.fit_stock_models(self.sample_data)
        
        # Check fitted status
        self.assertTrue(self.model.fitted)
        
        # Check stock models exist
        for symbol in self.sample_data.keys():
            self.assertIn(symbol, self.model.stock_models)
    
    def test_classify_trend(self):
        """Test trend classification works"""
        ma_series = self.sample_data['AAPL']['MA']
        trends = self.model.classify_trend(ma_series)
        
        self.assertEqual(len(trends), len(ma_series))
        self.assertTrue(all(trend in self.model.trend_names for trend in trends.dropna()))
    
    def test_missing_columns_handling(self):
        """Test handling of missing required columns"""
        # Data without required columns
        bad_data = {
            'TEST': pd.DataFrame({
                'Close': [100, 101, 102],
                'Volume': [1000, 1100, 1200]
            })
        }
        
        # Should not crash, just skip the stock
        self.model.fit_global_prior(bad_data)
        # Should complete without error even with no valid data


class TestTrendAwareBBMarkovWrapper(unittest.TestCase):
    """Test TrendAwareBBMarkovWrapper API"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = TrendAwareBBMarkovWrapper(n_states=5, slope_window=5)
        
        # Create sample BB data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Generate realistic data
        bb_pos = np.random.normal(0, 0.3, 100)
        bb_pos = np.clip(bb_pos, -1, 1)
        ma_base = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        
        self.bb_data = pd.DataFrame({
            'BB_Position': bb_pos,
            'BB_Width': np.random.uniform(0.01, 0.05, 100),
            'MA': ma_base
        }, index=dates)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(len(self.model.models), 3)
        self.assertIn('up', self.model.models)
        self.assertIn('down', self.model.models)
        self.assertIn('sideways', self.model.models)
        self.assertFalse(self.model.fitted)
    
    def test_fit_api(self):
        """Test fit method exists and works"""
        # Should not raise exception
        self.model.fit(self.bb_data)
        
        # Check fitted status
        self.assertTrue(self.model.fitted)
    
    def test_fit_required_columns(self):
        """Test fit requires correct columns"""
        # Missing required columns
        bad_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        
        with self.assertRaises(ValueError):
            self.model.fit(bad_data)
    
    def test_classify_trend(self):
        """Test trend classification"""
        trends = self.model.classify_trend(self.bb_data['MA'])
        
        self.assertEqual(len(trends), len(self.bb_data))
        valid_trends = {'up', 'down', 'sideways'}
        self.assertTrue(all(trend in valid_trends for trend in trends.dropna()))
    
    def test_detect_trend(self):
        """Test trend detection"""
        trend = self.model.detect_trend(self.bb_data['MA'])
        self.assertIn(trend, ['up', 'down', 'sideways'])
    
    def test_get_model_before_fit(self):
        """Test get_model raises error before fitting"""
        with self.assertRaises(ValueError):
            self.model.get_model('up')
    
    def test_get_model_after_fit(self):
        """Test get_model works after fitting"""
        self.model.fit(self.bb_data)
        
        up_model = self.model.get_model('up')
        self.assertIsInstance(up_model, BollingerBandMarkovModel)
        
        # Test fallback to global model
        unknown_model = self.model.get_model('unknown_trend')
        self.assertIsInstance(unknown_model, BollingerBandMarkovModel)
    
    def test_sample_bb_position(self):
        """Test BB position sampling"""
        self.model.fit(self.bb_data)
        
        # Should not raise exception
        sample = self.model.sample_bb_position(state=2, trend='up')
        self.assertIsInstance(sample, (int, float))


class TestBollingerBandMarkovModel(unittest.TestCase):
    """Test base BollingerBandMarkovModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = BollingerBandMarkovModel(n_states=5)
        
        # Create sample BB position data
        np.random.seed(42)
        self.bb_positions = pd.Series(np.random.normal(0, 0.3, 100))
        self.bb_positions = self.bb_positions.clip(-1, 1)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.n_states, 5)
        self.assertFalse(self.model.fitted)
        self.assertIsNone(self.model.transition_matrix)
    
    def test_fit_api(self):
        """Test fit method works"""
        # Should not raise exception
        self.model.fit(self.bb_positions)
        
        # Check fitted status
        self.assertTrue(self.model.fitted)
        self.assertIsNotNone(self.model.transition_matrix)
        self.assertEqual(self.model.transition_matrix.shape, (5, 5))
    
    def test_sample_bb_position_before_fit(self):
        """Test sampling raises error before fitting"""
        with self.assertRaises(RuntimeError):
            self.model.sample_bb_position(state=2)
    
    def test_sample_bb_position_after_fit(self):
        """Test sampling works after fitting"""
        self.model.fit(self.bb_positions)
        
        sample = self.model.sample_bb_position(state=2)
        self.assertIsInstance(sample, (int, float))


if __name__ == '__main__':
    unittest.main()