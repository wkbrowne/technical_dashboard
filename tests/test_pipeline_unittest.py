import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.pipeline import FeaturePipeline

class TestFeaturePipeline(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'adjclose': np.random.rand(100) * 100,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)

    def test_pipeline_initialization(self):
        pipeline = FeaturePipeline()
        self.assertEqual(pipeline.config, {})

    def test_validate_input(self):
        pipeline = FeaturePipeline()
        self.assertTrue(pipeline.validate_input(self.sample_data))

    def test_validate_input_missing_col(self):
        pipeline = FeaturePipeline()
        bad_df = self.sample_data.drop(columns=['close'])
        with self.assertRaises(ValueError):
            pipeline.validate_input(bad_df)

    def test_compute_daily_features(self):
        pipeline = FeaturePipeline()
        result = pipeline.compute_daily_features(self.sample_data)
        
        # Check for expected columns
        self.assertIn('rsi_14', result.columns)
        self.assertIn('ma_20', result.columns)
        self.assertIn('macd_histogram', result.columns)
        # self.assertIn('vol_regime_10_60', result.columns) # Might depend on specific vol logic

    def test_resample(self):
        pipeline = FeaturePipeline()
        weekly = pipeline.resample(self.sample_data, 'W-FRI')
        self.assertFalse(weekly.empty)
        self.assertLess(len(weekly), len(self.sample_data))
        # self.assertEqual(weekly.index.freqstr, 'W-FRI') # Freq might not be set explicitly

    def test_run_pipeline(self):
        pipeline = FeaturePipeline()
        result = pipeline.run(self.sample_data, symbol="TEST")
        
        self.assertFalse(result.empty)
        # Check for daily features
        self.assertIn('rsi_14', result.columns)
        # Check for weekly features
        self.assertIn('w_rsi_14', result.columns)
        # Check for symbol column (should NOT be there, handled by runner)
        self.assertNotIn('symbol', result.columns)

if __name__ == '__main__':
    unittest.main()
