import pytest
import pandas as pd
import numpy as np
from src.features.pipeline import FeaturePipeline

@pytest.fixture
def sample_data():
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'adjclose': np.random.rand(100) * 100,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    return df

def test_pipeline_initialization():
    pipeline = FeaturePipeline()
    assert pipeline.config == {}

def test_validate_input(sample_data):
    pipeline = FeaturePipeline()
    assert pipeline.validate_input(sample_data) == True

def test_validate_input_missing_col(sample_data):
    pipeline = FeaturePipeline()
    bad_df = sample_data.drop(columns=['close'])
    with pytest.raises(ValueError):
        pipeline.validate_input(bad_df)

def test_compute_daily_features(sample_data):
    pipeline = FeaturePipeline()
    result = pipeline.compute_daily_features(sample_data)
    
    # Check for expected columns
    assert 'rsi_14' in result.columns
    assert 'ma_20' in result.columns
    assert 'macd_histogram' in result.columns
    assert 'vol_regime_10_60' in result.columns

def test_resample(sample_data):
    pipeline = FeaturePipeline()
    weekly = pipeline.resample(sample_data, 'W-FRI')
    assert not weekly.empty
    assert len(weekly) < len(sample_data)
    assert weekly.index.freqstr == 'W-FRI' or weekly.index.inferred_freq == 'W-FRI'

def test_run_pipeline(sample_data):
    pipeline = FeaturePipeline()
    result = pipeline.run(sample_data, symbol="TEST")
    
    assert not result.empty
    # Check for daily features
    assert 'rsi_14' in result.columns
    # Check for weekly features
    assert 'w_rsi_14' in result.columns
    # Check for symbol column (should NOT be there, handled by runner)
    assert 'symbol' not in result.columns

def test_run_pipeline_insufficient_data():
    pipeline = FeaturePipeline()
    # Create very short dataframe
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'open': np.random.rand(5),
        'high': np.random.rand(5),
        'low': np.random.rand(5),
        'close': np.random.rand(5),
        'volume': np.random.randint(100, 1000, 5)
    }, index=dates)
    
    result = pipeline.run(df, symbol="SHORT")
    # Should still return daily features, but maybe no weekly
    assert not result.empty
    assert 'rsi_14' in result.columns
