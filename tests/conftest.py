"""
Pytest fixtures for testing feature computation modules.

This module provides reusable test fixtures including synthetic data,
temporary directories, and mock configurations.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from typing import Dict


# Register custom pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def business_day_index():
    """Create a 120-row business day index for testing."""
    return pd.bdate_range(start='2023-01-01', periods=120, freq='B')


@pytest.fixture
def sample_prices(business_day_index):
    """Create synthetic price data with realistic patterns."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate log-normal price path
    n_days = len(business_day_index)
    daily_returns = np.random.normal(0.0005, 0.015, n_days)  # ~0.125% daily drift, 1.5% volatility
    log_prices = np.cumsum(daily_returns)
    prices = 100 * np.exp(log_prices)  # Start at $100
    
    return pd.Series(prices, index=business_day_index, name='adjclose')


@pytest.fixture
def sample_volumes(business_day_index):
    """Create synthetic volume data."""
    np.random.seed(43)
    n_days = len(business_day_index)
    
    # Base volume with some trend and noise
    base_volume = 1000000
    trend = np.linspace(0.9, 1.1, n_days)
    noise = np.random.lognormal(0, 0.3, n_days)
    volumes = base_volume * trend * noise
    
    return pd.Series(volumes, index=business_day_index, name='volume')


@pytest.fixture
def sample_ohlcv_df(business_day_index, sample_prices, sample_volumes):
    """Create a complete OHLCV DataFrame for testing."""
    np.random.seed(44)
    
    # Create OHLC based on adjusted close
    adjclose = sample_prices
    
    # Simple OHLC simulation (close-to-close + intraday noise)
    daily_range_pct = np.random.uniform(0.005, 0.03, len(adjclose))  # 0.5-3% daily range
    high_offset = np.random.uniform(0.3, 1.0, len(adjclose))  # High within range
    low_offset = np.random.uniform(0.0, 0.7, len(adjclose))   # Low within range
    open_offset = np.random.uniform(0.0, 1.0, len(adjclose))  # Open within range
    
    daily_range = adjclose * daily_range_pct
    high = adjclose + daily_range * high_offset
    low = adjclose - daily_range * low_offset
    open_price = low + (high - low) * open_offset
    close = adjclose  # Assuming no dividend adjustment needed
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'adjclose': adjclose,
        'volume': sample_volumes,
    }, index=business_day_index)
    
    # Add returns
    df['ret'] = np.log(df['adjclose']).diff()
    
    return df


@pytest.fixture
def two_symbol_panel(business_day_index):
    """Create a simple two-symbol panel for cross-sectional tests."""
    np.random.seed(45)
    
    symbols = ['AAPL', 'MSFT']
    data = {}
    
    for i, sym in enumerate(symbols):
        # Different return patterns for each symbol
        daily_returns = np.random.normal(0.0003 + i*0.0002, 0.012 + i*0.003, len(business_day_index))
        log_prices = np.cumsum(daily_returns)
        prices = (100 + i*50) * np.exp(log_prices)
        data[sym] = pd.Series(prices, index=business_day_index)
    
    return pd.DataFrame(data, index=business_day_index)


@pytest.fixture
def sector_map():
    """Create a simple sector mapping for testing."""
    return {
        'AAPL': 'Technology Services',
        'MSFT': 'Technology Services',
        'JPM': 'Finance',
        'BAC': 'Finance',
        'XOM': 'Energy Minerals',
        'CVX': 'Energy Minerals',
    }


@pytest.fixture
def sector_to_etf_map():
    """Create a sector-to-ETF mapping for testing."""
    return {
        'technology services': 'XLK',
        'finance': 'XLF',
        'energy minerals': 'XLE',
    }


@pytest.fixture
def indicators_dict(sample_ohlcv_df):
    """Create a dictionary of indicators by symbol for testing."""
    symbols = ['AAPL', 'MSFT', 'SPY']
    indicators_by_symbol = {}
    
    np.random.seed(46)
    
    for i, sym in enumerate(symbols):
        # Create slightly different data for each symbol
        df = sample_ohlcv_df.copy()
        
        # Add some symbol-specific variation
        variation = 1.0 + (i - 1) * 0.1  # -10%, 0%, +10% variation
        for col in ['open', 'high', 'low', 'close', 'adjclose']:
            df[col] *= variation
        
        # Recalculate returns
        df['ret'] = np.log(df['adjclose']).diff()
        
        indicators_by_symbol[sym] = df
    
    return indicators_by_symbol


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_wide_data(business_day_index):
    """Create mock wide-format OHLCV data for testing assemble module."""
    np.random.seed(47)
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    data = {}
    for metric in ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']:
        df = pd.DataFrame(index=business_day_index, columns=symbols)
        
        for i, sym in enumerate(symbols):
            if metric == 'Volume':
                base_val = 1000000 * (1 + i * 0.5)
                values = base_val * np.random.lognormal(0, 0.2, len(business_day_index))
            else:
                base_price = 100 + i * 50
                returns = np.random.normal(0.0005, 0.015, len(business_day_index))
                log_prices = np.cumsum(returns)
                prices = base_price * np.exp(log_prices)
                
                if metric in ['High', 'Low']:
                    # Add some intraday range
                    noise = np.random.uniform(0.98 if metric == 'Low' else 1.0, 
                                            1.0 if metric == 'Low' else 1.02, 
                                            len(business_day_index))
                    values = prices * noise
                else:
                    values = prices
            
            df[sym] = values
        
        data[metric] = df
    
    return data


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Automatically suppress warnings in tests."""
    import warnings
    warnings.filterwarnings("ignore", message="RANSAC did not reach consensus")
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    warnings.filterwarnings("ignore", category=RuntimeWarning)