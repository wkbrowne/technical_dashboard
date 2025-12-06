#!/usr/bin/env python3
"""
Test script for enhanced weekly features computation with error handling and chunked processing.

This script tests the improved error handling for pandas_ta functions and memory-efficient
chunked processing for large datasets.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.weekly_enhanced import (
    safe_pandas_ta_rsi, safe_pandas_ta_macd, safe_pandas_ta_sma,
    enhanced_compute_weekly_features_single_symbol, ChunkedWeeklyProcessor,
    ChunkProcessingConfig, add_weekly_features_to_daily_enhanced
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data() -> pd.DataFrame:
    """Create synthetic daily stock data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=500, freq='B')  # 500 business days
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        prices = [base_price]
        
        for i in range(1, len(dates)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000000, 10000000)
            
            data.append({
                'symbol': symbol,
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
    
    return pd.DataFrame(data)


def create_problematic_data() -> pd.DataFrame:
    """Create data that would cause pandas_ta failures."""
    # Data with various issues
    problematic_cases = []
    
    # Case 1: Very short series
    short_data = {
        'symbol': 'SHORT',
        'date': pd.date_range('2023-01-01', periods=5),
        'close': [100, 101, 102, 103, 104]
    }
    for i, (date, close) in enumerate(zip(short_data['date'], short_data['close'])):
        problematic_cases.append({
            'symbol': short_data['symbol'],
            'date': date,
            'open': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': 1000000
        })
    
    # Case 2: Constant prices (would cause division by zero)
    constant_price = 100.0
    for i in range(50):
        date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
        problematic_cases.append({
            'symbol': 'CONSTANT',
            'date': date,
            'open': constant_price,
            'high': constant_price,
            'low': constant_price,
            'close': constant_price,
            'volume': 1000000
        })
    
    # Case 3: Data with many NaNs
    for i in range(100):
        date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
        close = 100 + np.random.normal(0, 5) if i % 3 == 0 else np.nan  # 2/3 NaN values
        problematic_cases.append({
            'symbol': 'SPARSE',
            'date': date,
            'open': close,
            'high': close * 1.01 if pd.notna(close) else np.nan,
            'low': close * 0.99 if pd.notna(close) else np.nan,
            'close': close,
            'volume': 1000000 if pd.notna(close) else np.nan
        })
    
    return pd.DataFrame(problematic_cases)


def test_safe_pandas_ta_functions():
    """Test the safe wrapper functions for pandas_ta."""
    logger.info("Testing safe pandas_ta wrapper functions...")
    
    # Test with normal data
    normal_prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 10)
    
    # Test RSI
    rsi_result = safe_pandas_ta_rsi(normal_prices, length=14)
    assert rsi_result is not None, "RSI should succeed with normal data"
    logger.info("✅ RSI computation successful with normal data")
    
    # Test MACD
    macd_hist, macd_signal = safe_pandas_ta_macd(normal_prices, fast=12, slow=26, signal=9)
    assert macd_hist is not None, "MACD should succeed with normal data"
    logger.info("✅ MACD computation successful with normal data")
    
    # Test SMA
    sma_result = safe_pandas_ta_sma(normal_prices, length=20)
    assert sma_result is not None, "SMA should succeed with normal data"
    logger.info("✅ SMA computation successful with normal data")
    
    # Test with problematic data
    # Empty series
    empty_series = pd.Series([], dtype='float64')
    assert safe_pandas_ta_rsi(empty_series, 14) is None, "Should handle empty series"
    logger.info("✅ Empty series handled correctly")
    
    # Constant series
    constant_series = pd.Series([100.0] * 50)
    rsi_constant = safe_pandas_ta_rsi(constant_series, 14)
    assert rsi_constant is not None, "Should handle constant series"
    # Constant prices should result in neutral RSI around 50
    assert abs(rsi_constant.dropna().iloc[-1] - 50.0) < 1.0, "Constant prices should give neutral RSI"
    logger.info("✅ Constant series handled correctly")
    
    # Very short series
    short_series = pd.Series([100, 101, 102])
    assert safe_pandas_ta_rsi(short_series, 14) is None, "Should handle insufficient data"
    logger.info("✅ Short series handled correctly")
    
    logger.info("All safe pandas_ta function tests passed!")


def test_enhanced_weekly_features():
    """Test enhanced weekly feature computation."""
    logger.info("Testing enhanced weekly feature computation...")
    
    # Create test weekly data
    weekly_data = pd.DataFrame({
        'week_end': pd.date_range('2023-01-06', periods=50, freq='W-FRI'),
        'symbol': 'TEST',
        'open': np.random.uniform(90, 110, 50),
        'high': np.random.uniform(100, 120, 50),
        'low': np.random.uniform(80, 100, 50),
        'close': np.random.uniform(95, 105, 50),
        'volume': np.random.uniform(1000000, 10000000, 50)
    })
    
    # Test with normal data
    result = enhanced_compute_weekly_features_single_symbol(weekly_data, 'TEST')
    assert not result.empty, "Should produce results with normal data"
    
    # Check that some features were added
    original_cols = set(weekly_data.columns)
    new_cols = set(result.columns) - original_cols
    assert len(new_cols) > 0, "Should add new feature columns"
    logger.info(f"✅ Added {len(new_cols)} feature columns: {list(new_cols)[:5]}...")
    
    # Test with problematic data
    problematic_weekly = pd.DataFrame({
        'week_end': pd.date_range('2023-01-06', periods=5, freq='W-FRI'),
        'symbol': 'PROBLEM',
        'close': [100.0] * 5  # Constant prices
    })
    
    result_problem = enhanced_compute_weekly_features_single_symbol(problematic_weekly, 'PROBLEM')
    # Should not crash and should return some result
    assert not result_problem.empty, "Should handle problematic data gracefully"
    logger.info("✅ Problematic data handled gracefully")
    
    logger.info("Enhanced weekly feature tests passed!")


def test_chunked_processor():
    """Test the chunked weekly processor."""
    logger.info("Testing chunked weekly processor...")
    
    # Create test data
    test_data = create_test_data()
    
    # Test with small chunk size to force chunking
    config = ChunkProcessingConfig(
        chunk_size=2,  # Force chunking with only 2 symbols per chunk
        n_jobs=2,
        memory_limit_mb=1  # Force chunking due to "memory limit"
    )
    
    processor = ChunkedWeeklyProcessor(config)
    weekly_results = processor.process_weekly_features_chunked(test_data)
    
    assert len(weekly_results) > 0, "Should produce weekly results"
    logger.info(f"✅ Processed {len(weekly_results)} symbols with chunking")
    
    # Check processing summary
    summary = processor.get_processing_summary()
    assert summary['total_symbols'] == test_data['symbol'].nunique(), "Should track all symbols"
    assert summary['success_rate'] > 0, "Should have some successful processing"
    logger.info(f"✅ Processing summary: {summary['success_rate']:.1f}% success rate")
    
    # Test with problematic data
    problematic_data = create_problematic_data()
    processor_problem = ChunkedWeeklyProcessor(config)
    problem_results = processor_problem.process_weekly_features_chunked(problematic_data)
    
    # Should handle problematic data without crashing
    summary_problem = processor_problem.get_processing_summary()
    logger.info(f"✅ Handled problematic data: {summary_problem['success_rate']:.1f}% success rate")
    
    logger.info("Chunked processor tests passed!")


def test_full_enhanced_pipeline():
    """Test the full enhanced weekly features pipeline."""
    logger.info("Testing full enhanced weekly features pipeline...")
    
    # Create test data
    test_data = create_test_data()
    
    # Test enhanced pipeline
    config = ChunkProcessingConfig(
        chunk_size=3,
        n_jobs=2,
        memory_limit_mb=10
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress pandas warnings during testing
        
        result = add_weekly_features_to_daily_enhanced(
            test_data,
            spy_symbol="AAPL",  # Use AAPL as benchmark since SPY not in test data
            config=config
        )
    
    assert not result.empty, "Should produce results"
    assert len(result) == len(test_data), "Should maintain same number of rows"
    
    # Check for weekly features
    weekly_features = [col for col in result.columns if col.startswith('w_')]
    assert len(weekly_features) > 0, "Should add weekly features"
    logger.info(f"✅ Added {len(weekly_features)} weekly features to daily data")
    
    # Test data integrity
    original_symbols = set(test_data['symbol'].unique())
    result_symbols = set(result['symbol'].unique())
    assert original_symbols == result_symbols, "Should preserve all symbols"
    
    logger.info("Full enhanced pipeline tests passed!")


def performance_comparison():
    """Compare performance between standard and enhanced implementations."""
    logger.info("Running performance comparison...")
    
    # Create larger test dataset
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='B')
    symbols = [f'SYM_{i}' for i in range(20)]  # 20 symbols
    
    large_data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        for i, date in enumerate(dates):
            price = base_price * (1 + np.random.normal(0, 0.02) * i * 0.001)
            large_data.append({
                'symbol': symbol,
                'date': date,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': np.random.uniform(1000000, 10000000)
            })
    
    large_df = pd.DataFrame(large_data)
    logger.info(f"Created test dataset: {len(large_df)} rows, {large_df['symbol'].nunique()} symbols")
    
    # Test enhanced version
    import time
    config = ChunkProcessingConfig(chunk_size=5, n_jobs=2)
    
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enhanced_result = add_weekly_features_to_daily_enhanced(
            large_df, spy_symbol=symbols[0], config=config
        )
    enhanced_time = time.time() - start_time
    
    weekly_features_count = len([col for col in enhanced_result.columns if col.startswith('w_')])
    logger.info(f"✅ Enhanced version: {enhanced_time:.2f}s, {weekly_features_count} weekly features")
    
    logger.info("Performance comparison completed!")


def main():
    """Run all enhanced weekly features tests."""
    logger.info("Starting enhanced weekly features test suite")
    
    try:
        # Run individual test components
        test_safe_pandas_ta_functions()
        test_enhanced_weekly_features()
        test_chunked_processor()
        test_full_enhanced_pipeline()
        performance_comparison()
        
        logger.info("🎉 All enhanced weekly features tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()