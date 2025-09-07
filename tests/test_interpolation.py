#!/usr/bin/env python3
"""
Test script for parallelized NaN interpolation performance.

This script creates synthetic data with NaN gaps and tests the performance
improvement from parallelization.
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.postprocessing import interpolate_internal_gaps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_symbols: int = 100, n_rows: int = 1000, nan_prob: float = 0.05) -> dict:
    """
    Create synthetic test data with random NaN gaps.
    
    Args:
        n_symbols: Number of symbols to create
        n_rows: Number of rows per symbol
        nan_prob: Probability of NaN values (internal gaps only)
        
    Returns:
        Dictionary mapping symbol -> DataFrame with NaN gaps
    """
    logger.info(f"Creating test data: {n_symbols} symbols, {n_rows} rows each")
    
    test_data = {}
    
    # Create date range
    date_range = pd.date_range(start='2020-01-01', periods=n_rows, freq='D')
    
    for i in range(n_symbols):
        symbol = f"TEST_{i:03d}"
        
        # Generate synthetic price and feature data
        np.random.seed(42 + i)  # Reproducible but different per symbol
        base_price = 100 + np.random.randn() * 20
        price_series = base_price * np.exp(np.cumsum(np.random.randn(n_rows) * 0.02))
        
        # Create DataFrame with multiple numeric columns
        price_df = pd.DataFrame({'price': price_series}, index=date_range)
        
        df = pd.DataFrame({
            'adjclose': price_series,
            'volume': np.random.exponential(1000000, n_rows),
            'feature1': np.random.randn(n_rows),
            'feature2': np.random.randn(n_rows) * 10,
            'feature3': price_df['price'].rolling(20).mean(),
            'feature4': price_df['price'].rolling(50).std(),
            'feature5': np.random.randn(n_rows).cumsum(),
        }, index=date_range)
        
        # Add random NaN gaps (avoid first/last 10% to keep as internal gaps)
        start_safe = int(n_rows * 0.1)
        end_safe = int(n_rows * 0.9)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Create random gaps in the middle portion
            gap_indices = np.random.choice(
                range(start_safe, end_safe), 
                size=int((end_safe - start_safe) * nan_prob),
                replace=False
            )
            df.iloc[gap_indices, df.columns.get_loc(col)] = np.nan
        
        test_data[symbol] = df
    
    # Count total NaNs created
    total_nans = sum(df.isna().sum().sum() for df in test_data.values())
    logger.info(f"Created {total_nans:,} NaN values for interpolation testing")
    
    return test_data


def time_interpolation(test_data: dict, n_jobs: int, description: str) -> float:
    """
    Time the interpolation process with given parallelization settings.
    
    Args:
        test_data: Test dataset
        n_jobs: Number of parallel jobs
        description: Description for logging
        
    Returns:
        Execution time in seconds
    """
    logger.info(f"Testing {description}...")
    
    # Make a copy to avoid modifying original
    data_copy = {symbol: df.copy() for symbol, df in test_data.items()}
    
    start_time = time.time()
    
    result = interpolate_internal_gaps(data_copy, n_jobs=n_jobs, batch_size=16)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify interpolation worked
    total_nans_before = sum(df.isna().sum().sum() for df in data_copy.values())
    total_nans_after = sum(df.isna().sum().sum() for df in result.values())
    filled_count = total_nans_before - total_nans_after
    
    logger.info(f"{description} completed in {execution_time:.2f}s, filled {filled_count:,} values")
    
    return execution_time


def main():
    """Run interpolation performance test."""
    logger.info("Starting NaN interpolation performance test")
    
    # Create test dataset
    test_data = create_test_data(n_symbols=200, n_rows=500, nan_prob=0.03)
    
    logger.info("Running performance comparison...")
    
    # Test sequential processing
    sequential_time = time_interpolation(test_data, n_jobs=1, description="Sequential processing (1 job)")
    
    # Test parallel processing
    parallel_time = time_interpolation(test_data, n_jobs=-1, description="Parallel processing (all cores)")
    
    # Calculate speedup
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        logger.info(f"Performance Results:")
        logger.info(f"  Sequential: {sequential_time:.2f}s")
        logger.info(f"  Parallel:   {parallel_time:.2f}s")
        logger.info(f"  Speedup:    {speedup:.2f}x")
        
        if speedup > 1.5:
            logger.info("✅ Parallelization provides significant speedup!")
        elif speedup > 1.0:
            logger.info("✅ Parallelization provides some speedup")
        else:
            logger.warning("⚠️  Parallelization overhead may be too high for this dataset size")
    else:
        logger.error("❌ Sequential processing failed")
    
    logger.info("Performance test completed")


if __name__ == "__main__":
    main()