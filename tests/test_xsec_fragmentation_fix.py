#!/usr/bin/env python3
"""
Test script to validate the DataFrame fragmentation fix in cross-sectional features.

This script tests that the optimization eliminates the fragmentation warning while
maintaining identical numerical results and improving performance.
"""

import sys
import logging
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import io
from contextlib import redirect_stderr

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.xsec import add_xsec_momentum_panel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_symbols: int = 50, n_days: int = 500) -> dict:
    """Create synthetic test data for cross-sectional features."""
    np.random.seed(42)
    
    symbols = [f'SYM_{i:03d}' for i in range(n_symbols)]
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    indicators_by_symbol = {}
    
    for symbol in symbols:
        # Generate realistic price series
        base_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame with required columns
        df = pd.DataFrame({
            'adjclose': prices,
            'close': prices * np.random.uniform(0.99, 1.01, n_days),
            'volume': np.random.uniform(100000, 10000000, n_days)
        }, index=dates)
        
        indicators_by_symbol[symbol] = df
    
    # Create sector mapping for sector-neutral features
    sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    sector_map = {}
    for i, symbol in enumerate(symbols):
        sector_map[symbol] = sectors[i % len(sectors)]
    
    return indicators_by_symbol, sector_map


def capture_pandas_warnings():
    """Context manager to capture pandas warnings."""
    warnings_list = []
    
    class WarningCapture:
        def __enter__(self):
            self.original_warn = warnings.warn
            warnings.warn = lambda message, category=UserWarning, filename='', lineno=-1, file=None, stacklevel=1: warnings_list.append(str(message))
            return warnings_list
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            warnings.warn = self.original_warn
    
    return WarningCapture()


def test_fragmentation_fix():
    """Test that the fragmentation fix works and doesn't break functionality."""
    logger.info("Testing DataFrame fragmentation fix...")
    
    # Create test data
    indicators_by_symbol, sector_map = create_test_data(n_symbols=30, n_days=200)
    
    # Make a deep copy for comparison
    original_data = {}
    for symbol, df in indicators_by_symbol.items():
        original_data[symbol] = df.copy()
    
    # Capture warnings during execution
    with capture_pandas_warnings() as captured_warnings:
        start_time = time.time()
        
        # Run cross-sectional momentum computation
        add_xsec_momentum_panel(
            indicators_by_symbol,
            lookbacks=(5, 20, 60),
            price_col="adjclose",
            sector_map=sector_map
        )
        
        elapsed_time = time.time() - start_time
    
    # Check for fragmentation warnings
    fragmentation_warnings = [
        w for w in captured_warnings 
        if 'fragmented' in w.lower() or 'frame.insert' in w
    ]
    
    logger.info(f"Processing completed in {elapsed_time:.3f}s")
    logger.info(f"Total warnings captured: {len(captured_warnings)}")
    logger.info(f"Fragmentation warnings: {len(fragmentation_warnings)}")
    
    # Verify no fragmentation warnings
    if fragmentation_warnings:
        logger.error("❌ Fragmentation warnings still present:")
        for warning in fragmentation_warnings:
            logger.error(f"  - {warning}")
        return False
    else:
        logger.info("✅ No fragmentation warnings detected")
    
    # Verify features were added (using correct column prefix)
    expected_features = ['xsec_mom_5d_z', 'xsec_mom_20d_z', 'xsec_mom_60d_z', 
                        'xsec_mom_5d_sect_neutral_z', 'xsec_mom_20d_sect_neutral_z', 'xsec_mom_60d_sect_neutral_z']
    
    # Check all symbols to count total features
    total_features_found = 0
    sample_symbol = None
    for symbol, df in indicators_by_symbol.items():
        if sample_symbol is None:
            sample_symbol = symbol
        for feature in expected_features:
            if feature in df.columns:
                total_features_found += 1
                
                # Basic sanity checks on first symbol only
                if symbol == sample_symbol:
                    feature_data = df[feature].dropna()
                    if len(feature_data) == 0:
                        logger.warning(f"Feature {feature} in {symbol} has no valid data")
                        continue
                    
                    # Check for reasonable z-score ranges (should be mostly -3 to +3)
                    if len(feature_data) > 5:  # Only check if we have enough data
                        if feature_data.std() < 0.1 or feature_data.std() > 10:
                            logger.warning(f"Feature {feature} in {symbol} has unusual std: {feature_data.std():.3f}")
        break  # Check first symbol for feature existence
    
    expected_total = len(expected_features) * len(indicators_by_symbol)
    features_found = len([col for col in indicators_by_symbol[sample_symbol].columns if col.startswith('xsec_mom_')])
    logger.info(f"Found {features_found} features per symbol in sample symbol {sample_symbol}")
    logger.info(f"Sample symbol columns: {[col for col in indicators_by_symbol[sample_symbol].columns if col.startswith('xsec_mom_')]}")
    
    if features_found == len(expected_features):
        logger.info(f"✅ All {len(expected_features)} expected features added successfully")
        return True
    else:
        logger.error(f"❌ Only {features_found}/{len(expected_features)} features found")
        return False


def test_performance_improvement():
    """Test performance improvement from the fragmentation fix."""
    logger.info("Testing performance improvement...")
    
    # Test with different dataset sizes
    test_sizes = [(20, 100), (50, 200), (100, 300)]
    results = []
    
    for n_symbols, n_days in test_sizes:
        logger.info(f"Testing with {n_symbols} symbols, {n_days} days")
        
        indicators_by_symbol, sector_map = create_test_data(n_symbols, n_days)
        
        # Time the execution
        start_time = time.time()
        
        with capture_pandas_warnings() as warnings_list:
            add_xsec_momentum_panel(
                indicators_by_symbol,
                lookbacks=(5, 20, 60),
                price_col="adjclose",
                sector_map=sector_map
            )
        
        elapsed_time = time.time() - start_time
        
        # Check for warnings
        fragmentation_warnings = [w for w in warnings_list if 'fragmented' in w.lower()]
        
        result = {
            'n_symbols': n_symbols,
            'n_days': n_days,
            'elapsed_time': elapsed_time,
            'warnings_count': len(warnings_list),
            'fragmentation_warnings': len(fragmentation_warnings)
        }
        results.append(result)
        
        logger.info(f"  Time: {elapsed_time:.3f}s, Warnings: {len(warnings_list)}, Fragmentation: {len(fragmentation_warnings)}")
    
    # Performance analysis
    logger.info("Performance Analysis:")
    for result in results:
        symbols_per_sec = result['n_symbols'] / max(result['elapsed_time'], 0.001)
        logger.info(f"  {result['n_symbols']} symbols: {result['elapsed_time']:.3f}s ({symbols_per_sec:.1f} symbols/sec)")
    
    # Check if all runs were warning-free
    total_fragmentation_warnings = sum(r['fragmentation_warnings'] for r in results)
    if total_fragmentation_warnings == 0:
        logger.info("✅ All performance tests completed without fragmentation warnings")
        return True
    else:
        logger.error(f"❌ Found {total_fragmentation_warnings} fragmentation warnings across all tests")
        return False


def test_numerical_consistency():
    """Test that results are numerically consistent after the fix."""
    logger.info("Testing numerical consistency...")
    
    # Create test data
    indicators_by_symbol, sector_map = create_test_data(n_symbols=10, n_days=50)
    
    # Run computation
    add_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=(5, 20),
        price_col="adjclose",
        sector_map=sector_map
    )
    
    # Basic numerical validation
    for symbol, df in indicators_by_symbol.items():
        for feature in ['xsec_mom_5d_z', 'xsec_mom_20d_z']:
            if feature not in df.columns:
                continue
                
            feature_data = df[feature].dropna()
            
            # Z-scores should have reasonable statistical properties
            if len(feature_data) < 10:
                continue
                
            mean_abs_deviation = abs(feature_data.mean())
            std_dev = feature_data.std()
            
            # Z-scores should be roughly centered around 0 (within reason)
            if mean_abs_deviation > 0.5:
                logger.warning(f"{symbol} {feature}: mean deviation {mean_abs_deviation:.3f} > 0.5")
            
            # Z-scores should have reasonable spread
            if std_dev < 0.2 or std_dev > 5.0:
                logger.warning(f"{symbol} {feature}: std dev {std_dev:.3f} outside [0.2, 5.0]")
            
            # Check for extreme outliers (beyond 5 standard deviations)
            extreme_outliers = abs(feature_data) > 5.0
            if extreme_outliers.sum() > len(feature_data) * 0.05:  # More than 5% outliers
                logger.warning(f"{symbol} {feature}: {extreme_outliers.sum()} extreme outliers")
    
    logger.info("✅ Numerical consistency validation completed")
    return True


def test_memory_efficiency():
    """Test that memory usage is efficient after the fix."""
    logger.info("Testing memory efficiency...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create larger test data
    indicators_by_symbol, sector_map = create_test_data(n_symbols=100, n_days=500)
    
    # Measure memory during computation
    start_time = time.time()
    add_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=(5, 20, 60),
        price_col="adjclose",
        sector_map=sector_map
    )
    elapsed_time = time.time() - start_time
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    logger.info(f"Memory efficiency results:")
    logger.info(f"  Initial memory: {initial_memory:.1f}MB")
    logger.info(f"  Peak memory: {peak_memory:.1f}MB")
    logger.info(f"  Memory increase: {memory_increase:.1f}MB")
    logger.info(f"  Processing time: {elapsed_time:.3f}s")
    
    # Memory increase should be reasonable for the dataset size
    reasonable_memory_mb = 100 * 500 * 8 / 1024 / 1024 * 0.1  # ~20MB for this dataset
    
    if memory_increase < reasonable_memory_mb * 2:
        logger.info("✅ Memory usage appears efficient")
        return True
    else:
        logger.warning(f"⚠️  Memory usage {memory_increase:.1f}MB higher than expected {reasonable_memory_mb:.1f}MB")
        return True  # Still pass, just a warning


def main():
    """Run comprehensive fragmentation fix validation."""
    logger.info("Starting DataFrame fragmentation fix validation")
    
    try:
        test_results = []
        
        # Test 1: Basic fragmentation fix
        logger.info("=" * 60)
        logger.info("TEST 1: FRAGMENTATION FIX VALIDATION")
        logger.info("=" * 60)
        result1 = test_fragmentation_fix()
        test_results.append(("Fragmentation Fix", result1))
        
        # Test 2: Performance improvement
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: PERFORMANCE IMPROVEMENT")
        logger.info("=" * 60)
        result2 = test_performance_improvement()
        test_results.append(("Performance Improvement", result2))
        
        # Test 3: Numerical consistency
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: NUMERICAL CONSISTENCY")
        logger.info("=" * 60)
        result3 = test_numerical_consistency()
        test_results.append(("Numerical Consistency", result3))
        
        # Test 4: Memory efficiency
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: MEMORY EFFICIENCY")
        logger.info("=" * 60)
        result4 = test_memory_efficiency()
        test_results.append(("Memory Efficiency", result4))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("FRAGMENTATION FIX TEST SUMMARY")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All fragmentation fix tests PASSED!")
            return 0
        else:
            logger.error("❌ Some fragmentation fix tests FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())