#!/usr/bin/env python3
"""
Test script for the adaptive weekly merge selection logic.

This script validates that the system correctly chooses between sequential and parallel
merge approaches based on dataset characteristics.
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.weekly_merge_parallel import optimized_merge_weekly_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_symbols: int, n_days: int) -> tuple:
    """Create test datasets with specified size."""
    np.random.seed(42)
    
    symbols = [f'SYM_{i:04d}' for i in range(n_symbols)]
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    
    # Daily data
    daily_data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        for date in dates:
            daily_data.append({
                'symbol': symbol,
                'date': date,
                'close': base_price * np.random.uniform(0.95, 1.05)
            })
    
    daily_df = pd.DataFrame(daily_data).sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Weekly data
    weekly_dates = pd.date_range('2020-01-03', periods=n_days//5, freq='W-FRI')
    weekly_data = []
    
    for symbol in symbols:
        for week_end in weekly_dates:
            weekly_data.append({
                'symbol': symbol,
                'week_end': week_end,
                'w_feature_1': np.random.normal(0, 1),
                'w_feature_2': np.random.uniform(0, 100),
                'w_feature_3': np.random.exponential(1)
            })
    
    weekly_df = pd.DataFrame(weekly_data).sort_values(['symbol', 'week_end']).reset_index(drop=True)
    
    return daily_df, weekly_df


def test_adaptive_selection():
    """Test that adaptive selection chooses the correct strategy."""
    logger.info("Testing adaptive merge strategy selection...")
    
    test_cases = [
        (10, 100, "Small dataset - should use sequential"),
        (50, 500, "Medium dataset - should use sequential"), 
        (150, 1000, "Large dataset - should use parallel"),
        (300, 2000, "Extra large dataset - should use parallel")
    ]
    
    results = []
    
    for n_symbols, n_days, description in test_cases:
        logger.info(f"\nTesting: {description}")
        logger.info(f"Parameters: {n_symbols} symbols, {n_days} days")
        
        # Create test data
        daily_df, weekly_df = create_test_data(n_symbols, n_days)
        
        # Time the merge operation
        start_time = time.time()
        
        # Capture log messages to verify strategy selection
        import io
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        temp_logger = logging.getLogger('src.features.weekly_merge_parallel')
        temp_logger.addHandler(handler)
        temp_logger.setLevel(logging.INFO)
        
        try:
            result = optimized_merge_weekly_features(daily_df, weekly_df)
            elapsed_time = time.time() - start_time
            
            # Check log output for strategy selection
            log_output = log_stream.getvalue()
            used_sequential = "Using sequential merge" in log_output
            used_parallel = "Using parallel merge" in log_output
            
            # Validate results
            assert not result.empty, "Result should not be empty"
            assert len(result) == len(daily_df), "Should preserve all daily rows"
            assert result['symbol'].nunique() == n_symbols, "Should preserve all symbols"
            
            # Check for weekly features
            weekly_cols = [col for col in result.columns if col.startswith('w_')]
            assert len(weekly_cols) > 0, "Should have weekly features"
            
            results.append({
                'description': description,
                'n_symbols': n_symbols,
                'n_days': n_days,
                'elapsed_time': elapsed_time,
                'used_sequential': used_sequential,
                'used_parallel': used_parallel,
                'weekly_features': len(weekly_cols)
            })
            
            strategy = "Sequential" if used_sequential else "Parallel" if used_parallel else "Unknown"
            logger.info(f"✅ Strategy used: {strategy}")
            logger.info(f"✅ Completed in {elapsed_time:.2f}s with {len(weekly_cols)} weekly features")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            results.append({
                'description': description,
                'n_symbols': n_symbols, 
                'n_days': n_days,
                'elapsed_time': float('inf'),
                'used_sequential': False,
                'used_parallel': False,
                'weekly_features': 0,
                'error': str(e)
            })
        
        finally:
            temp_logger.removeHandler(handler)
    
    return results


def test_performance_consistency():
    """Test that the adaptive system provides consistent performance benefits."""
    logger.info("Testing performance consistency...")
    
    # Test the same dataset multiple times to ensure consistency
    n_symbols, n_days = 200, 1500
    daily_df, weekly_df = create_test_data(n_symbols, n_days)
    
    times = []
    for i in range(3):
        logger.info(f"Run {i+1}/3...")
        start_time = time.time()
        result = optimized_merge_weekly_features(daily_df, weekly_df)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        # Verify result quality
        assert not result.empty, "Result should not be empty"
        weekly_features = len([col for col in result.columns if col.startswith('w_')])
        logger.info(f"  Time: {elapsed_time:.2f}s, Features: {weekly_features}")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    cv = (std_time / avg_time) * 100  # Coefficient of variation
    
    logger.info(f"Performance consistency: {avg_time:.2f}s ± {std_time:.2f}s (CV: {cv:.1f}%)")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'coefficient_variation': cv,
        'all_times': times
    }


def test_memory_efficiency():
    """Test memory efficiency of the adaptive approach."""
    logger.info("Testing memory efficiency...")
    
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Test with a large dataset
    daily_df, weekly_df = create_test_data(100, 1000)
    
    start_time = time.time()
    result = optimized_merge_weekly_features(daily_df, weekly_df)
    elapsed_time = time.time() - start_time
    
    # Get peak memory usage
    peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_increase = peak_memory - initial_memory
    
    logger.info(f"Memory efficiency test:")
    logger.info(f"  Initial memory: {initial_memory:.1f}MB")
    logger.info(f"  Peak memory: {peak_memory:.1f}MB")
    logger.info(f"  Memory increase: {memory_increase:.1f}MB")
    logger.info(f"  Processing time: {elapsed_time:.2f}s")
    
    return {
        'initial_memory': initial_memory,
        'peak_memory': peak_memory,
        'memory_increase': memory_increase,
        'processing_time': elapsed_time
    }


def main():
    """Run adaptive merge testing suite."""
    logger.info("Starting adaptive weekly merge testing suite")
    
    try:
        # Test 1: Adaptive strategy selection
        logger.info("="*60)
        logger.info("TEST 1: ADAPTIVE STRATEGY SELECTION")
        logger.info("="*60)
        
        selection_results = test_adaptive_selection()
        
        # Verify expected strategy selection
        expected_strategies = {
            'Small dataset': 'sequential',
            'Medium dataset': 'sequential', 
            'Large dataset': 'parallel',
            'Extra large dataset': 'parallel'
        }
        
        strategy_correct = 0
        for result in selection_results:
            if 'error' not in result:
                desc_key = result['description'].split(' - should')[0]
                expected = expected_strategies.get(desc_key, 'unknown')
                
                if expected == 'sequential' and result['used_sequential']:
                    strategy_correct += 1
                elif expected == 'parallel' and result['used_parallel']:
                    strategy_correct += 1
        
        logger.info(f"Strategy selection accuracy: {strategy_correct}/{len(selection_results)} correct")
        
        # Test 2: Performance consistency
        logger.info("\n" + "="*60)
        logger.info("TEST 2: PERFORMANCE CONSISTENCY")
        logger.info("="*60)
        
        consistency_results = test_performance_consistency()
        
        # Test 3: Memory efficiency
        logger.info("\n" + "="*60)
        logger.info("TEST 3: MEMORY EFFICIENCY")
        logger.info("="*60)
        
        memory_results = test_memory_efficiency()
        
        # Summary report
        logger.info("\n" + "="*60)
        logger.info("ADAPTIVE MERGE TEST SUMMARY")
        logger.info("="*60)
        
        successful_tests = len([r for r in selection_results if 'error' not in r])
        logger.info(f"Strategy Selection: {successful_tests}/{len(selection_results)} tests passed")
        logger.info(f"Strategy Accuracy: {strategy_correct}/{len(selection_results)} correct selections")
        logger.info(f"Performance Consistency: CV = {consistency_results['coefficient_variation']:.1f}%")
        logger.info(f"Memory Efficiency: {memory_results['memory_increase']:.1f}MB increase")
        
        # Performance comparison
        avg_times = {}
        for result in selection_results:
            if 'error' not in result:
                strategy = "Sequential" if result['used_sequential'] else "Parallel"
                if strategy not in avg_times:
                    avg_times[strategy] = []
                avg_times[strategy].append(result['elapsed_time'])
        
        for strategy, times in avg_times.items():
            avg_time = np.mean(times)
            logger.info(f"{strategy} average time: {avg_time:.2f}s ({len(times)} tests)")
        
        # Overall assessment
        if successful_tests == len(selection_results) and strategy_correct >= len(selection_results) * 0.75:
            logger.info("🎉 All adaptive merge tests PASSED!")
            return 0
        else:
            logger.error("❌ Some adaptive merge tests FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())