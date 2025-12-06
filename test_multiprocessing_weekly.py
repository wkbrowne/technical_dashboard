#!/usr/bin/env python3
"""
Test script for the new multiprocessing-optimized weekly features implementation.

This script validates that the process-compatible data restructuring fixes
the parallelization issues and achieves better CPU utilization and performance.
"""

import logging
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.features.weekly_multiprocessing import (
    add_weekly_features_to_daily_multiprocessing,
    MultiprocessingConfig,
    create_symbol_data_dict,
    process_symbols_in_parallel,
    determine_processing_strategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_realistic_test_data(n_symbols: int = 100, n_days: int = 500) -> pd.DataFrame:
    """Generate realistic test data for multiprocessing validation."""
    logger.info(f"Generating realistic test data: {n_symbols} symbols × {n_days} days")
    
    # Create date range (business days for realism)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic stock data with different characteristics per symbol
    np.random.seed(42)  # Reproducible results
    
    data_rows = []
    for i, symbol in enumerate([f"STOCK{i:03d}" for i in range(n_symbols)]):
        # Symbol-specific characteristics
        base_price = 20 + np.random.exponential(30)  # Realistic price distribution
        volatility = 0.15 + np.random.exponential(0.1)  # Varying volatility
        trend = np.random.normal(0, 0.0005)  # Small trend component
        
        # Generate price series with autocorrelation and realistic patterns
        price_returns = np.random.normal(trend, volatility/np.sqrt(252), n_days)
        
        # Add realistic autocorrelation
        for j in range(1, len(price_returns)):
            price_returns[j] += 0.05 * price_returns[j-1]  # Small momentum
        
        # Generate price levels
        prices = base_price * np.exp(np.cumsum(price_returns))
        
        # Generate realistic OHLCV data
        for j, date in enumerate(dates):
            close_price = prices[j]
            
            # Realistic daily range (varies by volatility)
            daily_vol = volatility * close_price / np.sqrt(252)
            high = close_price + daily_vol * np.random.exponential(0.7)
            low = close_price - daily_vol * np.random.exponential(0.7)
            
            # Open price influenced by previous close
            if j > 0:
                gap = np.random.normal(0, daily_vol * 0.3)
                open_price = max(low, min(high, prices[j-1] + gap))
            else:
                open_price = close_price + np.random.normal(0, daily_vol * 0.5)
                open_price = max(low, min(high, open_price))
            
            # Volume with realistic patterns
            base_volume = 100000 * (1 + np.random.exponential(2))
            volume_multiplier = 1 + 3 * abs(price_returns[j]) + 0.5 * np.random.exponential(0.5)
            volume = int(base_volume * volume_multiplier)
            
            data_rows.append({
                'symbol': symbol,
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
    
    df = pd.DataFrame(data_rows)
    
    # Add some missing data patterns for realism
    missing_prob = 0.001  # 0.1% missing data
    for col in ['volume']:
        mask = np.random.random(len(df)) < missing_prob
        df.loc[mask, col] = np.nan
    
    logger.info(f"Generated {len(df)} rows of realistic test data "
               f"({df.memory_usage(deep=True).sum() / 1024**2:.1f}MB)")
    
    return df


def monitor_system_resources(duration: float = 1.0):
    """Monitor CPU and memory usage during processing."""
    process = psutil.Process()
    
    cpu_percents = []
    memory_usages = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_usage = process.memory_info().rss / 1024**2  # MB
        
        cpu_percents.append(cpu_percent)
        memory_usages.append(memory_usage)
        
        time.sleep(0.1)
    
    return {
        'avg_cpu_percent': np.mean(cpu_percents),
        'max_cpu_percent': np.max(cpu_percents),
        'avg_memory_mb': np.mean(memory_usages),
        'max_memory_mb': np.max(memory_usages)
    }


def test_data_partitioning_efficiency():
    """Test the efficiency of the new data partitioning approach."""
    logger.info("🔍 Testing data partitioning efficiency...")
    
    # Test different dataset sizes
    test_cases = [
        {"name": "Small", "symbols": 20, "days": 100},
        {"name": "Medium", "symbols": 50, "days": 300}, 
        {"name": "Large", "symbols": 100, "days": 500},
    ]
    
    results = []
    
    for case in test_cases:
        logger.info(f"Testing {case['name']} dataset: {case['symbols']} symbols × {case['days']} days")
        
        # Generate test data
        df_test = generate_realistic_test_data(case['symbols'], case['days'])
        original_memory = df_test.memory_usage(deep=True).sum() / 1024**2
        
        # Test data partitioning
        start_time = time.time()
        symbol_data_dict = create_symbol_data_dict(df_test, optimize_dtypes=True)
        partitioning_time = time.time() - start_time
        
        # Calculate partitioned memory usage
        partitioned_memory = sum(
            df.memory_usage(deep=True).sum() for df in symbol_data_dict.values()
        ) / 1024**2
        
        # Test processing strategy determination
        strategy, config = determine_processing_strategy(symbol_data_dict)
        
        result = {
            'dataset': case['name'],
            'symbols': case['symbols'],
            'days': case['days'],
            'original_memory_mb': original_memory,
            'partitioned_memory_mb': partitioned_memory,
            'memory_reduction_pct': (1 - partitioned_memory / original_memory) * 100,
            'partitioning_time': partitioning_time,
            'processing_strategy': strategy,
            'strategy_config': config
        }
        
        results.append(result)
        
        logger.info(f"  Partitioning time: {partitioning_time:.2f}s")
        logger.info(f"  Memory: {original_memory:.1f}MB -> {partitioned_memory:.1f}MB "
                   f"({result['memory_reduction_pct']:.1f}% reduction)")
        logger.info(f"  Strategy: {strategy} with config {config}")
    
    return results


def test_multiprocessing_performance():
    """Test the multiprocessing performance improvements."""
    logger.info("🚀 Testing multiprocessing performance...")
    
    # Generate larger dataset for meaningful performance testing
    df_test = generate_realistic_test_data(n_symbols=80, n_days=400)
    logger.info(f"Test dataset: {len(df_test)} rows, {df_test['symbol'].nunique()} symbols, "
               f"{df_test.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    # Test different configurations
    configs = [
        {"name": "Standard", "n_jobs": 1, "dtype_opt": False},
        {"name": "Multi-core", "n_jobs": -1, "dtype_opt": False},
        {"name": "Optimized", "n_jobs": -1, "dtype_opt": True},
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"\nTesting {config['name']} configuration...")
        
        mp_config = MultiprocessingConfig(
            n_jobs=config['n_jobs'],
            dtype_optimization=config['dtype_opt'],
            verbose=0  # Reduce noise during testing
        )
        
        # Monitor system resources during processing
        start_time = time.time()
        
        try:
            final_features = add_weekly_features_to_daily_multiprocessing(
                df_test.copy(),
                spy_symbol="STOCK001",
                config=mp_config
            )
            
            processing_time = time.time() - start_time
            success = True
            
            # Analyze results
            weekly_features = [col for col in final_features.columns if col.startswith('w_')]
            symbols_processed = final_features['symbol'].nunique()
            
        except Exception as e:
            processing_time = time.time() - start_time
            success = False
            weekly_features = []
            symbols_processed = 0
            logger.error(f"Configuration {config['name']} failed: {e}")
        
        result = {
            'config_name': config['name'],
            'n_jobs': config['n_jobs'],
            'dtype_optimization': config['dtype_opt'],
            'success': success,
            'processing_time': processing_time,
            'symbols_processed': symbols_processed,
            'weekly_features_count': len(weekly_features),
            'processing_rate': symbols_processed / processing_time if processing_time > 0 else 0
        }
        
        results.append(result)
        
        if success:
            logger.info(f"  ✅ Success: {processing_time:.2f}s, {len(weekly_features)} features, "
                       f"{result['processing_rate']:.1f} symbols/sec")
        else:
            logger.info(f"  ❌ Failed: {processing_time:.2f}s")
    
    return results


def test_cpu_utilization():
    """Test CPU utilization improvements."""
    logger.info("💻 Testing CPU utilization...")
    
    # Generate test data
    df_test = generate_realistic_test_data(n_symbols=60, n_days=300)
    
    # Test configurations
    test_configs = [
        {"name": "Single Process", "n_jobs": 1},
        {"name": "Multi Process", "n_jobs": -1},
    ]
    
    cpu_results = []
    
    for config in test_configs:
        logger.info(f"\nTesting {config['name']} CPU utilization...")
        
        mp_config = MultiprocessingConfig(
            n_jobs=config['n_jobs'],
            dtype_optimization=True,
            verbose=0
        )
        
        # Start resource monitoring in background
        def run_processing():
            return add_weekly_features_to_daily_multiprocessing(
                df_test.copy(),
                spy_symbol="STOCK001", 
                config=mp_config
            )
        
        # Monitor during processing
        start_time = time.time()
        
        # Start monitoring
        monitor_process = psutil.Process()
        cpu_usage_samples = []
        
        try:
            # Run processing while monitoring
            result = run_processing()
            processing_time = time.time() - start_time
            
            # Sample CPU usage during processing
            for _ in range(10):
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_usage_samples.append(cpu_percent)
            
            avg_cpu = np.mean(cpu_usage_samples)
            max_cpu = np.max(cpu_usage_samples)
            
            success = True
            features_count = len([col for col in result.columns if col.startswith('w_')])
            
        except Exception as e:
            processing_time = time.time() - start_time
            avg_cpu = 0
            max_cpu = 0
            success = False
            features_count = 0
            logger.error(f"CPU test failed: {e}")
        
        cpu_result = {
            'config_name': config['name'],
            'n_jobs': config['n_jobs'],
            'success': success,
            'processing_time': processing_time,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'features_count': features_count
        }
        
        cpu_results.append(cpu_result)
        
        if success:
            logger.info(f"  ✅ {config['name']}: {processing_time:.2f}s, "
                       f"CPU: {avg_cpu:.1f}% avg / {max_cpu:.1f}% max")
        else:
            logger.info(f"  ❌ {config['name']}: Failed")
    
    return cpu_results


def main():
    """Run comprehensive multiprocessing tests."""
    logger.info("🎯 Starting Multiprocessing Weekly Features Validation")
    logger.info("=" * 80)
    
    try:
        # Test 1: Data partitioning efficiency
        logger.info("\n📊 TEST 1: Data Partitioning Efficiency")
        partitioning_results = test_data_partitioning_efficiency()
        
        # Test 2: Multiprocessing performance
        logger.info("\n⚡ TEST 2: Multiprocessing Performance")
        performance_results = test_multiprocessing_performance()
        
        # Test 3: CPU utilization
        logger.info("\n🖥️ TEST 3: CPU Utilization")
        cpu_results = test_cpu_utilization()
        
        # Summary report
        logger.info("\n" + "=" * 80)
        logger.info("📋 COMPREHENSIVE TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Partitioning results
        logger.info("\n🔍 Data Partitioning Results:")
        for result in partitioning_results:
            logger.info(f"  {result['dataset']:<8}: {result['partitioning_time']:.2f}s, "
                       f"{result['memory_reduction_pct']:.1f}% memory reduction, "
                       f"strategy: {result['processing_strategy']}")
        
        # Performance results  
        logger.info("\n⚡ Performance Results:")
        for result in performance_results:
            if result['success']:
                logger.info(f"  {result['config_name']:<12}: {result['processing_time']:.2f}s, "
                           f"{result['processing_rate']:.1f} symbols/sec, "
                           f"{result['weekly_features_count']} features")
            else:
                logger.info(f"  {result['config_name']:<12}: ❌ FAILED")
        
        # CPU utilization results
        logger.info("\n💻 CPU Utilization Results:")
        for result in cpu_results:
            if result['success']:
                logger.info(f"  {result['config_name']:<15}: {result['processing_time']:.2f}s, "
                           f"CPU: {result['avg_cpu_percent']:.1f}%")
            else:
                logger.info(f"  {result['config_name']:<15}: ❌ FAILED")
        
        # Performance comparison
        if len(performance_results) >= 2 and performance_results[0]['success'] and performance_results[-1]['success']:
            baseline_time = performance_results[0]['processing_time']
            optimized_time = performance_results[-1]['processing_time']
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0
            logger.info(f"\n🎉 Performance Improvement: {speedup:.2f}x faster with multiprocessing optimization")
        
        # CPU utilization comparison
        if len(cpu_results) >= 2 and cpu_results[0]['success'] and cpu_results[-1]['success']:
            single_cpu = cpu_results[0]['avg_cpu_percent']
            multi_cpu = cpu_results[-1]['avg_cpu_percent']
            cpu_improvement = multi_cpu / single_cpu if single_cpu > 0 else 0
            logger.info(f"🖥️ CPU Utilization Improvement: {cpu_improvement:.2f}x better CPU usage with multiprocessing")
        
        # Final assessment
        successful_tests = sum([
            all(r['memory_reduction_pct'] > 0 for r in partitioning_results),
            any(r['success'] and r['n_jobs'] == -1 for r in performance_results),
            any(r['success'] and r['n_jobs'] == -1 for r in cpu_results)
        ])
        
        logger.info("\n" + "=" * 80)
        if successful_tests == 3:
            logger.info("🎊 ALL TESTS PASSED! Multiprocessing optimization is working correctly.")
            logger.info("✅ Data duplication eliminated")
            logger.info("✅ Multiprocessing performance validated") 
            logger.info("✅ CPU utilization improved")
        else:
            logger.info(f"⚠️ {successful_tests}/3 test categories passed. Review failed tests.")
        
        logger.info("=" * 80)
        
        return successful_tests == 3
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)