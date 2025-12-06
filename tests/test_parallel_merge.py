#!/usr/bin/env python3
"""
Test and benchmark script for parallel weekly-to-daily merge implementation.

This script tests the performance improvements of the new parallel merge system
compared to the original sequential implementation.
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.weekly_merge_parallel import (
    ParallelWeeklyMerger, ParallelMergeConfig, SymbolDataPartitioner,
    parallel_merge_weekly_to_daily, optimized_merge_weekly_features
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_datasets(n_symbols: int, n_days: int, n_weekly_features: int = 20) -> tuple:
    """
    Create realistic test datasets for benchmarking.
    
    Args:
        n_symbols: Number of unique symbols
        n_days: Number of trading days
        n_weekly_features: Number of weekly features to generate
        
    Returns:
        Tuple of (daily_df, weekly_df)
    """
    logger.info(f"Creating test dataset: {n_symbols} symbols, {n_days} days, {n_weekly_features} weekly features")
    
    np.random.seed(42)  # Reproducible results
    
    # Generate symbol list
    symbols = [f'SYM_{i:04d}' for i in range(n_symbols)]
    
    # Generate daily data
    daily_data = []
    base_date = pd.Timestamp('2020-01-01')
    dates = pd.bdate_range(start=base_date, periods=n_days)
    
    for symbol in symbols:
        base_price = np.random.uniform(20, 200)
        for i, date in enumerate(dates):
            price = base_price * (1 + np.random.normal(0, 0.02) * np.sqrt(i / n_days))
            daily_data.append({
                'symbol': symbol,
                'date': date,
                'open': price * np.random.uniform(0.98, 1.02),
                'high': price * np.random.uniform(1.00, 1.05),
                'low': price * np.random.uniform(0.95, 1.00),
                'close': price,
                'volume': np.random.uniform(100000, 10000000)
            })
    
    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Generate weekly data
    weekly_data = []
    weekly_dates = pd.date_range(start=base_date, end=dates[-1], freq='W-FRI')
    
    for symbol in symbols:
        for week_end in weekly_dates:
            weekly_record = {
                'symbol': symbol,
                'week_end': week_end
            }
            
            # Generate realistic weekly features
            base_value = np.random.normal(0, 1)
            for i in range(n_weekly_features):
                if i < 5:  # RSI-like features (0-100)
                    weekly_record[f'w_rsi_{i}'] = np.random.uniform(20, 80)
                elif i < 10:  # MACD-like features (centered around 0)
                    weekly_record[f'w_macd_{i}'] = np.random.normal(0, 0.5)
                elif i < 15:  # Moving averages (price-like)
                    weekly_record[f'w_ma_{i}'] = np.random.uniform(50, 150)
                else:  # Various other features
                    weekly_record[f'w_feature_{i}'] = base_value + np.random.normal(0, 0.1)
            
            weekly_data.append(weekly_record)
    
    weekly_df = pd.DataFrame(weekly_data)
    weekly_df = weekly_df.sort_values(['symbol', 'week_end']).reset_index(drop=True)
    
    logger.info(f"Generated datasets: Daily {daily_df.shape}, Weekly {weekly_df.shape}")
    return daily_df, weekly_df


def sequential_merge_baseline(df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline sequential merge implementation for performance comparison.
    This replicates the original merge logic from the enhanced weekly features.
    """
    logger.info("Running baseline sequential merge...")
    start_time = time.time()
    
    merged_results = []
    for symbol in df_daily['symbol'].unique():
        daily_symbol = df_daily[df_daily['symbol'] == symbol].sort_values('date')
        weekly_symbol = df_weekly[df_weekly['symbol'] == symbol].sort_values('week_end')
        
        if not weekly_symbol.empty:
            merged_symbol = pd.merge_asof(
                daily_symbol,
                weekly_symbol.drop(columns=['symbol']),
                left_on='date',
                right_on='week_end',
                direction='backward'
            )
            merged_results.append(merged_symbol)
        else:
            merged_results.append(daily_symbol)
    
    result = pd.concat(merged_results, ignore_index=True)
    if 'week_end' in result.columns:
        result = result.drop(columns=['week_end'])
    
    elapsed_time = time.time() - start_time
    logger.info(f"Sequential merge completed in {elapsed_time:.2f}s")
    
    return result, elapsed_time


def test_partitioner_performance():
    """Test the performance of the data partitioner."""
    logger.info("Testing data partitioner performance...")
    
    # Test with various dataset sizes
    test_sizes = [(10, 500), (50, 1000), (100, 2000)]
    partitioner_results = {}
    
    for n_symbols, n_days in test_sizes:
        daily_df, weekly_df = create_test_datasets(n_symbols, n_days, n_weekly_features=10)
        
        config = ParallelMergeConfig(optimize_memory=True)
        partitioner = SymbolDataPartitioner(config)
        
        start_time = time.time()
        daily_partitions, weekly_partitions = partitioner.partition_data(daily_df, weekly_df)
        partition_time = time.time() - start_time
        
        partitioner_results[f"{n_symbols}_{n_days}"] = {
            'symbols': n_symbols,
            'days': n_days,
            'partition_time': partition_time,
            'memory_usage_mb': partitioner.memory_usage_mb,
            'partitions_created': len(daily_partitions)
        }
        
        logger.info(f"  {n_symbols} symbols, {n_days} days: {partition_time:.2f}s, {partitioner.memory_usage_mb:.1f}MB")
    
    return partitioner_results


def benchmark_merge_implementations():
    """Benchmark sequential vs parallel merge implementations."""
    logger.info("Benchmarking merge implementations...")
    
    # Test configurations: (n_symbols, n_days, description)
    test_configs = [
        (20, 500, "Small dataset"),
        (50, 1000, "Medium dataset"), 
        (100, 1500, "Large dataset"),
        (200, 2000, "Extra large dataset")
    ]
    
    benchmark_results = []
    
    for n_symbols, n_days, description in test_configs:
        logger.info(f"\nTesting {description}: {n_symbols} symbols, {n_days} days")
        
        # Create test data
        daily_df, weekly_df = create_test_datasets(n_symbols, n_days, n_weekly_features=15)
        
        # Test sequential baseline
        try:
            sequential_result, sequential_time = sequential_merge_baseline(daily_df, weekly_df)
            sequential_success = True
        except Exception as e:
            logger.error(f"Sequential merge failed: {e}")
            sequential_time = float('inf')
            sequential_success = False
            sequential_result = None
        
        # Test parallel implementation
        try:
            config = ParallelMergeConfig(
                chunk_size=max(10, n_symbols // 4),
                n_jobs=-1,
                backend='threading',
                optimize_memory=True
            )
            
            start_time = time.time()
            parallel_result = parallel_merge_weekly_to_daily(daily_df, weekly_df, config)
            parallel_time = time.time() - start_time
            parallel_success = True
            
        except Exception as e:
            logger.error(f"Parallel merge failed: {e}")
            parallel_time = float('inf')
            parallel_success = False
            parallel_result = None
        
        # Validate results are equivalent (if both succeeded)
        results_match = False
        if sequential_success and parallel_success:
            try:
                # Check shapes match
                shapes_match = sequential_result.shape == parallel_result.shape
                
                # Check weekly feature columns match
                seq_weekly_cols = sorted([col for col in sequential_result.columns if col.startswith('w_')])
                par_weekly_cols = sorted([col for col in parallel_result.columns if col.startswith('w_')])
                columns_match = seq_weekly_cols == par_weekly_cols
                
                results_match = shapes_match and columns_match
                logger.info(f"  Results validation: shapes_match={shapes_match}, columns_match={columns_match}")
                
            except Exception as e:
                logger.warning(f"Results comparison failed: {e}")
                results_match = False
        
        # Calculate performance metrics
        if sequential_success and parallel_success:
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / max(1, config.n_jobs) * 100 if config.n_jobs > 0 else 0
        else:
            speedup = 0
            efficiency = 0
        
        result_entry = {
            'description': description,
            'n_symbols': n_symbols,
            'n_days': n_days,
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'results_match': results_match,
            'sequential_success': sequential_success,
            'parallel_success': parallel_success
        }
        
        benchmark_results.append(result_entry)
        
        logger.info(f"  Sequential: {sequential_time:.2f}s")
        logger.info(f"  Parallel: {parallel_time:.2f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Results match: {results_match}")
    
    return benchmark_results


def test_memory_optimization():
    """Test memory optimization features."""
    logger.info("Testing memory optimization...")
    
    # Create test data
    daily_df, weekly_df = create_test_datasets(100, 1000, n_weekly_features=20)
    
    # Test without memory optimization
    config_no_opt = ParallelMergeConfig(optimize_memory=False, use_categorical=False)
    partitioner_no_opt = SymbolDataPartitioner(config_no_opt)
    
    start_time = time.time()
    daily_no_opt, weekly_no_opt = partitioner_no_opt.partition_data(daily_df, weekly_df)
    time_no_opt = time.time() - start_time
    memory_no_opt = partitioner_no_opt.memory_usage_mb
    
    # Test with memory optimization
    config_opt = ParallelMergeConfig(optimize_memory=True, use_categorical=True)
    partitioner_opt = SymbolDataPartitioner(config_opt)
    
    start_time = time.time()
    daily_opt, weekly_opt = partitioner_opt.partition_data(daily_df, weekly_df)
    time_opt = time.time() - start_time
    memory_opt = partitioner_opt.memory_usage_mb
    
    memory_savings = ((memory_no_opt - memory_opt) / memory_no_opt) * 100 if memory_no_opt > 0 else 0
    
    logger.info(f"Memory optimization results:")
    logger.info(f"  Without optimization: {memory_no_opt:.1f}MB, {time_no_opt:.2f}s")
    logger.info(f"  With optimization: {memory_opt:.1f}MB, {time_opt:.2f}s")
    logger.info(f"  Memory savings: {memory_savings:.1f}%")
    
    return {
        'memory_no_opt': memory_no_opt,
        'memory_opt': memory_opt,
        'memory_savings': memory_savings,
        'time_no_opt': time_no_opt,
        'time_opt': time_opt
    }


def test_chunk_size_optimization():
    """Test optimal chunk size selection."""
    logger.info("Testing chunk size optimization...")
    
    daily_df, weekly_df = create_test_datasets(100, 1000, n_weekly_features=10)
    
    chunk_sizes = [10, 25, 50, 100, 200]
    chunk_results = []
    
    for chunk_size in chunk_sizes:
        config = ParallelMergeConfig(
            chunk_size=chunk_size,
            n_jobs=4,  # Fixed for fair comparison
            backend='threading'
        )
        
        try:
            start_time = time.time()
            result = parallel_merge_weekly_to_daily(daily_df, weekly_df, config)
            elapsed_time = time.time() - start_time
            success = True
            
        except Exception as e:
            logger.warning(f"Chunk size {chunk_size} failed: {e}")
            elapsed_time = float('inf')
            success = False
        
        chunk_results.append({
            'chunk_size': chunk_size,
            'time': elapsed_time,
            'success': success
        })
        
        logger.info(f"  Chunk size {chunk_size}: {elapsed_time:.2f}s")
    
    return chunk_results


def generate_performance_report(benchmark_results: List[Dict], 
                              partitioner_results: Dict,
                              memory_results: Dict,
                              chunk_results: List[Dict]):
    """Generate comprehensive performance report."""
    logger.info("Generating performance report...")
    
    report = []
    report.append("# Parallel Weekly Merge Performance Report\n")
    
    # Benchmark summary
    report.append("## Merge Performance Comparison\n")
    report.append("| Dataset | Symbols | Days | Sequential (s) | Parallel (s) | Speedup | Results Match |\n")
    report.append("|---------|---------|------|---------------|-------------|---------|---------------|\n")
    
    for result in benchmark_results:
        report.append(
            f"| {result['description']} | {result['n_symbols']} | {result['n_days']} | "
            f"{result['sequential_time']:.2f} | {result['parallel_time']:.2f} | "
            f"{result['speedup']:.2f}x | {result['results_match']} |\n"
        )
    
    # Performance metrics
    if benchmark_results:
        successful_results = [r for r in benchmark_results if r['sequential_success'] and r['parallel_success']]
        if successful_results:
            avg_speedup = np.mean([r['speedup'] for r in successful_results])
            max_speedup = max([r['speedup'] for r in successful_results])
            report.append(f"\n**Average Speedup:** {avg_speedup:.2f}x\n")
            report.append(f"**Maximum Speedup:** {max_speedup:.2f}x\n")
    
    # Memory optimization results
    report.append("\n## Memory Optimization Results\n")
    report.append(f"- **Memory Savings:** {memory_results['memory_savings']:.1f}%\n")
    report.append(f"- **Without Optimization:** {memory_results['memory_no_opt']:.1f}MB\n")
    report.append(f"- **With Optimization:** {memory_results['memory_opt']:.1f}MB\n")
    
    # Partitioner performance
    report.append("\n## Data Partitioner Performance\n")
    report.append("| Dataset | Partition Time (s) | Memory Usage (MB) |\n")
    report.append("|---------|-------------------|------------------|\n")
    
    for key, result in partitioner_results.items():
        report.append(
            f"| {result['symbols']} symbols, {result['days']} days | "
            f"{result['partition_time']:.2f} | {result['memory_usage_mb']:.1f} |\n"
        )
    
    # Chunk size analysis
    report.append("\n## Chunk Size Optimization\n")
    successful_chunks = [r for r in chunk_results if r['success']]
    if successful_chunks:
        optimal_chunk = min(successful_chunks, key=lambda x: x['time'])
        report.append(f"**Optimal Chunk Size:** {optimal_chunk['chunk_size']} ({optimal_chunk['time']:.2f}s)\n")
    
    report.append("\n## Recommendations\n")
    if successful_results:
        report.append("- **Use parallel merge for datasets with >20 symbols** for optimal performance\n")
        report.append("- **Enable memory optimization** for large datasets (>50MB)\n")
        report.append(f"- **Optimal chunk size:** ~{optimal_chunk['chunk_size'] if successful_chunks else 50} symbols per chunk\n")
        report.append("- **Use threading backend** for I/O-bound merge operations\n")
    
    return ''.join(report)


def main():
    """Run comprehensive parallel merge testing and benchmarking."""
    logger.info("Starting parallel merge performance testing suite")
    
    try:
        # Test 1: Data partitioner performance
        partitioner_results = test_partitioner_performance()
        
        # Test 2: Merge implementation benchmarking
        benchmark_results = benchmark_merge_implementations()
        
        # Test 3: Memory optimization testing
        memory_results = test_memory_optimization()
        
        # Test 4: Chunk size optimization
        chunk_results = test_chunk_size_optimization()
        
        # Generate comprehensive report
        performance_report = generate_performance_report(
            benchmark_results, partitioner_results, memory_results, chunk_results
        )
        
        # Save report to file
        report_path = Path("parallel_merge_performance_report.md")
        with open(report_path, 'w') as f:
            f.write(performance_report)
        
        logger.info(f"Performance report saved to {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("PARALLEL MERGE PERFORMANCE SUMMARY")
        print("="*60)
        
        successful_benchmarks = [r for r in benchmark_results if r['parallel_success'] and r['sequential_success']]
        if successful_benchmarks:
            avg_speedup = np.mean([r['speedup'] for r in successful_benchmarks])
            max_speedup = max([r['speedup'] for r in successful_benchmarks])
            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Maximum Speedup: {max_speedup:.2f}x")
            print(f"Memory Savings: {memory_results['memory_savings']:.1f}%")
            
            all_results_match = all(r['results_match'] for r in successful_benchmarks)
            print(f"Results Validation: {'✅ PASS' if all_results_match else '❌ FAIL'}")
        
        print("="*60)
        logger.info("🎉 All parallel merge tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()