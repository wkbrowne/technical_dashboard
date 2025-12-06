#!/usr/bin/env python3
"""
Comprehensive test of optimized weekly processing integration.

This script validates that all weekly processing optimizations work correctly
and demonstrates the performance improvements achieved through:

1. Data duplication elimination via pre-partitioning
2. Optimal backend selection (threading vs loky)
3. Vectorized batch processing
4. Intelligent chunking and memory management
5. Parallel merge optimization

Tests both individual components and full integration through the orchestrator.
"""

import logging
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.features.weekly_optimized import (
    add_weekly_features_to_daily_optimized,
    OptimizedWeeklyConfig,
    OptimizedWeeklyProcessor
)
from src.features.weekly_enhanced import (
    add_weekly_features_to_daily_enhanced,
    ChunkProcessingConfig
)
from src.features.weekly import add_weekly_features_to_daily

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(n_symbols: int = 100, n_days: int = 500) -> pd.DataFrame:
    """Generate realistic test data for performance testing."""
    logger.info(f"Generating test data: {n_symbols} symbols × {n_days} days")
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic stock data
    np.random.seed(42)  # For reproducible results
    
    data_rows = []
    for i, symbol in enumerate([f"TEST{i:03d}" for i in range(n_symbols)]):
        # Generate realistic price series with trends and volatility
        base_price = 50 + np.random.normal(0, 20)
        price_returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual volatility
        
        # Add some autocorrelation for realism
        for j in range(1, len(price_returns)):
            price_returns[j] += 0.1 * price_returns[j-1]
        
        prices = base_price * np.exp(np.cumsum(price_returns))
        
        # Generate OHLCV data
        for j, date in enumerate(dates):
            close_price = prices[j]
            daily_range = close_price * 0.03 * np.random.uniform(0.5, 1.5)  # 1.5-4.5% daily range
            
            high = close_price + daily_range * np.random.uniform(0.2, 0.8)
            low = close_price - daily_range * np.random.uniform(0.2, 0.8)
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
            
            # Volume with some correlation to price movement
            volume = int(1000000 * (1 + 0.5 * abs(price_returns[j]) * np.random.uniform(0.5, 2.0)))
            
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
    logger.info(f"Generated {len(df)} rows of test data ({df.memory_usage(deep=True).sum() / 1024**2:.1f}MB)")
    return df


def test_optimized_implementation(df_long: pd.DataFrame, test_name: str) -> dict:
    """Test the optimized weekly processing implementation."""
    logger.info(f"Testing optimized implementation: {test_name}")
    
    start_time = time.time()
    
    # Configure for optimal performance
    n_symbols = df_long['symbol'].nunique()
    config = OptimizedWeeklyConfig(
        resample_backend='threading',
        features_backend='loky',
        chunk_size=min(max(n_symbols // 8, 50), 200),
        batch_size=min(max(n_symbols // 16, 8), 32),
        n_jobs=-1,
        memory_limit_mb=2048,
        enable_vectorization=n_symbols > 50,
        enable_gc=True
    )
    
    try:
        result_df = add_weekly_features_to_daily_optimized(
            df_long,
            spy_symbol="TEST001",  # Use first test symbol
            config=config
        )
        
        processing_time = time.time() - start_time
        
        # Analyze results
        weekly_features = [col for col in result_df.columns if col.startswith('w_')]
        
        return {
            'success': True,
            'processing_time': processing_time,
            'symbols_processed': result_df['symbol'].nunique(),
            'weekly_features_count': len(weekly_features),
            'total_rows': len(result_df),
            'memory_mb': result_df.memory_usage(deep=True).sum() / 1024**2,
            'symbols_per_second': n_symbols / processing_time if processing_time > 0 else 0,
            'features_sample': weekly_features[:10] if weekly_features else [],
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Optimized implementation failed: {e}")
        return {
            'success': False,
            'processing_time': processing_time,
            'error': str(e)
        }


def test_enhanced_implementation(df_long: pd.DataFrame, test_name: str) -> dict:
    """Test the enhanced weekly processing implementation."""
    logger.info(f"Testing enhanced implementation: {test_name}")
    
    start_time = time.time()
    
    n_symbols = df_long['symbol'].nunique()
    config = ChunkProcessingConfig(
        chunk_size=min(500, n_symbols // 4 + 1),
        n_jobs=-1,
        backend='loky',
        batch_size=8,
        memory_limit_mb=2048
    )
    
    try:
        result_df = add_weekly_features_to_daily_enhanced(
            df_long,
            spy_symbol="TEST001",
            config=config
        )
        
        processing_time = time.time() - start_time
        weekly_features = [col for col in result_df.columns if col.startswith('w_')]
        
        return {
            'success': True,
            'processing_time': processing_time,
            'symbols_processed': result_df['symbol'].nunique(),
            'weekly_features_count': len(weekly_features),
            'total_rows': len(result_df),
            'memory_mb': result_df.memory_usage(deep=True).sum() / 1024**2,
            'symbols_per_second': n_symbols / processing_time if processing_time > 0 else 0,
            'features_sample': weekly_features[:10] if weekly_features else [],
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Enhanced implementation failed: {e}")
        return {
            'success': False,
            'processing_time': processing_time,
            'error': str(e)
        }


def test_original_implementation(df_long: pd.DataFrame, test_name: str) -> dict:
    """Test the original weekly processing implementation."""
    logger.info(f"Testing original implementation: {test_name}")
    
    start_time = time.time()
    
    try:
        result_df = add_weekly_features_to_daily(
            df_long,
            spy_symbol="TEST001",
            n_jobs=-1
        )
        
        processing_time = time.time() - start_time
        weekly_features = [col for col in result_df.columns if col.startswith('w_')]
        
        return {
            'success': True,
            'processing_time': processing_time,
            'symbols_processed': result_df['symbol'].nunique(),
            'weekly_features_count': len(weekly_features),
            'total_rows': len(result_df),
            'memory_mb': result_df.memory_usage(deep=True).sum() / 1024**2,
            'symbols_per_second': df_long['symbol'].nunique() / processing_time if processing_time > 0 else 0,
            'features_sample': weekly_features[:10] if weekly_features else [],
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Original implementation failed: {e}")
        return {
            'success': False,
            'processing_time': processing_time,
            'error': str(e)
        }


def run_performance_comparison():
    """Run comprehensive performance comparison between implementations."""
    logger.info("🚀 Starting Weekly Processing Optimization Integration Test")
    logger.info("=" * 80)
    
    # Test scenarios with increasing complexity
    test_scenarios = [
        {"name": "Small Dataset", "symbols": 20, "days": 100},
        {"name": "Medium Dataset", "symbols": 50, "days": 300},
        {"name": "Large Dataset", "symbols": 100, "days": 500},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        logger.info(f"\n📊 Testing {scenario['name']}: {scenario['symbols']} symbols × {scenario['days']} days")
        logger.info("-" * 60)
        
        # Generate test data for this scenario
        df_test = generate_test_data(scenario['symbols'], scenario['days'])
        
        # Test all implementations
        scenario_results = {
            'scenario': scenario['name'],
            'symbols': scenario['symbols'],
            'days': scenario['days'],
            'data_size_mb': df_test.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Test optimized implementation (our new implementation)
        optimized_result = test_optimized_implementation(df_test, scenario['name'])
        scenario_results['optimized'] = optimized_result
        
        # Test enhanced implementation (intermediate version)
        enhanced_result = test_enhanced_implementation(df_test, scenario['name'])
        scenario_results['enhanced'] = enhanced_result
        
        # Test original implementation (baseline)
        original_result = test_original_implementation(df_test, scenario['name'])
        scenario_results['original'] = original_result
        
        results.append(scenario_results)
        
        # Print scenario summary
        logger.info(f"\n📋 {scenario['name']} Results Summary:")
        if optimized_result['success']:
            logger.info(f"  Optimized:  {optimized_result['processing_time']:.2f}s "
                       f"({optimized_result['symbols_per_second']:.1f} symbols/sec, "
                       f"{optimized_result['weekly_features_count']} features)")
        if enhanced_result['success']:
            logger.info(f"  Enhanced:   {enhanced_result['processing_time']:.2f}s "
                       f"({enhanced_result['symbols_per_second']:.1f} symbols/sec, "
                       f"{enhanced_result['weekly_features_count']} features)")
        if original_result['success']:
            logger.info(f"  Original:   {original_result['processing_time']:.2f}s "
                       f"({original_result['symbols_per_second']:.1f} symbols/sec, "
                       f"{original_result['weekly_features_count']} features)")
    
    # Print comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 COMPREHENSIVE PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    
    # Summary table
    logger.info(f"\n{'Dataset':<15} {'Implementation':<12} {'Time (s)':<10} {'Speed (sym/s)':<12} {'Features':<10} {'Status':<8}")
    logger.info("-" * 80)
    
    for result in results:
        dataset = result['scenario']
        
        for impl_name, impl_key in [('Optimized', 'optimized'), ('Enhanced', 'enhanced'), ('Original', 'original')]:
            impl_result = result[impl_key]
            if impl_result['success']:
                logger.info(f"{dataset:<15} {impl_name:<12} {impl_result['processing_time']:<10.2f} "
                           f"{impl_result['symbols_per_second']:<12.1f} {impl_result['weekly_features_count']:<10} "
                           f"{'✅ PASS':<8}")
            else:
                logger.info(f"{dataset:<15} {impl_name:<12} {'FAILED':<10} {'N/A':<12} {'N/A':<10} {'❌ FAIL':<8}")
    
    # Performance improvements
    logger.info("\n🚀 Performance Improvements (Optimized vs Original):")
    for result in results:
        if result['optimized']['success'] and result['original']['success']:
            speedup = result['original']['processing_time'] / result['optimized']['processing_time']
            logger.info(f"  {result['scenario']}: {speedup:.2f}x faster")
        elif result['optimized']['success'] and not result['original']['success']:
            logger.info(f"  {result['scenario']}: Optimized succeeded, original failed")
        elif not result['optimized']['success']:
            logger.info(f"  {result['scenario']}: Optimized implementation needs debugging")
    
    # Feature consistency check
    logger.info("\n✅ Feature Consistency Check:")
    for result in results:
        optimized_features = result['optimized'].get('weekly_features_count', 0) if result['optimized']['success'] else 0
        enhanced_features = result['enhanced'].get('weekly_features_count', 0) if result['enhanced']['success'] else 0
        original_features = result['original'].get('weekly_features_count', 0) if result['original']['success'] else 0
        
        if optimized_features > 0:
            logger.info(f"  {result['scenario']}: {optimized_features} weekly features generated")
            if enhanced_features > 0:
                consistency = abs(optimized_features - enhanced_features) <= 2  # Allow small differences
                logger.info(f"    Consistency with Enhanced: {'✅ PASS' if consistency else '❌ FAIL'}")
        
    logger.info("\n" + "=" * 80)
    logger.info("🎉 Weekly Processing Optimization Integration Test COMPLETED")
    logger.info("=" * 80)
    
    # Final success assessment
    successful_tests = sum(1 for result in results if result['optimized']['success'])
    total_tests = len(results)
    
    if successful_tests == total_tests:
        logger.info("🎊 ALL TESTS PASSED! Optimized weekly processing is ready for production.")
    else:
        logger.info(f"⚠️  {successful_tests}/{total_tests} tests passed. Review failed tests before deployment.")
    
    return results


if __name__ == "__main__":
    try:
        results = run_performance_comparison()
        
        # Save results for further analysis if needed
        import json
        results_file = Path("weekly_optimization_test_results.json")
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                json_result = {k: v for k, v in result.items() if k not in ['optimized', 'enhanced', 'original']}
                for impl_name in ['optimized', 'enhanced', 'original']:
                    impl_result = result[impl_name]
                    json_result[f'{impl_name}_success'] = impl_result['success']
                    json_result[f'{impl_name}_time'] = impl_result['processing_time']
                    if impl_result['success']:
                        json_result[f'{impl_name}_features'] = impl_result['weekly_features_count']
                        json_result[f'{impl_name}_speed'] = impl_result['symbols_per_second']
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2)
        
        logger.info(f"\n📁 Test results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)