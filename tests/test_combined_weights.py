#!/usr/bin/env python3
"""
Test script for validating combined weight generation in triple barrier targets.

This script tests the overlap weighting, class balance weighting, and combined multiplicative
weighting system with clipping and normalization.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.target_generation import _add_combined_weights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_targets() -> pd.DataFrame:
    """Create a synthetic targets DataFrame for testing weights."""
    np.random.seed(42)
    
    # Create diverse overlap and class scenarios
    data = []
    
    # Symbol A: High overlap, balanced classes
    for i in range(100):
        data.append({
            'symbol': 'A',
            't0': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            'hit': np.random.choice([-1, 0, 1], p=[0.33, 0.34, 0.33]),  # Balanced
            'n_overlapping_trajs': np.random.choice([5, 6, 7, 8], p=[0.25, 0.25, 0.25, 0.25]),  # High overlap
        })
    
    # Symbol B: Low overlap, imbalanced classes  
    for i in range(80):
        data.append({
            'symbol': 'B', 
            't0': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            'hit': np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]),  # Very imbalanced (mostly 0s)
            'n_overlapping_trajs': np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]),  # Low overlap
        })
    
    # Symbol C: Medium overlap, moderate imbalance
    for i in range(60):
        data.append({
            'symbol': 'C',
            't0': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            'hit': np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]),  # Moderate imbalance
            'n_overlapping_trajs': np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2]),  # Medium overlap
        })
    
    df = pd.DataFrame(data)
    return df


def test_weight_components():
    """Test individual weight components."""
    logger.info("Testing individual weight components...")
    
    targets_df = create_test_targets()
    
    # Test with default clipping
    result = _add_combined_weights(targets_df, weight_min_clip=0.01, weight_max_clip=10.0)
    
    # Verify all weight columns exist
    required_columns = ['weight_overlap', 'weight_class_balance', 'weight_final']
    for col in required_columns:
        assert col in result.columns, f"Missing weight column: {col}"
    
    # Test overlap weights
    assert (result['weight_overlap'] > 0).all(), "All overlap weights should be positive"
    # Higher overlap should give lower weights
    high_overlap = result[result['n_overlapping_trajs'] > 5]['weight_overlap']
    low_overlap = result[result['n_overlapping_trajs'] <= 2]['weight_overlap']
    assert high_overlap.mean() < low_overlap.mean(), "Higher overlap should produce lower weights"
    
    # Test class balance weights
    assert (result['weight_class_balance'] > 0).all(), "All class balance weights should be positive"
    
    # Test final weights properties
    assert (result['weight_final'] >= 0.01).all(), "All weights should be >= min_clip"
    assert (result['weight_final'] <= 10.0).all(), "All weights should be <= max_clip"
    
    # Test normalization (should sum to number of samples)
    expected_sum = len(result)
    actual_sum = result['weight_final'].sum()
    assert abs(actual_sum - expected_sum) < 0.1, f"Weight sum should be ~{expected_sum}, got {actual_sum:.2f}"
    
    logger.info("✅ Weight component tests passed")
    return result


def test_class_balance_effectiveness():
    """Test that class balance weights properly counter class imbalance."""
    logger.info("Testing class balance effectiveness...")
    
    targets_df = create_test_targets()
    result = _add_combined_weights(targets_df)
    
    # Group by class and analyze weights
    class_stats = result.groupby('hit').agg({
        'weight_class_balance': ['mean', 'count'],
        'weight_final': 'mean'
    }).round(4)
    
    logger.info("Class balance weight statistics by hit class:")
    logger.info(class_stats)
    
    # Rare classes should have higher class balance weights
    class_counts = result['hit'].value_counts()
    rare_class = class_counts.idxmin()
    common_class = class_counts.idxmax()
    
    rare_class_weight = result[result['hit'] == rare_class]['weight_class_balance'].mean()
    common_class_weight = result[result['hit'] == common_class]['weight_class_balance'].mean()
    
    assert rare_class_weight > common_class_weight, \
        f"Rare class ({rare_class}) should have higher weight than common class ({common_class})"
    
    logger.info(f"Rare class {rare_class}: avg class weight = {rare_class_weight:.3f}")
    logger.info(f"Common class {common_class}: avg class weight = {common_class_weight:.3f}")
    logger.info("✅ Class balance effectiveness test passed")


def test_extreme_clipping():
    """Test weight clipping with extreme values."""
    logger.info("Testing extreme weight clipping...")
    
    # Create data that would produce extreme weights
    extreme_data = []
    
    # Very rare class with very low overlap (should produce very high weights)
    for i in range(5):
        extreme_data.append({
            'symbol': 'EXTREME',
            't0': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            'hit': 1,  # Rare class
            'n_overlapping_trajs': 1,  # Very low overlap
        })
    
    # Very common class with very high overlap (should produce very low weights)
    for i in range(200):
        extreme_data.append({
            'symbol': 'COMMON',
            't0': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            'hit': 0,  # Common class
            'n_overlapping_trajs': 20,  # Very high overlap
        })
    
    extreme_df = pd.DataFrame(extreme_data)
    
    # Test with tight clipping
    result = _add_combined_weights(extreme_df, weight_min_clip=0.1, weight_max_clip=5.0)
    
    # Verify clipping worked
    assert (result['weight_final'] >= 0.1).all(), "All weights should be >= min_clip"
    assert (result['weight_final'] <= 5.0).all(), "All weights should be <= max_clip"
    
    # Check that extreme values were actually clipped
    raw_weights = result['weight_overlap'] * result['weight_class_balance']
    clipped_count = sum((raw_weights < 0.1) | (raw_weights > 5.0))
    logger.info(f"Clipped {clipped_count} extreme weights out of {len(result)}")
    
    logger.info("✅ Extreme clipping test passed")


def test_empty_data_handling():
    """Test handling of edge cases."""
    logger.info("Testing edge case handling...")
    
    # Test empty DataFrame
    empty_df = pd.DataFrame()
    result = _add_combined_weights(empty_df)
    assert result.empty, "Empty input should return empty DataFrame"
    
    # Test DataFrame missing 'hit' column
    no_hit_df = pd.DataFrame({
        'symbol': ['A', 'B'],
        'n_overlapping_trajs': [1, 2]
    })
    result = _add_combined_weights(no_hit_df)
    assert 'weight_overlap' in result.columns, "Should still have overlap weights"
    assert 'weight_final' in result.columns, "Should have final weights (fallback to overlap only)"
    
    logger.info("✅ Edge case handling test passed")


def validate_weight_distributions(result: pd.DataFrame):
    """Validate and visualize weight distributions."""
    logger.info("Validating weight distributions...")
    
    # Basic statistics
    stats = {
        'overlap': result['weight_overlap'].describe(),
        'class_balance': result['weight_class_balance'].describe(), 
        'final': result['weight_final'].describe()
    }
    
    logger.info("Weight distribution statistics:")
    for weight_type, stat in stats.items():
        logger.info(f"\n{weight_type.upper()} weights:")
        logger.info(f"  min: {stat['min']:.4f}, max: {stat['max']:.4f}")
        logger.info(f"  mean: {stat['mean']:.4f}, std: {stat['std']:.4f}")
    
    # Check for outliers (values beyond 3 standard deviations)
    final_mean = result['weight_final'].mean()
    final_std = result['weight_final'].std()
    outliers = result[abs(result['weight_final'] - final_mean) > 3 * final_std]
    
    if len(outliers) > 0:
        logger.warning(f"Found {len(outliers)} weight outliers (>3 std devs from mean)")
    else:
        logger.info("No extreme weight outliers detected")
    
    # Correlation analysis
    correlation = result[['weight_overlap', 'weight_class_balance', 'weight_final']].corr()
    logger.info("\nWeight correlations:")
    logger.info(correlation.round(3))
    
    logger.info("✅ Weight distribution validation completed")


def main():
    """Run all weight generation tests."""
    logger.info("Starting combined weight generation tests")
    
    try:
        # Run individual tests
        result = test_weight_components()
        test_class_balance_effectiveness() 
        test_extreme_clipping()
        test_empty_data_handling()
        
        # Validate overall distributions
        validate_weight_distributions(result)
        
        logger.info("🎉 All combined weight tests passed!")
        
        # Sample output for inspection
        logger.info("\nSample weight data:")
        sample = result.groupby(['symbol', 'hit']).agg({
            'weight_overlap': 'mean',
            'weight_class_balance': 'mean', 
            'weight_final': 'mean',
            'n_overlapping_trajs': 'mean'
        }).round(4)
        logger.info(sample.head(10))
        
    except Exception as e:
        logger.error(f"❌ Weight test failed: {e}")
        raise


if __name__ == "__main__":
    main()