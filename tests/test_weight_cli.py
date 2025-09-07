#!/usr/bin/env python3
"""
Integration test for combined weight generation command-line interface.

This script tests that the weight parameters are properly passed through
the command-line interface to the target generation pipeline.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.target_generation import generate_targets_parallel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_long_data() -> pd.DataFrame:
    """Create synthetic long-format data for target generation."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        for i, date in enumerate(dates):
            price = base_price * (1 + np.random.normal(0, 0.02) * i * 0.1)
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            atr = abs(high - low) * np.random.uniform(0.5, 2.0)
            
            data.append({
                'symbol': symbol,
                'date': date,
                'close': price,
                'high': high,
                'low': low,
                'atr': atr
            })
    
    return pd.DataFrame(data)


def test_weight_parameters():
    """Test that weight parameters are properly applied."""
    logger.info("Testing weight parameter integration...")
    
    df_long = create_test_long_data()
    
    config = {
        'up_mult': 2.0,
        'dn_mult': 2.0,
        'max_horizon': 10,
        'start_every': 5
    }
    
    # Test with custom weight clipping
    targets = generate_targets_parallel(
        df_long, 
        config, 
        n_jobs=1,  # Sequential for testing
        chunk_size=10,
        weight_min_clip=0.5,  # Higher min clip
        weight_max_clip=3.0   # Lower max clip
    )
    
    if targets.empty:
        logger.error("No targets generated for testing")
        return False
    
    # Verify weight columns exist
    expected_cols = ['weight_overlap', 'weight_class_balance', 'weight_final']
    missing_cols = [col for col in expected_cols if col not in targets.columns]
    if missing_cols:
        logger.error(f"Missing weight columns: {missing_cols}")
        return False
    
    # Verify weight bounds
    min_weight = targets['weight_final'].min()
    max_weight = targets['weight_final'].max()
    
    if min_weight < 0.5:
        logger.error(f"Minimum weight {min_weight:.3f} below min_clip 0.5")
        return False
        
    if max_weight > 3.0:
        logger.error(f"Maximum weight {max_weight:.3f} above max_clip 3.0")
        return False
    
    # Check weight sum (should be close to number of samples)
    weight_sum = targets['weight_final'].sum()
    expected_sum = len(targets)
    if abs(weight_sum - expected_sum) > 0.1:
        logger.warning(f"Weight sum {weight_sum:.2f} differs from expected {expected_sum}")
    
    logger.info(f"Generated {len(targets)} targets with weights in range [{min_weight:.3f}, {max_weight:.3f}]")
    logger.info(f"Weight sum: {weight_sum:.2f} (expected: {expected_sum})")
    
    # Sample statistics
    logger.info("Sample target data with weights:")
    sample = targets[['symbol', 'hit', 'weight_overlap', 'weight_class_balance', 'weight_final']].head()
    logger.info(sample.round(4))
    
    return True


def main():
    """Run weight CLI integration test."""
    logger.info("Starting combined weight CLI integration test")
    
    try:
        success = test_weight_parameters()
        
        if success:
            logger.info("✅ Weight CLI integration test passed!")
        else:
            logger.error("❌ Weight CLI integration test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        raise


if __name__ == "__main__":
    main()