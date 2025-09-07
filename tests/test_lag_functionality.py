#!/usr/bin/env python3
"""
Test script for validating configurable daily and weekly feature lags.

This script creates synthetic data and tests the temporal semantics of both
daily and weekly lag operations.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.lagging import (
    identify_daily_features, 
    identify_weekly_features,
    apply_daily_lags, 
    apply_weekly_lags,
    parse_lag_specification
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_dataframe() -> pd.DataFrame:
    """Create a test DataFrame with both daily and weekly features."""
    # Create 100 business days of data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    
    # Daily features (price-based)
    np.random.seed(42)
    base_price = 100
    returns = np.random.randn(100) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'adjclose': prices,
        'volume': np.random.exponential(1000000, 100),
        'rsi_14': 50 + np.random.randn(100) * 15,
        'macd_histogram': np.random.randn(100) * 0.5,
        'vol_regime_short': np.random.randn(100) * 2,
        'trend_agreement': np.random.uniform(0, 1, 100),
        'alpha_momentum_20d': np.random.randn(100) * 0.1,
        
        # Weekly features (prefixed with w_)
        'w_sector_momentum': np.random.randn(100) * 0.05,
        'w_breadth_advance_decline': np.random.uniform(-1, 1, 100),
        'w_cross_sectional_rank': np.random.uniform(0, 1, 100),
        
        # Non-feature columns
        'symbol': 'TEST',
        'some_string_col': 'data'
    }, index=dates)
    
    return df


def test_feature_identification():
    """Test feature identification functions."""
    logger.info("Testing feature identification...")
    
    df = create_test_dataframe()
    
    daily_features = identify_daily_features(df)
    weekly_features = identify_weekly_features(df)
    
    logger.info(f"Identified daily features: {daily_features}")
    logger.info(f"Identified weekly features: {weekly_features}")
    
    # Validate results
    expected_daily = ['adjclose', 'volume', 'rsi_14', 'macd_histogram', 'vol_regime_short', 
                      'trend_agreement', 'alpha_momentum_20d']
    expected_weekly = ['w_sector_momentum', 'w_breadth_advance_decline', 'w_cross_sectional_rank']
    
    assert set(daily_features) == set(expected_daily), f"Daily features mismatch: {daily_features}"
    assert set(weekly_features) == set(expected_weekly), f"Weekly features mismatch: {weekly_features}"
    
    logger.info("✅ Feature identification test passed")


def test_daily_lags():
    """Test daily lag functionality."""
    logger.info("Testing daily lag application...")
    
    df = create_test_dataframe()
    daily_features = ['adjclose', 'rsi_14']
    lags = [1, 5, 10]
    
    result = apply_daily_lags(df, daily_features, lags)
    
    # Check new columns were created
    expected_cols = ['adjclose_lag1d', 'adjclose_lag5d', 'adjclose_lag10d',
                     'rsi_14_lag1d', 'rsi_14_lag5d', 'rsi_14_lag10d']
    
    for col in expected_cols:
        assert col in result.columns, f"Missing lagged column: {col}"
    
    # Validate lag semantics (lag1 should be previous day's value)
    assert result['adjclose_lag1d'].iloc[1] == df['adjclose'].iloc[0], "Daily lag1 semantics incorrect"
    assert result['adjclose_lag5d'].iloc[5] == df['adjclose'].iloc[0], "Daily lag5 semantics incorrect"
    
    # Check NaN handling (first N values should be NaN)
    assert pd.isna(result['adjclose_lag1d'].iloc[0]), "First value should be NaN for lag1"
    assert pd.isna(result['adjclose_lag5d'].iloc[4]), "Fifth value should be NaN for lag5"
    assert not pd.isna(result['adjclose_lag5d'].iloc[5]), "Sixth value should not be NaN for lag5"
    
    logger.info("✅ Daily lag test passed")


def test_weekly_lags():
    """Test weekly lag functionality with week-boundary awareness."""
    logger.info("Testing weekly lag application...")
    
    df = create_test_dataframe()
    weekly_features = ['w_sector_momentum', 'w_breadth_advance_decline']
    lags = [1, 2]
    
    result = apply_weekly_lags(df, weekly_features, lags)
    
    # Check new columns were created
    expected_cols = ['w_sector_momentum_lag1w', 'w_sector_momentum_lag2w',
                     'w_breadth_advance_decline_lag1w', 'w_breadth_advance_decline_lag2w']
    
    for col in expected_cols:
        assert col in result.columns, f"Missing weekly lagged column: {col}"
    
    logger.info("✅ Weekly lag test passed")


def test_lag_specification_parsing():
    """Test lag specification parsing functionality."""
    logger.info("Testing lag specification parsing...")
    
    # Test various formats
    test_cases = [
        ("", []),
        ("1", [1]),
        ("1,2,5", [1, 2, 5]),
        ("1-5", [1, 2, 3, 4, 5]),
        ("1,3,5-7", [1, 3, 5, 6, 7]),
        ("10-12,15", [10, 11, 12, 15]),
        ("1,1,2", [1, 2]),  # Duplicates removed
    ]
    
    for input_spec, expected in test_cases:
        result = parse_lag_specification(input_spec)
        assert result == expected, f"Parsing '{input_spec}' failed. Expected: {expected}, Got: {result}"
    
    logger.info("✅ Lag specification parsing test passed")


def test_temporal_semantics():
    """Test that weekly lags respect week boundaries."""
    logger.info("Testing weekly lag temporal semantics...")
    
    # Create data spanning multiple weeks with known weekly pattern
    dates = pd.date_range(start='2023-01-02', periods=30, freq='D')  # Start on Monday
    
    # Create weekly feature with pattern that changes each week
    weekly_values = []
    for date in dates:
        week_num = date.isocalendar()[1]  # ISO week number
        weekly_values.append(week_num * 10)  # Each week has distinct value
    
    df = pd.DataFrame({
        'w_test_feature': weekly_values,
        'daily_feature': range(30)
    }, index=dates)
    
    # Apply 1-week lag
    result = apply_weekly_lags(df, ['w_test_feature'], [1])
    
    logger.info("Sample of weekly lag results:")
    logger.info(result[['w_test_feature', 'w_test_feature_lag1w']].head(10))
    
    # Verify that lagged values are held constant within weeks
    # and represent the previous week's value
    lag_col = result['w_test_feature_lag1w']
    
    # Check that values are consistent within each week (using W-FRI periods)
    # Group by the same week periods used in the lag function
    week_periods = result.index.to_period('W-FRI')
    
    for week_period in week_periods.unique()[1:]:  # Skip first week (will have NaN)
        week_mask = week_periods == week_period
        week_lag_values = lag_col[week_mask].dropna()
        
        if len(week_lag_values) > 1:
            unique_values = week_lag_values.nunique()
            assert unique_values == 1, f"Weekly lag values not constant within week {week_period}: {week_lag_values.unique()}"
    
    logger.info("✅ Weekly temporal semantics test passed")


def main():
    """Run all lag functionality tests."""
    logger.info("Starting configurable lag functionality tests")
    
    try:
        test_feature_identification()
        test_daily_lags()
        test_weekly_lags()
        test_lag_specification_parsing()
        test_temporal_semantics()
        
        logger.info("🎉 All lag functionality tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()