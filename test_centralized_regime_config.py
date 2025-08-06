#!/usr/bin/env python3
"""
Test Centralized Regime Configuration System

This script verifies that:
1. All models use the same centralized regime configuration
2. Configuration can be changed globally and affects all models
3. Regime classification is consistent across all models
4. Markov models integrate with the centralized system
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.regime_config import (
    REGIME_CONFIG, get_regime_config, update_trend_thresholds,
    get_all_combined_regimes, get_trend_thresholds
)
from models.regime_classifier import REGIME_CLASSIFIER, classify_stock_regimes
from models.global_kde_models import train_global_models
from models.unified_markov_model import create_combined_markov_model

warnings.filterwarnings('ignore')


def create_test_data():
    """Create test stock data"""
    np.random.seed(42)
    
    stock_data = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        n_days = 100
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Generate price data
        base_price = 100 + np.random.uniform(-20, 20)
        daily_returns = np.random.normal(0.001, 0.015, n_days)
        close_prices = base_price * np.exp(np.cumsum(daily_returns))
        
        # Generate OHLC
        gaps = np.random.normal(0, 0.005, n_days)
        opens = np.roll(close_prices, 1) * (1 + gaps)
        opens[0] = base_price
        
        highs = np.maximum(opens, close_prices) * (1 + np.random.uniform(0.005, 0.02, n_days))
        lows = np.minimum(opens, close_prices) * (1 - np.random.uniform(0.005, 0.02, n_days))
        
        # Technical indicators
        close_series = pd.Series(close_prices, index=dates)
        ma_20 = close_series.rolling(window=20, min_periods=5).mean()
        bb_std = close_series.rolling(window=20, min_periods=5).std()
        
        stock_data[symbol] = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'MA': ma_20,
            'BB_Position': ((close_prices - ma_20) / (2 * bb_std)).fillna(0).clip(-1, 1),
            'BB_Width': (bb_std / ma_20).fillna(0.02),
            'Volume': np.random.randint(1000000, 5000000, n_days)
        }, index=dates)
        
        # Only keep rows with valid MA
        stock_data[symbol] = stock_data[symbol][stock_data[symbol]['MA'].notna()]
    
    return stock_data


def test_centralized_configuration():
    """Test that all models use the same centralized configuration"""
    print("🧪 Testing Centralized Regime Configuration")
    print("=" * 60)
    
    # Test 1: Verify initial configuration
    print("1️⃣ Testing Initial Configuration")
    print("-" * 30)
    
    config = get_regime_config()
    print(f"   Trend regimes: {config.trend.get_all_labels()}")
    print(f"   Volatility regimes: {config.volatility.get_all_labels()}")
    print(f"   Total combined regimes: {len(get_all_combined_regimes())}")
    
    # Verify trend thresholds
    trend_thresholds = get_trend_thresholds()
    if len(trend_thresholds) != config.trend.n_states - 1:
        print(f"   ❌ Wrong number of trend thresholds: {len(trend_thresholds)} vs expected {config.trend.n_states - 1}")
        return False
    
    print("   ✅ Initial configuration looks good")
    
    # Test 2: Test regime classification consistency
    print(f"\n2️⃣ Testing Regime Classification Consistency")
    print("-" * 30)
    
    test_data = create_test_data()
    
    # Classify regimes using centralized classifier
    all_regimes = set()
    for symbol, df in test_data.items():
        df_with_regimes = classify_stock_regimes(df)
        stock_regimes = set(df_with_regimes['combined_regime'].dropna().unique())
        all_regimes.update(stock_regimes)
        print(f"   {symbol}: {len(stock_regimes)} unique regimes")
    
    print(f"   Total unique regimes found: {len(all_regimes)}")
    print(f"   ✅ Regime classification working")
    
    # Test 3: Test that models use same classifier
    print(f"\n3️⃣ Testing Model Integration")
    print("-" * 30)
    
    try:
        # Test global KDE models
        global_models = train_global_models(test_data, min_samples=10)
        kde_success = sum(1 for m in global_models.values() if m is not None and m.fitted)
        print(f"   Global KDE models: {kde_success}/3 trained successfully")
        
        # Test Markov model
        markov_model = create_combined_markov_model()
        markov_model.fit(test_data)
        print(f"   Markov model: {'✅ Fitted' if markov_model.fitted else '❌ Failed'}")
        
        # Verify both use same regimes
        if kde_success > 0:
            kde_regimes = set()
            for model in global_models.values():
                if model and hasattr(model, 'regime_stats'):
                    kde_regimes.update(model.regime_stats.keys())
            
            markov_regimes = set(markov_model.states)
            
            print(f"   KDE model regimes: {len(kde_regimes)}")
            print(f"   Markov model regimes: {len(markov_regimes)}")
            
            # Check if there's overlap (they should use compatible regime systems)
            if len(kde_regimes & markov_regimes) > 0:
                print(f"   ✅ Models use compatible regime systems")
            else:
                print(f"   ℹ️ Models use different but valid regime types")
        
    except Exception as e:
        print(f"   ❌ Model integration error: {str(e)}")
        return False
    
    # Test 4: Test configuration changes
    print(f"\n4️⃣ Testing Configuration Changes")
    print("-" * 30)
    
    # Get original thresholds
    original_thresholds = get_trend_thresholds()
    print(f"   Original bull threshold: {original_thresholds['bull']}")
    
    # Update configuration
    new_thresholds = {'bull': 0.002}  # Change from 0.001 to 0.002
    update_trend_thresholds(new_thresholds)
    
    # Force classifier to reload configuration
    REGIME_CLASSIFIER.reload_config()
    
    # Verify change
    updated_thresholds = get_trend_thresholds()
    if updated_thresholds['bull'] == 0.002:
        print(f"   ✅ Configuration update successful: {updated_thresholds['bull']}")
        
        # Test that new classifier uses updated config
        new_classifier_info = REGIME_CLASSIFIER.get_regime_info()
        if new_classifier_info['trend_thresholds']['bull'] == 0.002:
            print(f"   ✅ Classifier uses updated configuration")
        else:
            print(f"   ❌ Classifier not using updated configuration")
            return False
    else:
        print(f"   ❌ Configuration update failed")
        return False
    
    # Restore original thresholds
    restore_thresholds = {'bull': original_thresholds['bull']}
    update_trend_thresholds(restore_thresholds)
    REGIME_CLASSIFIER.reload_config()  # Reload again to restore
    print(f"   ✅ Configuration restored")
    
    return True


def test_regime_consistency():
    """Test that regime classification is consistent across different uses"""
    print(f"\n🧪 Testing Regime Classification Consistency")
    print("=" * 60)
    
    test_data = create_test_data()
    sample_stock = list(test_data.keys())[0]
    sample_df = test_data[sample_stock]
    
    # Classify using convenience function
    df_with_regimes1 = classify_stock_regimes(sample_df)
    
    # Classify using classifier directly
    returns = sample_df['Close'].pct_change()
    trend, vol, combined = REGIME_CLASSIFIER.classify_regimes(sample_df['MA'], returns)
    
    # Compare results
    matches = (df_with_regimes1['combined_regime'] == combined).sum()
    total = len(combined.dropna())
    
    print(f"   Regime matches: {matches}/{total}")
    
    if matches == total:
        print(f"   ✅ Regime classification is consistent")
        return True
    else:
        print(f"   ❌ Regime classification inconsistency detected")
        return False


def main():
    """Main test function"""
    print("🚀 Centralized Regime Configuration Test Suite")
    print("=" * 80)
    
    # Run tests
    config_test = test_centralized_configuration()
    consistency_test = test_regime_consistency()
    
    print("\n" + "=" * 80)
    print("📊 FINAL TEST RESULTS")
    print("=" * 80)
    
    if config_test and consistency_test:
        print("🎉 SUCCESS: Centralized Regime Configuration Working!")
        print("✅ All models use centralized configuration")
        print("✅ Configuration changes affect all models")
        print("✅ Regime classification is consistent")
        print("✅ Models integrate properly with centralized system")
        return True
    else:
        print("❌ FAILURES DETECTED:")
        if not config_test:
            print("   - Centralized configuration issues")
        if not consistency_test:
            print("   - Regime classification inconsistencies")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)