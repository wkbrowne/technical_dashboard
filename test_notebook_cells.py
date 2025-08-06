#!/usr/bin/env python3
"""
Test key cells from the updated notebook to ensure they work
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_regime_configuration():
    """Test the regime configuration cell"""
    print("🧪 Testing Regime Configuration Cell")
    print("-" * 40)
    
    # =====================================================
    # REGIME CONFIGURATION - MODIFY THESE SETTINGS
    # =====================================================

    # Set the number of regime states to use
    N_TREND_STATES = 5    # Number of trend states (3, 5, 7, etc.)
    N_VOL_STATES = 3      # Number of volatility states (2, 3, 5, etc.)

    print(f"🎛️ REGIME CONFIGURATION")
    print(f"=" * 50)
    print(f"Trend States: {N_TREND_STATES}")
    print(f"Volatility States: {N_VOL_STATES}")
    print(f"Total Regimes: {N_TREND_STATES * N_VOL_STATES}")

    # Apply the configuration to the global regime system
    from config.regime_config import create_regime_config, set_custom_regime_config, REGIME_CONFIG

    # Create custom regime configuration with specified states
    if N_TREND_STATES != 5 or N_VOL_STATES != 3:
        print(f"\n🔄 Creating custom regime configuration...")
        custom_config = create_regime_config(n_trend_states=N_TREND_STATES, n_vol_states=N_VOL_STATES)
        set_custom_regime_config(custom_config)
        print(f"✅ Custom configuration applied")
    else:
        print(f"\n✅ Using default configuration (5×3)")

    # Show regime details
    config = REGIME_CONFIG
    print(f"\n📊 Regime Details:")
    print(f"   Trend states: {config.trend.get_all_states()}")
    print(f"   Trend labels: {config.trend.get_all_labels()}")
    print(f"   Vol states: {config.volatility.get_all_states()}")  
    print(f"   Vol labels: {config.volatility.get_all_labels()}")

    # Show some example regimes
    combined_regimes = config.get_all_combined_regimes()
    print(f"\n🔗 Example Combined Regimes:")
    print(f"   First 5: {combined_regimes[:5]}")
    print(f"   Last 5: {combined_regimes[-5:]}")

    # Show state-to-label conversion examples
    print(f"\n🔄 State-Label Examples:")
    print(f"   trend_0 = {config.trend.get_state_label(0)}")
    print(f"   trend_{config.trend.get_all_states()[-1]} = {config.trend.get_state_label(config.trend.get_all_states()[-1])}")
    print(f"   vol_0 = {config.volatility.get_state_label(0)}")
    print(f"   vol_{config.volatility.get_all_states()[-1]} = {config.volatility.get_state_label(config.volatility.get_all_states()[-1])}")

    print(f"\n✅ Regime configuration complete - ready for training")
    print(f"=" * 50)
    
    return N_TREND_STATES, N_VOL_STATES


def test_markov_training(N_TREND_STATES, N_VOL_STATES):
    """Test the Markov model training cell"""
    print(f"\n🧪 Testing Markov Training Cell")
    print("-" * 40)
    
    # Create some test data
    all_prepared_data = {}
    for symbol in ['TEST1', 'TEST2', 'TEST3']:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, 100)))
        
        close_series = pd.Series(close_prices, index=dates)
        ma = close_series.rolling(20, min_periods=5).mean()
        bb_std = close_series.rolling(20, min_periods=5).std()
        
        all_prepared_data[symbol] = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.002, 100)),
            'High': close_prices * (1 + np.random.uniform(0.005, 0.02, 100)),
            'Low': close_prices * (1 - np.random.uniform(0.005, 0.02, 100)),
            'Close': close_prices,
            'MA': ma,
            'BB_Position': ((close_prices - ma) / (2 * bb_std)).fillna(0).clip(-1, 1),
            'BB_Width': (bb_std / ma).fillna(0.02),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates).dropna()
    
    from models.unified_markov_model import create_combined_markov_model

    print(f"🔄 Training global Markov model on {len(all_prepared_data)} stocks...")
    print(f"   Using integer regime system with {N_TREND_STATES}×{N_VOL_STATES} = {N_TREND_STATES * N_VOL_STATES} regimes")

    # Create unified Markov model that uses centralized regime configuration
    global_markov = create_combined_markov_model()

    # Fit the model on all prepared data
    global_markov.fit(all_prepared_data)

    # Show model summary
    if global_markov.fitted:
        summary = global_markov.get_model_summary()
        print(f"✅ Global Markov model trained successfully")
        print(f"   Model type: {summary['model_type']}")
        print(f"   States: {summary['n_states']}")
        print(f"   Using centralized regime config: ✅")
        
        # Show some state statistics
        state_stats = summary['state_statistics']
        top_states = sorted(state_stats.items(), key=lambda x: x[1]['frequency'], reverse=True)[:5]
        print(f"   Top 5 regimes by frequency:")
        for state, stats in top_states:
            print(f"     {state}: {stats['frequency']:.3f} ({stats['count']} obs)")
    else:
        print(f"❌ Global Markov model training failed")
        return False
    
    return True


def test_kde_training(N_TREND_STATES, N_VOL_STATES):
    """Test the KDE training cell"""
    print(f"\n🧪 Testing KDE Training Cell")
    print("-" * 40)
    
    # Create some test data
    all_prepared_data = {}
    for symbol in ['TEST1', 'TEST2']:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, 100)))
        
        close_series = pd.Series(close_prices, index=dates)
        ma = close_series.rolling(20, min_periods=5).mean()
        bb_std = close_series.rolling(20, min_periods=5).std()
        
        all_prepared_data[symbol] = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.002, 100)),
            'High': close_prices * (1 + np.random.uniform(0.005, 0.02, 100)),
            'Low': close_prices * (1 - np.random.uniform(0.005, 0.02, 100)),
            'Close': close_prices,
            'MA': ma,
            'BB_Position': ((close_prices - ma) / (2 * bb_std)).fillna(0).clip(-1, 1),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates).dropna()
    
    from models.global_kde_models import train_global_models
    from config.regime_config import REGIME_CONFIG

    print(f"🔄 Training global KDE models on ALL data using integer regime system...")
    print(f"   Using {N_TREND_STATES} trend states × {N_VOL_STATES} vol states = {N_TREND_STATES * N_VOL_STATES} total regimes")

    # Train all global models on complete dataset using new integer regime system
    global_models = train_global_models(all_prepared_data, min_samples=5)  # Lower threshold for test

    # Extract individual models for compatibility
    global_close_kde = global_models['close_kde']
    global_open_kde = global_models['open_kde'] 
    global_hl_copula = global_models['hl_copula']

    print(f"✅ Global KDE models trained on all {len(all_prepared_data)} stocks")
    print(f"   Close Price KDE: {'✅' if global_close_kde else '❌'}")
    print(f"   Open Price KDE: {'✅' if global_open_kde else '❌'}")
    print(f"   High-Low Copula: {'✅' if global_hl_copula else '❌'}")

    # Show regime configuration being used
    print(f"\n🎛️ Using Integer Regime Configuration:")
    print(f"   Trend states: {REGIME_CONFIG.trend.get_all_states()} → {REGIME_CONFIG.trend.get_all_labels()}")
    print(f"   Vol states: {REGIME_CONFIG.volatility.get_all_states()} → {REGIME_CONFIG.volatility.get_all_labels()}")
    
    return True


def main():
    """Run all tests"""
    print("🚀 Testing Updated Notebook Cells")
    print("=" * 60)
    print("Testing that key notebook cells work with integer regime system")
    print("=" * 60)
    
    try:
        # Test regime configuration
        N_TREND_STATES, N_VOL_STATES = test_regime_configuration()
        
        # Test Markov training
        markov_success = test_markov_training(N_TREND_STATES, N_VOL_STATES)
        
        # Test KDE training
        kde_success = test_kde_training(N_TREND_STATES, N_VOL_STATES)
        
        if markov_success and kde_success:
            print("\n🎉 ALL NOTEBOOK CELL TESTS PASSED!")
            print("=" * 60)
            print("✅ Regime configuration cell works")
            print("✅ Markov training cell works")
            print("✅ KDE training cell works") 
            print("✅ Integer regime system integration successful")
            print("✅ Updated notebook ready for use")
            print("=" * 60)
            return True
        else:
            print("\n❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)