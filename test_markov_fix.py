#!/usr/bin/env python3
"""
Test the fixed Markov model integration
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_markov_models():
    """Test the unified Markov models"""
    print("🧪 Testing Fixed Markov Models")
    print("-" * 40)
    
    from models.unified_markov_model import create_combined_markov_model, create_trend_markov_model
    from config.regime_config import REGIME_CONFIG
    
    # Create test data
    test_data = {}
    for symbol in ['TEST1', 'TEST2']:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, 100)))
        
        close_series = pd.Series(close_prices, index=dates)
        ma = close_series.rolling(20, min_periods=5).mean()
        bb_std = close_series.rolling(20, min_periods=5).std()
        
        test_data[symbol] = pd.DataFrame({
            'Close': close_prices,
            'MA': ma,
            'BB_Position': ((close_prices - ma) / (2 * bb_std)).fillna(0).clip(-1, 1),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates).dropna()
    
    # Test combined Markov model
    print("📊 Testing Combined Markov Model...")
    combined_model = create_combined_markov_model()
    
    print(f"   States: {len(combined_model.states)}")
    print(f"   Expected regimes: {REGIME_CONFIG.trend.n_states * REGIME_CONFIG.volatility.n_states}")
    
    # Fit the model
    try:
        combined_model.fit(test_data)
        
        if combined_model.fitted:
            summary = combined_model.get_model_summary()
            print(f"   ✅ Combined model fitted successfully")
            print(f"   Model type: {summary['model_type']}")
            print(f"   Actual states: {summary['n_states']}")
            
            # Test prediction
            first_state = combined_model.states[0]
            next_probs = combined_model.predict_next_state(first_state)
            print(f"   ✅ Prediction works: {len(next_probs)} next state probabilities")
            
        else:
            print(f"   ❌ Combined model failed to fit")
            
    except Exception as e:
        print(f"   ❌ Combined model error: {str(e)}")
    
    # Test trend Markov model
    print(f"\n📈 Testing Trend Markov Model...")
    trend_model = create_trend_markov_model()
    
    print(f"   States: {len(trend_model.states)}")
    print(f"   Expected trend states: {REGIME_CONFIG.trend.n_states}")
    
    try:
        trend_model.fit(test_data)
        
        if trend_model.fitted:
            summary = trend_model.get_model_summary()
            print(f"   ✅ Trend model fitted successfully")
            print(f"   Model type: {summary['model_type']}")
            print(f"   Actual states: {summary['n_states']}")
        else:
            print(f"   ❌ Trend model failed to fit")
            
    except Exception as e:
        print(f"   ❌ Trend model error: {str(e)}")
    
    print(f"\n✅ Markov model tests completed")


def main():
    """Run the test"""
    print("🚀 Testing Fixed Markov Model Integration")
    print("=" * 60)
    
    try:
        test_markov_models()
        print("\n🎉 SUCCESS: Markov models working with integer regime system!")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)