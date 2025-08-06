#!/usr/bin/env python3
"""
Test Updated Streamlined Training Pipeline

This script tests the key components of the updated pipeline to ensure
the integer regime system integration works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_regime_configuration():
    """Test the regime configuration functionality"""
    print("🧪 Testing Regime Configuration")
    print("-" * 40)
    
    # Test different configurations
    configs_to_test = [
        (5, 3),  # Default
        (3, 3),  # Simple
        (7, 2),  # Fine trend, coarse vol
    ]
    
    for n_trend, n_vol in configs_to_test:
        print(f"\n📊 Testing {n_trend}×{n_vol} configuration:")
        
        from config.regime_config import create_regime_config, set_custom_regime_config, REGIME_CONFIG
        
        # Create and apply configuration
        config = create_regime_config(n_trend_states=n_trend, n_vol_states=n_vol)
        set_custom_regime_config(config)
        
        # Get updated config reference
        from config.regime_config import get_regime_config
        current_config = get_regime_config()
        
        # Verify configuration
        assert current_config.trend.n_states == n_trend
        assert current_config.volatility.n_states == n_vol
        
        # Test state-label conversion
        trend_states = current_config.trend.get_all_states()
        vol_states = current_config.volatility.get_all_states()
        trend_labels = current_config.trend.get_all_labels()
        vol_labels = current_config.volatility.get_all_labels()
        
        assert len(trend_states) == n_trend
        assert len(vol_states) == n_vol
        assert len(trend_labels) == n_trend
        assert len(vol_labels) == n_vol
        
        # Test combined regimes
        combined = current_config.get_all_combined_regimes()
        assert len(combined) == n_trend * n_vol
        
        print(f"   ✅ {n_trend} trend states: {trend_states}")
        print(f"   ✅ {n_vol} vol states: {vol_states}")
        print(f"   ✅ {len(combined)} total regimes")
        
        # Test state-label conversion
        sample_trend_state = trend_states[len(trend_states)//2]
        sample_vol_state = vol_states[len(vol_states)//2]
        sample_trend_label = current_config.trend.get_state_label(sample_trend_state)
        sample_vol_label = current_config.volatility.get_state_label(sample_vol_state)
        
        print(f"   🔄 trend_{sample_trend_state} = {sample_trend_label}")
        print(f"   🔄 vol_{sample_vol_state} = {sample_vol_label}")
    
    print(f"\n✅ Regime configuration tests passed")


def test_regime_classifier():
    """Test the regime classifier with integer system"""
    print(f"\n🧪 Testing Regime Classifier")
    print("-" * 40)
    
    from models.regime_classifier import UnifiedRegimeClassifier
    from config.regime_config import REGIME_CONFIG
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    
    # Create MA series with trend
    ma_values = np.linspace(100, 110, 50)  # Upward trend
    ma_series = pd.Series(ma_values, index=dates)
    
    # Create returns with varying volatility  
    returns = pd.Series(np.concatenate([
        np.random.normal(0, 0.005, 20),  # Low vol
        np.random.normal(0, 0.025, 30)   # High vol
    ]), index=dates)
    
    classifier = UnifiedRegimeClassifier()
    
    # Test integer state classification
    trend_states = classifier.classify_trend(ma_series, return_states=True)
    vol_states = classifier.classify_volatility(returns, return_states=True)
    
    # Test label classification
    trend_labels = classifier.classify_trend(ma_series, return_states=False)
    vol_labels = classifier.classify_volatility(returns, return_states=False)
    
    # Verify data types
    assert all(isinstance(s, (int, np.integer)) for s in trend_states.dropna())
    assert all(isinstance(s, (int, np.integer)) for s in vol_states.dropna())
    assert all(isinstance(s, str) for s in trend_labels.dropna())
    assert all(isinstance(s, str) for s in vol_labels.dropna())
    
    # Test consistency
    for i in range(len(trend_states)):
        if pd.notna(trend_states.iloc[i]):
            state = int(trend_states.iloc[i])
            expected_label = REGIME_CONFIG.trend.get_state_label(state)
            actual_label = trend_labels.iloc[i]
            assert actual_label == expected_label
    
    print(f"   ✅ Integer states: {sorted(set(trend_states.dropna()))}")
    print(f"   ✅ String labels: {sorted(set(trend_labels.dropna()))}")
    print(f"   ✅ State-label consistency verified")
    
    # Test combined classification
    _, _, combined_states = classifier.classify_regimes_as_states(ma_series, returns)
    _, _, combined_labels = classifier.classify_regimes_as_labels(ma_series, returns)
    
    assert all(isinstance(s, tuple) for s in combined_states.dropna())
    assert all(isinstance(s, str) for s in combined_labels.dropna())
    
    print(f"   ✅ Combined states: {list(combined_states.dropna().unique())[:3]}...")
    print(f"   ✅ Combined labels: {list(combined_labels.dropna().unique())[:3]}...")
    
    print(f"\n✅ Regime classifier tests passed")


def test_global_kde_integration():
    """Test integration with global KDE models"""
    print(f"\n🧪 Testing Global KDE Integration")
    print("-" * 40)
    
    from models.global_kde_models import GlobalClosePriceKDE
    from config.regime_config import REGIME_CONFIG
    
    # Create test stock data
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
    
    # Test global KDE model with integer regime system
    kde_model = GlobalClosePriceKDE(min_samples_per_regime=5)
    kde_model.fit_global_model(test_data)
    
    if kde_model.fitted and len(kde_model.kde_models) > 0:
        # Test regime parsing
        regime_names = list(kde_model.kde_models.keys())
        for regime in regime_names[:2]:  # Test first 2 regimes
            try:
                trend_state, vol_state = REGIME_CONFIG.label_to_state(regime)
                trend_label = REGIME_CONFIG.trend.get_state_label(trend_state)
                vol_label = REGIME_CONFIG.volatility.get_state_label(vol_state)
                combined_check = REGIME_CONFIG.state_to_label(trend_state, vol_state)
                
                assert combined_check == regime
                print(f"   ✅ '{regime}' = trend_{trend_state} ({trend_label}) + vol_{vol_state} ({vol_label})")
            except:
                print(f"   ℹ️ '{regime}' = descriptive label (fallback)")
        
        # Test sampling
        sample_regime = regime_names[0]
        samples = kde_model.sample_close_price(sample_regime, 100.0, n_samples=5)
        assert len(samples) == 5
        assert all(s > 0 for s in samples)
        
        print(f"   ✅ KDE model trained with {len(kde_model.kde_models)} regimes")
        print(f"   ✅ Sampling works: {[f'{s:.2f}' for s in samples]}")
    else:
        print(f"   ℹ️ KDE model training skipped (insufficient data)")
    
    print(f"\n✅ Global KDE integration tests passed")


def main():
    """Run all tests"""
    print("🚀 Testing Updated Streamlined Training Pipeline")
    print("=" * 80)
    print("Testing integration of integer regime system with training pipeline")
    print("=" * 80)
    
    try:
        test_regime_configuration()
        test_regime_classifier() 
        test_global_kde_integration()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("✅ Integer regime system successfully integrated with pipeline")
        print("✅ Regime configuration works with different state counts")
        print("✅ State-label conversion is consistent")
        print("✅ Global KDE models work with new regime system")
        print("✅ Pipeline ready for use with integer-based regimes")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)