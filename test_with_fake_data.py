#!/usr/bin/env python3
"""
Test sophisticated model integration with fake data to avoid API calls.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('/mnt/a61cc0e8-1b32-4574-a771-4ad77e8faab6/conda/technical_dashboard')

def create_fake_stock_data(symbol='AAPL', days=500):
    """Create fake OHLC data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducibility
    
    # Start price
    initial_price = 150.0
    prices = [initial_price]
    
    # Generate price path
    for i in range(1, days):
        # Random walk with slight upward drift
        daily_return = np.random.normal(0.0005, 0.02)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Create realistic OHLC from close price
        volatility = 0.015
        high = close * (1 + abs(np.random.normal(0, volatility/2)))
        low = close * (1 - abs(np.random.normal(0, volatility/2)))
        
        if i == 0:
            open_price = close
        else:
            # Gap from previous close
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': np.random.randint(50000000, 200000000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df

def patch_data_loader():
    """Patch the data loader to return fake data."""
    def fake_get_multiple_stocks(symbols, interval="1d", update=False, cache_file=None, rate_limit=1.0):
        """Return fake data for testing."""
        print(f"📊 Generating fake data for {symbols} (testing mode)")
        
        result = {
            'Open': pd.DataFrame(),
            'High': pd.DataFrame(),
            'Low': pd.DataFrame(),
            'Close': pd.DataFrame(),
            'Volume': pd.DataFrame()
        }
        
        for symbol in symbols:
            fake_data = create_fake_stock_data(symbol)
            
            for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                result[column][symbol] = fake_data[column]
        
        return result
    
    # Monkey patch the loader
    import src.data.loader
    src.data.loader.get_multiple_stocks = fake_get_multiple_stocks
    
    print("✅ Data loader patched with fake data for testing")

def test_sophisticated_models_with_fake_data():
    """Test sophisticated models with fake data."""
    print("🧪 Testing sophisticated models with fake data...")
    
    # Patch the data loader
    patch_data_loader()
    
    try:
        from src.models.monte_carlo_simple import MonteCarloTrajectoryGenerator, MonteCarloOptionsAnalyzer, StrategyConfig
        
        # Create generator with sophisticated models
        generator = MonteCarloTrajectoryGenerator(n_trajectories=200, symbol='AAPL')
        
        print(f"📊 Model Status: {generator.get_model_status()}")
        
        if generator.get_model_status()['sophisticated_models_available']:
            generator.enable_sophisticated_models('AAPL')
            
            # Generate trajectories
            current_price = 150.0
            print(f"\n🎲 Generating trajectories with current price: ${current_price}")
            
            trajectories = generator.generate_trajectories(
                current_price=current_price,
                max_days=10,
                volatility=0.02
            )
            
            print(f"✅ Generated trajectories: {trajectories.shape}")
            print(f"   Model type: {generator.trajectory_metadata.get('model_type', 'unknown')}")
            
            if 'sophisticated' in generator.trajectory_metadata.get('model_type', ''):
                print("🎉 SUCCESS: Sophisticated models are working with fake data!")
                
                # Test options analysis
                print("\n🧪 Testing options analysis...")
                config = StrategyConfig(
                    covered_call_keep_prob=0.95,
                    long_call_win_prob=0.50,
                    csp_avoid_assignment_prob=0.95,
                    long_put_win_prob=0.50
                )
                
                analyzer = MonteCarloOptionsAnalyzer(generator, config)
                results = analyzer.find_optimal_strikes(current_price, max_dte=5)
                
                if results:
                    print("✅ Options analysis with sophisticated models completed!")
                    
                    # Show sample results
                    for strategy, dte_results in results.items():
                        if 5 in dte_results:  # Show 5DTE results
                            result = dte_results[5]
                            print(f"   {result.strategy_name} 5DTE: Strike ${result.strike_price:.2f}, Success {result.success_probability:.1%}")
                    
                    return True
                else:
                    print("❌ Options analysis failed")
                    return False
            else:
                print("⚠️ Simple model was used instead of sophisticated models")
                return False
        else:
            print("❌ Sophisticated models not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sophisticated models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Sophisticated Models with Fake Data")
    print("=" * 60)
    
    success = test_sophisticated_models_with_fake_data()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Sophisticated models work with fake data!")
        print("   Your ARIMA/GARCH/Copula models are properly integrated!")
    else:
        print("❌ Test failed - check sophisticated model integration")
    print("=" * 60)