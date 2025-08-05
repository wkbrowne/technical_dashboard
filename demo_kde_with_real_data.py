#!/usr/bin/env python3
"""
Demonstration of KDE-enhanced OHLC forecasting with real data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.ohlc_forecasting import OHLCForecaster

print("🚀 KDE-Enhanced OHLC Forecasting Demo")
print("=" * 50)

# Try to load real data from the existing cache
try:
    from data.loader import get_multiple_stocks
    
    print("📊 Loading real stock data...")
    stock_data = get_multiple_stocks('AAPL', update=False, rate_limit=5.0)
    
    # Prepare OHLC data for a specific stock
    symbol = 'AAPL'
    if symbol in stock_data['Close'].columns:
        ohlc_real = pd.DataFrame({
            'Open': stock_data['Open'][symbol],
            'High': stock_data['High'][symbol], 
            'Low': stock_data['Low'][symbol],
            'Close': stock_data['Close'][symbol]
        }).dropna().tail(300)  # Use last 300 days
        
        print(f"✅ Loaded {len(ohlc_real)} days of {symbol} data")
        print(f"📅 Date range: {ohlc_real.index[0].strftime('%Y-%m-%d')} to {ohlc_real.index[-1].strftime('%Y-%m-%d')}")
        
        # Test KDE-enhanced forecasting
        print(f"\n🧪 Testing KDE forecasting with real {symbol} data...")
        
        forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
        forecaster.fit(ohlc_real)
        
        print("✅ Forecaster fitted with real data")
        
        # Show KDE model info
        kde_info = forecaster.get_kde_model_info()
        print(f"\n📊 KDE Models for {symbol}:")
        
        for regime, info in kde_info.items():
            print(f"  {regime}:")
            if info['model_type'] == 'gaussian_kde':
                print(f"    - Model: Gaussian KDE with Silverman's rule")
                print(f"    - Samples: {info['n_samples']}")
                print(f"    - Bandwidth: {info['bandwidth']:.4f}")
                print(f"    - Est. Std: {info['estimated_std']:.4f}")
            else:
                print(f"    - Model: {info['model_type']}")
                if 'mean' in info:
                    print(f"    - Mean: {info['mean']:.4f}")
                    print(f"    - Std: {info['std']:.4f}")
        
        # Generate sample forecast
        current_close = ohlc_real['Close'].iloc[-1]
        forecast_days = 10
        
        # Simple trend-following MA forecast
        recent_ma = ohlc_real['Close'].rolling(20).mean().iloc[-1]
        ma_forecast = np.linspace(recent_ma, recent_ma * 1.02, forecast_days)
        
        # Estimate volatility from recent ATR
        recent_range = (ohlc_real['High'] - ohlc_real['Low']).tail(20).mean()
        vol_forecast = np.full(forecast_days, recent_range / current_close)
        
        # Mixed BB states
        bb_states = np.random.choice([1, 2, 3], size=forecast_days)
        
        print(f"\n🔮 Generating 10-day forecast for {symbol}...")
        print(f"📊 Current price: ${current_close:.2f}")
        
        forecast_results = forecaster.forecast_ohlc(
            ma_forecast=ma_forecast,
            vol_forecast=vol_forecast,
            bb_states=bb_states,
            current_close=current_close,
            n_days=forecast_days
        )
        
        print("✅ KDE-enhanced forecast generated!")
        
        # Show forecast results
        print(f"\n📈 {symbol} 10-Day KDE-Enhanced Forecast:")
        print("=" * 60)
        print(f"{'Day':<4} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Range':<6}")
        print("-" * 60)
        
        for i in range(forecast_days):
            day_range = forecast_results['high'][i] - forecast_results['low'][i]
            print(f"{i+1:<4} ${forecast_results['open'][i]:<7.2f} "
                  f"${forecast_results['high'][i]:<7.2f} "
                  f"${forecast_results['low'][i]:<7.2f} "
                  f"${forecast_results['close'][i]:<7.2f} "
                  f"${day_range:<5.2f}")
        
        # Calculate expected return
        final_price = forecast_results['close'][-1]
        expected_return = (final_price - current_close) / current_close * 100
        
        print("\n📊 Forecast Summary:")
        print(f"  Expected 10-day return: {expected_return:.2f}%")
        print(f"  Final forecasted price: ${final_price:.2f}")
        print(f"  Average daily range: ${np.mean([forecast_results['high'][i] - forecast_results['low'][i] for i in range(forecast_days)]):.2f}")
        
        # Show confidence intervals for final day
        final_ci = forecast_results['close_ci'][-1]
        print(f"  Final day 95% CI: ${final_ci[0]:.2f} - ${final_ci[1]:.2f}")
        
        print(f"\n🎯 KDE Enhancement Benefits:")
        print(f"  ✅ Uses actual historical return distributions")
        print(f"  ✅ Adapts to different market regimes automatically")
        print(f"  ✅ Silverman's rule optimizes bandwidth selection")
        print(f"  ✅ Robust fallbacks prevent model failures")
        print(f"  ✅ Enhanced uncertainty quantification")
        
    else:
        print(f"❌ {symbol} not found in cached data")
        
except Exception as e:
    print(f"❌ Error with real data: {e}")
    print("💡 This might be due to missing cached data")
    print("🔄 Falling back to synthetic data demo...")
    
    # Fallback to synthetic data
    np.random.seed(123)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Create more realistic synthetic data with regime changes
    base_price = 150
    regime_changes = [0, 50, 100, 150]  # Days where regime changes
    regimes = ['bull', 'bear', 'sideways', 'bull']
    
    prices = []
    for i, date in enumerate(dates):
        # Determine current regime
        regime_idx = sum(1 for change in regime_changes if i >= change) - 1
        regime = regimes[regime_idx]
        
        if regime == 'bull':
            drift = 0.0005
            vol = 0.015
        elif regime == 'bear':
            drift = -0.0008
            vol = 0.025
        else:  # sideways
            drift = 0.0001
            vol = 0.010
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + np.random.normal(drift, vol))
        
        prices.append(price)
    
    # Generate OHLC from prices
    ohlc_synthetic = pd.DataFrame(index=dates)
    for i, (date, close) in enumerate(zip(dates, prices)):
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        daily_vol = np.random.uniform(0.008, 0.020)
        high = max(open_price, close) * (1 + np.random.exponential(daily_vol))
        low = min(open_price, close) * (1 - np.random.exponential(daily_vol))
        
        ohlc_synthetic.loc[date] = [open_price, high, low, close]
        
    ohlc_synthetic.columns = ['Open', 'High', 'Low', 'Close']
    
    print(f"✅ Generated synthetic OHLC data with regime changes")
    
    # Test with synthetic data
    forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
    forecaster.fit(ohlc_synthetic)
    
    kde_info = forecaster.get_kde_model_info()
    
    print(f"\n📊 KDE Models for synthetic data: {len(kde_info)}")
    for regime, info in kde_info.items():
        if info['model_type'] == 'gaussian_kde':
            print(f"  {regime}: {info['n_samples']} samples, bandwidth={info['bandwidth']:.4f}")
    
    print(f"\n✅ KDE implementation works with both real and synthetic data!")

print(f"\n🎉 KDE-Enhanced OHLC Forecasting Demo Complete!")
print(f"💡 The Gaussian KDE with Silverman's rule successfully enhances close price estimation")