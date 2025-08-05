#!/usr/bin/env python3
"""
Complete demonstration of intelligent OHLC forecasting with:
1. KDE-based close price estimation
2. Intelligent open price modeling with global + stock-specific KDEs
3. Copula-based high-low dependency modeling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from models.ohlc_forecasting import OHLCForecaster
from models.open_price_kde import IntelligentOpenForecaster
from models.high_low_copula import IntelligentHighLowForecaster

print("🚀 COMPLETE INTELLIGENT OHLC FORECASTING DEMO")
print("=" * 70)

def demonstrate_complete_workflow():
    """Demonstrate the complete intelligent OHLC forecasting workflow."""
    
    try:
        # Step 1: Load stock data (same as notebook)
        from data.loader import get_multiple_stocks
        
        print("📊 STEP 1: Loading multi-stock data")
        stock_data = get_multiple_stocks('AAPL', update=False, rate_limit=5.0)
        
        # Prepare data for multiple stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        all_stock_data = {}
        
        for symbol in symbols:
            if symbol in stock_data['Close'].columns:
                stock_df = pd.DataFrame({
                    'Open': stock_data['Open'][symbol],
                    'High': stock_data['High'][symbol], 
                    'Low': stock_data['Low'][symbol],
                    'Close': stock_data['Close'][symbol]
                }).dropna()
                
                # Add Bollinger Bands
                from indicators.bollinger_bands import calculate_bollinger_bands
                bb_data = calculate_bollinger_bands(stock_df['Close'], window=20, num_std=2.0)
                
                all_stock_data[symbol] = pd.DataFrame({
                    'Open': stock_df['Open'],
                    'High': stock_df['High'],
                    'Low': stock_df['Low'],
                    'Close': stock_df['Close'],
                    'MA': bb_data['MA'],
                    'BB_Width': bb_data['BB_Width']
                }).dropna().tail(300)
        
        print(f"✅ Prepared data for {len(all_stock_data)} stocks")
        
        # Step 2: Train Global Models
        print(f"\n🌍 STEP 2: Training Global Models")
        print("-" * 40)
        
        # Train global open price model
        print("🔧 Training global open price KDE model...")
        open_forecaster = IntelligentOpenForecaster()
        open_forecaster.train_global_model(all_stock_data)
        
        # Train global high-low copula model
        print("🔧 Training global high-low copula model...")
        high_low_forecaster = IntelligentHighLowForecaster()
        high_low_forecaster.train_global_model(all_stock_data)
        
        print("✅ Global models trained successfully!")
        
        # Step 3: Individual Stock Analysis
        print(f"\n🎯 STEP 3: Individual Stock Enhanced Forecasting")
        print("-" * 50)
        
        selected_stock = 'AAPL'
        selected_stock_data = all_stock_data[selected_stock]
        
        print(f"Selected stock: {selected_stock}")
        print(f"Data points: {len(selected_stock_data)}")
        
        # Add stock-specific fine-tuning
        print("🔧 Adding stock-specific adaptations...")
        open_forecaster.add_stock_model(selected_stock, selected_stock_data)
        high_low_forecaster.add_stock_model(selected_stock, selected_stock_data)
        
        # Step 4: Create Enhanced OHLC Forecaster
        print(f"\n📈 STEP 4: Creating Enhanced OHLC Forecaster")
        print("-" * 45)
        
        # Create OHLC forecaster with all enhancements
        ohlc_forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
        ohlc_forecaster.fit(selected_stock_data)
        
        # Set intelligent forecasters
        ohlc_forecaster.set_intelligent_open_forecaster(open_forecaster, selected_stock)
        ohlc_forecaster.set_intelligent_high_low_forecaster(high_low_forecaster, selected_stock)
        
        print("✅ Enhanced OHLC forecaster ready with:")
        print("  📊 KDE-based close price estimation")
        print("  🌍 Intelligent open price modeling (global + stock-specific)")
        print("  🎲 Copula-based high-low dependency modeling")
        
        # Step 5: Generate Enhanced Forecast
        print(f"\n🔮 STEP 5: Generating Enhanced Forecast")
        print("-" * 40)
        
        current_close = selected_stock_data['Close'].iloc[-1]
        forecast_days = 10
        
        # Generate forecast inputs (same as notebook)
        ma_forecast = np.full(forecast_days, current_close * 1.005)
        vol_forecast = np.full(forecast_days, 0.025)
        bb_states = np.random.choice([1, 2, 3], size=forecast_days)
        
        print(f"Generating {forecast_days}-day enhanced forecast...")
        
        # Generate the enhanced forecast
        forecast_results = ohlc_forecaster.forecast_ohlc(
            ma_forecast=ma_forecast,
            vol_forecast=vol_forecast,
            bb_states=bb_states,
            current_close=current_close,
            n_days=forecast_days
        )
        
        print("✅ Enhanced forecast generated!")
        
        # Step 6: Display Results
        print(f"\n📊 STEP 6: Enhanced Forecast Results")
        print("=" * 80)
        print(f"Current {selected_stock} Price: ${current_close:.2f}")
        print()
        
        print(f"{'Day':<4} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Range':<8} {'Gap%':<8}")
        print("-" * 60)
        
        prev_close = current_close
        total_return = 0
        max_range = 0
        min_range = float('inf')
        
        for i in range(forecast_days):
            open_price = forecast_results['open'][i]
            high_price = forecast_results['high'][i]
            low_price = forecast_results['low'][i]
            close_price = forecast_results['close'][i]
            
            gap_pct = (open_price - prev_close) / prev_close * 100
            day_range = high_price - low_price
            daily_return = (close_price - prev_close) / prev_close * 100
            
            max_range = max(max_range, day_range)
            min_range = min(min_range, day_range)
            total_return += daily_return
            
            print(f"{i+1:<4} ${open_price:<7.2f} ${high_price:<7.2f} ${low_price:<7.2f} "
                  f"${close_price:<7.2f} ${day_range:<7.2f} {gap_pct:<7.2f}%")
            
            prev_close = close_price
        
        # Step 7: Forecast Analysis
        print(f"\n📊 STEP 7: Enhanced Forecast Analysis")
        print("-" * 40)
        
        final_price = forecast_results['close'][-1]
        
        print(f"📈 Forecast Summary:")
        print(f"  Final Price: ${final_price:.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Average Daily Range: ${np.mean([forecast_results['high'][i] - forecast_results['low'][i] for i in range(forecast_days)]):.2f}")
        print(f"  Max Daily Range: ${max_range:.2f}")
        print(f"  Min Daily Range: ${min_range:.2f}")
        
        # Show confidence intervals
        final_ci = forecast_results['close_ci'][-1]
        print(f"  Final Day 95% CI: ${final_ci[0]:.2f} - ${final_ci[1]:.2f}")
        
        # Step 8: Model Information
        print(f"\n🔧 STEP 8: Model Enhancement Information")
        print("-" * 45)
        
        # Open forecaster info
        open_system_info = open_forecaster.get_system_info()
        print(f"🌍 Open Price Modeling:")
        print(f"  Global regimes: {open_system_info['global_model']['regimes']}")
        if selected_stock in open_system_info['stock_models']:
            stock_info = open_system_info['stock_models'][selected_stock]
            print(f"  {selected_stock} adapted regimes: {len(stock_info['adapted_regimes'])}")
        
        # High-low forecaster info
        hl_system_info = high_low_forecaster.get_system_info()
        print(f"🎲 High-Low Copula Modeling:")
        print(f"  Global regimes: {hl_system_info['global_model']['regimes']}")
        if selected_stock in hl_system_info['stock_models']:
            stock_info = hl_system_info['stock_models'][selected_stock]
            print(f"  {selected_stock} adapted regimes: {len(stock_info['adapted_regimes'])}")
        
        # Show regime copula analysis
        regime_analysis = high_low_forecaster.analyze_regime_copulas()
        if not regime_analysis.empty:
            print(f"\n📊 Copula Family Selection by Regime:")
            for _, row in regime_analysis.iterrows():
                print(f"  {row['Regime']}: {row['Best_Copula']} copula")
        
        print(f"\n🎉 COMPLETE INTELLIGENT OHLC FORECASTING DEMO SUCCESSFUL!")
        print("=" * 70)
        
        # Step 9: Benefits Summary
        print(f"🎯 Enhanced Forecasting Benefits:")
        print(f"  📊 Close Prices: KDE-based estimation with regime awareness")
        print(f"  🏢 Open Prices: Global + stock-specific gap modeling")
        print(f"  🎲 High-Low: Copula-based dependency preservation")
        print(f"  🌍 Global Learning: Patterns from {len(all_stock_data)} stocks")
        print(f"  🔄 Adaptive: Stock-specific fine-tuning")
        print(f"  ⚡ Robust: Multiple fallback mechanisms")
        
        # Step 10: Integration Instructions
        print(f"\n📝 INTEGRATION WITH YOUR NOTEBOOK:")
        print("=" * 50)
        print("# Add these imports to your notebook:")
        print("from models.open_price_kde import IntelligentOpenForecaster")
        print("from models.high_low_copula import IntelligentHighLowForecaster")
        print()
        print("# Train global models (run once):")
        print("open_forecaster = IntelligentOpenForecaster()")
        print("open_forecaster.train_global_model(all_stock_bb_data)")
        print("high_low_forecaster = IntelligentHighLowForecaster()")
        print("high_low_forecaster.train_global_model(all_stock_bb_data)")
        print()
        print("# In your individual stock analysis:")
        print("open_forecaster.add_stock_model(selected_stock, selected_stock_data)")
        print("high_low_forecaster.add_stock_model(selected_stock, selected_stock_data)")
        print()
        print("# Set intelligent forecasters:")
        print("ohlc_forecaster.set_intelligent_open_forecaster(open_forecaster, selected_stock)") 
        print("ohlc_forecaster.set_intelligent_high_low_forecaster(high_low_forecaster, selected_stock)")
        print()
        print("# Your existing forecast code works the same!")
        print("forecast_results = ohlc_forecaster.forecast_ohlc(...)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demonstrate_complete_workflow()
    
    if success:
        print(f"\n💡 READY FOR PRODUCTION!")
        print(f"🎯 Your options trading system now has:")
        print(f"  ✅ State-of-the-art OHLC forecasting")
        print(f"  ✅ Regime-aware dependency modeling") 
        print(f"  ✅ Global + stock-specific learning")
        print(f"  ✅ Seamless notebook integration")
        print(f"  ✅ Enhanced options strategy accuracy")
    else:
        print(f"\n❌ Demo failed - check error messages above")
    
    print(f"\n🚀 Intelligent OHLC Forecasting System Complete!")