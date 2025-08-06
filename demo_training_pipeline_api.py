#!/usr/bin/env python3
"""
Complete Training Pipeline API Demo
==================================

This script demonstrates the clean 6-stage training pipeline API:

Stage 1: Global Markov Model Training
Stage 2: Stock-Specific Markov Models
Stage 3: OHLC Models (Copulas + KDEs)  
Stage 4: Stock-Specific OHLC Models
Stage 5: ARIMA GARCH Model Training
Stage 6: Single Prediction Generation

Each stage can be run independently or chained together.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def demo_individual_stages():
    """Demonstrate running each stage individually."""
    print("🎯 INDIVIDUAL STAGES DEMO")
    print("=" * 50)
    
    try:
        from models.training_pipeline_api import TrainingPipelineAPI
        
        # Create pipeline API
        api = TrainingPipelineAPI(cache_dir="demo_cache")
        
        # Demo symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        target_symbol = 'AAPL'
        
        print(f"📊 Demo symbols: {symbols}")
        print(f"🎯 Target symbol: {target_symbol}")
        print()
        
        # STAGE 1: Global Markov Model
        print("🌍 Running Stage 1: Global Markov Model...")
        stage1_results = api.train_global_markov_model(symbols, force_retrain=False)
        print(f"✅ Stage 1 completed in {stage1_results['training_time_seconds']:.1f}s")
        print(f"   Symbols trained: {stage1_results['n_symbols']}")
        print(f"   Model fitted: {stage1_results['model_fitted']}")
        print()
        
        # STAGE 2: Stock-Specific Markov Models  
        print("🎯 Running Stage 2: Stock-Specific Markov Models...")
        stage2_results = api.train_stock_specific_markov_models(symbols, force_retrain=False)
        print(f"✅ Stage 2 completed in {stage2_results['training_time_seconds']:.1f}s")
        print(f"   Successful models: {stage2_results['successful_models']}/{stage2_results['successful_models'] + stage2_results['failed_models']}")
        print()
        
        # STAGE 3: OHLC Models (Copulas + KDEs)
        print("📊 Running Stage 3: OHLC Models (Copulas + KDEs)...")
        stage3_results = api.train_ohlc_models_with_copulas_kdes(symbols, force_retrain=False)
        print(f"✅ Stage 3 completed in {stage3_results['training_time_seconds']:.1f}s")
        print(f"   Open forecaster regimes: {stage3_results['open_forecaster_regimes']}")
        print(f"   Copula forecaster regimes: {stage3_results['copula_forecaster_regimes']}")
        print()
        
        # STAGE 4: Stock-Specific OHLC Models
        print("🎯 Running Stage 4: Stock-Specific OHLC Models...")
        stage4_results = api.train_stock_specific_ohlc_models(symbols, force_retrain=False)
        print(f"✅ Stage 4 completed in {stage4_results['training_time_seconds']:.1f}s")
        print(f"   Successful models: {stage4_results['successful_models']}/{stage4_results['successful_models'] + stage4_results['failed_models']}")
        print()
        
        # STAGE 5: ARIMA GARCH Models
        print("📈 Running Stage 5: ARIMA GARCH Models...")
        stage5_results = api.train_arima_garch_model(symbols, force_retrain=False)
        print(f"✅ Stage 5 completed in {stage5_results['training_time_seconds']:.1f}s")
        print(f"   Successful models: {stage5_results['successful_models']}/{stage5_results['successful_models'] + stage5_results['failed_models']}")
        print()
        
        # STAGE 6: Single Prediction
        print(f"🔮 Running Stage 6: Single Prediction for {target_symbol}...")
        stage6_results = api.make_single_prediction(target_symbol, forecast_days=10)
        print(f"✅ Stage 6 completed in {stage6_results['prediction_time_seconds']:.2f}s")
        print(f"   Current price: ${stage6_results['current_price']:.2f}")
        print(f"   Predicted final price: ${stage6_results['final_price']:.2f}")
        print(f"   Total return: {stage6_results['total_return_pct']:.2f}%")
        print()
        
        # Pipeline Status
        status = api.get_pipeline_status()
        print("📊 Final Pipeline Status:")
        print(f"   Stages completed: {sum(status['stages_completed'].values())}/5")
        print(f"   Models available: {sum(status['models_available'].values())}")
        print(f"   Ready for predictions: {status['pipeline_ready_for_predictions']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual stages demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_complete_pipeline():
    """Demonstrate running the complete pipeline at once."""
    print("\n🚀 COMPLETE PIPELINE DEMO")
    print("=" * 50)
    
    try:
        from models.training_pipeline_api import TrainingPipelineAPI
        
        # Create pipeline API
        api = TrainingPipelineAPI(cache_dir="complete_demo_cache")
        
        # Demo symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        target_symbol = 'AAPL'
        
        print(f"📊 Pipeline symbols: {symbols}")
        print(f"🎯 Target symbol: {target_symbol}")
        print()
        
        # Run complete pipeline
        print("🎬 Running Complete 6-Stage Pipeline...")
        complete_results = api.run_complete_pipeline(symbols, target_symbol, force_retrain=False)
        
        if complete_results['pipeline_success']:
            print("🎉 COMPLETE PIPELINE SUCCESS!")
            print(f"   Total pipeline time: {complete_results['total_pipeline_time_seconds']:.1f}s")
            print(f"   Stages completed: {complete_results['stages_completed']}/6")
            print(f"   Target symbol: {complete_results['target_symbol']}")
            
            # Show final prediction
            final_pred = complete_results['final_prediction']
            print(f"\n🔮 Final Prediction for {target_symbol}:")
            print(f"   Current price: ${final_pred['current_price']:.2f}")
            print(f"   Predicted final price: ${final_pred['final_price']:.2f}")
            print(f"   Total return: {final_pred['total_return_pct']:.2f}%")
            print(f"   Max drawdown: {final_pred['max_drawdown_pct']:.2f}%")
            print(f"   Max runup: {final_pred['max_runup_pct']:.2f}%")
            print(f"   Avg daily range: ${final_pred['avg_daily_range']:.2f}")
            
            # Show models used
            models_used = final_pred['models_used']
            print(f"\n🔧 Models Used in Prediction:")
            print(f"   OHLC Forecaster: {'✅' if models_used['ohlc_forecaster'] else '❌'}")
            print(f"   ARIMA-GARCH: {'✅' if models_used['arima_garch'] else '❌'}")
            print(f"   Markov Regime: {'✅' if models_used['markov_regime'] else '❌'}")
            print(f"   Intelligent Open: {'✅' if models_used['intelligent_open'] else '❌'}")
            print(f"   Intelligent High-Low: {'✅' if models_used['intelligent_high_low'] else '❌'}")
            
            return True
        else:
            print("❌ Complete pipeline failed")
            print(f"   Error: {complete_results.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Complete pipeline demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_quick_convenience_functions():
    """Demonstrate the convenience functions."""
    print("\n🎬 CONVENIENCE FUNCTIONS DEMO")
    print("=" * 50)
    
    try:
        from models.training_pipeline_api import quick_pipeline_demo, single_stock_prediction
        
        # Demo 1: Quick Pipeline Demo
        print("🚀 Running quick_pipeline_demo()...")
        quick_results = quick_pipeline_demo(['AAPL', 'MSFT', 'TSLA'], 'AAPL')
        
        if quick_results['pipeline_success']:
            print("✅ Quick pipeline demo successful!")
            final_price = quick_results['final_prediction']['final_price']
            current_price = quick_results['final_prediction']['current_price']
            print(f"   AAPL prediction: ${current_price:.2f} → ${final_price:.2f}")
        else:
            print("❌ Quick pipeline demo failed")
        
        print()
        
        # Demo 2: Single Stock Prediction
        print("🎯 Running single_stock_prediction('AAPL')...")
        prediction_results = single_stock_prediction('AAPL', forecast_days=5)
        
        print("✅ Single stock prediction successful!")
        print(f"   Current price: ${prediction_results['current_price']:.2f}")
        print(f"   5-day prediction: ${prediction_results['final_price']:.2f}")
        print(f"   Expected return: {prediction_results['total_return_pct']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_detailed_forecast_output():
    """Show detailed forecast output from Stage 6."""
    print("\n📊 DETAILED FORECAST OUTPUT DEMO")
    print("=" * 50)
    
    try:
        from models.training_pipeline_api import TrainingPipelineAPI
        
        # Create pipeline and ensure models are trained
        api = TrainingPipelineAPI(cache_dir="forecast_demo_cache")
        
        symbols = ['AAPL', 'MSFT', 'SPY']
        target_symbol = 'AAPL'
        
        # Run pipeline if needed
        status = api.get_pipeline_status()
        if not status['pipeline_ready_for_predictions']:
            print("🔄 Training models first...")
            api.run_complete_pipeline(symbols, target_symbol, force_retrain=False)
        
        # Generate detailed prediction
        print(f"🔮 Generating detailed forecast for {target_symbol}...")
        forecast_results = api.make_single_prediction(target_symbol, forecast_days=7)
        
        # Display detailed day-by-day forecast
        print(f"\n📈 7-Day Detailed Forecast for {target_symbol}")
        print("=" * 80)
        print(f"Current Price: ${forecast_results['current_price']:.2f}")
        print()
        print(f"{'Day':<4} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Range':<8} {'CI Low':<8} {'CI High':<8}")
        print("-" * 80)
        
        forecast_data = forecast_results['forecast_data']
        for i in range(len(forecast_data['open'])):
            day = i + 1
            open_price = forecast_data['open'][i]
            high_price = forecast_data['high'][i]
            low_price = forecast_data['low'][i]
            close_price = forecast_data['close'][i]
            range_price = high_price - low_price
            ci_low = forecast_data['close_ci_lower'][i]
            ci_high = forecast_data['close_ci_upper'][i]
            
            print(f"{day:<4} ${open_price:<7.2f} ${high_price:<7.2f} ${low_price:<7.2f} "
                  f"${close_price:<7.2f} ${range_price:<7.2f} ${ci_low:<7.2f} ${ci_high:<7.2f}")
        
        # Summary statistics
        print(f"\n📊 Forecast Summary:")
        print(f"   Final Price: ${forecast_results['final_price']:.2f}")
        print(f"   Total Return: {forecast_results['total_return_pct']:.2f}%")
        print(f"   Max Drawdown: {forecast_results['max_drawdown_pct']:.2f}%")
        print(f"   Max Runup: {forecast_results['max_runup_pct']:.2f}%")
        print(f"   Average Daily Range: ${forecast_results['avg_daily_range']:.2f}")
        
        # Model diagnostics
        print(f"\n🔧 Model Diagnostics:")
        models_used = forecast_results['models_used']
        print(f"   OHLC Forecaster: {'✅ Active' if models_used['ohlc_forecaster'] else '❌ Inactive'}")
        print(f"   ARIMA-GARCH: {'✅ Active' if models_used['arima_garch'] else '❌ Fallback used'}")
        print(f"   Markov Regime: {'✅ Active' if models_used['markov_regime'] else '❌ Random states'}")
        print(f"   Intelligent Open: {'✅ Active' if models_used['intelligent_open'] else '❌ Simple model'}")
        print(f"   Intelligent High-Low: {'✅ Active' if models_used['intelligent_high_low'] else '❌ Simple model'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed forecast demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_api_features():
    """Demonstrate key API features."""
    print("\n🛠️ API FEATURES DEMO")
    print("=" * 50)
    
    try:
        from models.training_pipeline_api import TrainingPipelineAPI
        
        # Create API instance
        api = TrainingPipelineAPI(cache_dir="features_demo_cache")
        
        # Feature 1: Pipeline Status Checking
        print("📊 Feature 1: Pipeline Status Checking")
        status = api.get_pipeline_status()
        print(f"   Stages completed: {status['stages_completed']}")
        print(f"   Models available: {status['models_available']}")
        print(f"   Cache directory: {status['cache_directory']}")
        print(f"   Ready for predictions: {status['pipeline_ready_for_predictions']}")
        print()
        
        # Feature 2: Selective Stage Training
        print("🎯 Feature 2: Selective Stage Training")
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Train just global Markov
        print("   Training only Stage 1 (Global Markov)...")
        stage1_result = api.train_global_markov_model(symbols, force_retrain=False)
        print(f"   ✅ Stage 1: {stage1_result['training_time_seconds']:.1f}s")
        
        # Check status after Stage 1
        status = api.get_pipeline_status()
        print(f"   Updated status: {status['stages_completed']}")
        print()
        
        # Feature 3: Cache Management
        print("🗑️ Feature 3: Cache Management")
        print("   Cache files before clearing:")
        cache_files = list(api.cache_dir.glob("*.pkl"))
        for f in cache_files:
            print(f"      {f.name}")
        
        # Clear cache
        api.clear_all_cache()
        print("   ✅ Cache cleared")
        
        # Check status after clearing
        status = api.get_pipeline_status()
        print(f"   Status after clearing: {status['stages_completed']}")
        print()
        
        # Feature 4: Force Retraining
        print("🔄 Feature 4: Force Retraining")
        print("   Training Stage 1 with force_retrain=True...")
        stage1_retrain = api.train_global_markov_model(['AAPL', 'SPY'], force_retrain=True)
        print(f"   ✅ Forced retrain: {stage1_retrain['training_time_seconds']:.1f}s")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ API features demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all demo sections."""
    print("🎬 COMPLETE TRAINING PIPELINE API DEMONSTRATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    demo_results = {}
    
    # Run individual demos
    print("Running demos in sequence...")
    print()
    
    demo_results['individual_stages'] = demo_individual_stages()
    demo_results['complete_pipeline'] = demo_complete_pipeline()
    demo_results['convenience_functions'] = demo_quick_convenience_functions()
    demo_results['detailed_forecast'] = demo_detailed_forecast_output()
    demo_results['api_features'] = demo_api_features()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 DEMO SUMMARY")
    print("=" * 70)
    
    total_success = sum(demo_results.values())
    total_demos = len(demo_results)
    
    for demo_name, success in demo_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {demo_name.replace('_', ' ').title():<25} {status}")
    
    print()
    print(f"Overall Result: {total_success}/{total_demos} demos passed")
    
    if total_success == total_demos:
        print("🎉 ALL DEMOS SUCCESSFUL!")
        print("\n🚀 Training Pipeline API is ready for production use!")
        print("\nKey Features Demonstrated:")
        print("  ✅ 6-stage clean training pipeline")
        print("  ✅ Individual stage execution")
        print("  ✅ Complete pipeline execution") 
        print("  ✅ Intelligent caching system")
        print("  ✅ Comprehensive prediction generation")
        print("  ✅ Convenience functions")
        print("  ✅ Detailed forecast output")
        print("  ✅ Pipeline status monitoring")
        print("\n📖 Usage Instructions:")
        print("1. Import: from models.training_pipeline_api import TrainingPipelineAPI")
        print("2. Create: api = TrainingPipelineAPI()")
        print("3. Train: api.run_complete_pipeline(['AAPL', 'MSFT'], 'AAPL')")
        print("4. Predict: api.make_single_prediction('AAPL', forecast_days=10)")
        
    else:
        print("⚠️ Some demos failed - check error messages above")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()