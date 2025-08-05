#!/usr/bin/env python3
"""
Test script to verify sophisticated model integration works properly.
"""

import sys
import os
sys.path.append('/mnt/a61cc0e8-1b32-4574-a771-4ad77e8faab6/conda/technical_dashboard')

def test_sophisticated_model_imports():
    """Test if we can import sophisticated models."""
    print("🧪 Testing sophisticated model imports...")
    
    try:
        from src.models.monte_carlo_simple import MonteCarloTrajectoryGenerator, MonteCarloOptionsAnalyzer, StrategyConfig
        print("✅ Monte Carlo classes imported successfully")
        
        # Test model availability
        generator = MonteCarloTrajectoryGenerator(n_trajectories=100, symbol='AAPL')
        status = generator.get_model_status()
        
        print(f"📊 Model Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        if status['sophisticated_models_available']:
            print("🚀 Sophisticated models are available!")
            
            # Test enabling sophisticated models
            success = generator.enable_sophisticated_models('AAPL')
            if success:
                print("✅ Successfully enabled sophisticated models")
                
                # Test trajectory generation (just a few for testing)
                print("🎲 Testing trajectory generation...")
                current_price = 150.0
                trajectories = generator.generate_trajectories(
                    current_price=current_price,
                    max_days=5,
                    volatility=0.02
                )
                
                print(f"✅ Generated trajectories: {trajectories.shape}")
                print(f"   Model used: {generator.trajectory_metadata.get('model_type', 'unknown')}")
                
                if 'sophisticated' in generator.trajectory_metadata.get('model_type', ''):
                    print("🎉 SUCCESS: Sophisticated models are working!")
                    return True
                else:
                    print("⚠️ WARNING: Simple model was used instead")
                    return False
            else:
                print("❌ Failed to enable sophisticated models")
                return False
        else:
            print("⚠️ Sophisticated models not available - this is expected if imports fail")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sophisticated models: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_options_analysis():
    """Test options analysis with sophisticated models."""
    print("\n🧪 Testing options analysis...")
    
    try:
        from src.models.monte_carlo_simple import MonteCarloTrajectoryGenerator, MonteCarloOptionsAnalyzer, StrategyConfig
        
        # Create generator with sophisticated models
        generator = MonteCarloTrajectoryGenerator(n_trajectories=500, symbol='AAPL')
        
        if generator.get_model_status()['sophisticated_models_available']:
            generator.enable_sophisticated_models('AAPL')
            
            # Generate trajectories
            current_price = 150.0
            trajectories = generator.generate_trajectories(
                current_price=current_price,
                max_days=10,
                volatility=0.02
            )
            
            # Create options analyzer
            config = StrategyConfig(
                covered_call_keep_prob=0.95,
                long_call_win_prob=0.50,
                csp_avoid_assignment_prob=0.95,
                long_put_win_prob=0.50
            )
            
            analyzer = MonteCarloOptionsAnalyzer(generator, config)
            
            # Find optimal strikes
            results = analyzer.find_optimal_strikes(current_price, max_dte=5)
            
            if results:
                print("✅ Options analysis completed successfully")
                print(f"   Strategies analyzed: {list(results.keys())}")
                
                # Show summary
                summary_df = analyzer.get_strategy_summary_table()
                if not summary_df.empty:
                    print("📊 Sample results:")
                    print(summary_df.head())
                    
                return True
            else:
                print("❌ No results from options analysis")
                return False
        else:
            print("⚠️ Skipping options analysis - sophisticated models not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing options analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Sophisticated Model Integration")
    print("=" * 60)
    
    success1 = test_sophisticated_model_imports()
    success2 = test_options_analysis()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED - Sophisticated models are working!")
    elif success1:
        print("✅ Sophisticated models work, but options analysis had issues")
    else:
        print("❌ Tests failed - check import issues")
    print("=" * 60)