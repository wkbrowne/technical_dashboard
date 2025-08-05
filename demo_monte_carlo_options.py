#!/usr/bin/env python3
"""
Monte Carlo Options Analysis System - Complete Demo
===================================================

This demo showcases the complete Monte Carlo-based options strategy system with:
1. 1000+ trajectory simulation for ensemble close prices
2. Configurable probability targets for each strategy
3. Multi-DTE analysis (0-10 days to expiry)
4. Smart strike selection based on probability requirements

Key Features:
- Easy probability reconfiguration
- Real-time strike optimization
- Clean interface for Monte Carlo manipulation
- Separated complex modeling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from models.monte_carlo_options import (
    MonteCarloTrajectoryGenerator,
    MonteCarloOptionsAnalyzer,
    StrategyConfig
)
from data.loader import get_multiple_stocks

print("🎲 MONTE CARLO OPTIONS ANALYSIS SYSTEM - COMPLETE DEMO")
print("=" * 80)

def demo_complete_system():
    """Demonstrate the complete Monte Carlo options system."""
    
    # Step 1: Load Real Stock Data
    print("📊 STEP 1: Loading Real Stock Data")
    print("-" * 40)
    
    try:
        stock_data = get_multiple_stocks('AAPL', update=False, rate_limit=2.0)
        
        # Get AAPL data
        if 'AAPL' in stock_data['Close'].columns:
            aapl_price = stock_data['Close']['AAPL'].iloc[-1]
            aapl_volatility = stock_data['Close']['AAPL'].pct_change().std()
            
            print(f"✅ Loaded AAPL data:")
            print(f"   Current Price: ${aapl_price:.2f}")
            print(f"   Daily Volatility: {aapl_volatility:.1%}")
        else:
            # Fallback
            aapl_price = 175.0
            aapl_volatility = 0.025
            print(f"🔄 Using example data: ${aapl_price:.2f} with {aapl_volatility:.1%} volatility")
            
    except Exception as e:
        print(f"⚠️ Data loading issue: {e}")
        # Use example values
        aapl_price = 175.0
        aapl_volatility = 0.025
        print(f"🔄 Using example data: ${aapl_price:.2f} with {aapl_volatility:.1%} volatility")
    
    # Step 2: Configure Strategy Probabilities
    print(f"\n🎯 STEP 2: Configure Strategy Probabilities")
    print("-" * 45)
    
    # Example: Conservative configuration
    conservative_config = StrategyConfig(
        covered_call_keep_prob=0.95,      # 95% chance of keeping stock
        long_call_win_prob=0.40,          # 40% chance of call winning (conservative)
        csp_avoid_assignment_prob=0.90,   # 90% chance of avoiding assignment
        long_put_win_prob=0.40            # 40% chance of put winning (conservative)
    )
    
    print("📊 Conservative Configuration:")
    for key, value in conservative_config.to_dict().items():
        print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
    
    # Step 3: Generate Monte Carlo Trajectories
    print(f"\n🎲 STEP 3: Generate Monte Carlo Trajectories")
    print("-" * 45)
    
    trajectory_gen = MonteCarloTrajectoryGenerator(
        n_trajectories=1000,
        use_garch_vol=True,
        use_regime_switching=True,
        random_seed=42  # For reproducible demo
    )
    
    trajectories = trajectory_gen.generate_trajectories(
        current_price=aapl_price,
        volatility=aapl_volatility,
        max_days=10,
        drift=0.0005,
        regime_probs={'bull': 0.4, 'neutral': 0.4, 'bear': 0.2}
    )
    
    print(f"✅ Generated {trajectories.shape[0]:,} trajectories over {trajectories.shape[1]-1} days")
    
    # Show trajectory statistics
    day_5_stats = trajectory_gen.get_price_statistics(5)
    day_10_stats = trajectory_gen.get_price_statistics(10)
    
    print(f"📊 Trajectory Statistics:")
    print(f"   Day 5 Price Range: ${day_5_stats['q05']:.2f} - ${day_5_stats['q95']:.2f}")
    print(f"   Day 10 Price Range: ${day_10_stats['q05']:.2f} - ${day_10_stats['q95']:.2f}")
    print(f"   Final Mean Price: ${day_10_stats['mean']:.2f}")
    
    # Step 4: Find Optimal Strikes
    print(f"\n🎯 STEP 4: Find Optimal Strikes for All Strategies")
    print("-" * 50)
    
    analyzer = MonteCarloOptionsAnalyzer(
        trajectory_generator=trajectory_gen,
        strategy_config=conservative_config
    )
    
    results = analyzer.find_optimal_strikes(
        current_price=aapl_price,
        max_dte=10,
        risk_free_rate=0.05
    )
    
    print(f"✅ Found optimal strikes for all strategies")
    
    # Step 5: Display Results
    print(f"\n📊 STEP 5: Strategy Results Summary")
    print("-" * 40)
    
    summary_table = analyzer.get_strategy_summary_table()
    
    # Show results by strategy type
    strategies = summary_table['Strategy'].unique()
    
    for strategy in strategies:
        strategy_data = summary_table[summary_table['Strategy'] == strategy]
        print(f"\n🔸 {strategy.upper()}:")
        
        for _, row in strategy_data.head(3).iterrows():  # Show first 3 DTEs
            print(f"   {row['DTE']}DTE: Strike {row['Strike']}, "
                  f"Success {row['Success_Prob']} (target {row['Target_Prob']}), "
                  f"P&L {row['Expected_PnL']}")
    
    # Step 6: Best Strategies
    print(f"\n🏆 STEP 6: Best Strategy by DTE")
    print("-" * 35)
    
    # Find best strategy for each DTE
    for dte in range(0, 6):  # Show 0-5 DTE
        dte_data = summary_table[summary_table['DTE'] == dte]
        if not dte_data.empty:
            # Find highest expected P&L
            dte_data_copy = dte_data.copy()
            dte_data_copy['PnL_Numeric'] = dte_data_copy['Expected_PnL'].str.replace('$', '').str.replace(',', '').astype(float)
            best_idx = dte_data_copy['PnL_Numeric'].idxmax()
            best = dte_data.loc[best_idx]
            
            print(f"   {dte}DTE: {best['Strategy']} at {best['Strike']} "
                  f"({best['Success_Prob']} success, {best['Expected_PnL']} P&L)")
    
    # Step 7: Try Different Configuration
    print(f"\n🔄 STEP 7: Try Aggressive Configuration")
    print("-" * 40)
    
    # Example: Aggressive configuration
    aggressive_config = StrategyConfig(
        covered_call_keep_prob=0.80,      # 80% chance (more aggressive)
        long_call_win_prob=0.60,          # 60% chance (higher confidence)
        csp_avoid_assignment_prob=0.75,   # 75% chance (more aggressive)
        long_put_win_prob=0.60            # 60% chance (higher confidence)
    )
    
    print("📊 Aggressive Configuration:")
    for key, value in aggressive_config.to_dict().items():
        print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
    
    # Update analyzer configuration
    analyzer.update_config(aggressive_config)
    
    # Re-run analysis
    aggressive_results = analyzer.find_optimal_strikes(
        current_price=aapl_price,
        max_dte=5,  # Shorter horizon for demo
        risk_free_rate=0.05
    )
    
    aggressive_summary = analyzer.get_strategy_summary_table()
    
    print(f"\n📊 Aggressive vs Conservative Comparison (5DTE):")
    
    for strategy in ['Covered Call', 'Long Call']:
        if strategy in aggressive_summary['Strategy'].values:
            agg_row = aggressive_summary[(aggressive_summary['Strategy'] == strategy) & 
                                       (aggressive_summary['DTE'] == 5)]
            if not agg_row.empty:
                agg_row = agg_row.iloc[0]
                print(f"   {strategy}:")
                print(f"     Aggressive: {agg_row['Strike']} strike, {agg_row['Success_Prob']} success")
                
                # Find corresponding conservative result
                cons_row = summary_table[(summary_table['Strategy'] == strategy) & 
                                       (summary_table['DTE'] == 5)]
                if not cons_row.empty:
                    cons_row = cons_row.iloc[0]
                    print(f"     Conservative: {cons_row['Strike']} strike, {cons_row['Success_Prob']} success")
    
    # Step 8: Visualization
    print(f"\n📈 STEP 8: Generate Visualizations")
    print("-" * 35)
    
    print("🎲 Generating trajectory plot...")
    trajectory_gen.plot_trajectories(n_show=100, show_percentiles=True, figsize=(12, 8))
    
    print("📊 Generating strategy analysis plots...")
    analyzer.plot_strategy_analysis(figsize=(15, 12))
    
    print(f"\n✅ Complete Monte Carlo analysis finished!")
    
    return {
        'trajectory_generator': trajectory_gen,
        'analyzer': analyzer,
        'conservative_results': results,
        'aggressive_results': aggressive_results,
        'summary_table': summary_table,
        'aggressive_summary': aggressive_summary
    }

def demo_easy_reconfiguration():
    """Demonstrate easy probability reconfiguration."""
    
    print(f"\n🔧 BONUS: Easy Probability Reconfiguration Demo")
    print("=" * 60)
    
    # Show how easy it is to change probabilities
    configs = {
        'Ultra Conservative': {
            'covered_call_keep_prob': 0.98,
            'long_call_win_prob': 0.30,
            'csp_avoid_assignment_prob': 0.95,
            'long_put_win_prob': 0.30
        },
        'Balanced': {
            'covered_call_keep_prob': 0.85,
            'long_call_win_prob': 0.50,
            'csp_avoid_assignment_prob': 0.85,
            'long_put_win_prob': 0.50
        },
        'Aggressive': {
            'covered_call_keep_prob': 0.70,
            'long_call_win_prob': 0.70,
            'csp_avoid_assignment_prob': 0.70,
            'long_put_win_prob': 0.70
        }
    }
    
    print("📊 Different Configuration Examples:")
    
    for config_name, config_dict in configs.items():
        print(f"\n🔸 {config_name.upper()} CONFIGURATION:")
        config = StrategyConfig.from_dict(config_dict)
        
        for key, value in config.to_dict().items():
            strategy_name = key.replace('_prob', '').replace('_', ' ').title()
            print(f"   {strategy_name}: {value:.1%}")
    
    print(f"\n💡 TO USE DIFFERENT CONFIGURATIONS:")
    print("=" * 40)
    print("# Create your custom configuration")
    print("my_config = StrategyConfig(")
    print("    covered_call_keep_prob=0.90,")
    print("    long_call_win_prob=0.55,")
    print("    csp_avoid_assignment_prob=0.85,")
    print("    long_put_win_prob=0.45")
    print(")")
    print()
    print("# Update analyzer and re-run")
    print("analyzer.update_config(my_config)")
    print("results = analyzer.find_optimal_strikes(current_price, max_dte=10)")
    print("summary = analyzer.get_strategy_summary_table()")

if __name__ == "__main__":
    print("🚀 Starting complete Monte Carlo options system demo...\n")
    
    try:
        # Run complete system demo
        demo_results = demo_complete_system()
        
        # Show easy reconfiguration
        demo_easy_reconfiguration()
        
        print(f"\n🎉 MONTE CARLO DEMO COMPLETE!")
        print("=" * 80)
        print("🎯 Key Features Demonstrated:")
        print("  ✅ 1000+ trajectory Monte Carlo simulation")
        print("  ✅ Configurable probability-based strategy selection")
        print("  ✅ Multi-DTE analysis (0-10 days to expiry)")
        print("  ✅ Smart strike optimization for each strategy")
        print("  ✅ Easy probability reconfiguration")
        print("  ✅ Comprehensive visualization and analysis")
        print("  ✅ Real-time comparison of different configurations")
        
        print(f"\n📝 NEXT STEPS:")
        print("  1. Open Monte_Carlo_Options_Analysis.ipynb for interactive analysis")
        print("  2. Modify probability targets in the notebook")
        print("  3. Run analysis for your preferred stocks")
        print("  4. Compare different probability configurations")
        print("  5. Use the visualization tools to understand trajectories")
        
        print(f"\n💡 SYSTEM BENEFITS:")
        print("  🎲 Ensemble-based: More robust than single forecasts")
        print("  🎯 Probability-driven: You control the risk/reward balance")
        print("  📅 Multi-timeframe: Covers all DTEs from 0 to 10+ days")
        print("  🔧 Configurable: Easy to adjust for your trading style")
        print("  📊 Visual: Clear plots and tables for decision making")
        print("  ⚡ Fast: Optimized for real-time strategy analysis")
        
        print(f"\n🎲 MONTE CARLO OPTIONS SYSTEM READY FOR PRODUCTION!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n💫 Thank you for using the Monte Carlo Options Analysis System!")