#!/usr/bin/env python3
"""
OHLC Trajectory Candlestick Visualization Tool
==============================================

This script creates candlestick charts of full OHLC trajectories from stochastic simulations,
with optional trend regime annotations on each candle.

Usage:
    python visualize_ohlc_trajectory.py [--symbol SYMBOL] [--days DAYS] [--with-regimes]

Features:
1. Generates a complete OHLC forecast using trained models
2. Creates professional candlestick charts
3. Optionally annotates trend regimes on each candle
4. Supports both matplotlib and plotly rendering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import argparse
import warnings

warnings.filterwarnings('ignore')

# Import our models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.training_pipeline_api import TrainingPipelineAPI
from data.loader import get_multiple_stocks


class OHLCTrajectoryVisualizer:
    """Visualize OHLC trajectories from stochastic simulations."""
    
    def __init__(self, cache_dir: str = "demo_cache"):
        self.cache_dir = Path(cache_dir)
        self.pipeline = TrainingPipelineAPI(cache_dir=cache_dir)
        self.ohlc_data = None
        self.trajectory_data = None
        
    def load_trained_models(self) -> bool:
        """Load all trained models needed for simulation."""
        required_files = [
            "stage1_global_markov.pkl",
            "stage3_ohlc_copulas_kdes.pkl", 
            "stage5_arima_garch.pkl"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.cache_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            print("   Please run the training pipeline first:")
            print("   python demo_training_pipeline_api.py")
            return False
        
        # Load models into pipeline
        try:
            # Load global Markov
            with open(self.cache_dir / "stage1_global_markov.pkl", 'rb') as f:
                cached_data = pickle.load(f)
                self.pipeline.global_markov_model = cached_data['model']
                self.pipeline.stages_completed['global_markov'] = True
            
            # Load OHLC models
            with open(self.cache_dir / "stage3_ohlc_copulas_kdes.pkl", 'rb') as f:
                cached_data = pickle.load(f)
                self.pipeline.open_forecaster = cached_data['models']['open_forecaster']
                self.pipeline.high_low_forecaster = cached_data['models']['high_low_forecaster'] 
                self.pipeline.stages_completed['global_ohlc_copulas_kdes'] = True
            
            # Load ARIMA-GARCH models
            with open(self.cache_dir / "stage5_arima_garch.pkl", 'rb') as f:
                cached_data = pickle.load(f)
                self.pipeline.arima_garch_models = cached_data['models']
                self.pipeline.stages_completed['arima_garch'] = True
            
            print("✅ All required models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
    
    def generate_ohlc_trajectory(self, symbol: str, n_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Generate a complete OHLC trajectory using all trained models.
        
        Parameters
        ----------
        symbol : str
            Stock symbol to simulate
        n_days : int
            Number of days to simulate
            
        Returns
        -------
        pd.DataFrame or None
            OHLC trajectory with columns: Date, Open, High, Low, Close, Trend_Regime
        """
        print(f"🎯 Generating {n_days}-day OHLC trajectory for {symbol}")
        
        # Load current stock data
        try:
            stock_data = get_multiple_stocks([symbol], update=False, rate_limit=1.0)
            if symbol not in stock_data['Close'].columns:
                print(f"❌ No data available for {symbol}")
                return None
            
            current_close = stock_data['Close'][symbol].iloc[-1]
            print(f"📊 Current close price: ${current_close:.2f}")
            
        except Exception as e:
            print(f"❌ Failed to load stock data: {e}")
            return None
        
        # Generate ARIMA-GARCH forecasts
        arima_forecasts = None
        if symbol in self.pipeline.arima_garch_models:
            try:
                arima_model = self.pipeline.arima_garch_models[symbol]
                if hasattr(arima_model, 'forecast') and arima_model.fitted:
                    forecast_result = arima_model.forecast(horizon=n_days)
                    if 'ma_forecast' in forecast_result:
                        arima_forecasts = forecast_result['ma_forecast']
                        print(f"✅ ARIMA forecasts generated for {n_days} days")
            except Exception as e:
                print(f"⚠️ ARIMA forecast failed: {e}")
        
        # Generate Markov state sequence
        try:
            # Use global Markov model to predict regime sequence
            markov_model = self.pipeline.global_markov_model
            
            # Get recent stock data for context
            recent_data = pd.DataFrame({
                'Open': stock_data['Open'][symbol].tail(50),
                'High': stock_data['High'][symbol].tail(50),
                'Low': stock_data['Low'][symbol].tail(50),
                'Close': stock_data['Close'][symbol].tail(50)
            }).dropna()
            
            # Calculate BB positions and trends
            recent_data['BB_MA'] = recent_data['Close'].rolling(20).mean()
            recent_data['BB_Std'] = recent_data['Close'].rolling(20).std()
            recent_data['BB_Upper'] = recent_data['BB_MA'] + 2 * recent_data['BB_Std']
            recent_data['BB_Lower'] = recent_data['BB_MA'] - 2 * recent_data['BB_Std']
            
            # Calculate BB position
            close_prices = recent_data['Close']
            bb_position = np.where(close_prices > recent_data['BB_Upper'], 5,
                         np.where(close_prices > recent_data['BB_MA'] + recent_data['BB_Std'], 4,
                         np.where(close_prices > recent_data['BB_MA'], 3,
                         np.where(close_prices > recent_data['BB_MA'] - recent_data['BB_Std'], 2, 1))))
            
            recent_data['BB_Position'] = bb_position
            recent_data['Trend'] = markov_model.classify_trend(recent_data['BB_MA'])
            
            # Predict future states
            current_state = recent_data['BB_Position'].iloc[-1]
            current_trend = recent_data['Trend'].iloc[-1]
            
            if hasattr(markov_model, 'sample_future_states'):
                future_states = markov_model.sample_future_states(n_days, current_state, current_trend)
            else:
                # Fallback: use simple Markov chain simulation
                future_states = np.random.choice([1, 2, 3, 4, 5], size=n_days)
            
        except Exception as e:
            print(f"⚠️ Markov state generation failed: {e}")
            future_states = np.random.choice([1, 2, 3, 4, 5], size=n_days)
        
        # Generate volatility forecasts (simple)
        recent_returns = recent_data['Close'].pct_change().dropna()
        base_vol = recent_returns.std()
        vol_forecasts = np.random.normal(base_vol, base_vol * 0.1, n_days)
        vol_forecasts = np.abs(vol_forecasts)  # Ensure positive
        
        # Generate MA forecasts if no ARIMA
        if arima_forecasts is None:
            current_ma = recent_data['BB_MA'].iloc[-1]
            trend_factor = 0.001 if current_trend == 'up' else (-0.001 if current_trend == 'down' else 0)
            ma_forecasts = [current_ma + i * trend_factor + np.random.normal(0, current_ma * 0.01) 
                           for i in range(n_days)]
        else:
            ma_forecasts = np.exp(arima_forecasts)  # Convert from log prices
        
        # Simulate OHLC using the forecasting models
        print("🎲 Simulating OHLC trajectory...")
        
        dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), 
                             periods=n_days, freq='D')
        
        trajectory = []
        prev_close = current_close
        
        for i in range(n_days):
            # Determine trend regime for this day
            bb_state = future_states[i]
            volatility = vol_forecasts[i]
            
            # Simple trend classification based on BB state
            if bb_state >= 4:
                trend_regime = 'Bullish'
            elif bb_state <= 2:
                trend_regime = 'Bearish'  
            else:
                trend_regime = 'Neutral'
            
            # Generate Close price
            if arima_forecasts is not None:
                close_price = ma_forecasts[i]
            else:
                # Use MA + noise
                close_price = ma_forecasts[i] + np.random.normal(0, prev_close * volatility)
            
            # Generate Open (with gap)
            gap_factor = np.random.normal(0, 0.002)  # Small random gap
            open_price = prev_close * (1 + gap_factor)
            
            # Generate High and Low based on volatility and trend
            daily_range = close_price * volatility * np.random.uniform(0.5, 2.0)
            
            if trend_regime == 'Bullish':
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0.001, 0.01))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0.001, 0.005))
            elif trend_regime == 'Bearish':
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0.001, 0.005))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0.001, 0.01))
            else:  # Neutral
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0.001, 0.007))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0.001, 0.007))
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            trajectory.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Trend_Regime': trend_regime,
                'BB_State': bb_state,
                'Volatility': volatility
            })
            
            prev_close = close_price
        
        self.trajectory_data = pd.DataFrame(trajectory)
        print(f"✅ Generated {n_days}-day OHLC trajectory")
        return self.trajectory_data
    
    def plot_candlestick_chart(self, with_regimes: bool = True, figsize: tuple = (15, 10)):
        """
        Create a candlestick chart of the OHLC trajectory.
        
        Parameters
        ----------
        with_regimes : bool
            Whether to annotate trend regimes
        figsize : tuple
            Figure size
        """
        if self.trajectory_data is None:
            print("❌ No trajectory data. Generate trajectory first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], 
                                      sharex=True)
        
        # Plot candlesticks
        self._plot_candlesticks(ax1, with_regimes)
        
        # Plot volatility
        ax2.plot(self.trajectory_data['Date'], self.trajectory_data['Volatility'], 
                color='purple', alpha=0.7, linewidth=1.5)
        ax2.set_ylabel('Volatility', fontsize=10)
        ax2.set_title('Forecasted Volatility', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.trajectory_data) // 10)))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = f"ohlc_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved candlestick chart to {output_file}")
        
        plt.show()
    
    def _plot_candlesticks(self, ax, with_regimes: bool):
        """Plot candlestick chart on given axis."""
        data = self.trajectory_data
        
        # Define colors for regimes
        regime_colors = {
            'Bullish': 'green',
            'Bearish': 'red', 
            'Neutral': 'gray'
        }
        
        for i, row in data.iterrows():
            date = mdates.date2num(row['Date'])
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            regime = row['Trend_Regime']
            
            # Determine candle color
            if with_regimes:
                color = regime_colors.get(regime, 'blue')
                edge_color = color
            else:
                color = 'green' if close_price > open_price else 'red'
                edge_color = 'darkgreen' if close_price > open_price else 'darkred'
            
            # Draw high-low line
            ax.plot([date, date], [low_price, high_price], color=edge_color, linewidth=1)
            
            # Draw candle body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            candle = Rectangle((date - 0.3, body_bottom), 0.6, body_height,
                             facecolor=color, edgecolor=edge_color, alpha=0.8)
            ax.add_patch(candle)
            
            # Add regime annotation if requested
            if with_regimes and i % max(1, len(data) // 20) == 0:  # Annotate every nth candle
                ax.annotate(regime[:4], (date, high_price), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, 
                           color=edge_color, fontweight='bold')
        
        # Formatting
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Simulated OHLC Trajectory with Trend Regimes' if with_regimes 
                    else 'Simulated OHLC Trajectory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend if using regimes
        if with_regimes:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=regime) 
                             for regime, color in regime_colors.items()]
            ax.legend(handles=legend_elements, loc='upper left')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize OHLC trajectory from stochastic simulation')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol to simulate (default: AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to simulate (default: 30)')
    parser.add_argument('--with-regimes', action='store_true', 
                       help='Annotate trend regimes on candlesticks')
    parser.add_argument('--cache-dir', default='demo_cache', 
                       help='Directory containing trained models (default: demo_cache)')
    
    args = parser.parse_args()
    
    print("📈 OHLC Trajectory Candlestick Visualizer")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"With regimes: {args.with_regimes}")
    
    # Initialize visualizer
    visualizer = OHLCTrajectoryVisualizer(cache_dir=args.cache_dir)
    
    # Load trained models
    if not visualizer.load_trained_models():
        return
    
    # Generate trajectory
    trajectory = visualizer.generate_ohlc_trajectory(args.symbol, args.days)
    if trajectory is None:
        return
    
    # Create visualization
    print("\\n🎨 Creating candlestick chart...")
    visualizer.plot_candlestick_chart(with_regimes=args.with_regimes)
    
    print("\\n✅ Visualization complete!")


if __name__ == "__main__":
    main()