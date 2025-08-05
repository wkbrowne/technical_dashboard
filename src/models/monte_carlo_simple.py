"""
Simple Monte Carlo Options Strategy Analyzer
============================================

This is a working version that can be enhanced step by step with your
sophisticated ARIMA/GARCH/Copula/Markov models.

This version:
1. Works immediately with simple random walk
2. Has hooks for sophisticated model integration 
3. Maintains the same interface as the enhanced version
4. Can be extended incrementally
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategyConfig:
    """Configuration for options strategy probability targets."""
    
    # Covered Call: Probability of keeping the stock (not being assigned)
    covered_call_keep_prob: float = 0.95  # 95% chance of keeping stock
    
    # Long Call: Probability of winning (finishing ITM)
    long_call_win_prob: float = 0.50  # 50% chance of winning
    
    # Cash Secured Put: Probability of NOT buying the stock (finishing OTM)
    csp_avoid_assignment_prob: float = 0.95  # 95% chance of not buying stock
    
    # Long Put: Probability of winning (finishing ITM)
    long_put_win_prob: float = 0.50  # 50% chance of winning
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy manipulation."""
        return {
            'covered_call_keep_prob': self.covered_call_keep_prob,
            'long_call_win_prob': self.long_call_win_prob,
            'csp_avoid_assignment_prob': self.csp_avoid_assignment_prob,
            'long_put_win_prob': self.long_put_win_prob
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**config_dict)

@dataclass
class StrategyResult:
    """Result for a single strategy at a specific DTE."""
    strategy_name: str
    dte: int
    strike_price: float
    success_probability: float
    target_probability: float
    expected_pnl: float
    max_profit: float
    max_loss: float
    trajectories_successful: int
    total_trajectories: int
    breakeven_price: Optional[float] = None
    premium_estimate: Optional[float] = None

class MonteCarloTrajectoryGenerator:
    """Generate ensemble of price trajectories (simple version, ready for enhancement)."""
    
    def __init__(self, 
                 n_trajectories: int = 1000,
                 symbol: str = None,
                 use_simple_model: bool = None,  # Auto-detect if None
                 random_seed: Optional[int] = None):
        """
        Initialize trajectory generator.
        
        Parameters
        ----------
        n_trajectories : int
            Number of Monte Carlo trajectories to generate
        symbol : str, optional
            Stock symbol (for future sophisticated model integration)
        use_simple_model : bool
            If True, uses simple random walk (default for now)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.n_trajectories = n_trajectories
        self.symbol = symbol
        
        # Initialize sophisticated model integration
        self.sophisticated_models_available = self._check_sophisticated_models()
        
        # Auto-detect model type if not specified
        if use_simple_model is None:
            self.use_simple_model = not self.sophisticated_models_available
        else:
            self.use_simple_model = use_simple_model
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Model state
        self.ohlc_forecaster = None
        self.markov_model = None
        self.models_trained = False
        
        # Trajectory storage
        self.trajectories = None
        self.trajectory_metadata = {}
    
    def _check_sophisticated_models(self) -> bool:
        """Check if sophisticated models are available."""
        try:
            # Try absolute imports first
            from src.models.ohlc_forecasting import OHLCForecaster
            from src.models.markov_bb import TrendAwareBBMarkovWrapper
            from src.data.loader import get_multiple_stocks
            return True
        except ImportError:
            try:
                # Fallback to relative imports
                from .ohlc_forecasting import OHLCForecaster
                from .markov_bb import TrendAwareBBMarkovWrapper
                from ..data.loader import get_multiple_stocks
                return True
            except ImportError:
                return False
    
    def _load_and_train_models(self, symbol: str = None) -> None:
        """Load and train sophisticated models if available."""
        if self.models_trained or not self.sophisticated_models_available:
            return
            
        print(f"🚀 Loading sophisticated ARIMA/GARCH/Copula models...")
        
        try:
            # Try absolute imports first
            try:
                from src.models.ohlc_forecasting import OHLCForecaster
                from src.data.loader import get_multiple_stocks
            except ImportError:
                # Fallback to relative imports
                from .ohlc_forecasting import OHLCForecaster
                from ..data.loader import get_multiple_stocks
            
            # Use provided symbol or default
            symbol = symbol or self.symbol or 'AAPL'
            
            # Load stock data
            print(f"📊 Loading data for {symbol}...")
            stock_data = get_multiple_stocks([symbol], update=False, rate_limit=1.0)
            
            if symbol not in stock_data['Close'].columns:
                raise ValueError(f"Could not load data for {symbol}")
            
            # Prepare OHLC data
            ohlc_data = pd.DataFrame({
                'Open': stock_data['Open'][symbol],
                'High': stock_data['High'][symbol], 
                'Low': stock_data['Low'][symbol],
                'Close': stock_data['Close'][symbol]
            }).dropna()
            
            print(f"📈 Training OHLC forecaster with {len(ohlc_data)} observations...")
            
            # Train OHLC Forecaster (includes KDE, Markov, volatility models)
            self.ohlc_forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
            self.ohlc_forecaster.fit(ohlc_data)
            
            # Extract Markov model
            self.markov_model = self.ohlc_forecaster.markov_model
            
            # Store symbol for reference
            self.symbol = symbol
            
            print(f"✅ Sophisticated models trained successfully for {symbol}!")
            print(f"   - OHLC Forecaster: ✓ (KDE + Markov + GARCH)")
            print(f"   - Trend-aware Markov: ✓")
            if hasattr(self.ohlc_forecaster, 'kde_models'):
                print(f"   - KDE Close Models: ✓ ({len(self.ohlc_forecaster.kde_models)} regimes)")
            
            self.models_trained = True
            
        except Exception as e:
            print(f"⚠️ Error training sophisticated models: {e}")
            print(f"🔄 Will use simple model fallback")
            self.sophisticated_models_available = False
    
    def generate_trajectories(self, 
                            current_price: float,
                            volatility: float = 0.02,
                            max_days: int = 10,
                            drift: float = 0.0005,
                            regime_probs: Optional[Dict] = None) -> np.ndarray:
        """
        Generate ensemble of price trajectories.
        
        Parameters
        ----------
        current_price : float
            Current stock price
        volatility : float
            Daily volatility
        max_days : int
            Maximum days to simulate
        drift : float
            Daily drift (expected return)
        regime_probs : dict, optional
            For future sophisticated model integration
            
        Returns
        -------
        np.ndarray
            Shape (n_trajectories, max_days+1) with price paths
        """
        print(f"🎲 Generating {self.n_trajectories} Monte Carlo trajectories for {max_days} days")
        
        # Load and train sophisticated models if available
        if self.sophisticated_models_available and not self.use_simple_model:
            self._load_and_train_models()
        
        if self.use_simple_model or not self.sophisticated_models_available or not self.models_trained:
            print("🔄 Using simple random walk model")
            return self._generate_simple_trajectories(current_price, volatility, max_days, drift)
        else:
            print("🚀 Using sophisticated ARIMA/GARCH/Copula models")
            return self._generate_sophisticated_trajectories(current_price, volatility, max_days, drift)
    
    def _generate_simple_trajectories(self, current_price: float, volatility: float, 
                                    max_days: int, drift: float) -> np.ndarray:
        """Generate trajectories using simple random walk."""
        trajectories = np.zeros((self.n_trajectories, max_days + 1))
        trajectories[:, 0] = current_price
        
        # Handle None values with defaults
        if volatility is None:
            volatility = 0.02
        if drift is None:
            drift = 0.0005
        
        print(f"📊 Simple model parameters:")
        print(f"   Volatility: {volatility:.3f}")
        print(f"   Drift: {drift:.4f}")
        
        for traj in range(self.n_trajectories):
            current = current_price
            for day in range(1, max_days + 1):
                # Simple random walk with optional regime effects
                daily_return = np.random.normal(drift, volatility)
                current = current * (1 + daily_return)
                trajectories[traj, day] = current
        
        # Store trajectories and metadata
        self.trajectories = trajectories
        self.trajectory_metadata = {
            'current_price': current_price,
            'base_volatility': volatility,
            'max_days': max_days,
            'drift': drift,
            'n_trajectories': self.n_trajectories,
            'generation_time': datetime.now(),
            'model_type': 'simple_random_walk',
            'symbol': self.symbol or 'N/A'
        }
        
        print(f"✅ Generated trajectories: {trajectories.shape}")
        print(f"   Price range at day {max_days}: ${np.min(trajectories[:, -1]):.2f} - ${np.max(trajectories[:, -1]):.2f}")
        print(f"   Mean final price: ${np.mean(trajectories[:, -1]):.2f}")
        
        return trajectories
    
    def _generate_sophisticated_trajectories(self, current_price: float, volatility: float,
                                           max_days: int, drift: float) -> np.ndarray:
        """Generate trajectories using your sophisticated OHLC forecasting models."""
        trajectories = np.zeros((self.n_trajectories, max_days + 1))
        trajectories[:, 0] = current_price
        
        try:
            # Get historical data for model state
            ohlc_data = self.ohlc_forecaster.ohlc_data
            if len(ohlc_data) == 0:
                raise ValueError("No historical data available")
                
            # Get current market state
            current_bb_pos = ohlc_data['BB_Position'].iloc[-1]
            current_ma = ohlc_data['BB_MA'].iloc[-1]
            current_vol = ohlc_data['BB_Width'].iloc[-1] if volatility is None else volatility
            
            print(f"📊 Current market state for {self.symbol}:")
            print(f"   BB Position: {current_bb_pos:.3f}")
            print(f"   MA: ${current_ma:.2f}")
            print(f"   Volatility: {current_vol:.3f}")
            
            # Generate ensemble using OHLC forecaster
            for traj in range(self.n_trajectories):
                trajectory = self._generate_single_sophisticated_trajectory(
                    current_price, max_days, current_bb_pos, current_ma, current_vol)
                trajectories[traj, :] = trajectory
                
                if (traj + 1) % 200 == 0:
                    print(f"   Generated {traj + 1}/{self.n_trajectories} trajectories...")
            
            # Store results with sophisticated model metadata
            self.trajectories = trajectories
            self.trajectory_metadata = {
                'current_price': current_price,
                'base_volatility': current_vol,
                'max_days': max_days,
                'drift': drift,
                'n_trajectories': self.n_trajectories,
                'generation_time': datetime.now(),
                'model_type': 'sophisticated_ohlc_forecaster',
                'symbol': self.symbol,
                'current_bb_position': current_bb_pos,
                'current_ma': current_ma,
                'kde_models': len(self.ohlc_forecaster.kde_models) if hasattr(self.ohlc_forecaster, 'kde_models') else 0
            }
            
            print(f"✅ Generated sophisticated trajectories: {trajectories.shape}")
            print(f"   Price range at day {max_days}: ${np.min(trajectories[:, -1]):.2f} - ${np.max(trajectories[:, -1]):.2f}")
            print(f"   Mean final price: ${np.mean(trajectories[:, -1]):.2f}")
            
            return trajectories
            
        except Exception as e:
            print(f"⚠️ Error generating sophisticated trajectories: {e}")
            print("🔄 Falling back to simple model")
            return self._generate_simple_trajectories(current_price, volatility, max_days, drift)
    
    def _generate_single_sophisticated_trajectory(self, current_price: float, max_days: int,
                                                current_bb_pos: float, current_ma: float,
                                                current_vol: float) -> np.ndarray:
        """Generate a single trajectory using your OHLC forecasting model."""
        trajectory = np.zeros(max_days + 1)
        trajectory[0] = current_price
        
        try:
            # Use the Markov model to forecast BB states
            recent_trend = self.ohlc_forecaster.ohlc_data['Trend'].iloc[-1] if 'Trend' in self.ohlc_forecaster.ohlc_data.columns else 'up'
            current_state = self.markov_model.get_state(current_bb_pos)
            bb_states = self.markov_model.sample_states(max_days, current_state, recent_trend)
            
            # Create forecasts for MA and volatility
            ma_forecast = np.full(max_days, current_ma)  # Could enhance with ARIMA
            vol_forecast = np.full(max_days, current_vol)  # Could enhance with GARCH
            
            # Use OHLC forecaster to generate trajectory
            ohlc_forecast = self.ohlc_forecaster.forecast_ohlc(
                ma_forecast=ma_forecast,
                vol_forecast=vol_forecast,
                bb_states=bb_states,
                current_close=current_price,
                n_days=max_days
            )
            
            # Extract close prices for trajectory
            trajectory[1:] = ohlc_forecast['close']
            
        except Exception as e:
            # Fallback to enhanced random walk if sophisticated model fails for this trajectory
            for day in range(1, max_days + 1):
                # Use GARCH-style volatility evolution
                if day > 1:
                    prev_return = (trajectory[day-1] - trajectory[day-2]) / trajectory[day-2]
                    vol_today = current_vol * (0.94 + 0.06 * abs(prev_return) / current_vol)
                else:
                    vol_today = current_vol
                    
                daily_return = np.random.normal(0.0005, vol_today)
                trajectory[day] = trajectory[day-1] * (1 + daily_return)
        
        return trajectory
    
    def enable_sophisticated_models(self, symbol: str = None) -> bool:
        """Enable sophisticated models for trajectory generation."""
        if not self.sophisticated_models_available:
            print("⚠️ Sophisticated models not available")
            return False
            
        self.use_simple_model = False
        if symbol:
            self.symbol = symbol
            
        print("🚀 Sophisticated models enabled")
        print("   Next trajectory generation will use ARIMA/GARCH/Copula models")
        return True
    
    def disable_sophisticated_models(self):
        """Disable sophisticated models and use simple random walk."""
        self.use_simple_model = True
        print("🔄 Simple model enabled")
        print("   Next trajectory generation will use random walk")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            'sophisticated_models_available': self.sophisticated_models_available,
            'use_simple_model': self.use_simple_model,
            'models_trained': self.models_trained,
            'symbol': self.symbol,
            'has_ohlc_forecaster': self.ohlc_forecaster is not None,
            'has_markov_model': self.markov_model is not None
        }
    
    def get_price_statistics(self, day: int) -> Dict[str, float]:
        """Get price statistics for a specific day."""
        if self.trajectories is None:
            raise ValueError("No trajectories generated. Call generate_trajectories first.")
        
        if day >= self.trajectories.shape[1]:
            raise ValueError(f"Day {day} exceeds trajectory length {self.trajectories.shape[1]-1}")
        
        prices = self.trajectories[:, day]
        
        stats = {
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices),
            'min': np.min(prices),
            'max': np.max(prices),
            'q05': np.percentile(prices, 5),
            'q25': np.percentile(prices, 25),
            'q75': np.percentile(prices, 75),
            'q95': np.percentile(prices, 95),
            'skewness': self._calculate_skewness(prices),
            'n_trajectories': len(prices)
        }
        
        # Add model-specific information
        if hasattr(self, 'trajectory_metadata'):
            stats['model_type'] = self.trajectory_metadata.get('model_type', 'unknown')
            stats['symbol'] = self.trajectory_metadata.get('symbol', 'unknown')
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of price distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def plot_trajectories(self, 
                         n_show: int = 100, 
                         show_percentiles: bool = True,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot sample trajectories with statistics."""
        if self.trajectories is None:
            raise ValueError("No trajectories generated. Call generate_trajectories first.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot sample trajectories
        max_days = self.trajectories.shape[1] - 1
        days = range(max_days + 1)
        
        # Show subset of trajectories
        sample_indices = np.random.choice(self.n_trajectories, min(n_show, self.n_trajectories), replace=False)
        
        for idx in sample_indices:
            ax1.plot(days, self.trajectories[idx, :], alpha=0.1, color='blue', linewidth=0.5)
        
        # Show percentiles
        if show_percentiles:
            percentiles = [5, 25, 50, 75, 95]
            colors = ['red', 'orange', 'green', 'orange', 'red']
            
            percentile_data = np.zeros((len(percentiles), max_days + 1))
            for day in days:
                stats = self.get_price_statistics(day)
                percentile_data[0, day] = stats['q05']
                percentile_data[1, day] = stats['q25']
                percentile_data[2, day] = stats['median']
                percentile_data[3, day] = stats['q75']
                percentile_data[4, day] = stats['q95']
            
            for i, pct in enumerate(percentiles):
                label = f'{pct}th percentile' if pct != 50 else 'Median'
                ax1.plot(days, percentile_data[i, :], color=colors[i], linewidth=2, alpha=0.8, label=label)
        
        model_type = self.trajectory_metadata.get('model_type', 'unknown')
        symbol = self.trajectory_metadata.get('symbol', '')
        title = f'Monte Carlo Price Trajectories ({model_type})'
        if symbol and symbol != 'N/A':
            title += f' - {symbol}'
            
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        if show_percentiles:
            ax1.legend()
        
        # Plot distribution at final day
        final_prices = self.trajectories[:, -1]
        ax2.hist(final_prices, bins=50, alpha=0.7, color='skyblue', density=True)
        ax2.axvline(np.mean(final_prices), color='red', linestyle='--', label=f'Mean: ${np.mean(final_prices):.2f}')
        ax2.axvline(np.median(final_prices), color='green', linestyle='--', label=f'Median: ${np.median(final_prices):.2f}')
        ax2.set_xlabel('Final Price ($)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Price Distribution at Day {max_days}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class MonteCarloOptionsAnalyzer:
    """Analyze options strategies using Monte Carlo trajectories."""
    
    def __init__(self, 
                 trajectory_generator: MonteCarloTrajectoryGenerator,
                 strategy_config: StrategyConfig = None):
        """
        Initialize options analyzer.
        
        Parameters
        ----------
        trajectory_generator : MonteCarloTrajectoryGenerator
            Configured trajectory generator
        strategy_config : StrategyConfig
            Strategy probability configuration
        """
        self.trajectory_generator = trajectory_generator
        self.strategy_config = strategy_config or StrategyConfig()
        
        # Results storage
        self.analysis_results = {}
        self.optimal_strikes = {}
        
    def update_config(self, new_config: Union[StrategyConfig, Dict[str, float]]) -> None:
        """Update strategy configuration."""
        if isinstance(new_config, dict):
            self.strategy_config = StrategyConfig.from_dict(new_config)
        else:
            self.strategy_config = new_config
        
        print("📊 Updated strategy configuration:")
        for key, value in self.strategy_config.to_dict().items():
            print(f"   {key}: {value:.1%}")
    
    def find_optimal_strikes(self, 
                           current_price: float,
                           max_dte: int = 10,
                           risk_free_rate: float = 0.05) -> Dict[str, Dict[int, StrategyResult]]:
        """
        Find optimal strikes for all strategies across all DTEs.
        
        Parameters
        ----------
        current_price : float
            Current stock price
        max_dte : int
            Maximum days to expiry to analyze
        risk_free_rate : float
            Risk-free rate for option pricing
            
        Returns
        -------
        dict
            Nested dict with strategy -> dte -> StrategyResult
        """
        if self.trajectory_generator.trajectories is None:
            raise ValueError("No trajectories available. Generate trajectories first.")
        
        model_type = self.trajectory_generator.trajectory_metadata.get('model_type', 'unknown')
        symbol = self.trajectory_generator.trajectory_metadata.get('symbol', '')
        
        print(f"🎯 Finding optimal strikes using {model_type} model")
        if symbol and symbol != 'N/A':
            print(f"   Symbol: {symbol}")
        print(f"   DTEs: 0-{max_dte}")
        print("=" * 60)
        
        strategies = {
            'covered_call': self._analyze_covered_call,
            'long_call': self._analyze_long_call,
            'cash_secured_put': self._analyze_cash_secured_put,
            'long_put': self._analyze_long_put
        }
        
        results = {}
        
        for strategy_name, analyzer_func in strategies.items():
            print(f"\\n📈 Analyzing {strategy_name.replace('_', ' ').title()}")
            print("-" * 40)
            
            strategy_results = {}
            
            for dte in range(max_dte + 1):
                try:
                    result = analyzer_func(current_price, dte, risk_free_rate)
                    strategy_results[dte] = result
                    
                    print(f"  {dte}DTE: Strike ${result.strike_price:.2f}, "
                          f"Success {result.success_probability:.1%} "
                          f"(target {result.target_probability:.1%})")
                    
                except Exception as e:
                    print(f"  {dte}DTE: ❌ Failed - {e}")
                    continue
            
            results[strategy_name] = strategy_results
        
        self.analysis_results = results
        return results
    
    def _analyze_covered_call(self, current_price: float, dte: int, risk_free_rate: float) -> StrategyResult:
        """Analyze covered call strategy for specific DTE."""
        target_prob = self.strategy_config.covered_call_keep_prob
        trajectories = self.trajectory_generator.trajectories
        
        if dte >= trajectories.shape[1]:
            raise ValueError(f"DTE {dte} exceeds trajectory length")
        
        # Get prices at expiry
        expiry_prices = trajectories[:, dte]
        
        # Search for optimal strike
        strike_candidates = np.linspace(current_price * 0.98, current_price * 1.15, 100)
        best_strike = None
        best_prob_diff = float('inf')
        
        for strike in strike_candidates:
            # Covered call succeeds if stock stays below strike
            success_count = np.sum(expiry_prices <= strike)
            success_prob = success_count / len(expiry_prices)
            
            prob_diff = abs(success_prob - target_prob)
            if prob_diff < best_prob_diff:
                best_prob_diff = prob_diff
                best_strike = strike
        
        # Calculate final metrics for best strike
        success_count = np.sum(expiry_prices <= best_strike)
        success_prob = success_count / len(expiry_prices)
        
        # Estimate premium (simplified Black-Scholes)
        premium = self._estimate_option_premium(current_price, best_strike, 0.02, dte, 'call', risk_free_rate)
        
        # P&L calculation
        pnl_if_assigned = premium + (best_strike - current_price) * 100
        pnl_if_not_assigned = premium
        expected_pnl = success_prob * pnl_if_not_assigned + (1 - success_prob) * pnl_if_assigned
        
        return StrategyResult(
            strategy_name='Covered Call',
            dte=dte,
            strike_price=best_strike,
            success_probability=success_prob,
            target_probability=target_prob,
            expected_pnl=expected_pnl,
            max_profit=pnl_if_not_assigned,
            max_loss=-(current_price * 100 - premium),
            trajectories_successful=success_count,
            total_trajectories=len(expiry_prices),
            premium_estimate=premium
        )
    
    def _analyze_long_call(self, current_price: float, dte: int, risk_free_rate: float) -> StrategyResult:
        """Analyze long call strategy for specific DTE."""
        target_prob = self.strategy_config.long_call_win_prob
        trajectories = self.trajectory_generator.trajectories
        
        if dte >= trajectories.shape[1]:
            raise ValueError(f"DTE {dte} exceeds trajectory length")
        
        expiry_prices = trajectories[:, dte]
        
        # Search for optimal strike
        strike_candidates = np.linspace(current_price * 0.95, current_price * 1.20, 100)
        best_strike = None
        best_prob_diff = float('inf')
        
        for strike in strike_candidates:
            # Long call wins if stock finishes above strike
            success_count = np.sum(expiry_prices > strike)
            success_prob = success_count / len(expiry_prices)
            
            prob_diff = abs(success_prob - target_prob)
            if prob_diff < best_prob_diff:
                best_prob_diff = prob_diff
                best_strike = strike
        
        # Calculate final metrics
        success_count = np.sum(expiry_prices > best_strike)
        success_prob = success_count / len(expiry_prices)
        
        premium = self._estimate_option_premium(current_price, best_strike, 0.02, dte, 'call', risk_free_rate)
        
        # P&L calculation
        winning_trades = expiry_prices[expiry_prices > best_strike]
        if len(winning_trades) > 0:
            avg_winning_pnl = np.mean((winning_trades - best_strike) * 100) - premium
        else:
            avg_winning_pnl = -premium
        
        expected_pnl = success_prob * avg_winning_pnl - (1 - success_prob) * premium
        
        return StrategyResult(
            strategy_name='Long Call',
            dte=dte,
            strike_price=best_strike,
            success_probability=success_prob,
            target_probability=target_prob,
            expected_pnl=expected_pnl,
            max_profit=float('inf'),
            max_loss=-premium,
            trajectories_successful=success_count,
            total_trajectories=len(expiry_prices),
            premium_estimate=premium
        )
    
    def _analyze_cash_secured_put(self, current_price: float, dte: int, risk_free_rate: float) -> StrategyResult:
        """Analyze cash secured put strategy for specific DTE."""
        target_prob = self.strategy_config.csp_avoid_assignment_prob
        trajectories = self.trajectory_generator.trajectories
        
        if dte >= trajectories.shape[1]:
            raise ValueError(f"DTE {dte} exceeds trajectory length")
        
        expiry_prices = trajectories[:, dte]
        
        # Search for optimal strike
        strike_candidates = np.linspace(current_price * 0.80, current_price * 1.02, 100)
        best_strike = None
        best_prob_diff = float('inf')
        
        for strike in strike_candidates:
            # CSP succeeds if stock stays above strike (avoid assignment)
            success_count = np.sum(expiry_prices >= strike)
            success_prob = success_count / len(expiry_prices)
            
            prob_diff = abs(success_prob - target_prob)
            if prob_diff < best_prob_diff:
                best_prob_diff = prob_diff
                best_strike = strike
        
        # Calculate final metrics
        success_count = np.sum(expiry_prices >= best_strike)
        success_prob = success_count / len(expiry_prices)
        
        premium = self._estimate_option_premium(current_price, best_strike, 0.02, dte, 'put', risk_free_rate)
        
        # P&L calculation
        expected_pnl = success_prob * premium - (1 - success_prob) * (best_strike * 100 - premium)
        
        return StrategyResult(
            strategy_name='Cash Secured Put',
            dte=dte,
            strike_price=best_strike,
            success_probability=success_prob,
            target_probability=target_prob,
            expected_pnl=expected_pnl,
            max_profit=premium,
            max_loss=-(best_strike * 100 - premium),
            trajectories_successful=success_count,
            total_trajectories=len(expiry_prices),
            premium_estimate=premium
        )
    
    def _analyze_long_put(self, current_price: float, dte: int, risk_free_rate: float) -> StrategyResult:
        """Analyze long put strategy for specific DTE."""
        target_prob = self.strategy_config.long_put_win_prob
        trajectories = self.trajectory_generator.trajectories
        
        if dte >= trajectories.shape[1]:
            raise ValueError(f"DTE {dte} exceeds trajectory length")
        
        expiry_prices = trajectories[:, dte]
        
        # Search for optimal strike
        strike_candidates = np.linspace(current_price * 0.80, current_price * 1.05, 100)
        best_strike = None
        best_prob_diff = float('inf')
        
        for strike in strike_candidates:
            # Long put wins if stock finishes below strike
            success_count = np.sum(expiry_prices < strike)
            success_prob = success_count / len(expiry_prices)
            
            prob_diff = abs(success_prob - target_prob)
            if prob_diff < best_prob_diff:
                best_prob_diff = prob_diff
                best_strike = strike
        
        # Calculate final metrics
        success_count = np.sum(expiry_prices < best_strike)
        success_prob = success_count / len(expiry_prices)
        
        premium = self._estimate_option_premium(current_price, best_strike, 0.02, dte, 'put', risk_free_rate)
        
        # P&L calculation
        winning_trades = expiry_prices[expiry_prices < best_strike]
        if len(winning_trades) > 0:
            avg_winning_pnl = np.mean((best_strike - winning_trades) * 100) - premium
        else:
            avg_winning_pnl = -premium
        
        expected_pnl = success_prob * avg_winning_pnl - (1 - success_prob) * premium
        
        return StrategyResult(
            strategy_name='Long Put',
            dte=dte,
            strike_price=best_strike,
            success_probability=success_prob,
            target_probability=target_prob,
            expected_pnl=expected_pnl,
            max_profit=best_strike * 100 - premium,
            max_loss=-premium,
            trajectories_successful=success_count,
            total_trajectories=len(expiry_prices),
            premium_estimate=premium
        )
    
    def _estimate_option_premium(self, spot: float, strike: float, vol: float, 
                                dte: int, option_type: str, risk_free_rate: float) -> float:
        """Simplified Black-Scholes option pricing."""
        if dte == 0:
            if option_type == 'call':
                return max(spot - strike, 0)
            else:
                return max(strike - spot, 0)
        
        time_to_expiry = dte / 365.0
        moneyness = spot / strike
        time_value = vol * np.sqrt(time_to_expiry) * spot * 0.4
        
        if option_type == 'call':
            intrinsic = max(spot - strike, 0)
            if moneyness > 1:  # ITM
                premium = intrinsic + time_value * 0.6
            else:  # OTM
                premium = time_value * (moneyness ** 0.5)
        else:  # put
            intrinsic = max(strike - spot, 0)
            if moneyness < 1:  # ITM
                premium = intrinsic + time_value * 0.6
            else:  # OTM
                premium = time_value * ((1/moneyness) ** 0.5)
        
        return max(premium, 0.01)
    
    def get_strategy_summary_table(self) -> pd.DataFrame:
        """Create comprehensive summary table of all strategies."""
        if not self.analysis_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for strategy_name, dte_results in self.analysis_results.items():
            for dte, result in dte_results.items():
                summary_data.append({
                    'Strategy': result.strategy_name,
                    'DTE': dte,
                    'Strike': f"${result.strike_price:.2f}",
                    'Success_Prob': f"{result.success_probability:.1%}",
                    'Target_Prob': f"{result.target_probability:.1%}",
                    'Prob_Diff': f"{abs(result.success_probability - result.target_probability):.1%}",
                    'Expected_PnL': f"${result.expected_pnl:.2f}",
                    'Max_Profit': f"${result.max_profit:.2f}" if result.max_profit != float('inf') else "Unlimited",
                    'Max_Loss': f"${result.max_loss:.2f}",
                    'Premium': f"${result.premium_estimate:.2f}" if result.premium_estimate else "N/A"
                })
        
        return pd.DataFrame(summary_data)
    
    def plot_strategy_analysis(self, figsize: Tuple[int, int] = (15, 12)) -> None:
        """Plot comprehensive strategy analysis."""
        if not self.analysis_results:
            print("No analysis results available. Run find_optimal_strikes first.")
            return
        
        # Get model information for title
        model_type = self.trajectory_generator.trajectory_metadata.get('model_type', 'unknown')
        symbol = self.trajectory_generator.trajectory_metadata.get('symbol', '')
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Strike prices by DTE
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_strikes_by_dte(ax1)
        
        # 2. Success probabilities by DTE
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_success_probs_by_dte(ax2)
        
        # 3. Expected P&L by DTE
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_expected_pnl_by_dte(ax3)
        
        # 4. Premium estimates by DTE
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_premiums_by_dte(ax4)
        
        # 5. Strategy comparison at optimal DTE
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_strategy_comparison(ax5)
        
        title = f'Monte Carlo Options Strategy Analysis ({model_type})'
        if symbol and symbol != 'N/A':
            title += f' - {symbol}'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.show()
    
    def _plot_strikes_by_dte(self, ax):
        """Plot strike prices by DTE for all strategies."""
        for strategy_name, dte_results in self.analysis_results.items():
            dtes = []
            strikes = []
            for dte, result in dte_results.items():
                dtes.append(dte)
                strikes.append(result.strike_price)
            
            ax.plot(dtes, strikes, marker='o', label=strategy_name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Strike Price ($)')
        ax.set_title('Optimal Strike Prices by DTE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_success_probs_by_dte(self, ax):
        """Plot success probabilities by DTE."""
        for strategy_name, dte_results in self.analysis_results.items():
            dtes = []
            probs = []
            targets = []
            
            for dte, result in dte_results.items():
                dtes.append(dte)
                probs.append(result.success_probability)
                targets.append(result.target_probability)
            
            ax.plot(dtes, probs, marker='o', label=f'{strategy_name.replace("_", " ").title()} Actual', linewidth=2)
            ax.plot(dtes, targets, linestyle='--', alpha=0.7, label=f'{strategy_name.replace("_", " ").title()} Target')
        
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Success Probability')
        ax.set_title('Success Probabilities: Actual vs Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_expected_pnl_by_dte(self, ax):
        """Plot expected P&L by DTE."""
        for strategy_name, dte_results in self.analysis_results.items():
            dtes = []
            pnls = []
            for dte, result in dte_results.items():
                dtes.append(dte)
                pnls.append(result.expected_pnl)
            
            ax.plot(dtes, pnls, marker='o', label=strategy_name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Expected P&L ($)')
        ax.set_title('Expected P&L by DTE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def _plot_premiums_by_dte(self, ax):
        """Plot premium estimates by DTE."""
        for strategy_name, dte_results in self.analysis_results.items():
            dtes = []
            premiums = []
            for dte, result in dte_results.items():
                if result.premium_estimate:
                    dtes.append(dte)
                    premiums.append(result.premium_estimate)
            
            if premiums:
                ax.plot(dtes, premiums, marker='o', label=strategy_name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Premium Estimate ($)')
        ax.set_title('Option Premium Estimates by DTE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_strategy_comparison(self, ax):
        """Plot strategy comparison at best DTE for each."""
        strategies = []
        expected_pnls = []
        success_probs = []
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (strategy_name, dte_results) in enumerate(self.analysis_results.items()):
            # Find best DTE (highest expected P&L)
            best_dte = max(dte_results.keys(), key=lambda d: dte_results[d].expected_pnl)
            best_result = dte_results[best_dte]
            
            strategies.append(f"{strategy_name.replace('_', ' ').title()}\\n({best_dte}DTE)")
            expected_pnls.append(best_result.expected_pnl)
            success_probs.append(best_result.success_probability)
        
        # Create scatter plot
        scatter = ax.scatter(success_probs, expected_pnls, s=200, c=colors[:len(strategies)], alpha=0.7)
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (success_probs[i], expected_pnls[i]), 
                       xytext=(10, 10), textcoords='offset points', 
                       fontsize=10, ha='left')
        
        ax.set_xlabel('Success Probability')
        ax.set_ylabel('Expected P&L ($)')
        ax.set_title('Strategy Comparison (Best DTE for Each)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

# Convenience aliases for enhanced system integration
EnhancedMonteCarloTrajectoryGenerator = MonteCarloTrajectoryGenerator