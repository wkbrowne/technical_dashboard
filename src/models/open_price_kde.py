"""
Intelligent Open Price Modeling using Global + Stock-Specific KDEs
Resolved by Trend and Volatility Regimes with Silverman's Bandwidth Selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import gaussian_kde
from collections import defaultdict
import pickle
import os
from pathlib import Path

class GlobalOpenPriceKDE:
    """
    Global KDE model for open price patterns across all stocks.
    Trained on gap patterns resolved by trend and volatility regimes.
    """
    
    def __init__(self, min_samples_per_regime: int = 50):
        self.min_samples_per_regime = min_samples_per_regime
        self.global_kde_models = {}  # {regime_key: kde_model}
        self.global_regime_stats = {}  # {regime_key: {'mean', 'std', 'count'}}
        self.fitted = False
        
        # Define trend and volatility thresholds
        self.trend_thresholds = {
            'strong_bull': 0.002,    # >0.2% daily MA slope
            'bull': 0.0005,          # 0.05-0.2% daily MA slope
            'neutral': -0.0005,      # -0.05% to 0.05% daily MA slope
            'bear': -0.002,          # -0.2% to -0.05% daily MA slope
            'strong_bear': float('-inf')  # <-0.2% daily MA slope
        }
        
    def fit_global_model(self, all_stock_data: Dict[str, pd.DataFrame]) -> 'GlobalOpenPriceKDE':
        """
        Fit global KDE models using data from all stocks.
        
        Parameters:
        -----------
        all_stock_data : Dict[str, pd.DataFrame]
            Dictionary of stock data {symbol: ohlc_dataframe}
            Each DataFrame must have Open, High, Low, Close, BB_Width, BB_MA columns
        """
        print("🌍 Training Global Open Price KDE Models")
        print("=" * 60)
        
        # Collect gap data from all stocks
        all_gap_data = []
        stock_count = 0
        
        for symbol, stock_df in all_stock_data.items():
            try:
                gap_data = self._extract_gap_features(stock_df, symbol)
                if len(gap_data) > 10:  # Minimum data requirement
                    all_gap_data.append(gap_data)
                    stock_count += 1
                    if stock_count % 100 == 0:
                        print(f"  Processed {stock_count} stocks...")
            except Exception as e:
                print(f"  ⚠️ Error processing {symbol}: {e}")
                continue
        
        if not all_gap_data:
            raise ValueError("No valid stock data found for global model training")
        
        # Combine all gap data
        combined_data = pd.concat(all_gap_data, ignore_index=True)
        print(f"  📊 Combined data: {len(combined_data)} gap observations from {stock_count} stocks")
        
        # Train KDE models for each regime
        self._train_regime_kdes(combined_data)
        
        self.fitted = True
        print(f"✅ Global model training complete!")
        return self
    
    def _extract_gap_features(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract gap features (overnight returns) with regime classification.
        """
        df = stock_df.copy()
        
        # Calculate overnight gaps (open vs previous close)
        df['Prev_Close'] = df['Close'].shift(1)
        df['Gap'] = df['Open'] - df['Prev_Close']
        df['Gap_Return'] = df['Gap'] / df['Prev_Close']
        
        # Calculate moving average slope for trend classification
        if 'BB_MA' in df.columns:
            ma_col = 'BB_MA'
        else:
            # Calculate MA if not present
            df['BB_MA'] = df['Close'].rolling(window=20).mean()
            ma_col = 'BB_MA'
            
        df['MA_Slope'] = df[ma_col].pct_change(periods=5)  # 5-day slope
        
        # Classify trend regime
        df['Trend_Regime'] = self._classify_trend_regime(df['MA_Slope'])
        
        # Classify volatility regime
        if 'BB_Width' in df.columns:
            vol_median = df['BB_Width'].median()
            df['Vol_Regime'] = np.where(df['BB_Width'] > vol_median, 'High_Vol', 'Low_Vol')
        else:
            # Calculate volatility if not present
            returns = df['Close'].pct_change()
            vol_rolling = returns.rolling(window=20).std()
            vol_median = vol_rolling.median()
            df['Vol_Regime'] = np.where(vol_rolling > vol_median, 'High_Vol', 'Low_Vol')
        
        # Create combined regime
        df['Combined_Regime'] = df['Trend_Regime'] + '_' + df['Vol_Regime']
        
        # Additional features
        df['Day_of_Week'] = df.index.dayofweek
        df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
        df['Previous_Intraday_Return'] = df['Intraday_Return'].shift(1)
        
        # Select relevant columns
        feature_cols = [
            'Gap_Return', 'Combined_Regime', 'Trend_Regime', 'Vol_Regime',
            'Day_of_Week', 'Previous_Intraday_Return', 'MA_Slope'
        ]
        
        result = df[feature_cols].dropna().copy()
        result['Symbol'] = symbol
        
        return result
    
    def _classify_trend_regime(self, ma_slopes: pd.Series) -> pd.Series:
        """Classify trend regime based on MA slope."""
        conditions = [
            ma_slopes > self.trend_thresholds['strong_bull'],
            ma_slopes > self.trend_thresholds['bull'],
            ma_slopes > self.trend_thresholds['neutral'],
            ma_slopes > self.trend_thresholds['bear']
        ]
        
        choices = ['Strong_Bull', 'Bull', 'Neutral', 'Bear']
        
        return pd.Series(np.select(conditions, choices, default='Strong_Bear'), 
                        index=ma_slopes.index)
    
    def _train_regime_kdes(self, combined_data: pd.DataFrame) -> None:
        """Train KDE models for each regime."""
        print(f"\n🎯 Training regime-specific KDE models...")
        
        regimes = combined_data['Combined_Regime'].value_counts()
        print(f"  📊 Found {len(regimes)} regimes:")
        
        successful_models = 0
        
        for regime, count in regimes.items():
            print(f"    {regime}: {count} samples", end="")
            
            if count >= self.min_samples_per_regime:
                regime_data = combined_data[combined_data['Combined_Regime'] == regime]
                gap_returns = regime_data['Gap_Return'].values
                
                # Remove extreme outliers (beyond 3 standard deviations)
                mean_gap = np.mean(gap_returns)
                std_gap = np.std(gap_returns)
                filtered_gaps = gap_returns[
                    np.abs(gap_returns - mean_gap) <= 3 * std_gap
                ]
                
                if len(filtered_gaps) >= self.min_samples_per_regime:
                    try:
                        # Fit KDE with Silverman's rule
                        kde_model = gaussian_kde(filtered_gaps)
                        kde_model.set_bandwidth(bw_method='silverman')
                        
                        self.global_kde_models[regime] = kde_model
                        self.global_regime_stats[regime] = {
                            'mean': np.mean(filtered_gaps),
                            'std': np.std(filtered_gaps),
                            'count': len(filtered_gaps),
                            'raw_count': count
                        }
                        
                        successful_models += 1
                        print(" ✅")
                        
                    except Exception as e:
                        print(f" ❌ KDE failed: {e}")
                        # Fallback to normal distribution
                        self.global_regime_stats[regime] = {
                            'mean': np.mean(filtered_gaps),
                            'std': np.std(filtered_gaps),
                            'count': len(filtered_gaps),
                            'raw_count': count,
                            'model_type': 'normal_fallback'
                        }
                else:
                    print(f" ⚠️ Insufficient data after filtering")
            else:
                print(f" ⚠️ Below minimum threshold ({self.min_samples_per_regime})")
        
        print(f"  ✅ Successfully trained {successful_models} KDE models")
        print(f"  📊 Total regimes with stats: {len(self.global_regime_stats)}")
    
    def sample_gap_return(self, regime: str, n_samples: int = 1) -> np.ndarray:
        """Sample gap returns from the appropriate regime model."""
        if regime in self.global_kde_models:
            # Use trained KDE
            return self.global_kde_models[regime].resample(n_samples)[0]
        elif regime in self.global_regime_stats:
            # Use normal distribution fallback
            stats = self.global_regime_stats[regime]
            return np.random.normal(stats['mean'], stats['std'], n_samples)
        else:
            # Default fallback: small random gap
            return np.random.normal(0.0, 0.002, n_samples)  # 0.2% std
    
    def get_regime_info(self) -> Dict:
        """Get information about trained regimes."""
        info = {}
        
        for regime in self.global_regime_stats:
            stats = self.global_regime_stats[regime]
            info[regime] = {
                'samples_used': stats['count'],
                'raw_samples': stats['raw_count'],
                'mean_gap': stats['mean'],
                'std_gap': stats['std'],
                'has_kde': regime in self.global_kde_models,
                'model_type': stats.get('model_type', 'kde')
            }
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the trained global model."""
        model_data = {
            'global_kde_models': self.global_kde_models,
            'global_regime_stats': self.global_regime_stats,
            'trend_thresholds': self.trend_thresholds,
            'min_samples_per_regime': self.min_samples_per_regime,
            'fitted': self.fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Global model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'GlobalOpenPriceKDE':
        """Load a trained global model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.global_kde_models = model_data['global_kde_models']
        self.global_regime_stats = model_data['global_regime_stats']
        self.trend_thresholds = model_data['trend_thresholds']
        self.min_samples_per_regime = model_data['min_samples_per_regime']
        self.fitted = model_data['fitted']
        
        print(f"✅ Global model loaded from {filepath}")
        return self


class StockSpecificOpenKDE:
    """
    Stock-specific KDE fine-tuning using global priors.
    Combines global market patterns with individual stock characteristics.
    """
    
    def __init__(self, global_model: GlobalOpenPriceKDE, 
                 adaptation_weight: float = 0.3,
                 min_stock_samples: int = 20):
        self.global_model = global_model
        self.adaptation_weight = adaptation_weight  # How much to adapt from global
        self.min_stock_samples = min_stock_samples
        self.stock_kde_models = {}
        self.stock_regime_stats = {}
        self.fitted_stocks = set()
        
    def fit_stock_model(self, symbol: str, stock_data: pd.DataFrame) -> 'StockSpecificOpenKDE':
        """
        Fit stock-specific model using global priors + stock-specific data.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        stock_data : pd.DataFrame
            Stock OHLC data
        """
        print(f"🏢 Fitting stock-specific model for {symbol}")
        
        # Extract stock-specific gap features
        stock_gap_data = self.global_model._extract_gap_features(stock_data, symbol)
        
        if len(stock_gap_data) < self.min_stock_samples:
            print(f"  ⚠️ Insufficient data ({len(stock_gap_data)} < {self.min_stock_samples})")
            print(f"  🌍 Using global model only for {symbol}")
            self.fitted_stocks.add(symbol)
            return self
        
        print(f"  📊 Stock data: {len(stock_gap_data)} gap observations")
        
        # Fit regime-specific models
        stock_regimes = stock_gap_data['Combined_Regime'].value_counts()
        adapted_models = 0
        
        for regime, count in stock_regimes.items():
            if count >= 10:  # Minimum for stock-specific adaptation
                stock_regime_data = stock_gap_data[
                    stock_gap_data['Combined_Regime'] == regime
                ]['Gap_Return'].values
                
                # Combine with global data for better estimation
                if regime in self.global_model.global_kde_models:
                    # Sample from global model for augmentation
                    global_samples = self.global_model.sample_gap_return(regime, 
                                                                       n_samples=max(50, count))
                    
                    # Weighted combination: more stock data = less global influence
                    stock_weight = min(count / 100.0, 0.8)  # Cap at 80% stock-specific
                    global_weight = 1 - stock_weight
                    
                    # Create hybrid dataset
                    n_global = int(global_weight * len(stock_regime_data))
                    combined_data = np.concatenate([
                        stock_regime_data,
                        global_samples[:n_global]
                    ])
                    
                    try:
                        # Fit hybrid KDE
                        hybrid_kde = gaussian_kde(combined_data)
                        hybrid_kde.set_bandwidth(bw_method='silverman')
                        
                        regime_key = f"{symbol}_{regime}"
                        self.stock_kde_models[regime_key] = hybrid_kde
                        self.stock_regime_stats[regime_key] = {
                            'stock_samples': count,
                            'global_samples': n_global,
                            'mean': np.mean(combined_data),
                            'std': np.std(combined_data),
                            'stock_weight': stock_weight
                        }
                        
                        adapted_models += 1
                        
                    except Exception as e:
                        print(f"    ❌ Failed to fit {regime}: {e}")
        
        print(f"  ✅ Adapted {adapted_models} regime models for {symbol}")
        self.fitted_stocks.add(symbol)
        return self
    
    def sample_gap_return(self, symbol: str, regime: str, n_samples: int = 1) -> np.ndarray:
        """
        Sample gap returns using stock-specific model if available, 
        otherwise fall back to global model.
        """
        stock_regime_key = f"{symbol}_{regime}"
        
        if stock_regime_key in self.stock_kde_models:
            # Use stock-specific adapted model
            return self.stock_kde_models[stock_regime_key].resample(n_samples)[0]
        else:
            # Fall back to global model
            return self.global_model.sample_gap_return(regime, n_samples)
    
    def get_model_info(self, symbol: str) -> Dict:
        """Get information about stock-specific adaptations."""
        info = {
            'symbol': symbol,
            'is_fitted': symbol in self.fitted_stocks,
            'adapted_regimes': {},
            'global_fallback_regimes': []
        }
        
        if symbol in self.fitted_stocks:
            # Find adapted regimes
            for key, stats in self.stock_regime_stats.items():
                if key.startswith(f"{symbol}_"):
                    regime = key.replace(f"{symbol}_", "")
                    info['adapted_regimes'][regime] = stats
            
            # Find global fallback regimes
            for regime in self.global_model.global_regime_stats:
                if f"{symbol}_{regime}" not in self.stock_kde_models:
                    info['global_fallback_regimes'].append(regime)
        
        return info


class IntelligentOpenForecaster:
    """
    Main interface for intelligent open price forecasting using 
    global + stock-specific KDE models.
    """
    
    def __init__(self, global_model_path: Optional[str] = None):
        self.global_model = GlobalOpenPriceKDE()
        self.stock_models = {}  # {symbol: StockSpecificOpenKDE}
        self.global_model_path = global_model_path
        
        # Load global model if path provided and exists
        if global_model_path and os.path.exists(global_model_path):
            self.global_model.load_model(global_model_path)
    
    def train_global_model(self, all_stock_data: Dict[str, pd.DataFrame], 
                          save_path: Optional[str] = None) -> 'IntelligentOpenForecaster':
        """Train the global model on all available stock data."""
        self.global_model.fit_global_model(all_stock_data)
        
        if save_path:
            self.global_model.save_model(save_path)
            self.global_model_path = save_path
        
        return self
    
    def add_stock_model(self, symbol: str, stock_data: pd.DataFrame) -> 'IntelligentOpenForecaster':
        """Add stock-specific fine-tuning for a particular symbol."""
        if not self.global_model.fitted:
            raise ValueError("Global model must be trained first")
        
        stock_model = StockSpecificOpenKDE(self.global_model)
        stock_model.fit_stock_model(symbol, stock_data)
        self.stock_models[symbol] = stock_model
        
        return self
    
    def forecast_open(self, symbol: str, prev_close: float, 
                     trend_regime: str, vol_regime: str,
                     additional_features: Optional[Dict] = None) -> Dict:
        """
        Forecast opening price using intelligent KDE models.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        prev_close : float
            Previous day's closing price
        trend_regime : str
            Current trend regime
        vol_regime : str
            Current volatility regime
        additional_features : dict, optional
            Additional features like day_of_week, etc.
            
        Returns:
        --------
        dict
            Forecast results with mean, confidence intervals, etc.
        """
        combined_regime = f"{trend_regime}_{vol_regime}"
        
        # Sample gap return
        if symbol in self.stock_models:
            gap_return = self.stock_models[symbol].sample_gap_return(
                symbol, combined_regime, n_samples=1)[0]
            model_used = 'stock_specific'
        else:
            gap_return = self.global_model.sample_gap_return(
                combined_regime, n_samples=1)[0]
            model_used = 'global_only'
        
        # Calculate forecasted open
        forecasted_open = prev_close * (1 + gap_return)
        
        # Estimate uncertainty by sampling multiple times
        n_uncertainty_samples = 100
        if symbol in self.stock_models:
            gap_samples = self.stock_models[symbol].sample_gap_return(
                symbol, combined_regime, n_samples=n_uncertainty_samples)
        else:
            gap_samples = self.global_model.sample_gap_return(
                combined_regime, n_samples=n_uncertainty_samples)
        
        open_samples = prev_close * (1 + gap_samples)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(open_samples, 2.5)
        ci_upper = np.percentile(open_samples, 97.5)
        
        return {
            'forecasted_open': forecasted_open,
            'gap_return': gap_return,
            'confidence_interval': (ci_lower, ci_upper),
            'uncertainty_std': np.std(open_samples),
            'model_used': model_used,
            'regime': combined_regime
        }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive information about the forecasting system."""
        info = {
            'global_model': {
                'fitted': self.global_model.fitted,
                'regimes': len(self.global_model.global_regime_stats),
                'kde_models': len(self.global_model.global_kde_models)
            },
            'stock_models': {}
        }
        
        for symbol, stock_model in self.stock_models.items():
            info['stock_models'][symbol] = stock_model.get_model_info(symbol)
        
        return info