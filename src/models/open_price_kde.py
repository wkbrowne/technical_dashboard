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


# StockSpecificOpenKDE class removed - using global-only modeling architecture


class IntelligentOpenForecaster:
    """
    Main interface for intelligent open price forecasting using 
    global-only KDE models.
    """
    
    def __init__(self, global_model_path: Optional[str] = None):
        self.global_model = GlobalOpenPriceKDE()
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
    
    def forecast_open(self, symbol: str, prev_close: float, 
                     trend_regime: str, vol_regime: str,
                     additional_features: Optional[Dict] = None) -> Dict:
        """
        Forecast opening price using global-only KDE models.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (used for logging/tracking only)
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
        
        # Sample gap return using global model only
        gap_return = self.global_model.sample_gap_return(
            combined_regime, n_samples=1)[0]
        model_used = 'global_only'
        
        # Calculate forecasted open
        forecasted_open = prev_close * (1 + gap_return)
        
        # Estimate uncertainty by sampling multiple times
        n_uncertainty_samples = 100
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
        """Get comprehensive information about the global-only forecasting system."""
        info = {
            'global_model': {
                'fitted': self.global_model.fitted,
                'regimes': len(self.global_model.global_regime_stats),
                'kde_models': len(self.global_model.global_kde_models)
            },
            'architecture': 'global_only'
        }
        
        return info