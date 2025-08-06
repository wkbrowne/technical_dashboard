"""
Global KDE Models for Training on All Data with Regime Resolution

This module implements the corrected approach where all models (close price KDE, 
open price KDE, and copula models) are trained on ALL data and resolved by:
- Volatility regime (low, medium, high volatility)
- Trend regime (bull, bear, sideways)

The key correction is that we train ONE global model on all stocks' data,
not individual models per stock.

Uses centralized regime configuration from config.regime_config.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import gaussian_kde
from scipy import stats
from collections import defaultdict

# Import centralized regime configuration and classifier
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.regime_config import REGIME_CONFIG, ModelRegimeConfig
from models.regime_classifier import REGIME_CLASSIFIER

warnings.filterwarnings('ignore')

try:
    from scipy.stats import gaussian_kde
    KDE_AVAILABLE = True
except ImportError:
    KDE_AVAILABLE = False
    print("Warning: scipy KDE not available")


# RegimeClassifier is now imported from models.regime_classifier as REGIME_CLASSIFIER


class GlobalClosePriceKDE:
    """
    Global Close Price KDE model trained on ALL stocks' data.
    Resolved by trend and volatility regimes.
    """
    
    def __init__(self, min_samples_per_regime: int = 100):
        self.min_samples_per_regime = min_samples_per_regime
        self.regime_classifier = REGIME_CLASSIFIER  # Use centralized classifier
        self.kde_models = {}  # {regime: kde_model}
        self.regime_stats = {}  # {regime: stats}
        self.fitted = False
        
    def fit_global_model(self, all_stock_data: Dict[str, pd.DataFrame]) -> 'GlobalClosePriceKDE':
        """
        Train global close price KDE on all stocks' data.
        
        Parameters
        ----------
        all_stock_data : Dict[str, pd.DataFrame]
            Dictionary of stock data {symbol: dataframe}
            Each DataFrame must have 'Close', 'MA', 'BB_Position' columns
        """
        print("🌍 Training Global Close Price KDE Models")
        print("=" * 60)
        
        # Collect close price features from all stocks
        all_features = []
        stock_count = 0
        
        for symbol, stock_df in all_stock_data.items():
            try:
                features = self._extract_close_features(stock_df, symbol)
                if len(features) > 20:  # Minimum requirement
                    all_features.append(features)
                    stock_count += 1
                    if stock_count % 50 == 0:
                        print(f"  📊 Processed {stock_count} stocks...")
            except Exception as e:
                print(f"  ⚠️ Error processing {symbol}: {str(e)[:50]}")
                continue
        
        if not all_features:
            raise ValueError("No valid stock data found for global close KDE training")
        
        # Combine all features
        combined_data = pd.concat(all_features, ignore_index=True)
        print(f"  📊 Combined data: {len(combined_data)} close price observations from {stock_count} stocks")
        
        # Train KDE models for each regime
        self._train_regime_kdes(combined_data)
        
        self.fitted = True
        print(f"✅ Global Close Price KDE training complete!")
        return self
    
    def _extract_close_features(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract close price features with regime classification.
        """
        df = stock_df.copy()
        required_cols = ['Close', 'MA']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Calculate returns for volatility classification
        returns = df['Close'].pct_change()
        
        # Classify regimes
        trend = self.regime_classifier.classify_trend(df['MA'])
        volatility = self.regime_classifier.classify_volatility(returns)
        combined_regime = self.regime_classifier.get_combined_regime(trend, volatility)
        
        # Calculate close price relative features
        close_to_ma = df['Close'] / df['MA'] - 1  # Deviation from MA
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'symbol': symbol,
            'close_to_ma_ratio': close_to_ma,
            'close_return': returns,
            'trend_regime': trend,
            'vol_regime': volatility,
            'combined_regime': combined_regime,
            'bb_position': df.get('BB_Position', 0),  # Use if available
        }).dropna()
        
        return features
    
    def _train_regime_kdes(self, combined_data: pd.DataFrame):
        """Train KDE models for each regime."""
        print(f"🎯 Training regime-specific Close Price KDE models...")
        
        regime_counts = combined_data['combined_regime'].value_counts()
        print(f"  📊 Found {len(regime_counts)} regimes:")
        
        successful_kdes = 0
        for regime, count in regime_counts.items():
            print(f"    {regime}: {count} samples", end="")
            
            if count >= self.min_samples_per_regime:
                regime_data = combined_data[combined_data['combined_regime'] == regime]
                
                try:
                    # Use close_to_ma_ratio as the main feature for KDE
                    feature_data = regime_data['close_to_ma_ratio'].values
                    
                    # Remove extreme outliers
                    q1, q99 = np.percentile(feature_data, [1, 99])
                    clean_data = feature_data[(feature_data >= q1) & (feature_data <= q99)]
                    
                    if len(clean_data) >= 20:
                        kde_model = gaussian_kde(clean_data)
                        self.kde_models[regime] = kde_model
                        
                        # Store regime statistics
                        self.regime_stats[regime] = {
                            'mean': np.mean(clean_data),
                            'std': np.std(clean_data),
                            'count': len(clean_data),
                            'q25': np.percentile(clean_data, 25),
                            'q75': np.percentile(clean_data, 75)
                        }
                        
                        print(f" ✅ KDE trained")
                        successful_kdes += 1
                    else:
                        print(f" ⚠️ Too few samples after cleaning ({len(clean_data)})")
                        
                except Exception as e:
                    print(f" ❌ KDE training failed: {str(e)[:30]}")
            else:
                print(f" ⚠️ Below minimum threshold ({self.min_samples_per_regime})")
        
        print(f"  ✅ Successfully trained {successful_kdes} Close Price KDE models")
        print(f"  📊 Total regimes with KDEs: {len(self.kde_models)}")
    
    def sample_close_price(self, regime: str, ma_value: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample close price for given regime and MA value.
        
        Parameters
        ----------
        regime : str
            Combined regime (e.g., 'bull_low')
        ma_value : float
            Moving average value
        n_samples : int
            Number of samples
            
        Returns
        -------
        np.ndarray
            Sampled close prices
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if regime in self.kde_models:
            # Sample from KDE
            ratios = self.kde_models[regime].resample(n_samples)[0]
            close_prices = ma_value * (1 + ratios)
        else:
            # Fallback to regime statistics or simple sampling
            if regime in self.regime_stats:
                mean_ratio = self.regime_stats[regime]['mean']
                std_ratio = self.regime_stats[regime]['std']
            else:
                # Ultimate fallback
                mean_ratio = 0.0
                std_ratio = 0.02
            
            ratios = np.random.normal(mean_ratio, std_ratio, n_samples)
            close_prices = ma_value * (1 + ratios)
        
        return close_prices


class GlobalOpenPriceKDE:
    """
    Global Open Price KDE model trained on ALL stocks' data.
    Models overnight gaps resolved by trend and volatility regimes.
    """
    
    def __init__(self, min_samples_per_regime: int = 100):
        self.min_samples_per_regime = min_samples_per_regime
        self.regime_classifier = REGIME_CLASSIFIER  # Use centralized classifier
        self.kde_models = {}
        self.regime_stats = {}
        self.fitted = False
        
    def fit_global_model(self, all_stock_data: Dict[str, pd.DataFrame]) -> 'GlobalOpenPriceKDE':
        """
        Train global open price KDE on all stocks' data.
        
        Parameters
        ----------
        all_stock_data : Dict[str, pd.DataFrame]
            Dictionary of stock data {symbol: dataframe}
        """
        print("🌍 Training Global Open Price KDE Models")
        print("=" * 60)
        
        # Collect gap features from all stocks
        all_features = []
        stock_count = 0
        
        for symbol, stock_df in all_stock_data.items():
            try:
                features = self._extract_gap_features(stock_df, symbol)
                if len(features) > 20:
                    all_features.append(features)
                    stock_count += 1
                    if stock_count % 50 == 0:
                        print(f"  📊 Processed {stock_count} stocks...")
            except Exception as e:
                print(f"  ⚠️ Error processing {symbol}: {str(e)[:50]}")
                continue
        
        if not all_features:
            raise ValueError("No valid stock data found for global open KDE training")
        
        # Combine all features
        combined_data = pd.concat(all_features, ignore_index=True)
        print(f"  📊 Combined data: {len(combined_data)} gap observations from {stock_count} stocks")
        
        # Train KDE models for each regime
        self._train_regime_kdes(combined_data)
        
        self.fitted = True
        print(f"✅ Global Open Price KDE training complete!")
        return self
    
    def _extract_gap_features(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract overnight gap features with regime classification."""
        df = stock_df.copy()
        
        # Calculate overnight gaps
        prev_close = df['Close'].shift(1)
        gaps = (df['Open'] - prev_close) / prev_close
        
        # Calculate returns for volatility classification
        returns = df['Close'].pct_change()
        
        # Classify regimes using previous day's MA (for gap prediction)
        ma_prev = df['MA'].shift(1)
        trend = self.regime_classifier.classify_trend(ma_prev)
        volatility = self.regime_classifier.classify_volatility(returns)
        combined_regime = self.regime_classifier.get_combined_regime(trend, volatility)
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'symbol': symbol,
            'gap_return': gaps,
            'prev_close': prev_close,
            'open_price': df['Open'],
            'trend_regime': trend,
            'vol_regime': volatility,
            'combined_regime': combined_regime,
            'prev_bb_position': df.get('BB_Position', 0).shift(1),
        }).dropna()
        
        return features
    
    def _train_regime_kdes(self, combined_data: pd.DataFrame):
        """Train KDE models for each regime."""
        print(f"🎯 Training regime-specific Open Price KDE models...")
        
        regime_counts = combined_data['combined_regime'].value_counts()
        print(f"  📊 Found {len(regime_counts)} regimes:")
        
        successful_kdes = 0
        for regime, count in regime_counts.items():
            print(f"    {regime}: {count} samples", end="")
            
            if count >= self.min_samples_per_regime:
                regime_data = combined_data[combined_data['combined_regime'] == regime]
                
                try:
                    # Use gap returns as main feature
                    gap_data = regime_data['gap_return'].values
                    
                    # Remove extreme outliers (beyond 5% gaps)
                    clean_gaps = gap_data[np.abs(gap_data) < 0.05]
                    
                    if len(clean_gaps) >= 20:
                        kde_model = gaussian_kde(clean_gaps)
                        self.kde_models[regime] = kde_model
                        
                        # Store regime statistics
                        self.regime_stats[regime] = {
                            'mean': np.mean(clean_gaps),
                            'std': np.std(clean_gaps),
                            'count': len(clean_gaps),
                            'q25': np.percentile(clean_gaps, 25),
                            'q75': np.percentile(clean_gaps, 75)
                        }
                        
                        print(f" ✅ KDE trained")
                        successful_kdes += 1
                    else:
                        print(f" ⚠️ Too few samples after cleaning ({len(clean_gaps)})")
                        
                except Exception as e:
                    print(f" ❌ KDE training failed: {str(e)[:30]}")
            else:
                print(f" ⚠️ Below minimum threshold ({self.min_samples_per_regime})")
        
        print(f"  ✅ Successfully trained {successful_kdes} Open Price KDE models")
        print(f"  📊 Total regimes with KDEs: {len(self.kde_models)}")
    
    def sample_gap(self, regime: str, n_samples: int = 1) -> np.ndarray:
        """Sample overnight gap for given regime."""
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if regime in self.kde_models:
            gaps = self.kde_models[regime].resample(n_samples)[0]
        else:
            # Fallback
            if regime in self.regime_stats:
                mean_gap = self.regime_stats[regime]['mean']
                std_gap = self.regime_stats[regime]['std']
            else:
                mean_gap = 0.0
                std_gap = 0.005  # 0.5% default gap volatility
            
            gaps = np.random.normal(mean_gap, std_gap, n_samples)
        
        return gaps


class GlobalHighLowCopula:
    """
    Global High-Low Copula model trained on ALL stocks' data.
    Models the joint distribution of high and low prices relative to close/open.
    """
    
    def __init__(self, min_samples_per_regime: int = 150):
        self.min_samples_per_regime = min_samples_per_regime
        self.regime_classifier = REGIME_CLASSIFIER  # Use centralized classifier
        self.copula_models = {}  # {regime: copula_data}
        self.regime_stats = {}
        self.fitted = False
        
    def fit_global_model(self, all_stock_data: Dict[str, pd.DataFrame]) -> 'GlobalHighLowCopula':
        """
        Train global high-low copula on all stocks' data.
        
        Parameters
        ----------
        all_stock_data : Dict[str, pd.DataFrame]
            Dictionary of stock data {symbol: dataframe}
        """
        print("🌍 Training Global High-Low Copula Models")
        print("=" * 60)
        
        # Collect high-low features from all stocks
        all_features = []
        stock_count = 0
        
        for symbol, stock_df in all_stock_data.items():
            try:
                features = self._extract_highlow_features(stock_df, symbol)
                if len(features) > 20:
                    all_features.append(features)
                    stock_count += 1
                    if stock_count % 50 == 0:
                        print(f"  📊 Processed {stock_count} stocks...")
            except Exception as e:
                print(f"  ⚠️ Error processing {symbol}: {str(e)[:50]}")
                continue
        
        if not all_features:
            raise ValueError("No valid stock data found for global copula training")
        
        # Combine all features
        combined_data = pd.concat(all_features, ignore_index=True)
        print(f"  📊 Combined data: {len(combined_data)} high-low observations from {stock_count} stocks")
        
        # Train copula models for each regime
        self._train_regime_copulas(combined_data)
        
        self.fitted = True
        print(f"✅ Global High-Low Copula training complete!")
        return self
    
    def _extract_highlow_features(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract high-low features with regime classification."""
        df = stock_df.copy()
        
        # Calculate reference price (average of open and close)
        ref_price = (df['Open'] + df['Close']) / 2
        
        # Calculate high/low relative to reference
        high_ratio = (df['High'] - ref_price) / ref_price
        low_ratio = (ref_price - df['Low']) / ref_price  # Positive values
        
        # Calculate returns for volatility classification
        returns = df['Close'].pct_change()
        
        # Classify regimes
        trend = self.regime_classifier.classify_trend(df['MA'])
        volatility = self.regime_classifier.classify_volatility(returns)
        combined_regime = self.regime_classifier.get_combined_regime(trend, volatility)
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'symbol': symbol,
            'high_ratio': high_ratio,
            'low_ratio': low_ratio,
            'ref_price': ref_price,
            'trend_regime': trend,
            'vol_regime': volatility,
            'combined_regime': combined_regime,
            'bb_position': df.get('BB_Position', 0),
        }).dropna()
        
        return features
    
    def _train_regime_copulas(self, combined_data: pd.DataFrame):
        """Train copula models for each regime."""
        print(f"🎯 Training regime-specific Copula models...")
        
        regime_counts = combined_data['combined_regime'].value_counts()
        print(f"  📊 Found {len(regime_counts)} regimes:")
        
        successful_copulas = 0
        for regime, count in regime_counts.items():
            print(f"    {regime}: {count} samples", end="")
            
            if count >= self.min_samples_per_regime:
                regime_data = combined_data[combined_data['combined_regime'] == regime]
                
                try:
                    # Extract high and low ratios
                    high_ratios = regime_data['high_ratio'].values
                    low_ratios = regime_data['low_ratio'].values
                    
                    # Remove extreme outliers
                    high_clean = high_ratios[(high_ratios >= 0) & (high_ratios <= 0.1)]  # Max 10% moves
                    low_clean = low_ratios[(low_ratios >= 0) & (low_ratios <= 0.1)]
                    
                    if len(high_clean) >= 50 and len(low_clean) >= 50:
                        # Store cleaned data for sampling (simple empirical copula)
                        self.copula_models[regime] = {
                            'high_data': high_clean,
                            'low_data': low_clean,
                            'correlation': np.corrcoef(high_clean[:min(len(high_clean), len(low_clean))], 
                                                     low_clean[:min(len(high_clean), len(low_clean))])[0, 1]
                        }
                        
                        # Store regime statistics
                        self.regime_stats[regime] = {
                            'high_mean': np.mean(high_clean),
                            'high_std': np.std(high_clean),
                            'low_mean': np.mean(low_clean),
                            'low_std': np.std(low_clean),
                            'correlation': self.copula_models[regime]['correlation'],
                            'count': min(len(high_clean), len(low_clean))
                        }
                        
                        print(f" ✅ Copula trained")
                        successful_copulas += 1
                    else:
                        print(f" ⚠️ Too few samples after cleaning")
                        
                except Exception as e:
                    print(f" ❌ Copula training failed: {str(e)[:30]}")
            else:
                print(f" ⚠️ Below minimum threshold ({self.min_samples_per_regime})")
        
        print(f"  ✅ Successfully trained {successful_copulas} Copula models")
        print(f"  📊 Total regimes with Copulas: {len(self.copula_models)}")
    
    def sample_high_low(self, regime: str, ref_price: float, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Sample high and low prices for given regime and reference price.
        
        Parameters
        ----------
        regime : str
            Combined regime
        ref_price : float
            Reference price (average of open/close)
        n_samples : int
            Number of samples
            
        Returns
        -------
        dict
            {'high': high_prices, 'low': low_prices}
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if regime in self.copula_models:
            # Sample from empirical copula
            copula_data = self.copula_models[regime]
            high_data = copula_data['high_data']
            low_data = copula_data['low_data']
            
            # Simple empirical sampling
            indices = np.random.choice(len(high_data), n_samples, replace=True)
            high_ratios = high_data[indices]
            low_ratios = low_data[indices[:len(low_data)]]  # Handle different lengths
            
        else:
            # Fallback
            if regime in self.regime_stats:
                high_mean = self.regime_stats[regime]['high_mean']
                high_std = self.regime_stats[regime]['high_std']
                low_mean = self.regime_stats[regime]['low_mean']
                low_std = self.regime_stats[regime]['low_std']
            else:
                high_mean, high_std = 0.01, 0.005  # 1% ± 0.5%
                low_mean, low_std = 0.01, 0.005
            
            high_ratios = np.random.normal(high_mean, high_std, n_samples)
            low_ratios = np.random.normal(low_mean, low_std, n_samples)
        
        # Convert ratios to prices
        high_prices = ref_price * (1 + high_ratios)
        low_prices = ref_price * (1 - low_ratios)
        
        return {'high': high_prices, 'low': low_prices}


# Convenience functions for easy integration
def train_global_models(all_stock_data: Dict[str, pd.DataFrame], 
                       min_samples: int = 100) -> Dict[str, object]:
    """
    Train all global models on the provided data.
    
    Parameters
    ----------
    all_stock_data : Dict[str, pd.DataFrame]
        Dictionary of stock data
    min_samples : int
        Minimum samples per regime
        
    Returns
    -------
    dict
        Dictionary of trained models
    """
    print("🚀 Training All Global Models on All Data")
    print("=" * 80)
    
    models = {}
    
    # Train Close Price KDE
    try:
        close_kde = GlobalClosePriceKDE(min_samples_per_regime=min_samples)
        close_kde.fit_global_model(all_stock_data)
        models['close_kde'] = close_kde
        print("✅ Global Close Price KDE trained")
    except Exception as e:
        print(f"❌ Global Close Price KDE failed: {str(e)[:60]}")
        models['close_kde'] = None
    
    # Train Open Price KDE
    try:
        open_kde = GlobalOpenPriceKDE(min_samples_per_regime=min_samples)
        open_kde.fit_global_model(all_stock_data)
        models['open_kde'] = open_kde
        print("✅ Global Open Price KDE trained")
    except Exception as e:
        print(f"❌ Global Open Price KDE failed: {str(e)[:60]}")
        models['open_kde'] = None
    
    # Train High-Low Copula
    try:
        hl_copula = GlobalHighLowCopula(min_samples_per_regime=min_samples + 50)  # Need more samples for copula
        hl_copula.fit_global_model(all_stock_data)
        models['hl_copula'] = hl_copula
        print("✅ Global High-Low Copula trained")
    except Exception as e:
        print(f"❌ Global High-Low Copula failed: {str(e)[:60]}")
        models['hl_copula'] = None
    
    successful_models = sum(1 for model in models.values() if model is not None)
    print(f"\n🎯 Global Model Training Results: {successful_models}/3 models trained successfully")
    
    return models