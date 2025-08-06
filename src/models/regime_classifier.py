"""
Centralized Regime Classifier for All Models

This module provides a unified regime classification system that can be used
across all models including:
- Markov models
- Close price KDE models  
- Open price KDE models
- High-low copula models

Uses the centralized regime configuration from config.regime_config.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import sys
import os

# Import centralized regime configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.regime_config import REGIME_CONFIG, ModelRegimeConfig


class UnifiedRegimeClassifier:
    """
    Unified regime classification for trend and volatility.
    Used across all models for consistent regime identification.
    Uses centralized regime configuration.
    """
    
    def __init__(self):
        self.vol_thresholds = None  # Will be set after seeing data
        self._reload_config()
    
    def _reload_config(self):
        """Reload configuration from centralized source"""
        # Use centralized configuration
        self.trend_thresholds = REGIME_CONFIG.trend.thresholds.copy()
        self.vol_percentiles = REGIME_CONFIG.volatility.percentile_thresholds.copy()
        
        # Configuration parameters
        self.trend_window = ModelRegimeConfig.get_trend_lookback()
        self.vol_window = ModelRegimeConfig.get_volatility_lookback()
        self.vol_min_periods = ModelRegimeConfig.get_volatility_min_periods()
    
    def classify_trend(self, ma_series: pd.Series, window: int = None, return_states: bool = False) -> pd.Series:
        """
        Classify trend based on moving average slope using integer states.
        
        Parameters
        ----------
        ma_series : pd.Series
            Moving average series
        window : int, optional
            Window for slope calculation (uses config default if None)
        return_states : bool, optional
            If True, return integer states; if False, return descriptive labels
            
        Returns
        -------
        pd.Series
            Trend classifications (integer states or labels)
        """
        # Use configured window if not specified
        if window is None:
            window = self.trend_window
            
        # Calculate slope as percentage change over window
        slope = ma_series.pct_change(periods=window)
        
        # Initialize with fallback state
        trend_states = pd.Series(index=ma_series.index, dtype='int32')
        trend_states[:] = REGIME_CONFIG.fallback_trend_state
        
        # Classify trends using configured thresholds
        thresholds = self.trend_thresholds
        valid_slope = slope.dropna()
        
        if len(valid_slope) > 0:
            # Assign states based on thresholds (from lowest to highest)
            for i, threshold in enumerate(thresholds):
                trend_states[slope >= threshold] = i + 1
            
            # Values below first threshold get state 0
            trend_states[slope < thresholds[0]] = 0
        
        # Forward fill missing values
        trend_states = trend_states.ffill()
        # Ensure all values are integers
        trend_states = trend_states.astype('int32')
        
        if return_states:
            return trend_states
        else:
            # Convert to labels, handling NaN values
            trend_labels = pd.Series(index=trend_states.index, dtype='object')
            for idx, state in trend_states.items():
                if pd.notna(state):
                    trend_labels[idx] = REGIME_CONFIG.trend.get_state_label(int(state))
                else:
                    trend_labels[idx] = REGIME_CONFIG.fallback_trend
            return trend_labels
    
    def classify_volatility(self, returns: pd.Series, window: int = None, return_states: bool = False) -> pd.Series:
        """
        Classify volatility regime based on realized volatility percentiles using integer states.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        window : int, optional
            Window for volatility calculation (uses config default if None)
        return_states : bool, optional
            If True, return integer states; if False, return descriptive labels
            
        Returns
        -------
        pd.Series
            Volatility classifications (integer states or labels)
        """
        # Use configured window if not specified
        if window is None:
            window = self.vol_window
            
        # Calculate realized volatility
        realized_vol = returns.rolling(window=window, min_periods=self.vol_min_periods).std()
        
        # Set thresholds if not already set
        if self.vol_thresholds is None:
            vol_values = realized_vol.dropna()
            
            if len(vol_values) == 0:
                # Fallback thresholds when no data
                percentiles = self.vol_percentiles
                self.vol_thresholds = [0.01, 0.02]  # Simple fallback
            elif len(vol_values) < 10:
                # Use simple thresholds for small datasets
                vol_mean = vol_values.mean()
                vol_std = vol_values.std()
                # Create thresholds around mean
                self.vol_thresholds = [
                    vol_mean - 0.5 * vol_std,
                    vol_mean + 0.5 * vol_std
                ]
            else:
                # Use percentiles for larger datasets
                self.vol_thresholds = [
                    np.percentile(vol_values, p) for p in self.vol_percentiles
                ]
        
        # Initialize with fallback state
        vol_states = pd.Series(index=returns.index, dtype='int32')
        vol_states[:] = REGIME_CONFIG.fallback_volatility_state
        
        # Classify volatility using thresholds
        valid_vol = realized_vol.dropna()
        
        if len(valid_vol) > 0 and len(self.vol_thresholds) > 0:
            # Assign states based on thresholds (from lowest to highest)
            for i, threshold in enumerate(self.vol_thresholds):
                vol_states[realized_vol >= threshold] = i + 1
            
            # Values below first threshold get state 0
            vol_states[realized_vol < self.vol_thresholds[0]] = 0
        
        # Forward fill missing values
        vol_states = vol_states.ffill()
        # Ensure all values are integers
        vol_states = vol_states.astype('int32')
        
        if return_states:
            return vol_states
        else:
            # Convert to labels, handling NaN values
            vol_labels = pd.Series(index=vol_states.index, dtype='object')
            for idx, state in vol_states.items():
                if pd.notna(state):
                    vol_labels[idx] = REGIME_CONFIG.volatility.get_state_label(int(state))
                else:
                    vol_labels[idx] = REGIME_CONFIG.fallback_volatility
            return vol_labels
    
    def get_combined_regime(self, trend: pd.Series, volatility: pd.Series) -> pd.Series:
        """
        Combine trend and volatility into unified regime.
        
        Parameters
        ----------
        trend : pd.Series
            Trend classifications
        volatility : pd.Series
            Volatility classifications
            
        Returns
        -------
        pd.Series
            Combined regime (e.g., 'bull_low', 'bear_high')
        """
        return trend.astype(str) + REGIME_CONFIG.regime_separator + volatility.astype(str)
    
    def classify_regimes(self, ma_series: pd.Series, returns: pd.Series, return_states: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Classify both trend and volatility regimes in one call.
        
        Parameters
        ----------
        ma_series : pd.Series
            Moving average series for trend classification
        returns : pd.Series
            Return series for volatility classification
        return_states : bool, optional
            If True, return integer states; if False, return descriptive labels
            
        Returns
        -------
        tuple
            (trend_regime, vol_regime, combined_regime)
        """
        trend_regime = self.classify_trend(ma_series, return_states=return_states)
        vol_regime = self.classify_volatility(returns, return_states=return_states)
        
        if return_states:
            # For states, combine as tuple pairs
            combined_regime = pd.Series(
                [(t, v) for t, v in zip(trend_regime, vol_regime)], 
                index=trend_regime.index
            )
        else:
            # For labels, combine as string
            combined_regime = self.get_combined_regime(trend_regime, vol_regime)
        
        return trend_regime, vol_regime, combined_regime
    
    def classify_regimes_as_states(self, ma_series: pd.Series, returns: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Classify regimes and return integer states.
        
        Parameters
        ----------
        ma_series : pd.Series
            Moving average series for trend classification
        returns : pd.Series
            Return series for volatility classification
            
        Returns
        -------
        tuple
            (trend_states, vol_states, combined_state_tuples)
        """
        return self.classify_regimes(ma_series, returns, return_states=True)
    
    def classify_regimes_as_labels(self, ma_series: pd.Series, returns: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Classify regimes and return descriptive labels.
        
        Parameters
        ----------
        ma_series : pd.Series
            Moving average series for trend classification
        returns : pd.Series
            Return series for volatility classification
            
        Returns
        -------
        tuple
            (trend_labels, vol_labels, combined_labels)
        """
        return self.classify_regimes(ma_series, returns, return_states=False)
    
    def parse_combined_regime(self, combined_regime: str) -> Tuple[str, str]:
        """
        Parse combined regime into trend and volatility components.
        
        Parameters
        ----------
        combined_regime : str
            Combined regime string (e.g., 'bull_low')
            
        Returns
        -------
        tuple
            (trend_component, volatility_component)
        """
        return REGIME_CONFIG.parse_combined_regime(combined_regime)
    
    def reload_config(self):
        """Force reload of configuration from centralized source"""
        self._reload_config() 
        # Reset volatility thresholds so they'll be recalculated
        self.vol_thresholds = None
    
    def get_regime_info(self) -> dict:
        """Get information about current regime configuration"""
        return {
            'trend_thresholds': self.trend_thresholds.copy(),
            'vol_percentiles': self.vol_percentiles.copy(), 
            'vol_thresholds': self.vol_thresholds.copy() if self.vol_thresholds else None,
            'trend_window': self.trend_window,
            'vol_window': self.vol_window,
            'vol_min_periods': self.vol_min_periods,
            'n_trend_states': REGIME_CONFIG.trend.n_states,
            'n_vol_states': REGIME_CONFIG.volatility.n_states,
            'trend_state_labels': REGIME_CONFIG.trend.get_all_labels(),
            'vol_state_labels': REGIME_CONFIG.volatility.get_all_labels(),
            'fallback_trend': REGIME_CONFIG.fallback_trend,
            'fallback_volatility': REGIME_CONFIG.fallback_volatility,
            'fallback_trend_state': REGIME_CONFIG.fallback_trend_state,
            'fallback_volatility_state': REGIME_CONFIG.fallback_volatility_state,
            'all_possible_regimes': REGIME_CONFIG.get_all_combined_regimes(),
            'all_possible_states': REGIME_CONFIG.get_all_combined_states()
        }


# Create global instance for easy importing
REGIME_CLASSIFIER = UnifiedRegimeClassifier()


def get_regime_classifier() -> UnifiedRegimeClassifier:
    """Get the global regime classifier instance"""
    return REGIME_CLASSIFIER


def classify_stock_regimes(stock_data: pd.DataFrame, 
                          ma_column: str = 'MA',
                          close_column: str = 'Close') -> pd.DataFrame:
    """
    Convenience function to classify regimes for a stock DataFrame.
    
    Parameters
    ----------
    stock_data : pd.DataFrame
        Stock data with MA and Close columns
    ma_column : str
        Name of moving average column
    close_column : str  
        Name of close price column
        
    Returns
    -------
    pd.DataFrame
        Original data with added regime columns
    """
    df = stock_data.copy()
    
    # Calculate returns
    returns = df[close_column].pct_change()
    
    # Classify regimes
    trend, vol, combined = REGIME_CLASSIFIER.classify_regimes(df[ma_column], returns)
    
    # Add regime columns
    df['trend_regime'] = trend
    df['vol_regime'] = vol  
    df['combined_regime'] = combined
    
    return df


def classify_hurst_regime(hurst_series: pd.Series, 
                         thresholds: Tuple[float, float] = (0.4, 0.6)) -> pd.Series:
    """
    Classify regime based on Hurst exponent values with flexible cutoff specification.
    
    Parameters
    ----------
    hurst_series : pd.Series
        Series of Hurst exponent values
    thresholds : tuple of float, default=(0.4, 0.6)
        Tuple of (mean_reverting_threshold, trending_threshold)
        
    Returns
    -------
    pd.Series
        Series of regime labels per point ('mean-reverting', 'neutral', 'trending', 'unknown')
    """
    mean_reverting_threshold, trending_threshold = thresholds
    
    # Validate thresholds
    if not (0 < mean_reverting_threshold < trending_threshold < 1):
        raise ValueError("Thresholds must satisfy: 0 < mean_reverting < trending < 1")
    
    def classify_single_value(hurst_value):
        if pd.isna(hurst_value):
            return 'unknown'
        elif hurst_value < mean_reverting_threshold:
            return 'mean-reverting'
        elif hurst_value > trending_threshold:
            return 'trending'
        else:
            return 'neutral'
    
    regime_series = hurst_series.apply(classify_single_value)
    regime_series.name = 'hurst_regime'
    return regime_series


# Export key components
__all__ = [
    'UnifiedRegimeClassifier',
    'REGIME_CLASSIFIER', 
    'get_regime_classifier',
    'classify_stock_regimes',
    'classify_hurst_regime'
]