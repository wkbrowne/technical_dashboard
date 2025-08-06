"""
Global Regime Configuration for All Models

This module provides centralized configuration for trend and volatility regimes
used across all models including:
- Markov models  
- Close price KDE models
- Open price KDE models
- High-low copula models

All regime thresholds and classifications should be defined here to ensure
consistency across the entire modeling system.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class TrendRegimeConfig:
    """Configuration for trend regime classification using integer states"""
    
    # Number of trend states (integer-based)
    n_states: int = 5
    
    # Trend thresholds based on daily MA slope (percentage change)
    # These define the boundaries between integer states
    thresholds: List[float] = None
    
    # Lookback window for trend classification (days)
    lookback_window: int = 5
    
    def __post_init__(self):
        if self.thresholds is None:
            # Default thresholds for 5-state system (from most bearish to most bullish)
            self.thresholds = [
                -0.003,  # Below this: trend_0 (strong_bear)
                -0.001,  # Below this: trend_1 (bear)
                0.001,   # Below this: trend_2 (sideways)
                0.003    # Below this: trend_3 (bull), above: trend_4 (strong_bull)
            ]
        
        # Validate thresholds length
        if len(self.thresholds) != self.n_states - 1:
            raise ValueError(f"Need {self.n_states - 1} thresholds for {self.n_states} states")
    
    def get_state_label(self, state: int) -> str:
        """Get descriptive label for integer state"""
        # Handle NaN values
        if pd.isna(state):
            return f'trend_{self.n_states // 2}'  # Fallback to middle state
        
        state = int(state)  # Ensure integer
        
        if self.n_states == 5:
            labels = ['strong_bear', 'bear', 'sideways', 'bull', 'strong_bull']
        elif self.n_states == 3:
            labels = ['bear', 'sideways', 'bull']
        elif self.n_states == 7:
            labels = ['very_strong_bear', 'strong_bear', 'bear', 'sideways', 'bull', 'strong_bull', 'very_strong_bull']
        else:
            # Generic labels for other n_states
            labels = [f'trend_{i}' for i in range(self.n_states)]
        
        if 0 <= state < len(labels):
            return labels[state]
        return f'trend_{state}'
    
    def get_all_states(self) -> List[int]:
        """Get all possible integer states"""
        return list(range(self.n_states))
    
    def get_all_labels(self) -> List[str]:
        """Get all descriptive labels"""
        return [self.get_state_label(i) for i in range(self.n_states)]


@dataclass 
class VolatilityRegimeConfig:
    """Configuration for volatility regime classification using integer states"""
    
    # Number of volatility states (integer-based)
    n_states: int = 3
    
    # Volatility percentile thresholds (boundaries between states)
    percentile_thresholds: List[float] = None
    
    # Lookback window for volatility calculation (days)
    lookback_window: int = 20
    
    # Minimum periods required for rolling calculations
    min_periods: int = 10
    
    def __post_init__(self):
        if self.percentile_thresholds is None:
            # Default percentile thresholds for 3-state system
            self.percentile_thresholds = [
                33.33,   # Below this percentile: vol_0 (low)
                66.67    # Below this percentile: vol_1 (medium), above: vol_2 (high)
            ]
        
        # Validate thresholds length
        if len(self.percentile_thresholds) != self.n_states - 1:
            raise ValueError(f"Need {self.n_states - 1} percentile thresholds for {self.n_states} states")
    
    def get_state_label(self, state: int) -> str:
        """Get descriptive label for integer state"""
        # Handle NaN values
        if pd.isna(state):
            return f'vol_{self.n_states // 2}'  # Fallback to middle state
        
        state = int(state)  # Ensure integer
        
        if self.n_states == 2:
            labels = ['low', 'high']
        elif self.n_states == 3:
            labels = ['low', 'medium', 'high']
        elif self.n_states == 5:
            labels = ['very_low', 'low', 'medium', 'high', 'very_high']
        else:
            # Generic labels for other n_states
            labels = [f'vol_{i}' for i in range(self.n_states)]
        
        if 0 <= state < len(labels):
            return labels[state]
        return f'vol_{state}'
    
    def get_all_states(self) -> List[int]:
        """Get all possible integer states"""
        return list(range(self.n_states))
    
    def get_all_labels(self) -> List[str]:
        """Get all descriptive labels"""
        return [self.get_state_label(i) for i in range(self.n_states)]


@dataclass
class GlobalRegimeConfig:
    """Master configuration for all regime classifications"""
    
    trend: TrendRegimeConfig = None
    volatility: VolatilityRegimeConfig = None
    
    # Combined regime separator
    regime_separator: str = '_'
    
    # Fallback regimes when classification fails (using middle states)
    @property
    def fallback_trend_state(self) -> int:
        return self.trend.n_states // 2
    
    @property 
    def fallback_volatility_state(self) -> int:
        return self.volatility.n_states // 2
    
    @property
    def fallback_trend(self) -> str:
        return self.trend.get_state_label(self.fallback_trend_state)
    
    @property
    def fallback_volatility(self) -> str:
        return self.volatility.get_state_label(self.fallback_volatility_state)
    
    def __post_init__(self):
        if self.trend is None:
            self.trend = TrendRegimeConfig()
        if self.volatility is None:
            self.volatility = VolatilityRegimeConfig()
    
    def get_all_combined_regimes(self) -> List[str]:
        """Get all possible combined regime combinations using integer states"""
        combined_regimes = []
        for trend_state in self.trend.get_all_states():
            for vol_state in self.volatility.get_all_states():
                trend_label = self.trend.get_state_label(trend_state)
                vol_label = self.volatility.get_state_label(vol_state)
                combined_regimes.append(f"{trend_label}{self.regime_separator}{vol_label}")
        return combined_regimes
    
    def get_all_combined_states(self) -> List[Tuple[int, int]]:
        """Get all possible combined regime states as integer tuples"""
        combined_states = []
        for trend_state in self.trend.get_all_states():
            for vol_state in self.volatility.get_all_states():
                combined_states.append((trend_state, vol_state))
        return combined_states
    
    def state_to_label(self, trend_state: int, vol_state: int) -> str:
        """Convert integer states to descriptive label"""
        trend_label = self.trend.get_state_label(trend_state)
        vol_label = self.volatility.get_state_label(vol_state)
        return f"{trend_label}{self.regime_separator}{vol_label}"
    
    def label_to_state(self, combined_label: str) -> Tuple[int, int]:
        """Convert descriptive label to integer states"""
        trend_label, vol_label = self.parse_combined_regime(combined_label)
        
        # Find trend state
        trend_state = None
        for i in self.trend.get_all_states():
            if self.trend.get_state_label(i) == trend_label:
                trend_state = i
                break
        
        # Find vol state
        vol_state = None
        for i in self.volatility.get_all_states():
            if self.volatility.get_state_label(i) == vol_label:
                vol_state = i
                break
        
        # Use fallback states if not found
        if trend_state is None:
            trend_state = self.trend.n_states // 2  # Middle state
        if vol_state is None:
            vol_state = self.volatility.n_states // 2  # Middle state
            
        return trend_state, vol_state
    
    def parse_combined_regime(self, combined_regime: str) -> Tuple[str, str]:
        """Parse combined regime into trend and volatility components"""
        if self.regime_separator not in combined_regime:
            return self.fallback_trend, self.fallback_volatility
        
        parts = combined_regime.split(self.regime_separator)
        
        # Handle cases like 'strong_bull_high' where trend has underscore
        if len(parts) == 3 and parts[0] == 'strong':
            trend_part = f"{parts[0]}_{parts[1]}"  # 'strong_bull'
            vol_part = parts[2]  # 'high'
        elif len(parts) == 2:
            trend_part, vol_part = parts
        else:
            # Fallback for unexpected formats
            trend_part = self.fallback_trend
            vol_part = self.fallback_volatility
        
        # Validate components against available labels
        trend_labels = self.trend.get_all_labels()
        vol_labels = self.volatility.get_all_labels()
        
        if trend_part not in trend_labels:
            trend_part = self.fallback_trend
        if vol_part not in vol_labels:
            vol_part = self.fallback_volatility
            
        return trend_part, vol_part
    
    def combine_regimes(self, trend_regime: str, vol_regime: str) -> str:
        """Combine trend and volatility regimes into single string"""
        return f"{trend_regime}{self.regime_separator}{vol_regime}"


# Global singleton instance - import this across all models
REGIME_CONFIG = GlobalRegimeConfig()


def get_regime_config() -> GlobalRegimeConfig:
    """Get the global regime configuration instance"""
    return REGIME_CONFIG


def update_trend_thresholds(new_thresholds: List[float]) -> None:
    """Update trend regime thresholds globally"""
    global REGIME_CONFIG
    if len(new_thresholds) != REGIME_CONFIG.trend.n_states - 1:
        raise ValueError(f"Need {REGIME_CONFIG.trend.n_states - 1} thresholds for {REGIME_CONFIG.trend.n_states} states")
    REGIME_CONFIG.trend.thresholds = new_thresholds.copy()


def update_volatility_thresholds(new_thresholds: List[float]) -> None:
    """Update volatility regime thresholds globally"""
    global REGIME_CONFIG  
    if len(new_thresholds) != REGIME_CONFIG.volatility.n_states - 1:
        raise ValueError(f"Need {REGIME_CONFIG.volatility.n_states - 1} thresholds for {REGIME_CONFIG.volatility.n_states} states")
    REGIME_CONFIG.volatility.percentile_thresholds = new_thresholds.copy()


def set_custom_regime_config(config: GlobalRegimeConfig) -> None:
    """Set a completely custom regime configuration"""
    global REGIME_CONFIG
    REGIME_CONFIG = config


# Convenience functions for common operations
def get_trend_thresholds() -> List[float]:
    """Get current trend thresholds"""
    return REGIME_CONFIG.trend.thresholds.copy()


def get_volatility_thresholds() -> List[float]:
    """Get current volatility thresholds"""
    return REGIME_CONFIG.volatility.percentile_thresholds.copy()


def get_all_trend_regimes() -> List[str]:
    """Get all trend regime labels"""
    return REGIME_CONFIG.trend.get_all_labels()


def get_all_volatility_regimes() -> List[str]:
    """Get all volatility regime labels"""
    return REGIME_CONFIG.volatility.get_all_labels()


def get_all_trend_states() -> List[int]:
    """Get all trend regime integer states"""
    return REGIME_CONFIG.trend.get_all_states()


def get_all_volatility_states() -> List[int]:
    """Get all volatility regime integer states"""
    return REGIME_CONFIG.volatility.get_all_states()


def create_regime_config(n_trend_states: int, n_vol_states: int) -> GlobalRegimeConfig:
    """Create a new regime configuration with specified number of states"""
    # Create appropriate thresholds for the number of states
    trend_thresholds = None
    if n_trend_states == 3:
        trend_thresholds = [-0.001, 0.001]  # bear, sideways, bull
    elif n_trend_states == 5:
        trend_thresholds = [-0.003, -0.001, 0.001, 0.003]  # Default 5-state
    elif n_trend_states == 7:
        trend_thresholds = [-0.005, -0.003, -0.001, 0.001, 0.003, 0.005]  # 7-state
    # Add more as needed
    
    vol_thresholds = None
    if n_vol_states == 2:
        vol_thresholds = [50.0]  # low, high
    elif n_vol_states == 3:
        vol_thresholds = [33.33, 66.67]  # Default 3-state
    elif n_vol_states == 5:
        vol_thresholds = [20.0, 40.0, 60.0, 80.0]  # 5-state
    # Add more as needed
    
    trend_config = TrendRegimeConfig(n_states=n_trend_states, thresholds=trend_thresholds)
    vol_config = VolatilityRegimeConfig(n_states=n_vol_states, percentile_thresholds=vol_thresholds)
    return GlobalRegimeConfig(trend=trend_config, volatility=vol_config)


def get_all_combined_regimes() -> List[str]:
    """Get all possible combined regime names"""
    return REGIME_CONFIG.get_all_combined_regimes()


# Model-specific regime configuration helpers
class ModelRegimeConfig:
    """Helper class for models to use regime configuration"""
    
    @staticmethod
    def get_trend_lookback() -> int:
        """Get trend classification lookback window"""
        return REGIME_CONFIG.trend.lookback_window
    
    @staticmethod
    def get_volatility_lookback() -> int:
        """Get volatility classification lookback window"""
        return REGIME_CONFIG.volatility.lookback_window
    
    @staticmethod
    def get_volatility_min_periods() -> int:
        """Get minimum periods for volatility calculation"""
        return REGIME_CONFIG.volatility.min_periods
    
    @staticmethod
    def get_fallback_regime() -> str:
        """Get fallback combined regime"""
        return REGIME_CONFIG.combine_regimes(
            REGIME_CONFIG.fallback_trend,
            REGIME_CONFIG.fallback_volatility
        )


# Export key components for easy importing
__all__ = [
    'REGIME_CONFIG',
    'GlobalRegimeConfig', 
    'TrendRegimeConfig',
    'VolatilityRegimeConfig',
    'get_regime_config',
    'update_trend_thresholds',
    'update_volatility_thresholds',
    'set_custom_regime_config',
    'get_trend_thresholds',
    'get_volatility_thresholds', 
    'get_all_trend_regimes',
    'get_all_volatility_regimes',
    'get_all_trend_states',
    'get_all_volatility_states',
    'get_all_combined_regimes',
    'create_regime_config',
    'ModelRegimeConfig'
]