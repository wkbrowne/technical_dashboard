import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional

def calculate_bollinger_bands(prices: pd.Series, 
                             window: int = 20, 
                             num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically close prices)
    window : int
        Moving average window (default 20)
    num_std : float
        Number of standard deviations for bands (default 2.0)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: MA, Upper_Band, Lower_Band, BB_Width, BB_Position
    """
    # Calculate moving average
    ma = prices.rolling(window=window).mean()
    
    # Calculate rolling standard deviation
    std = prices.rolling(window=window).std()
    
    # Calculate bands
    upper_band = ma + (num_std * std)
    lower_band = ma - (num_std * std)
    
    # Calculate band width (volatility measure)
    bb_width = (upper_band - lower_band) / ma
    
    # Calculate position within bands (-1 to 1, 0 = at MA)
    bb_position = (prices - ma) / (upper_band - ma)
    bb_position = np.clip(bb_position, -1, 1)
    
    # Create DataFrame
    bb_df = pd.DataFrame({
        'Price': prices,
        'MA': ma,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band,
        'BB_Width': bb_width,
        'BB_Position': bb_position
    }, index=prices.index)
    
    return bb_df

def calculate_bb_squeeze(bb_df: pd.DataFrame, 
                        squeeze_threshold: float = 0.1,
                        window: int = 20) -> pd.Series:
    """
    Identify Bollinger Band squeeze periods (low volatility).
    
    Parameters
    ----------
    bb_df : pd.DataFrame
        Bollinger Bands DataFrame from calculate_bollinger_bands
    squeeze_threshold : float
        BB_Width threshold for squeeze identification
    window : int
        Lookback window for squeeze comparison
        
    Returns
    -------
    pd.Series
        Boolean series indicating squeeze periods
    """
    # Calculate if current BB width is at lowest in lookback window
    bb_width = bb_df['BB_Width']
    rolling_min = bb_width.rolling(window=window).min()
    
    # Squeeze when current width equals rolling minimum and below threshold
    squeeze = (bb_width == rolling_min) & (bb_width < squeeze_threshold)
    
    return squeeze

def calculate_bb_breakout(bb_df: pd.DataFrame, 
                         breakout_threshold: float = 0.8) -> pd.DataFrame:
    """
    Identify Bollinger Band breakouts.
    
    Parameters
    ----------
    bb_df : pd.DataFrame
        Bollinger Bands DataFrame
    breakout_threshold : float
        BB_Position threshold for breakout (default 0.8 = 80% to band)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with breakout signals
    """
    bb_position = bb_df['BB_Position']
    
    # Breakout signals
    upper_breakout = bb_position > breakout_threshold
    lower_breakout = bb_position < -breakout_threshold
    
    # Mean reversion signals (opposite direction)
    upper_reversion = (bb_position.shift(1) > breakout_threshold) & (bb_position <= breakout_threshold)
    lower_reversion = (bb_position.shift(1) < -breakout_threshold) & (bb_position >= -breakout_threshold)
    
    breakout_df = pd.DataFrame({
        'Upper_Breakout': upper_breakout,
        'Lower_Breakout': lower_breakout,
        'Upper_Reversion': upper_reversion,
        'Lower_Reversion': lower_reversion
    }, index=bb_df.index)
    
    return breakout_df

def plot_bollinger_bands(bb_df: pd.DataFrame, 
                        title: str = "Bollinger Bands Analysis",
                        figsize: Tuple[int, int] = (15, 10),
                        show_position: bool = True) -> None:
    """
    Plot Bollinger Bands with price data.
    
    Parameters
    ----------
    bb_df : pd.DataFrame
        Bollinger Bands DataFrame
    title : str
        Plot title
    figsize : tuple
        Figure size
    show_position : bool
        Whether to show BB position subplot
    """
    if show_position:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 1])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Main price and bands plot
    ax1.plot(bb_df.index, bb_df['Price'], label='Price', linewidth=2, color='black')
    ax1.plot(bb_df.index, bb_df['MA'], label='20-day MA', linewidth=2, color='blue')
    ax1.fill_between(bb_df.index, bb_df['Upper_Band'], bb_df['Lower_Band'], 
                     alpha=0.2, color='gray', label='Bollinger Bands')
    ax1.plot(bb_df.index, bb_df['Upper_Band'], linewidth=1, color='red', linestyle='--')
    ax1.plot(bb_df.index, bb_df['Lower_Band'], linewidth=1, color='red', linestyle='--')
    
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if show_position:
        # BB Position plot
        ax2.plot(bb_df.index, bb_df['BB_Position'], linewidth=2, color='purple')
        ax2.axhline(0, color='blue', linestyle='-', alpha=0.7)
        ax2.axhline(0.8, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(-0.8, color='red', linestyle='--', alpha=0.7)
        ax2.fill_between(bb_df.index, 0.8, 1, alpha=0.2, color='red', label='Overbought')
        ax2.fill_between(bb_df.index, -0.8, -1, alpha=0.2, color='green', label='Oversold')
        ax2.set_ylabel('BB Position', fontsize=12)
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # BB Width plot
        ax3.plot(bb_df.index, bb_df['BB_Width'], linewidth=2, color='orange')
        ax3.axhline(bb_df['BB_Width'].median(), color='gray', linestyle='--', alpha=0.7, label='Median Width')
        ax3.set_ylabel('BB Width', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def get_bb_regime(bb_df: pd.DataFrame) -> pd.Series:
    """
    Classify market regime based on Bollinger Band position and width.
    
    Parameters
    ----------
    bb_df : pd.DataFrame
        Bollinger Bands DataFrame
        
    Returns
    -------
    pd.Series
        Regime classification: 'Squeeze', 'Trending_Up', 'Trending_Down', 'Ranging'
    """
    bb_position = bb_df['BB_Position']
    bb_width = bb_df['BB_Width']
    
    # Define thresholds
    width_threshold = bb_width.quantile(0.3)  # Bottom 30% for squeeze
    position_high = 0.5
    position_low = -0.5
    
    # Classify regimes
    regime = pd.Series('Ranging', index=bb_df.index)
    
    # Squeeze regime (low volatility)
    regime[bb_width < width_threshold] = 'Squeeze'
    
    # Trending regimes (high BB position)
    trending_mask = bb_width >= width_threshold
    regime[(bb_position > position_high) & trending_mask] = 'Trending_Up'
    regime[(bb_position < position_low) & trending_mask] = 'Trending_Down'
    
    return regime

def calculate_bb_statistics(bb_df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for Bollinger Bands analysis.
    
    Parameters
    ----------
    bb_df : pd.DataFrame
        Bollinger Bands DataFrame
        
    Returns
    -------
    dict
        Dictionary with summary statistics
    """
    bb_position = bb_df['BB_Position'].dropna()
    bb_width = bb_df['BB_Width'].dropna()
    
    # Position statistics
    position_stats = {
        'mean_position': bb_position.mean(),
        'std_position': bb_position.std(),
        'time_above_ma': (bb_position > 0).mean(),
        'time_in_upper_band': (bb_position > 0.8).mean(),
        'time_in_lower_band': (bb_position < -0.8).mean(),
    }
    
    # Width statistics
    width_stats = {
        'mean_width': bb_width.mean(),
        'median_width': bb_width.median(),
        'width_volatility': bb_width.std(),
        'squeeze_frequency': (bb_width < bb_width.quantile(0.2)).mean(),
    }
    
    # Regime analysis
    regime = get_bb_regime(bb_df)
    regime_stats = regime.value_counts(normalize=True).to_dict()
    regime_stats = {f'regime_{k.lower()}': v for k, v in regime_stats.items()}
    
    return {
        **position_stats,
        **width_stats,
        **regime_stats
    }