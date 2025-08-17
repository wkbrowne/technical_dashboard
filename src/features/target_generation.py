"""
Triple barrier target generation for financial time series.

This module generates triple barrier targets for stock trajectories using configurable
barriers based on ATR (Average True Range). Each target represents a potential trade
with upper/lower barriers and time horizon constraints.
"""
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_triple_barrier_targets(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Generate triple barrier targets from long-format price data.
    
    Creates targets by setting upper and lower barriers around entry prices
    and tracking which barrier (if any) is hit within the maximum horizon.
    
    Args:
        df: Long-format DataFrame with columns: symbol, date, close, high, low, atr
        config: Configuration dictionary with keys:
            - up_mult: Upper barrier multiplier (e.g., 3.0 = 3x ATR above entry)
            - dn_mult: Lower barrier multiplier (e.g., 3.0 = 3x ATR below entry)  
            - max_horizon: Maximum days to track each target
            - start_every: Minimum days between new targets (when not hit early)
            
    Returns:
        DataFrame with columns: symbol, t0, t_hit, hit, entry_px, top, bot, 
                               h_used, price_hit, ret_from_entry
        
    Notes:
        - hit: 1 = upper barrier hit, -1 = lower barrier hit, 0 = time expired
        - ret_from_entry: Log return from entry to exit price
        - Assumes df is sorted by symbol and date
    """
    required_cols = {'symbol', 'date', 'close', 'high', 'low', 'atr'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate config
    required_config = {'up_mult', 'dn_mult', 'max_horizon', 'start_every'}
    missing_config = required_config - set(config.keys())
    if missing_config:
        raise ValueError(f"Missing required config keys: {missing_config}")
    
    logger.debug(f"Generating targets with config: {config}")
    
    # Process each symbol independently
    results = []
    symbols = df['symbol'].unique()
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        if len(symbol_df) < config['max_horizon'] + 2:
            logger.debug(f"Skipping {symbol}: insufficient data ({len(symbol_df)} rows)")
            continue
            
        symbol_targets = _compute_triple_barriers_numpy(symbol, symbol_df, config)
        if not symbol_targets.empty:
            results.append(symbol_targets)
    
    if not results:
        logger.warning("No targets generated for any symbols")
        return pd.DataFrame()
    
    targets_df = pd.concat(results, ignore_index=True)
    logger.info(f"Generated {len(targets_df)} targets across {len(results)} symbols")
    
    return targets_df


def _compute_triple_barriers_numpy(symbol: str, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compute triple barrier targets for a single symbol using NumPy for speed.
    
    Args:
        symbol: Symbol identifier
        df: DataFrame for single symbol, sorted by date
        config: Configuration dictionary
        
    Returns:
        DataFrame with target information for this symbol
    """
    # Sort and reset index to ensure consistent positioning
    df = df.sort_values('date').reset_index(drop=True)
    
    # Convert to NumPy arrays for speed
    dates = df['date'].to_numpy('datetime64[ns]')
    close = df['close'].to_numpy(dtype=float)
    atr = df['atr'].to_numpy(dtype=float)
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    
    n = len(df)
    pos = 0
    targets = []
    
    up_mult = config['up_mult']
    dn_mult = config['dn_mult']
    max_horizon = config['max_horizon']
    start_every = config['start_every']
    
    while pos < n - max_horizon - 1:
        c0, a0 = close[pos], atr[pos]
        
        # Skip if entry price or ATR is invalid
        if not (np.isfinite(c0) and np.isfinite(a0) and a0 > 0):
            pos += 1
            continue
        
        # Define barriers
        top_barrier = c0 + up_mult * a0
        bot_barrier = c0 - dn_mult * a0
        
        # Define analysis window (next day through max horizon)
        start_idx = pos + 1
        end_idx = start_idx + max_horizon
        
        if end_idx > n:
            break  # Not enough data for full horizon
        
        # Slice arrays for the analysis window
        h_slice = high[start_idx:end_idx]
        l_slice = low[start_idx:end_idx]
        c_slice = close[start_idx:end_idx]
        
        # Find barrier hits
        hit_top_indices = np.where(h_slice >= top_barrier)[0]
        hit_bot_indices = np.where(l_slice <= bot_barrier)[0]
        
        # Determine which barrier was hit first (if any)
        hit_type = 0  # 0 = time expired
        hit_idx = max_horizon - 1  # Default to last day
        price_hit = c_slice[hit_idx] if len(c_slice) > hit_idx else np.nan
        
        if len(hit_top_indices) > 0 and (len(hit_bot_indices) == 0 or hit_top_indices[0] <= hit_bot_indices[0]):
            # Upper barrier hit first
            hit_type = 1
            hit_idx = hit_top_indices[0]
            price_hit = top_barrier
        elif len(hit_bot_indices) > 0:
            # Lower barrier hit first  
            hit_type = -1
            hit_idx = hit_bot_indices[0]
            price_hit = bot_barrier
        
        # Calculate metrics
        horizon_used = hit_idx + 1
        t_hit = dates[start_idx + hit_idx]
        ret_from_entry = np.log(price_hit / c0) if price_hit > 0 and c0 > 0 else np.nan
        
        targets.append({
            'symbol': symbol,
            't0': dates[pos],
            't_hit': t_hit,
            'hit': hit_type,
            'entry_px': c0,
            'top': top_barrier,
            'bot': bot_barrier,
            'h_used': horizon_used,
            'price_hit': price_hit,
            'ret_from_entry': ret_from_entry
        })
        
        # Advance position: if barrier hit early, use that horizon, otherwise use start_every
        if hit_type != 0 and horizon_used < start_every:
            pos += horizon_used
        else:
            pos += start_every
    
    return pd.DataFrame(targets)


def get_target_summary(targets_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for triple barrier targets.
    
    Args:
        targets_df: Output from generate_triple_barrier_targets()
        
    Returns:
        Dictionary with summary statistics
    """
    if targets_df.empty:
        return {'total_targets': 0}
    
    hit_counts = targets_df['hit'].value_counts()
    
    return {
        'total_targets': len(targets_df),
        'symbols': targets_df['symbol'].nunique(),
        'hit_upper': hit_counts.get(1, 0),
        'hit_lower': hit_counts.get(-1, 0), 
        'hit_time': hit_counts.get(0, 0),
        'avg_horizon_used': targets_df['h_used'].mean(),
        'avg_return': targets_df['ret_from_entry'].mean(),
        'return_std': targets_df['ret_from_entry'].std(),
        'date_range': (targets_df['t0'].min(), targets_df['t0'].max())
    }