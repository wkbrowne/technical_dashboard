"""
Weekly features computation for daily stock data.

This module resamples daily OHLCV data to weekly timeframe and computes comprehensive
technical indicators including RSI, MACD, moving averages, trend features, distance features,
and all relative strength metrics. Features are merged back to daily data in a leakage-safe manner.
"""
import logging
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
from joblib import Parallel, delayed

# Import existing feature functions
from .trend import add_trend_features, add_rsi_features, add_macd_features
from .distance import add_distance_to_ma_features
from .relstrength import _compute_relative_strength_block

logger = logging.getLogger(__name__)


def _resample_to_weekly(df_symbol: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Resample a single symbol's daily data to weekly (W-FRI) bars.
    
    Args:
        df_symbol: Daily OHLCV data for one symbol
        symbol: Symbol name for logging
        
    Returns:
        Weekly resampled DataFrame with week_end column
    """
    if df_symbol.empty or 'date' not in df_symbol.columns:
        logger.warning(f"Empty or invalid data for {symbol}")
        return pd.DataFrame()
    
    # Set date as index for resampling
    df_weekly = df_symbol.set_index('date').copy()
    
    # Define aggregation rules for OHLCV
    agg_rules = {
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Add any other numeric columns with 'last' aggregation
    for col in df_weekly.columns:
        if col not in agg_rules and pd.api.types.is_numeric_dtype(df_weekly[col]):
            agg_rules[col] = 'last'
    
    try:
        # Resample to weekly Friday bars
        weekly_data = df_weekly.resample('W-FRI').agg(agg_rules)
        
        # Remove rows with no data
        weekly_data = weekly_data.dropna(subset=['close'])
        
        if weekly_data.empty:
            logger.warning(f"No weekly data generated for {symbol}")
            return pd.DataFrame()
        
        # Add week_end column and symbol
        weekly_data['week_end'] = weekly_data.index
        weekly_data['symbol'] = symbol
        
        # Calculate weekly returns
        weekly_data['w_ret'] = np.log(pd.to_numeric(weekly_data['close'], errors='coerce')).diff()
        
        # Reset index for easier handling
        weekly_data = weekly_data.reset_index(drop=True)
        
        logger.debug(f"Resampled {symbol} to {len(weekly_data)} weekly bars")
        return weekly_data
        
    except Exception as e:
        logger.error(f"Error resampling {symbol} to weekly: {e}")
        return pd.DataFrame()


def _compute_weekly_features_single_symbol(weekly_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compute all weekly technical features for a single symbol.
    
    Args:
        weekly_df: Weekly OHLCV data for one symbol
        symbol: Symbol name
        
    Returns:
        Weekly DataFrame with all computed features
    """
    if weekly_df.empty:
        return pd.DataFrame()
    
    try:
        # Make a copy to avoid modifying original
        df = weekly_df.copy()
        
        # 1) Basic RSI features (weekly)
        df = add_rsi_features(
            df,
            src_col='close',
            periods=(14, 21, 30)
        )
        
        # 2) MACD features (weekly)
        df = add_macd_features(
            df,
            src_col='close',
            fast=12,
            slow=26,
            signal=9,
            derivative_ema_span=3
        )
        
        # 3) Trend features (weekly MAs and slopes)
        df = add_trend_features(
            df,
            src_col='close',
            ma_periods=(20, 50, 100, 200),  # Weekly MAs
            slope_window=4,  # 4-week slope window
            eps=1e-5
        )
        
        # 4) Distance to MA features (weekly)
        df = add_distance_to_ma_features(
            df,
            src_col='close',
            ma_lengths=(20, 50, 100, 200),
            z_window=12  # 12-week z-score window
        )
        
        # 5) Additional weekly-specific moving averages using pandas-ta
        close_series = pd.to_numeric(df['close'], errors='coerce')
        
        for period in [20, 50, 100, 200]:
            try:
                sma_values = ta.sma(close_series, length=period)
                if sma_values is not None:
                    df[f'sma{period}'] = sma_values.astype('float32')
                else:
                    # Fallback to pandas rolling mean
                    df[f'sma{period}'] = close_series.rolling(window=period, min_periods=period//2).mean().astype('float32')
            except Exception as e:
                logger.warning(f"Error computing SMA{period} for {symbol}: {e}")
                # Fallback to pandas rolling mean
                df[f'sma{period}'] = close_series.rolling(window=period, min_periods=period//2).mean().astype('float32')
        
        logger.debug(f"Computed weekly features for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error computing weekly features for {symbol}: {e}")
        return weekly_df  # Return original if computation fails


def _compute_weekly_relative_strength(weekly_data: Dict[str, pd.DataFrame], 
                                    enhanced_mappings: Optional[Dict[str, Dict]] = None,
                                    spy_symbol: str = "SPY") -> Dict[str, pd.DataFrame]:
    """
    Compute comprehensive weekly relative strength features for all symbols.
    
    Args:
        weekly_data: Dictionary of {symbol: weekly_df}
        enhanced_mappings: Enhanced sector/subsector mappings
        spy_symbol: Market benchmark symbol
        
    Returns:
        Updated weekly_data with relative strength features
    """
    logger.info("Computing weekly relative strength features...")
    
    # Get benchmark data
    spy_weekly = weekly_data.get(spy_symbol)
    rsp_weekly = weekly_data.get('RSP')  # Equal-weight market
    
    if spy_weekly is None:
        logger.warning(f"No weekly data for {spy_symbol}, skipping market relative strength")
        return weekly_data
    
    spy_close = pd.to_numeric(spy_weekly['close'], errors='coerce')
    rsp_close = pd.to_numeric(rsp_weekly['close'], errors='coerce') if rsp_weekly is not None else None
    
    # Create date index for alignment
    spy_dates = spy_weekly['week_end']
    
    for symbol, df in weekly_data.items():
        if df.empty or symbol == spy_symbol:
            continue
        
        try:
            stock_close = pd.to_numeric(df['close'], errors='coerce')
            stock_dates = df['week_end']
            
            # Align dates between stock and benchmarks
            common_dates = stock_dates[stock_dates.isin(spy_dates)]
            if len(common_dates) < 10:  # Need minimum data
                continue
            
            stock_aligned = stock_close[stock_dates.isin(common_dates)]
            spy_aligned = spy_close[spy_dates.isin(common_dates)]
            
            # Market relative strength (vs SPY)
            if len(stock_aligned) == len(spy_aligned):
                rs_spy, rs_spy_norm, rs_spy_slope = _compute_relative_strength_block(
                    stock_aligned, spy_aligned, look=12, slope_win=4  # Weekly parameters
                )
                
                # Map back to original dataframe
                mask = stock_dates.isin(common_dates)
                df.loc[mask, 'w_rel_strength_spy'] = rs_spy.astype('float32')
                df.loc[mask, 'w_rel_strength_spy_norm'] = rs_spy_norm.astype('float32')
                df.loc[mask, 'w_rel_strength_spy_slope4'] = rs_spy_slope.astype('float32')
            
            # Equal-weight market relative strength (vs RSP)
            if rsp_close is not None:
                rsp_aligned = rsp_close[spy_dates.isin(common_dates)]
                if len(stock_aligned) == len(rsp_aligned):
                    rs_rsp, rs_rsp_norm, rs_rsp_slope = _compute_relative_strength_block(
                        stock_aligned, rsp_aligned, look=12, slope_win=4
                    )
                    
                    df.loc[mask, 'w_rel_strength_rsp'] = rs_rsp.astype('float32')
                    df.loc[mask, 'w_rel_strength_rsp_norm'] = rs_rsp_norm.astype('float32')
                    df.loc[mask, 'w_rel_strength_rsp_slope4'] = rs_rsp_slope.astype('float32')
            
            # Sector and subsector relative strength (if mappings provided)
            if enhanced_mappings and symbol in enhanced_mappings:
                mapping = enhanced_mappings[symbol]
                
                # Sector ETF relative strength
                sector_etf = mapping.get('sector_etf')
                if sector_etf and sector_etf in weekly_data:
                    sector_weekly = weekly_data[sector_etf]
                    sector_close = pd.to_numeric(sector_weekly['close'], errors='coerce')
                    sector_dates = sector_weekly['week_end']
                    
                    sector_aligned = sector_close[sector_dates.isin(common_dates)]
                    if len(stock_aligned) == len(sector_aligned):
                        rs_sect, rs_sect_norm, rs_sect_slope = _compute_relative_strength_block(
                            stock_aligned, sector_aligned, look=12, slope_win=4
                        )
                        
                        df.loc[mask, 'w_rel_strength_sector'] = rs_sect.astype('float32')
                        df.loc[mask, 'w_rel_strength_sector_norm'] = rs_sect_norm.astype('float32')
                        df.loc[mask, 'w_rel_strength_sector_slope4'] = rs_sect_slope.astype('float32')
                
                # Equal-weight sector ETF relative strength
                ew_etf = mapping.get('equal_weight_etf')
                if ew_etf and ew_etf in weekly_data:
                    ew_weekly = weekly_data[ew_etf]
                    ew_close = pd.to_numeric(ew_weekly['close'], errors='coerce')
                    ew_dates = ew_weekly['week_end']
                    
                    ew_aligned = ew_close[ew_dates.isin(common_dates)]
                    if len(stock_aligned) == len(ew_aligned):
                        rs_ew, rs_ew_norm, rs_ew_slope = _compute_relative_strength_block(
                            stock_aligned, ew_aligned, look=12, slope_win=4
                        )
                        
                        df.loc[mask, 'w_rel_strength_sector_ew'] = rs_ew.astype('float32')
                        df.loc[mask, 'w_rel_strength_sector_ew_norm'] = rs_ew_norm.astype('float32')
                        df.loc[mask, 'w_rel_strength_sector_ew_slope4'] = rs_ew_slope.astype('float32')
                
                # Subsector ETF relative strength
                subsector_etf = mapping.get('subsector_etf')
                if subsector_etf and subsector_etf in weekly_data:
                    sub_weekly = weekly_data[subsector_etf]
                    sub_close = pd.to_numeric(sub_weekly['close'], errors='coerce')
                    sub_dates = sub_weekly['week_end']
                    
                    sub_aligned = sub_close[sub_dates.isin(common_dates)]
                    if len(stock_aligned) == len(sub_aligned):
                        rs_sub, rs_sub_norm, rs_sub_slope = _compute_relative_strength_block(
                            stock_aligned, sub_aligned, look=12, slope_win=4
                        )
                        
                        df.loc[mask, 'w_rel_strength_subsector'] = rs_sub.astype('float32')
                        df.loc[mask, 'w_rel_strength_subsector_norm'] = rs_sub_norm.astype('float32')
                        df.loc[mask, 'w_rel_strength_subsector_slope4'] = rs_sub_slope.astype('float32')
        
        except Exception as e:
            logger.warning(f"Error computing weekly relative strength for {symbol}: {e}")
            continue
    
    logger.info("Weekly relative strength computation completed")
    return weekly_data


def _add_weekly_prefix_and_convert_types(weekly_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Add 'w_' prefix to feature columns and convert to float32.
    
    Args:
        weekly_data: Dictionary of weekly DataFrames
        
    Returns:
        Updated weekly_data with prefixed columns
    """
    # Columns to exclude from prefixing
    exclude_cols = {'symbol', 'week_end', 'date', 'open', 'high', 'low', 'close', 'volume'}
    
    for symbol, df in weekly_data.items():
        if df.empty:
            continue
        
        # Get columns to prefix
        cols_to_prefix = [col for col in df.columns if col not in exclude_cols]
        
        # Add prefix and convert types
        for col in cols_to_prefix:
            new_col = f'w_{col}'
            if pd.api.types.is_numeric_dtype(df[col]):
                df[new_col] = df[col].astype('float32')
            else:
                df[new_col] = df[col]
            # Remove original column
            df.drop(columns=[col], inplace=True)
    
    return weekly_data


def add_weekly_features_to_daily(df_long: pd.DataFrame, 
                               spy_symbol: str = "SPY", 
                               n_jobs: int = -1,
                               enhanced_mappings: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Add comprehensive weekly features to daily long-format stock dataframe.
    
    This function resamples daily OHLCV data to weekly timeframe, computes comprehensive
    technical indicators, and merges them back to daily data in a leakage-safe manner.
    
    Args:
        df_long: Daily long-format DataFrame with columns ['symbol','date','open','high','low','close','volume']
        spy_symbol: Market benchmark symbol (default: "SPY")
        n_jobs: Number of parallel jobs for processing (-1 = all cores)
        enhanced_mappings: Optional enhanced sector/subsector mappings for relative strength
        
    Returns:
        Daily DataFrame with added weekly features (prefixed with 'w_')
        
    Features added (~51 per symbol):
    - Basic: RSI, MACD, SMA indicators
    - Trend: MA slopes, trend scores, alignment
    - Distance: Distance to MA, z-scores  
    - Relative Strength: Market, sector, subsector comparisons
    """
    logger.info(f"Adding weekly features to {len(df_long)} daily rows across {df_long['symbol'].nunique()} symbols")
    
    # Step 1: Data validation
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df_long.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure date is datetime
    df_long['date'] = pd.to_datetime(df_long['date'])
    
    # Sort by symbol and date
    df_long = df_long.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    logger.info("Step 1: Data validation completed")
    
    # Step 2: Weekly resampling (parallelized by symbol)
    unique_symbols = df_long['symbol'].unique()
    logger.info(f"Step 2: Resampling {len(unique_symbols)} symbols to weekly timeframe")
    
    # Create symbol groups for parallel processing
    symbol_groups = [(symbol, df_long[df_long['symbol'] == symbol]) for symbol in unique_symbols]
    
    # Parallel weekly resampling
    weekly_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(
        delayed(_resample_to_weekly)(group_df, symbol) 
        for symbol, group_df in symbol_groups
    )
    
    # Convert results to dictionary
    weekly_data = {}
    for result in weekly_results:
        if not result.empty and 'symbol' in result.columns:
            symbol = result['symbol'].iloc[0]
            weekly_data[symbol] = result
    
    logger.info(f"Step 2: Resampled to {len(weekly_data)} weekly symbol datasets")
    
    # Step 3: Compute weekly technical features (parallelized)
    logger.info("Step 3: Computing weekly technical features")
    
    weekly_feature_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(
        delayed(_compute_weekly_features_single_symbol)(df, symbol)
        for symbol, df in weekly_data.items()
    )
    
    # Update weekly_data with computed features
    for i, (symbol, _) in enumerate(weekly_data.items()):
        if not weekly_feature_results[i].empty:
            weekly_data[symbol] = weekly_feature_results[i]
    
    logger.info("Step 3: Weekly technical features completed")
    
    # Step 4: Compute weekly relative strength features
    logger.info("Step 4: Computing weekly relative strength features")
    weekly_data = _compute_weekly_relative_strength(weekly_data, enhanced_mappings, spy_symbol)
    
    # Step 5: Add prefixes and convert types
    logger.info("Step 5: Adding prefixes and converting data types")
    weekly_data = _add_weekly_prefix_and_convert_types(weekly_data)
    
    # Step 6: Leakage-safe merge back to daily data
    logger.info("Step 6: Performing leakage-safe merge to daily data")
    
    # Combine all weekly data
    weekly_features_list = []
    for symbol, df in weekly_data.items():
        if not df.empty:
            # Keep only feature columns + merge keys
            feature_cols = [col for col in df.columns if col.startswith('w_')] + ['symbol', 'week_end']
            if len(feature_cols) > 2:  # More than just symbol and week_end
                weekly_features_list.append(df[feature_cols])
    
    if not weekly_features_list:
        logger.warning("No weekly features computed, returning original dataframe")
        return df_long
    
    weekly_features_combined = pd.concat(weekly_features_list, ignore_index=True)
    
    # Sort for merge_asof - MUST be sorted by merge keys within each group
    df_daily_sorted = df_long.sort_values(['symbol', 'date']).reset_index(drop=True)
    weekly_features_sorted = weekly_features_combined.sort_values(['symbol', 'week_end']).reset_index(drop=True)
    
    # Perform leakage-safe merge using merge_asof by symbol groups to ensure proper sorting
    merged_results = []
    for symbol in df_daily_sorted['symbol'].unique():
        daily_symbol = df_daily_sorted[df_daily_sorted['symbol'] == symbol].sort_values('date')
        weekly_symbol = weekly_features_sorted[weekly_features_sorted['symbol'] == symbol].sort_values('week_end')
        
        if not weekly_symbol.empty:
            merged_symbol = pd.merge_asof(
                daily_symbol,
                weekly_symbol.drop(columns=['symbol']),  # Remove duplicate symbol column
                left_on='date',
                right_on='week_end',
                direction='backward'
            )
            merged_results.append(merged_symbol)
        else:
            # If no weekly features for this symbol, keep original daily data
            merged_results.append(daily_symbol)
    
    df_with_weekly = pd.concat(merged_results, ignore_index=True)
    
    # Drop temporary merge column
    if 'week_end' in df_with_weekly.columns:
        df_with_weekly.drop(columns=['week_end'], inplace=True)
    
    logger.info(f"Step 6: Added {len([col for col in df_with_weekly.columns if col.startswith('w_')])} weekly features")
    
    # Step 7: Final cleanup and validation
    logger.info("Step 7: Final cleanup")
    
    # Ensure original order is maintained
    final_df = df_with_weekly.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    weekly_feature_count = len([col for col in final_df.columns if col.startswith('w_')])
    logger.info(f"Successfully added {weekly_feature_count} weekly features to daily dataframe")
    
    return final_df