"""
Sector ETF Breadth Proxy - Market breadth features using the 11 Select Sector SPDR ETFs.

This module provides a survivorship-bias-free alternative to traditional NYSE advance/decline
data, which is not reliably available via free APIs. Instead, we compute breadth indicators
from the fixed set of 11 sector ETFs that together cover the entire S&P 500.

Features are computed globally (same value for all symbols on a given date) and then
broadcast to all symbols.

Replaces the prior breadth.py logic that depended on external advance/decline data sources.
"""
import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 11 Select Sector SPDR ETFs covering the S&P 500
SECTOR_ETFS = [
    'XLB',   # Materials
    'XLC',   # Communication Services
    'XLE',   # Energy
    'XLF',   # Financials
    'XLI',   # Industrials
    'XLK',   # Technology
    'XLP',   # Consumer Staples
    'XLRE',  # Real Estate
    'XLU',   # Utilities
    'XLV',   # Health Care
    'XLY',   # Consumer Discretionary
]

N_SECTORS = len(SECTOR_ETFS)  # 11


def _load_sector_etf_prices(
    etf_data: Dict[str, pd.DataFrame],
    price_col: str = 'adjclose'
) -> Optional[pd.DataFrame]:
    """
    Extract price series for the 11 sector ETFs from ETF data dictionary.

    Args:
        etf_data: Dictionary mapping ETF symbol to DataFrame with OHLCV data
        price_col: Column to use for price (default: 'adjclose')

    Returns:
        DataFrame with DatetimeIndex and sector ETF prices as columns,
        or None if insufficient data.
    """
    prices = {}
    missing = []

    for etf in SECTOR_ETFS:
        if etf in etf_data:
            df = etf_data[etf]
            # Handle both lowercase and mixed case columns
            col = price_col.lower()
            if col in df.columns:
                prices[etf] = df[col]
            elif price_col in df.columns:
                prices[etf] = df[price_col]
            else:
                # Try 'close' as fallback
                if 'close' in df.columns:
                    prices[etf] = df['close']
                else:
                    missing.append(etf)
        else:
            missing.append(etf)

    if missing:
        logger.warning(f"Missing sector ETFs for breadth calculation: {missing}")

    if len(prices) < 6:  # Need at least half the sectors
        logger.error(f"Insufficient sector ETF data: only {len(prices)}/{N_SECTORS} available")
        return None

    # Combine into panel, aligning on dates
    panel = pd.DataFrame(prices)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    logger.info(f"Loaded {len(prices)} sector ETFs for breadth calculation "
                f"({panel.index.min().date()} to {panel.index.max().date()})")

    return panel


def compute_sector_breadth_daily(
    etf_data: Dict[str, pd.DataFrame],
    price_col: str = 'adjclose'
) -> Optional[pd.DataFrame]:
    """
    Compute daily sector ETF breadth features.

    Args:
        etf_data: Dictionary mapping ETF symbol to DataFrame with OHLCV data
        price_col: Column to use for price

    Returns:
        DataFrame with DatetimeIndex and breadth features as columns:
        - sector_breadth_adv: Count of advancing sectors (0-11)
        - sector_breadth_dec: Count of declining sectors (0-11)
        - sector_breadth_net_adv: Net advancers (adv - dec, range -11 to +11)
        - sector_breadth_ad_line: Cumulative A/D line
        - sector_breadth_pct_above_ma50: Pct of sectors above 50-day MA (0-1)
        - sector_breadth_pct_above_ma200: Pct of sectors above 200-day MA (0-1)
        - sector_breadth_mcclellan_osc: McClellan Oscillator
        - sector_breadth_mcclellan_sum: McClellan Summation Index
    """
    prices = _load_sector_etf_prices(etf_data, price_col)
    if prices is None:
        return None

    n_sectors = prices.shape[1]

    # Daily returns
    returns = prices.pct_change()

    # Advance/Decline counts
    advancing = (returns > 0).sum(axis=1).astype('float32')
    declining = (returns < 0).sum(axis=1).astype('float32')
    net_adv = advancing - declining

    # Cumulative A/D line
    ad_line = net_adv.cumsum()

    # Percent above moving averages
    ma50 = prices.rolling(50, min_periods=25).mean()
    ma200 = prices.rolling(200, min_periods=100).mean()

    pct_above_ma50 = (prices > ma50).sum(axis=1) / n_sectors
    pct_above_ma200 = (prices > ma200).sum(axis=1) / n_sectors

    # McClellan Oscillator: EMA(19) - EMA(39) of net advances
    ema_fast = net_adv.ewm(span=19, adjust=False).mean()
    ema_slow = net_adv.ewm(span=39, adjust=False).mean()
    mcclellan_osc = ema_fast - ema_slow

    # McClellan Summation Index: cumulative sum of oscillator
    mcclellan_sum = mcclellan_osc.cumsum()

    # Assemble output DataFrame
    result = pd.DataFrame({
        'sector_breadth_adv': advancing,
        'sector_breadth_dec': declining,
        'sector_breadth_net_adv': net_adv,
        'sector_breadth_ad_line': ad_line,
        'sector_breadth_pct_above_ma50': pct_above_ma50.astype('float32'),
        'sector_breadth_pct_above_ma200': pct_above_ma200.astype('float32'),
        'sector_breadth_mcclellan_osc': mcclellan_osc.astype('float32'),
        'sector_breadth_mcclellan_sum': mcclellan_sum.astype('float32'),
    }, index=prices.index)

    logger.info(f"Computed {len(result.columns)} daily sector breadth features "
                f"for {len(result)} dates")

    return result


def compute_sector_breadth_weekly(
    etf_data: Dict[str, pd.DataFrame],
    price_col: str = 'adjclose'
) -> Optional[pd.DataFrame]:
    """
    Compute weekly sector ETF breadth features.

    Args:
        etf_data: Dictionary mapping ETF symbol to DataFrame with OHLCV data
                  (should be weekly-resampled data)
        price_col: Column to use for price

    Returns:
        DataFrame with DatetimeIndex (Fridays) and weekly breadth features:
        - w_sector_breadth_adv: Weekly advancing sectors count
        - w_sector_breadth_dec: Weekly declining sectors count
        - w_sector_breadth_net_adv: Weekly net advancers
        - w_sector_breadth_ad_line: Cumulative weekly A/D line
        - w_sector_breadth_pct_above_ma10: Pct above 10-week MA
        - w_sector_breadth_pct_above_ma40: Pct above 40-week MA
        - w_sector_breadth_mcclellan_osc: Weekly McClellan Oscillator
        - w_sector_breadth_mcclellan_sum: Weekly McClellan Summation Index
    """
    prices = _load_sector_etf_prices(etf_data, price_col)
    if prices is None:
        return None

    n_sectors = prices.shape[1]

    # Weekly returns
    returns = prices.pct_change()

    # Advance/Decline counts
    advancing = (returns > 0).sum(axis=1).astype('float32')
    declining = (returns < 0).sum(axis=1).astype('float32')
    net_adv = advancing - declining

    # Cumulative A/D line
    ad_line = net_adv.cumsum()

    # Percent above moving averages (10-week ≈ 50-day, 40-week ≈ 200-day)
    ma10 = prices.rolling(10, min_periods=5).mean()
    ma40 = prices.rolling(40, min_periods=20).mean()

    pct_above_ma10 = (prices > ma10).sum(axis=1) / n_sectors
    pct_above_ma40 = (prices > ma40).sum(axis=1) / n_sectors

    # Weekly McClellan Oscillator: EMA(4) - EMA(8) of net advances
    # (4 and 8 weeks ≈ 19 and 39 days)
    ema_fast = net_adv.ewm(span=4, adjust=False).mean()
    ema_slow = net_adv.ewm(span=8, adjust=False).mean()
    mcclellan_osc = ema_fast - ema_slow

    # McClellan Summation Index
    mcclellan_sum = mcclellan_osc.cumsum()

    # Assemble output DataFrame with w_ prefix
    result = pd.DataFrame({
        'w_sector_breadth_adv': advancing,
        'w_sector_breadth_dec': declining,
        'w_sector_breadth_net_adv': net_adv,
        'w_sector_breadth_ad_line': ad_line,
        'w_sector_breadth_pct_above_ma10': pct_above_ma10.astype('float32'),
        'w_sector_breadth_pct_above_ma40': pct_above_ma40.astype('float32'),
        'w_sector_breadth_mcclellan_osc': mcclellan_osc.astype('float32'),
        'w_sector_breadth_mcclellan_sum': mcclellan_sum.astype('float32'),
    }, index=prices.index)

    logger.info(f"Computed {len(result.columns)} weekly sector breadth features "
                f"for {len(result)} weeks")

    return result


def add_sector_breadth_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    etf_data: Dict[str, pd.DataFrame],
    weekly_etf_data: Optional[Dict[str, pd.DataFrame]] = None,
    price_col: str = 'adjclose'
) -> None:
    """
    Add sector ETF breadth features to all symbols.

    Features are global-by-date (same value for all symbols on a given date).
    Computed once from the 11 sector ETFs, then broadcast to all symbols.

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        etf_data: Dictionary mapping ETF symbol to DataFrame with daily OHLCV
        weekly_etf_data: Optional dictionary with weekly-resampled ETF data.
                         If None, weekly features are skipped.
        price_col: Column to use for price
    """
    if not indicators_by_symbol:
        logger.warning("No symbols to add breadth features to")
        return

    # Compute daily breadth features
    daily_breadth = compute_sector_breadth_daily(etf_data, price_col)

    if daily_breadth is None:
        logger.error("Failed to compute daily sector breadth features")
        return

    # Compute weekly breadth features if weekly data provided
    weekly_breadth = None
    if weekly_etf_data is not None:
        weekly_breadth = compute_sector_breadth_weekly(weekly_etf_data, price_col)

    # Broadcast to all symbols
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        breadth_data = {}

        # Add daily features
        for col in daily_breadth.columns:
            if col not in df.columns:
                aligned = daily_breadth[col].reindex(df.index)
                breadth_data[col] = aligned.astype('float32')

        # Add weekly features (forward-filled to daily)
        if weekly_breadth is not None:
            for col in weekly_breadth.columns:
                if col not in df.columns:
                    # Reindex to daily dates and forward-fill
                    aligned = weekly_breadth[col].reindex(df.index, method='ffill')
                    breadth_data[col] = aligned.astype('float32')

        # Attach all at once using pd.concat to avoid fragmentation
        if breadth_data:
            new_cols_df = pd.DataFrame(breadth_data, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_cols_df], axis=1)
            attached_count += 1

    n_daily = len(daily_breadth.columns) if daily_breadth is not None else 0
    n_weekly = len(weekly_breadth.columns) if weekly_breadth is not None else 0

    logger.info(f"Attached {n_daily} daily + {n_weekly} weekly sector breadth features "
                f"to {attached_count} symbols")


def get_sector_breadth_feature_names(include_weekly: bool = True) -> List[str]:
    """
    Return list of sector breadth feature names.

    Args:
        include_weekly: Whether to include weekly feature names

    Returns:
        List of feature names
    """
    daily = [
        'sector_breadth_adv',
        'sector_breadth_dec',
        'sector_breadth_net_adv',
        'sector_breadth_ad_line',
        'sector_breadth_pct_above_ma50',
        'sector_breadth_pct_above_ma200',
        'sector_breadth_mcclellan_osc',
        'sector_breadth_mcclellan_sum',
    ]

    if not include_weekly:
        return daily

    weekly = [
        'w_sector_breadth_adv',
        'w_sector_breadth_dec',
        'w_sector_breadth_net_adv',
        'w_sector_breadth_ad_line',
        'w_sector_breadth_pct_above_ma10',
        'w_sector_breadth_pct_above_ma40',
        'w_sector_breadth_mcclellan_osc',
        'w_sector_breadth_mcclellan_sum',
    ]

    return daily + weekly
