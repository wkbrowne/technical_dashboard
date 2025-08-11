"""
Market breadth features based on percentage of stocks above moving averages and A/D line.

This module computes market breadth indicators that measure the participation
of stocks in market moves, including percentage above MAs and advance/decline metrics.
"""
import logging
from typing import Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _above_ma_series(p: pd.Series, L: int) -> pd.Series:
    """
    Create binary series indicating whether price is above its moving average.
    
    Args:
        p: Price series
        L: Moving average length
        
    Returns:
        Binary series (1.0 if above MA, 0.0 if below, NaN if insufficient data)
    """
    if p.isna().all():
        return pd.Series(index=p.index, dtype='float32')
    
    ma = p.rolling(L, min_periods=max(1, L//2)).mean()
    return (p > ma).astype('float32')


def add_breadth_series(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    universe_tickers: List[str]
) -> None:
    """
    Add market breadth features based on percentage of universe stocks above moving averages.
    
    Features added to all symbols:
    - pct_universe_above_ma20: Percentage of universe above 20-day MA
    - pct_universe_above_ma50: Percentage of universe above 50-day MA  
    - pct_universe_above_ma200: Percentage of universe above 200-day MA
    - ad_line_universe: Cumulative advance/decline line for universe
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        universe_tickers: List of ticker symbols to include in breadth calculation
        
    Note:
        The breadth features are computed across the specified universe and then
        attached to ALL symbols in indicators_by_symbol, not just universe members.
    """
    logger.info(f"Computing breadth features for universe of {len(universe_tickers)} tickers")
    
    # Find intersection of universe tickers and available symbols
    available_symbols = set(indicators_by_symbol.keys())
    matched_tickers = [t for t in universe_tickers if t in available_symbols]
    
    logger.info(f"Matched {len(matched_tickers)} tickers for breadth calculation "
                f"(out of {len(universe_tickers)} universe tickers)")
    
    if len(matched_tickers) < 10:
        logger.warning("Very few tickers matched for breadth calculation")
        logger.debug(f"Sample universe tickers: {universe_tickers[:5]}")
        logger.debug(f"Sample available symbols: {list(available_symbols)[:5]}")

    # Collect price series and compute breadth indicators
    above20, above50, above200, advdec = {}, {}, {}, {}
    processed_count = 0
    
    for sym in matched_tickers:
        df = indicators_by_symbol[sym]
        
        # Choose best available price column
        price_col = "close" if "close" in df.columns else ("adjclose" if "adjclose" in df.columns else None)
        if price_col is None:
            continue
            
        px = pd.to_numeric(df[price_col], errors='coerce')
        if px.isna().all():
            continue
        
        # Daily return direction for advance/decline
        ret1 = px.pct_change(fill_method=None)
        advdec[sym] = np.sign(ret1).fillna(0.0)
        
        # Above moving average indicators
        above20[sym] = _above_ma_series(px, 20)
        above50[sym] = _above_ma_series(px, 50)  
        above200[sym] = _above_ma_series(px, 200)
        
        processed_count += 1

    logger.info(f"Processed {processed_count} symbols for breadth calculation")
    
    if not above50:  # Check if we have any data
        logger.warning("No valid data for breadth calculation")
        return

    def _pct(df_map: Dict[str, pd.Series]) -> pd.Series:
        """Convert dictionary of binary series to percentage series."""
        panel = pd.DataFrame(df_map)
        return (panel.mean(axis=1, skipna=True) * 100.0).rename(None)

    # Compute percentage above each MA
    pct20 = _pct(above20).rename("pct_universe_above_ma20")
    pct50 = _pct(above50).rename("pct_universe_above_ma50")
    pct200 = _pct(above200).rename("pct_universe_above_ma200")

    # Compute advance/decline line
    ad_panel = pd.DataFrame(advdec)
    ad_net = ad_panel.sum(axis=1, skipna=True)  # Net advances per day
    ad_line = ad_net.cumsum().rename("ad_line_universe")  # Cumulative A/D line

    # Attach breadth features to ALL symbols
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        for s in [pct20, pct50, pct200, ad_line]:
            if len(s) > 0:
                # Reindex to symbol's date range and convert to float32
                df[s.name] = pd.to_numeric(s.reindex(df.index), errors='coerce').astype('float32')
        attached_count += 1

    logger.info(f"Breadth features attached to {attached_count} symbols "
                f"(computed from {processed_count} universe members)")