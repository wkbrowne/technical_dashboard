"""
Relative strength features comparing individual securities to market and sector benchmarks.

This module computes relative strength metrics by comparing each symbol's performance
to broader market indices (like SPY) and sector-specific ETFs.
"""
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _compute_relative_strength_block(
    price: pd.Series, 
    bench: pd.Series, 
    look: int = 60, 
    slope_win: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute relative strength metrics for a single price series vs benchmark.
    
    Args:
        price: Price series for the security
        bench: Benchmark price series
        look: Lookback window for RS normalization
        slope_win: Window for computing RS slope
        
    Returns:
        Tuple of (raw_rs, normalized_rs, rs_slope)
    """
    # Align benchmark to price index and handle missing values
    bench_aligned = pd.to_numeric(bench.reindex(price.index), errors='coerce').replace(0, np.nan)
    
    # Raw relative strength ratio
    rs = price / bench_aligned
    
    # Normalized RS: (current_rs / rolling_mean_rs) - 1
    roll = rs.rolling(look, min_periods=max(5, look//3)).mean()
    rs_norm = (rs / roll) - 1.0
    
    # RS slope: change in RS over slope_win periods, normalized per period
    rs_slope = (rs - rs.shift(slope_win)) / float(slope_win)
    
    return rs.astype('float32'), rs_norm.astype('float32'), rs_slope.astype('float32')


def add_relative_strength(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    spy_symbol: str = "SPY"
) -> None:
    """
    Add relative strength features comparing each symbol to market and sector benchmarks.
    
    Features added (if benchmarks available):
    - rel_strength_spy: Raw relative strength vs SPY
    - rel_strength_spy_norm: Normalized relative strength vs SPY  
    - rel_strength_spy_slope20: 20-period slope of RS vs SPY
    - rel_strength_sector: Raw relative strength vs sector ETF
    - rel_strength_sector_norm: Normalized relative strength vs sector ETF
    - rel_strength_sector_slope20: 20-period slope of RS vs sector ETF
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        sectors: Optional mapping of symbol -> sector name
        sector_to_etf: Optional mapping of lowercase sector name -> ETF symbol
        spy_symbol: Symbol to use as market benchmark
    """
    logger.info(f"Computing relative strength features using {spy_symbol} as market benchmark")
    
    # Get SPY benchmark data
    spy_data = indicators_by_symbol.get(spy_symbol)
    spy_px = None
    if spy_data is not None and "adjclose" in spy_data.columns:
        spy_px = pd.to_numeric(spy_data["adjclose"], errors='coerce')
        logger.debug(f"Using {spy_symbol} as market benchmark ({len(spy_px)} data points)")

    # Prepare sector mappings (convert to lowercase)
    sector_to_etf = sector_to_etf or {}
    sector_to_etf_lc = {k.lower(): v for k, v in sector_to_etf.items()}

    # Pre-load sector benchmark data
    sector_benchmarks: Dict[str, pd.Series] = {}
    if sectors and sector_to_etf:
        # Find all ETFs needed for sector benchmarks
        etfs_needed = set()
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sector_to_etf_lc.get(sec.lower())
                if etf:
                    etfs_needed.add(etf)
        
        # Load price data for each needed sector ETF
        for etf in etfs_needed:
            etf_df = indicators_by_symbol.get(etf)
            if etf_df is not None and "adjclose" in etf_df.columns:
                sector_benchmarks[etf] = pd.to_numeric(etf_df["adjclose"], errors='coerce')
        
        logger.debug(f"Loaded {len(sector_benchmarks)} sector benchmarks: {list(sector_benchmarks.keys())}")

    rs_spy_count = 0
    rs_sector_count = 0
    
    for sym, df in indicators_by_symbol.items():
        if "adjclose" not in df.columns:
            continue
        
        px = pd.to_numeric(df["adjclose"], errors='coerce')

        # Market relative strength (vs SPY)
        if spy_px is not None and sym != spy_symbol:
            rs, rs_norm, rs_slope = _compute_relative_strength_block(px, spy_px)
            df["rel_strength_spy"] = rs
            df["rel_strength_spy_norm"] = rs_norm  
            df["rel_strength_spy_slope20"] = rs_slope
            rs_spy_count += 1

        # Sector relative strength
        if sectors and sector_to_etf:
            sec = sectors.get(sym)
            etf = sector_to_etf_lc.get(sec.lower()) if isinstance(sec, str) else None
            bench = sector_benchmarks.get(etf) if etf else None
            
            if bench is not None:
                rs, rs_norm, rs_slope = _compute_relative_strength_block(px, bench)
                df["rel_strength_sector"] = rs
                df["rel_strength_sector_norm"] = rs_norm
                df["rel_strength_sector_slope20"] = rs_slope
                rs_sector_count += 1

    logger.info(f"Added SPY relative strength to {rs_spy_count} symbols, "
                f"sector relative strength to {rs_sector_count} symbols")