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
    spy_symbol: str = "SPY",
    enhanced_mappings: Optional[Dict[str, Dict]] = None
) -> None:
    """
    Add comprehensive relative strength features with cap-weighted, equal-weighted, and subsector analysis.
    
    Standard features (always added if benchmarks available):
    - rel_strength_spy: Raw relative strength vs SPY
    - rel_strength_spy_norm: Normalized relative strength vs SPY  
    - rel_strength_spy_slope20: 20-period slope of RS vs SPY
    - rel_strength_sector: Raw relative strength vs sector ETF
    - rel_strength_sector_norm: Normalized relative strength vs sector ETF
    - rel_strength_sector_slope20: 20-period slope of RS vs sector ETF
    
    Enhanced features (if enhanced_mappings provided):
    - rel_strength_rsp: Raw relative strength vs RSP (equal-weight market)
    - rel_strength_rsp_norm: Normalized relative strength vs RSP
    - rel_strength_rsp_slope20: 20-period slope of RS vs RSP
    - rel_strength_sector_ew: Raw relative strength vs equal-weight sector ETF
    - rel_strength_sector_ew_norm: Normalized relative strength vs equal-weight sector ETF
    - rel_strength_sector_ew_slope20: 20-period slope of RS vs equal-weight sector ETF
    - rel_strength_subsector: Raw relative strength vs subsector ETF
    - rel_strength_subsector_norm: Normalized relative strength vs subsector ETF
    - rel_strength_subsector_slope20: 20-period slope of RS vs subsector ETF
    - rel_strength_sector_vs_market: Sector ETF relative strength vs market
    - rel_strength_sector_vs_market_norm: Normalized sector vs market strength
    - rel_strength_sector_vs_market_slope20: 20-period slope of sector vs market
    
    Total: 24 relative strength features when enhanced_mappings is provided
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        sectors: Optional mapping of symbol -> sector name
        sector_to_etf: Optional mapping of lowercase sector name -> ETF symbol
        spy_symbol: Symbol to use as market benchmark
        enhanced_mappings: Optional enhanced sector/subsector mappings from sector_mapping module
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

    # Enhanced relative strength features (if enhanced mappings provided)
    rs_subsector_count = 0
    rs_equal_weight_count = 0
    
    if enhanced_mappings:
        logger.info("Computing enhanced relative strength with equal-weighted ETFs and subsector mappings...")
        
        # Pre-load all required benchmark data
        subsector_benchmarks: Dict[str, pd.Series] = {}
        equal_weight_benchmarks: Dict[str, pd.Series] = {}
        
        # Get required ETFs from enhanced mappings
        required_subsector_etfs = {
            mapping['subsector_etf'] for mapping in enhanced_mappings.values()
            if mapping['subsector_etf'] is not None
        }
        required_equal_weight_etfs = {
            mapping['equal_weight_etf'] for mapping in enhanced_mappings.values()
            if mapping['equal_weight_etf'] is not None
        }
        
        # Load subsector benchmarks
        for etf in required_subsector_etfs:
            etf_df = indicators_by_symbol.get(etf)
            if etf_df is not None and "adjclose" in etf_df.columns:
                subsector_benchmarks[etf] = pd.to_numeric(etf_df["adjclose"], errors='coerce')
        
        # Load equal-weight benchmarks (including RSP for market)
        for etf in required_equal_weight_etfs | {'RSP'}:
            etf_df = indicators_by_symbol.get(etf)
            if etf_df is not None and "adjclose" in etf_df.columns:
                equal_weight_benchmarks[etf] = pd.to_numeric(etf_df["adjclose"], errors='coerce')
        
        logger.debug(f"Loaded {len(subsector_benchmarks)} subsector benchmarks")
        logger.debug(f"Loaded {len(equal_weight_benchmarks)} equal-weight benchmarks")
        
        # Compute enhanced relative strength for each symbol
        for sym, df in indicators_by_symbol.items():
            if "adjclose" not in df.columns or sym not in enhanced_mappings:
                continue
            
            px = pd.to_numeric(df["adjclose"], errors='coerce')
            mapping = enhanced_mappings[sym]
            
            # Equal-weighted market relative strength (vs RSP)
            if 'RSP' in equal_weight_benchmarks:
                rsp_bench = equal_weight_benchmarks['RSP']
                rs, rs_norm, rs_slope = _compute_relative_strength_block(px, rsp_bench)
                df["rel_strength_rsp"] = rs
                df["rel_strength_rsp_norm"] = rs_norm  
                df["rel_strength_rsp_slope20"] = rs_slope
                rs_equal_weight_count += 1
            
            # Equal-weighted sector relative strength
            equal_weight_etf = mapping.get('equal_weight_etf')
            if equal_weight_etf and equal_weight_etf in equal_weight_benchmarks:
                ew_bench = equal_weight_benchmarks[equal_weight_etf]
                rs, rs_norm, rs_slope = _compute_relative_strength_block(px, ew_bench)
                df["rel_strength_sector_ew"] = rs
                df["rel_strength_sector_ew_norm"] = rs_norm
                df["rel_strength_sector_ew_slope20"] = rs_slope
            
            # Subsector relative strength
            subsector_etf = mapping.get('subsector_etf')
            if subsector_etf and subsector_etf in subsector_benchmarks:
                subsector_bench = subsector_benchmarks[subsector_etf]
                rs, rs_norm, rs_slope = _compute_relative_strength_block(px, subsector_bench)
                df["rel_strength_subsector"] = rs
                df["rel_strength_subsector_norm"] = rs_norm
                df["rel_strength_subsector_slope20"] = rs_slope
                rs_subsector_count += 1
            
            # Cross-benchmark features: sector vs market comparisons
            sector_etf = mapping.get('sector_etf')
            if sector_etf in sector_benchmarks and 'SPY' in indicators_by_symbol:
                sector_bench = sector_benchmarks[sector_etf]
                spy_bench = spy_px
                if sector_bench is not None and spy_bench is not None:
                    # Sector vs market strength
                    rs, rs_norm, rs_slope = _compute_relative_strength_block(sector_bench, spy_bench)
                    df["rel_strength_sector_vs_market"] = rs
                    df["rel_strength_sector_vs_market_norm"] = rs_norm
                    df["rel_strength_sector_vs_market_slope20"] = rs_slope

    total_enhanced = rs_subsector_count if enhanced_mappings else 0
    total_equal_weight = rs_equal_weight_count if enhanced_mappings else 0
    
    logger.info(f"Added SPY relative strength to {rs_spy_count} symbols, "
                f"sector relative strength to {rs_sector_count} symbols"
                + (f", equal-weight relative strength to {rs_equal_weight_count} symbols" if total_equal_weight > 0 else "")
                + (f", subsector relative strength to {rs_subsector_count} symbols" if total_enhanced > 0 else ""))