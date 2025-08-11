"""
Data assembly and indicator construction from wide-format market data.

This module contains functions for converting wide-format OHLCV data into
per-symbol DataFrames with basic indicators and returns.
"""
import logging
from typing import Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame without modifying column names.
    
    Note: Column names are kept as-is since they represent stock symbols.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Copy of the DataFrame
    """
    return df.copy()


def assemble_indicators_from_wide(
    data: Dict[str, pd.DataFrame],
    adjust_ohlc_with_factor: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convert wide-format OHLCV data into per-symbol DataFrames with indicators.
    
    Args:
        data: Dictionary with keys like 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'
              where values are wide DataFrames (dates x symbols)
        adjust_ohlc_with_factor: Whether to adjust OHLC prices using the adjustment factor
        
    Returns:
        Dictionary mapping symbol -> DataFrame with columns:
        - open, high, low, close, adjclose, volume (all lowercase)
        - ret: log returns calculated from adjclose
        
    Raises:
        ValueError: If required 'AdjClose' key is missing from data
    """
    logger.debug(f"Assembling indicators from wide data with keys: {list(data.keys())}")
    
    req = ["AdjClose"]
    for r in req:
        if r not in data:
            raise ValueError(f"Expected '{r}' in loaded data keys; got {list(data.keys())}")

    keys = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    frames = {
        k.lower(): _safe_lower_columns(v) 
        for k, v in data.items() 
        if k in keys and isinstance(v, pd.DataFrame)
    }
    
    all_syms = set()
    for dfw in frames.values():
        all_syms |= set(dfw.columns)

    indicators_by_symbol: Dict[str, pd.DataFrame] = {}
    
    for sym in sorted(all_syms):
        parts = []
        for k, dfw in frames.items():
            if sym in dfw.columns:
                s = pd.to_numeric(dfw[sym], errors='coerce').rename(k)
                parts.append(s)
        
        if not parts:
            continue
            
        df = pd.concat(parts, axis=1).sort_index()
        
        # Ensure all required columns exist
        for col in ["open", "high", "low", "close", "adjclose", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        # Adjust OHLC prices using adjustment factor
        if (adjust_ohlc_with_factor and 
            ("close" in df.columns) and 
            ("adjclose" in df.columns)):
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = df["adjclose"] / df["close"]
            for c in ["open", "high", "low"]:
                df[c] = df[c] * factor

        # Calculate log returns
        df["ret"] = np.log(df["adjclose"]).diff()
        indicators_by_symbol[sym] = df

    logger.info(f"Assembled indicators for {len(indicators_by_symbol)} symbols")
    return indicators_by_symbol