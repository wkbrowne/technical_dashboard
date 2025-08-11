"""
Data saving utilities for persisting computed features.

This module provides functions for saving feature data in various formats,
including per-symbol parquet files and long-format consolidated files.
"""
import logging
from pathlib import Path
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


def save_symbol_frames(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    out_dir: Path
) -> None:
    """
    Save per-symbol DataFrames as individual parquet files.
    
    Each symbol's DataFrame is saved as a separate parquet file named {symbol}.parquet
    in the specified output directory. This format is efficient for accessing
    individual symbol data.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        out_dir: Output directory path
        
    Raises:
        IOError: If directory creation or file writing fails
    """
    logger.info(f"Saving {len(indicators_by_symbol)} symbol frames to {out_dir}")
    
    # Ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for sym, df in indicators_by_symbol.items():
        try:
            output_path = out_dir / f"{sym}.parquet"
            df.to_parquet(
                output_path, 
                engine="pyarrow", 
                compression="snappy"
            )
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save {sym}: {e}")
    
    logger.info(f"Successfully saved {saved_count} symbol frames")


def save_long_parquet(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    out_path: Path
) -> None:
    """
    Save all symbols as a single long-format parquet file.
    
    Converts the symbol-keyed dictionary of DataFrames into a single long-format
    DataFrame with 'symbol' and 'date' columns, suitable for analytics tools
    that prefer long-format data.
    
    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame
        out_path: Output file path
        
    Raises:
        IOError: If file writing fails
    """
    logger.info(f"Saving long-format parquet with {len(indicators_by_symbol)} symbols to {out_path}")
    
    if not indicators_by_symbol:
        logger.warning("No data to save")
        return
    
    # Collect all DataFrames with symbol and date columns
    parts = []
    for sym, df in indicators_by_symbol.items():
        if df.empty:
            continue
            
        # Make a copy and add metadata columns
        x = df.copy()
        x["symbol"] = sym
        x["date"] = x.index
        parts.append(x)
    
    if not parts:
        logger.warning("No non-empty DataFrames to save")
        return
    
    # Concatenate all DataFrames
    logger.debug("Concatenating DataFrames...")
    long_df = pd.concat(parts, axis=0, ignore_index=True)
    
    # Ensure proper data types
    long_df["symbol"] = long_df["symbol"].astype("string")
    long_df["date"] = pd.to_datetime(long_df["date"])
    
    # Save to parquet
    try:
        # Ensure parent directory exists
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        long_df.to_parquet(
            out_path, 
            engine="pyarrow", 
            compression="snappy", 
            index=False
        )
        logger.info(f"Successfully saved long-format data: {len(long_df)} rows, {len(long_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to save long-format parquet: {e}")
        raise


def load_symbol_frames(
    symbols_dir: Path,
    symbols: list = None
) -> Dict[str, pd.DataFrame]:
    """
    Load per-symbol DataFrames from parquet files.
    
    Args:
        symbols_dir: Directory containing symbol parquet files
        symbols: List of symbols to load (None for all)
        
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    logger.info(f"Loading symbol frames from {symbols_dir}")
    
    symbols_dir = Path(symbols_dir)
    if not symbols_dir.exists():
        logger.error(f"Symbols directory does not exist: {symbols_dir}")
        return {}
    
    # Find all parquet files
    parquet_files = list(symbols_dir.glob("*.parquet"))
    logger.debug(f"Found {len(parquet_files)} parquet files")
    
    loaded = {}
    for pq_file in parquet_files:
        symbol = pq_file.stem  # Filename without extension
        
        # Skip if symbols list provided and this symbol not in it
        if symbols is not None and symbol not in symbols:
            continue
            
        try:
            df = pd.read_parquet(pq_file)
            loaded[symbol] = df
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
    
    logger.info(f"Successfully loaded {len(loaded)} symbol frames")
    return loaded


def load_long_parquet(parquet_path: Path) -> pd.DataFrame:
    """
    Load long-format feature data from parquet file.
    
    Args:
        parquet_path: Path to long-format parquet file
        
    Returns:
        DataFrame with long-format feature data
    """
    logger.info(f"Loading long-format parquet from {parquet_path}")
    
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded long-format data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load long-format parquet: {e}")
        raise