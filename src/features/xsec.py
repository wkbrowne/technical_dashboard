"""
Cross-sectional momentum features comparing each symbol's performance to the universe.

This module computes cross-sectional z-scores of momentum for different lookback periods,
with optional sector-neutral adjustments to control for sector effects.
"""
import logging
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_xsec_momentum_panel(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (5, 20, 60),
    price_col: str = "adjclose",
    sector_map: Optional[Dict[str, str]] = None,
    col_prefix: str = "xsec_mom"
) -> None:
    """
    Add cross-sectional momentum z-scores for each lookback period.
    
    For each lookback L:
    1. Compute L-day log-return per symbol
    2. For each date, z-score across symbols: z = (ret - cross_median) / cross_std
    3. If sector_map provided: sector-neutral first (subtract sector median on each date),
       then z-score across all symbols (so values are directly comparable)
    
    Features added:
    - {col_prefix}_{L}d_z: Plain cross-sectional momentum z-score
    - {col_prefix}_{L}d_sect_neutral_z: Sector-neutral cross-sectional momentum z-score (if sector_map provided)
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        lookbacks: Lookback periods for momentum calculation
        price_col: Column name for price data
        sector_map: Optional mapping of symbol -> sector (any taxonomy)
        col_prefix: Prefix for output column names
        
    Note:
        Cross-sectional features provide relative performance measures - they indicate
        how each symbol performs relative to the broader universe on each date.
    """
    logger.info(f"Computing cross-sectional momentum for lookbacks: {list(lookbacks)}")
    
    # Build price panel (date index, columns = symbols)
    syms = [s for s, df in indicators_by_symbol.items() if price_col in df.columns]
    if not syms:
        logger.warning(f"No symbols have {price_col} column")
        return
    
    logger.debug(f"Building price panel for {len(syms)} symbols")
    
    panel = pd.DataFrame(index=pd.Index([], name=None))
    for s in syms:
        panel[s] = pd.to_numeric(indicators_by_symbol[s][price_col], errors='coerce')

    # Align all indexes (outer join) and sort
    panel = panel.sort_index()

    # Compute log prices and daily log returns (vectorized)
    logp = np.log(panel.replace(0, np.nan))
    ret1 = logp.diff()  # Daily log return

    for L in lookbacks:
        logger.debug(f"Processing {L}-day momentum")
        
        # L-day log return: sum of daily log-returns over L days
        momL = ret1.rolling(L, min_periods=max(3, L//3)).sum()

        # Sector-neutral adjustment (optional)
        if sector_map:
            logger.debug(f"Applying sector-neutral adjustment for {L}-day momentum")
            
            # Group columns by sector
            sect_groups: Dict[str, list] = {}
            for s in syms:
                sec = sector_map.get(s)
                if isinstance(sec, str):
                    sect_groups.setdefault(sec, []).append(s)

            # Subtract sector median per date within each sector
            sector_neutral_dfs = []
            for sec, cols in sect_groups.items():
                sub = momL[cols]
                med = sub.median(axis=1, skipna=True)
                sector_neutral_dfs.append(sub.sub(med, axis=0))

            # Handle symbols without sector mapping
            missing = [s for s in syms if s not in sum([list(cols) for cols in sect_groups.values()], [])]
            if missing:
                sector_neutral_dfs.append(momL[missing])

            # Combine all sector-neutral data
            mom_sect_neutral = pd.concat(sector_neutral_dfs, axis=1) if sector_neutral_dfs else pd.DataFrame(index=momL.index)

            # Final z-score across ALL symbols (so values are comparable across sectors)
            row_std = mom_sect_neutral.std(axis=1, ddof=0).replace(0, np.nan)
            row_med = mom_sect_neutral.median(axis=1, skipna=True)
            z_sect = mom_sect_neutral.sub(row_med, axis=0).div(row_std, axis=0)

        # Plain cross-sectional z-score (no sector neutral)
        row_std_plain = momL.std(axis=1, ddof=0).replace(0, np.nan)
        row_med_plain = momL.median(axis=1, skipna=True)
        z_plain = momL.sub(row_med_plain, axis=0).div(row_std_plain, axis=0)

        # Write back to each symbol DataFrame
        plain_name = f"{col_prefix}_{L}d_z"
        for s in syms:
            indicators_by_symbol[s][plain_name] = pd.to_numeric(
                z_plain[s].reindex(indicators_by_symbol[s].index), errors='coerce'
            ).astype('float32')

        if sector_map:
            sect_name = f"{col_prefix}_{L}d_sect_neutral_z"
            for s in syms:
                indicators_by_symbol[s][sect_name] = pd.to_numeric(
                    z_sect[s].reindex(indicators_by_symbol[s].index), errors='coerce'
                ).astype('float32')

    features_added = len(lookbacks) * (2 if sector_map else 1)
    logger.info(f"Added {features_added} cross-sectional momentum features to {len(syms)} symbols")