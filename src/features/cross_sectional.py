"""
Cross-sectional feature computation requiring data from multiple stocks.

This module provides a centralized function for computing all features that require
comparing a stock against other stocks in the universe (cross-sectional features).
"""
import logging
from typing import Dict, List, Optional

import pandas as pd

# Feature module imports with fallback for both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from .volatility import add_vol_regime_cs_context
    from .alpha import add_alpha_momentum_features
    from .relstrength import add_relative_strength
    from .breadth import add_breadth_series
    from .xsec import add_xsec_momentum_panel
except ImportError:
    # Fallback to absolute imports (when run directly)
    from src.features.volatility import add_vol_regime_cs_context
    from src.features.alpha import add_alpha_momentum_features
    from src.features.relstrength import add_relative_strength
    from src.features.breadth import add_breadth_series
    from src.features.xsec import add_xsec_momentum_panel

logger = logging.getLogger(__name__)


def compute_cross_sectional_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    market_symbol: str = "SPY",
    sp500_tickers: Optional[List[str]] = None,
    enhanced_mappings: Optional[Dict[str, Dict]] = None,
    beta_win: int = 60,
    alpha_windows: tuple = (20, 60, 120),
    alpha_ema_span: int = 10,
    xsec_lookbacks: tuple = (5, 20, 60),
    price_col: str = "adjclose"
) -> None:
    """
    Compute all cross-sectional features that require data from multiple stocks.

    This function modifies the indicators_by_symbol dictionary in place, adding
    cross-sectional features to each symbol's DataFrame.

    Features computed:
    1. Cross-sectional volatility regime context (vol_regime_cs_median, vol_regime_rel)
    2. Alpha-momentum features (beta-adjusted returns vs market and sectors)
    3. Relative strength features (vs SPY, sectors, and subsectors)
    4. Market breadth features (advancing/declining stocks, new highs/lows)
    5. Cross-sectional momentum rankings (percentile ranks within universe/sector)

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame with features
                             (modified in place)
        sectors: Dictionary mapping symbol -> sector name (optional)
        sector_to_etf: Dictionary mapping sector name -> ETF symbol (optional)
        market_symbol: Symbol to use as market benchmark (default: "SPY")
        sp500_tickers: List of S&P 500 tickers for breadth calculation (optional)
        enhanced_mappings: Enhanced sector/subsector mappings dictionary (optional)
        beta_win: Window for beta calculation in alpha features (default: 60)
        alpha_windows: Windows for alpha-momentum computation (default: (20, 60, 120))
        alpha_ema_span: EMA span for alpha smoothing (default: 10)
        xsec_lookbacks: Lookback periods for cross-sectional momentum (default: (5, 20, 60))
        price_col: Column name for price data (default: "adjclose")

    Returns:
        None (modifies indicators_by_symbol in place)

    Example:
        >>> from src.features import compute_cross_sectional_features
        >>>
        >>> # First compute single-stock features for all symbols
        >>> indicators_by_symbol = {...}  # Your dict of symbol DataFrames
        >>>
        >>> # Then add cross-sectional features
        >>> compute_cross_sectional_features(
        ...     indicators_by_symbol,
        ...     sectors=sector_map,
        ...     sector_to_etf=sector_etf_map,
        ...     market_symbol="SPY",
        ...     sp500_tickers=sp500_list
        ... )
        >>>
        >>> # Now all DataFrames have cross-sectional features added
        >>> aapl_df = indicators_by_symbol['AAPL']
        >>> print(aapl_df[['vol_regime_rel', 'alpha_60d', 'rs_spy_20d']].head())
    """
    logger.info("Computing cross-sectional features for all symbols")

    n_symbols = len(indicators_by_symbol)
    logger.info(f"Processing {n_symbols} symbols")

    # 1) Cross-sectional volatility regime context
    logger.info("Adding cross-sectional volatility regime context...")
    add_vol_regime_cs_context(indicators_by_symbol)
    logger.debug("  ✓ Volatility context completed")

    # 2) Alpha-momentum features (requires sectors, sector_to_etf, and market_symbol)
    if sectors is not None and sector_to_etf is not None:
        logger.info("Adding alpha-momentum features...")
        add_alpha_momentum_features(
            indicators_by_symbol,
            sectors=sectors,
            sector_to_etf=sector_to_etf,
            market_symbol=market_symbol,
            beta_win=beta_win,
            windows=alpha_windows,
            ema_span=alpha_ema_span
        )
        logger.debug(f"  ✓ Alpha-momentum completed (beta_win={beta_win}, windows={alpha_windows})")
    else:
        logger.warning("Skipping alpha-momentum features (sectors or sector_to_etf not provided)")

    # 3) Relative strength vs SPY, sectors, and subsectors
    logger.info("Adding relative strength features...")
    add_relative_strength(
        indicators_by_symbol,
        sectors=sectors,
        sector_to_etf=sector_to_etf,
        spy_symbol=market_symbol,
        enhanced_mappings=enhanced_mappings
    )
    logger.debug("  ✓ Relative strength completed")

    # 4) Breadth features (requires SP500 ticker list)
    if sp500_tickers:
        logger.info("Adding market breadth features...")
        add_breadth_series(indicators_by_symbol, sp500_tickers)
        logger.debug(f"  ✓ Breadth features completed ({len(sp500_tickers)} tickers)")
    else:
        logger.warning("Skipping breadth features (sp500_tickers not provided)")

    # 5) Cross-sectional momentum rankings
    logger.info("Adding cross-sectional momentum features...")
    add_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=xsec_lookbacks,
        price_col=price_col,
        sector_map=sectors
    )
    logger.debug(f"  ✓ Cross-sectional momentum completed (lookbacks={xsec_lookbacks})")

    logger.info(f"Cross-sectional feature computation completed for {n_symbols} symbols")


def compute_cross_sectional_features_safe(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    **kwargs
) -> bool:
    """
    Safe wrapper for compute_cross_sectional_features with error handling.

    This version catches exceptions and logs errors instead of raising them,
    allowing the pipeline to continue even if cross-sectional computation fails.

    Args:
        indicators_by_symbol: Dictionary mapping symbol -> DataFrame with features
        **kwargs: Additional arguments passed to compute_cross_sectional_features

    Returns:
        True if successful, False if computation failed
    """
    try:
        compute_cross_sectional_features(indicators_by_symbol, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Cross-sectional feature computation failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage for testing
    import logging
    import numpy as np

    logging.basicConfig(level=logging.DEBUG)

    # Create sample multi-stock data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']

    indicators_by_symbol = {}
    for sym in symbols:
        df = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'adjclose': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'ret': np.random.randn(len(dates)) * 0.02,
            'rv_ratio_20_100': np.random.randn(len(dates)) * 0.5 + 1.0,
            'vol_regime': np.random.randn(len(dates)) * 0.1
        }, index=dates)
        indicators_by_symbol[sym] = df

    # Sector mapping
    sectors = {
        'AAPL': 'technology',
        'MSFT': 'technology',
        'GOOGL': 'technology',
        'SPY': 'market'
    }

    sector_to_etf = {
        'technology': 'XLK',
        'market': 'SPY'
    }

    # Compute cross-sectional features
    compute_cross_sectional_features(
        indicators_by_symbol,
        sectors=sectors,
        sector_to_etf=sector_to_etf,
        market_symbol='SPY',
        sp500_tickers=['AAPL', 'MSFT', 'GOOGL']
    )

    # Show what was added
    sample_df = indicators_by_symbol['AAPL']
    print(f"\nTotal columns in AAPL: {len(sample_df.columns)}")

    # Show cross-sectional feature columns
    cs_cols = [col for col in sample_df.columns if any(x in col for x in
               ['_cs_', 'vol_regime_rel', 'alpha_', 'rs_', 'breadth_', 'xmom_'])]
    print(f"Cross-sectional feature columns: {len(cs_cols)}")
    print(f"Sample: {cs_cols[:10]}")
