"""
Single-stock feature computation without cross-sectional dependencies.

This module provides a standalone function for computing all features that can be
calculated for a single stock without requiring data from other stocks.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

# Feature module imports with fallback for both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from .trend import add_trend_features, add_rsi_features, add_macd_features
    from .volatility import add_multiscale_vol_regime
    from .distance import add_distance_to_ma_features
    from .range_breakout import add_range_breakout_features
    from .volume import add_volume_features, add_volume_shock_features
except ImportError:
    # Fallback to absolute imports (when run directly)
    from src.features.trend import add_trend_features, add_rsi_features, add_macd_features
    from src.features.volatility import add_multiscale_vol_regime
    from src.features.distance import add_distance_to_ma_features
    from src.features.range_breakout import add_range_breakout_features
    from src.features.volume import add_volume_features, add_volume_shock_features

logger = logging.getLogger(__name__)


def compute_single_stock_features(
    df: pd.DataFrame,
    price_col: str = 'adjclose',
    ret_col: str = 'ret',
    vol_col: str = 'volume',
    ensure_returns: bool = True
) -> pd.DataFrame:
    """
    Compute all single-stock features that don't require cross-sectional data.

    This function applies the complete single-stock feature stack without OHLC adjustment.
    It assumes the input DataFrame already has properly adjusted price data.

    Features computed:
    1. Returns (if not present and ensure_returns=True)
    2. Trend features (MA slopes, agreement, etc.)
    3. RSI features (momentum oscillator)
    4. MACD features (histogram and derivative)
    5. Multi-scale volatility regime (single-stock only)
    6. Distance-to-MA features (z-scores)
    7. Range/breakout features (including ATR14)
    8. Volume features
    9. Volume shock features

    Note: Cross-sectional volatility features (vol_regime_cs_median, vol_regime_rel)
    are NOT included here as they require data from multiple stocks. Use
    add_vol_regime_cs_context() separately for those features.

    Args:
        df: Input DataFrame with OHLCV data (must have index as DatetimeIndex)
        price_col: Column name for price data (default: 'adjclose')
        ret_col: Column name for returns (default: 'ret')
        vol_col: Column name for volume data (default: 'volume')
        ensure_returns: If True, calculate returns from price_col if not present

    Returns:
        DataFrame with all single-stock features added

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> import pandas as pd
        >>> # Load your stock data
        >>> stock_df = pd.read_parquet('AAPL.parquet')
        >>> # Compute features
        >>> features_df = compute_single_stock_features(stock_df)
        >>> # Inspect what was added
        >>> new_cols = set(features_df.columns) - set(stock_df.columns)
        >>> print(f"Added {len(new_cols)} feature columns")
    """
    # Validate input
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex, features may not compute correctly")

    # Copy to avoid modifying original
    out = df.copy()

    # Ensure returns exist if requested
    if ensure_returns and ret_col not in out.columns:
        if price_col not in out.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")

        logger.debug(f"Calculating returns from {price_col}")
        out[ret_col] = np.log(pd.to_numeric(out[price_col], errors="coerce")).diff()

    # Validate required columns exist
    if price_col not in out.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    if ret_col not in out.columns:
        raise ValueError(f"Returns column '{ret_col}' not found in DataFrame")

    # 1) Trend features (MA slopes, agreement, etc.)
    logger.debug("Adding trend features")
    out = add_trend_features(
        out,
        src_col=price_col,
        ma_periods=(10, 20, 30, 50, 75, 100, 150, 200),
        slope_window=20,
        eps=1e-5
    )

    # 2) RSI features (momentum oscillator)
    logger.debug("Adding RSI features")
    out = add_rsi_features(
        out,
        src_col=price_col,
        periods=(14, 21, 30)
    )

    # 3) MACD features (histogram and derivative)
    logger.debug("Adding MACD features")
    out = add_macd_features(
        out,
        src_col=price_col,
        fast=12,
        slow=26,
        signal=9,
        derivative_ema_span=3
    )

    # 4) Enhanced multi-scale vol regime (single-stock only, no cross-sectional context)
    logger.debug("Adding volatility regime features")
    out = add_multiscale_vol_regime(
        out,
        ret_col=ret_col,
        short_windows=(10, 20),
        long_windows=(60, 100),
        z_window=60,
        ema_span=10,
        slope_win=20,
        cs_ratio_median=None,  # No cross-sectional context in single-stock mode
    )

    # 5) Distance-to-MA + z-scores
    logger.debug("Adding distance-to-MA features")
    out = add_distance_to_ma_features(
        out,
        src_col=price_col,
        ma_lengths=(20, 50, 100, 200),
        z_window=60
    )

    # 6) Range / breakout features (includes ATR14)
    logger.debug("Adding range/breakout features")
    out = add_range_breakout_features(out, win_list=(5, 10, 20))

    # 7) Volume features
    logger.debug("Adding volume features")
    out = add_volume_features(out)

    # 8) Volume shock + alignment features
    logger.debug("Adding volume shock features")
    out = add_volume_shock_features(
        out,
        vol_col=vol_col,
        price_col_primary="close",
        price_col_fallback=price_col,
        lookback=20,
        ema_span=10,
        prefix="volshock"
    )

    logger.debug("Single-stock feature computation completed")
    return out


def compute_single_stock_features_safe(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    **kwargs
) -> pd.DataFrame:
    """
    Safe wrapper for compute_single_stock_features with error handling.

    This version catches exceptions and returns the original DataFrame if
    feature computation fails, logging the error instead of raising.

    Args:
        df: Input DataFrame with OHLCV data
        symbol: Symbol name for logging (default: "UNKNOWN")
        **kwargs: Additional arguments passed to compute_single_stock_features

    Returns:
        DataFrame with features added, or original DataFrame if computation fails
    """
    try:
        return compute_single_stock_features(df, **kwargs)
    except Exception as e:
        logger.error(f"Feature computation failed for {symbol}: {e}")
        return df


if __name__ == "__main__":
    # Example usage for testing
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'adjclose': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Compute features
    result = compute_single_stock_features(sample_data)

    # Show what was added
    original_cols = set(sample_data.columns)
    new_cols = set(result.columns) - original_cols

    print(f"\nOriginal columns: {len(original_cols)}")
    print(f"New feature columns: {len(new_cols)}")
    print(f"Total columns: {len(result.columns)}")
    print(f"\nSample of new features: {sorted(list(new_cols))[:10]}")
