"""
Factor spread features for regime/momentum analysis.

Computes market-level features from factor spreads:
- QQQ, SPY absolute cumulative returns and momentum
- QQQ-SPY spread (growth vs value)
- RSP-SPY spread (equal-weight vs cap-weight / breadth)

These are cross-sectional features (same for all symbols), computed once
and broadcast to all stocks. Per-symbol bestmatch-SPY spread is handled
separately in factor_regression.py.
"""
import logging
import warnings
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global spreads (same for all symbols)
GLOBAL_SPREADS = {
    'qqq': ('QQQ', None),           # Absolute QQQ returns
    'spy': ('SPY', None),           # Absolute SPY returns
    'qqq_spy': ('QQQ', 'SPY'),      # QQQ - SPY (growth premium)
    'rsp_spy': ('RSP', 'SPY'),      # RSP - SPY (breadth spread)
}

DAILY_WINDOWS = (20, 60, 120)
WEEKLY_WINDOWS = (4, 12, 24)  # ~20, 60, 120 days


def _get_return_series(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    symbol: str
) -> Optional[pd.Series]:
    """Extract return series from indicators dictionary."""
    if symbol not in indicators_by_symbol:
        return None

    df = indicators_by_symbol[symbol]
    if 'ret' not in df.columns:
        return None

    return pd.to_numeric(df['ret'], errors='coerce')


def _rolling_slope(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Compute rolling linear regression slope.

    Args:
        series: Input series (typically returns)
        window: Rolling window size
        min_periods: Minimum valid periods (default: window // 2)

    Returns:
        Series of rolling slopes
    """
    if min_periods is None:
        # Ensure min_periods is at least 3 but never exceeds window
        min_periods = min(window, max(3, window // 2))

    def slope_func(arr):
        y = arr
        valid = ~np.isnan(y)
        n_valid = valid.sum()
        if n_valid < min_periods:
            return np.nan
        x = np.arange(len(y))
        x_v, y_v = x[valid], y[valid]
        # Simple OLS: slope = Cov(x,y) / Var(x)
        x_mean = x_v.mean()
        y_mean = y_v.mean()
        cov_xy = ((x_v - x_mean) * (y_v - y_mean)).sum()
        var_x = ((x_v - x_mean) ** 2).sum()
        if var_x == 0:
            return np.nan
        return cov_xy / var_x

    return series.rolling(window, min_periods=min_periods).apply(slope_func, raw=True)


def _compute_spread_metrics(
    spread: pd.Series,
    name: str,
    windows: Tuple[int, ...],
    prefix: str = ""
) -> Dict[str, pd.Series]:
    """Compute all metrics for a single spread.

    Args:
        spread: Return spread series (e.g., QQQ - SPY daily returns)
        name: Base name for features
        windows: Windows for rolling computations
        prefix: Prefix for feature names (e.g., 'w_' for weekly)

    Returns:
        Dict of feature_name -> pd.Series
    """
    features = {}

    # Cumulative return over horizons (sum of returns)
    for w in windows:
        min_periods = max(3, w // 3)
        features[f'{prefix}{name}_cumret_{w}'] = spread.rolling(
            w, min_periods=min_periods
        ).sum().astype('float32')

    # Z-score (use middle window: 60d daily, 12w weekly)
    zscore_win = windows[1]
    min_periods_z = max(10, zscore_win // 2)
    mean = spread.rolling(zscore_win, min_periods=min_periods_z).mean()
    std = spread.rolling(zscore_win, min_periods=min_periods_z).std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zscore = (spread - mean) / std.replace(0, np.nan)
    features[f'{prefix}{name}_zscore_{zscore_win}'] = zscore.astype('float32')

    # Slope (momentum direction) for first two windows
    for w in windows[:2]:
        features[f'{prefix}{name}_slope_{w}'] = _rolling_slope(spread, w).astype('float32')

    return features


def compute_global_spread_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    windows: Tuple[int, ...] = DAILY_WINDOWS,
    prefix: str = ""
) -> Dict[str, pd.Series]:
    """Compute global spread features at market level.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame
        windows: Rolling windows
        prefix: Feature name prefix

    Returns:
        Dict of feature_name -> pd.Series to be broadcast to all symbols
    """
    features = {}
    missing = []

    for spread_name, (sym1, sym2) in GLOBAL_SPREADS.items():
        ret1 = _get_return_series(indicators_by_symbol, sym1)
        if ret1 is None:
            missing.append(sym1)
            continue

        if sym2 is None:
            # Absolute return (QQQ or SPY)
            spread = ret1
        else:
            ret2 = _get_return_series(indicators_by_symbol, sym2)
            if ret2 is None:
                missing.append(sym2)
                continue
            # Spread return: sym1 - sym2
            spread = ret1 - ret2.reindex(ret1.index)

        # Compute metrics for this spread
        spread_features = _compute_spread_metrics(spread, spread_name, windows, prefix)
        features.update(spread_features)

        logger.debug(f"  + {spread_name} spread: {len(spread_features)} features")

    if missing:
        logger.warning(f"Missing symbols for spread features: {set(missing)}")

    return features


def add_spread_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lag_days: int = 1,
) -> None:
    """Add daily spread features to all symbols.

    Computes global spread features and broadcasts to all symbols.
    Features are lagged by lag_days to prevent look-ahead bias.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame (modified in place)
        lag_days: Number of days to lag features (default 1)
    """
    logger.info(f"Computing daily spread features (lag={lag_days})")

    # Compute features at market level
    features = compute_global_spread_features(
        indicators_by_symbol,
        windows=DAILY_WINDOWS,
        prefix=""
    )

    if not features:
        logger.warning("No spread features computed")
        return

    # Broadcast to all symbols
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        new_cols = {}

        # Get date index
        if isinstance(df.index, pd.DatetimeIndex):
            date_index = df.index
        elif 'date' in df.columns:
            date_index = pd.DatetimeIndex(pd.to_datetime(df['date']))
        else:
            continue

        for name, series in features.items():
            if len(series) > 0:
                # Reindex to symbol's dates, apply lag
                reindexed = series.reindex(date_index)
                if lag_days > 0:
                    lagged = reindexed.shift(lag_days)
                else:
                    lagged = reindexed
                new_cols[name] = lagged.astype('float32')

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_df], axis=1)
        attached_count += 1

    logger.info(f"Daily spread features ({len(features)} total) attached to {attached_count} symbols")


def add_weekly_spread_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    prefix: str = "w_",
) -> None:
    """Add weekly spread features to all symbols.

    Resamples returns to weekly, computes spread metrics, then merges back to daily.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame (modified in place)
        prefix: Prefix for weekly feature names (default 'w_')
    """
    logger.info("Computing weekly spread features")

    # First, resample returns to weekly for each required symbol
    required_symbols = set()
    for sym1, sym2 in GLOBAL_SPREADS.values():
        required_symbols.add(sym1)
        if sym2:
            required_symbols.add(sym2)

    weekly_returns = {}
    for sym in required_symbols:
        ret = _get_return_series(indicators_by_symbol, sym)
        if ret is None:
            continue
        # Resample to weekly (Friday end-of-week)
        weekly_ret = ret.resample('W-FRI').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
        )
        weekly_returns[sym] = weekly_ret

    if len(weekly_returns) < 2:
        logger.warning("Insufficient data for weekly spread features")
        return

    # Compute weekly spread features
    features = {}
    for spread_name, (sym1, sym2) in GLOBAL_SPREADS.items():
        if sym1 not in weekly_returns:
            continue

        w_ret1 = weekly_returns[sym1]

        if sym2 is None:
            spread = w_ret1
        else:
            if sym2 not in weekly_returns:
                continue
            w_ret2 = weekly_returns[sym2]
            spread = w_ret1 - w_ret2.reindex(w_ret1.index)

        spread_features = _compute_spread_metrics(spread, spread_name, WEEKLY_WINDOWS, prefix)
        features.update(spread_features)

        logger.debug(f"  + {prefix}{spread_name}: {len(spread_features)} weekly features")

    if not features:
        logger.warning("No weekly spread features computed")
        return

    # Broadcast to all symbols with forward-fill from weekly to daily
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        new_cols = {}

        if isinstance(df.index, pd.DatetimeIndex):
            daily_idx = df.index
        elif 'date' in df.columns:
            daily_idx = pd.DatetimeIndex(pd.to_datetime(df['date']))
        else:
            continue

        for name, series in features.items():
            if len(series) > 0:
                # Union indices, forward-fill, then select daily dates
                combined_idx = series.index.union(daily_idx).sort_values()
                filled = series.reindex(combined_idx).ffill()
                new_cols[name] = filled.reindex(daily_idx).astype('float32')

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_df], axis=1)
        attached_count += 1

    logger.info(f"Weekly spread features ({len(features)} total) attached to {attached_count} symbols")


def compute_bestmatch_spread_features(
    stock_ret_index: pd.Index,
    bestmatch_ret: pd.Series,
    spy_ret: pd.Series,
    windows: Tuple[int, ...] = DAILY_WINDOWS,
    prefix: str = ""
) -> Dict[str, pd.Series]:
    """Compute bestmatch-SPY spread features for a single symbol.

    This is called per-symbol since each stock has its own best-match ETF.

    Args:
        stock_ret_index: Date index to align results to
        bestmatch_ret: Best-match ETF return series
        spy_ret: SPY return series
        windows: Rolling windows
        prefix: Feature name prefix

    Returns:
        Dict of feature_name -> pd.Series
    """
    # Align returns
    bestmatch_aligned = bestmatch_ret.reindex(stock_ret_index)
    spy_aligned = spy_ret.reindex(stock_ret_index)

    # Compute spread
    spread = bestmatch_aligned - spy_aligned

    # Compute metrics
    return _compute_spread_metrics(spread, 'bestmatch_spy', windows, prefix)


def compute_bestmatch_ew_spread_features(
    stock_ret_index: pd.Index,
    bestmatch_ew_ret: pd.Series,
    rsp_ret: pd.Series,
    windows: Tuple[int, ...] = DAILY_WINDOWS,
    prefix: str = ""
) -> Dict[str, pd.Series]:
    """Compute equal-weight bestmatch-RSP spread features for a single symbol.

    Parallel to compute_bestmatch_spread_features but in equal-weight space:
    - bestmatch-SPY: cap-weighted sector vs cap-weighted market
    - bestmatch_ew-RSP: equal-weight sector vs equal-weight market

    This captures relative performance of a stock's sector in equal-weight terms,
    removing large-cap concentration effects.

    Args:
        stock_ret_index: Date index to align results to
        bestmatch_ew_ret: Best-match equal-weight ETF return series (e.g., RSPT)
        rsp_ret: RSP (S&P 500 Equal Weight) return series
        windows: Rolling windows
        prefix: Feature name prefix

    Returns:
        Dict of feature_name -> pd.Series
    """
    # Align returns
    bestmatch_aligned = bestmatch_ew_ret.reindex(stock_ret_index)
    rsp_aligned = rsp_ret.reindex(stock_ret_index)

    # Compute spread (equal-weight sector - equal-weight market)
    spread = bestmatch_aligned - rsp_aligned

    # Compute metrics with distinct name
    return _compute_spread_metrics(spread, 'bestmatch_ew_rsp', windows, prefix)


def compute_bestmatch_spread_features_weekly(
    daily_index: pd.Index,
    bestmatch_ret_daily: pd.Series,
    spy_ret_daily: pd.Series,
    windows: Tuple[int, ...] = WEEKLY_WINDOWS,
    prefix: str = "w_"
) -> Dict[str, pd.Series]:
    """Compute weekly bestmatch-SPY spread features for a single symbol.

    Args:
        daily_index: Daily date index for output alignment
        bestmatch_ret_daily: Best-match ETF daily returns
        spy_ret_daily: SPY daily returns
        windows: Weekly rolling windows
        prefix: Feature name prefix

    Returns:
        Dict of feature_name -> pd.Series (aligned to daily_index)
    """
    # Resample to weekly
    def resample_to_weekly(ret):
        return ret.resample('W-FRI').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
        )

    w_bestmatch = resample_to_weekly(bestmatch_ret_daily)
    w_spy = resample_to_weekly(spy_ret_daily)

    # Compute spread
    spread = w_bestmatch - w_spy.reindex(w_bestmatch.index)

    # Compute metrics on weekly data
    weekly_features = _compute_spread_metrics(spread, 'bestmatch_spy', windows, prefix)

    # Map back to daily index
    daily_features = {}
    for name, series in weekly_features.items():
        combined_idx = series.index.union(daily_index).sort_values()
        filled = series.reindex(combined_idx).ffill()
        daily_features[name] = filled.reindex(daily_index).astype('float32')

    return daily_features
