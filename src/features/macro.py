"""
Macro volatility regime features based on VIX and VXN indices.

This module computes features that capture the overall market volatility regime,
which can help ML models understand whether the market is in a fear/greed state.

All features are lagged by 1 day by default to prevent look-ahead bias
(VIX closes at 4:15pm ET while stocks close at 4:00pm ET).
"""
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_vix_series(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    vix_symbol: str = "^VIX"
) -> Optional[pd.Series]:
    """
    Extract VIX close series from indicators dictionary with proper handling
    of missing/zero values.

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames
        vix_symbol: Symbol for VIX index (default ^VIX)

    Returns:
        VIX close price series with zeros replaced and forward-filled, or None if not available
    """
    if vix_symbol not in indicators_by_symbol:
        # Try alternative symbol formats
        alt_symbols = ["VIX", "^VIX", "$VIX", "CBOE:VIX"]
        for alt in alt_symbols:
            if alt in indicators_by_symbol:
                vix_symbol = alt
                break
        else:
            return None

    df = indicators_by_symbol[vix_symbol]
    price_col = "close" if "close" in df.columns else ("adjclose" if "adjclose" in df.columns else None)
    if price_col is None:
        return None

    vix = pd.to_numeric(df[price_col], errors='coerce')

    # Replace zeros with NaN (VIX can't be 0 - these are missing values from holidays)
    # VIX is typically in the 10-80 range, so anything <= 0 is clearly invalid
    vix = vix.replace(0, np.nan)
    vix = vix.where(vix > 0, np.nan)

    # Forward-fill missing values (carry forward last known VIX value)
    # This is appropriate for VIX since it represents market conditions that persist
    vix = vix.ffill()

    return vix


def add_vix_regime_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    vix_symbol: str = "^VIX",
    vxn_symbol: str = "^VXN",
    lag_days: int = 1
) -> None:
    """
    Add VIX-based volatility regime features to all symbols.

    IMPORTANT: All features are lagged by `lag_days` (default 1) to prevent
    look-ahead bias. VIX closes at 4:15pm ET while stocks close at 4:00pm ET,
    so using same-day VIX values would be look-ahead bias.

    Features added (all lagged by lag_days):
    - vix_level: Raw VIX level (useful for regime classification)
    - vix_percentile_252d: 252-day percentile rank (0-100)
    - vix_zscore_60d: 60-day z-score of VIX
    - vix_regime: Categorical regime (0=low, 1=normal, 2=elevated, 3=crisis)
    - vix_ma20_ratio: VIX / 20-day MA ratio (mean reversion signal)
    - vix_change_5d: 5-day change in VIX (fear spike indicator)
    - vix_change_20d: 20-day change in VIX (trend in fear)
    - vix_ema10: 10-day EMA of VIX (smoothed level)
    - vxn_level: Raw VXN level (Nasdaq implied vol)
    - vxn_percentile_252d: 252-day percentile rank for VXN
    - vix_vxn_spread: VIX - VXN spread (equity vs tech fear)
    - vix_vxn_ratio: VIX/VXN ratio

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        vix_symbol: Symbol for VIX index
        vxn_symbol: Symbol for VXN index
        lag_days: Number of days to lag features (default 1 to avoid look-ahead bias)
    """
    logger.info(f"Computing VIX regime features (lag={lag_days} days)")

    # Get VIX series
    vix = _get_vix_series(indicators_by_symbol, vix_symbol)
    if vix is None or vix.isna().all():
        logger.warning(f"VIX data not available ({vix_symbol}). Skipping VIX features.")
        return

    logger.info(f"VIX data found: {len(vix.dropna())} valid observations")

    # Get VXN series (optional)
    vxn = _get_vix_series(indicators_by_symbol, vxn_symbol)
    has_vxn = vxn is not None and not vxn.isna().all()
    if has_vxn:
        logger.info(f"VXN data found: {len(vxn.dropna())} valid observations")
    else:
        logger.info("VXN data not available, skipping VXN features")

    # Compute VIX features
    features = {}

    # Raw level
    features['vix_level'] = vix.rename('vix_level')

    # 252-day rolling percentile (handles NaN gracefully)
    def rolling_percentile(s, window=252):
        """Compute rolling percentile rank."""
        def pct_rank(x):
            valid = x.dropna()
            if len(valid) < 10:
                return np.nan
            return (valid.iloc[-1] > valid.iloc[:-1]).mean() * 100
        return s.rolling(window, min_periods=50).apply(pct_rank, raw=False)

    features['vix_percentile_252d'] = rolling_percentile(vix).rename('vix_percentile_252d')

    # 60-day z-score
    vix_mean = vix.rolling(60, min_periods=20).mean()
    vix_std = vix.rolling(60, min_periods=20).std()
    features['vix_zscore_60d'] = ((vix - vix_mean) / vix_std.replace(0, np.nan)).rename('vix_zscore_60d')

    # VIX regime classification
    # Low: < 15, Normal: 15-20, Elevated: 20-30, Crisis: > 30
    def classify_regime(x):
        if pd.isna(x):
            return np.nan
        if x < 15:
            return 0  # Low volatility
        elif x < 20:
            return 1  # Normal
        elif x < 30:
            return 2  # Elevated
        else:
            return 3  # Crisis

    features['vix_regime'] = vix.apply(classify_regime).rename('vix_regime')

    # VIX / MA20 ratio (mean reversion indicator)
    vix_ma20 = vix.rolling(20, min_periods=10).mean()
    features['vix_ma20_ratio'] = (vix / vix_ma20.replace(0, np.nan)).rename('vix_ma20_ratio')

    # VIX changes (fear spike detection)
    features['vix_change_5d'] = vix.pct_change(5, fill_method=None).rename('vix_change_5d')
    features['vix_change_20d'] = vix.pct_change(20, fill_method=None).rename('vix_change_20d')

    # EMA for smoother signals
    features['vix_ema10'] = vix.ewm(span=10, min_periods=5).mean().rename('vix_ema10')

    # VIX term structure (vs VXN) if available
    if has_vxn:
        features['vxn_level'] = vxn.rename('vxn_level')
        features['vxn_percentile_252d'] = rolling_percentile(vxn).rename('vxn_percentile_252d')

        # VIX - VXN spread (negative = tech more fearful than broad market)
        features['vix_vxn_spread'] = (vix - vxn).rename('vix_vxn_spread')

        # VIX/VXN ratio
        features['vix_vxn_ratio'] = (vix / vxn.replace(0, np.nan)).rename('vix_vxn_ratio')

    # Attach features to ALL symbols with lag to prevent look-ahead bias
    # Use pd.concat to batch add all features at once (avoids DataFrame fragmentation)
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        new_cols = {}

        # Handle both DatetimeIndex and date column cases
        if isinstance(df.index, pd.DatetimeIndex):
            date_index = df.index
            use_date_col = False
        elif 'date' in df.columns:
            date_index = pd.DatetimeIndex(pd.to_datetime(df['date']))
            use_date_col = True
        else:
            # Skip if no valid date reference
            continue

        for name, series in features.items():
            if len(series) > 0:
                # Reindex to symbol's date range, apply lag, and convert to float32
                reindexed = series.reindex(date_index)
                # Apply lag: shift forward so today's row gets yesterday's VIX value
                if lag_days > 0:
                    lagged = reindexed.shift(lag_days)
                else:
                    lagged = reindexed
                # Reset index to match original df if using date column
                if use_date_col:
                    lagged = lagged.values
                new_cols[name] = pd.to_numeric(lagged, errors='coerce').astype('float32')
        # Batch add all new columns at once using pd.concat
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_df], axis=1)
        attached_count += 1

    logger.info(f"VIX features ({len(features)} total) attached to {attached_count} symbols (lag={lag_days}d)")


def add_volatility_term_structure(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    vxx_symbol: str = "VXX",
    vixy_symbol: str = "VIXY",
    lag_days: int = 1
) -> None:
    """
    Add volatility term structure features using VXX/VIXY ETPs.

    These ETPs track VIX futures, and their behavior relative to spot VIX
    provides insights into the volatility term structure (contango/backwardation).

    IMPORTANT: All features are lagged by `lag_days` (default 1) to prevent
    look-ahead bias.

    Features added (all lagged by lag_days):
    - vxx_ret_5d: 5-day VXX return (volatility trend proxy)
    - vxx_ret_20d: 20-day VXX return
    - vxx_rsi_14: RSI of VXX (fear mean reversion)

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        vxx_symbol: Symbol for VXX
        vixy_symbol: Symbol for VIXY (alternative)
        lag_days: Number of days to lag features (default 1 to avoid look-ahead bias)
    """
    # Try VXX first, then VIXY
    vol_sym = None
    for sym in [vxx_symbol, vixy_symbol]:
        if sym in indicators_by_symbol:
            vol_sym = sym
            break

    if vol_sym is None:
        logger.info("No VXX/VIXY data available, skipping term structure features")
        return

    logger.info(f"Computing volatility term structure features from {vol_sym}")

    df = indicators_by_symbol[vol_sym]
    price_col = "close" if "close" in df.columns else ("adjclose" if "adjclose" in df.columns else None)
    if price_col is None:
        return

    px = pd.to_numeric(df[price_col], errors='coerce')

    features = {}

    # Returns at different horizons
    features['vxx_ret_5d'] = px.pct_change(5, fill_method=None).rename('vxx_ret_5d')
    features['vxx_ret_20d'] = px.pct_change(20, fill_method=None).rename('vxx_ret_20d')

    # RSI of VXX
    delta = px.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    features['vxx_rsi_14'] = (100 - (100 / (1 + rs))).rename('vxx_rsi_14')

    # Attach to all symbols with lag to prevent look-ahead bias
    # Use pd.concat to batch add all features at once (avoids DataFrame fragmentation)
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        new_cols = {}

        # Handle both DatetimeIndex and date column cases
        if isinstance(df.index, pd.DatetimeIndex):
            date_index = df.index
            use_date_col = False
        elif 'date' in df.columns:
            date_index = pd.DatetimeIndex(pd.to_datetime(df['date']))
            use_date_col = True
        else:
            continue

        for name, series in features.items():
            if len(series) > 0:
                reindexed = series.reindex(date_index)
                # Apply lag: shift forward so today's row gets yesterday's value
                if lag_days > 0:
                    lagged = reindexed.shift(lag_days)
                else:
                    lagged = reindexed
                if use_date_col:
                    lagged = lagged.values
                new_cols[name] = pd.to_numeric(lagged, errors='coerce').astype('float32')
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_df], axis=1)
        attached_count += 1

    logger.info(f"VXX term structure features ({len(features)}) attached to {attached_count} symbols (lag={lag_days}d)")


def add_weekly_vix_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    vix_symbol: str = "^VIX",
    vxn_symbol: str = "^VXN",
    lag_days: int = 1
) -> None:
    """
    Add weekly VIX-based volatility regime features to all symbols.

    This computes VIX features on weekly-resampled data, providing a longer-term
    view of volatility regime. Features are merged back to daily with 'w_' prefix.

    IMPORTANT: All features are lagged by `lag_days` (default 1) after forward-fill
    to prevent look-ahead bias.

    Weekly features added (all prefixed with 'w_'):
    - w_vix_level: Weekly closing VIX level
    - w_vix_percentile_52w: 52-week percentile rank (0-100)
    - w_vix_zscore_12w: 12-week z-score of VIX
    - w_vix_regime: Categorical regime on weekly basis
    - w_vix_ma4_ratio: VIX / 4-week MA ratio
    - w_vix_change_1w: 1-week change in VIX
    - w_vix_change_4w: 4-week change in VIX
    - w_vix_ema4: 4-week EMA of VIX
    - w_vxn_level: Weekly closing VXN level
    - w_vix_vxn_spread: Weekly VIX - VXN spread
    - w_vix_vxn_ratio: Weekly VIX/VXN ratio

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        vix_symbol: Symbol for VIX index
        vxn_symbol: Symbol for VXN index
        lag_days: Number of days to lag features after ffill (default 1)
    """
    logger.info(f"Computing weekly VIX regime features (lag={lag_days} days)")

    # Get VIX series
    vix = _get_vix_series(indicators_by_symbol, vix_symbol)
    if vix is None or vix.isna().all():
        logger.warning(f"VIX data not available ({vix_symbol}). Skipping weekly VIX features.")
        return

    # Get VXN series (optional)
    vxn = _get_vix_series(indicators_by_symbol, vxn_symbol)
    has_vxn = vxn is not None and not vxn.isna().all()

    # Resample VIX to weekly (using Friday close, standard market convention)
    vix_weekly = vix.resample('W-FRI').last()
    logger.info(f"Resampled VIX to weekly: {len(vix_weekly.dropna())} weeks")

    if has_vxn:
        vxn_weekly = vxn.resample('W-FRI').last()

    # Compute weekly features
    weekly_features = {}

    # Raw weekly level
    weekly_features['w_vix_level'] = vix_weekly.rename('w_vix_level')

    # 52-week rolling percentile
    def rolling_percentile(s, window=52):
        """Compute rolling percentile rank."""
        def pct_rank(x):
            valid = x.dropna()
            if len(valid) < 10:
                return np.nan
            return (valid.iloc[-1] > valid.iloc[:-1]).mean() * 100
        return s.rolling(window, min_periods=12).apply(pct_rank, raw=False)

    weekly_features['w_vix_percentile_52w'] = rolling_percentile(vix_weekly, 52).rename('w_vix_percentile_52w')

    # 12-week z-score
    vix_mean_12w = vix_weekly.rolling(12, min_periods=4).mean()
    vix_std_12w = vix_weekly.rolling(12, min_periods=4).std()
    weekly_features['w_vix_zscore_12w'] = ((vix_weekly - vix_mean_12w) / vix_std_12w.replace(0, np.nan)).rename('w_vix_zscore_12w')

    # Weekly VIX regime classification (same thresholds as daily)
    def classify_regime(x):
        if pd.isna(x):
            return np.nan
        if x < 15:
            return 0  # Low volatility
        elif x < 20:
            return 1  # Normal
        elif x < 30:
            return 2  # Elevated
        else:
            return 3  # Crisis

    weekly_features['w_vix_regime'] = vix_weekly.apply(classify_regime).rename('w_vix_regime')

    # VIX / 4-week MA ratio
    vix_ma4 = vix_weekly.rolling(4, min_periods=2).mean()
    weekly_features['w_vix_ma4_ratio'] = (vix_weekly / vix_ma4.replace(0, np.nan)).rename('w_vix_ma4_ratio')

    # Weekly changes
    weekly_features['w_vix_change_1w'] = vix_weekly.pct_change(1, fill_method=None).rename('w_vix_change_1w')
    weekly_features['w_vix_change_4w'] = vix_weekly.pct_change(4, fill_method=None).rename('w_vix_change_4w')

    # 4-week EMA
    weekly_features['w_vix_ema4'] = vix_weekly.ewm(span=4, min_periods=2).mean().rename('w_vix_ema4')

    # VXN weekly features if available
    if has_vxn:
        weekly_features['w_vxn_level'] = vxn_weekly.rename('w_vxn_level')
        weekly_features['w_vix_vxn_spread'] = (vix_weekly - vxn_weekly).rename('w_vix_vxn_spread')
        weekly_features['w_vix_vxn_ratio'] = (vix_weekly / vxn_weekly.replace(0, np.nan)).rename('w_vix_vxn_ratio')

    # Merge weekly features back to daily using forward-fill then apply lag
    # Each daily row gets the value from the most recent completed week, then lagged
    # Use pd.concat to batch add all features at once (avoids DataFrame fragmentation)
    attached_count = 0
    for sym, df in indicators_by_symbol.items():
        new_cols = {}

        # Handle both DatetimeIndex and date column cases
        if isinstance(df.index, pd.DatetimeIndex):
            date_index = df.index
            use_date_col = False
        elif 'date' in df.columns:
            date_index = pd.DatetimeIndex(pd.to_datetime(df['date']))
            use_date_col = True
        else:
            # Skip if no valid date reference
            continue

        for name, weekly_series in weekly_features.items():
            if len(weekly_series) > 0:
                # Reindex weekly to daily dates using ffill (forward fill from last week)
                daily_values = weekly_series.reindex(date_index, method='ffill')
                # Apply additional lag to prevent look-ahead bias
                if lag_days > 0:
                    daily_values = daily_values.shift(lag_days)
                # Reset index to match original df
                if use_date_col:
                    daily_values = daily_values.values
                new_cols[name] = pd.to_numeric(daily_values, errors='coerce').astype('float32')
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_df], axis=1)
        attached_count += 1

    logger.info(f"Weekly VIX features ({len(weekly_features)} total) attached to {attached_count} symbols (lag={lag_days}d)")
