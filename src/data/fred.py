"""
FRED (Federal Reserve Economic Data) data loader and feature generator.

Fetches macroeconomic time series from FRED and creates features for equity prediction.

IMPORTANT: Publication lag handling
- Each series has a 'pub_lag_days' field indicating how many days after the reference
  date the data is actually published/available
- This is IN ADDITION to any explicit lag_days parameter
- Example: Jobless claims for week ending Saturday are released Thursday (5 days later)
- Treasury yields are typically available same day or next day

Requires FRED_API_KEY environment variable.
Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series definitions with publication lags and feature transformations
# pub_lag_days: days between reference date and when data is actually available
FRED_SERIES = {
    # Treasury Yields - available same day (after market close) or next morning
    # Conservative: use 1 day pub lag
    'DGS10': {
        'name': '10Y Treasury Yield',
        'category': 'yields',
        'frequency': 'daily',
        'pub_lag_days': 1,  # Available next day
        'transforms': ['level', 'change_5d', 'change_20d', 'zscore_60d', 'percentile_252d'],
    },
    'DGS2': {
        'name': '2Y Treasury Yield',
        'category': 'yields',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'change_5d', 'change_20d'],
    },
    'T10Y2Y': {
        'name': '10Y-2Y Yield Spread (Yield Curve)',
        'category': 'yields',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'change_5d', 'zscore_60d', 'percentile_252d'],
    },
    'T10Y3M': {
        'name': '10Y-3M Yield Spread',
        'category': 'yields',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'zscore_60d'],
    },

    # Credit Spreads - ICE BofA indices typically have 1-day lag
    'BAMLH0A0HYM2': {
        'name': 'High Yield OAS (ICE BofA)',
        'category': 'credit',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'change_5d', 'change_20d', 'zscore_60d', 'percentile_252d'],
    },
    'BAMLC0A4CBBB': {
        'name': 'BBB Corporate OAS',
        'category': 'credit',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'change_5d', 'zscore_60d'],
    },

    # Fed Policy - changes are announced immediately, but use 1 day for safety
    'DFEDTARU': {
        'name': 'Fed Funds Upper Target',
        'category': 'fed',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'change_20d'],
    },

    # Initial Jobless Claims - released Thursday 8:30am for week ending prior Saturday
    # Week ends Saturday, released Thursday = 5 days later
    # To use on Friday, need data from week ending 8 days ago
    'ICSA': {
        'name': 'Initial Jobless Claims',
        'category': 'labor',
        'frequency': 'weekly',
        'pub_lag_days': 5,  # Released Thursday for week ending Saturday
        'transforms': ['level', 'change_4w', 'zscore_52w', 'percentile_104w'],
    },
    'CCSA': {
        'name': 'Continued Claims',
        'category': 'labor',
        'frequency': 'weekly',
        'pub_lag_days': 12,  # Released with additional week lag vs initial claims
        'transforms': ['level', 'change_4w', 'zscore_52w'],
    },

    # Financial Conditions - released weekly with ~1 week lag
    'NFCI': {
        'name': 'Chicago Fed Financial Conditions Index',
        'category': 'conditions',
        'frequency': 'weekly',
        'pub_lag_days': 7,  # Released ~1 week after reference date
        'transforms': ['level', 'change_4w', 'zscore_52w'],
    },

    # VIX - available same day (real-time)
    # But FRED updates next day, so use 1 day lag
    'VIXCLS': {
        'name': 'VIX Close',
        'category': 'volatility',
        'frequency': 'daily',
        'pub_lag_days': 1,
        'transforms': ['level', 'change_5d', 'zscore_60d', 'percentile_252d'],
    },
}


def get_fred_api():
    """Get FRED API instance."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("fredapi not installed. Run: pip install fredapi")

    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable not set. "
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    return Fred(api_key=api_key)


def download_fred_series(
    series_ids: Optional[List[str]] = None,
    start_date: str = '2010-01-01',
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download FRED series and return as DataFrame.

    Args:
        series_ids: List of FRED series IDs (default: all defined series)
        start_date: Start date for data
        cache_path: Path to cache the data (optional)

    Returns:
        DataFrame with date index and series as columns
    """
    if series_ids is None:
        series_ids = list(FRED_SERIES.keys())

    # Check cache first
    if cache_path and cache_path.exists():
        logger.info(f"Loading FRED data from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        # Check if we need to update (data older than 1 day)
        if df.index.max() >= pd.Timestamp.now() - pd.Timedelta(days=2):
            return df
        logger.info("Cache is stale, refreshing...")

    fred = get_fred_api()

    all_series = {}
    for series_id in series_ids:
        try:
            logger.info(f"Downloading {series_id}: {FRED_SERIES.get(series_id, {}).get('name', 'Unknown')}")
            data = fred.get_series(series_id, observation_start=start_date)
            all_series[series_id] = data
        except Exception as e:
            logger.warning(f"Failed to download {series_id}: {e}")

    if not all_series:
        raise ValueError("No FRED series downloaded successfully")

    # Combine into DataFrame
    df = pd.DataFrame(all_series)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward-fill for weekly/monthly series (carry forward last value)
    df = df.ffill()

    # Cache the data
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info(f"Cached FRED data to {cache_path}")

    return df


def compute_fred_features(
    fred_df: pd.DataFrame,
    additional_lag_days: int = 0,
) -> pd.DataFrame:
    """
    Compute features from raw FRED series with proper publication lag handling.

    Each series is lagged by its pub_lag_days (when data becomes available)
    plus any additional_lag_days for extra safety margin.

    Args:
        fred_df: DataFrame with FRED series
        additional_lag_days: Extra days to lag beyond publication lag (default 0)

    Returns:
        DataFrame with computed features (properly lagged per series)
    """
    features = {}

    for series_id, config in FRED_SERIES.items():
        if series_id not in fred_df.columns:
            continue

        series = fred_df[series_id].copy()
        prefix = f"fred_{series_id.lower()}"
        transforms = config.get('transforms', ['level'])

        # Get publication lag for this series
        pub_lag = config.get('pub_lag_days', 1)
        total_lag = pub_lag + additional_lag_days

        # Compute all transforms for this series
        series_features = {}

        for transform in transforms:
            if transform == 'level':
                series_features[f"{prefix}"] = series

            elif transform == 'change_5d':
                series_features[f"{prefix}_chg5d"] = series.diff(5)

            elif transform == 'change_20d':
                series_features[f"{prefix}_chg20d"] = series.diff(20)

            elif transform == 'change_4w':
                # For weekly data (4 weeks of daily data = ~20 trading days)
                # But since weekly data is forward-filled, diff(20) on daily index
                series_features[f"{prefix}_chg4w"] = series.diff(20)

            elif transform == 'zscore_60d':
                mean = series.rolling(60, min_periods=20).mean()
                std = series.rolling(60, min_periods=20).std()
                series_features[f"{prefix}_z60"] = (series - mean) / std.replace(0, np.nan)

            elif transform == 'zscore_52w':
                # For weekly data - use ~252 trading days for 1 year
                mean = series.rolling(252, min_periods=60).mean()
                std = series.rolling(252, min_periods=60).std()
                series_features[f"{prefix}_z52w"] = (series - mean) / std.replace(0, np.nan)

            elif transform == 'percentile_252d':
                def pct_rank(x):
                    valid = x.dropna()
                    if len(valid) < 50:
                        return np.nan
                    return (valid.iloc[-1] > valid.iloc[:-1]).mean() * 100
                series_features[f"{prefix}_pct252"] = series.rolling(252, min_periods=50).apply(pct_rank, raw=False)

            elif transform == 'percentile_104w':
                # For weekly data - use ~504 trading days for 2 years
                def pct_rank(x):
                    valid = x.dropna()
                    if len(valid) < 100:
                        return np.nan
                    return (valid.iloc[-1] > valid.iloc[:-1]).mean() * 100
                series_features[f"{prefix}_pct104w"] = series.rolling(504, min_periods=100).apply(pct_rank, raw=False)

        # Apply publication lag to all features from this series
        for feat_name, feat_series in series_features.items():
            features[feat_name] = feat_series.shift(total_lag)

        logger.debug(f"  {series_id}: {len(series_features)} features, lag={total_lag}d (pub={pub_lag}d + extra={additional_lag_days}d)")

    # Combine into DataFrame
    features_df = pd.DataFrame(features)

    # Convert to float32 for memory efficiency
    for col in features_df.columns:
        features_df[col] = features_df[col].astype('float32')

    logger.info(f"Computed {len(features_df.columns)} FRED features with publication lag handling")

    return features_df


def add_fred_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    fred_df: Optional[pd.DataFrame] = None,
    cache_path: Optional[Path] = None,
    additional_lag_days: int = 0,
) -> None:
    """
    Add FRED features to all symbols with proper publication lag handling.

    Each series is automatically lagged by its publication delay (when data
    becomes publicly available). Additional lag can be added for extra safety.

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        fred_df: Pre-loaded FRED data (optional, will download if not provided)
        cache_path: Path to cache FRED data
        additional_lag_days: Extra days to lag beyond publication lag (default 0)
    """
    logger.info(f"Adding FRED macro features (additional_lag={additional_lag_days} days)")

    # Download or load FRED data
    if fred_df is None:
        try:
            fred_df = download_fred_series(cache_path=cache_path)
        except Exception as e:
            logger.error(f"Failed to load FRED data: {e}")
            return

    # Compute features with publication lag
    fred_features = compute_fred_features(fred_df, additional_lag_days=additional_lag_days)

    # Attach to each symbol
    attached = 0
    for sym, df in indicators_by_symbol.items():
        if df.empty:
            continue

        # Reindex FRED features to symbol's dates
        symbol_fred = fred_features.reindex(df.index)

        # Concatenate
        for col in symbol_fred.columns:
            df[col] = symbol_fred[col].values

        attached += 1

    logger.info(f"FRED features attached to {attached} symbols")


def get_fred_feature_names() -> List[str]:
    """Get list of all FRED feature names that will be generated."""
    names = []
    for series_id, config in FRED_SERIES.items():
        prefix = f"fred_{series_id.lower()}"
        transforms = config.get('transforms', ['level'])

        for transform in transforms:
            if transform == 'level':
                names.append(prefix)
            elif transform == 'change_5d':
                names.append(f"{prefix}_chg5d")
            elif transform == 'change_20d':
                names.append(f"{prefix}_chg20d")
            elif transform == 'change_4w':
                names.append(f"{prefix}_chg4w")
            elif transform == 'zscore_60d':
                names.append(f"{prefix}_z60")
            elif transform == 'zscore_52w':
                names.append(f"{prefix}_z52w")
            elif transform == 'percentile_252d':
                names.append(f"{prefix}_pct252")
            elif transform == 'percentile_104w':
                names.append(f"{prefix}_pct104w")

    return names
