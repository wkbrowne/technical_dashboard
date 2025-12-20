"""
Divergence features.

Divergence occurs when price and an indicator move in opposite directions,
often signaling potential reversals. This module computes simplified divergence
signals without requiring swing high/low detection.

Approach:
- Compare direction of change (not absolute levels) over N periods
- Positive divergence: price down but indicator up (bullish)
- Negative divergence: price up but indicator down (bearish)

Features:
- RSI-price divergence: momentum vs price disagreement
- MACD-price divergence: trend momentum vs price
- Trend-momentum divergence: trend score vs RSI disagreement
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _direction_divergence(
    series1: pd.Series,
    series2: pd.Series,
    window: int
) -> pd.Series:
    """
    Compute divergence between two series based on direction of change.

    Returns:
        Divergence score in [-1, 1]:
        - Positive: series1 up, series2 down (or vice versa)
        - Negative: both moving same direction
        - Near zero: no clear divergence

    The sign indicates: series1_direction - series2_direction
    So if price (series1) is up but RSI (series2) is down, result is positive.
    """
    # Direction of change over window
    change1 = series1.diff(window)
    change2 = series2.diff(window)

    # Normalize to [-1, 1] using tanh of z-score
    def normalize_direction(change: pd.Series) -> pd.Series:
        std = change.rolling(60, min_periods=20).std()
        z = change / std.clip(lower=1e-8)
        return np.tanh(z)  # Squash to [-1, 1]

    dir1 = normalize_direction(change1)
    dir2 = normalize_direction(change2)

    # Divergence = difference in directions
    # Positive when series1 going up and series2 going down (or vice versa)
    divergence = dir1 - dir2

    return divergence.astype('float32')


def _cumulative_divergence(
    price: pd.Series,
    indicator: pd.Series,
    window: int,
    smooth: int = 5
) -> pd.Series:
    """
    Compute cumulative divergence signal.

    Counts instances where price and indicator move opposite directions,
    smoothed with EMA.
    """
    # Daily direction
    price_up = (price.diff() > 0).astype(float)
    ind_up = (indicator.diff() > 0).astype(float)

    # Divergence: 1 when opposite, -1 when same, 0 when either is flat
    daily_div = (price_up - ind_up)  # +1 if price up & ind down, -1 if price down & ind up

    # Rolling sum and smooth
    cum_div = daily_div.rolling(window, min_periods=window//2).sum()
    smoothed = cum_div.ewm(span=smooth, adjust=False).mean()

    # Normalize by window size
    normalized = smoothed / window

    return normalized.astype('float32')


def add_divergence_features(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """
    Add divergence features between price and momentum indicators.

    Requires these columns to already exist:
    - close or adjclose (price)
    - rsi_14 (RSI indicator)
    - macd_histogram (MACD histogram)
    - trend_score_sign (trend direction)

    Features computed:
    - rsi_price_div_{w}d: RSI vs price divergence
    - macd_price_div_{w}d: MACD histogram vs price divergence
    - trend_rsi_div_{w}d: Trend score vs RSI divergence
    - rsi_price_div_cum_{w}d: Cumulative RSI-price divergence

    Args:
        df: DataFrame with price and indicator columns
        windows: Lookback windows for divergence calculation

    Returns:
        DataFrame with divergence features added (mutates in-place)
    """
    # Get price
    if 'close' in df.columns:
        price = pd.to_numeric(df['close'], errors='coerce')
    elif 'adjclose' in df.columns:
        price = pd.to_numeric(df['adjclose'], errors='coerce')
    else:
        logger.warning("No price column found; skipping divergence features")
        return df

    new_cols = {}

    # --- RSI-Price Divergence ---
    if 'rsi_14' in df.columns:
        rsi = pd.to_numeric(df['rsi_14'], errors='coerce')

        for w in windows:
            # Direction-based divergence
            div = _direction_divergence(price, rsi, w)
            new_cols[f'rsi_price_div_{w}d'] = div

            # Cumulative divergence (how persistent is the divergence)
            cum_div = _cumulative_divergence(price, rsi, w)
            new_cols[f'rsi_price_div_cum_{w}d'] = cum_div
    else:
        logger.debug("rsi_14 not found; skipping RSI divergence features")

    # --- MACD-Price Divergence ---
    if 'macd_histogram' in df.columns:
        macd = pd.to_numeric(df['macd_histogram'], errors='coerce')

        for w in windows:
            div = _direction_divergence(price, macd, w)
            new_cols[f'macd_price_div_{w}d'] = div
    else:
        logger.debug("macd_histogram not found; skipping MACD divergence features")

    # --- Trend-Momentum Divergence ---
    # When trend is positive but RSI is falling (or vice versa)
    if 'trend_score_sign' in df.columns and 'rsi_14' in df.columns:
        trend = pd.to_numeric(df['trend_score_sign'], errors='coerce')
        rsi = pd.to_numeric(df['rsi_14'], errors='coerce')

        for w in windows:
            div = _direction_divergence(trend, rsi, w)
            new_cols[f'trend_rsi_div_{w}d'] = div
    else:
        logger.debug("trend_score_sign or rsi_14 not found; skipping trend-RSI divergence")

    # --- Volatility-Trend Divergence ---
    # Volatility expanding while trend weakening (potential reversal)
    if 'vol_regime_ema10' in df.columns and 'trend_score_sign' in df.columns:
        vol = pd.to_numeric(df['vol_regime_ema10'], errors='coerce')
        trend = pd.to_numeric(df['trend_score_sign'], errors='coerce')

        for w in windows:
            div = _direction_divergence(vol, trend, w)
            new_cols[f'vol_trend_div_{w}d'] = div
    else:
        logger.debug("vol_regime_ema10 or trend_score_sign not found; skipping vol-trend divergence")

    # Add all columns at once
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        for col in new_df.columns:
            df[col] = new_df[col]
        logger.debug(f"Added {len(new_cols)} divergence features")

    return df
