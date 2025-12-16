"""
Liquidity and market microstructure features.

These features capture trading conditions that affect momentum execution:
- Spread proxies (bid-ask estimation from OHLC)
- Intraday volatility patterns
- Volume-price relationships
- Illiquidity measures
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def add_liquidity_features(
    df: pd.DataFrame,
    windows: tuple = (5, 10, 20),
) -> pd.DataFrame:
    """Add liquidity and microstructure features.

    Args:
        df: DataFrame with OHLCV data (open, high, low, close, volume)
        windows: Rolling windows for smoothed features

    Returns:
        DataFrame with liquidity features added
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for liquidity features: {missing}")
        return df

    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

    # =========================================================================
    # 1. SPREAD PROXIES (bid-ask estimation)
    # =========================================================================

    # Parkinson volatility (high-low based) - proxy for spread + volatility
    # Higher values indicate wider spreads or higher volatility
    hl_range = (h - l).clip(lower=1e-8)
    df['hl_spread_proxy'] = (hl_range / c).astype('float32')

    # Corwin-Schultz spread estimator (simplified)
    # Based on the idea that high-low range captures spread component
    # Beta = sum of squared log(H/L) over 2 days
    log_hl = np.log(h / l.clip(lower=1e-8))
    log_hl_sq = log_hl ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)
    # Gamma = log(max(H_t, H_t-1) / min(L_t, L_t-1))^2
    h2 = np.maximum(h, h.shift(1))
    l2 = np.minimum(l, l.shift(1))
    gamma = np.log(h2 / l2.clip(lower=1e-8)) ** 2
    # Spread estimate (can be negative, so we floor at 0)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    df['cs_spread_est'] = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).astype('float32')

    # Roll spread estimator (based on serial covariance of price changes)
    # Negative serial covariance implies bid-ask bounce
    price_change = c.diff()
    # Suppress numpy warnings from covariance calculation (expected when variance is 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roll_cov = price_change.rolling(20, min_periods=10).cov(price_change.shift(1))
    # Clip denominator to avoid inf for low-priced stocks, replace any remaining inf with NaN
    roll_spread = 2 * np.sqrt((-roll_cov).clip(lower=0)) / c.clip(lower=1e-8)
    df['roll_spread_est'] = roll_spread.replace([np.inf, -np.inf], np.nan).astype('float32')

    # =========================================================================
    # 2. INTRADAY PATTERNS
    # =========================================================================

    # Gap vs continuation ratio
    # High ratio = price moves mostly happen at open (overnight), not intraday
    overnight_move = (o - c.shift(1)).abs()
    intraday_move = (c - o).abs()
    total_move = overnight_move + intraday_move + 1e-8
    df['overnight_ratio'] = (overnight_move / total_move).astype('float32')

    # Intraday range efficiency
    # How much of the high-low range was captured in open-close move
    # Low efficiency = lots of intraday reversal (indecision)
    oc_range = (c - o).abs()
    df['range_efficiency'] = (oc_range / hl_range).clip(upper=1.0).astype('float32')

    # Upper shadow ratio (selling pressure)
    # Large upper shadow = price rejected higher levels
    body_top = np.maximum(o, c)
    df['upper_shadow_ratio'] = ((h - body_top) / hl_range).astype('float32')

    # Lower shadow ratio (buying pressure)
    body_bottom = np.minimum(o, c)
    df['lower_shadow_ratio'] = ((body_bottom - l) / hl_range).astype('float32')

    # =========================================================================
    # 3. VWAP (Volume-Weighted Average Price)
    # =========================================================================

    # Typical price for VWAP calculation
    typical_price = (h + l + c) / 3

    # Rolling VWAP (cumulative would reset daily, but we use rolling for daily data)
    for w in windows:
        cum_tp_vol = (typical_price * v).rolling(w, min_periods=w//2).sum()
        cum_vol = v.rolling(w, min_periods=w//2).sum()
        vwap = cum_tp_vol / cum_vol.clip(lower=1)

        # Distance from VWAP (percentage)
        df[f'vwap_dist_{w}d'] = ((c - vwap) / vwap.clip(lower=1e-8) * 100).astype('float32')

        # VWAP distance z-score
        vwap_dist = (c - vwap) / vwap.clip(lower=1e-8)
        vwap_dist_std = vwap_dist.rolling(60, min_periods=20).std()
        df[f'vwap_dist_{w}d_zscore'] = (vwap_dist / vwap_dist_std.clip(lower=1e-8)).clip(-5, 5).astype('float32')

    # =========================================================================
    # 4. VOLUME-PRICE RELATIONSHIPS
    # =========================================================================

    # Volume-weighted price direction (buying vs selling pressure)
    # Positive = volume on up moves, negative = volume on down moves
    price_direction = np.sign(c - o)
    df['volume_direction'] = (price_direction * v).astype('float32')

    # Relative volume (vs rolling average)
    for w in windows:
        vol_ma = v.rolling(w, min_periods=w//2).mean()
        df[f'rel_volume_{w}d'] = (v / vol_ma.clip(lower=1)).astype('float32')

    # Volume trend (is volume increasing or decreasing)
    df['volume_trend_10d'] = (v.rolling(5, min_periods=2).mean() / v.rolling(20, min_periods=5).mean().clip(lower=1)).astype('float32')

    # Price-volume divergence
    # Price up but volume down = weak rally (bearish divergence)
    price_roc_5 = c.pct_change(5, fill_method=None)
    # Volume pct_change can produce inf when going from 0 to non-zero, replace with NaN
    vol_roc_5 = v.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
    df['pv_divergence_5d'] = (price_roc_5 - vol_roc_5).astype('float32')

    # =========================================================================
    # 4. ILLIQUIDITY MEASURES
    # =========================================================================

    # Amihud illiquidity ratio
    # abs(return) / dollar volume - price impact per dollar traded
    ret = c.pct_change(fill_method=None).abs()
    dollar_volume = c * v
    amihud = ret / dollar_volume.clip(lower=1)
    # Scale to reasonable range and smooth
    df['amihud_illiq'] = (amihud * 1e9).rolling(5, min_periods=1).mean().astype('float32')

    # Amihud ratio relative to rolling average (illiquidity shock)
    amihud_ma = df['amihud_illiq'].rolling(20, min_periods=10).mean()
    df['amihud_illiq_ratio'] = (df['amihud_illiq'] / amihud_ma.clip(lower=1e-8)).astype('float32')

    # Zero-volume days in rolling window (extreme illiquidity)
    zero_vol = (v == 0).astype(float)
    df['zero_vol_pct_20d'] = zero_vol.rolling(20, min_periods=10).mean().astype('float32')

    # =========================================================================
    # 5. MICROSTRUCTURE REGIME
    # =========================================================================

    # Composite liquidity score (normalized)
    # Combine multiple measures into single score
    spread_roll = df['hl_spread_proxy'].rolling(60, min_periods=20)
    spread_z = (df['hl_spread_proxy'] - spread_roll.mean()) / spread_roll.std().clip(lower=1e-8)
    amihud_roll = df['amihud_illiq'].rolling(60, min_periods=20)
    amihud_z = (df['amihud_illiq'] - amihud_roll.mean()) / amihud_roll.std().clip(lower=1e-8)
    vol_ratio_z = (df['rel_volume_20d'] - 1)  # Already relative, center at 0

    # High score = illiquid conditions (wide spread, high impact, low volume)
    df['illiquidity_score'] = ((spread_z + amihud_z - vol_ratio_z) / 3).clip(-3, 3).astype('float32')

    # Liquidity regime (categorical)
    # 1 = liquid, 2 = normal, 3 = illiquid
    illiq_20 = df['illiquidity_score'].rolling(20, min_periods=10).mean()
    df['liquidity_regime'] = pd.cut(
        illiq_20,
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=[1, 2, 3]
    ).astype('float32')

    return df


def add_liquidity_features_batch(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    windows: tuple = (5, 10, 20),
) -> None:
    """Add liquidity features to all symbols in-place.

    Args:
        indicators_by_symbol: Dict mapping symbol -> DataFrame
        windows: Rolling windows for smoothed features
    """
    for symbol, df in indicators_by_symbol.items():
        try:
            updated = add_liquidity_features(df, windows=windows)
            # Update in place - use loc to avoid chained assignment warning
            new_cols = [col for col in updated.columns if col not in df.columns]
            for col in new_cols:
                df.loc[:, col] = updated[col]
        except Exception as e:
            logger.warning(f"Failed to add liquidity features for {symbol}: {e}")
