"""
Volatility regime features based on realized volatility patterns.

This module computes multi-scale volatility features including volatility ratios,
regime indicators, volatility-of-volatility metrics, and squeeze detection.
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("pandas_ta is required for volatility calculations. Please: pip install pandas-ta")

logger = logging.getLogger(__name__)


def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    """
    Compute rolling z-score of a series.
    
    Args:
        s: Input series
        win: Rolling window size
        
    Returns:
        Rolling z-score series
    """
    m = s.rolling(win, min_periods=max(5, win//3)).mean()
    sd = s.rolling(win, min_periods=max(5, win//3)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def add_multiscale_vol_regime(
    df: pd.DataFrame,
    ret_col: str = "ret",
    short_windows: Tuple[int, ...] = (10, 20),
    long_windows: Tuple[int, ...] = (60, 100),
    z_window: int = 60,
    ema_span: int = 10,
    slope_win: int = 20,
    prefix: str = "rv",
    cs_ratio_median: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Add comprehensive multi-scale volatility regime features.
    
    Features added:
    - rv_{w}: Rolling standard deviation for each window
    - rv_ratio_10_60, rv_ratio_20_100: Short/long volatility ratios
    - vol_regime: log1p transform of rv_ratio_20_100
    - vol_regime_ema{ema_span}: EMA-smoothed vol_regime
    - rv_z_{z_window}: Z-score of rv_20 over z_window
    - vol_of_vol_20d: Volatility of volatility over 20 days
    - rv60_slope_norm, rv100_slope_norm: Normalized volatility slopes
    - vol_regime_cs_median, vol_regime_rel: Cross-sectional context (optional)
    - quiet_trend: Trend strength gated by low volatility (if trend exists)
    
    Args:
        df: Input DataFrame with return data
        ret_col: Column name for returns
        short_windows: Windows for short-term volatility
        long_windows: Windows for long-term volatility
        z_window: Window for computing volatility z-scores
        ema_span: EMA span for smoothing
        slope_win: Window for computing volatility slopes
        prefix: Prefix for volatility column names
        cs_ratio_median: Cross-sectional median of volatility ratios (optional)
        
    Returns:
        DataFrame with added volatility features (mutates input DataFrame)
    """
    if ret_col not in df.columns:
        logger.warning(f"Return column '{ret_col}' not found, skipping volatility features")
        return df

    logger.debug("Computing multi-scale volatility regime features")
    
    out = df.copy()
    r = pd.to_numeric(out[ret_col], errors="coerce")

    # 1) Core realized volatilities
    all_wins = sorted(set(list(short_windows) + list(long_windows)))
    for w in all_wins:
        out[f"{prefix}_{w}"] = r.rolling(w, min_periods=max(5, w//3)).std(ddof=0).astype("float32")

    # 2) Volatility ratios (short vs long)
    def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        if a is None or b is None:
            return pd.Series(index=out.index, dtype="float32")
        return (a / b.replace(0, np.nan)).astype("float32")

    if (10 in short_windows) and (60 in long_windows):
        out["rv_ratio_10_60"] = _safe_ratio(out.get("rv_10"), out.get("rv_60"))
    if (20 in short_windows) and (100 in long_windows):
        out["rv_ratio_20_100"] = _safe_ratio(out.get("rv_20"), out.get("rv_100"))

    # 3) Canonical regime score + smoothing
    if "rv_ratio_20_100" in out.columns:
        out["vol_regime"] = np.log1p(out["rv_ratio_20_100"]).astype("float32")
        out["vol_regime_ema10"] = (
            out["vol_regime"].ewm(span=ema_span, adjust=False, min_periods=1).mean().astype("float32")
        )

    # 4) Z-score of rv_20 in local context
    if "rv_20" in out.columns and z_window:
        mu = out["rv_20"].rolling(z_window, min_periods=max(5, z_window//3)).mean()
        sd = out["rv_20"].rolling(z_window, min_periods=max(5, z_window//3)).std(ddof=0)
        out[f"rv_z_{z_window}"] = ((out["rv_20"] - mu) / sd.replace(0, np.nan)).astype("float32")

    # 5) Volatility-of-volatility
    if "rv_20" in out.columns:
        d_rv20 = pd.to_numeric(out["rv_20"], errors="coerce").diff()
        out["vol_of_vol_20d"] = d_rv20.rolling(20, min_periods=5).std(ddof=0).astype("float32")

    # 6) Directional volatility trends (normalized slopes)
    def _norm_slope(s: pd.Series, win: int) -> pd.Series:
        if s is None:
            return pd.Series(index=out.index, dtype="float32")
        # Percent change per bar over 'win' bars
        pct_per_bar = (s / s.shift(win) - 1.0) / float(win)
        return pct_per_bar.replace([np.inf, -np.inf], np.nan).astype("float32")

    if "rv_60" in out.columns:
        out["rv60_slope_norm"] = _norm_slope(out["rv_60"], slope_win)
    if "rv_100" in out.columns:
        out["rv100_slope_norm"] = _norm_slope(out["rv_100"], slope_win)

    # 7) Cross-sectional regime context (optional)
    if cs_ratio_median is not None and "vol_regime" in out.columns:
        cs_med = pd.to_numeric(cs_ratio_median.reindex(out.index), errors="coerce")
        vol_regime_cs = np.log1p(cs_med)  # Same transform as vol_regime
        out["vol_regime_cs_median"] = vol_regime_cs.astype("float32")
        out["vol_regime_rel"] = (out["vol_regime"] - out["vol_regime_cs_median"]).astype("float32")

    # 8) Quiet-trend interaction (only if upstream trend exists)
    if "trend_score_granular" in out.columns and "vol_regime_ema10" in out.columns:
        # Emphasize trend in quiet regimes; suppress in turbulent regimes
        # vol_regime_ema10 = log1p(rv_ratio_20_100), median is ~log1p(1.0) = 0.69
        # "Quiet" = below median volatility regime (short-term vol < long-term vol)
        vol_median = out["vol_regime_ema10"].rolling(252, min_periods=50).median()
        quiet_gate = (out["vol_regime_ema10"] < vol_median).astype("float32")
        out["quiet_trend"] = (out["trend_score_granular"] * quiet_gate).astype("float32")

    logger.debug("Multi-scale volatility regime features completed")
    return out


def add_vol_regime_cs_context(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    ratio_col: str = "rv_ratio_20_100",
    out_cs_col: str = "vol_regime_cs_median",
    out_rel_col: str = "vol_regime_rel",
) -> None:
    """
    Add cross-sectional volatility regime context to all symbols.
    
    Computes cross-sectional median of volatility ratios and adds:
    - vol_regime_cs_median: Cross-sectional median volatility regime
    - vol_regime_rel: Symbol's vol_regime relative to cross-sectional median
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        ratio_col: Column name for volatility ratio
        out_cs_col: Output column name for cross-sectional median
        out_rel_col: Output column name for relative volatility regime
    """
    logger.info("Computing cross-sectional volatility regime context")
    
    # Collect all ratio series
    cols = {}
    for sym, df in indicators_by_symbol.items():
        if ratio_col in df.columns:
            cols[sym] = pd.to_numeric(df[ratio_col], errors="coerce")
    
    if not cols:
        logger.warning(f"No symbols have {ratio_col} column")
        return

    # Build panel and compute median per date
    panel = pd.DataFrame(cols).sort_index()
    cs_ratio_median = panel.median(axis=1, skipna=True)

    # Attach to each symbol
    attached = 0
    for sym, df in indicators_by_symbol.items():
        if df.empty:
            continue

        # Apply same transform used for vol_regime
        cs_log = np.log1p(pd.to_numeric(cs_ratio_median.reindex(df.index), errors="coerce")).astype("float32")
        df[out_cs_col] = cs_log

        if "vol_regime" in df.columns:
            df[out_rel_col] = (df["vol_regime"].astype("float32") - df[out_cs_col]).astype("float32")

        attached += 1

    logger.info(f"Cross-sectional vol regime context attached to {attached} symbols "
                f"(median over {len(cols)} symbols per date)")


def add_bollinger_bandwidth_features(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
    z_window: int = 60
) -> pd.DataFrame:
    """
    Add Bollinger Bandwidth features using pandas-ta.

    Bollinger Bandwidth measures the width of Bollinger Bands relative to the middle band.
    Low bandwidth indicates volatility compression, high bandwidth indicates expansion.

    Features added:
    - bb_width_{length}_{std}: Bollinger Bandwidth = (upper - lower) / middle
    - bb_width_{length}_{std}_z{z_window}: Z-score of bandwidth over z_window

    Args:
        df: Input DataFrame with price data (requires 'close' or 'adjclose')
        length: Lookback period for Bollinger Bands (default: 20)
        std: Standard deviation multiplier (default: 2.0)
        z_window: Window for z-score calculation (default: 60)

    Returns:
        DataFrame with added Bollinger Bandwidth features (mutates input DataFrame)
    """
    logger.debug(f"Adding Bollinger Bandwidth features (length={length}, std={std})")

    # Get close price
    close_col = 'close' if 'close' in df.columns else 'adjclose'
    if close_col not in df.columns:
        logger.warning(f"Price column not found for Bollinger Bandwidth calculation")
        return df

    close = pd.to_numeric(df[close_col], errors='coerce')

    try:
        # Use pandas-ta for Bollinger Bands
        # Returns DataFrame with: BBL_{length}_{std}, BBM_{length}_{std}, BBU_{length}_{std}, BBB_{length}_{std}, BBP_{length}_{std}
        bb_result = ta.bbands(close=close, length=length, std=std)

        if bb_result is not None and not bb_result.empty:
            # Extract columns - pandas-ta names vary by version
            # Try different naming conventions: BBL_20_2.0, BBL_20_2.0_2.0, etc.
            bbl_col = None
            bbm_col = None
            bbu_col = None

            # Search for matching column names
            for col in bb_result.columns:
                if col.startswith('BBL_'):
                    bbl_col = col
                elif col.startswith('BBM_'):
                    bbm_col = col
                elif col.startswith('BBU_'):
                    bbu_col = col

            # Calculate width as (upper - lower) / middle
            if bbu_col and bbl_col and bbm_col:
                upper = bb_result[bbu_col]
                lower = bb_result[bbl_col]
                middle = bb_result[bbm_col]

                # Compute bandwidth = (upper - lower) / middle
                # Note: BBB in pandas-ta is already computed, but we calculate to ensure consistency
                bb_width = (upper - lower) / middle.replace(0, np.nan)

                # Feature naming: bb_width_20_2 (length_std as integer)
                std_int = int(std) if std == int(std) else std
                width_name = f'bb_width_{length}_{std_int}'
                z_name = f'{width_name}_z{z_window}'

                df[width_name] = bb_width.astype('float32')

                # Compute z-score of bandwidth
                bb_width_z = _rolling_z(bb_width, z_window)
                df[z_name] = bb_width_z.astype('float32')

                logger.debug(f"Added {width_name} and {z_name} features")
            else:
                logger.warning(f"Expected Bollinger columns not found. Available: {list(bb_result.columns)}")
        else:
            logger.debug("Bollinger Bands calculation returned None/empty, creating NaN series")
            std_int = int(std) if std == int(std) else std
            df[f'bb_width_{length}_{std_int}'] = pd.Series(np.nan, index=df.index, dtype='float32')
            df[f'bb_width_{length}_{std_int}_z{z_window}'] = pd.Series(np.nan, index=df.index, dtype='float32')

    except Exception as e:
        logger.warning(f"Failed to compute Bollinger Bandwidth: {e}")
        std_int = int(std) if std == int(std) else std
        df[f'bb_width_{length}_{std_int}'] = pd.Series(np.nan, index=df.index, dtype='float32')
        df[f'bb_width_{length}_{std_int}_z{z_window}'] = pd.Series(np.nan, index=df.index, dtype='float32')

    logger.debug("Bollinger Bandwidth features computation completed")
    return df


def add_squeeze_features(
    df: pd.DataFrame,
    length: int = 20,
    bb_std: float = 2.0,
    kc_scalar_narrow: float = 1.5,
    kc_scalar_wide: float = 2.0,
    max_squeeze_days: int = 50
) -> pd.DataFrame:
    """
    Add volatility squeeze features using pandas-ta for building blocks.

    Squeeze occurs when Bollinger Bands contract inside Keltner Channels,
    indicating volatility compression that often precedes a breakout.

    Features added:
    - squeeze_on_{length}: 1 if BB inside KC(narrow), else 0
    - squeeze_on_wide_{length}: 1 if BB inside KC(wide), else 0 (stricter squeeze)
    - squeeze_intensity_{length}: How much BB is inside KC, clipped [-1, 1]
    - squeeze_release_{length}: 1 on transition from squeeze_on to squeeze_off
    - days_in_squeeze_{length}: Consecutive days in squeeze (capped at max_squeeze_days)

    Args:
        df: Input DataFrame with OHLC data
        length: Lookback period for BB and KC (default: 20)
        bb_std: Bollinger Bands standard deviation (default: 2.0)
        kc_scalar_narrow: Keltner Channel multiplier for narrow (default: 1.5)
        kc_scalar_wide: Keltner Channel multiplier for wide (default: 2.0)
        max_squeeze_days: Cap for days_in_squeeze feature (default: 50)

    Returns:
        DataFrame with added squeeze features (mutates input DataFrame)
    """
    logger.debug(f"Adding squeeze features (length={length}, bb_std={bb_std}, kc_scalar={kc_scalar_narrow}/{kc_scalar_wide})")

    required = {'high', 'low'}
    if not required.issubset(df.columns):
        logger.warning(f"Required columns {required} not found for squeeze calculation")
        return df

    close_col = 'close' if 'close' in df.columns else 'adjclose'
    if close_col not in df.columns:
        logger.warning(f"Price column not found for squeeze calculation")
        return df

    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    close = pd.to_numeric(df[close_col], errors='coerce')

    try:
        # Get Bollinger Bands
        bb_result = ta.bbands(close=close, length=length, std=bb_std)

        # Get Keltner Channels (narrow and wide)
        kc_narrow = ta.kc(high=high, low=low, close=close, length=length, scalar=kc_scalar_narrow)
        kc_wide = ta.kc(high=high, low=low, close=close, length=length, scalar=kc_scalar_wide)

        if bb_result is None or kc_narrow is None:
            logger.debug("BB or KC calculation returned None, creating NaN series")
            _create_nan_squeeze_features(df, length)
            return df

        # Extract Bollinger Bands - use prefix search due to pandas-ta naming variance
        bbu_col = None
        bbl_col = None
        for col in bb_result.columns:
            if col.startswith('BBU_'):
                bbu_col = col
            elif col.startswith('BBL_'):
                bbl_col = col

        if bbu_col is None or bbl_col is None:
            logger.warning(f"BB columns not found. Available: {list(bb_result.columns)}")
            _create_nan_squeeze_features(df, length)
            return df

        bb_upper = bb_result[bbu_col]
        bb_lower = bb_result[bbl_col]

        # Extract Keltner Channels (narrow) - use prefix search due to naming variance
        kcu_narrow_col = None
        kcl_narrow_col = None
        for col in kc_narrow.columns:
            # Match KCU or KCUe prefix (upper channel)
            if col.startswith('KCU'):
                kcu_narrow_col = col
            # Match KCL or KCLe prefix (lower channel)
            elif col.startswith('KCL'):
                kcl_narrow_col = col

        if kcu_narrow_col is None or kcl_narrow_col is None:
            logger.warning(f"KC narrow columns not found. Available: {list(kc_narrow.columns)}")
            _create_nan_squeeze_features(df, length)
            return df

        kc_upper_narrow = kc_narrow[kcu_narrow_col]
        kc_lower_narrow = kc_narrow[kcl_narrow_col]

        # Squeeze on (narrow): BB is inside KC
        # Condition: bb_upper < kc_upper AND bb_lower > kc_lower
        squeeze_on = ((bb_upper < kc_upper_narrow) & (bb_lower > kc_lower_narrow)).astype('int8')
        df[f'squeeze_on_{length}'] = squeeze_on

        # Squeeze on (wide): stricter condition with wider KC
        if kc_wide is not None:
            # Use prefix search for wide KC columns
            kcu_wide_col = None
            kcl_wide_col = None
            for col in kc_wide.columns:
                if col.startswith('KCU'):
                    kcu_wide_col = col
                elif col.startswith('KCL'):
                    kcl_wide_col = col

            if kcu_wide_col is not None and kcl_wide_col is not None:
                kc_upper_wide = kc_wide[kcu_wide_col]
                kc_lower_wide = kc_wide[kcl_wide_col]
                squeeze_on_wide = ((bb_upper < kc_upper_wide) & (bb_lower > kc_lower_wide)).astype('int8')
                df[f'squeeze_on_wide_{length}'] = squeeze_on_wide
            else:
                df[f'squeeze_on_wide_{length}'] = pd.Series(np.nan, index=df.index, dtype='int8')
        else:
            df[f'squeeze_on_wide_{length}'] = pd.Series(np.nan, index=df.index, dtype='int8')

        # Squeeze intensity: how compressed BB is relative to KC
        # = (kc_width - bb_width) / kc_width, clipped to [-1, 1]
        bb_width = bb_upper - bb_lower
        kc_width = kc_upper_narrow - kc_lower_narrow

        intensity = (kc_width - bb_width) / kc_width.replace(0, np.nan)
        intensity = intensity.clip(-1, 1)
        df[f'squeeze_intensity_{length}'] = intensity.astype('float32')

        # Squeeze release: transition from squeeze_on=1 to squeeze_on=0
        squeeze_release = ((squeeze_on.shift(1) == 1) & (squeeze_on == 0)).astype('int8')
        df[f'squeeze_release_{length}'] = squeeze_release

        # Days in squeeze: consecutive days with squeeze_on=1, capped
        days_in_squeeze = _consecutive_count(squeeze_on, max_count=max_squeeze_days)
        df[f'days_in_squeeze_{length}'] = days_in_squeeze.astype('float32')

        logger.debug(f"Added squeeze features for length={length}")

    except Exception as e:
        logger.warning(f"Failed to compute squeeze features: {e}")
        _create_nan_squeeze_features(df, length)

    logger.debug("Squeeze features computation completed")
    return df


def _consecutive_count(series: pd.Series, max_count: int = 50) -> pd.Series:
    """
    Count consecutive True/1 values in a series, resetting on False/0.

    Args:
        series: Boolean or int series
        max_count: Maximum count to cap at (default: 50)

    Returns:
        Series of consecutive counts, capped at max_count
    """
    # Convert to boolean
    mask = series.astype(bool)

    # Use cumsum of not-mask to create groups
    # When mask is False, we start a new group
    groups = (~mask).cumsum()

    # Within each group, cumsum of mask gives consecutive count
    counts = mask.groupby(groups).cumsum()

    # Cap at max_count
    return counts.clip(upper=max_count)


def _create_nan_squeeze_features(df: pd.DataFrame, length: int) -> None:
    """Create NaN squeeze features when calculation fails."""
    df[f'squeeze_on_{length}'] = pd.Series(np.nan, index=df.index, dtype='int8')
    df[f'squeeze_on_wide_{length}'] = pd.Series(np.nan, index=df.index, dtype='int8')
    df[f'squeeze_intensity_{length}'] = pd.Series(np.nan, index=df.index, dtype='float32')
    df[f'squeeze_release_{length}'] = pd.Series(np.nan, index=df.index, dtype='int8')
    df[f'days_in_squeeze_{length}'] = pd.Series(np.nan, index=df.index, dtype='float32')