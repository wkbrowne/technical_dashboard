"""
Volatility regime features based on realized volatility patterns.

This module computes multi-scale volatility features including volatility ratios,
regime indicators, and volatility-of-volatility metrics.
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

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
        quiet_gate = (out["vol_regime_ema10"] < 0).astype("float32")
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