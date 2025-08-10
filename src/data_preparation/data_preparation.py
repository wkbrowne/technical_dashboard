# data_preparation.py
# ------------------------------------------------------------
# Build modeling-ready feature sets for all symbols.
# - Requires: src/data/loader.py providing load_stock_universe(), load_etf_universe()
# - Outputs:  ./artifacts/features_long.parquet  (long, tidy)
#             ./artifacts/symbol_frames/<SYM>.parquet  (one wide DF per symbol)
# ------------------------------------------------------------
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings("ignore", message="RANSAC did not reach consensus, using numpy's polyfit")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# Add project root to sys.path: .../project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- loader imports ----------
from src.data.loader import load_stock_universe, load_etf_universe

# Load S&P 500 tickers from cache
from cache.sp500_list import SP500_TICKERS

# ---------- nolds (required) ----------
try:
    import nolds
except Exception:
    raise ImportError("nolds is required for Hurst features. Please: pip install nolds")

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = Path("./artifacts")
SYMBOL_FRAMES_DIR = OUTPUT_DIR / "symbol_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SYMBOL_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Universe / fetch knobs
MAX_STOCKS: Optional[int] = None   # None => full universe from CSV
RATE_LIMIT_REQ_PER_SEC: float = 1.0
INTERVAL = "1d"

# ETF list to fetch (ensure SPY is included for RS)
DEFAULT_ETFS = [
    "SPY","QQQ","IWM","DIA","TLT","IEF","HYG","LQD",
    "XLF","XLK","XLE","XLY","XLI","XLP","XLV","XLU","XLB","XLC",
    "EFA","EEM","GLD","SLV","USO","UNG",
    # breadth / sector extras (optional)
    "SMH","XRT","ITA","KBE","KRE","IBB","IHE","IYT","XLRE"
]
SPY_SYMBOL = "SPY"

# Sector→ETF (lower-case keys)
SECTOR_TO_ETF = {
    "technology services": "XLK",
    "electronic technology": "XLK",
    "finance": "XLF",
    "retail trade": "XRT",
    "health technology": "XLV",
    "consumer non-durables": "XLP",
    "producer manufacturing": "XLI",
    "energy minerals": "XLE",
    "consumer services": "XLY",
    "consumer durables": "XLY",
    "utilities": "XLU",
    "non-energy minerals": "XLB",
    "industrial services": "XLI",
    "transportation": "IYT",
    "commercial services": "XLC",
    "process industries": "XLB",
    "communications": "XLC",
    "health services": "XLV",
    "distribution services": "XLI",
    "miscellaneous": "SPY",
}

# ============================================================
# Helpers: assemble per-symbol frames (lower-case)
# ============================================================

def _feature_worker(sym, df, cs_ratio_median=None):
    """Runs the core feature stack for one symbol; returns (sym, out_df)."""
    try:
        out = df.copy()

        # Ensure returns exist
        if "ret" not in out.columns:
            if "adjclose" in out.columns:
                out["ret"] = np.log(pd.to_numeric(out["adjclose"], errors="coerce")).diff()
            else:
                print(f"⚠️ {sym}: No adjclose column for returns calculation")
                return sym, out

        # 1) Trend features (MA slopes, agreement, etc.)
        out = add_trend_features(
            out,
            src_col='adjclose',
            ma_periods=(10, 20, 30, 50, 75, 100, 150, 200),
            slope_window=20,
            eps=1e-5
        )

        # 2) Enhanced multi‑scale vol regime (includes rv ratios, vol_regime, z, slopes, quiet_trend)
        #    Pass cs_ratio_median=None on first pass; you can append CS context after parallel (see below).
        out = add_multiscale_vol_regime(
            out,
            ret_col="ret",
            short_windows=(10, 20),
            long_windows=(60, 100),
            z_window=60,
            ema_span=10,
            slope_win=20,
            cs_ratio_median=None,   # Will be computed after parallel processing
        )

        # 3) Hurst (slow + smoothed) — keep this if you want the robust trending/MR gauge
        out = add_hurst_features(out, ret_col="ret", windows=(64, 128), ema_halflife=5)

        # 4) Distance-to-MA + z-scores
        out = add_distance_to_ma_features(
            out,
            src_col='adjclose',
            ma_lengths=(20, 50, 100, 200),
            z_window=60
        )

        # 5) Range / breakout features
        out = add_range_breakout_features(out, win_list=(5, 10, 20))

        # 6) Volume features
        out = add_volume_features(out)

        # 7) Volume shock + alignment features
        out = add_volume_shock_features(
            out,
            vol_col="volume",
            price_col_primary="close",
            price_col_fallback="adjclose",
            lookback=20,
            ema_span=10,
            prefix="volshock"
        )

        return sym, out

    except Exception as e:
        print(f"[features] {sym} failed: {e}")
        return sym, df

def _safe_lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Don't lowercase the column names since they are stock symbols
    return df.copy()

def assemble_indicators_from_wide(data: Dict[str, pd.DataFrame],
                                  adjust_ohlc_with_factor: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Input `data` is a dict of wide frames like:
      data = {'Open': df_wide, 'High': df_wide, 'Low': df_wide, 'Close': df_wide,
              'AdjClose': df_wide, 'Volume': df_wide}
    Return dict[symbol -> df] with lower-case columns: open/high/low/close/adjclose/volume
    Also:
      - adds factor = adjclose/close and optionally adjusts O/H/L by factor
      - adds ret = log(adjclose).diff()
    """
    req = ["AdjClose"]
    for r in req:
        if r not in data:
            raise ValueError(f"Expected '{r}' in loaded data keys; got {list(data.keys())}")

    keys = ["Open","High","Low","Close","AdjClose","Volume"]
    frames = {k.lower(): _safe_lower_columns(v) for k, v in data.items() if k in keys and isinstance(v, pd.DataFrame)}
    all_syms = set()
    for dfw in frames.values():
        all_syms |= set(dfw.columns)

    indicators_by_symbol: Dict[str, pd.DataFrame] = {}
    for sym in sorted(all_syms):
        parts = []
        for k, dfw in frames.items():
            if sym in dfw.columns:
                s = pd.to_numeric(dfw[sym], errors='coerce').rename(k)
                parts.append(s)
        if not parts:
            continue
        df = pd.concat(parts, axis=1).sort_index()
        for col in ["open","high","low","close","adjclose","volume"]:
            if col not in df.columns:
                df[col] = np.nan

        if adjust_ohlc_with_factor and ("close" in df.columns) and ("adjclose" in df.columns):
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = df["adjclose"] / df["close"]
            for c in ["open","high","low"]:
                df[c] = df[c] * factor

        df["ret"] = np.log(df["adjclose"]).diff()
        indicators_by_symbol[sym] = df

    return indicators_by_symbol

# ============================================================
# Feature builders

# --- Hurst helpers ---

def _safe_hurst_rs(x: np.ndarray) -> float:
    """R/S Hurst exponent calculation via nolds."""
    try:
        x_clean = np.asarray(x, dtype=float)
        x_clean = x_clean[~np.isnan(x_clean)]  # Remove NaNs
        if len(x_clean) < 10:  # Need minimum data
            return np.nan
        result = float(nolds.hurst_rs(x_clean, fit="poly"))
        return result
    except Exception:
        return np.nan

def add_hurst_features(df: pd.DataFrame,
                       ret_col: str = "ret",
                       windows=(128,),
                       ema_halflife: int = 5,
                       prefix: str = "hurst_ret"
                       ) -> pd.DataFrame:
    """
    Rolling Hurst exponent on returns with optional EMA smoothing.
    """
    if ret_col not in df.columns:
        return df
    s = pd.to_numeric(df[ret_col], errors="coerce")

    for w in windows:
        col = f"{prefix}_{w}"
        min_periods = max(50, w//2)
        df[col] = s.rolling(window=w, min_periods=min_periods).apply(_safe_hurst_rs, raw=False)

    if ema_halflife and len(windows):
        base = f"{prefix}_{windows[0]}"
        if base in df.columns:
            df[f"{base}_emaHL{ema_halflife}"] = (
                df[base].ewm(halflife=ema_halflife, adjust=False, min_periods=1).mean()
            )
    return df


def add_vol_regime_cs_context(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    ratio_col: str = "rv_ratio_20_100",
    out_cs_col: str = "vol_regime_cs_median",
    out_rel_col: str = "vol_regime_rel",
) -> None:
    """
    Cross-sectional context for volatility regime:
      1) Build a date-indexed panel of rv_ratio_20_100 across all symbols
      2) Take median per date
      3) For each symbol:
           vol_regime_cs_median = log1p(cs_median_ratio)
           vol_regime_rel       = vol_regime - vol_regime_cs_median
    Writes columns in-place; returns None.
    """
    # Collect all ratio series we have
    cols = {}
    for sym, df in indicators_by_symbol.items():
        if ratio_col in df.columns:
            cols[sym] = pd.to_numeric(df[ratio_col], errors="coerce")
    if not cols:
        print("⚠️ add_vol_regime_cs_context: no symbols have", ratio_col)
        return

    # Build panel and compute median per date
    panel = pd.DataFrame(cols).sort_index()
    cs_ratio_median = panel.median(axis=1, skipna=True)

    # Attach to each symbol
    attached = 0
    for sym, df in indicators_by_symbol.items():
        if df.empty:
            continue

        # same transform used for vol_regime
        cs_log = np.log1p(pd.to_numeric(cs_ratio_median.reindex(df.index), errors="coerce")).astype("float32")
        df[out_cs_col] = cs_log

        if "vol_regime" in df.columns:
            df[out_rel_col] = (df["vol_regime"].astype("float32") - df[out_cs_col]).astype("float32")

        attached += 1

    print(f"✅ Cross-sectional vol regime context attached to {attached} symbols "
          f"(median over {len(cols)} symbols per date).")

from typing import Iterable

def add_xsec_momentum_panel(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (5, 20, 60),
    price_col: str = "adjclose",
    sector_map: Optional[Dict[str, str]] = None,  # sym -> sector (any taxonomy)
    col_prefix: str = "xsec_mom"
) -> None:
    """
    Adds cross-sectional momentum z-scores for each lookback to every symbol's DF.

    For each lookback L:
      1) Compute L-day log-return per symbol.
      2) For each date, z-score across symbols:
         z = (ret - cross_median) / cross_std
      3) If sector_map is provided: sector-neutral first (subtract sector median on each date),
         then z-score across all symbols (so values are directly comparable).

    Writes columns: f"{col_prefix}_{L}d_z" (and f"{col_prefix}_{L}d_sect_neutral_z" if sector_map provided).
    """
    # 1) Build a price panel (date index, columns = symbols)
    syms = [s for s, df in indicators_by_symbol.items() if price_col in df.columns]
    if not syms:
        return

    panel = pd.DataFrame(index=pd.Index([], name=None))
    for s in syms:
        panel[s] = pd.to_numeric(indicators_by_symbol[s][price_col], errors='coerce')

    # Align all indexes (outer join) then forward-fill to reduce accidental NaNs
    panel = panel.sort_index()

    # 2) Compute log-returns & L-day momentum per symbol (vectorized)
    logp = np.log(panel.replace(0, np.nan))
    # daily log return
    ret1 = logp.diff()

    for L in lookbacks:
        # L‑day log return: sum of daily log-returns over L days
        momL = ret1.rolling(L, min_periods=max(3, L//3)).sum()

        # --- sector-neutral (optional) ---
        if sector_map:
            # group columns by sector
            sect_groups: Dict[str, list] = {}
            for s in syms:
                sec = sector_map.get(s)
                if isinstance(sec, str):
                    sect_groups.setdefault(sec, []).append(s)

            # subtract sector median per date within each sector
            sector_neutral_dfs = []
            for sec, cols in sect_groups.items():
                sub = momL[cols]
                med = sub.median(axis=1, skipna=True)
                sector_neutral_dfs.append(sub.sub(med, axis=0))

            # fall back: any symbols without sector get original (won't be sector-neutralized)
            missing = [s for s in syms if s not in sum([list(cols) for cols in sect_groups.values()], [])]
            if missing:
                sector_neutral_dfs.append(momL[missing])

            # Combine all sector-neutral data at once
            mom_sect_neutral = pd.concat(sector_neutral_dfs, axis=1) if sector_neutral_dfs else pd.DataFrame(index=momL.index)

            # final z across *all* symbols (so values are comparable across sectors)
            row_std = mom_sect_neutral.std(axis=1, ddof=0).replace(0, np.nan)
            row_med = mom_sect_neutral.median(axis=1, skipna=True)
            z_sect = mom_sect_neutral.sub(row_med, axis=0).div(row_std, axis=0)

        # --- plain cross‑sectional z (no sector neutral) ---
        row_std_plain = momL.std(axis=1, ddof=0).replace(0, np.nan)
        row_med_plain = momL.median(axis=1, skipna=True)
        z_plain = momL.sub(row_med_plain, axis=0).div(row_std_plain, axis=0)

        # 3) Write back to each symbol DF
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

def add_volume_shock_features(df: pd.DataFrame,
                              vol_col: str = "volume",
                              price_col_primary: str = "close",
                              price_col_fallback: str = "adjclose",
                              lookback: int = 20,
                              ema_span: int = 10,
                              prefix: str = "volshock") -> pd.DataFrame:
    """
    Volume Shock + Trend Alignment
      - {prefix}_z       : Z-score of volume vs rolling mean/std (lookback)
      - {prefix}_dir     : Directional shock (shock sign * price move sign over lookback//2)
      - {prefix}_ema     : Smoothed directional shock (EMA)
    """
    out = df.copy()
    if vol_col not in out.columns:
        return out

    # Choose price column (close -> adjclose)
    price_col = price_col_primary if price_col_primary in out.columns else (
        price_col_fallback if price_col_fallback in out.columns else None
    )
    if price_col is None:
        return out

    vol = pd.to_numeric(out[vol_col], errors="coerce")
    px  = pd.to_numeric(out[price_col], errors="coerce")

    # Rolling stats on volume
    roll_mean = vol.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    roll_std  = vol.rolling(lookback, min_periods=max(5, lookback // 3)).std(ddof=0)

    # Volume shock Z
    z = (vol - roll_mean) / roll_std.replace(0, np.nan)
    out[f"{prefix}_z"] = z.astype("float32")

    # Direction over a shorter horizon to align shock with price impulse
    half = max(1, lookback // 2)
    px_dir = np.sign(px - px.shift(half)).astype("float32")

    # Directional shock
    out[f"{prefix}_dir"] = (z * px_dir).astype("float32")

    # Smoothed directional shock
    out[f"{prefix}_ema"] = (
        out[f"{prefix}_dir"].ewm(span=ema_span, adjust=False, min_periods=1).mean().astype("float32")
    )

    return out

# --- Rolling z-score helper ---
def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(5, win//3)).mean()
    sd = s.rolling(win, min_periods=max(5, win//3)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

# ============================================================
# Public: add_regime_features
# ============================================================
# --- Regime wrapper (drop-in) ---------------------------------

def _rolling_autocorr(x: pd.Series, w: int) -> pd.Series:
    """
    Fast-ish rolling lag-1 autocorrelation using cov/var.
    Returns NaN where not enough data.
    """
    x = pd.to_numeric(x, errors='coerce')
    x_lag = x.shift(1)
    # rolling means
    m  = x.rolling(w, min_periods=max(3, w//3)).mean()
    mL = x_lag.rolling(w, min_periods=max(3, w//3)).mean()
    # rolling cov and var
    cov = (x * x_lag).rolling(w, min_periods=max(3, w//3)).mean() - (m * mL)
    var = (x * x).rolling(w, min_periods=max(3, w//3)).mean() - (m * m)
    with np.errstate(divide='ignore', invalid='ignore'):
        acf1 = cov / var.replace(0, np.nan)
    return acf1

def _ensure_ma(df: pd.DataFrame, src='adjclose', p=20, minp=None) -> pd.Series:
    if minp is None:
        minp = max(5, p//2)
    if f"ma_{p}" in df.columns:
        return pd.to_numeric(df[f"ma_{p}"], errors='coerce')
    if src in df.columns:
        return pd.to_numeric(df[src], errors='coerce').rolling(p, min_periods=minp).mean()
    return pd.Series(index=df.index, dtype='float64')

def add_trend_features(
    df: pd.DataFrame,
    src_col: str = 'adjclose',
    ma_periods=(10,20,30,50,75,100,150,200),
    slope_window: int = 20,
    eps: float = 1e-5
) -> pd.DataFrame:
    sign_cols = []
    for p in ma_periods:
        ma = _ensure_ma(df, src=src_col, p=p)
        df[f"ma_{p}"] = ma
        slope = (ma / ma.shift(slope_window) - 1.0)
        df[f"pct_slope_ma_{p}"] = slope.astype('float32')
        sign = np.where(slope > eps, 1.0, np.where(slope < -eps, -1.0, 0.0))
        df[f"sign_ma_{p}"] = sign.astype('float32')
        sign_cols.append(f"sign_ma_{p}")

    if sign_cols:
        sign_mat = df[sign_cols].to_numpy(dtype='float32')
        nz = (sign_mat != 0).sum(axis=1)
        sums = sign_mat.sum(axis=1)
        trend_score = np.divide(
            sums, np.where(nz == 0, np.nan, nz),
            out=np.zeros_like(sums, dtype='float32'), where=nz != 0
        )
        df["trend_score_granular"] = trend_score.astype('float32')
        df["trend_score_sign"] = np.sign(trend_score).astype('float32')
        df["trend_score_slope"] = pd.Series(trend_score, index=df.index).diff().astype('float32')
        df["trend_persist_ema"] = (
            df["trend_score_sign"].ewm(span=10, adjust=False, min_periods=1).mean().astype('float32')
        )
        pos = (sign_mat > 0).sum(axis=1)
        neg = (sign_mat < 0).sum(axis=1)
        denom = (pos + neg).astype('float32')
        df["trend_alignment"] = (pos / np.where(denom == 0, np.nan, denom)).astype('float32')
    return df

# -------- Multi‑scale price volatility regime (enhanced) --------
def add_multiscale_vol_regime(
    df: pd.DataFrame,
    ret_col: str = "ret",
    short_windows=(10, 20),
    long_windows=(60, 100),
    z_window: int = 60,
    ema_span: int = 10,
    slope_win: int = 20,
    prefix: str = "rv",
    # Optional cross‑sectional context:
    # pass a pd.Series indexed by date with the universe median of rv_ratio_20_100
    cs_ratio_median: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Adds realized‑volatility regime features:
      - rv_{w}: rolling std of log returns (w in short_windows ∪ long_windows)
      - rv_ratio_10_60, rv_ratio_20_100 (short/long vol ratios)
      - vol_regime = log1p(rv_ratio_20_100), vol_regime_ema{ema_span}
      - rv_z_{z_window}: z‑score of rv_20 over a z_window
      - vol_of_vol_20d: std of daily change in rv_20 over 20d
      - rv60_slope_norm, rv100_slope_norm: normalized slopes of intermediate/long vol
      - vol_regime_cs_median, vol_regime_rel: cross‑sectional context (optional)
      - quiet_trend: trend strength gated by low‑vol regime (if trend_score_granular exists)
    """
    if ret_col not in df.columns:
        return df

    out = df.copy()
    r = pd.to_numeric(out[ret_col], errors="coerce")

    # 1) Core realized vols
    all_wins = sorted(set(list(short_windows) + list(long_windows)))
    for w in all_wins:
        out[f"{prefix}_{w}"] = r.rolling(w, min_periods=max(5, w//3)).std(ddof=0).astype("float32")

    # 2) Ratios (short vs long)
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

    # 4) Z‑score of rv_20 in a local context
    if "rv_20" in out.columns and z_window:
        mu = out["rv_20"].rolling(z_window, min_periods=max(5, z_window//3)).mean()
        sd = out["rv_20"].rolling(z_window, min_periods=max(5, z_window//3)).std(ddof=0)
        out[f"rv_z_{z_window}"] = ((out["rv_20"] - mu) / sd.replace(0, np.nan)).astype("float32")

    # 5) Volatility‑of‑volatility (price vol of vol)
    if "rv_20" in out.columns:
        d_rv20 = pd.to_numeric(out["rv_20"], errors="coerce").diff()
        out["vol_of_vol_20d"] = d_rv20.rolling(20, min_periods=5).std(ddof=0).astype("float32")

    # 6) Directional vol trend (normalized slopes)
    def _norm_slope(s: pd.Series, win: int) -> pd.Series:
        if s is None:
            return pd.Series(index=out.index, dtype="float32")
        # percent change per bar over 'win' bars
        pct_per_bar = (s / s.shift(win) - 1.0) / float(win)
        return pct_per_bar.replace([np.inf, -np.inf], np.nan).astype("float32")

    if "rv_60" in out.columns:
        out["rv60_slope_norm"] = _norm_slope(out["rv_60"], slope_win)
    if "rv_100" in out.columns:
        out["rv100_slope_norm"] = _norm_slope(out["rv_100"], slope_win)

    # 7) Cross‑sectional regime context (optional)
    #    Pass cs_ratio_median as the universe‑median of rv_ratio_20_100 for each date.
    if cs_ratio_median is not None and "vol_regime" in out.columns:
        cs_med = pd.to_numeric(cs_ratio_median.reindex(out.index), errors="coerce")
        vol_regime_cs = np.log1p(cs_med)  # keep same transform as vol_regime
        out["vol_regime_cs_median"] = vol_regime_cs.astype("float32")
        out["vol_regime_rel"] = (out["vol_regime"] - out["vol_regime_cs_median"]).astype("float32")

    # 8) Quiet‑trend interaction (only if upstream trend exists)
    if "trend_score_granular" in out.columns and "vol_regime_ema10" in out.columns:
        # Emphasize trend in quiet regimes; suppress in turbulent regimes
        quiet_gate = (out["vol_regime_ema10"] < 0).astype("float32")
        out["quiet_trend"] = (out["trend_score_granular"] * quiet_gate).astype("float32")

    return out
# -----------------------------------------------------------

def add_distance_to_ma_features(
    df: pd.DataFrame,
    src_col='adjclose',
    ma_lengths=(20,50,100,200),
    z_window=60
) -> pd.DataFrame:
    if src_col not in df.columns:
        return df
    px = pd.to_numeric(df[src_col], errors='coerce')
    pct_cols = []
    for L in ma_lengths:
        ma = _ensure_ma(df, src=src_col, p=L)
        col = f"pct_dist_ma_{L}"
        df[col] = (px - ma) / ma.replace(0, np.nan)
        pct_cols.append(col)
        if z_window:
            m = df[col].rolling(z_window, min_periods=max(5, z_window//3)).mean()
            s = df[col].rolling(z_window, min_periods=max(5, z_window//3)).std(ddof=0)
            df[f"{col}_z"] = (df[col] - m) / s.replace(0, np.nan)
    if pct_cols:
        df["min_pct_dist_ma"] = df[pct_cols].abs().min(axis=1)
        if "ma_20" in df.columns and "ma_50" in df.columns:
            df["relative_dist_20_50"] = (df["ma_20"] - df["ma_50"]) / df["ma_50"].replace(0, np.nan)
            if z_window:
                m = df["relative_dist_20_50"].rolling(z_window, min_periods=max(5, z_window//3)).mean()
                s = df["relative_dist_20_50"].rolling(z_window, min_periods=max(5, z_window//3)).std(ddof=0)
                df["relative_dist_20_50_z"] = (df["relative_dist_20_50"] - m) / s.replace(0, np.nan)
    return df

def add_range_breakout_features(
    df: pd.DataFrame,
    win_list=(5,10,20)
) -> pd.DataFrame:
    if not set(["high","low"]).issubset(df.columns):
        return df
    close = pd.to_numeric(df["close"] if "close" in df.columns else df["adjclose"], errors='coerce')
    high  = pd.to_numeric(df["high"], errors='coerce')
    low   = pd.to_numeric(df["low"],  errors='coerce')

    prev_close = close.shift(1)
    hl_range = (high - low)
    df["hl_range"] = hl_range
    df["hl_range_pct_close"] = hl_range / close.replace(0, np.nan)

    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    df["true_range"] = tr
    df["tr_pct_close"] = tr / close.replace(0, np.nan)
    atr = tr.rolling(14, min_periods=5).mean()
    df["atr_percent"] = (atr / close.replace(0, np.nan)).astype('float32')

    df["gap_pct"] = (close / prev_close - 1.0)
    df["gap_atr_ratio"] = df["gap_pct"] / df["atr_percent"].replace(0, np.nan)

    for w in win_list:
        hi = high.rolling(w, min_periods=max(2, w//3)).max()
        lo = low .rolling(w, min_periods=max(2, w//3)).min()
        rng = hi - lo
        df[f"{w}d_high"]  = hi
        df[f"{w}d_low"]   = lo
        df[f"{w}d_range"] = rng
        df[f"{w}d_range_pct_close"] = rng / close.replace(0, np.nan)
        df[f"pos_in_{w}d_range"] = (close - lo) / rng.replace(0, np.nan)
        df[f"breakout_up_{w}d"] = (close > hi.shift(1)).astype('float32')
        df[f"breakout_dn_{w}d"] = (close < lo.shift(1)).astype('float32')
        df[f"range_expansion_{w}d"] = (rng / rng.shift(1) - 1.0)
        mu = rng.rolling(60, min_periods=20).mean()
        sd = rng.rolling(60, min_periods=20).std(ddof=0)
        df[f"range_z_{w}d"] = (rng - mu) / sd.replace(0, np.nan)

    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors='coerce')
        vol_ma20 = vol.rolling(20, min_periods=5).mean()
        df["range_x_rvol20"] = (hl_range / close.replace(0, np.nan)) / (vol / vol_ma20.replace(0, np.nan))
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        return df
    vol = pd.to_numeric(df["volume"], errors='coerce')
    df["vol_ma_20"] = vol.rolling(20, min_periods=5).mean()
    df["vol_ma_50"] = vol.rolling(50, min_periods=10).mean()

    mu20 = vol.rolling(20, min_periods=5).mean()
    sd20 = vol.rolling(20, min_periods=5).std(ddof=0)
    df["vol_z_20"] = (vol - mu20) / sd20.replace(0, np.nan)

    mu60 = vol.rolling(60, min_periods=15).mean()
    sd60 = vol.rolling(60, min_periods=15).std(ddof=0)
    df["vol_z_60"] = (vol - mu60) / sd60.replace(0, np.nan)

    df["rvol_20"] = vol / df["vol_ma_20"].replace(0, np.nan)
    df["rvol_50"] = vol / df["vol_ma_50"].replace(0, np.nan)

    px_col = "adjclose" if "adjclose" in df.columns else ("close" if "close" in df.columns else None)
    if px_col:
        px = pd.to_numeric(df[px_col], errors='coerce')
        dvol = px * vol
        dvol_ma20 = dvol.rolling(20, min_periods=5).mean()
        df["dollar_vol_ma_20"] = dvol_ma20
        df["rdollar_vol_20"]   = dvol / dvol_ma20.replace(0, np.nan)

        obv = (np.sign(px.diff()).fillna(0.0) * vol).fillna(0.0).cumsum()
        df["obv"] = obv
        obv_mu = obv.rolling(60, min_periods=20).mean()
        obv_sd = obv.rolling(60, min_periods=20).std(ddof=0)
        df["obv_z_60"] = (obv - obv_mu) / obv_sd.replace(0, np.nan)

    # volatility of **volume** (you already had this; leaving as-is)
    lv = np.log(vol.replace(0, np.nan))
    d_lv = lv.diff()
    df["vol_rolling_20d"] = vol.rolling(20, min_periods=5).mean()
    df["vol_rolling_60d"] = vol.rolling(60, min_periods=15).mean()
    df["vol_of_vol_20d"]  = d_lv.rolling(20, min_periods=5).std(ddof=0)

    return df

# ============================================================
# Relative strength
# ============================================================
def add_relative_strength(indicators_by_symbol: Dict[str, pd.DataFrame],
                          sectors: Optional[Dict[str, str]] = None,
                          sector_to_etf: Optional[Dict[str, str]] = None,
                          spy_symbol: str = SPY_SYMBOL) -> None:
    def _compute_block(price: pd.Series, bench: pd.Series, look=60, slope_win=20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        bench_a = pd.to_numeric(bench.reindex(price.index), errors='coerce').replace(0, np.nan)
        rs = price / bench_a
        roll = rs.rolling(look, min_periods=max(5, look//3)).mean()
        rs_norm = (rs / roll) - 1.0
        rs_slope = (rs - rs.shift(slope_win)) / float(slope_win)
        return rs.astype('float32'), rs_norm.astype('float32'), rs_slope.astype('float32')

    spy = indicators_by_symbol.get(spy_symbol)
    spy_px = None
    if spy is not None and "adjclose" in spy.columns:
        spy_px = pd.to_numeric(spy["adjclose"], errors='coerce')

    sector_to_etf = (sector_to_etf or {})
    sector_to_etf_lc = {k.lower(): v for k, v in sector_to_etf.items()}

    sector_benchmarks: Dict[str, pd.Series] = {}
    if sectors:
        etfs_needed = set()
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sector_to_etf_lc.get(sec.lower())
                if etf:
                    etfs_needed.add(etf)
        for etf in etfs_needed:
            etf_df = indicators_by_symbol.get(etf)
            if etf_df is not None and "adjclose" in etf_df.columns:
                sector_benchmarks[etf] = pd.to_numeric(etf_df["adjclose"], errors='coerce')

    rs_spy_count = 0
    rs_sector_count = 0
    for sym, df in indicators_by_symbol.items():
        if "adjclose" not in df.columns:
            continue
        px = pd.to_numeric(df["adjclose"], errors='coerce')

        if spy_px is not None and sym != spy_symbol:
            rs, rsn, rss = _compute_block(px, spy_px)
            df["rel_strength_spy"] = rs
            df["rel_strength_spy_norm"] = rsn
            df["rel_strength_spy_slope20"] = rss
            rs_spy_count += 1

        if sectors:
            sec = sectors.get(sym)
            etf = sector_to_etf_lc.get(sec.lower()) if isinstance(sec, str) else None
            bench = sector_benchmarks.get(etf) if etf else None
            if bench is not None:
                rs, rsn, rss = _compute_block(px, bench)
                df["rel_strength_sector"] = rs
                df["rel_strength_sector_norm"] = rsn
                df["rel_strength_sector_slope20"] = rss
                rs_sector_count += 1

    print(f"✅ Added SPY RS to {rs_spy_count} symbols, Sector RS to {rs_sector_count} symbols")

def _rolling_beta_alpha(ret, bench_ret, win):
    """
    Rolling CAPM beta & alpha:
      beta_t = Cov(ret, bench)/Var(bench)
      alpha_t = ret - beta_t * bench
    """
    ret = pd.to_numeric(ret, errors='coerce')
    bench_ret = pd.to_numeric(bench_ret, errors='coerce')

    cov = (ret.rolling(win).cov(bench_ret))
    var = (bench_ret.rolling(win).var())
    beta = cov / var.replace(0, np.nan)
    alpha = ret - beta * bench_ret
    return beta.astype('float32'), alpha.astype('float32')

def _alpha_momentum_from_residual(alpha_ret: pd.Series, windows=(20, 60, 120), ema_span=10, prefix="alpha_mom"):
    out = {}
    # EMA of residual returns (fast “flow”)
    out[f"{prefix}_ema{ema_span}"] = alpha_ret.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype('float32')
    # Windowed sums (cumulative alpha over horizon)
    for w in windows:
        mom = alpha_ret.rolling(w, min_periods=max(3, w//3)).sum()
        out[f"{prefix}_{w}_ema{ema_span}"] = mom.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype('float32')
    return out

def add_alpha_momentum_features(
    indicators_by_symbol: dict,
    sectors: dict = None,                    # dict[symbol -> sector_name] (optional)
    sector_to_etf: dict = None,              # dict[lower-sector-name -> ETF ticker] (optional)
    market_symbol: str = "SPY",
    beta_win: int = 60,                       # rolling window for beta/alpha
    windows=(20, 60, 120),                    # momentum horizons
    ema_span: int = 10
):
    """
    Mutates indicators_by_symbol in place, adding alpha momentum features.
    Requires each symbol df to have 'adjclose' and 'ret' (log-return).
    """
    # --- Market benchmark (SPY) ---
    mkt_df = indicators_by_symbol.get(market_symbol)
    if mkt_df is None or 'ret' not in mkt_df.columns:
        print(f"⚠️ {market_symbol} missing or has no 'ret' — only sector alphas (if any) will be computed.")
        mkt_ret = None
    else:
        mkt_ret = pd.to_numeric(mkt_df['ret'], errors='coerce')

    # --- Pre-build sector ETF returns (if mapping available) ---
    sec_map_lc = {k.lower(): v for k, v in (sector_to_etf or {}).items()}
    sector_bench_ret = {}
    if sectors and sector_to_etf:
        # Find all ETFs we need
        needed = set()
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sec_map_lc.get(sec.lower())
                if etf: needed.add(etf)
        # Build returns for each ETF
        for etf in needed:
            df_etf = indicators_by_symbol.get(etf)
            if df_etf is not None and 'ret' in df_etf.columns:
                sector_bench_ret[etf] = pd.to_numeric(df_etf['ret'], errors='coerce')

    added_count = 0
    for sym, df in indicators_by_symbol.items():
        if 'ret' not in df.columns or 'adjclose' not in df.columns:
            continue
        idx = df.index
        r = pd.to_numeric(df['ret'], errors='coerce')

        # ========== Alpha vs Market ==========
        if mkt_ret is not None and sym != market_symbol:
            beta_mkt, alpha_mkt = _rolling_beta_alpha(r, mkt_ret.reindex(idx), win=beta_win)
            df['alpha_resid_spy'] = alpha_mkt
            # Momentum transforms
            mkt_feats = _alpha_momentum_from_residual(alpha_mkt, windows=windows, ema_span=ema_span, prefix="alpha_mom_spy")
            for k, v in mkt_feats.items():
                df[k] = v

        # ========== Alpha vs Sector ETF (if available) ==========
        alpha_sec = None
        if sectors and sector_to_etf:
            sec = sectors.get(sym)
            etf = sec_map_lc.get(sec.lower()) if isinstance(sec, str) else None
            if etf and etf in sector_bench_ret:
                sec_ret = sector_bench_ret[etf].reindex(idx)
                _, alpha_sec = _rolling_beta_alpha(r, sec_ret, win=beta_win)
                df['alpha_resid_sector'] = alpha_sec
                sec_feats = _alpha_momentum_from_residual(alpha_sec, windows=windows, ema_span=ema_span, prefix="alpha_mom_sector")
                for k, v in sec_feats.items():
                    df[k] = v

        # ========== Blended (if both exist) ==========
        if ('alpha_resid_spy' in df.columns) and (alpha_sec is not None):
            alpha_combo = 0.5 * df['alpha_resid_spy'] + 0.5 * df['alpha_resid_sector']
            combo_feats = _alpha_momentum_from_residual(alpha_combo, windows=windows, ema_span=ema_span, prefix="alpha_mom_combo")
            for k, v in combo_feats.items():
                df[k] = v

        added_count += 1

    print(f"✅ Alpha momentum features added for {added_count} symbols "
          f"(β window={beta_win}, horizons={list(windows)}, EMA={ema_span}).")

# ============================================================
# Breadth: % above MA and AD line for a list (e.g., S&P 500)
# ============================================================
def add_breadth_series(indicators_by_symbol: Dict[str, pd.DataFrame],
                       universe_tickers: List[str]) -> None:
    print(f"🔍 Breadth: {len(universe_tickers)} universe tickers provided")
    print(f"🔍 Available symbols: {len(indicators_by_symbol)}")
    tickers = [t for t in universe_tickers if t in indicators_by_symbol]
    print(f"🔍 Matched {len(tickers)} tickers for breadth calculation")

    if len(tickers) < 10:
        print(f"⚠️ Few matches found. Sample universe: {universe_tickers[:5]}")
        print(f"⚠️ Sample available: {list(indicators_by_symbol.keys())[:5]}")

    def _above_ma_series(p, L):
        if p.isna().all():
            return pd.Series(index=p.index, dtype='float32')
        ma = p.rolling(L, min_periods=max(1, L//2)).mean()
        return (p > ma).astype('float32')

    above20, above50, above200, advdec = {}, {}, {}, {}
    processed_count = 0
    for sym in tickers:
        df = indicators_by_symbol[sym]
        price_col = "close" if "close" in df.columns else ("adjclose" if "adjclose" in df.columns else None)
        if price_col is None:
            continue
        px = pd.to_numeric(df[price_col], errors='coerce')
        if px.isna().all():
            continue
        ret1 = px.pct_change()
        advdec[sym] = np.sign(ret1).fillna(0.0)
        above20[sym]  = _above_ma_series(px, 20)
        above50[sym]  = _above_ma_series(px, 50)
        above200[sym] = _above_ma_series(px, 200)
        processed_count += 1

    print(f"✅ Processed {processed_count} symbols for breadth")
    if not above50:
        return

    def _pct(df_map: Dict[str, pd.Series]) -> pd.Series:
        panel = pd.DataFrame(df_map)
        return (panel.mean(axis=1, skipna=True) * 100.0).rename(None)

    pct20  = _pct(above20).rename("pct_universe_above_ma20")
    pct50  = _pct(above50).rename("pct_universe_above_ma50")
    pct200 = _pct(above200).rename("pct_universe_above_ma200")

    ad_panel = pd.DataFrame(advdec)
    ad_net = ad_panel.sum(axis=1, skipna=True)
    ad_line = ad_net.cumsum().rename("ad_line_universe")

    for sym, df in indicators_by_symbol.items():
        for s in [pct20, pct50, pct200, ad_line]:
            if len(s) > 0:
                df[s.name] = pd.to_numeric(s.reindex(df.index), errors='coerce').astype('float32')

# ============================================================
# Save helpers
# ============================================================
def save_symbol_frames(indicators_by_symbol: Dict[str, pd.DataFrame],
                       out_dir: Path = SYMBOL_FRAMES_DIR) -> None:
    for sym, df in indicators_by_symbol.items():
        df.to_parquet(out_dir / f"{sym}.parquet", engine="pyarrow", compression="snappy")

def save_long_parquet(indicators_by_symbol: Dict[str, pd.DataFrame],
                      out_path: Path = OUTPUT_DIR / "features_long.parquet") -> None:
    parts = []
    for sym, df in indicators_by_symbol.items():
        x = df.copy()
        x["symbol"] = sym
        x["date"] = x.index
        parts.append(x)
    long_df = pd.concat(parts, axis=0, ignore_index=True)
    long_df["symbol"] = long_df["symbol"].astype("string")
    long_df["date"]   = pd.to_datetime(long_df["date"])
    long_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

# ============================================================
# Main
# ============================================================
def main(include_sectors: bool = True) -> None:
    sp500_tickers = SP500_TICKERS

    # 1) Load stocks + ETFs
    stocks, sectors = load_stock_universe(max_symbols=MAX_STOCKS, update=False,
                                 rate_limit=RATE_LIMIT_REQ_PER_SEC, interval=INTERVAL, include_sectors=True)
    if not stocks:
        raise RuntimeError("Failed to load stock universe.")
    etfs = load_etf_universe(etf_symbols=DEFAULT_ETFS, update=False,
                             rate_limit=RATE_LIMIT_REQ_PER_SEC, interval=INTERVAL)
    if not etfs:
        raise RuntimeError("Failed to load ETF universe.")

    # 2) Assemble per‑symbol frames (lower-case)
    data = {
        k: pd.concat([stocks.get(k, pd.DataFrame()),
                      etfs.get(k, pd.DataFrame())], axis=1).sort_index()
        for k in (set(stocks) | set(etfs))
    }
    indicators_by_symbol = assemble_indicators_from_wide(data, adjust_ohlc_with_factor=True)

    # 4) Core features (parallelized)
    print(f"🔧 Processing {len(indicators_by_symbol)} symbols in parallel...")
    results = Parallel(
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        backend="loky",
        batch_size=8,
        verbose=0,
    )(
        delayed(_feature_worker)(sym, df) for sym, df in indicators_by_symbol.items()
    )

    indicators_by_symbol = {sym: df for sym, df in results}
    print(f"✅ Core features completed for {len(indicators_by_symbol)} symbols")

    # 4.1) Cross-sectional volatility regime context
    print("🔧 Adding cross-sectional volatility regime context...")
    add_vol_regime_cs_context(indicators_by_symbol)
    print("✅ Cross-sectional volatility regime context added")

    # 5) Alpha-momentum features (requires access to all symbols, so done after parallel processing)
    print("🔧 Adding alpha-momentum features...")
    add_alpha_momentum_features(
        indicators_by_symbol,
        sectors=sectors,
        sector_to_etf=SECTOR_TO_ETF,
        market_symbol=SPY_SYMBOL,
        beta_win=60,
        windows=(20, 60, 120),
        ema_span=10
    )
    print("✅ Alpha-momentum features added")

    # 6) Relative strength vs SPY and sectors
    print("🔧 Adding relative strength features...")
    add_relative_strength(indicators_by_symbol, sectors=sectors, sector_to_etf=SECTOR_TO_ETF, spy_symbol=SPY_SYMBOL)
    print("✅ Relative strength features added")

    # 7) Breadth (attach to all symbols) — if S&P500 list is provided
    if sp500_tickers:
        print("🔧 Adding breadth series...")
        add_breadth_series(indicators_by_symbol, sp500_tickers)
        print("✅ Breadth series added")

    # Add x‑sec momentum (plain and/or sector‑neutral if you have `sectors: dict[sym->sector]`)
    add_xsec_momentum_panel(
        indicators_by_symbol,
        lookbacks=(5, 20, 60),
        price_col="adjclose",
        sector_map=sectors  # or None if you don’t want sector-neutralization
    )

    # 8) Save
    save_symbol_frames(indicators_by_symbol, out_dir=SYMBOL_FRAMES_DIR)
    save_long_parquet(indicators_by_symbol, out_path=OUTPUT_DIR / "features_long.parquet")
    print(f"✅ Saved features → {OUTPUT_DIR}")

if __name__ == "__main__":
    main(include_sectors=True)