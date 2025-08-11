"""
Alpha momentum features based on CAPM residuals and risk-adjusted returns.

This module computes alpha (excess return) features by calculating rolling CAPM
beta and alpha against market and sector benchmarks, then deriving momentum
features from these alpha streams.
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_beta_alpha(ret: pd.Series, bench_ret: pd.Series, win: int) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling CAPM beta and alpha using rolling covariance and variance.
    
    Args:
        ret: Return series for the security
        bench_ret: Benchmark return series
        win: Rolling window for beta/alpha calculation
        
    Returns:
        Tuple of (beta, alpha) where:
        - beta_t = Cov(ret, bench_ret) / Var(bench_ret)
        - alpha_t = ret - beta_t * bench_ret
    """
    ret = pd.to_numeric(ret, errors='coerce')
    bench_ret = pd.to_numeric(bench_ret, errors='coerce')

    # Rolling covariance and variance
    cov = ret.rolling(win).cov(bench_ret)
    var = bench_ret.rolling(win).var()
    
    # Beta calculation
    beta = cov / var.replace(0, np.nan)
    
    # Alpha: actual return minus expected return (beta * benchmark return)
    alpha = ret - beta * bench_ret
    
    return beta.astype('float32'), alpha.astype('float32')


def _alpha_momentum_from_residual(
    alpha_ret: pd.Series, 
    windows: Tuple[int, ...] = (20, 60, 120), 
    ema_span: int = 10, 
    prefix: str = "alpha_mom"
) -> Dict[str, pd.Series]:
    """
    Create momentum features from alpha residual return series.
    
    Args:
        alpha_ret: Alpha residual return series
        windows: Windows for cumulative alpha momentum
        ema_span: EMA span for smoothing
        prefix: Prefix for feature names
        
    Returns:
        Dictionary of momentum features
    """
    out = {}
    
    # EMA of residual returns (fast "flow" indicator)
    out[f"{prefix}_ema{ema_span}"] = (
        alpha_ret.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype('float32')
    )
    
    # Windowed cumulative alpha with EMA smoothing
    for w in windows:
        # Cumulative alpha over horizon (sum of residual returns)
        mom = alpha_ret.rolling(w, min_periods=max(3, w//3)).sum()
        # EMA smoothing of cumulative alpha
        out[f"{prefix}_{w}_ema{ema_span}"] = (
            mom.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype('float32')
        )
    
    return out


def add_alpha_momentum_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    sectors: Optional[Dict[str, str]] = None,
    sector_to_etf: Optional[Dict[str, str]] = None,
    market_symbol: str = "SPY",
    beta_win: int = 60,
    windows: Tuple[int, ...] = (20, 60, 120),
    ema_span: int = 10
) -> None:
    """
    Add alpha momentum features based on CAPM residuals vs market and sector benchmarks.
    
    Features added (if benchmarks available):
    - alpha_resid_spy: CAPM alpha residuals vs market
    - alpha_mom_spy_ema{ema_span}: EMA of market alpha residuals  
    - alpha_mom_spy_{w}_ema{ema_span}: EMA of {w}-day cumulative market alpha
    - alpha_resid_sector: CAPM alpha residuals vs sector
    - alpha_mom_sector_ema{ema_span}: EMA of sector alpha residuals
    - alpha_mom_sector_{w}_ema{ema_span}: EMA of {w}-day cumulative sector alpha
    - alpha_mom_combo_*: Blended features (50/50 market/sector) if both exist
    
    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        sectors: Optional mapping of symbol -> sector name
        sector_to_etf: Optional mapping of lowercase sector name -> ETF symbol  
        market_symbol: Symbol to use as market benchmark
        beta_win: Rolling window for beta/alpha calculation
        windows: Momentum calculation windows
        ema_span: EMA span for smoothing momentum features
        
    Raises:
        None, but logs warnings if benchmark data is missing
    """
    logger.info(f"Computing alpha momentum features using {market_symbol} as market benchmark")
    
    # Market benchmark (SPY) data
    mkt_df = indicators_by_symbol.get(market_symbol)
    if mkt_df is None or 'ret' not in mkt_df.columns:
        logger.warning(f"{market_symbol} missing or has no 'ret' column - only sector alphas will be computed")
        mkt_ret = None
    else:
        mkt_ret = pd.to_numeric(mkt_df['ret'], errors='coerce')
        logger.debug(f"Using {market_symbol} returns as market benchmark")

    # Pre-build sector ETF returns (if mapping available)
    sec_map_lc = {k.lower(): v for k, v in (sector_to_etf or {}).items()}
    sector_bench_ret = {}
    
    if sectors and sector_to_etf:
        # Find all ETFs we need
        needed = set()
        for sym, sec in sectors.items():
            if isinstance(sec, str):
                etf = sec_map_lc.get(sec.lower())
                if etf: 
                    needed.add(etf)
        
        # Build returns for each ETF
        for etf in needed:
            df_etf = indicators_by_symbol.get(etf)
            if df_etf is not None and 'ret' in df_etf.columns:
                sector_bench_ret[etf] = pd.to_numeric(df_etf['ret'], errors='coerce')
        
        logger.debug(f"Loaded {len(sector_bench_ret)} sector benchmark returns")

    added_count = 0
    
    for sym, df in indicators_by_symbol.items():
        if 'ret' not in df.columns or 'adjclose' not in df.columns:
            continue
            
        idx = df.index
        r = pd.to_numeric(df['ret'], errors='coerce')

        # Market alpha (vs SPY)
        if mkt_ret is not None and sym != market_symbol:
            beta_mkt, alpha_mkt = _rolling_beta_alpha(r, mkt_ret.reindex(idx), win=beta_win)
            df['alpha_resid_spy'] = alpha_mkt
            
            # Create momentum features from market alpha
            mkt_feats = _alpha_momentum_from_residual(
                alpha_mkt, windows=windows, ema_span=ema_span, prefix="alpha_mom_spy"
            )
            for k, v in mkt_feats.items():
                df[k] = v

        # Sector alpha (vs sector ETF if available)
        alpha_sec = None
        if sectors and sector_to_etf:
            sec = sectors.get(sym)
            etf = sec_map_lc.get(sec.lower()) if isinstance(sec, str) else None
            if etf and etf in sector_bench_ret:
                sec_ret = sector_bench_ret[etf].reindex(idx)
                _, alpha_sec = _rolling_beta_alpha(r, sec_ret, win=beta_win)
                df['alpha_resid_sector'] = alpha_sec
                
                # Create momentum features from sector alpha
                sec_feats = _alpha_momentum_from_residual(
                    alpha_sec, windows=windows, ema_span=ema_span, prefix="alpha_mom_sector"
                )
                for k, v in sec_feats.items():
                    df[k] = v

        # Blended alpha features (if both market and sector exist)
        if ('alpha_resid_spy' in df.columns) and (alpha_sec is not None):
            alpha_combo = 0.5 * df['alpha_resid_spy'] + 0.5 * alpha_sec
            combo_feats = _alpha_momentum_from_residual(
                alpha_combo, windows=windows, ema_span=ema_span, prefix="alpha_mom_combo"
            )
            for k, v in combo_feats.items():
                df[k] = v

        added_count += 1

    logger.info(f"Alpha momentum features added for {added_count} symbols "
                f"(beta window={beta_win}, horizons={list(windows)}, EMA={ema_span})")