"""
Joint Factor Regression Block for clean exposure computation.

This module computes factor exposures and alpha jointly via multivariate rolling
ridge regression. Instead of computing betas one factor at a time, this approach
computes all factor exposures simultaneously, producing cleaner estimates that
properly account for factor correlations.

Factor Model (orthogonalized design):
    R_stock = α + β_market × R_SPY
                + β_qqq × (R_QQQ - R_SPY)
                + β_bestmatch × (R_bestmatch - R_SPY)
                + β_breadth × (R_RSP - R_SPY)
                + ε

Factors:
- market: R_SPY (broad market exposure)
- qqq: R_QQQ - R_SPY (growth/tech premium over market)
- bestmatch: R_bestmatch - R_SPY (sector/subsector premium over market)
- breadth: R_RSP - R_SPY (equal-weight vs cap-weight spread)

The spread factors are orthogonalized vs market to avoid multicollinearity
and provide cleaner interpretation of factor loadings.

Features computed per frequency (daily/weekly) and per window:
- beta_market, beta_qqq, beta_bestmatch, beta_breadth (factor exposures)
- alpha (regression intercept / idiosyncratic return)
- residual_mean, residual_cumret, residual_vol (residual statistics)
"""
import gc
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import Ridge

# Import parallel config for stocks_per_worker based parallelism
try:
    from ..config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER
except ImportError:
    from src.config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER

# Import spread features for bestmatch spread computation
try:
    from .spread_features import (
        compute_bestmatch_spread_features,
        compute_bestmatch_ew_spread_features,
        DAILY_WINDOWS as SPREAD_DAILY_WINDOWS
    )
except ImportError:
    from src.features.spread_features import (
        compute_bestmatch_spread_features,
        compute_bestmatch_ew_spread_features,
        DAILY_WINDOWS as SPREAD_DAILY_WINDOWS
    )

logger = logging.getLogger(__name__)

# Default beta windows - matches existing alpha.py
DAILY_BETA_WINDOWS = [60]  # Existing window from alpha.py
WEEKLY_BETA_WINDOWS = [12]  # ~60 days = ~12 weeks

# Factor ETF definitions
MARKET_FACTOR = 'SPY'
GROWTH_FACTOR = 'QQQ'
BREADTH_SPREAD = ('RSP', 'SPY')  # RSP - SPY spread for concentration/breadth

# Best-match candidate pool (cap-weighted sector + subsector ETFs, excluding broad market)
BEST_MATCH_CANDIDATES = [
    # Sector ETFs (cap-weighted)
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLC', 'XLRE',
    # Tech subsectors
    'SMH', 'IGV', 'SKYY', 'HACK',
    # Finance subsectors
    'KBE', 'KRE',
    # Healthcare/Biotech
    'IBB', 'XBI', 'IHE',
    # Industrial subsectors
    'ITA', 'ITB',
    # Energy subsectors
    'XOP',
    # Consumer subsectors
    'XRT',
    # Clean energy/materials
    'TAN', 'URA', 'LIT', 'COPX',
]

# Equal-weight best-match candidate pool (Invesco S&P 500 Equal Weight Sector ETFs)
# Symbols follow pattern: RSP + sector letter (matching cap-weighted XL* sectors)
BEST_MATCH_EW_CANDIDATES = [
    'RSPT',  # Technology (XLK equal weight)
    'RSPF',  # Financials (XLF equal weight)
    'RSPH',  # Health Care (XLV equal weight)
    'RSPE',  # Energy (XLE equal weight)
    'RSPN',  # Industrials (XLI equal weight)
    'RSPC',  # Consumer Discretionary (XLY equal weight)
    'RSPS',  # Consumer Staples (XLP equal weight)
    'RSPU',  # Utilities (XLU equal weight)
    'RSPM',  # Materials (XLB equal weight)
    'RSPD',  # Communication Services (XLC equal weight)
    # Note: Real Estate (XLRE) doesn't have an RSP equivalent
    # The algorithm will gracefully handle missing ETFs
]


def _resample_returns_to_weekly(
    ret_index: pd.Index,
    ret_values: np.ndarray
) -> Tuple[pd.Index, np.ndarray, pd.Series]:
    """
    Resample daily returns to weekly returns.

    Weekly returns are computed as the cumulative (compounded) return over each week.
    Uses Friday close (week-end) as the weekly timestamp.

    Args:
        ret_index: Daily DatetimeIndex
        ret_values: Daily return values

    Returns:
        Tuple of:
        - weekly_index: Weekly DatetimeIndex (Fridays)
        - weekly_values: Weekly return values (compounded daily returns)
        - daily_to_weekly_map: Series mapping daily dates to weekly dates (for ffill back)
    """
    daily_ret = pd.Series(ret_values, index=ret_index, dtype='float64')

    # Resample to weekly using Friday end-of-week
    # Weekly return = product of (1 + daily_ret) - 1
    weekly_ret = daily_ret.resample('W-FRI').apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
    )

    # Create mapping from daily to weekly dates (for forward-filling results back)
    # Each daily date maps to its Friday week-end
    daily_to_weekly = daily_ret.resample('W-FRI').apply(lambda x: x.index[-1] if len(x) > 0 else pd.NaT)
    # Actually we need the other direction: for each daily date, what's its week's Friday
    weekly_dates = pd.Series(index=daily_ret.index, dtype='datetime64[ns]')
    for week_end, group in daily_ret.groupby(pd.Grouper(freq='W-FRI')):
        if len(group) > 0:
            weekly_dates.loc[group.index] = week_end

    return weekly_ret.index, weekly_ret.values, weekly_dates


def _resample_factor_returns_to_weekly(
    factor_returns: Dict[str, Tuple[pd.Index, np.ndarray]]
) -> Dict[str, Tuple[pd.Index, np.ndarray]]:
    """
    Resample all factor returns from daily to weekly.

    Args:
        factor_returns: Dict of factor name -> (daily_index, daily_values)

    Returns:
        Dict of factor name -> (weekly_index, weekly_values)
    """
    weekly_factor_returns = {}
    for factor_name, (ret_index, ret_values) in factor_returns.items():
        weekly_index, weekly_values, _ = _resample_returns_to_weekly(ret_index, ret_values)
        weekly_factor_returns[factor_name] = (weekly_index, weekly_values)
    return weekly_factor_returns


def _get_required_factors_for_batch(
    work_items: List[Tuple[str, pd.Index, np.ndarray, Optional[str], Optional[str]]],
    all_factor_returns: Dict[str, Tuple[pd.Index, np.ndarray]]
) -> Dict[str, Tuple[pd.Index, np.ndarray]]:
    """
    Extract only the factor returns needed by a specific batch of symbols.

    This reduces memory usage by only sending relevant factor data to each worker
    instead of the entire factor returns dictionary.

    Args:
        work_items: List of (symbol, ret_index, ret_values, best_match_etf, best_match_ew_etf) tuples
        all_factor_returns: Complete factor returns dictionary

    Returns:
        Subset of factor returns needed by this batch
    """
    # Core factors always needed
    required = {MARKET_FACTOR, GROWTH_FACTOR, BREADTH_SPREAD[0], BREADTH_SPREAD[1]}

    # Add best-match ETFs (cap-weighted and equal-weight) for symbols in this batch
    for item in work_items:
        if len(item) >= 4 and item[3]:  # best_match_etf (cap-weighted)
            required.add(item[3])
        if len(item) >= 5 and item[4]:  # best_match_ew_etf (equal-weight)
            required.add(item[4])

    # Extract only required factors
    subset = {}
    for key in required:
        if key in all_factor_returns:
            subset[key] = all_factor_returns[key]

    return subset


@dataclass
class FactorRegressionConfig:
    """Configuration for joint factor regression."""
    daily_windows: List[int] = None
    weekly_windows: List[int] = None
    ridge_alpha: float = 0.01  # L2 regularization strength
    min_periods_ratio: float = 0.33  # min_periods = window * ratio
    min_periods_floor: int = 20  # absolute minimum

    def __post_init__(self):
        if self.daily_windows is None:
            self.daily_windows = DAILY_BETA_WINDOWS.copy()
        if self.weekly_windows is None:
            self.weekly_windows = WEEKLY_BETA_WINDOWS.copy()


def _compute_rolling_ridge_regression(
    y: np.ndarray,
    X: np.ndarray,
    window: int,
    min_periods: int,
    alpha: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling multivariate ridge regression using sklearn.

    Args:
        y: Target returns array (n_obs,)
        X: Factor returns array (n_obs, n_factors)
        window: Rolling window size
        min_periods: Minimum valid observations in window
        alpha: Ridge regularization parameter (L2)

    Returns:
        Tuple of (betas, residuals) where:
        - betas: array of shape (n_obs, n_factors + 1) with [intercept, beta1, beta2, ...]
        - residuals: array of shape (n_obs,) with regression residuals
    """
    n_obs = len(y)
    n_factors = X.shape[1]

    # Output arrays
    betas = np.full((n_obs, n_factors + 1), np.nan, dtype=np.float32)
    residuals = np.full(n_obs, np.nan, dtype=np.float32)

    # Create sklearn Ridge model (fit_intercept=True handles intercept automatically)
    model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky')

    for i in range(window - 1, n_obs):
        start_idx = i - window + 1
        y_win = y[start_idx:i + 1]
        X_win = X[start_idx:i + 1]

        # Check for valid data
        valid_mask = ~(np.isnan(y_win) | np.any(np.isnan(X_win), axis=1))
        n_valid = np.sum(valid_mask)

        if n_valid < min_periods:
            continue

        y_valid = y_win[valid_mask]
        X_valid = X_win[valid_mask]

        try:
            # Fit sklearn Ridge - handles numerical stability internally
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_valid, y_valid)

            # Extract coefficients: [intercept, beta1, beta2, ...]
            betas[i, 0] = model.intercept_
            betas[i, 1:] = model.coef_

            # Compute residual at current point (if current point has valid data)
            if not np.isnan(y[i]) and not np.any(np.isnan(X[i])):
                residuals[i] = y[i] - model.predict(X[i:i+1])[0]

        except Exception:
            # Skip on any error (singular matrix, convergence issues, etc.)
            continue

    return betas, residuals


def _compute_residual_features(
    residuals: np.ndarray,
    window: int,
    min_periods: int
) -> Dict[str, np.ndarray]:
    """
    Compute features from regression residuals.

    Args:
        residuals: Array of regression residuals
        window: Rolling window for statistics
        min_periods: Minimum valid observations

    Returns:
        Dict with residual_mean, residual_cumret, residual_vol
    """
    n_obs = len(residuals)
    result = {
        'residual_mean': np.full(n_obs, np.nan, dtype=np.float32),
        'residual_cumret': np.full(n_obs, np.nan, dtype=np.float32),
        'residual_vol': np.full(n_obs, np.nan, dtype=np.float32),
    }

    for i in range(window - 1, n_obs):
        start_idx = i - window + 1
        res_win = residuals[start_idx:i + 1]
        valid_mask = ~np.isnan(res_win)
        n_valid = np.sum(valid_mask)

        if n_valid < min_periods:
            continue

        valid_res = res_win[valid_mask]
        result['residual_mean'][i] = np.mean(valid_res)
        result['residual_cumret'][i] = np.sum(valid_res)  # Cumulative residual return
        result['residual_vol'][i] = np.std(valid_res, ddof=1) if n_valid > 1 else np.nan

    return result


def find_best_match_etf(
    stock_returns: pd.Series,
    etf_returns: Dict[str, pd.Series],
    candidates: List[str],
    window: int = 60,
    min_overlap: int = 100
) -> Tuple[Optional[str], float]:
    """
    Find the best-matching reference ETF for a stock based on R-squared.

    Uses a stable selection based on R-squared from univariate regression
    over the full available history (or longest window).

    Args:
        stock_returns: Stock return series
        etf_returns: Dict of ETF return series
        candidates: List of candidate ETF symbols
        window: Lookback window for matching
        min_overlap: Minimum overlapping data points required

    Returns:
        Tuple of (best_etf_symbol, r_squared) or (None, 0.0) if no match found
    """
    best_etf = None
    best_r2 = 0.0

    # Use all available data for stable matching
    stock_ret = stock_returns.dropna()
    if len(stock_ret) < min_overlap:
        return None, 0.0

    for etf in candidates:
        if etf not in etf_returns:
            continue

        etf_ret = etf_returns[etf].dropna()

        # Find overlapping dates
        common_idx = stock_ret.index.intersection(etf_ret.index)
        if len(common_idx) < min_overlap:
            continue

        # Compute R-squared
        y = stock_ret.loc[common_idx].values
        x = etf_ret.loc[common_idx].values

        # Remove any remaining NaNs
        valid = ~(np.isnan(y) | np.isnan(x))
        if np.sum(valid) < min_overlap:
            continue

        y_valid = y[valid]
        x_valid = x[valid]

        # Simple linear regression R-squared
        try:
            # Suppress numerical warnings for regression calculations
            with np.errstate(invalid='ignore', divide='ignore'):
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)

                ss_tot = np.sum((y_valid - y_mean) ** 2)
                if ss_tot == 0:
                    continue

                # Compute beta and predicted values
                cov_xy = np.sum((x_valid - x_mean) * (y_valid - y_mean))
                var_x = np.sum((x_valid - x_mean) ** 2)
                if var_x == 0:
                    continue

                beta = cov_xy / var_x
                alpha_reg = y_mean - beta * x_mean
                y_pred = alpha_reg + beta * x_valid

                ss_res = np.sum((y_valid - y_pred) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                if r2 > best_r2:
                    best_r2 = r2
                    best_etf = etf

        except Exception:
            continue

    return best_etf, best_r2


def _find_best_match_batch(
    work_items: List[Tuple[str, pd.Series]],
    etf_returns: Dict[str, pd.Series],
    candidates: List[str],
    window: int,
    min_overlap: int
) -> List[Tuple[str, Optional[str], float]]:
    """
    Process a batch of symbols for best-match ETF computation.

    Args:
        work_items: List of (symbol, stock_returns_series) tuples
        etf_returns: Dict of ETF return series (shared across batch)
        candidates: List of candidate ETF symbols
        window: Lookback window for matching
        min_overlap: Minimum overlapping data points

    Returns:
        List of (symbol, best_etf, r_squared) tuples
    """
    results = []
    for sym, stock_ret in work_items:
        best_etf, r2 = find_best_match_etf(
            stock_ret, etf_returns, candidates, window, min_overlap
        )
        results.append((sym, best_etf, r2))
    return results


def compute_joint_factor_features_for_symbol(
    symbol: str,
    ret_index: pd.Index,
    ret_values: np.ndarray,
    factor_returns: Dict[str, Tuple[pd.Index, np.ndarray]],
    best_match_etf: Optional[str],
    config: FactorRegressionConfig,
    frequency: str = 'daily',
    best_match_ew_etf: Optional[str] = None
) -> Dict[str, pd.Series]:
    """
    Compute joint factor regression features for a single symbol.

    Args:
        symbol: Stock symbol
        ret_index: DatetimeIndex for returns
        ret_values: Return values array
        factor_returns: Dict of factor name -> (index, values) tuples
        best_match_etf: Best-match reference ETF for this stock (cap-weighted, or None)
        config: FactorRegressionConfig
        frequency: 'daily' or 'weekly'
        best_match_ew_etf: Best-match equal-weight ETF for this stock (or None)

    Returns:
        Dict of feature_name -> pd.Series
    """
    result = {}

    # Select windows based on frequency
    windows = config.daily_windows if frequency == 'daily' else config.weekly_windows
    prefix = '' if frequency == 'daily' else 'w_'

    # Build aligned factor matrix
    stock_ret = pd.Series(ret_values, index=ret_index, dtype='float64')
    n_obs = len(stock_ret)

    # Factors to include (orthogonalized design):
    # - market: R_SPY (broad market exposure)
    # - qqq: R_QQQ - R_SPY (growth premium over market)
    # - bestmatch: R_bestmatch - R_SPY (sector/subsector premium over market)
    # - breadth: R_RSP - R_SPY (equal-weight vs cap-weight spread)
    factor_names = ['market']
    factor_series = []

    # Helper to align factor returns to target index (handles weekly date mismatches)
    def align_factor_to_index(factor_idx, factor_vals, target_index):
        """Align factor returns to target index using ffill/bfill for date mismatches."""
        factor_series = pd.Series(factor_vals, index=factor_idx, dtype='float64')
        if len(factor_series) == 0 or len(target_index) == 0:
            return pd.Series(index=target_index, dtype='float64')
        # Union indices, ffill/bfill, then select target dates
        # This handles weekly data where dates may not match exactly
        combined_idx = factor_series.index.union(target_index).sort_values()
        filled = factor_series.reindex(combined_idx).ffill().bfill()
        return filled.reindex(target_index)

    # Market factor (SPY) - base factor
    if MARKET_FACTOR in factor_returns:
        mkt_idx, mkt_vals = factor_returns[MARKET_FACTOR]
        mkt_ret = align_factor_to_index(mkt_idx, mkt_vals, ret_index)
        factor_series.append(mkt_ret)
    else:
        return result  # Need market factor

    # Growth spread (QQQ - SPY) - captures growth/tech premium over market
    if GROWTH_FACTOR in factor_returns:
        qqq_idx, qqq_vals = factor_returns[GROWTH_FACTOR]
        qqq_ret = align_factor_to_index(qqq_idx, qqq_vals, ret_index)
        qqq_spread = qqq_ret - mkt_ret  # Orthogonalize vs market
        factor_series.append(qqq_spread)
        factor_names.append('qqq')

    # Best-match spread (bestmatch - SPY) - captures sector/subsector premium over market
    if best_match_etf and best_match_etf in factor_returns:
        bm_idx, bm_vals = factor_returns[best_match_etf]
        bm_ret = align_factor_to_index(bm_idx, bm_vals, ret_index)
        bm_spread = bm_ret - mkt_ret  # Orthogonalize vs market
        factor_series.append(bm_spread)
        factor_names.append('bestmatch')

    # Breadth spread (RSP - SPY) - captures equal-weight vs cap-weight
    rsp_sym, spy_sym = BREADTH_SPREAD
    if rsp_sym in factor_returns and spy_sym in factor_returns:
        rsp_idx, rsp_vals = factor_returns[rsp_sym]
        rsp_ret = align_factor_to_index(rsp_idx, rsp_vals, ret_index)
        breadth_spread = rsp_ret - mkt_ret  # Use mkt_ret directly (same as SPY)
        factor_series.append(breadth_spread)
        factor_names.append('breadth')

    # Build factor matrix
    if not factor_series:
        return result

    X = np.column_stack([s.values for s in factor_series])
    y = stock_ret.values

    # Compute for each window
    for window in windows:
        # BUG FIX: min_periods must not exceed window size
        # For weekly data with 12-week window, can't require 20 min_periods
        min_periods_computed = max(config.min_periods_floor, int(window * config.min_periods_ratio))
        min_periods = min(min_periods_computed, window)  # Cap at window size

        # Run joint ridge regression
        betas, residuals = _compute_rolling_ridge_regression(
            y, X, window, min_periods, config.ridge_alpha
        )

        # Window suffix for feature names
        win_suffix = f'_{window}' if len(windows) > 1 else ''

        # Extract betas (column 0 is intercept)
        result[f'{prefix}alpha{win_suffix}'] = pd.Series(
            betas[:, 0], index=ret_index, dtype='float32'
        )

        for i, factor_name in enumerate(factor_names):
            result[f'{prefix}beta_{factor_name}{win_suffix}'] = pd.Series(
                betas[:, i + 1], index=ret_index, dtype='float32'
            )

        # Residual features
        resid_features = _compute_residual_features(residuals, window, min_periods)
        for feat_name, feat_vals in resid_features.items():
            result[f'{prefix}{feat_name}{win_suffix}'] = pd.Series(
                feat_vals, index=ret_index, dtype='float32'
            )

    # Compute bestmatch-SPY spread features if best_match_etf is available (cap-weighted)
    if best_match_etf and best_match_etf in factor_returns and MARKET_FACTOR in factor_returns:
        bm_idx, bm_vals = factor_returns[best_match_etf]
        bm_ret = pd.Series(bm_vals, index=bm_idx, dtype='float64')
        mkt_idx, mkt_vals = factor_returns[MARKET_FACTOR]
        spy_ret = pd.Series(mkt_vals, index=mkt_idx, dtype='float64')

        # Use appropriate windows based on frequency
        spread_windows = SPREAD_DAILY_WINDOWS if frequency == 'daily' else (4, 12, 24)

        # Compute spread features (bestmatch - SPY)
        spread_features = compute_bestmatch_spread_features(
            stock_ret_index=ret_index,
            bestmatch_ret=bm_ret,
            spy_ret=spy_ret,
            windows=spread_windows,
            prefix=prefix
        )
        result.update(spread_features)

    # Compute equal-weight bestmatch-RSP spread features if best_match_ew_etf is available
    # This captures sector performance in equal-weight terms (removes large-cap concentration)
    rsp_sym = BREADTH_SPREAD[0]  # RSP
    if best_match_ew_etf and best_match_ew_etf in factor_returns and rsp_sym in factor_returns:
        ew_idx, ew_vals = factor_returns[best_match_ew_etf]
        ew_ret = pd.Series(ew_vals, index=ew_idx, dtype='float64')
        rsp_idx, rsp_vals = factor_returns[rsp_sym]
        rsp_ret = pd.Series(rsp_vals, index=rsp_idx, dtype='float64')

        # Use appropriate windows based on frequency
        spread_windows = SPREAD_DAILY_WINDOWS if frequency == 'daily' else (4, 12, 24)

        # Compute spread features (bestmatch_ew - RSP)
        ew_spread_features = compute_bestmatch_ew_spread_features(
            stock_ret_index=ret_index,
            bestmatch_ew_ret=ew_ret,
            rsp_ret=rsp_ret,
            windows=spread_windows,
            prefix=prefix
        )
        result.update(ew_spread_features)

    # Store best-match info
    if best_match_etf:
        # Store as a constant series (for reference, not ML feature)
        result[f'{prefix}bestmatch_etf'] = pd.Series(
            best_match_etf, index=ret_index, dtype='object'
        )
    if best_match_ew_etf:
        result[f'{prefix}bestmatch_ew_etf'] = pd.Series(
            best_match_ew_etf, index=ret_index, dtype='object'
        )

    return result


def compute_factor_features_batch(
    work_items: List[Tuple[str, pd.Index, np.ndarray, Optional[str], Optional[str]]],
    factor_returns: Dict[str, Tuple[pd.Index, np.ndarray]],
    config: FactorRegressionConfig,
    frequency: str = 'daily'
) -> List[Tuple[str, Dict[str, pd.Series]]]:
    """
    Process a batch of symbols for factor feature computation.

    Args:
        work_items: List of (symbol, ret_index, ret_values, best_match_etf, best_match_ew_etf) tuples
        factor_returns: Pre-extracted factor returns
        config: FactorRegressionConfig
        frequency: 'daily' or 'weekly'

    Returns:
        List of (symbol, features_dict) tuples
    """
    results = []
    for item in work_items:
        # Support both old 4-tuple and new 5-tuple format for backwards compatibility
        if len(item) == 5:
            sym, ret_index, ret_values, best_match_etf, best_match_ew_etf = item
        else:
            sym, ret_index, ret_values, best_match_etf = item
            best_match_ew_etf = None
        features = compute_joint_factor_features_for_symbol(
            sym, ret_index, ret_values, factor_returns,
            best_match_etf, config, frequency, best_match_ew_etf
        )
        results.append((sym, features))
    return results


def compute_factor_features_batch_weekly(
    work_items: List[Tuple[str, pd.Index, np.ndarray, Optional[str], Optional[str]]],
    factor_returns: Dict[str, Tuple[pd.Index, np.ndarray]],
    config: FactorRegressionConfig
) -> List[Tuple[str, Dict[str, pd.Series]]]:
    """
    Process a batch of symbols for WEEKLY factor computation with internal resampling.

    This function handles weekly resampling inside the worker, avoiding sequential
    resampling in the main thread. Each worker resamples its own stocks.

    Args:
        work_items: List of (symbol, daily_ret_index, daily_ret_values, best_match_etf, best_match_ew_etf) tuples
                   NOTE: Daily data, NOT pre-resampled weekly data
        factor_returns: Pre-extracted factor returns (already resampled to weekly)
        config: FactorRegressionConfig

    Returns:
        List of (symbol, features_dict) tuples
    """
    results = []
    for item in work_items:
        # Support both old 4-tuple and new 5-tuple format for backwards compatibility
        if len(item) == 5:
            sym, daily_ret_index, daily_ret_values, best_match_etf, best_match_ew_etf = item
        else:
            sym, daily_ret_index, daily_ret_values, best_match_etf = item
            best_match_ew_etf = None

        # Resample this stock's daily returns to weekly INSIDE the worker
        weekly_index, weekly_values, _ = _resample_returns_to_weekly(
            daily_ret_index, daily_ret_values
        )

        features = compute_joint_factor_features_for_symbol(
            sym, weekly_index, weekly_values, factor_returns,
            best_match_etf, config, 'weekly', best_match_ew_etf
        )
        results.append((sym, features))
    return results


def add_joint_factor_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    best_match_mappings: Optional[Dict[str, str]] = None,
    config: Optional[FactorRegressionConfig] = None,
    frequency: str = 'daily',
    n_jobs: int = -1,
    best_match_ew_mappings: Optional[Dict[str, str]] = None
) -> None:
    """
    Add joint factor regression features to all symbols.

    This function computes factor exposures and residuals using multivariate
    ridge regression with factors computed jointly rather than one at a time.

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame (modified in place)
        best_match_mappings: Dict of symbol -> best-match cap-weighted ETF symbol
        config: FactorRegressionConfig (None for defaults)
        frequency: 'daily' or 'weekly'
        n_jobs: Number of parallel jobs (-1 for all cores)
        best_match_ew_mappings: Dict of symbol -> best-match equal-weight ETF symbol
    """
    from joblib import Parallel, delayed

    if config is None:
        config = FactorRegressionConfig()

    logger.info(f"Computing joint factor features ({frequency})")

    # Pre-extract factor returns
    factor_returns: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    # All potential factor ETFs (cap-weighted + equal-weight)
    factor_etfs = {MARKET_FACTOR, GROWTH_FACTOR, BREADTH_SPREAD[0], BREADTH_SPREAD[1]}
    if best_match_mappings:
        factor_etfs.update(set(best_match_mappings.values()))
    if best_match_ew_mappings:
        factor_etfs.update(set(best_match_ew_mappings.values()))

    for etf in factor_etfs:
        if etf not in indicators_by_symbol:
            continue
        df = indicators_by_symbol[etf]
        if 'ret' not in df.columns:
            continue
        ret = pd.to_numeric(df['ret'], errors='coerce')
        factor_returns[etf] = (ret.index, ret.values)

    logger.debug(f"Loaded {len(factor_returns)} factor return series")

    # For weekly frequency, resample factor returns to weekly
    daily_indices = {}  # Store daily indices for mapping results back
    if frequency == 'weekly':
        logger.info("Resampling factor returns to weekly for weekly factor computation")
        factor_returns = _resample_factor_returns_to_weekly(factor_returns)

    # Build work items - for weekly, pass DAILY data and let workers handle resampling
    work_items = []
    excluded = factor_etfs | {MARKET_FACTOR, GROWTH_FACTOR}

    for sym, df in indicators_by_symbol.items():
        if sym in excluded:
            continue
        if 'ret' not in df.columns:
            continue

        ret = pd.to_numeric(df['ret'], errors='coerce')
        best_match = best_match_mappings.get(sym) if best_match_mappings else None
        best_match_ew = best_match_ew_mappings.get(sym) if best_match_ew_mappings else None

        if frequency == 'weekly':
            # Store daily index for mapping results back later
            # Pass DAILY data to workers - they will resample in parallel
            daily_indices[sym] = ret.index
            work_items.append((sym, ret.index, ret.values, best_match, best_match_ew))
        else:
            work_items.append((sym, ret.index, ret.values, best_match, best_match_ew))

    n_symbols = len(work_items)
    logger.info(f"Computing factor features for {n_symbols} symbols")

    # Calculate workers based on stocks_per_worker (200 stocks/worker by default)
    chunk_size = DEFAULT_STOCKS_PER_WORKER
    n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)

    # Select batch function based on frequency
    # Weekly uses a special batch function that handles resampling in parallel
    is_weekly = (frequency == 'weekly')
    batch_func = compute_factor_features_batch_weekly if is_weekly else compute_factor_features_batch

    # Sequential for small datasets or debugging
    if n_symbols < 10 or n_jobs == 1 or n_workers == 1:
        if is_weekly:
            results = batch_func(work_items, factor_returns, config)
        else:
            results = batch_func(work_items, factor_returns, config, frequency)
    else:
        # Parallel with batching and subsetted factor returns
        chunks = [work_items[i:i + chunk_size] for i in range(0, n_symbols, chunk_size)]

        # Pre-compute subsetted factor returns for each chunk to reduce serialization
        # Each worker only gets the factors it needs (core + batch-specific best-match ETFs)
        chunk_factor_returns = []
        for chunk in chunks:
            factor_subset = _get_required_factors_for_batch(chunk, factor_returns)
            chunk_factor_returns.append(factor_subset)

        # Log factor reduction
        total_factors = len(factor_returns)
        avg_factors = sum(len(f) for f in chunk_factor_returns) / len(chunk_factor_returns) if chunk_factor_returns else 0
        logger.info(f"Factor subsetting: {total_factors} total -> {avg_factors:.1f} avg per batch")

        try:
            if is_weekly:
                # Weekly: workers resample in parallel (no pre-resampling bottleneck)
                batch_results = Parallel(
                    n_jobs=n_workers,
                    backend='loky',
                    verbose=0
                )(
                    delayed(compute_factor_features_batch_weekly)(
                        chunk, chunk_factor_returns[i], config
                    )
                    for i, chunk in enumerate(chunks)
                )
            else:
                # Daily: use standard batch function
                batch_results = Parallel(
                    n_jobs=n_workers,
                    backend='loky',
                    verbose=0
                )(
                    delayed(compute_factor_features_batch)(
                        chunk, chunk_factor_returns[i], config, frequency
                    )
                    for i, chunk in enumerate(chunks)
                )

            # Flatten results
            results = []
            for batch in batch_results:
                results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel factor computation failed ({e}), falling back to sequential")
            if is_weekly:
                results = compute_factor_features_batch_weekly(
                    work_items, factor_returns, config
                )
            else:
                results = compute_factor_features_batch(
                    work_items, factor_returns, config, frequency
                )

    # Apply results to DataFrames
    # Use pd.concat to avoid PerformanceWarning from repeated column insertion
    added_count = 0
    for sym, features in results:
        if not features:
            continue
        df = indicators_by_symbol[sym]
        # Filter out metadata columns and collect new columns
        new_cols = {col_name: series for col_name, series in features.items()
                    if not col_name.endswith('_etf')}
        if new_cols:
            if frequency == 'weekly' and sym in daily_indices:
                # Map weekly results back to daily index using forward-fill
                # Weekly features are computed on Friday close, propagate to next week's daily dates
                daily_idx = daily_indices[sym]
                mapped_cols = {}
                for col_name, series in new_cols.items():
                    # BUG FIX: reindex(method='ffill') only works when source values exist in target
                    # Instead, union the indices, ffill, then select daily dates
                    # This handles cases where Friday dates don't match daily dates exactly
                    combined_idx = series.index.union(daily_idx).sort_values()
                    filled = series.reindex(combined_idx).ffill()
                    mapped_cols[col_name] = filled.reindex(daily_idx).astype('float32')
                new_df = pd.DataFrame(mapped_cols, index=df.index)
            else:
                # Daily frequency - use results as-is
                new_df = pd.DataFrame(new_cols, index=df.index)
            indicators_by_symbol[sym] = pd.concat([df, new_df], axis=1)
        added_count += 1

    logger.info(f"Joint factor features added for {added_count} symbols ({frequency})")

    # Force garbage collection
    gc.collect()


def compute_best_match_mappings(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    candidates: Optional[List[str]] = None,
    window: int = 252,
    min_overlap: int = 100,
    n_jobs: int = -1
) -> Dict[str, str]:
    """
    Compute best-match ETF mappings for all stocks (cap-weighted).

    Uses parallel processing for large datasets (200 stocks per worker).

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame
        candidates: List of candidate ETF symbols (None for defaults)
        window: Lookback window for matching (default 252 = 1 year)
        min_overlap: Minimum overlapping data points
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dict mapping stock symbol -> best-match ETF symbol
    """
    from joblib import Parallel, delayed

    if candidates is None:
        candidates = BEST_MATCH_CANDIDATES.copy()

    logger.info(f"Computing best-match ETF mappings with {len(candidates)} candidates")

    # Pre-extract ETF returns
    etf_returns: Dict[str, pd.Series] = {}
    for etf in candidates:
        if etf not in indicators_by_symbol:
            continue
        df = indicators_by_symbol[etf]
        if 'ret' not in df.columns:
            continue
        etf_returns[etf] = pd.to_numeric(df['ret'], errors='coerce')

    logger.debug(f"Loaded {len(etf_returns)} candidate ETF return series")

    # Excluded symbols (benchmarks and ETFs)
    excluded = set(candidates) | {'SPY', 'QQQ', 'RSP', 'IWM', 'DIA', 'VTI', 'VOO'}

    # Build work items: (symbol, stock_returns) tuples
    work_items = []
    for sym, df in indicators_by_symbol.items():
        if sym in excluded:
            continue
        if 'ret' not in df.columns:
            continue
        stock_ret = pd.to_numeric(df['ret'], errors='coerce')
        work_items.append((sym, stock_ret))

    n_symbols = len(work_items)
    logger.info(f"Computing best-match for {n_symbols} symbols")

    # Calculate workers based on stocks_per_worker (200 stocks/worker default)
    chunk_size = DEFAULT_STOCKS_PER_WORKER
    n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)

    # Sequential for small datasets
    if n_symbols < 20 or n_jobs == 1 or n_workers == 1:
        all_results = _find_best_match_batch(
            work_items, etf_returns, candidates, window, min_overlap
        )
    else:
        # Parallel with batching
        chunks = [work_items[i:i + chunk_size] for i in range(0, n_symbols, chunk_size)]
        logger.info(f"Parallel best-match: {len(chunks)} batches, {n_workers} workers")

        try:
            batch_results = Parallel(
                n_jobs=n_workers,
                backend='loky',
                verbose=0
            )(
                delayed(_find_best_match_batch)(
                    chunk, etf_returns, candidates, window, min_overlap
                )
                for chunk in chunks
            )

            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel best-match failed ({e}), falling back to sequential")
            all_results = _find_best_match_batch(
                work_items, etf_returns, candidates, window, min_overlap
            )

    # Aggregate results
    mappings = {}
    match_count = 0
    r2_sum = 0.0

    for sym, best_etf, r2 in all_results:
        if best_etf:
            mappings[sym] = best_etf
            match_count += 1
            r2_sum += r2

    avg_r2 = r2_sum / match_count if match_count > 0 else 0.0
    logger.info(f"Best-match mappings: {match_count} stocks matched, avg R2={avg_r2:.3f}")

    # Log distribution of matches
    match_dist = {}
    for etf in mappings.values():
        match_dist[etf] = match_dist.get(etf, 0) + 1

    top_matches = sorted(match_dist.items(), key=lambda x: -x[1])[:10]
    logger.info(f"Top best-match ETFs: {top_matches}")

    return mappings


def compute_best_match_ew_mappings(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    candidates: Optional[List[str]] = None,
    window: int = 252,
    min_overlap: int = 100,
    n_jobs: int = -1
) -> Dict[str, str]:
    """
    Compute best-match equal-weight ETF mappings for all stocks.

    Same algorithm as compute_best_match_mappings but uses equal-weight
    ETF candidates instead of cap-weighted. Uses parallel processing
    for large datasets (100 stocks per worker).

    Args:
        indicators_by_symbol: Dict of symbol -> DataFrame
        candidates: List of equal-weight candidate ETF symbols (None for defaults)
        window: Lookback window for matching (default 252 = 1 year)
        min_overlap: Minimum overlapping data points
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dict mapping stock symbol -> best-match equal-weight ETF symbol
    """
    from joblib import Parallel, delayed

    if candidates is None:
        candidates = BEST_MATCH_EW_CANDIDATES.copy()

    logger.info(f"Computing best-match equal-weight ETF mappings with {len(candidates)} candidates")

    # Pre-extract ETF returns
    etf_returns: Dict[str, pd.Series] = {}
    for etf in candidates:
        if etf not in indicators_by_symbol:
            continue
        df = indicators_by_symbol[etf]
        if 'ret' not in df.columns:
            continue
        etf_returns[etf] = pd.to_numeric(df['ret'], errors='coerce')

    available_count = len(etf_returns)
    logger.debug(f"Loaded {available_count} equal-weight candidate ETF return series")

    if available_count == 0:
        logger.warning("No equal-weight ETF candidates available - skipping EW best-match")
        return {}

    # Excluded symbols (benchmarks and ETFs)
    all_etf_candidates = set(candidates) | set(BEST_MATCH_CANDIDATES) | {'SPY', 'QQQ', 'RSP', 'IWM', 'DIA', 'VTI', 'VOO'}

    # Build work items: (symbol, stock_returns) tuples
    available_candidates = list(etf_returns.keys())
    work_items = []
    for sym, df in indicators_by_symbol.items():
        if sym in all_etf_candidates:
            continue
        if 'ret' not in df.columns:
            continue
        stock_ret = pd.to_numeric(df['ret'], errors='coerce')
        work_items.append((sym, stock_ret))

    n_symbols = len(work_items)
    logger.info(f"Computing EW best-match for {n_symbols} symbols")

    # Calculate workers based on stocks_per_worker (200 stocks/worker default)
    chunk_size = DEFAULT_STOCKS_PER_WORKER
    n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)

    # Sequential for small datasets
    if n_symbols < 20 or n_jobs == 1 or n_workers == 1:
        all_results = _find_best_match_batch(
            work_items, etf_returns, available_candidates, window, min_overlap
        )
    else:
        # Parallel with batching
        chunks = [work_items[i:i + chunk_size] for i in range(0, n_symbols, chunk_size)]
        logger.info(f"Parallel EW best-match: {len(chunks)} batches, {n_workers} workers")

        try:
            batch_results = Parallel(
                n_jobs=n_workers,
                backend='loky',
                verbose=0
            )(
                delayed(_find_best_match_batch)(
                    chunk, etf_returns, available_candidates, window, min_overlap
                )
                for chunk in chunks
            )

            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel EW best-match failed ({e}), falling back to sequential")
            all_results = _find_best_match_batch(
                work_items, etf_returns, available_candidates, window, min_overlap
            )

    # Aggregate results
    mappings = {}
    match_count = 0
    r2_sum = 0.0

    for sym, best_etf, r2 in all_results:
        if best_etf:
            mappings[sym] = best_etf
            match_count += 1
            r2_sum += r2

    avg_r2 = r2_sum / match_count if match_count > 0 else 0.0
    logger.info(f"Best-match EW mappings: {match_count} stocks matched, avg R2={avg_r2:.3f}")

    # Log distribution of matches
    if match_count > 0:
        match_dist = {}
        for etf in mappings.values():
            match_dist[etf] = match_dist.get(etf, 0) + 1
        top_matches = sorted(match_dist.items(), key=lambda x: -x[1])[:10]
        logger.info(f"Top best-match EW ETFs: {top_matches}")

    return mappings
