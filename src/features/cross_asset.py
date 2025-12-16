"""
Cross-asset signal features for regime detection and risk-on/risk-off indicators.

These features capture macro-level market dynamics using ETF proxies for various
asset classes. All features are lagged by 1 day to prevent look-ahead bias.

Feature Categories:
1. Risk Regime: Gold/SPY ratio, equity-bond correlation
2. Growth Expectations: Copper/Gold ratio, oil momentum
3. Yield Curve: TLT/SHY ratio as slope proxy, credit spreads (HYG/LQD)
4. Sector Rotation: Cyclical vs defensive (XLY/XLP), tech leadership (XLK/SPY)
5. Dollar Strength: UUP momentum (impacts multinationals, commodities)

ETF Symbols Used:
- SPY: S&P 500 (equity benchmark)
- GLD: Gold (fear/safe haven)
- SLV: Silver (risk-on metal)
- TLT: Long-term treasuries (duration/rates)
- SHY: Short-term treasuries (cash proxy)
- HYG: High-yield bonds (credit risk)
- LQD: Investment-grade bonds (credit safe)
- USO: Oil (growth/inflation)
- COPX: Copper miners (growth proxy, or use copper futures)
- UUP: Dollar index
- XLK: Tech sector
- XLY: Consumer discretionary (cyclical)
- XLP: Consumer staples (defensive)
- XLU: Utilities (defensive)
- XLF: Financials (rate sensitive)
"""
import logging
import warnings
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ETF symbols for cross-asset features (with alternatives)
CROSS_ASSET_ETFS = {
    'equity': ['SPY'],
    'gold': ['GLD'],
    'silver': ['SLV'],
    'long_bond': ['TLT'],
    'short_bond': ['SHY', 'IEF', 'BIL', 'VGSH'],  # Alternatives for short duration
    'high_yield': ['HYG'],
    'inv_grade': ['LQD'],
    'oil': ['USO'],
    'copper': ['COPX'],  # Copper miners as proxy
    'dollar': ['UUP', 'USDU'],  # Dollar index ETFs
    'tech': ['XLK'],
    'cyclical': ['XLY'],
    'defensive': ['XLP'],
    'utilities': ['XLU'],
    'financials': ['XLF'],
}


def _get_price_series(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    symbol: str,
    price_col: str = 'close'
) -> Optional[pd.Series]:
    """Extract price series from indicators dictionary with forward-fill for missing values."""
    if symbol not in indicators_by_symbol:
        return None

    df = indicators_by_symbol[symbol]
    if price_col not in df.columns:
        price_col = 'adjclose' if 'adjclose' in df.columns else None
    if price_col is None:
        return None

    series = pd.to_numeric(df[price_col], errors='coerce')

    # Replace zeros with NaN (price can't be 0 - these are missing values)
    series = series.replace(0, np.nan)
    series = series.where(series > 0, np.nan)

    # Forward-fill missing values (carry forward last known price)
    series = series.ffill()

    return series


def _get_price_with_alternatives(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
    price_col: str = 'close'
) -> Optional[pd.Series]:
    """Try to get price series from a list of alternative symbols."""
    for symbol in symbols:
        series = _get_price_series(indicators_by_symbol, symbol, price_col)
        if series is not None and not series.isna().all():
            return series
    return None


def _compute_ratio(
    series1: pd.Series,
    series2: pd.Series,
    name: str
) -> pd.Series:
    """Compute ratio of two series, handling division by zero."""
    ratio = series1 / series2.replace(0, np.nan)
    return ratio.rename(name)


def _compute_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling percentile rank (0-100)."""
    def pct_rank(x):
        valid = x.dropna()
        if len(valid) < 10:
            return np.nan
        return (valid.iloc[-1] > valid.iloc[:-1]).mean() * 100
    return series.rolling(window, min_periods=50).apply(pct_rank, raw=False)


def _compute_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=20).mean()
    std = series.rolling(window, min_periods=20).std()
    return (series - mean) / std.replace(0, np.nan)


def _compute_momentum(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute momentum (percent change over window)."""
    return series.pct_change(window, fill_method=None)


def _compute_rolling_corr(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 60
) -> pd.Series:
    """Compute rolling correlation between two series."""
    # Suppress numpy warnings from correlation calculation (expected when variance is 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return series1.rolling(window, min_periods=30).corr(series2)


def add_cross_asset_features(
    indicators_by_symbol: Dict[str, pd.DataFrame],
    lag_days: int = 1
) -> None:
    """
    Add cross-asset signal features to all symbols.

    IMPORTANT: All features are lagged by `lag_days` (default 1) to prevent
    look-ahead bias since ETF prices are available at market close.

    Features added (all lagged by lag_days):

    Risk Regime:
    - gold_spy_ratio: GLD/SPY ratio (fear vs greed)
    - gold_spy_ratio_zscore: Z-score of gold/spy ratio
    - equity_bond_corr_60d: Rolling 60d correlation SPY vs TLT

    Growth Expectations:
    - copper_gold_ratio: COPX/GLD ratio (growth optimism)
    - copper_gold_zscore: Z-score of copper/gold
    - oil_momentum_20d: USO 20-day momentum

    Yield Curve / Credit:
    - yield_curve_proxy: TLT/SHY ratio (steeper = higher ratio)
    - yield_curve_zscore: Z-score of yield curve proxy
    - credit_spread_proxy: HYG/LQD ratio (tighter = higher ratio = risk-on)
    - credit_spread_zscore: Z-score of credit spread

    Sector Rotation:
    - cyclical_defensive_ratio: XLY/XLP ratio (risk appetite)
    - tech_spy_ratio: XLK/SPY ratio (tech leadership)
    - financials_utilities_ratio: XLF/XLU ratio (rate expectations)

    Dollar:
    - dollar_momentum_20d: UUP 20-day momentum
    - dollar_percentile_252d: Dollar strength percentile

    Args:
        indicators_by_symbol: Dictionary of symbol DataFrames (modified in place)
        lag_days: Number of days to lag features (default 1 to avoid look-ahead bias)
    """
    logger.info(f"Computing cross-asset features (lag={lag_days} days)")

    # Extract price series for each ETF (try alternatives if primary missing)
    prices = {}
    missing = []
    for name, symbols in CROSS_ASSET_ETFS.items():
        series = _get_price_with_alternatives(indicators_by_symbol, symbols)
        if series is not None and not series.isna().all():
            prices[name] = series
            logger.debug(f"  Found {name}: {len(series.dropna())} observations")
        else:
            missing.append(name)

    if missing:
        logger.warning(f"Missing ETF data for: {missing}")

    if len(prices) < 3:
        logger.warning("Insufficient cross-asset data. Skipping cross-asset features.")
        return

    features = {}

    # =========================================================================
    # 1. Risk Regime Features
    # =========================================================================
    if 'gold' in prices and 'equity' in prices:
        gold_spy = _compute_ratio(prices['gold'], prices['equity'], 'gold_spy_ratio')
        features['gold_spy_ratio'] = gold_spy
        features['gold_spy_ratio_zscore'] = _compute_zscore(gold_spy, 60).rename('gold_spy_ratio_zscore')
        logger.debug("  + Gold/SPY ratio features")

    if 'equity' in prices and 'long_bond' in prices:
        features['equity_bond_corr_60d'] = _compute_rolling_corr(
            prices['equity'].pct_change(),
            prices['long_bond'].pct_change(),
            60
        ).rename('equity_bond_corr_60d')
        logger.debug("  + Equity-bond correlation")

    # =========================================================================
    # 2. Growth Expectations
    # =========================================================================
    if 'copper' in prices and 'gold' in prices:
        copper_gold = _compute_ratio(prices['copper'], prices['gold'], 'copper_gold_ratio')
        features['copper_gold_ratio'] = copper_gold
        features['copper_gold_zscore'] = _compute_zscore(copper_gold, 60).rename('copper_gold_zscore')
        logger.debug("  + Copper/Gold ratio features")

    if 'oil' in prices:
        features['oil_momentum_20d'] = _compute_momentum(prices['oil'], 20).rename('oil_momentum_20d')
        logger.debug("  + Oil momentum")

    # =========================================================================
    # 3. Yield Curve / Credit Features
    # =========================================================================
    if 'long_bond' in prices and 'short_bond' in prices:
        yield_curve = _compute_ratio(prices['long_bond'], prices['short_bond'], 'yield_curve_proxy')
        features['yield_curve_proxy'] = yield_curve
        features['yield_curve_zscore'] = _compute_zscore(yield_curve, 60).rename('yield_curve_zscore')
        logger.debug("  + Yield curve proxy features")

    if 'high_yield' in prices and 'inv_grade' in prices:
        credit_spread = _compute_ratio(prices['high_yield'], prices['inv_grade'], 'credit_spread_proxy')
        features['credit_spread_proxy'] = credit_spread
        features['credit_spread_zscore'] = _compute_zscore(credit_spread, 60).rename('credit_spread_zscore')
        logger.debug("  + Credit spread features")

    # =========================================================================
    # 4. Sector Rotation
    # =========================================================================
    if 'cyclical' in prices and 'defensive' in prices:
        features['cyclical_defensive_ratio'] = _compute_ratio(
            prices['cyclical'], prices['defensive'], 'cyclical_defensive_ratio'
        )
        logger.debug("  + Cyclical/Defensive ratio")

    if 'tech' in prices and 'equity' in prices:
        features['tech_spy_ratio'] = _compute_ratio(
            prices['tech'], prices['equity'], 'tech_spy_ratio'
        )
        logger.debug("  + Tech/SPY ratio")

    if 'financials' in prices and 'utilities' in prices:
        features['financials_utilities_ratio'] = _compute_ratio(
            prices['financials'], prices['utilities'], 'financials_utilities_ratio'
        )
        logger.debug("  + Financials/Utilities ratio")

    # =========================================================================
    # 5. Dollar Strength
    # =========================================================================
    if 'dollar' in prices:
        features['dollar_momentum_20d'] = _compute_momentum(prices['dollar'], 20).rename('dollar_momentum_20d')
        features['dollar_percentile_252d'] = _compute_percentile(prices['dollar'], 252).rename('dollar_percentile_252d')
        logger.debug("  + Dollar features")

    # =========================================================================
    # Attach features to ALL symbols with lag
    # Use pd.concat to batch add all features at once (avoids DataFrame fragmentation)
    # =========================================================================
    if not features:
        logger.warning("No cross-asset features computed")
        return

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
                # Reindex to symbol's date range, apply lag, convert to float32
                reindexed = series.reindex(date_index)
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

    logger.info(f"Cross-asset features ({len(features)} total) attached to {attached_count} symbols (lag={lag_days}d)")


def get_required_etfs() -> List[str]:
    """Return list of ETF symbols required for cross-asset features (including alternatives)."""
    etfs = []
    for symbols in CROSS_ASSET_ETFS.values():
        etfs.extend(symbols)
    return list(set(etfs))
