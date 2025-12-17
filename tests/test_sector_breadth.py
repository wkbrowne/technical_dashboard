"""
Tests for the sector_breadth module.

Tests the Sector ETF Breadth Proxy feature computation using synthetic
sector ETF data.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.sector_breadth import (
    SECTOR_ETFS,
    N_SECTORS,
    compute_sector_breadth_daily,
    compute_sector_breadth_weekly,
    add_sector_breadth_features,
    get_sector_breadth_feature_names
)


@pytest.fixture
def business_day_index():
    """Create a 252-row business day index for testing (1 year)."""
    return pd.bdate_range(start='2023-01-01', periods=252, freq='B')


@pytest.fixture
def weekly_index():
    """Create a 52-week index for testing (1 year)."""
    return pd.date_range(start='2023-01-06', periods=52, freq='W-FRI')


@pytest.fixture
def synthetic_etf_data(business_day_index):
    """
    Create synthetic OHLCV data for the 11 sector ETFs.

    Returns:
        Dict mapping ETF symbol -> DataFrame with OHLCV data.
    """
    np.random.seed(42)
    etf_data = {}

    for i, etf in enumerate(SECTOR_ETFS):
        # Generate correlated but different price paths for each sector
        n_days = len(business_day_index)
        # Base market return + sector-specific component
        market_return = np.random.normal(0.0004, 0.01, n_days)
        sector_return = np.random.normal(0, 0.005, n_days)
        daily_returns = market_return + sector_return + (i - 5) * 0.0001  # Slight sector bias

        log_prices = np.cumsum(daily_returns)
        adjclose = 100 * np.exp(log_prices)

        # Create OHLC
        daily_range = adjclose * np.random.uniform(0.005, 0.02, n_days)
        high = adjclose + daily_range * np.random.uniform(0.3, 1.0, n_days)
        low = adjclose - daily_range * np.random.uniform(0.3, 1.0, n_days)
        open_price = low + (high - low) * np.random.uniform(0, 1, n_days)
        volume = np.random.lognormal(15, 0.5, n_days)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': adjclose,
            'adjclose': adjclose,
            'volume': volume,
        }, index=business_day_index)

        etf_data[etf] = df

    return etf_data


@pytest.fixture
def synthetic_weekly_etf_data(weekly_index):
    """Create synthetic weekly OHLCV data for the 11 sector ETFs."""
    np.random.seed(43)
    etf_data = {}

    for i, etf in enumerate(SECTOR_ETFS):
        n_weeks = len(weekly_index)
        weekly_returns = np.random.normal(0.002, 0.025, n_weeks)
        log_prices = np.cumsum(weekly_returns)
        adjclose = 100 * np.exp(log_prices)

        daily_range = adjclose * np.random.uniform(0.01, 0.04, n_weeks)
        high = adjclose + daily_range * 0.7
        low = adjclose - daily_range * 0.7
        open_price = low + (high - low) * 0.5
        volume = np.random.lognormal(17, 0.5, n_weeks)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': adjclose,
            'adjclose': adjclose,
            'volume': volume,
        }, index=weekly_index)

        etf_data[etf] = df

    return etf_data


@pytest.fixture
def sample_indicators_by_symbol(business_day_index):
    """Create sample stock data to attach breadth features to."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    np.random.seed(44)
    indicators = {}

    for sym in symbols:
        n_days = len(business_day_index)
        returns = np.random.normal(0.0005, 0.02, n_days)
        log_prices = np.cumsum(returns)
        adjclose = 150 * np.exp(log_prices)

        df = pd.DataFrame({
            'adjclose': adjclose,
            'ret': returns,
        }, index=business_day_index)
        indicators[sym] = df

    return indicators


class TestSectorBreadthConstants:
    """Tests for module-level constants."""

    def test_sector_etfs_count(self):
        """Test that we have exactly 11 sector ETFs."""
        assert len(SECTOR_ETFS) == 11
        assert N_SECTORS == 11

    def test_sector_etfs_format(self):
        """Test that all ETF symbols are uppercase strings."""
        for etf in SECTOR_ETFS:
            assert isinstance(etf, str)
            assert etf.isupper()

    def test_expected_sector_etfs(self):
        """Test that the expected sector ETFs are present."""
        expected = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        assert sorted(SECTOR_ETFS) == sorted(expected)


class TestComputeSectorBreadthDaily:
    """Tests for compute_sector_breadth_daily function."""

    def test_returns_dataframe(self, synthetic_etf_data):
        """Test that function returns a DataFrame."""
        result = compute_sector_breadth_daily(synthetic_etf_data)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, synthetic_etf_data):
        """Test that all expected columns are present."""
        result = compute_sector_breadth_daily(synthetic_etf_data)
        expected_cols = [
            'sector_breadth_adv',
            'sector_breadth_dec',
            'sector_breadth_net_adv',
            'sector_breadth_ad_line',
            'sector_breadth_pct_above_ma50',
            'sector_breadth_pct_above_ma200',
            'sector_breadth_mcclellan_osc',
            'sector_breadth_mcclellan_sum',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_adv_dec_range(self, synthetic_etf_data):
        """Test that advancing/declining counts are in valid range."""
        result = compute_sector_breadth_daily(synthetic_etf_data)

        # Drop first row which has NaN returns
        adv = result['sector_breadth_adv'].dropna()
        dec = result['sector_breadth_dec'].dropna()

        assert (adv >= 0).all()
        assert (adv <= N_SECTORS).all()
        assert (dec >= 0).all()
        assert (dec <= N_SECTORS).all()

    def test_net_adv_calculation(self, synthetic_etf_data):
        """Test that net_adv = adv - dec."""
        result = compute_sector_breadth_daily(synthetic_etf_data)
        expected = result['sector_breadth_adv'] - result['sector_breadth_dec']
        pd.testing.assert_series_equal(
            result['sector_breadth_net_adv'],
            expected,
            check_names=False
        )

    def test_pct_above_ma_range(self, synthetic_etf_data):
        """Test that pct_above_ma values are between 0 and 1."""
        result = compute_sector_breadth_daily(synthetic_etf_data)

        # Skip initial NaN period for MA calculation
        pct50 = result['sector_breadth_pct_above_ma50'].dropna()
        pct200 = result['sector_breadth_pct_above_ma200'].dropna()

        assert (pct50 >= 0).all() and (pct50 <= 1).all()
        assert (pct200 >= 0).all() and (pct200 <= 1).all()

    def test_datetime_index(self, synthetic_etf_data):
        """Test that result has DatetimeIndex."""
        result = compute_sector_breadth_daily(synthetic_etf_data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_insufficient_etfs_returns_none(self, synthetic_etf_data):
        """Test that None is returned when too few ETFs are available."""
        # Keep only 3 ETFs
        limited_data = {k: v for k, v in list(synthetic_etf_data.items())[:3]}
        result = compute_sector_breadth_daily(limited_data)
        assert result is None


class TestComputeSectorBreadthWeekly:
    """Tests for compute_sector_breadth_weekly function."""

    def test_returns_dataframe(self, synthetic_weekly_etf_data):
        """Test that function returns a DataFrame."""
        result = compute_sector_breadth_weekly(synthetic_weekly_etf_data)
        assert isinstance(result, pd.DataFrame)

    def test_expected_weekly_columns(self, synthetic_weekly_etf_data):
        """Test that all expected weekly columns are present."""
        result = compute_sector_breadth_weekly(synthetic_weekly_etf_data)
        expected_cols = [
            'w_sector_breadth_adv',
            'w_sector_breadth_dec',
            'w_sector_breadth_net_adv',
            'w_sector_breadth_ad_line',
            'w_sector_breadth_pct_above_ma10',
            'w_sector_breadth_pct_above_ma40',
            'w_sector_breadth_mcclellan_osc',
            'w_sector_breadth_mcclellan_sum',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing weekly column: {col}"

    def test_weekly_prefix(self, synthetic_weekly_etf_data):
        """Test that all columns have w_ prefix."""
        result = compute_sector_breadth_weekly(synthetic_weekly_etf_data)
        for col in result.columns:
            assert col.startswith('w_'), f"Column {col} missing w_ prefix"


class TestAddSectorBreadthFeatures:
    """Tests for add_sector_breadth_features function."""

    def test_adds_features_to_all_symbols(self, sample_indicators_by_symbol, synthetic_etf_data):
        """Test that features are added to all symbols."""
        add_sector_breadth_features(
            sample_indicators_by_symbol,
            etf_data=synthetic_etf_data
        )

        for sym, df in sample_indicators_by_symbol.items():
            assert 'sector_breadth_adv' in df.columns, f"Feature not added to {sym}"
            assert 'sector_breadth_mcclellan_osc' in df.columns, f"Feature not added to {sym}"

    def test_same_values_for_all_symbols(self, sample_indicators_by_symbol, synthetic_etf_data):
        """Test that breadth features are identical across all symbols (global-by-date)."""
        add_sector_breadth_features(
            sample_indicators_by_symbol,
            etf_data=synthetic_etf_data
        )

        symbols = list(sample_indicators_by_symbol.keys())
        first_sym = symbols[0]
        first_df = sample_indicators_by_symbol[first_sym]

        for sym in symbols[1:]:
            df = sample_indicators_by_symbol[sym]
            for col in get_sector_breadth_feature_names(include_weekly=False):
                if col in first_df.columns and col in df.columns:
                    pd.testing.assert_series_equal(
                        first_df[col].dropna(),
                        df[col].dropna(),
                        check_names=False,
                        obj=f"Feature {col} differs between {first_sym} and {sym}"
                    )


class TestGetSectorBreadthFeatureNames:
    """Tests for get_sector_breadth_feature_names function."""

    def test_daily_only(self):
        """Test that daily-only list has correct features."""
        names = get_sector_breadth_feature_names(include_weekly=False)
        assert 'sector_breadth_adv' in names
        assert 'sector_breadth_mcclellan_osc' in names
        assert 'w_sector_breadth_adv' not in names
        assert len(names) == 8

    def test_with_weekly(self):
        """Test that full list includes weekly features."""
        names = get_sector_breadth_feature_names(include_weekly=True)
        assert 'sector_breadth_adv' in names
        assert 'w_sector_breadth_adv' in names
        assert len(names) == 16
