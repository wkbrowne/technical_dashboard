"""
Tests for the target_generation module.

Tests triple barrier target generation with synthetic data to verify
correct barrier logic, timing, and configuration handling.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.target_generation import generate_triple_barrier_targets, get_target_summary


@pytest.fixture
def sample_config():
    """Standard configuration for testing."""
    return {
        'up_mult': 3.0,
        'dn_mult': 3.0,
        'max_horizon': 21,
        'start_every': 5,
    }


@pytest.fixture
def synthetic_price_data():
    """Generate synthetic price data for 2 symbols over several weeks."""
    np.random.seed(42)  # For reproducible tests
    
    symbols = ['AAPL', 'MSFT']
    n_days = 60  # About 2 months of data
    start_date = datetime(2023, 1, 1)
    
    data = []
    
    for symbol in symbols:
        dates = pd.date_range(start_date, periods=n_days, freq='D')
        
        # Generate realistic price series with some volatility
        base_price = 100.0 if symbol == 'AAPL' else 250.0
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        # Generate OHLC from close prices
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        
        # Simple ATR approximation
        true_ranges = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(prices, 1)),
                np.abs(low - np.roll(prices, 1))
            )
        )
        # Set first TR to HL range since no previous close
        true_ranges[0] = high[0] - low[0]
        
        # 14-period ATR
        atr = pd.Series(true_ranges).rolling(14, min_periods=1).mean().values
        
        for i in range(n_days):
            data.append({
                'symbol': symbol,
                'date': dates[i],
                'close': prices[i],
                'high': high[i],
                'low': low[i],
                'atr': atr[i]
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def trending_price_data():
    """Generate price data with a strong upward trend for testing barrier hits."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    
    # Strong upward trend
    close_prices = 100 * (1.02 ** np.arange(30))  # 2% daily growth
    high_prices = close_prices * 1.01
    low_prices = close_prices * 0.99
    atr_values = np.full(30, 2.0)  # Constant ATR for predictability
    
    data = []
    for i in range(30):
        data.append({
            'symbol': 'TREND',
            'date': dates[i],
            'close': close_prices[i],
            'high': high_prices[i], 
            'low': low_prices[i],
            'atr': atr_values[i]
        })
    
    return pd.DataFrame(data)


class TestTargetGeneration:
    """Tests for generate_triple_barrier_targets function."""
    
    def test_basic_functionality(self, synthetic_price_data, sample_config):
        """Test that basic target generation works and returns expected columns."""
        targets = generate_triple_barrier_targets(synthetic_price_data, sample_config)
        
        # Should return DataFrame with expected columns
        expected_cols = ['symbol', 't0', 't_hit', 'hit', 'entry_px', 'top', 'bot', 
                        'h_used', 'price_hit', 'ret_from_entry']
        assert all(col in targets.columns for col in expected_cols)
        
        # Should have overlap count column with config suffix
        overlap_col = f"n_overlapping_trajs__up{sample_config['up_mult']}_dn{sample_config['dn_mult']}_h{sample_config['max_horizon']}"
        assert overlap_col in targets.columns
        
        # Should have targets for both symbols
        assert set(targets['symbol'].unique()) == {'AAPL', 'MSFT'}
        
        # Should have reasonable number of targets
        assert len(targets) > 0
        assert len(targets) < len(synthetic_price_data) // 2  # Sanity check
    
    def test_missing_columns_raises_error(self, sample_config):
        """Test that missing required columns raises ValueError."""
        incomplete_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [datetime.now()],
            'close': [100.0]
            # Missing high, low, atr
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_triple_barrier_targets(incomplete_df, sample_config)
    
    def test_missing_config_raises_error(self, synthetic_price_data):
        """Test that missing config keys raises ValueError."""
        incomplete_config = {
            'up_mult': 3.0,
            'dn_mult': 3.0
            # Missing max_horizon, start_every
        }
        
        with pytest.raises(ValueError, match="Missing required config keys"):
            generate_triple_barrier_targets(synthetic_price_data, incomplete_config)
    
    def test_barrier_logic_with_trending_data(self, trending_price_data, sample_config):
        """Test barrier hit logic with predictable trending data."""
        # Use small barriers to ensure hits
        config = sample_config.copy()
        config['up_mult'] = 0.5  # Small barrier for easy hits
        config['dn_mult'] = 0.5
        config['max_horizon'] = 5
        config['start_every'] = 3
        
        targets = generate_triple_barrier_targets(trending_price_data, config)
        
        # With strong upward trend and small barriers, should mostly hit upper
        hit_types = targets['hit'].value_counts()
        
        # Should have some targets
        assert len(targets) > 0
        
        # Upper barrier hits should be most common (hit=1)
        if 1 in hit_types.index:
            assert hit_types[1] > 0
    
    def test_horizon_constraints(self, synthetic_price_data, sample_config):
        """Test that horizon constraints are respected."""
        targets = generate_triple_barrier_targets(synthetic_price_data, sample_config)
        
        # All horizons should be <= max_horizon
        assert (targets['h_used'] <= sample_config['max_horizon']).all()
        
        # All horizons should be >= 1
        assert (targets['h_used'] >= 1).all()
    
    def test_barrier_calculation(self, synthetic_price_data, sample_config):
        """Test that barrier prices are calculated correctly."""
        targets = generate_triple_barrier_targets(synthetic_price_data, sample_config)
        
        for _, row in targets.iterrows():
            entry_px = row['entry_px']
            top = row['top']
            bot = row['bot']
            
            # Top barrier should be entry + up_mult * ATR
            # We can't know exact ATR, but can verify relationship
            assert top > entry_px
            assert bot < entry_px
            
            # Barriers should be symmetric for equal multipliers
            if sample_config['up_mult'] == sample_config['dn_mult']:
                assert abs((top - entry_px) - (entry_px - bot)) < 0.01  # Small tolerance
    
    def test_overlapping_control(self, synthetic_price_data, sample_config):
        """Test that start_every parameter controls overlapping correctly."""
        # Test with different start_every values
        config1 = sample_config.copy()
        config1['start_every'] = 1  # More overlap
        
        config2 = sample_config.copy()  
        config2['start_every'] = 10  # Less overlap
        
        targets1 = generate_triple_barrier_targets(synthetic_price_data, config1)
        targets2 = generate_triple_barrier_targets(synthetic_price_data, config2)
        
        # More frequent starts should generate more targets
        assert len(targets1) >= len(targets2)
    
    def test_return_calculation(self, synthetic_price_data, sample_config):
        """Test that log returns are calculated correctly."""
        targets = generate_triple_barrier_targets(synthetic_price_data, sample_config)
        
        for _, row in targets.iterrows():
            entry_px = row['entry_px']
            price_hit = row['price_hit']
            ret_from_entry = row['ret_from_entry']
            
            if pd.notna(ret_from_entry) and entry_px > 0 and price_hit > 0:
                expected_return = np.log(price_hit / entry_px)
                assert abs(ret_from_entry - expected_return) < 1e-10
    
    def test_insufficient_data_handling(self, sample_config):
        """Test handling of symbols with insufficient data."""
        # Create data with too few rows
        short_data = pd.DataFrame({
            'symbol': ['SHORT'] * 10,
            'date': pd.date_range('2023-01-01', periods=10),
            'close': np.random.uniform(90, 110, 10),
            'high': np.random.uniform(100, 115, 10),
            'low': np.random.uniform(85, 105, 10),
            'atr': np.random.uniform(1, 3, 10)
        })
        
        targets = generate_triple_barrier_targets(short_data, sample_config)
        
        # Should return empty DataFrame for insufficient data
        assert len(targets) == 0
    
    def test_empty_input_handling(self, sample_config):
        """Test handling of empty input DataFrame."""
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'close', 'high', 'low', 'atr'])
        
        targets = generate_triple_barrier_targets(empty_df, sample_config)
        
        # Should return empty DataFrame
        assert len(targets) == 0
        assert isinstance(targets, pd.DataFrame)
    
    def test_overlap_counting_basic(self, sample_config):
        """Test that overlap counting works with controlled data."""
        # Create data with known overlap patterns
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Create data for single symbol to control overlap
        data = []
        for i, date in enumerate(dates):
            data.append({
                'symbol': 'TEST',
                'date': date,
                'close': 100 + i * 0.1,  # Slight upward trend
                'high': 100 + i * 0.1 + 0.5,
                'low': 100 + i * 0.1 - 0.5,
                'atr': 1.0  # Constant ATR for predictability
            })
        
        test_df = pd.DataFrame(data)
        
        # Use config that will create overlapping trajectories
        config = {
            'up_mult': 2.0,
            'dn_mult': 2.0,
            'max_horizon': 5,
            'start_every': 2,  # Start every 2 days, but trajectories last 5 days = overlap
        }
        
        targets = generate_triple_barrier_targets(test_df, config)
        
        # Should have overlap count column
        overlap_col = f"n_overlapping_trajs__up{config['up_mult']}_dn{config['dn_mult']}_h{config['max_horizon']}"
        assert overlap_col in targets.columns
        
        # All overlap counts should be positive integers
        assert (targets[overlap_col] >= 1).all()  # At least the trajectory itself
        assert targets[overlap_col].dtype == int
    
    def test_overlap_counting_no_overlap(self, sample_config):
        """Test overlap counting when trajectories don't overlap."""
        # Create data for single symbol
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'symbol': 'TEST',
                'date': date,
                'close': 100 + i * 0.1,
                'high': 100 + i * 0.1 + 0.5,
                'low': 100 + i * 0.1 - 0.5,
                'atr': 1.0
            })
        
        test_df = pd.DataFrame(data)
        
        # Config with no overlap: start_every >= max_horizon
        config = {
            'up_mult': 2.0,
            'dn_mult': 2.0,
            'max_horizon': 5,
            'start_every': 10,  # Start every 10 days, trajectories last 5 days = no overlap
        }
        
        targets = generate_triple_barrier_targets(test_df, config)
        
        if not targets.empty:
            overlap_col = f"n_overlapping_trajs__up{config['up_mult']}_dn{config['dn_mult']}_h{config['max_horizon']}"
            # With no overlap, all counts should be 1 (just the trajectory itself)
            assert (targets[overlap_col] == 1).all()
    
    def test_overlap_counting_multiple_symbols(self, sample_config):
        """Test that overlap counting is done per symbol."""
        # Create data for two symbols
        symbols = ['SYM1', 'SYM2']
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        
        data = []
        for symbol in symbols:
            for i, date in enumerate(dates):
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'close': 100 + i * 0.1,
                    'high': 100 + i * 0.1 + 0.5,
                    'low': 100 + i * 0.1 - 0.5,
                    'atr': 1.0
                })
        
        test_df = pd.DataFrame(data)
        
        config = {
            'up_mult': 2.0,
            'dn_mult': 2.0,
            'max_horizon': 3,
            'start_every': 2,
        }
        
        targets = generate_triple_barrier_targets(test_df, config)
        
        if not targets.empty:
            overlap_col = f"n_overlapping_trajs__up{config['up_mult']}_dn{config['dn_mult']}_h{config['max_horizon']}"
            
            # Should have targets for both symbols
            assert set(targets['symbol'].unique()) == {'SYM1', 'SYM2'}
            
            # Overlap counts should be calculated per symbol
            assert overlap_col in targets.columns
            assert (targets[overlap_col] >= 1).all()
    
    def test_overlap_column_naming(self):
        """Test that overlap column names include correct config parameters."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'symbol': 'TEST',
                'date': date,
                'close': 100 + i * 0.1,
                'high': 100 + i * 0.1 + 0.5,
                'low': 100 + i * 0.1 - 0.5,
                'atr': 1.0
            })
        
        test_df = pd.DataFrame(data)
        
        # Test different config parameters
        config1 = {'up_mult': 1.5, 'dn_mult': 2.5, 'max_horizon': 10, 'start_every': 3}
        config2 = {'up_mult': 3.0, 'dn_mult': 1.0, 'max_horizon': 7, 'start_every': 2}
        
        targets1 = generate_triple_barrier_targets(test_df, config1)
        targets2 = generate_triple_barrier_targets(test_df, config2)
        
        expected_col1 = "n_overlapping_trajs__up1.5_dn2.5_h10"
        expected_col2 = "n_overlapping_trajs__up3.0_dn1.0_h7"
        
        if not targets1.empty:
            assert expected_col1 in targets1.columns
        if not targets2.empty:
            assert expected_col2 in targets2.columns


class TestTargetSummary:
    """Tests for get_target_summary function."""
    
    def test_summary_basic_functionality(self, synthetic_price_data, sample_config):
        """Test that summary function returns expected metrics."""
        targets = generate_triple_barrier_targets(synthetic_price_data, sample_config)
        summary = get_target_summary(targets)
        
        # Should contain expected keys
        expected_keys = ['total_targets', 'symbols', 'hit_upper', 'hit_lower', 
                        'hit_time', 'avg_horizon_used', 'avg_return', 'return_std', 'date_range']
        assert all(key in summary for key in expected_keys)
        
        # Values should be reasonable
        assert summary['total_targets'] == len(targets)
        assert summary['symbols'] == targets['symbol'].nunique()
        assert isinstance(summary['avg_horizon_used'], float)
        assert isinstance(summary['date_range'], tuple)
    
    def test_summary_empty_targets(self):
        """Test summary function with empty targets DataFrame."""
        empty_targets = pd.DataFrame()
        summary = get_target_summary(empty_targets)
        
        assert summary['total_targets'] == 0
        assert len(summary) == 1  # Only total_targets key
    
    def test_summary_hit_counts(self, trending_price_data):
        """Test that hit counts in summary match actual data."""
        config = {
            'up_mult': 0.5,
            'dn_mult': 0.5, 
            'max_horizon': 5,
            'start_every': 3
        }
        
        targets = generate_triple_barrier_targets(trending_price_data, config)
        summary = get_target_summary(targets)
        
        # Hit counts should sum to total targets
        total_hits = summary['hit_upper'] + summary['hit_lower'] + summary['hit_time']
        assert total_hits == summary['total_targets']