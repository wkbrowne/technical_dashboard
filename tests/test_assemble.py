"""
Tests for the assemble module.

Tests the conversion of wide-format OHLCV data into per-symbol DataFrames
with basic indicators and returns.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.assemble import assemble_indicators_from_wide, _safe_lower_columns


class TestSafeLowerColumns:
    """Tests for the _safe_lower_columns helper function."""
    
    def test_safe_lower_columns_returns_copy(self):
        """Test that _safe_lower_columns returns a copy of the DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = _safe_lower_columns(df)
        
        assert result is not df  # Should be a different object
        pd.testing.assert_frame_equal(result, df)  # But with same content


class TestAssembleIndicatorsFromWide:
    """Tests for the main assemble_indicators_from_wide function."""
    
    def test_assemble_basic_functionality(self, mock_wide_data):
        """Test basic assembly of wide data into per-symbol DataFrames."""
        result = assemble_indicators_from_wide(mock_wide_data, adjust_ohlc_with_factor=True)
        
        # Should return a dictionary
        assert isinstance(result, dict)
        
        # Should have entries for each symbol
        expected_symbols = set(mock_wide_data['AdjClose'].columns)
        assert set(result.keys()) == expected_symbols
        
        # Each DataFrame should have the expected columns
        expected_cols = ['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ret']
        for sym, df in result.items():
            assert isinstance(df, pd.DataFrame)
            for col in expected_cols:
                assert col in df.columns, f"Missing column {col} in {sym}"
    
    def test_assemble_column_types(self, mock_wide_data):
        """Test that assembled DataFrames have appropriate column types."""
        result = assemble_indicators_from_wide(mock_wide_data)
        
        for sym, df in result.items():
            # All price/volume columns should be numeric
            for col in ['open', 'high', 'low', 'close', 'adjclose', 'volume']:
                assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric in {sym}"
    
    def test_returns_calculation(self, mock_wide_data):
        """Test that log returns are calculated correctly."""
        result = assemble_indicators_from_wide(mock_wide_data)
        
        for sym, df in result.items():
            # Returns should be the log difference of adjclose
            expected_ret = np.log(df['adjclose']).diff()
            pd.testing.assert_series_equal(df['ret'], expected_ret, check_names=False)
    
    def test_adjustment_factor_application(self, business_day_index):
        """Test that OHLC adjustment factor is applied correctly."""
        # Create test data where we control the adjustment factor
        symbols = ['TEST']
        data = {}
        
        # Create Close and AdjClose with known relationship
        close_prices = pd.Series([100.0, 101.0, 102.0], index=business_day_index[:3])
        adj_close_prices = close_prices * 0.95  # 5% adjustment
        
        for metric in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
            df = pd.DataFrame(index=business_day_index[:3], columns=symbols)
            if metric == 'Close':
                df['TEST'] = close_prices
            elif metric == 'AdjClose':
                df['TEST'] = adj_close_prices
            else:
                df['TEST'] = close_prices * (1.01 if metric == 'High' else 0.99)
            data[metric] = df
        
        result = assemble_indicators_from_wide(data, adjust_ohlc_with_factor=True)
        df = result['TEST']
        
        # After adjustment, the adjustment factor should be applied to OHLC
        # The original close should remain unchanged
        expected_factor = adj_close_prices / close_prices
        expected_open = close_prices * 0.99 * expected_factor  # Original open * factor
        
        # Check that open was adjusted by the factor
        pd.testing.assert_series_equal(df['open'], expected_open, rtol=1e-10, check_names=False)
        
        # Close should remain as original close
        pd.testing.assert_series_equal(df['close'], close_prices, rtol=1e-10, check_names=False)
        
        # AdjClose should remain as original adjclose  
        pd.testing.assert_series_equal(df['adjclose'], adj_close_prices, rtol=1e-10, check_names=False)
    
    def test_no_adjustment_factor(self, mock_wide_data):
        """Test behavior when adjustment factor is disabled."""
        result = assemble_indicators_from_wide(mock_wide_data, adjust_ohlc_with_factor=False)
        
        for sym, df in result.items():
            # Without adjustment, close and adjclose should be different
            # (assuming they were different in the input)
            assert not df['close'].equals(df['adjclose'])
    
    def test_missing_adjclose_raises_error(self, mock_wide_data):
        """Test that missing AdjClose key raises ValueError."""
        # Remove AdjClose from the data
        incomplete_data = {k: v for k, v in mock_wide_data.items() if k != 'AdjClose'}
        
        with pytest.raises(ValueError, match="Expected 'AdjClose'"):
            assemble_indicators_from_wide(incomplete_data)
    
    def test_missing_columns_filled_with_nan(self, business_day_index):
        """Test that missing OHLCV columns are filled with NaN."""
        # Create minimal data with only AdjClose
        minimal_data = {
            'AdjClose': pd.DataFrame({
                'TEST': pd.Series([100, 101, 102], index=business_day_index[:3])
            })
        }
        
        result = assemble_indicators_from_wide(minimal_data)
        df = result['TEST']
        
        # Missing columns should be present but filled with NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df.columns
            assert df[col].isna().all()
        
        # AdjClose and ret should have real values
        assert not df['adjclose'].isna().all()
        assert df['ret'].isna().sum() == 1  # Only first value should be NaN due to diff()
    
    def test_empty_symbol_handling(self, business_day_index):
        """Test handling of symbols with no data."""
        # Create data where one symbol has no valid data
        data = {
            'AdjClose': pd.DataFrame({
                'GOOD': [100, 101, 102],
                'EMPTY': [np.nan, np.nan, np.nan]
            }, index=business_day_index[:3])
        }
        
        result = assemble_indicators_from_wide(data)
        
        # Both symbols should be present
        assert 'GOOD' in result
        assert 'EMPTY' in result
        
        # GOOD should have valid adjclose data
        assert not result['GOOD']['adjclose'].isna().all()
        
        # EMPTY should have NaN adjclose data
        assert result['EMPTY']['adjclose'].isna().all()
    
    def test_index_preservation(self, mock_wide_data):
        """Test that date indexes are preserved correctly."""
        result = assemble_indicators_from_wide(mock_wide_data)
        
        original_index = mock_wide_data['AdjClose'].index
        
        for sym, df in result.items():
            # Each DataFrame should have the same index as the original data
            pd.testing.assert_index_equal(df.index, original_index)
    
    def test_symbol_sorting(self, business_day_index):
        """Test that symbols are processed in sorted order."""
        # Create data with symbols in non-alphabetical order
        symbols = ['ZETA', 'ALPHA', 'BETA']
        data = {
            'AdjClose': pd.DataFrame({
                sym: [100 + i for i in range(3)] 
                for sym in symbols
            }, index=business_day_index[:3])
        }
        
        result = assemble_indicators_from_wide(data)
        
        # Result keys should be in sorted order
        assert list(result.keys()) == ['ALPHA', 'BETA', 'ZETA']