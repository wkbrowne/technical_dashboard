# Training API Test Suite

This directory contains comprehensive unit tests for the training pipeline API, ensuring all components work correctly and the API is stable.

## Test Structure

### Core Unit Tests
- `test_markov_models.py` - Tests for Markov chain models (MultiStockBBMarkovModel, TrendAwareBBMarkovWrapper)
- `test_ohlc_forecaster.py` - Tests for OHLC forecasting models
- `test_intelligent_forecasters.py` - Tests for intelligent open/high-low forecasters
- `test_garch_models.py` - Tests for GARCH volatility models
- `test_training_pipeline_integration.py` - Full integration tests

### Smoke Tests
- `test_api_smoke.py` - Quick API existence and basic functionality tests

## Running Tests

### Quick API Verification
```bash
python test_training_api.py
```
This runs a comprehensive test of the entire training pipeline API and verifies all components work together.

### Individual Test Modules
```bash
python -m pytest tests/test_markov_models.py -v
python -m pytest tests/test_ohlc_forecaster.py -v
python -m pytest tests/test_garch_models.py -v
```

### All Tests
```bash
python run_tests.py
```

## Test Coverage

The test suite covers:

1. **API Existence**: All required methods exist and are callable
2. **Basic Functionality**: Core operations work with valid inputs
3. **Error Handling**: Proper error handling for invalid inputs
4. **Integration**: Full training pipeline workflow
5. **Edge Cases**: Handling of missing data, extreme values, etc.

## Key APIs Tested

### MultiStockBBMarkovModel
- `fit_global_prior(all_stock_data)`
- `fit_stock_models(all_stock_data)`
- `classify_trend(ma_series)`

### TrendAwareBBMarkovWrapper
- `fit(bb_data)`
- `get_model(trend)`
- `detect_trend(ma_series)`

### OHLCForecaster
- `fit(ohlc_data)`
- `forecast_ohlc(ma_forecast, vol_forecast, bb_states, current_close, n_days)`
- `set_intelligent_open_forecaster(forecaster, symbol)`
- `set_intelligent_high_low_forecaster(forecaster, symbol)`

### IntelligentOpenForecaster
- `train_global_model(all_stock_data)`
- `add_stock_model(symbol, stock_data)`
- `forecast_open(symbol, prev_close, trend_regime, vol_regime)`

### IntelligentHighLowForecaster
- `train_global_model(all_stock_data)`
- `add_stock_model(symbol, stock_data)`
- `forecast_high_low(symbol, reference_price, trend_regime, vol_regime)`

### GARCH Volatility Models
- `calculate_returns(prices, method)`
- `fit_garch_model(returns)`
- `forecast_garch_volatility(fitted_model, horizon)`
- `simple_volatility_forecast(returns, horizon)`

## Test Philosophy

These tests focus on:
1. **API Stability**: Ensuring the interface remains consistent
2. **Functional Correctness**: Verifying expected behavior
3. **Robustness**: Handling edge cases gracefully
4. **Integration**: Full pipeline functionality

The tests are designed to be run frequently during development to catch API changes and regressions.