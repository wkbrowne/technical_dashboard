# Centralized Regime Configuration System

This document explains the centralized regime configuration system that provides a single global place to set trend regimes and volatility regimes used across all models.

## Overview

The centralized regime configuration system ensures that all models (Markov models, close price KDE models, open price KDE models, and high-low copula models) use the same regime definitions and thresholds.

## Key Components

### 1. Configuration Files

- **`src/config/regime_config.py`** - Central configuration definitions
- **`src/models/regime_classifier.py`** - Unified regime classifier used by all models

### 2. Configuration Structure

```python
from config.regime_config import REGIME_CONFIG

# Access trend regimes
trend_regimes = REGIME_CONFIG.trend.regime_order
# ['strong_bear', 'bear', 'sideways', 'bull', 'strong_bull']

# Access volatility regimes  
vol_regimes = REGIME_CONFIG.volatility.regime_order
# ['low', 'medium', 'high']

# Get all combined regimes
combined_regimes = REGIME_CONFIG.get_all_combined_regimes()
# ['strong_bear_low', 'strong_bear_medium', 'strong_bear_high', ...]
```

## Using the Centralized Configuration

### 1. For New Models

When creating new models, use the centralized regime classifier:

```python
from models.regime_classifier import REGIME_CLASSIFIER, classify_stock_regimes

# Classify regimes for a stock DataFrame
stock_with_regimes = classify_stock_regimes(stock_data)

# Or use the classifier directly
trend, vol, combined = REGIME_CLASSIFIER.classify_regimes(ma_series, returns)
```

### 2. For Existing Models

Update existing models to use the centralized classifier:

```python
# OLD WAY (don't do this)
class MyModel:
    def __init__(self):
        self.trend_thresholds = {'bull': 0.001, 'bear': -0.001}  # Hardcoded

# NEW WAY (do this)
from models.regime_classifier import REGIME_CLASSIFIER

class MyModel:
    def __init__(self):
        self.regime_classifier = REGIME_CLASSIFIER  # Use centralized classifier
```

### 3. For Markov Models

Use the unified Markov model that integrates with centralized config:

```python
from models.unified_markov_model import (
    create_trend_markov_model,
    create_volatility_markov_model,
    create_combined_markov_model
)

# Create different types of Markov models
trend_model = create_trend_markov_model()
vol_model = create_volatility_markov_model()
combined_model = create_combined_markov_model()

# Fit on stock data
combined_model.fit(stock_data)
```

## Changing Configuration Globally

### 1. Update Trend Thresholds

```python
from config.regime_config import update_trend_thresholds
from models.regime_classifier import REGIME_CLASSIFIER

# Update trend thresholds
new_thresholds = {
    'bull': 0.002,      # Change from 0.001 to 0.002
    'strong_bull': 0.004 # Change from 0.003 to 0.004
}
update_trend_thresholds(new_thresholds)

# Force all models to reload configuration
REGIME_CLASSIFIER.reload_config()
```

### 2. Update Volatility Thresholds

```python
from config.regime_config import update_volatility_thresholds

# Update volatility percentile thresholds
new_vol_thresholds = {
    'low': 25,      # Change from 33.33 to 25
    'medium': 75    # Change from 66.67 to 75
}
update_volatility_thresholds(new_vol_thresholds)

# Reload configuration
REGIME_CLASSIFIER.reload_config()
```

### 3. Complete Custom Configuration

```python
from config.regime_config import (
    set_custom_regime_config, 
    GlobalRegimeConfig,
    TrendRegimeConfig,
    VolatilityRegimeConfig
)

# Create completely custom configuration
custom_trend = TrendRegimeConfig(
    thresholds={
        'strong_bull': 0.005,
        'bull': 0.002,
        'neutral': -0.002,
        'bear': -0.005,
        'strong_bear': float('-inf')
    },
    regime_order=['strong_bear', 'bear', 'neutral', 'bull', 'strong_bull'],
    lookback_window=10  # Different window
)

custom_vol = VolatilityRegimeConfig(
    percentile_thresholds={'low': 20, 'medium': 80, 'high': 100},
    regime_order=['low', 'medium', 'high'],
    lookback_window=30  # Different window
)

custom_config = GlobalRegimeConfig(
    trend=custom_trend,
    volatility=custom_vol,
    regime_separator='|'  # Different separator
)

# Apply custom configuration globally
set_custom_regime_config(custom_config)
REGIME_CLASSIFIER.reload_config()
```

## Models Using Centralized Configuration

### 1. Global KDE Models ✅

All global KDE models now use the centralized configuration:

```python
from models.global_kde_models import train_global_models

# All models will use centralized regime configuration
global_models = train_global_models(stock_data)
```

### 2. Unified Markov Models ✅

```python
from models.unified_markov_model import UnifiedMarkovModel

# Automatically uses centralized configuration
model = UnifiedMarkovModel(model_type='combined')
model.fit(stock_data)
```

### 3. Legacy Models (Need Updates)

Legacy models should be updated to use the centralized system:

```python
# Update existing models to use REGIME_CLASSIFIER instead of hardcoded values
from models.regime_classifier import REGIME_CLASSIFIER

class LegacyModel:
    def __init__(self):
        self.regime_classifier = REGIME_CLASSIFIER  # Add this line
        # Remove hardcoded thresholds
```

## Configuration Details

### Default Trend Regimes

| Regime | Threshold | Description |
|--------|-----------|-------------|
| `strong_bull` | > 0.3% daily MA slope | Very strong uptrend |
| `bull` | 0.1% to 0.3% | Moderate uptrend |
| `sideways` | -0.1% to 0.1% | No clear trend |
| `bear` | -0.3% to -0.1% | Moderate downtrend |
| `strong_bear` | < -0.3% | Very strong downtrend |

### Default Volatility Regimes

| Regime | Percentile | Description |
|--------|------------|-------------|
| `low` | Bottom 33% | Low volatility period |
| `medium` | Middle 33% | Normal volatility |
| `high` | Top 33% | High volatility period |

### Combined Regimes

Combined regimes use the format `{trend}_{volatility}`:

- `strong_bull_low` - Strong uptrend with low volatility
- `bear_high` - Downtrend with high volatility  
- `sideways_medium` - No trend with normal volatility
- etc. (15 total combinations)

## Testing

Run the comprehensive test to verify everything works:

```bash
python test_centralized_regime_config.py
```

This test verifies:
- ✅ All models use centralized configuration
- ✅ Configuration changes affect all models globally
- ✅ Regime classification is consistent across models
- ✅ Models integrate properly with centralized system

## Migration Guide

### For Existing Models

1. **Remove hardcoded regime definitions**:
   ```python
   # Remove these
   self.trend_thresholds = {'bull': 0.001, ...}
   self.vol_thresholds = {'low': 0.01, ...}
   ```

2. **Use centralized classifier**:
   ```python
   # Add this
   from models.regime_classifier import REGIME_CLASSIFIER
   self.regime_classifier = REGIME_CLASSIFIER
   ```

3. **Update regime classification calls**:
   ```python
   # OLD
   trend = self.classify_trend(ma_series)
   
   # NEW  
   trend = self.regime_classifier.classify_trend(ma_series)
   ```

### For New Models

1. Always import and use the centralized classifier
2. Never hardcode regime thresholds
3. Use the unified Markov model for new Markov chains
4. Follow the patterns in `global_kde_models.py`

## Benefits

1. **Consistency** - All models use the same regime definitions
2. **Maintainability** - Change configuration in one place
3. **Flexibility** - Easy to experiment with different regime definitions
4. **Testability** - Centralized testing of regime classification
5. **Documentation** - Clear documentation of all regime parameters

## Future Enhancements

Potential future improvements:
- Configuration persistence (save/load from files)
- Web UI for configuration management
- A/B testing framework for different configurations
- Automatic parameter optimization
- Configuration versioning and rollback