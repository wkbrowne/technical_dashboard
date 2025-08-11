# Technical Dashboard - Development Guide

This document provides setup and development instructions for the refactored technical dashboard feature computation pipeline.

## Overview

The codebase has been refactored from a monolithic `data_preparation.py` into a clean, modular package structure:

```
src/
├── features/           # Feature computation modules
│   ├── assemble.py     # Data assembly from wide format
│   ├── trend.py        # Trend and moving average features
│   ├── volatility.py   # Volatility regime features
│   ├── hurst.py        # Hurst exponent features
│   ├── distance.py     # Distance to moving average features
│   ├── range_breakout.py # Range and breakout features
│   ├── volume.py       # Volume-based features
│   ├── relstrength.py  # Relative strength features
│   ├── alpha.py        # Alpha momentum features
│   ├── breadth.py      # Market breadth features
│   └── xsec.py         # Cross-sectional features
├── pipelines/          # Pipeline orchestration
│   └── orchestrator.py # Main workflow and parallel processing
├── io/                 # I/O utilities
│   └── saving.py       # Data saving functions
├── data/               # Data loading (existing)
│   └── loader.py       # Market data loaders
└── config/             # Configuration (existing)
```

## Setup

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Environment Configuration

Set up your environment variables for data access:
- `RAPIDAPI_KEY`: Your RapidAPI key for Yahoo Finance data

### 3. Run the Pipeline

```bash
python data_preparation.py
```

## Development Workflow

### Running Tests

Execute the full test suite:

```bash
pytest -q
```

Run with coverage:

```bash
pytest --cov=src tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_trend.py -v
pytest tests/test_volatility.py -v
```

### Code Structure Guidelines

#### 1. Pure Functions
- All feature functions return new DataFrames or modify input DataFrames in-place (documented)
- No global state mutations
- Explicit input/output types

#### 2. Type Hints and Documentation
- All public functions have type hints
- Google-style or NumPy-style docstrings
- Private functions prefixed with `_`

#### 3. Logging
- Use module-level loggers: `logger = logging.getLogger(__name__)`
- No `print()` statements except in CLI entry points
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)

#### 4. Data Types
- Preserve float32 dtypes where specified in original code
- Maintain index alignment semantics
- Handle NaN values gracefully

### Adding New Features

To add a new feature module:

1. Create a new file in `src/features/`
2. Follow the existing pattern:
   ```python
   import logging
   from typing import Dict, Optional
   import pandas as pd
   import numpy as np
   
   logger = logging.getLogger(__name__)
   
   def add_new_features(df: pd.DataFrame, ...) -> pd.DataFrame:
       """Add new features to DataFrame."""
       # Implementation
       return df
   ```

3. Add the feature to the orchestrator pipeline
4. Write comprehensive tests in `tests/test_new_feature.py`

### Testing Guidelines

#### Test Structure
- One test file per feature module
- Use fixtures from `conftest.py` for common test data
- Test both happy path and edge cases

#### Test Categories
- **Unit tests**: Individual function behavior
- **Integration tests**: Feature interaction and data flow
- **Edge cases**: Empty data, insufficient data, NaN handling

#### Example Test Pattern
```python
def test_feature_computation(self, sample_ohlcv_df):
    """Test basic feature computation."""
    result = add_my_features(sample_ohlcv_df.copy())
    
    # Check columns are added
    assert 'my_feature' in result.columns
    
    # Check data types
    assert result['my_feature'].dtype == 'float32'
    
    # Check value ranges
    values = result['my_feature'].dropna()
    assert values.between(-10, 10).all()
```

## Architecture Decisions

### 1. Modular Design
- Each feature family in separate module for maintainability
- Clear separation of concerns
- Easy to test individual components

### 2. Parallel Processing
- Maintained joblib-based parallelization for per-symbol features
- Cross-sectional features computed after parallel phase
- Configurable parallelism via CPU count

### 3. Type Safety
- Comprehensive type hints for better IDE support
- Runtime type checking where critical
- Clear documentation of expected data formats

### 4. Backward Compatibility
- `data_preparation.py` remains as thin CLI wrapper
- Same output artifacts and column names
- Identical behavior for existing workflows

## Performance Considerations

### Memory Usage
- DataFrames copied selectively to avoid unnecessary memory overhead
- Float32 used for feature columns to reduce memory footprint
- Streaming processing for large symbol universes

### Computation Speed
- Vectorized operations preferred over loops
- Parallel processing for embarrassingly parallel tasks
- Efficient rolling window calculations

## Troubleshooting

### Common Issues

#### Import Errors
- Ensure `src/` directory is in Python path
- Check for circular imports in feature modules
- Verify all required dependencies are installed

#### Test Failures
- Run tests with `-v` flag for detailed output
- Check that nolds library is properly installed for Hurst tests
- Ensure test fixtures provide sufficient data for feature calculations

#### Memory Issues
- Reduce parallelism with fewer CPU cores
- Process smaller symbol universes
- Monitor memory usage during large runs

### Debugging Features

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Use smaller datasets for testing:
```python
from src.pipelines.orchestrator import run_pipeline
run_pipeline(max_stocks=10)  # Limit to 10 stocks
```

## Contributing

### Code Style
- Follow PEP 8 with line length ≤ 100 characters
- Use Black formatter if available
- Prefer clarity over clever one-liners

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Ensure all tests pass: `pytest -q`
4. Update documentation if needed
5. Submit PR with clear description

### Performance Testing
For performance-sensitive changes:
1. Profile with small dataset first
2. Compare before/after metrics
3. Document any performance implications

## Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Joblib Documentation](https://joblib.readthedocs.io/)
- [PyTest Documentation](https://docs.pytest.org/)