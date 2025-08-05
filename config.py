# Configuration file for technical dashboard

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Cache file for stock data
CACHE_FILE = BASE_DIR / "data" / "stock_cache.pkl"

# Data directory
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Ensure cache directory exists
CACHE_FILE.parent.mkdir(exist_ok=True)

# Other configuration settings
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
DEFAULT_INTERVAL = "1d"
DEFAULT_RATE_LIMIT = 1.0