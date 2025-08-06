"""
Basic configuration settings for the technical dashboard
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Cache file location
CACHE_FILE = PROJECT_ROOT / "cache" / "stock_data.pkl"

# Make sure cache directory exists
CACHE_FILE.parent.mkdir(exist_ok=True)

# Environment variables and API settings
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Other configuration settings
DEFAULT_RATE_LIMIT = 1.0
DEFAULT_INTERVAL = "1d"