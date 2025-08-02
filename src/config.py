import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cache directory + default cache file
DATA_CACHE_DIR = PROJECT_ROOT / 'cache'
CACHE_FILE = DATA_CACHE_DIR / 'stock_data.pkl'
