#!/usr/bin/env python3
"""
Quick test to verify API key is working
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print(f"📁 Loading environment from: {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    if key.strip() == 'RAPIDAPI_KEY':
                        print(f"✅ RAPIDAPI_KEY loaded: {value[:8]}...")
        return True
    else:
        print(f"⚠️ No .env file found at: {env_file}")
        return False

# Load .env file
load_env_file()

# Set RAPIDAPI_KEY directly if needed
if not os.getenv("RAPIDAPI_KEY"):
    print("🔧 Setting RAPIDAPI_KEY directly...")
    os.environ["RAPIDAPI_KEY"] = "2d530e3f25msh7e3021c97724690p18f0dajsn148cba8ab131"
    print("✅ RAPIDAPI_KEY set successfully")

# Test API
print("\n🧪 Testing API connection...")
try:
    from data.loader import get_stock_data
    
    print("📊 Fetching AAPL data...")
    test_data = get_stock_data("AAPL", "1d")
    
    if test_data is not None and len(test_data) > 0:
        print(f"✅ SUCCESS! Retrieved {len(test_data)} data points")
        print(f"   Columns: {list(test_data.columns)}")
        print(f"   Date range: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"   Latest close: ${test_data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ FAILED: No data returned")
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")