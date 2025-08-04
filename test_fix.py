#!/usr/bin/env python3
"""
Test script to verify the get_multiple_stocks fix
"""

import sys, os
sys.path.append('./src')
from dotenv import load_dotenv
load_dotenv('.env')

from data.loader import get_multiple_stocks

def test_data_loading():
    print("🧪 Testing get_multiple_stocks fix...")
    print("=" * 50)
    
    # Test 1: Load from cache (should work)
    print("📂 Test 1: Loading from cache...")
    try:
        result = get_multiple_stocks(['AAPL', 'MSFT'], update=False)
        if result:
            print("✅ Cache loading: SUCCESS")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Close shape: {result['Close'].shape}")
        else:
            print("❌ Cache loading: FAILED")
    except Exception as e:
        print(f"❌ Cache loading: ERROR - {e}")
    
    print()
    
    # Test 2: Fetch fresh data (tests the fix)
    print("🌐 Test 2: Fetching fresh data (small sample)...")
    try:
        result = get_multiple_stocks(['AAPL'], update=True, rate_limit=1.0)
        if result:
            print("✅ Fresh data loading: SUCCESS")
            print(f"   Keys: {list(result.keys())}")
            close_data = result['Close']
            print(f"   Close shape: {close_data.shape}")
            print(f"   Date range: {close_data.index.min()} to {close_data.index.max()}")
            print(f"   Sample values:\n{close_data.head()}")
        else:
            print("❌ Fresh data loading: FAILED")
    except Exception as e:
        print(f"❌ Fresh data loading: ERROR - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()