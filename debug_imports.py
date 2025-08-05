#!/usr/bin/env python3
"""
Debug import issues for sophisticated models.
"""

import sys
import os
sys.path.append('/mnt/a61cc0e8-1b32-4574-a771-4ad77e8faab6/conda/technical_dashboard')

print("🔧 Debugging import issues...")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test individual imports
tests = [
    ("src.models.ohlc_forecasting", "OHLCForecaster"),
    ("src.models.markov_bb", "TrendAwareBBMarkovWrapper"),  
    ("src.data.loader", "get_multiple_stocks"),
]

for module_name, class_name in tests:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ {module_name}.{class_name} imported successfully")
    except Exception as e:
        print(f"❌ Failed to import {module_name}.{class_name}: {e}")
        
# Test relative imports within the models package
print("\n🔧 Testing relative imports from within models package...")

# Simulate being inside the models package
sys.path.insert(0, '/mnt/a61cc0e8-1b32-4574-a771-4ad77e8faab6/conda/technical_dashboard/src/models')

try:
    from ohlc_forecasting import OHLCForecaster
    print("✅ Direct import of OHLCForecaster works")
except Exception as e:
    print(f"❌ Direct import failed: {e}")

try:
    from markov_bb import TrendAwareBBMarkovWrapper
    print("✅ Direct import of TrendAwareBBMarkovWrapper works")
except Exception as e:
    print(f"❌ Direct import failed: {e}")

# Test the loader import
sys.path.insert(0, '/mnt/a61cc0e8-1b32-4574-a771-4ad77e8faab6/conda/technical_dashboard/src')
try:
    from data.loader import get_multiple_stocks
    print("✅ Direct import of get_multiple_stocks works")
except Exception as e:
    print(f"❌ Direct import of loader failed: {e}")