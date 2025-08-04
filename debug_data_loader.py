#!/usr/bin/env python3
"""
Debug script for get_multiple_stocks function
This script will help identify why the data loading is failing
"""

import sys
import os
sys.path.append(os.path.abspath('./src'))

import requests
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

def debug_environment():
    """Check environment setup"""
    print("🔍 ENVIRONMENT DEBUG")
    print("=" * 50)
    
    # Check API key
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if rapidapi_key:
        print(f"✅ RAPIDAPI_KEY: {'*' * (len(rapidapi_key) - 4)}{rapidapi_key[-4:]}")
    else:
        print("❌ RAPIDAPI_KEY: Not found in environment")
        return False
    
    # Check cache directory
    cache_dir = "./cache"
    cache_file = "./cache/stock_data.pkl"
    print(f"📁 Cache directory exists: {os.path.exists(cache_dir)}")
    print(f"📄 Cache file exists: {os.path.exists(cache_file)}")
    
    if os.path.exists(cache_file):
        cache_size = os.path.getsize(cache_file)
        cache_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
        print(f"📊 Cache file size: {cache_size:,} bytes")
        print(f"🕒 Cache last modified: {cache_modified}")
    
    return True

def debug_api_connectivity():
    """Test basic API connectivity"""
    print("\n🌐 API CONNECTIVITY DEBUG")
    print("=" * 50)
    
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        print("❌ Cannot test API - no key available")
        return False
    
    # Test simple request
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    }
    
    test_symbol = "AAPL"
    querystring = {
        "symbol": test_symbol,
        "interval": "1d",
        "diffandsplits": "false"
    }
    
    try:
        print(f"🔗 Testing API with symbol: {test_symbol}")
        print(f"🌍 URL: {url}")
        print(f"📋 Headers: {{'x-rapidapi-key': '***', 'x-rapidapi-host': '{headers['x-rapidapi-host']}'}}")
        print(f"🔍 Query params: {querystring}")
        
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📏 Response size: {len(response.content):,} bytes")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ JSON parsing successful")
                
                # Analyze response structure
                print(f"🔑 Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
                if 'body' in data:
                    body = data['body']
                    if body:
                        print(f"📦 Body type: {type(body)}")
                        if isinstance(body, dict):
                            print(f"📊 Body keys count: {len(body)}")
                            print(f"🔑 First 3 body keys: {list(body.keys())[:3]}")
                            
                            # Sample first entry
                            first_key = list(body.keys())[0] if body else None
                            if first_key:
                                first_entry = body[first_key]
                                print(f"📝 Sample entry keys: {list(first_entry.keys()) if isinstance(first_entry, dict) else 'Not a dict'}")
                        else:
                            print(f"📦 Body content: {str(body)[:200]}...")
                    else:
                        print("❌ Body is empty")
                else:
                    print("❌ No 'body' key in response")
                    
                # Show full response structure (truncated)
                response_str = json.dumps(data, indent=2)[:1000]
                print(f"📋 Response sample:\n{response_str}...")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"📄 Raw response: {response.text[:500]}...")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📄 Response: {response.text[:500]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - check internet connectivity")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def debug_date_parsing():
    """Test date parsing with sample data"""
    print("\n📅 DATE PARSING DEBUG")
    print("=" * 50)
    
    # Test different date formats that might come from the API
    test_dates = [
        "2024-01-15",
        "15-01-2024", 
        "2024/01/15",
        "15/01/2024",
        "2024-1-15",
        "Jan 15 2024",
        "1642204800",  # timestamp
        "2024-01-15T00:00:00Z"
    ]
    
    for date_str in test_dates:
        try:
            # Test different parsing methods
            print(f"Testing: '{date_str}'")
            
            # Method 1: dayfirst=True (current)
            try:
                result1 = pd.to_datetime(date_str, dayfirst=True)
                print(f"  ✅ dayfirst=True: {result1}")
            except Exception as e:
                print(f"  ❌ dayfirst=True: {e}")
            
            # Method 2: infer_datetime_format
            try:
                result2 = pd.to_datetime(date_str, infer_datetime_format=True)
                print(f"  ✅ infer_format: {result2}")
            except Exception as e:
                print(f"  ❌ infer_format: {e}")
                
            # Method 3: default pandas
            try:
                result3 = pd.to_datetime(date_str)
                print(f"  ✅ default: {result3}")
            except Exception as e:
                print(f"  ❌ default: {e}")
                
            print()
            
        except Exception as e:
            print(f"  ❌ All methods failed: {e}")

def debug_get_stock_data_verbose(symbol="AAPL"):
    """Verbose version of get_stock_data with detailed logging"""
    print(f"\n🔍 DETAILED STOCK DATA DEBUG for {symbol}")
    print("=" * 50)
    
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        print("❌ No API key available")
        return None
    
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"
    querystring = {
        "symbol": symbol,
        "interval": "1d",
        "diffandsplits": "false"
    }
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    }
    
    try:
        print(f"📡 Making request for {symbol}...")
        response = requests.get(url, headers=headers, params=querystring)
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Bad status code: {response.text}")
            return None
            
        data = response.json()
        print(f"✅ JSON parsed successfully")
        
        if 'body' not in data:
            print(f"❌ No 'body' in response: {list(data.keys())}")
            return None
            
        body = data['body']
        if not body:
            print(f"❌ Empty body")
            return None
            
        print(f"📦 Body has {len(body)} entries")
        
        # Process data step by step
        print("🔄 Processing records...")
        records = []
        
        for i, (timestamp, values) in enumerate(body.items()):
            if i < 3:  # Show first 3 entries
                print(f"  Entry {i}: timestamp='{timestamp}', values={values}")
            
            record = values.copy()
            record['date'] = values.get('date')
            records.append(record)
            
        print(f"📝 Created {len(records)} records")
        
        # Create DataFrame
        print("📊 Creating DataFrame...")
        df = pd.DataFrame(records)
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check date column
        print("📅 Processing dates...")
        if 'date' in df.columns:
            print(f"  Date column sample: {df['date'].head().tolist()}")
            
            # Try different date parsing methods
            date_parsing_methods = [
                ("dayfirst=True", lambda x: pd.to_datetime(x, dayfirst=True)),
                ("infer_datetime_format", lambda x: pd.to_datetime(x, infer_datetime_format=True)),
                ("default", lambda x: pd.to_datetime(x)),
                ("format='%Y-%m-%d'", lambda x: pd.to_datetime(x, format='%Y-%m-%d')),
                ("format='%d-%m-%Y'", lambda x: pd.to_datetime(x, format='%d-%m-%Y')),
            ]
            
            successful_method = None
            for method_name, method_func in date_parsing_methods:
                try:
                    print(f"  Trying {method_name}...")
                    df['date_parsed'] = method_func(df['date'])
                    print(f"  ✅ {method_name} worked!")
                    print(f"    Sample parsed dates: {df['date_parsed'].head().tolist()}")
                    successful_method = method_name
                    break
                except Exception as e:
                    print(f"  ❌ {method_name} failed: {e}")
            
            if successful_method:
                df['date'] = df['date_parsed']
                df.drop('date_parsed', axis=1, inplace=True)
                df.set_index('date', inplace=True)
                print(f"  ✅ Date index set successfully")
            else:
                print(f"  ❌ All date parsing methods failed")
                return None
        else:
            print(f"  ❌ No 'date' column found")
            return None
            
        # Add symbol and clean columns
        df['symbol'] = symbol
        print(f"📋 Added symbol column")
        
        # Column mapping
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'adjclose': 'AdjClose',
            'volume': 'Volume'
        }
        
        print("🔄 Mapping columns...")
        available_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        print(f"  Available mappings: {available_mappings}")
        
        df = df.rename(columns=available_mappings)
        
        # Check required fields
        required_fields = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            print(f"⚠️ Missing fields: {missing_fields}")
        
        desired_columns = required_fields + ['symbol']
        final_columns = [col for col in desired_columns if col in df.columns]
        df = df[final_columns]
        
        print(f"✅ Final DataFrame:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Index: {df.index.name} ({len(df.index)} entries)")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Sample data:\n{df.head()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_cache_loading():
    """Test cache loading functionality"""
    print("\n💾 CACHE LOADING DEBUG")
    print("=" * 50)
    
    cache_file = "./cache/stock_data.pkl"
    
    if not os.path.exists(cache_file):
        print("❌ Cache file doesn't exist")
        return None
    
    try:
        print(f"📂 Loading cache from: {cache_file}")
        with open(cache_file, "rb") as f:
            import pickle
            data = pickle.load(f)
            
        print(f"✅ Cache loaded successfully")
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"🔑 Keys: {list(data.keys())}")
            
            for key, df in data.items():
                if hasattr(df, 'shape'):
                    print(f"  {key}: {df.shape} - {df.index.min()} to {df.index.max()}")
                else:
                    print(f"  {key}: {type(df)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Cache loading error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all debug tests"""
    print("🐛 DATA LOADER DEBUG SUITE")
    print("=" * 60)
    
    # Step 1: Environment
    if not debug_environment():
        print("\n❌ Environment check failed - stopping here")
        return
    
    # Step 2: API connectivity
    api_works = debug_api_connectivity()
    
    # Step 3: Date parsing
    debug_date_parsing()
    
    # Step 4: Detailed stock data fetch
    if api_works:
        debug_get_stock_data_verbose("AAPL")
    
    # Step 5: Cache loading
    debug_cache_loading()
    
    print("\n🏁 DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()