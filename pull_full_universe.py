#!/usr/bin/env python3
"""
Pull and record the full universe of stocks from US universe file.

This script loads all 5,013 stocks from the US universe_2025-08-05*.csv file
and caches the data for use in the training pipeline.
"""

import sys
import os
import time
from datetime import datetime

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

# Load .env file immediately
load_env_file()

# Alternative: Set RAPIDAPI_KEY directly if .env loading fails
if not os.getenv("RAPIDAPI_KEY"):
    print("🔧 Setting RAPIDAPI_KEY directly from config...")
    os.environ["RAPIDAPI_KEY"] = "2d530e3f25msh7e3021c97724690p18f0dajsn148cba8ab131"
    print("✅ RAPIDAPI_KEY set successfully")

def verify_api_key():
    """Verify that the API key is set and working"""
    print("\n🔑 VERIFYING API KEY")
    print("=" * 25)
    
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        print("❌ RAPIDAPI_KEY not found in environment")
        return False
    
    print(f"✅ RAPIDAPI_KEY found: {rapidapi_key[:8]}...")
    
    # Test API call with a simple request
    try:
        from data.loader import get_stock_data
        print("🧪 Testing API with AAPL...")
        test_data = get_stock_data("AAPL", "1d")
        
        if test_data is not None and len(test_data) > 0:
            print(f"✅ API test successful: {len(test_data)} data points")
            return True
        else:
            print("❌ API test failed: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False


def pull_full_universe(max_symbols=None, rate_limit=2.0, force_update=False):
    """
    Pull the full universe of stocks and cache the data.
    
    Parameters
    ----------
    max_symbols : int, optional
        Maximum number of symbols to pull (None for all 5,013)
    rate_limit : float
        API requests per second (2.0 recommended for stability)
    force_update : bool
        If True, forces fresh data fetch even if cache exists
    """
    
    print("🌍 PULLING FULL US UNIVERSE DATA")
    print("=" * 50)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        from data.loader import load_universe_data
        from config import CACHE_FILE
        import os
        
        # Check current cache status
        cache_dir = os.path.dirname(CACHE_FILE)
        universe_cache = os.path.join(cache_dir, 'stock_data_universe.pkl')
        
        if os.path.exists(universe_cache) and not force_update:
            print(f"⚠️  Universe cache already exists: {universe_cache}")
            response = input("Do you want to update with fresh data? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("✅ Using existing cache. Use force_update=True to override.")
                return universe_cache
        
        # Determine how many symbols to pull
        if max_symbols is None:
            print("🎯 Target: ALL universe stocks")
            print("⏱️  Estimated time: 45-60 minutes at 2 req/sec")
        else:
            print(f"🎯 Target: First {max_symbols} universe stocks")
            estimated_minutes = (max_symbols / rate_limit) / 60
            print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes at {rate_limit} req/sec")
        
        # Confirm before proceeding
        if max_symbols is None or max_symbols > 100:
            response = input("This will make many API calls. Continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Cancelled by user")
                return None
        
        print(f"\n🚀 Starting data pull...")
        print(f"   Rate limit: {rate_limit} requests/second")
        print(f"   Update mode: {'Fresh data' if force_update else 'Cache if available'}")
        
        start_time = time.time()
        
        # Pull the data
        stock_data = load_universe_data(
            max_symbols=max_symbols,
            update=True,  # Always fetch fresh data
            rate_limit=rate_limit
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if stock_data and 'Close' in stock_data:
            n_symbols = len(stock_data['Close'].columns)
            n_dates = len(stock_data['Close'])
            
            print(f"\n🎉 SUCCESS!")
            print("=" * 30)
            print(f"✅ Symbols loaded: {n_symbols}")
            print(f"✅ Date range: {n_dates} trading days")
            print(f"✅ Duration: {duration/60:.1f} minutes")
            print(f"✅ Cache location: {universe_cache}")
            
            # Show sample symbols
            symbols = list(stock_data['Close'].columns)
            print(f"\n📊 Sample symbols:")
            print(f"   First 10: {symbols[:10]}")
            print(f"   Last 10: {symbols[-10:]}")
            
            # Show data quality summary
            print(f"\n📈 Data Quality Summary:")
            completeness_scores = []
            for symbol in symbols[:20]:  # Check first 20 for speed
                valid_data = stock_data['Close'][symbol].dropna()
                completeness = len(valid_data) / n_dates
                completeness_scores.append(completeness)
            
            avg_completeness = sum(completeness_scores) / len(completeness_scores)
            print(f"   Average data completeness: {avg_completeness:.1%}")
            print(f"   Sample checked: {len(completeness_scores)} symbols")
            
            return universe_cache
        else:
            print(f"\n❌ FAILED!")
            print("No data was successfully loaded.")
            return None
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️ INTERRUPTED BY USER")
        print("Partial data may have been cached.")
        return None
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def check_universe_status():
    """Check the current status of universe data."""
    
    print("📊 UNIVERSE DATA STATUS")
    print("=" * 30)
    
    try:
        from config import CACHE_FILE
        import os
        import pandas as pd
        import pickle
        
        cache_dir = os.path.dirname(CACHE_FILE)
        universe_cache = os.path.join(cache_dir, 'stock_data_universe.pkl')
        universe_file = os.path.join(cache_dir, 'US universe_2025-08-05_5a49a.csv')
        
        # Check universe CSV file
        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
            print(f"✅ Universe CSV: {os.path.basename(universe_file)}")
            print(f"   Total symbols available: {len(df)}")
            print(f"   File size: {os.path.getsize(universe_file)/1024:.1f} KB")
        else:
            print(f"❌ Universe CSV not found")
            return
        
        # Check cached data
        if os.path.exists(universe_cache):
            cache_size = os.path.getsize(universe_cache) / (1024*1024)  # MB
            cache_date = datetime.fromtimestamp(os.path.getmtime(universe_cache))
            
            print(f"\n✅ Universe Cache: {os.path.basename(universe_cache)}")
            print(f"   File size: {cache_size:.1f} MB")
            print(f"   Last updated: {cache_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Try to load and check contents
            try:
                with open(universe_cache, 'rb') as f:
                    stock_data = pickle.load(f)
                
                if 'Close' in stock_data:
                    n_symbols = len(stock_data['Close'].columns)
                    n_dates = len(stock_data['Close'])
                    symbols = list(stock_data['Close'].columns)
                    
                    print(f"   Symbols cached: {n_symbols}")
                    print(f"   Date range: {n_dates} days")
                    print(f"   Top symbols: {symbols[:5]}")
                else:
                    print(f"   ⚠️ Invalid cache format")
                    
            except Exception as e:
                print(f"   ❌ Error reading cache: {str(e)}")
        else:
            print(f"\n❌ No universe cache found")
            print(f"   Expected location: {universe_cache}")
        
    except Exception as e:
        print(f"❌ Error checking status: {str(e)}")


def main():
    """Main script execution with user options."""
    
    print("🌍 UNIVERSE DATA MANAGER")
    print("=" * 40)
    print("1. Check current status")
    print("2. Verify API key")
    print("3. Pull sample data (first 50 stocks)")
    print("4. Pull medium dataset (first 500 stocks)")
    print("5. Pull FULL universe (all ~5,013 stocks)")
    print("6. Force update existing cache")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
                
            elif choice == '1':
                check_universe_status()
                
            elif choice == '2':
                verify_api_key()
                
            elif choice == '3':
                print("\n📊 Pulling sample data (50 stocks)...")
                if verify_api_key():
                    result = pull_full_universe(max_symbols=50, rate_limit=2.0)
                    if result:
                        print(f"✅ Sample data cached at: {result}")
                else:
                    print("❌ API key verification failed. Cannot proceed.")
                
            elif choice == '4':
                print("\n📊 Pulling medium dataset (500 stocks)...")
                if verify_api_key():
                    result = pull_full_universe(max_symbols=500, rate_limit=2.0)
                    if result:
                        print(f"✅ Medium dataset cached at: {result}")
                else:
                    print("❌ API key verification failed. Cannot proceed.")
                
            elif choice == '5':
                print("\n🌍 Pulling FULL universe data...")
                if verify_api_key():
                    result = pull_full_universe(max_symbols=None, rate_limit=2.0)
                    if result:
                        print(f"✅ Full universe cached at: {result}")
                else:
                    print("❌ API key verification failed. Cannot proceed.")
                
            elif choice == '6':
                print("\n🔄 Force updating existing cache...")
                if verify_api_key():
                    result = pull_full_universe(max_symbols=None, rate_limit=2.0, force_update=True)
                    if result:
                        print(f"✅ Cache updated at: {result}")
                else:
                    print("❌ API key verification failed. Cannot proceed.")
                
            else:
                print("❌ Invalid choice. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    # Quick mode - can be run with command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            check_universe_status()
        elif sys.argv[1] == 'sample':
            pull_full_universe(max_symbols=50)
        elif sys.argv[1] == 'medium':
            pull_full_universe(max_symbols=500)
        elif sys.argv[1] == 'full':
            pull_full_universe(max_symbols=None)
        else:
            print("Usage: python pull_full_universe.py [status|sample|medium|full]")
    else:
        # Interactive mode
        main()