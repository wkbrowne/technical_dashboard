"""
Model Manager for Monte Carlo Options Analysis
==============================================

This module abstracts away the complex model training and provides a clean 
interface for the Monte Carlo notebook. It handles:

1. Loading stock data
2. Training OHLC forecaster with GARCH/ARIMA/KDE models
3. Training Markov models for regime switching
4. Training intelligent open price forecaster
5. Training high-low copula models
6. Caching models for performance

Key Features:
- Automatic model training and caching
- Clean interface for Monte Carlo simulation
- Support for multiple stocks
- Configurable model parameters
- Performance monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from .ohlc_forecasting import OHLCForecaster
    from .markov_bb import TrendAwareBBMarkovWrapper, MultiStockBBMarkovModel
    from .open_price_kde import IntelligentOpenForecaster
    from .high_low_copula import IntelligentHighLowForecaster
    from ..data.loader import get_multiple_stocks
    from .. import config  # Import config to load .env variables
except ImportError:
    # Fallback for direct execution
    from ohlc_forecasting import OHLCForecaster
    from markov_bb import TrendAwareBBMarkovWrapper, MultiStockBBMarkovModel
    from open_price_kde import IntelligentOpenForecaster
    from high_low_copula import IntelligentHighLowForecaster
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.loader import get_multiple_stocks
    import config  # Import config to load .env variables

class ModelManager:
    """
    Manages training and caching of sophisticated forecasting models.
    Abstracts complexity away from Monte Carlo notebook.
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        Initialize model manager.
        
        Parameters:
        -----------
        cache_dir : str
            Directory to cache trained models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.ohlc_forecasters = {}  # {symbol: OHLCForecaster}
        self.markov_models = {}     # {symbol: TrendAwareBBMarkovWrapper}
        self.global_open_forecaster = None
        self.global_high_low_forecaster = None
        
        # Training status
        self.models_trained = {}    # {symbol: bool}
        self.global_models_trained = False
        
        print("📁 Model Manager initialized")
        print(f"   Cache directory: {self.cache_dir}")
    
    def train_models_for_symbol(self, symbol: str, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train all models for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to train models for
        force_retrain : bool
            Force retraining even if cached models exist
            
        Returns:
        --------
        dict
            Training results and model information
        """
        cache_file = self.cache_dir / f"{symbol}_models.pkl"
        
        if not force_retrain and cache_file.exists():
            print(f"📂 Loading cached models for {symbol}...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_models = pickle.load(f)
                
                self.ohlc_forecasters[symbol] = cached_models['ohlc_forecaster']
                self.markov_models[symbol] = cached_models['markov_model']
                self.models_trained[symbol] = True
                
                print(f"✅ Loaded cached models for {symbol}")
                return cached_models['training_info']
                
            except Exception as e:
                print(f"⚠️ Failed to load cached models for {symbol}: {e}")
                print("🔄 Proceeding with fresh training...")
        
        print(f"🚀 Training sophisticated models for {symbol}...")
        start_time = datetime.now()
        
        try:
            # 1. Load stock data
            print(f"📊 Step 1/4: Loading data for {symbol}...")
            print(f"   Using get_multiple_stocks() data loader...")
            stock_data = get_multiple_stocks([symbol], update=False, rate_limit=1.0)
            
            if symbol not in stock_data['Close'].columns:
                raise ValueError(f"Could not load data for {symbol}")
            
            # Prepare OHLC data
            print(f"   Preparing OHLC data structure...")
            ohlc_data = pd.DataFrame({
                'Open': stock_data['Open'][symbol],
                'High': stock_data['High'][symbol], 
                'Low': stock_data['Low'][symbol],
                'Close': stock_data['Close'][symbol]
            }).dropna()
            
            if len(ohlc_data) < 100:
                raise ValueError(f"Insufficient data for {symbol}: {len(ohlc_data)} observations")
            
            print(f"   ✅ Data loaded: {len(ohlc_data)} observations")
            print(f"   📅 Date range: {ohlc_data.index.min().date()} to {ohlc_data.index.max().date()}")
            print(f"   💰 Price range: ${ohlc_data['Close'].min():.2f} - ${ohlc_data['Close'].max():.2f}")
            print(f"✅ STOCKS READ: {symbol} data loaded successfully ({len(ohlc_data)} observations)")
            
            # 2. Train OHLC Forecaster (includes KDE, GARCH, volatility models)
            print(f"📈 Step 2/4: Training OHLC forecaster (KDE + GARCH + Markov)...")
            print(f"   Bollinger Bands window: 20, std: 2.0")
            print(f"   Training sophisticated volatility models...")
            
            ohlc_forecaster = OHLCForecaster(bb_window=20, bb_std=2.0)
            
            print(f"   🔄 Fitting ARIMA/GARCH models...")
            fit_start = datetime.now()
            ohlc_forecaster.fit(ohlc_data)
            fit_time = datetime.now() - fit_start
            print(f"   ✅ OHLC Forecaster fitted in {fit_time.total_seconds():.1f}s")
            print(f"✅ ARIMA/GARCH TRAINED: {symbol} sophisticated models fitted ({fit_time.total_seconds():.1f}s)")
            
            # 3. Extract and validate Markov model
            print(f"📊 Step 3/4: Extracting Markov regime model...")
            markov_model = ohlc_forecaster.markov_model
            print(f"   Markov model fitted: {markov_model.fitted}")
            print(f"   KDE models trained: {len(ohlc_forecaster.kde_models)} regimes")
            
            # 4. Store models in manager
            print(f"💾 Step 4/4: Storing models in manager...")
            self.ohlc_forecasters[symbol] = ohlc_forecaster
            self.markov_models[symbol] = markov_model
            self.models_trained[symbol] = True
            print(f"   ✅ Models stored for {symbol}")
            print(f"✅ MODEL UPDATED: {symbol} trained models stored in manager")
            
            # Training information with more details
            training_time = datetime.now() - start_time
            training_info = {
                'symbol': symbol,
                'training_time': training_time.total_seconds(),
                'data_points': len(ohlc_data),
                'kde_models': len(ohlc_forecaster.kde_models),
                'markov_fitted': markov_model.fitted,
                'bb_regimes': len(ohlc_forecaster.kde_models),
                'training_date': datetime.now(),
                'data_quality': {
                    'date_range_days': (ohlc_data.index.max() - ohlc_data.index.min()).days,
                    'price_volatility': ohlc_data['Close'].std() / ohlc_data['Close'].mean(),
                    'missing_data_pct': (1 - len(ohlc_data) / len(stock_data['Close'][symbol])) * 100
                }
            }
            
            # Cache models
            try:
                cache_data = {
                    'ohlc_forecaster': ohlc_forecaster,
                    'markov_model': markov_model,
                    'training_info': training_info
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
                print(f"💾 Models cached to {cache_file}")
                
            except Exception as e:
                print(f"⚠️ Failed to cache models: {e}")
            
            print(f"✅ Models trained successfully for {symbol}!")
            print(f"   Training time: {training_time.total_seconds():.1f}s")
            print(f"   OHLC Forecaster: ✓ (KDE + Markov + GARCH)")
            print(f"   KDE Close Models: ✓ ({len(ohlc_forecaster.kde_models)} regimes)")
            print(f"   Markov Model: ✓ ({markov_model.fitted})")
            
            return training_info
            
        except Exception as e:
            print(f"❌ Failed to train models for {symbol}: {e}")
            self.models_trained[symbol] = False
            raise
    
    def train_global_models(self, symbols: List[str], force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train global models (open price forecaster, high-low copula) using multiple stocks.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to use for global model training
        force_retrain : bool
            Force retraining even if cached models exist
            
        Returns:
        --------
        dict
            Training results for global models
        """
        cache_file = self.cache_dir / "global_models.pkl"
        
        if not force_retrain and cache_file.exists():
            print("📂 Loading cached global models...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_models = pickle.load(f)
                
                self.global_open_forecaster = cached_models['open_forecaster']
                self.global_high_low_forecaster = cached_models['high_low_forecaster']
                self.global_models_trained = True
                
                print("✅ Loaded cached global models")
                return cached_models['training_info']
                
            except Exception as e:
                print(f"⚠️ Failed to load cached global models: {e}")
                print("🔄 Proceeding with fresh training...")
        
        print(f"🌍 Training global models with {len(symbols)} stocks...")
        print("=" * 60)
        start_time = datetime.now()
        
        try:
            # Load data for all symbols
            print("📊 Step 1/3: Loading data for global model training...")
            print(f"   Using get_multiple_stocks() for batch data loading...")
            all_stock_data = {}
            successful_loads = 0
            
            for i, symbol in enumerate(symbols, 1):
                print(f"   📈 Loading {symbol} ({i}/{len(symbols)})...")
                try:
                    stock_data = get_multiple_stocks([symbol], update=False, rate_limit=1.0)
                    if symbol in stock_data['Close'].columns:
                        ohlc_data = pd.DataFrame({
                            'Open': stock_data['Open'][symbol],
                            'High': stock_data['High'][symbol], 
                            'Low': stock_data['Low'][symbol],
                            'Close': stock_data['Close'][symbol]
                        }).dropna()
                        
                        if len(ohlc_data) >= 100:  # Minimum data requirement
                            # Add required columns for global models
                            close = ohlc_data['Close']
                            ohlc_data['BB_MA'] = close.rolling(window=20).mean()
                            std = close.rolling(window=20).std()
                            ohlc_data['BB_Width'] = std / ohlc_data['BB_MA']
                            
                            all_stock_data[symbol] = ohlc_data.dropna()
                            successful_loads += 1
                            print(f"      ✅ {symbol}: {len(ohlc_data)} observations")
                        else:
                            print(f"      ❌ {symbol}: Insufficient data ({len(ohlc_data)} < 100)")
                    else:
                        print(f"      ❌ {symbol}: No data in response")
                            
                except Exception as e:
                    print(f"      ❌ {symbol}: {e}")
                    continue
            
            print(f"   📊 Data loading complete: {successful_loads}/{len(symbols)} symbols loaded")
            
            if len(all_stock_data) < 5:
                raise ValueError(f"Need at least 5 stocks for global training, got {len(all_stock_data)}")
            
            # 2. Train Global Open Price Forecaster
            print(f"🔄 Step 2/3: Training global open price forecaster...")
            print(f"   Training KDE models across {len(all_stock_data)} stocks...")
            open_start = datetime.now()
            open_forecaster = IntelligentOpenForecaster()
            open_forecaster.train_global_model(all_stock_data)
            open_time = datetime.now() - open_start
            print(f"   ✅ Open price forecaster trained in {open_time.total_seconds():.1f}s")
            print(f"   📊 Global KDE models: {len(open_forecaster.global_model.global_kde_models)}")
            
            # 3. Train Global High-Low Copula Forecaster
            print(f"🔄 Step 3/3: Training global high-low copula forecaster...")
            print(f"   Training copula models for range prediction...")
            copula_start = datetime.now()
            high_low_forecaster = IntelligentHighLowForecaster()
            high_low_forecaster.train_global_model(all_stock_data)
            copula_time = datetime.now() - copula_start
            print(f"   ✅ High-low copula trained in {copula_time.total_seconds():.1f}s")
            print(f"   📊 Global copula models: {len(high_low_forecaster.global_model.global_copulas)}")
            
            # Store models
            self.global_open_forecaster = open_forecaster
            self.global_high_low_forecaster = high_low_forecaster
            print(f"✅ GLOBAL MODELS TRAINED: Open forecaster + High-low copula ready ({len(all_stock_data)} stocks)")
            self.global_models_trained = True
            
            # Training information
            training_time = datetime.now() - start_time
            training_info = {
                'symbols_used': list(all_stock_data.keys()),
                'n_stocks': len(all_stock_data),
                'training_time': training_time.total_seconds(),
                'open_forecaster_regimes': len(open_forecaster.global_model.global_kde_models),
                'high_low_copula_regimes': len(high_low_forecaster.global_model.global_copulas),
                'training_date': datetime.now()
            }
            
            # Cache models
            try:
                cache_data = {
                    'open_forecaster': open_forecaster,
                    'high_low_forecaster': high_low_forecaster,
                    'training_info': training_info
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
                print(f"💾 Global models cached to {cache_file}")
                
            except Exception as e:
                print(f"⚠️ Failed to cache global models: {e}")
            
            print("✅ Global models trained successfully!")
            print(f"   Training time: {training_time.total_seconds():.1f}s")
            print(f"   Open Forecaster: ✓ ({len(open_forecaster.global_model.global_kde_models)} regimes)")
            print(f"   High-Low Copula: ✓ ({len(high_low_forecaster.global_model.global_copulas)} regimes)")
            
            return training_info
            
        except Exception as e:
            print(f"❌ Failed to train global models: {e}")
            self.global_models_trained = False
            raise
    
    def get_models_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get all trained models for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        dict
            Dictionary containing all models for the symbol
        """
        if symbol not in self.models_trained or not self.models_trained[symbol]:
            raise ValueError(f"Models not trained for {symbol}. Call train_models_for_symbol() first.")
        
        return {
            'ohlc_forecaster': self.ohlc_forecasters[symbol],
            'markov_model': self.markov_models[symbol],
            'global_open_forecaster': self.global_open_forecaster,
            'global_high_low_forecaster': self.global_high_low_forecaster,
            'symbol': symbol,
            'models_available': {
                'ohlc_forecaster': True,
                'markov_model': True,
                'global_open_forecaster': self.global_models_trained,
                'global_high_low_forecaster': self.global_models_trained
            }
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get status of all trained models."""
        return {
            'individual_models': {
                symbol: trained for symbol, trained in self.models_trained.items()
            },
            'global_models_trained': self.global_models_trained,
            'available_symbols': list(self.models_trained.keys()),
            'fully_trained_symbols': [
                symbol for symbol, trained in self.models_trained.items() if trained
            ],
            'cache_directory': str(self.cache_dir)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        print("🗑️ Clearing model cache...")
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                print(f"   Removed {cache_file}")
            except Exception as e:
                print(f"   Failed to remove {cache_file}: {e}")
        
        # Clear in-memory models
        self.ohlc_forecasters.clear()
        self.markov_models.clear()
        self.global_open_forecaster = None
        self.global_high_low_forecaster = None
        self.models_trained.clear()
        self.global_models_trained = False
        
        print("✅ Cache cleared")
    
    def quick_setup_for_demo(self, symbol: str = 'AAPL') -> Dict[str, Any]:
        """
        Quick setup for demonstration purposes.
        Trains models for one symbol with reasonable defaults.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to use for demo
            
        Returns:
        --------
        dict
            Setup results and model information
        """
        print(f"🚀 Quick setup for Monte Carlo demo with {symbol}")
        print("=" * 60)
        
        try:
            # Train individual models
            individual_results = self.train_models_for_symbol(symbol)
            
            # Train global models with a few popular stocks
            demo_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            global_results = self.train_global_models(demo_stocks)
            
            # Setup complete
            setup_results = {
                'primary_symbol': symbol,
                'individual_training': individual_results,
                'global_training': global_results,
                'models_ready': True,
                'setup_complete': True
            }
            
            print("🎉 Quick setup complete!")
            print(f"   Primary symbol: {symbol}")
            print(f"   Global models trained on: {len(demo_stocks)} stocks")
            print("   Ready for Monte Carlo analysis!")
            
            return setup_results
            
        except Exception as e:
            print(f"❌ Quick setup failed: {e}")
            return {
                'models_ready': False,
                'setup_complete': False,
                'error': str(e)
            }