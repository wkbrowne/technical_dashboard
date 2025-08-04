import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima

from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch package not available. Install with: pip install arch")

def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calculate returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    method : str
        'log' for log returns, 'simple' for simple returns
        
    Returns
    -------
    pd.Series
        Returns series
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns.dropna()

def fit_garch_model(returns: pd.Series, 
                   p: int = 1, 
                   q: int = 1,
                   mean_model: str = 'AR',
                   vol_model: str = 'GARCH',
                   dist: str = 't') -> Optional[object]:
    """
    Fit GARCH model to returns series.
    
    Parameters
    ----------
    returns : pd.Series
        Returns series (should be stationary)
    p : int
        GARCH lag order
    q : int  
        ARCH lag order
    mean_model : str
        Mean model specification ('AR', 'Constant', 'Zero')
    vol_model : str
        Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH')
    dist : str
        Error distribution ('normal', 't', 'skewt')
        
    Returns
    -------
    fitted_model or None
        Fitted GARCH model or None if arch not available
    """
    if not ARCH_AVAILABLE:
        print("ARCH package not available. Using simple volatility estimation.")
        return None
    
    try:
        # Remove extreme outliers (beyond 5 standard deviations)
        returns_clean = returns[np.abs(returns - returns.mean()) < 5 * returns.std()]
        
        # Scale returns to percentage (helps with numerical stability)
        returns_scaled = returns_clean * 100
        
        # Create and fit model
        if mean_model == 'AR':
            model = arch_model(returns_scaled, mean='AR', lags=1, vol=vol_model, p=p, q=q, dist=dist)
        elif mean_model == 'Constant':
            model = arch_model(returns_scaled, mean='Constant', vol=vol_model, p=p, q=q, dist=dist)
        else:  # Zero mean
            model = arch_model(returns_scaled, mean='Zero', vol=vol_model, p=p, q=q, dist=dist)
        
        fitted_model = model.fit(disp='off', show_warning=False)
        return fitted_model
        
    except Exception as e:
        print(f"GARCH model fitting failed: {e}")
        return None

def forecast_garch_volatility(
    fitted_model: object, 
    horizon: int = 20,
    returns_scale: float = 100,
    last_date: Optional[pd.Timestamp] = None
) -> Dict:
    """
    Forecast volatility using fitted GARCH model.

    Parameters
    ----------
    fitted_model : object
        Fitted GARCH model from fit_garch_model
    horizon : int
        Forecast horizon in days
    returns_scale : float
        Scale factor used in model fitting (default 100 for percentage)
    last_date : pd.Timestamp, optional
        Last known date from the original data, used to generate forecast_dates
        
    Returns
    -------
    dict
        Dictionary with forecasted volatility and confidence intervals
    """
    if fitted_model is None:
        return None

    try:
        # Generate forecast
        forecast = fitted_model.forecast(horizon=horizon, reindex=False)

        # Extract volatility forecasts (convert back from percentage scale)
        vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / returns_scale

        # Calculate confidence intervals
        if hasattr(fitted_model, 'dist') and getattr(fitted_model.dist, 'name', '') in ['t', 'skewt']:
            df = fitted_model.params.get('nu', 5)
            confidence_multiplier = stats.t.ppf(0.975, df)
        else:
            confidence_multiplier = 1.96

        # Estimate forecast uncertainty (simplified approach)
        vol_std = np.std(fitted_model.conditional_volatility) / returns_scale
        vol_upper = vol_forecast + confidence_multiplier * vol_std
        vol_lower = np.maximum(vol_forecast - confidence_multiplier * vol_std, 0)

        # Generate forecast dates
        if last_date is None:
            forecast_dates = pd.date_range(periods=horizon, freq='D')
        else:
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

        return {
            'forecast_dates': forecast_dates,
            'volatility_forecast': vol_forecast,
            'volatility_upper': vol_upper,
            'volatility_lower': vol_lower,
            'mean_volatility': np.mean(vol_forecast),
            'volatility_trend': vol_forecast[-1] - vol_forecast[0]
        }

    except Exception as e:
        print(f"GARCH forecasting failed: {e}")
        return None
        
    

def simple_volatility_forecast(returns: pd.Series, 
                              horizon: int = 20,
                              method: str = 'ewm') -> Dict:
    """
    Simple volatility forecasting when GARCH is not available.
    
    Parameters
    ----------
    returns : pd.Series
        Returns series
    horizon : int
        Forecast horizon
    method : str
        'ewm' for exponentially weighted, 'rolling' for rolling window
        
    Returns
    -------
    dict
        Dictionary with forecasted volatility
    """
    if method == 'ewm':
        # Exponentially weighted volatility
        vol_current = returns.ewm(span=30).std().iloc[-1]
        
        # Simple mean reversion model
        long_term_vol = returns.std()
        decay_factor = 0.95  # Daily decay
        
        vol_forecast = []
        for i in range(horizon):
            vol_t = long_term_vol + (vol_current - long_term_vol) * (decay_factor ** i)
            vol_forecast.append(vol_t)
        
    else:
        # Rolling window volatility
        vol_current = returns.rolling(window=30).std().iloc[-1]  
        vol_forecast = [vol_current] * horizon
    
    vol_forecast = np.array(vol_forecast)
    
    # Add uncertainty bands
    vol_std = returns.rolling(window=60).std().std()
    vol_upper = vol_forecast + 1.96 * vol_std
    vol_lower = np.maximum(vol_forecast - 1.96 * vol_std, 0)
    
    return {
        'forecast_dates': pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), 
                                      periods=horizon, freq='D'),
        'volatility_forecast': vol_forecast,
        'volatility_upper': vol_upper,
        'volatility_lower': vol_lower,
        'mean_volatility': np.mean(vol_forecast),
        'volatility_trend': vol_forecast[-1] - vol_forecast[0]
    }

def forecast_moving_average(prices: pd.Series, 
                            window: int = 20,
                            horizon: int = 20,
                            method: str = 'trend_continuation') -> Dict:
    """
    Forecast the moving average using various methods.

    Parameters
    ----------
    prices : pd.Series
        Price series
    window : int
        Moving average window
    horizon : int
        Forecast horizon
    method : str
        'trend_continuation', 'mean_reversion', 'arima'

    Returns
    -------
    dict
        Dictionary with MA forecast
    """
    # Calculate current moving average series
    ma_series = prices.rolling(window=window).mean().dropna()
    current_ma = ma_series.iloc[-1]
    current_price = prices.iloc[-1]
    
    # Calculate trend
    ma_returns = ma_series.pct_change().dropna()
    recent_trend = ma_returns.tail(10).mean()

    if method == 'trend_continuation':
        # Continue recent trend with decay
        decay_factor = 0.98
        ma_forecast = [
            current_ma * (1 + recent_trend * (decay_factor ** i)) 
            for i in range(horizon)
        ]

    elif method == 'mean_reversion':
        # Mean revert toward long-term average
        long_term_ma = ma_series.tail(252).mean()
        reversion_speed = 0.05
        ma_forecast = []
        current_forecast = current_ma
        for _ in range(horizon):
            current_forecast += reversion_speed * (long_term_ma - current_forecast)
            ma_forecast.append(current_forecast)

    elif method == 'arima':
        try:
            model = auto_arima(ma_series, seasonal=False, suppress_warnings=True, stepwise=True)
            ma_forecast = model.predict(n_periods=horizon)
        except Exception as e:
            print(f"ARIMA forecasting failed: {e}")
            # Fallback to flat MA
            ma_forecast = [current_ma] * horizon
    else:
        # Default to flat MA
        ma_forecast = [current_ma] * horizon

    ma_forecast = np.array(ma_forecast)

    # Add uncertainty bands based on historical MA volatility
    ma_volatility = ma_returns.std()
    ma_upper = ma_forecast + 1.96 * ma_volatility * np.sqrt(np.arange(1, horizon + 1))
    ma_lower = ma_forecast - 1.96 * ma_volatility * np.sqrt(np.arange(1, horizon + 1))

    return {
        'forecast_dates': pd.date_range(
            start=prices.index[-1] + pd.Timedelta(days=1), 
            periods=horizon, freq='D'
        ),
        'ma_forecast': ma_forecast,
        'ma_upper': ma_upper,
        'ma_lower': ma_lower,
        'current_ma': current_ma,
        'ma_trend': recent_trend
    }

def create_volatility_model(prices: pd.Series,
                           ma_window: int = 20,
                           forecast_horizon: int = 20) -> Dict:
    """
    Create comprehensive volatility and MA forecasting model.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    ma_window : int
        Moving average window
    forecast_horizon : int
        Forecast horizon in days
        
    Returns
    -------
    dict
        Complete forecasting results
    """
    # Calculate returns
    returns = calculate_returns(prices, method='log')
    
    # Fit GARCH model
    garch_model = fit_garch_model(returns)
    
    # Forecast volatility
    if garch_model is not None:
        vol_forecast = forecast_garch_volatility(
            garch_model, 
            forecast_horizon, 
            last_date=returns.index[-1]
        )
        model_type = 'GARCH'
    else:
        vol_forecast = simple_volatility_forecast(returns, forecast_horizon)
        model_type = 'Simple'
    
    # Forecast moving average
    ma_forecast = forecast_moving_average(prices, ma_window, forecast_horizon, 'arima')
    
    # Calculate current Bollinger Bands parameters
    current_ma = prices.rolling(window=ma_window).mean().iloc[-1]
    current_vol = returns.rolling(window=ma_window).std().iloc[-1]
    
    return {
        'model_type': model_type,
        'garch_model': garch_model,
        'volatility_forecast': vol_forecast,
        'ma_forecast': ma_forecast,
        'current_stats': {
            'current_price': prices.iloc[-1],
            'current_ma': current_ma,
            'current_volatility': current_vol,
            'price_to_ma_ratio': prices.iloc[-1] / current_ma,
            'returns_summary': returns.describe()
        }
    }

def plot_volatility_forecast(vol_forecast: Dict, 
                            returns: pd.Series,
                            title: str = "Volatility Forecast") -> None:
    """
    Plot volatility forecast with historical data.
    
    Parameters
    ----------
    vol_forecast : dict
        Volatility forecast results
    returns : pd.Series  
        Historical returns series
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Historical volatility (30-day rolling)
    hist_vol = returns.rolling(window=30).std()
    
    # Plot historical volatility
    ax1.plot(hist_vol.index, hist_vol.values, label='Historical Volatility (30d)', 
             linewidth=2, alpha=0.8, color='blue')
    
    # Plot forecast
    forecast_dates = vol_forecast['forecast_dates']
    ax1.plot(forecast_dates, vol_forecast['volatility_forecast'], 
             label='Forecasted Volatility', linewidth=2, color='red')
    
    # Plot confidence bands
    ax1.fill_between(forecast_dates, 
                     vol_forecast['volatility_lower'],
                     vol_forecast['volatility_upper'],
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    ax1.set_title(f'{title} - Volatility', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Volatility', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot returns distribution
    ax2.hist(returns.values, bins=50, alpha=0.7, density=True, color='skyblue', 
             label='Historical Returns')
    
    # Overlay normal distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = stats.norm.pdf(x, returns.mean(), returns.std())
    ax2.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal Distribution')
    
    ax2.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Returns', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_ma_forecast(ma_forecast: Dict,
                    prices: pd.Series,
                    ma_window: int = 20,
                    title: str = "Moving Average Forecast") -> None:
    """
    Plot moving average forecast with historical data.
    
    Parameters
    ----------
    ma_forecast : dict
        MA forecast results
    prices : pd.Series
        Historical price series
    ma_window : int
        Moving average window
    title : str
        Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Historical prices and MA
    historical_ma = prices.rolling(window=ma_window).mean()
    
    # Plot historical data
    ax.plot(prices.index, prices.values, label='Price', linewidth=1, alpha=0.8, color='black')
    ax.plot(historical_ma.index, historical_ma.values, label=f'{ma_window}d MA', 
            linewidth=2, color='blue')
    
    # Plot forecast
    forecast_dates = ma_forecast['forecast_dates']
    ax.plot(forecast_dates, ma_forecast['ma_forecast'], 
            label='Forecasted MA', linewidth=2, color='red', linestyle='--')
    
    # Plot confidence bands
    ax.fill_between(forecast_dates,
                    ma_forecast['ma_lower'],
                    ma_forecast['ma_upper'],
                    alpha=0.3, color='red', label='95% Confidence Interval')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()