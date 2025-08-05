"""
ARIMA-GARCH Models for Moving Average and Bollinger Band Forecasting

This module implements the proper separation:
- ARIMA models for 20-day moving average forecasting
- GARCH models for Bollinger Band volatility modeling
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Union
from pmdarima import auto_arima
from scipy import stats

warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch package not available. Install with: pip install arch")


class ARIMAMovingAverageModel:
    """
    ARIMA model specifically for forecasting 20-day moving averages.
    This models the trend component of stock prices.
    """
    
    def __init__(self, ma_window: int = 20):
        self.ma_window = ma_window
        self.arima_model = None
        self.fitted = False
        self.ma_series = None
        self.current_ma = None
        
    def fit(self, prices: pd.Series) -> 'ARIMAMovingAverageModel':
        """
        Fit ARIMA model to the moving average series.
        
        Parameters
        ----------
        prices : pd.Series
            Price series to calculate moving average from
            
        Returns
        -------
        self
        """
        # Calculate moving average series
        self.ma_series = prices.rolling(window=self.ma_window).mean().dropna()
        self.current_ma = self.ma_series.iloc[-1]
        
        if len(self.ma_series) < 50:
            raise ValueError(f"Need at least 50 observations for ARIMA, got {len(self.ma_series)}")
        
        try:
            # Fit auto ARIMA to moving average series
            print(f"🔄 Fitting ARIMA model for {self.ma_window}-day moving average...")
            self.arima_model = auto_arima(
                self.ma_series,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3,
                max_q=3,
                max_d=2,
                start_p=0,
                start_q=0,
                information_criterion='aic'
            )
            
            self.fitted = True
            print(f"✅ ARIMA model fitted: {self.arima_model.order}")
            
        except Exception as e:
            print(f"❌ ARIMA fitting failed: {str(e)}")
            # Create a simple trend model as fallback
            self._create_fallback_model()
            
        return self
    
    def _create_fallback_model(self):
        """Create a simple trend-based fallback model"""
        # Calculate recent trend
        recent_returns = self.ma_series.pct_change().tail(10).mean()
        self.fallback_trend = recent_returns
        self.fitted = True
        print("✅ Using simple trend fallback model")
    
    def forecast(self, horizon: int = 10) -> Dict:
        """
        Forecast moving average for given horizon.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
            
        Returns
        -------
        dict
            Forecast results with confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.arima_model is not None:
            # Use ARIMA model
            try:
                forecast = self.arima_model.predict(n_periods=horizon, return_conf_int=True)
                ma_forecast = forecast[0]
                conf_int = forecast[1]
                ma_lower = conf_int[:, 0]
                ma_upper = conf_int[:, 1]
                
            except Exception as e:
                print(f"⚠️ ARIMA forecast failed: {str(e)}, using fallback")
                return self._fallback_forecast(horizon)
        else:
            # Use fallback trend model
            return self._fallback_forecast(horizon)
        
        # Generate forecast dates
        if hasattr(self.ma_series.index, 'freq') and self.ma_series.index.freq is not None:
            last_date = self.ma_series.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
        else:
            forecast_dates = pd.date_range(
                start=pd.Timestamp.now().normalize(),
                periods=horizon,
                freq='D'
            )
        
        return {
            'ma_forecast': np.array(ma_forecast),
            'ma_lower': np.array(ma_lower),
            'ma_upper': np.array(ma_upper),
            'forecast_dates': forecast_dates,
            'current_ma': self.current_ma,
            'model_type': 'ARIMA',
            'arima_order': self.arima_model.order if self.arima_model else None
        }
    
    def _fallback_forecast(self, horizon: int) -> Dict:
        """Fallback trend-based forecast"""
        ma_forecast = []
        current_forecast = self.current_ma
        
        # Apply trend with slight decay
        for i in range(horizon):
            decay = 0.95 ** i  # Trend decays over time
            current_forecast = current_forecast * (1 + self.fallback_trend * decay)
            ma_forecast.append(current_forecast)
        
        ma_forecast = np.array(ma_forecast)
        
        # Simple confidence intervals based on historical MA volatility
        if len(self.ma_series) > 20:
            ma_std = self.ma_series.pct_change().std()
            expanding_std = ma_std * np.sqrt(np.arange(1, horizon + 1))
            ma_upper = ma_forecast + 1.96 * expanding_std * ma_forecast
            ma_lower = ma_forecast - 1.96 * expanding_std * ma_forecast
        else:
            ma_upper = ma_forecast * 1.05
            ma_lower = ma_forecast * 0.95
        
        forecast_dates = pd.date_range(
            start=pd.Timestamp.now().normalize(),
            periods=horizon,
            freq='D'
        )
        
        return {
            'ma_forecast': ma_forecast,
            'ma_lower': ma_lower,
            'ma_upper': ma_upper,
            'forecast_dates': forecast_dates,
            'current_ma': self.current_ma,
            'model_type': 'Trend',
            'arima_order': None
        }
    
    def get_model_summary(self) -> Dict:
        """Get summary of fitted model"""
        if not self.fitted:
            return {'status': 'not_fitted'}
        
        summary = {
            'status': 'fitted',
            'ma_window': self.ma_window,
            'current_ma': self.current_ma,
            'model_type': 'ARIMA' if self.arima_model else 'Trend'
        }
        
        if self.arima_model:
            summary.update({
                'arima_order': self.arima_model.order,
                'aic': self.arima_model.aic(),
                'bic': self.arima_model.bic()
            })
        
        return summary


class GARCHBollingerBandModel:
    """
    GARCH model specifically for Bollinger Band volatility.
    This models the volatility component around the moving average.
    """
    
    def __init__(self, bb_window: int = 20, bb_std: float = 2.0):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.garch_model = None
        self.fitted = False
        self.bb_position_series = None
        self.current_bb_width = None
        
    def fit(self, prices: pd.Series) -> 'GARCHBollingerBandModel':
        """
        Fit GARCH model to Bollinger Band positions.
        
        Parameters
        ----------
        prices : pd.Series
            Price series to calculate Bollinger Bands from
            
        Returns
        -------
        self
        """
        # Calculate Bollinger Bands
        ma = prices.rolling(window=self.bb_window).mean()
        bb_std = prices.rolling(window=self.bb_window).std()
        bb_upper = ma + self.bb_std * bb_std
        bb_lower = ma - self.bb_std * bb_std
        
        # Calculate BB position (-1 to +1, where 0 = at MA)
        self.bb_position_series = ((prices - ma) / (bb_upper - ma)).clip(-1, 1).dropna()
        
        # Calculate BB width (normalized by MA)
        bb_width_series = (bb_std / ma).dropna()
        self.current_bb_width = bb_width_series.iloc[-1]
        
        if len(self.bb_position_series) < 50:
            raise ValueError(f"Need at least 50 observations for GARCH, got {len(self.bb_position_series)}")
        
        # Use BB width changes as the series to model with GARCH
        bb_width_returns = bb_width_series.pct_change().dropna()
        
        try:
            print(f"🔄 Fitting GARCH model for Bollinger Band volatility...")
            self.garch_model = self._fit_garch_model(bb_width_returns)
            self.fitted = True
            print("✅ GARCH model fitted for BB volatility")
            
        except Exception as e:
            print(f"❌ GARCH fitting failed: {str(e)}")
            # Create fallback model
            self._create_fallback_model(bb_width_series)
            
        return self
    
    def _fit_garch_model(self, returns: pd.Series) -> Optional[object]:
        """Fit GARCH model to BB width returns"""
        if not ARCH_AVAILABLE:
            print("ARCH package not available. Using simple volatility estimation.")
            return None
        
        try:
            # Remove extreme outliers
            returns_clean = returns[np.abs(returns - returns.mean()) < 5 * returns.std()]
            
            # Scale returns for numerical stability
            returns_scaled = returns_clean * 100
            
            # Try different GARCH specifications
            garch_specs = [
                {'vol': 'GARCH', 'p': 1, 'q': 1, 'mean': 'Constant'},
                {'vol': 'GARCH', 'p': 1, 'q': 1, 'mean': 'Zero'},
                {'vol': 'GARCH', 'p': 2, 'q': 1, 'mean': 'Constant'},
            ]
            
            best_model = None
            best_aic = np.inf
            
            for spec in garch_specs:
                try:
                    model = arch_model(
                        returns_scaled,
                        mean=spec['mean'],
                        vol=spec['vol'],
                        p=spec['p'],
                        q=spec['q'],
                        dist='t'
                    )
                    fitted_model = model.fit(disp='off', show_warning=False)
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        
                except Exception:
                    continue
            
            return best_model
            
        except Exception as e:
            print(f"GARCH fitting error: {e}")
            return None
    
    def _create_fallback_model(self, bb_width_series: pd.Series):
        """Create simple volatility fallback model"""
        self.bb_width_vol = bb_width_series.rolling(30).std().iloc[-1]
        self.bb_width_mean = bb_width_series.rolling(30).mean().iloc[-1]
        self.fitted = True
        print("✅ Using simple volatility fallback model for BB")
    
    def forecast(self, horizon: int = 10) -> Dict:
        """
        Forecast Bollinger Band volatility.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
            
        Returns
        -------
        dict
            BB volatility forecast results
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.garch_model is not None:
            # Use GARCH model
            try:
                forecast = self.garch_model.forecast(horizon=horizon, reindex=False)
                bb_vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # Scale back
                
                # Convert to BB width forecast
                bb_width_forecast = self.current_bb_width * (1 + bb_vol_forecast)
                
                # Create confidence intervals
                vol_std = np.std(self.garch_model.conditional_volatility) / 100
                bb_width_upper = bb_width_forecast + 1.96 * vol_std
                bb_width_lower = np.maximum(bb_width_forecast - 1.96 * vol_std, 0.001)
                
                model_type = 'GARCH'
                
            except Exception as e:
                print(f"⚠️ GARCH forecast failed: {str(e)}, using fallback")
                return self._fallback_forecast(horizon)
        else:
            # Use fallback model
            return self._fallback_forecast(horizon)
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=pd.Timestamp.now().normalize(),
            periods=horizon,
            freq='D'
        )
        
        return {
            'bb_width_forecast': bb_width_forecast,
            'bb_width_lower': bb_width_lower,
            'bb_width_upper': bb_width_upper,
            'forecast_dates': forecast_dates,
            'current_bb_width': self.current_bb_width,
            'model_type': model_type
        }
    
    def _fallback_forecast(self, horizon: int) -> Dict:
        """Fallback BB volatility forecast"""
        # Mean reversion model for BB width
        long_term_width = self.bb_width_mean if hasattr(self, 'bb_width_mean') else self.current_bb_width
        reversion_speed = 0.05
        
        bb_width_forecast = []
        current_forecast = self.current_bb_width
        
        for _ in range(horizon):
            current_forecast += reversion_speed * (long_term_width - current_forecast)
            bb_width_forecast.append(current_forecast)
        
        bb_width_forecast = np.array(bb_width_forecast)
        
        # Simple confidence intervals
        width_vol = self.bb_width_vol if hasattr(self, 'bb_width_vol') else self.current_bb_width * 0.1
        bb_width_upper = bb_width_forecast + 1.96 * width_vol
        bb_width_lower = np.maximum(bb_width_forecast - 1.96 * width_vol, 0.001)
        
        forecast_dates = pd.date_range(
            start=pd.Timestamp.now().normalize(),
            periods=horizon,
            freq='D'
        )
        
        return {
            'bb_width_forecast': bb_width_forecast,
            'bb_width_lower': bb_width_lower,
            'bb_width_upper': bb_width_upper,
            'forecast_dates': forecast_dates,
            'current_bb_width': self.current_bb_width,
            'model_type': 'Simple'
        }
    
    def get_model_summary(self) -> Dict:
        """Get summary of fitted model"""
        if not self.fitted:
            return {'status': 'not_fitted'}
        
        summary = {
            'status': 'fitted',
            'bb_window': self.bb_window,
            'current_bb_width': self.current_bb_width,
            'model_type': 'GARCH' if self.garch_model else 'Simple'
        }
        
        if self.garch_model:
            summary.update({
                'garch_params': dict(self.garch_model.params),
                'aic': self.garch_model.aic,
                'bic': self.garch_model.bic
            })
        
        return summary


class CombinedARIMAGARCHModel:
    """
    Combined model that uses ARIMA for MA forecasting and GARCH for BB volatility.
    This is the main interface for the training pipeline.
    """
    
    def __init__(self, ma_window: int = 20, bb_std: float = 2.0):
        self.ma_window = ma_window
        self.bb_std = bb_std
        self.arima_model = ARIMAMovingAverageModel(ma_window)
        self.garch_model = GARCHBollingerBandModel(ma_window, bb_std)
        self.fitted = False
        
    def fit(self, prices: pd.Series) -> 'CombinedARIMAGARCHModel':
        """
        Fit both ARIMA and GARCH models.
        
        Parameters
        ----------
        prices : pd.Series
            Price series
            
        Returns
        -------
        self
        """
        print(f"🚀 Training Combined ARIMA-GARCH Model")
        print("=" * 50)
        
        try:
            # Fit ARIMA model for moving average
            self.arima_model.fit(prices)
            arima_success = True
        except Exception as e:
            print(f"❌ ARIMA fitting failed: {str(e)}")
            arima_success = False
        
        try:
            # Fit GARCH model for Bollinger Band volatility
            self.garch_model.fit(prices)
            garch_success = True
        except Exception as e:
            print(f"❌ GARCH fitting failed: {str(e)}")
            garch_success = False
        
        # Consider model fitted if at least one component works
        self.fitted = arima_success or garch_success
        
        if self.fitted:
            print("✅ Combined ARIMA-GARCH model fitted")
        else:
            print("❌ Both ARIMA and GARCH fitting failed")
            
        return self
    
    def forecast(self, horizon: int = 10) -> Dict:
        """
        Generate combined forecast.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon
            
        Returns
        -------
        dict
            Combined forecast results
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        results = {}
        
        # Get MA forecast from ARIMA
        if self.arima_model.fitted:
            ma_results = self.arima_model.forecast(horizon)
            results.update({
                'ma_forecast': ma_results['ma_forecast'],
                'ma_lower': ma_results['ma_lower'],
                'ma_upper': ma_results['ma_upper'],
                'arima_model_type': ma_results['model_type']
            })
        else:
            # Fallback: flat MA
            current_ma = self.arima_model.current_ma or 100.0
            results.update({
                'ma_forecast': np.full(horizon, current_ma),
                'ma_lower': np.full(horizon, current_ma * 0.95),
                'ma_upper': np.full(horizon, current_ma * 1.05),
                'arima_model_type': 'Fallback'
            })
        
        # Get BB volatility forecast from GARCH
        if self.garch_model.fitted:
            bb_results = self.garch_model.forecast(horizon)
            results.update({
                'bb_width_forecast': bb_results['bb_width_forecast'],
                'bb_width_lower': bb_results['bb_width_lower'],
                'bb_width_upper': bb_results['bb_width_upper'],
                'garch_model_type': bb_results['model_type']
            })
        else:
            # Fallback: constant BB width
            current_width = self.garch_model.current_bb_width or 0.02
            results.update({
                'bb_width_forecast': np.full(horizon, current_width),
                'bb_width_lower': np.full(horizon, current_width * 0.8),
                'bb_width_upper': np.full(horizon, current_width * 1.2),
                'garch_model_type': 'Fallback'
            })
        
        # Calculate derived forecasts
        ma_forecast = results['ma_forecast']
        bb_width_forecast = results['bb_width_forecast']
        
        # Calculate BB bands
        bb_upper_forecast = ma_forecast + self.bb_std * bb_width_forecast * ma_forecast
        bb_lower_forecast = ma_forecast - self.bb_std * bb_width_forecast * ma_forecast
        
        results.update({
            'bb_upper_forecast': bb_upper_forecast,
            'bb_lower_forecast': bb_lower_forecast,
            'forecast_dates': pd.date_range(
                start=pd.Timestamp.now().normalize(),
                periods=horizon,
                freq='D'
            ),
            'horizon': horizon,
            'fitted_models': {
                'arima_fitted': self.arima_model.fitted,
                'garch_fitted': self.garch_model.fitted
            }
        })
        
        return results
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        return {
            'combined_model_fitted': self.fitted,
            'arima_summary': self.arima_model.get_model_summary(),
            'garch_summary': self.garch_model.get_model_summary(),
            'parameters': {
                'ma_window': self.ma_window,
                'bb_std': self.bb_std
            }
        }


# Convenience functions for backward compatibility and easy access
def fit_arima_garch_model(prices: pd.Series, 
                         ma_window: int = 20, 
                         bb_std: float = 2.0) -> CombinedARIMAGARCHModel:
    """
    Fit combined ARIMA-GARCH model to price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    ma_window : int
        Moving average window
    bb_std : float
        Bollinger Band standard deviation multiplier
        
    Returns
    -------
    CombinedARIMAGARCHModel
        Fitted model
    """
    model = CombinedARIMAGARCHModel(ma_window, bb_std)
    return model.fit(prices)


def forecast_arima_garch(model: CombinedARIMAGARCHModel, 
                        horizon: int = 10) -> Dict:
    """
    Generate forecast from fitted ARIMA-GARCH model.
    
    Parameters
    ----------
    model : CombinedARIMAGARCHModel
        Fitted model
    horizon : int
        Forecast horizon
        
    Returns
    -------
    dict
        Forecast results
    """
    return model.forecast(horizon)