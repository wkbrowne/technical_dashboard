import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
from scipy import stats
from datetime import datetime, timedelta
from .markov_bb import TrendAwareBBMarkovWrapper

class OHLCForecaster:
    """
    OHLC (Open, High, Low, Close) forecasting model using Bollinger Bands,
    GARCH volatility, and Markov chain models.
    """
    
    def __init__(self, bb_window: int = 20, bb_std: float = 2.0):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.fitted = False
        self.markov_model = TrendAwareBBMarkovWrapper()
        
        
    def fit(self, ohlc_data: pd.DataFrame) -> 'OHLCForecaster':
        """
        Fit the OHLC forecasting model.
        
        Parameters
        ----------
        ohlc_data : pd.DataFrame
            DataFrame with OHLC columns: 'Open', 'High', 'Low', 'Close'
            
        Returns
        -------
        self
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        
        if not all(col in ohlc_data.columns for col in required_cols):
            raise ValueError(f"OHLC data must contain columns: {required_cols}")
        
        self.ohlc_data = ohlc_data.copy()
        
        # Calculate various metrics
        self._calculate_ohlc_metrics()
        self._analyze_intraday_patterns()
        self._calculate_volatility_patterns()
        self._calculate_bb_position_stats()
        self._calculate_trend_regimes()
        # Prepare data for Markov model (needs BB_Position and MA columns)
        markov_data = self.ohlc_data[['BB_Position', 'BB_MA']].copy()
        markov_data.rename(columns={'BB_MA': 'MA'}, inplace=True)
        self.markov_model.fit(markov_data)
        self.fitted = True
        return self
    
    def _calculate_bb_position_stats(self) -> None:
        """Learn empirical BB-relative close position distribution per Markov state."""
        state_bounds = [-np.inf, -0.9, -0.3, 0.3, 0.9, np.inf]
        bb_pos = self.ohlc_data['BB_Position'].dropna()
        
        self.bb_position_stats = {}
        for i in range(5):
            mask = (bb_pos > state_bounds[i]) & (bb_pos <= state_bounds[i+1])
            state_values = bb_pos[mask]
            if len(state_values) > 10:
                self.bb_position_stats[i] = {
                    'mean': state_values.mean(),
                    'std': state_values.std()
                }
            else:
                self.bb_position_stats[i] = {'mean': 0.0, 'std': 0.01}

    def _calculate_ohlc_metrics(self) -> None:
        """Calculate OHLC-derived metrics."""
        ohlc = self.ohlc_data
        
        # Basic metrics
        self.ohlc_data['True_Range'] = self._calculate_true_range(ohlc)
        self.ohlc_data['Range'] = ohlc['High'] - ohlc['Low']
        self.ohlc_data['Body'] = ohlc['Close'] - ohlc['Open']
        self.ohlc_data['Upper_Shadow'] = ohlc['High'] - np.maximum(ohlc['Open'], ohlc['Close'])
        self.ohlc_data['Lower_Shadow'] = np.minimum(ohlc['Open'], ohlc['Close']) - ohlc['Low']
        
        # Normalized metrics (as percentage of close price)
        close = ohlc['Close']
        self.ohlc_data['Range_Pct'] = self.ohlc_data['Range'] / close
        self.ohlc_data['Body_Pct'] = self.ohlc_data['Body'] / close
        self.ohlc_data['Upper_Shadow_Pct'] = self.ohlc_data['Upper_Shadow'] / close
        self.ohlc_data['Lower_Shadow_Pct'] = self.ohlc_data['Lower_Shadow'] / close
        
        # Bollinger Bands on close
        ma = close.rolling(window=self.bb_window).mean()
        std = close.rolling(window=self.bb_window).std()
        self.ohlc_data['BB_MA'] = ma
        self.ohlc_data['BB_Upper'] = ma + self.bb_std * std
        self.ohlc_data['BB_Lower'] = ma - self.bb_std * std
        self.ohlc_data['BB_Position'] = (close - ma) / (self.bb_std * std)
        self.ohlc_data['BB_Width'] = (self.bb_std * std) / ma
        self._calculate_trend_direction()
        
    def _calculate_true_range(self, ohlc: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        prev_close = ohlc['Close'].shift(1)
        tr1 = ohlc['High'] - ohlc['Low']
        tr2 = np.abs(ohlc['High'] - prev_close)
        tr3 = np.abs(ohlc['Low'] - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def _analyze_intraday_patterns(self) -> None:
        """Analyze patterns in intraday price movements."""
        # Gap analysis
        self.ohlc_data['Gap'] = self.ohlc_data['Open'] - self.ohlc_data['Close'].shift(1)
        self.ohlc_data['Gap_Pct'] = self.ohlc_data['Gap'] / self.ohlc_data['Close'].shift(1)
        
        # Opening patterns
        self.ohlc_data['Open_To_High'] = self.ohlc_data['High'] - self.ohlc_data['Open']
        self.ohlc_data['Open_To_Low'] = self.ohlc_data['Open'] - self.ohlc_data['Low']
        
        # Closing patterns
        self.ohlc_data['High_To_Close'] = self.ohlc_data['High'] - self.ohlc_data['Close']
        self.ohlc_data['Low_To_Close'] = self.ohlc_data['Close'] - self.ohlc_data['Low']
        
        # Calculate statistical relationships
        self.intraday_stats = self._calculate_intraday_statistics()
        
    def _calculate_intraday_statistics(self) -> Dict:
        """Calculate statistics for intraday patterns."""
        data = self.ohlc_data.dropna()
        
        stats_dict = {}
        
        # Range statistics conditioned on BB position
        bb_positions = ['Low', 'Middle', 'High']
        trends = ['Up', 'Down', 'Sideways']
        bb_quantiles = [0.33, 0.67]
        
        for i, position in enumerate(bb_positions):
            for trend in trends:
                if i == 0:
                    mask = data['BB_Position'] <= data['BB_Position'].quantile(bb_quantiles[0])
                elif i == 1:
                    mask = ((data['BB_Position'] > data['BB_Position'].quantile(bb_quantiles[0])) & 
                            (data['BB_Position'] <= data['BB_Position'].quantile(bb_quantiles[1])))
                else:
                    mask = data['BB_Position'] > data['BB_Position'].quantile(bb_quantiles[1])

                trend_mask = data['Trend'] == trend
                subset = data[mask & trend_mask]

                if len(subset) > 10:
                    stats_dict[f'{position}_{trend}'] = {
                        'range_pct_mean': subset['Range_Pct'].mean(),
                        'range_pct_std': subset['Range_Pct'].std(),
                        'body_pct_mean': subset['Body_Pct'].mean(),
                        'body_pct_std': subset['Body_Pct'].std(),
                        'upper_shadow_pct_mean': subset['Upper_Shadow_Pct'].mean(),
                        'lower_shadow_pct_mean': subset['Lower_Shadow_Pct'].mean(),
                        'gap_pct_mean': subset['Gap_Pct'].mean(),
                        'gap_pct_std': subset['Gap_Pct'].std()
                    }
        
        # Volatility regime statistics
        vol_high = data['BB_Width'] > data['BB_Width'].median()
        vol_low = data['BB_Width'] <= data['BB_Width'].median()
        
        stats_dict['High_Vol'] = {
            'range_pct_mean': data[vol_high]['Range_Pct'].mean(),
            'range_pct_std': data[vol_high]['Range_Pct'].std(),
            'body_pct_std': data[vol_high]['Body_Pct'].std()
        }
        
        stats_dict['Low_Vol'] = {
            'range_pct_mean': data[vol_low]['Range_Pct'].mean(),
            'range_pct_std': data[vol_low]['Range_Pct'].std(),
            'body_pct_std': data[vol_low]['Body_Pct'].std()
        }
        
        return stats_dict
        
    def _calculate_volatility_patterns(self) -> None:
        """Calculate volatility-related patterns."""
        # ATR (Average True Range)
        self.ohlc_data['ATR'] = self.ohlc_data['True_Range'].rolling(window=14).mean()
        
        # Volatility regimes based on ATR
        atr_median = self.ohlc_data['ATR'].median()
        self.ohlc_data['Vol_Regime'] = np.where(self.ohlc_data['ATR'] > atr_median, 'High', 'Low')
    def forecast_bb_states(self, n_days: int = 1) -> np.ndarray:
        recent_trend = self.ohlc_data['Trend'].iloc[-1]
        recent_state = self.markov_model.get_state(self.ohlc_data.iloc[-1]['BB_Position'])
        return self.markov_model.sample_states(n=n_days, current_state=recent_state, trend=recent_trend)

    def forecast_ohlc(self, 
                     ma_forecast: np.ndarray,
                     vol_forecast: np.ndarray,
                     bb_states: np.ndarray,
                     current_close: float,
                     n_days: int = 1) -> Dict:
        """
        Forecast OHLC values using MA, volatility, and BB state forecasts.
        
        Parameters
        ----------
        ma_forecast : np.ndarray
            Forecasted moving average values
        vol_forecast : np.ndarray
            Forecasted volatility values
        bb_states : np.ndarray
            Forecasted Bollinger Band states
        current_close : float
            Current closing price
        n_days : int
            Number of days to forecast
            
        Returns
        -------
        dict
            Forecasted OHLC values with confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        n_days = min(n_days, len(ma_forecast), len(vol_forecast), len(bb_states))
        
        forecasts = {
            'dates': pd.date_range(start=self.ohlc_data.index[-1] + timedelta(days=1), 
                                 periods=n_days, freq='D'),
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'open_ci': [],
            'high_ci': [],
            'low_ci': [],
            'close_ci': []
        }
        
        prev_close = current_close
        
        for day in range(n_days):
            # Get forecasted parameters
            forecasted_ma = ma_forecast[day]
            forecasted_vol = vol_forecast[day]
            bb_state = int(bb_states[day])
            
            # Get regime statistics
            trend_today = self.ohlc_data['Trend'].iloc[-1]

            regime_stats = self._get_regime_stats(bb_state, forecasted_vol, trend_today)
            
            # Forecast Close first (using MA and volatility)
            close_forecast = self._forecast_close(prev_close, forecasted_ma, forecasted_vol, regime_stats, bb_state)
            
            # Forecast Open (considering gaps)
            open_forecast = self._forecast_open(prev_close, close_forecast['mean'], regime_stats)
            
            # Forecast High and Low (considering range patterns)
            high_low_forecast = self._forecast_high_low(open_forecast['mean'], close_forecast['mean'], 
                                                       forecasted_vol, regime_stats)
            
            # Store forecasts
            forecasts['open'].append(open_forecast['mean'])
            forecasts['high'].append(high_low_forecast['high_mean'])
            forecasts['low'].append(high_low_forecast['low_mean'])
            forecasts['close'].append(close_forecast['mean'])
            
            # Store confidence intervals
            forecasts['open_ci'].append((open_forecast['lower'], open_forecast['upper']))
            forecasts['high_ci'].append((high_low_forecast['high_lower'], high_low_forecast['high_upper']))
            forecasts['low_ci'].append((high_low_forecast['low_lower'], high_low_forecast['low_upper']))
            forecasts['close_ci'].append((close_forecast['lower'], close_forecast['upper']))
            
            # Update for next iteration
            prev_close = close_forecast['mean']
        
        return forecasts
    def _calculate_trend_regimes(self, slope_window: int = 5, threshold: float = 0.0005) -> None:
        """Compute trend regimes using the slope of the moving average."""
        ma = self.ohlc_data['BB_MA']
        slope = ma.diff(slope_window) / slope_window

        # Simple classification
        trend = np.where(slope > threshold, 'Uptrend',
                np.where(slope < -threshold, 'Downtrend', 'Sideways'))

        self.ohlc_data['Trend'] = trend

    def _get_regime_stats(self, bb_state: int, vol_level: float, trend: str) -> Dict:
        """Get statistical parameters for current regime."""
        # Map BB state to regime
        if bb_state <= 1:
            bb_regime = 'Low_BB'
        elif bb_state >= 3:
            bb_regime = 'High_BB'
        else:
            bb_regime = 'Middle_BB'
        
        # Determine volatility regime
        vol_median = self.ohlc_data['BB_Width'].median()
        vol_regime = 'High_Vol' if vol_level > vol_median else 'Low_Vol'
        
        # Get base statistics
        regime_key = f"{bb_regime.replace('_BB','')}_{trend}"  # e.g. Low_Up, Middle_Down
        base_stats = self.intraday_stats.get(regime_key, self.intraday_stats.get('Middle_Sideways', {}))
        vol_stats = self.intraday_stats.get(vol_regime, {})
        
        # Combine statistics
        combined_stats = {
            'range_pct_mean': base_stats.get('range_pct_mean', 0.02),
            'range_pct_std': base_stats.get('range_pct_std', 0.01),
            'body_pct_mean': base_stats.get('body_pct_mean', 0.0),
            'body_pct_std': max(base_stats.get('body_pct_std', 0.015), vol_stats.get('body_pct_std', 0.015)),
            'gap_pct_mean': base_stats.get('gap_pct_mean', 0.0),
            'gap_pct_std': base_stats.get('gap_pct_std', 0.005),
            'upper_shadow_pct_mean': base_stats.get('upper_shadow_pct_mean', 0.005),
            'lower_shadow_pct_mean': base_stats.get('lower_shadow_pct_mean', 0.005)
        }
        
        return combined_stats
    def _calculate_trend_direction(self) -> None:
        """Add trend direction column based on MA slope."""
        ma = self.ohlc_data['BB_MA']
        ma_slope = ma.diff(self.bb_window // 2)
        threshold = ma.mean() * 0.002  # ~0.2% trend filter

        self.ohlc_data['Trend'] = np.select(
            [
                ma_slope > threshold,
                ma_slope < -threshold
            ],
            [
                'Up',
                'Down'
            ],
            default='Sideways'
        )

    def _forecast_close(self, prev_close: float, ma_forecast: float, vol_forecast: float, regime_stats: Dict, bb_state: int) -> Dict:
        trend = self.ohlc_data['Trend'].iloc[-1]
        # Use the state-specific BB position statistics to sample a position
        if bb_state in self.bb_position_stats:
            bb_pos_mean = self.bb_position_stats[bb_state]['mean']
            bb_pos_std = self.bb_position_stats[bb_state]['std']
            bb_position = np.random.normal(bb_pos_mean, bb_pos_std)
        else:
            # Fallback to state boundaries
            if self.markov_model.fitted and self.markov_model.global_model.state_boundaries:
                if bb_state == 0:
                    bb_position = self.markov_model.global_model.state_boundaries[0] - 0.1
                elif bb_state == self.markov_model.global_model.n_states - 1:
                    bb_position = self.markov_model.global_model.state_boundaries[-1] + 0.1
                else:
                    lower = self.markov_model.global_model.state_boundaries[bb_state-1]
                    upper = self.markov_model.global_model.state_boundaries[bb_state]
                    bb_position = (lower + upper) / 2
            else:
                bb_position = 0.0  # Default neutral position
        
        close_mean = ma_forecast + bb_position * vol_forecast * ma_forecast
        body_std = regime_stats.get('body_pct_std', 0.015)
        total_std = np.sqrt(vol_forecast**2 + body_std**2)
        close_std = prev_close * total_std
        return {
            'mean': close_mean,
            'lower': close_mean - 1.96 * close_std,
            'upper': close_mean + 1.96 * close_std
        }
    
    def _forecast_open(self, prev_close: float, close_forecast: float, regime_stats: Dict) -> Dict:
        """Forecast opening price."""
        # Gap statistics
        gap_mean = regime_stats['gap_pct_mean']
        gap_std = regime_stats['gap_pct_std']
        
        # Expected gap
        expected_gap = prev_close * gap_mean
        open_mean = prev_close + expected_gap
        
        # Confidence interval
        gap_uncertainty = prev_close * gap_std
        
        return {
            'mean': open_mean,
            'lower': open_mean - 1.96 * gap_uncertainty,
            'upper': open_mean + 1.96 * gap_uncertainty
        }
    
    def _forecast_high_low(self, open_forecast: float, close_forecast: float, vol_forecast: float, regime_stats: Dict) -> Dict:
        """Forecast high and low prices."""
        # Reference price (average of open and close)
        ref_price = (open_forecast + close_forecast) / 2
        
        # Expected range
        range_pct_mean = regime_stats['range_pct_mean']
        range_pct_std = regime_stats['range_pct_std']
        
        expected_range = ref_price * range_pct_mean
        range_std = ref_price * range_pct_std
        
        # Shadow statistics
        upper_shadow_pct = regime_stats['upper_shadow_pct_mean']
        lower_shadow_pct = regime_stats['lower_shadow_pct_mean']
        
        # Calculate high and low relative to open/close envelope
        body_high = max(open_forecast, close_forecast)
        body_low = min(open_forecast, close_forecast)
        
        # Add shadows
        high_mean = body_high + ref_price * upper_shadow_pct
        low_mean = body_low - ref_price * lower_shadow_pct
        
        # Ensure minimum range
        current_range = high_mean - low_mean
        if current_range < expected_range:
            range_extension = (expected_range - current_range) / 2
            high_mean += range_extension
            low_mean -= range_extension
        
        # Confidence intervals
        range_uncertainty = range_std
        
        return {
            'high_mean': high_mean,
            'high_lower': high_mean - range_uncertainty,
            'high_upper': high_mean + range_uncertainty,
            'low_mean': low_mean,
            'low_lower': low_mean - range_uncertainty,
            'low_upper': low_mean + range_uncertainty
        }
    
    def plot_ohlc_forecast(self, forecast_results: Dict, n_historical: int = 60) -> None:
        """Plot OHLC forecast with historical data."""
        # Get historical data
        hist_data = self.ohlc_data.tail(n_historical)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot background trend shading
        trend_colors = {'Uptrend': '#d2fbd4', 'Downtrend': '#fddede', 'Sideways': '#fdf8d2'}
        for i, (date, row) in enumerate(hist_data.iterrows()):
            trend = row.get('Trend', 'Sideways')
            color = trend_colors.get(trend, '#f0f0f0')
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.2, linewidth=0)

        # Plot historical candlesticks (simplified)
        for i, (date, row) in enumerate(hist_data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            # High-Low line
            ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
            # Body rectangle (simplified as line)
            ax.plot([i, i], [row['Open'], row['Close']], color=color, linewidth=4, alpha=0.7)
        
        # Plot forecasted values
        forecast_start = len(hist_data)
        n_forecast = len(forecast_results['open'])
        
        for i in range(n_forecast):
            x_pos = forecast_start + i
            
            # Forecasted OHLC
            o = forecast_results['open'][i]
            h = forecast_results['high'][i]
            l = forecast_results['low'][i]
            c = forecast_results['close'][i]
            
            color = 'darkgreen' if c >= o else 'darkred'
            
            # High-Low line
            ax.plot([x_pos, x_pos], [l, h], color='blue', linewidth=2, alpha=0.7)
            # Body
            ax.plot([x_pos, x_pos], [o, c], color=color, linewidth=6, alpha=0.8)
            
            # Confidence intervals (for close)
            close_ci = forecast_results['close_ci'][i]
            ax.fill_between([x_pos-0.3, x_pos+0.3], [close_ci[0], close_ci[0]], 
                           [close_ci[1], close_ci[1]], alpha=0.2, color='blue')
        
        # Add vertical line separating historical and forecast
        ax.axvline(x=forecast_start-0.5, color='orange', linestyle='--', linewidth=2, 
                  label='Forecast Start')
        
        ax.set_title('OHLC Forecast', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_forecast_summary(self, forecast_results: Dict) -> pd.DataFrame:
        """Get summary table of forecast results."""
        n_days = len(forecast_results['open'])
        
        summary_data = []
        for i in range(n_days):
            row = {
                'Date': forecast_results['dates'][i],
                'Open': forecast_results['open'][i],
                'High': forecast_results['high'][i],
                'Low': forecast_results['low'][i],
                'Close': forecast_results['close'][i],
                'Range': forecast_results['high'][i] - forecast_results['low'][i],
                'Body': forecast_results['close'][i] - forecast_results['open'][i],
                'Close_CI_Lower': forecast_results['close_ci'][i][0],
                'Close_CI_Upper': forecast_results['close_ci'][i][1]
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)