import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

class FactorBasedStockSelector:
    """
    Stock selection system using factor forecasts and loadings to identify
    stocks most likely to go up or down.
    """
    
    def __init__(self, forecast_horizon: int = 20, confidence_threshold: float = 0.6):
        """
        Initialize the factor-based stock selector.
        
        Parameters
        ----------
        forecast_horizon : int
            Number of days to forecast ahead
        confidence_threshold : float
            Minimum confidence threshold for predictions
        """
        self.forecast_horizon = forecast_horizon
        self.confidence_threshold = confidence_threshold
        self.factor_forecasts = None
        self.factor_loadings = None
        self.stock_scores = None
        self.fitted = False
        
    def fit(self, components_df: pd.DataFrame, loadings_df: pd.DataFrame) -> 'FactorBasedStockSelector':
        """
        Fit the stock selector with factor data.
        
        Parameters
        ----------
        components_df : pd.DataFrame
            Factor time series data
        loadings_df : pd.DataFrame
            Factor loadings (multi-indexed by date and asset)
            
        Returns
        -------
        self
        """
        self.components_df = components_df.copy()
        self.loadings_df = loadings_df.copy()
        
        # Generate factor forecasts
        print("🔮 Generating factor forecasts...")
        self.factor_forecasts = self._forecast_factors()
        
        # Get latest factor loadings
        print("📊 Extracting latest factor loadings...")
        self.factor_loadings = self._get_latest_loadings()
        
        # Calculate stock scores
        print("🎯 Calculating stock directional scores...")
        self.stock_scores = self._calculate_stock_scores()
        
        self.fitted = True
        return self
    
    def _forecast_factors(self) -> Dict:
        """Generate forecasts for each factor using ARIMA or simple methods."""
        forecasts = {}
        
        for factor in self.components_df.columns:
            factor_series = self.components_df[factor].dropna()
            
            if len(factor_series) < 50:
                print(f"⚠️ Insufficient data for {factor}, skipping...")
                continue
                
            try:
                if ARIMA_AVAILABLE:
                    forecast_result = self._arima_forecast(factor_series, factor)
                else:
                    forecast_result = self._simple_forecast(factor_series, factor)
                
                forecasts[factor] = forecast_result
                
            except Exception as e:
                print(f"❌ Error forecasting {factor}: {e}")
                # Fallback to simple forecast
                forecasts[factor] = self._simple_forecast(factor_series, factor)
        
        return forecasts
    
    def _arima_forecast(self, series: pd.Series, factor_name: str) -> Dict:
        """Generate ARIMA forecast for a factor."""
        # Find optimal ARIMA parameters (simplified search)
        best_aic = np.inf
        best_model = None
        best_params = (1, 1, 1)
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                            best_params = (p, d, q)
                    except:
                        continue
        
        if best_model is None:
            return self._simple_forecast(series, factor_name)
        
        # Generate forecast
        forecast = best_model.forecast(steps=self.forecast_horizon)
        forecast_ci = best_model.get_forecast(steps=self.forecast_horizon).conf_int()
        
        # Calculate forecast confidence (based on CI width)
        ci_width = forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]
        avg_ci_width = ci_width.mean()
        
        # Confidence inversely related to CI width
        forecast_confidence = 1.0 / (1.0 + avg_ci_width / series.std())
        
        return {
            'method': 'ARIMA',
            'params': best_params,
            'forecast': forecast.values,
            'confidence': forecast_confidence,
            'trend': np.mean(np.diff(forecast.values)),  # Average daily change
            'direction': 1 if np.mean(forecast.values) > series.iloc[-1] else -1,
            'magnitude': abs(np.mean(forecast.values) - series.iloc[-1]) / series.std()
        }
    
    def _simple_forecast(self, series: pd.Series, factor_name: str) -> Dict:
        """Generate simple forecast based on recent trends."""
        # Calculate recent trend
        recent_trend = series.tail(10).diff().mean()
        
        # Simple linear extrapolation
        current_value = series.iloc[-1]
        forecast_values = [current_value + recent_trend * (i + 1) for i in range(self.forecast_horizon)]
        
        # Estimate confidence based on trend consistency
        recent_changes = series.tail(20).diff().dropna()
        trend_consistency = 1.0 - (recent_changes.std() / (abs(recent_trend) + 1e-6))
        forecast_confidence = max(0.1, min(0.9, trend_consistency))
        
        return {
            'method': 'Simple',
            'params': None,
            'forecast': np.array(forecast_values),
            'confidence': forecast_confidence,
            'trend': recent_trend,
            'direction': 1 if recent_trend > 0 else -1,
            'magnitude': abs(recent_trend) * self.forecast_horizon / series.std()
        }
    
    def _get_latest_loadings(self) -> pd.DataFrame:
        """Get the most recent factor loadings."""
        latest_date = self.loadings_df.index.get_level_values('date').max()
        latest_loadings = self.loadings_df.xs(latest_date, level='date')
        return latest_loadings
    
    def _calculate_stock_scores(self) -> pd.DataFrame:
        """Calculate directional scores for each stock."""
        scores_data = []
        
        for asset in self.factor_loadings.index:
            asset_loadings = self.factor_loadings.loc[asset]
            
            # Calculate weighted score based on factor forecasts and loadings
            total_score = 0.0
            total_confidence = 0.0
            factor_contributions = {}
            
            for factor in self.factor_forecasts.keys():
                if factor in asset_loadings.index:
                    loading = asset_loadings[factor]
                    forecast_info = self.factor_forecasts[factor]
                    
                    # Factor contribution to stock return
                    factor_return_contribution = (
                        loading * 
                        forecast_info['direction'] * 
                        forecast_info['magnitude']
                    )
                    
                    # Weight by forecast confidence
                    weighted_contribution = (
                        factor_return_contribution * 
                        forecast_info['confidence']
                    )
                    
                    total_score += weighted_contribution
                    total_confidence += forecast_info['confidence']
                    
                    factor_contributions[factor] = {
                        'loading': loading,
                        'forecast_direction': forecast_info['direction'],
                        'forecast_magnitude': forecast_info['magnitude'],
                        'confidence': forecast_info['confidence'],
                        'contribution': weighted_contribution
                    }
            
            # Normalize by total confidence
            if total_confidence > 0:
                normalized_score = total_score / total_confidence
            else:
                normalized_score = 0.0
            
            # Calculate overall confidence
            avg_confidence = total_confidence / len(self.factor_forecasts) if len(self.factor_forecasts) > 0 else 0.0
            
            # Determine direction and strength
            direction = 'Long' if normalized_score > 0 else 'Short'
            strength = abs(normalized_score)
            
            scores_data.append({
                'Asset': asset,
                'Score': normalized_score,
                'Abs_Score': abs(normalized_score),
                'Direction': direction,
                'Strength': strength,
                'Confidence': avg_confidence,
                'Factor_Contributions': factor_contributions
            })
        
        scores_df = pd.DataFrame(scores_data)
        scores_df = scores_df.sort_values('Abs_Score', ascending=False)
        
        return scores_df
    
    def get_top_long_picks(self, n_picks: int = 20, min_confidence: float = None) -> pd.DataFrame:
        """Get top long (buy) recommendations."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        min_conf = min_confidence or self.confidence_threshold
        
        long_picks = self.stock_scores[
            (self.stock_scores['Direction'] == 'Long') & 
            (self.stock_scores['Confidence'] >= min_conf)
        ].head(n_picks)
        
        return long_picks[['Asset', 'Score', 'Strength', 'Confidence']].copy()
    
    def get_top_short_picks(self, n_picks: int = 20, min_confidence: float = None) -> pd.DataFrame:
        """Get top short (sell) recommendations."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        min_conf = min_confidence or self.confidence_threshold
        
        short_picks = self.stock_scores[
            (self.stock_scores['Direction'] == 'Short') & 
            (self.stock_scores['Confidence'] >= min_conf)
        ].head(n_picks)
        
        return short_picks[['Asset', 'Score', 'Strength', 'Confidence']].copy()
    
    def get_factor_attribution(self, asset: str) -> pd.DataFrame:
        """Get detailed factor attribution for a specific asset."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        asset_row = self.stock_scores[self.stock_scores['Asset'] == asset]
        if len(asset_row) == 0:
            raise ValueError(f"Asset {asset} not found")
        
        factor_contributions = asset_row.iloc[0]['Factor_Contributions']
        
        attribution_data = []
        for factor, contrib in factor_contributions.items():
            attribution_data.append({
                'Factor': factor,
                'Loading': contrib['loading'],
                'Forecast_Direction': '+' if contrib['forecast_direction'] > 0 else '-',
                'Forecast_Magnitude': contrib['forecast_magnitude'],
                'Confidence': contrib['confidence'],
                'Contribution': contrib['contribution']
            })
        
        attribution_df = pd.DataFrame(attribution_data)
        attribution_df = attribution_df.sort_values('Contribution', key=abs, ascending=False)
        
        return attribution_df
    
    def plot_stock_scores_distribution(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot distribution of stock scores."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Score distribution histogram
        ax1.hist(self.stock_scores['Score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax1.set_title('Stock Score Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Score vs Confidence scatter
        colors = ['green' if d == 'Long' else 'red' for d in self.stock_scores['Direction']]
        ax2.scatter(self.stock_scores['Confidence'], self.stock_scores['Score'], 
                   c=colors, alpha=0.6, s=30)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Score vs Confidence', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        
        # Top long picks
        top_longs = self.get_top_long_picks(10, min_confidence=0.0)
        if len(top_longs) > 0:
            ax3.barh(range(len(top_longs)), top_longs['Score'], color='green', alpha=0.7)
            ax3.set_yticks(range(len(top_longs)))
            ax3.set_yticklabels(top_longs['Asset'])
            ax3.set_title('Top 10 Long Picks', fontweight='bold', fontsize=14)
            ax3.set_xlabel('Score')
            ax3.grid(True, alpha=0.3)
        
        # Top short picks
        top_shorts = self.get_top_short_picks(10, min_confidence=0.0)
        if len(top_shorts) > 0:
            ax4.barh(range(len(top_shorts)), top_shorts['Score'], color='red', alpha=0.7)
            ax4.set_yticks(range(len(top_shorts)))
            ax4.set_yticklabels(top_shorts['Asset'])
            ax4.set_title('Top 10 Short Picks', fontweight='bold', fontsize=14)
            ax4.set_xlabel('Score')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_factor_attribution(self, asset: str, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot factor attribution for a specific asset."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        attribution_df = self.get_factor_attribution(asset)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Factor contributions
        colors = ['green' if x > 0 else 'red' for x in attribution_df['Contribution']]
        bars = ax1.barh(range(len(attribution_df)), attribution_df['Contribution'], 
                       color=colors, alpha=0.7)
        ax1.set_yticks(range(len(attribution_df)))
        ax1.set_yticklabels(attribution_df['Factor'])
        ax1.set_title(f'{asset} - Factor Contributions', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Contribution to Score')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.8)
        ax1.grid(True, alpha=0.3)
        
        # Factor loadings
        colors = ['blue' if x > 0 else 'orange' for x in attribution_df['Loading']]
        ax2.barh(range(len(attribution_df)), attribution_df['Loading'], 
                color=colors, alpha=0.7)
        ax2.set_yticks(range(len(attribution_df)))
        ax2.set_yticklabels(attribution_df['Factor'])
        ax2.set_title(f'{asset} - Factor Loadings', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Loading')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_portfolio_construction_report(self, long_picks: int = 20, short_picks: int = 20, 
                                        min_confidence: float = None) -> Dict:
        """Generate comprehensive portfolio construction report."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        min_conf = min_confidence or self.confidence_threshold
        
        # Get recommendations
        top_longs = self.get_top_long_picks(long_picks, min_conf)
        top_shorts = self.get_top_short_picks(short_picks, min_conf)
        
        # Calculate portfolio statistics
        long_avg_score = top_longs['Score'].mean() if len(top_longs) > 0 else 0
        short_avg_score = top_shorts['Score'].mean() if len(top_shorts) > 0 else 0
        long_avg_confidence = top_longs['Confidence'].mean() if len(top_longs) > 0 else 0
        short_avg_confidence = top_shorts['Confidence'].mean() if len(top_shorts) > 0 else 0
        
        # Factor exposure analysis
        factor_exposures = self._calculate_portfolio_factor_exposures(top_longs, top_shorts)
        
        report = {
            'summary': {
                'total_long_picks': len(top_longs),
                'total_short_picks': len(top_shorts),
                'long_avg_score': long_avg_score,
                'short_avg_score': short_avg_score,
                'long_avg_confidence': long_avg_confidence,
                'short_avg_confidence': short_avg_confidence,
                'forecast_horizon_days': self.forecast_horizon
            },
            'long_picks': top_longs,
            'short_picks': top_shorts,
            'factor_exposures': factor_exposures,
            'factor_forecasts_summary': self._get_factor_forecasts_summary()
        }
        
        return report
    
    def _calculate_portfolio_factor_exposures(self, long_picks: pd.DataFrame, 
                                           short_picks: pd.DataFrame) -> pd.DataFrame:
        """Calculate net factor exposures for the portfolio."""
        factor_exposures = {}
        
        # Get all factors
        all_factors = list(self.factor_forecasts.keys())
        
        for factor in all_factors:
            long_exposure = 0.0
            short_exposure = 0.0
            
            # Long exposure
            for asset in long_picks['Asset']:
                if asset in self.factor_loadings.index and factor in self.factor_loadings.columns:
                    loading = self.factor_loadings.loc[asset, factor]
                    long_exposure += loading / len(long_picks) if len(long_picks) > 0 else 0
            
            # Short exposure (negative)
            for asset in short_picks['Asset']:
                if asset in self.factor_loadings.index and factor in self.factor_loadings.columns:
                    loading = self.factor_loadings.loc[asset, factor]
                    short_exposure -= loading / len(short_picks) if len(short_picks) > 0 else 0
            
            net_exposure = long_exposure + short_exposure
            
            factor_exposures[factor] = {
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_exposure': net_exposure,
                'forecast_direction': self.factor_forecasts[factor]['direction'],
                'forecast_confidence': self.factor_forecasts[factor]['confidence']
            }
        
        # Convert to DataFrame
        exposure_data = []
        for factor, exposures in factor_exposures.items():
            exposure_data.append({
                'Factor': factor,
                'Long_Exposure': exposures['long_exposure'],
                'Short_Exposure': exposures['short_exposure'],
                'Net_Exposure': exposures['net_exposure'],
                'Forecast_Direction': exposures['forecast_direction'],
                'Forecast_Confidence': exposures['forecast_confidence']
            })
        
        return pd.DataFrame(exposure_data)
    
    def _get_factor_forecasts_summary(self) -> pd.DataFrame:
        """Get summary of factor forecasts."""
        summary_data = []
        
        for factor, forecast_info in self.factor_forecasts.items():
            summary_data.append({
                'Factor': factor,
                'Method': forecast_info['method'],
                'Direction': '+' if forecast_info['direction'] > 0 else '-',
                'Magnitude': forecast_info['magnitude'],
                'Confidence': forecast_info['confidence'],
                'Trend_Per_Day': forecast_info['trend']
            })
        
        return pd.DataFrame(summary_data)
    
    def print_portfolio_report(self, long_picks: int = 20, short_picks: int = 20, 
                             min_confidence: float = None) -> None:
        """Print comprehensive portfolio construction report."""
        report = self.get_portfolio_construction_report(long_picks, short_picks, min_confidence)
        
        print("\n" + "=" * 80)
        print("🎯 FACTOR-BASED STOCK SELECTION REPORT")
        print("=" * 80)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 PORTFOLIO SUMMARY:")
        print(f"  Long Positions: {summary['total_long_picks']}")
        print(f"  Short Positions: {summary['total_short_picks']}")
        print(f"  Forecast Horizon: {summary['forecast_horizon_days']} days")
        print(f"  Long Avg Score: {summary['long_avg_score']:+.4f}")
        print(f"  Short Avg Score: {summary['short_avg_score']:+.4f}")
        print(f"  Long Avg Confidence: {summary['long_avg_confidence']:.3f}")
        print(f"  Short Avg Confidence: {summary['short_avg_confidence']:.3f}")
        
        # Top picks
        if len(report['long_picks']) > 0:
            print(f"\n📈 TOP LONG PICKS:")
            for i, row in report['long_picks'].head(10).iterrows():
                print(f"  {row['Asset']}: Score={row['Score']:+.4f}, Confidence={row['Confidence']:.3f}")
        
        if len(report['short_picks']) > 0:
            print(f"\n📉 TOP SHORT PICKS:")
            for i, row in report['short_picks'].head(10).iterrows():
                print(f"  {row['Asset']}: Score={row['Score']:+.4f}, Confidence={row['Confidence']:.3f}")
        
        # Factor forecasts
        print(f"\n🔮 FACTOR FORECASTS:")
        forecasts_summary = report['factor_forecasts_summary']
        for i, row in forecasts_summary.iterrows():
            direction_icon = "📈" if row['Direction'] == '+' else "📉"
            print(f"  {direction_icon} {row['Factor']}: {row['Direction']} (Conf: {row['Confidence']:.3f}, Mag: {row['Magnitude']:.3f})")
        
        # Factor exposures
        print(f"\n⚖️ PORTFOLIO FACTOR EXPOSURES:")
        exposures = report['factor_exposures']
        for i, row in exposures.iterrows():
            exposure_icon = "🟢" if abs(row['Net_Exposure']) < 0.1 else "🔴"
            print(f"  {exposure_icon} {row['Factor']}: Net={row['Net_Exposure']:+.4f} (Long={row['Long_Exposure']:+.3f}, Short={row['Short_Exposure']:+.3f})")
        
        print("\n" + "=" * 80)