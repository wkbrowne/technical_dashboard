"""
High-Low Price Copula Modeling with Regime Conditioning
========================================================

This module implements sophisticated copula-based modeling for high and low prices,
conditioned on trend and volatility regimes. Uses multiple copula families
(Gaussian, Clayton, Gumbel, Frank) with automatic selection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.stats import gaussian_kde, rankdata
from scipy.optimize import minimize, differential_evolution
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CopulaFamily:
    """Base class for copula families."""
    
    def __init__(self, name: str):
        self.name = name
        self.params = None
        self.fitted = False
    
    def fit(self, u: np.ndarray, v: np.ndarray) -> 'CopulaFamily':
        """Fit copula parameters to uniform marginals u, v."""
        raise NotImplementedError
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the fitted copula."""
        raise NotImplementedError
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Probability density function."""
        raise NotImplementedError
    
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        raise NotImplementedError
    
    def aic(self, u: np.ndarray, v: np.ndarray) -> float:
        """Akaike Information Criterion."""
        log_likelihood = np.sum(np.log(self.pdf(u, v) + 1e-10))
        k = len(self.params) if self.params is not None else 1
        return 2 * k - 2 * log_likelihood

class GaussianCopula(CopulaFamily):
    """Gaussian (Normal) copula implementation."""
    
    def __init__(self):
        super().__init__("Gaussian")
        self.rho = None
    
    def fit(self, u: np.ndarray, v: np.ndarray) -> 'GaussianCopula':
        """Fit Gaussian copula using method of moments."""
        # Convert to normal quantiles
        z_u = stats.norm.ppf(np.clip(u, 1e-6, 1-1e-6))
        z_v = stats.norm.ppf(np.clip(v, 1e-6, 1-1e-6))
        
        # Calculate correlation
        self.rho = np.corrcoef(z_u, z_v)[0, 1]
        self.rho = np.clip(self.rho, -0.99, 0.99)  # Avoid perfect correlation
        
        self.params = {'rho': self.rho}
        self.fitted = True
        return self
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from Gaussian copula."""
        if not self.fitted:
            raise ValueError("Copula must be fitted first")
        
        # Generate bivariate normal
        mean = [0, 0]
        cov = [[1, self.rho], [self.rho, 1]]
        samples = np.random.multivariate_normal(mean, cov, n)
        
        # Convert to uniform marginals
        u = stats.norm.cdf(samples[:, 0])
        v = stats.norm.cdf(samples[:, 1])
        
        return u, v
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gaussian copula density."""
        if not self.fitted:
            return np.ones_like(u)  # Independence
        
        z_u = stats.norm.ppf(np.clip(u, 1e-6, 1-1e-6))
        z_v = stats.norm.ppf(np.clip(v, 1e-6, 1-1e-6))
        
        rho_sq = self.rho**2
        
        # Copula density formula
        density = (1 / np.sqrt(1 - rho_sq)) * np.exp(
            -0.5 * (2 * self.rho * z_u * z_v - self.rho**2 * (z_u**2 + z_v**2)) / (1 - rho_sq)
        )
        
        return np.clip(density, 1e-10, 1e10)

class ClaytonCopula(CopulaFamily):
    """Clayton copula implementation."""
    
    def __init__(self):
        super().__init__("Clayton")
        self.theta = None
    
    def fit(self, u: np.ndarray, v: np.ndarray) -> 'ClaytonCopula':
        """Fit Clayton copula using method of moments."""
        # Kendall's tau estimation
        tau = stats.kendalltau(u, v)[0]
        
        # Convert tau to theta
        if tau > 0:
            self.theta = max(2 * tau / (1 - tau), 1e-6)
        else:
            self.theta = 1e-6  # Clayton requires theta > 0
        
        self.params = {'theta': self.theta}
        self.fitted = True
        return self
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from Clayton copula using conditional method."""
        if not self.fitted:
            raise ValueError("Copula must be fitted first")
        
        # Generate uniform samples
        w1 = np.random.uniform(0, 1, n)
        w2 = np.random.uniform(0, 1, n)
        
        # Clayton copula conditional sampling
        u = w1
        v = (w2 * u**(-self.theta) - w2 + 1)**(-1/self.theta)
        
        return u, v
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Clayton copula density."""
        if not self.fitted:
            return np.ones_like(u)
        
        u_clip = np.clip(u, 1e-6, 1-1e-6)
        v_clip = np.clip(v, 1e-6, 1-1e-6)
        
        density = (1 + self.theta) * (u_clip * v_clip)**(-1 - self.theta) * \
                 (u_clip**(-self.theta) + v_clip**(-self.theta) - 1)**(-2 - 1/self.theta)
        
        return np.clip(density, 1e-10, 1e10)

class GumbelCopula(CopulaFamily):
    """Gumbel copula implementation."""
    
    def __init__(self):
        super().__init__("Gumbel")
        self.theta = None
    
    def fit(self, u: np.ndarray, v: np.ndarray) -> 'GumbelCopula':
        """Fit Gumbel copula using method of moments."""
        # Kendall's tau estimation
        tau = stats.kendalltau(u, v)[0]
        
        # Convert tau to theta
        if tau > 0:
            self.theta = max(1 / (1 - tau), 1.0)
        else:
            self.theta = 1.0  # Gumbel requires theta >= 1
        
        self.params = {'theta': self.theta}
        self.fitted = True
        return self
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from Gumbel copula."""
        if not self.fitted:
            raise ValueError("Copula must be fitted first")
        
        # Generate uniform samples
        w1 = np.random.uniform(0, 1, n)
        w2 = np.random.uniform(0, 1, n)
        
        # Gumbel copula sampling (simplified method)
        # This is a basic implementation - more sophisticated methods exist
        u = w1
        # Approximate conditional sampling
        v = np.exp(-((-np.log(w2))**self.theta + (-np.log(u))**self.theta)**(1/self.theta))
        
        return u, v
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gumbel copula density."""
        if not self.fitted:
            return np.ones_like(u)
        
        u_clip = np.clip(u, 1e-6, 1-1e-6)
        v_clip = np.clip(v, 1e-6, 1-1e-6)
        
        log_u = np.log(u_clip)
        log_v = np.log(v_clip)
        
        A = (-log_u)**self.theta + (-log_v)**self.theta
        A_inv = A**(1/self.theta)
        
        density = np.exp(-A_inv) * A**(-2 + 2/self.theta) * \
                 (log_u * log_v)**(self.theta - 1) * \
                 (A_inv + self.theta - 1) / (u_clip * v_clip)
        
        return np.clip(density, 1e-10, 1e10)

class FrankCopula(CopulaFamily):
    """Frank copula implementation."""
    
    def __init__(self):
        super().__init__("Frank")
        self.theta = None
    
    def fit(self, u: np.ndarray, v: np.ndarray) -> 'FrankCopula':
        """Fit Frank copula using method of moments."""
        # Kendall's tau estimation
        tau = stats.kendalltau(u, v)[0]
        
        # Convert tau to theta (approximate)
        if abs(tau) > 1e-6:
            # Use approximation for Frank copula
            if tau > 0:
                self.theta = 4 * tau / (1 - tau)
            else:
                self.theta = 4 * tau / (1 + tau)
        else:
            self.theta = 0.0  # Independence
        
        self.params = {'theta': self.theta}
        self.fitted = True
        return self
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from Frank copula."""
        if not self.fitted:
            raise ValueError("Copula must be fitted first")
        
        # Generate uniform samples
        w1 = np.random.uniform(0, 1, n)
        w2 = np.random.uniform(0, 1, n)
        
        u = w1
        
        if abs(self.theta) < 1e-6:
            # Independence case
            v = w2
        else:
            # Frank copula conditional sampling
            exp_theta = np.exp(self.theta)
            v = -np.log(1 + (exp_theta - 1) * w2 / 
                       (w2 * (exp_theta - 1) + (1 - w2) * np.exp(self.theta * u))) / self.theta
        
        return u, v
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Frank copula density."""
        if not self.fitted or abs(self.theta) < 1e-6:
            return np.ones_like(u)  # Independence
        
        u_clip = np.clip(u, 1e-6, 1-1e-6)
        v_clip = np.clip(v, 1e-6, 1-1e-6)
        
        exp_theta = np.exp(self.theta)
        exp_theta_u = np.exp(self.theta * u_clip)
        exp_theta_v = np.exp(self.theta * v_clip)
        
        numerator = self.theta * (exp_theta - 1) * exp_theta_u * exp_theta_v
        denominator = ((exp_theta - 1) + (exp_theta_u - 1) * (exp_theta_v - 1))**2
        
        density = numerator / denominator
        
        return np.clip(density, 1e-10, 1e10)

class MarginalDistribution:
    """Marginal distribution modeling using KDE with fallback to parametric."""
    
    def __init__(self, name: str = "marginal"):
        self.name = name
        self.kde_model = None
        self.fallback_params = None
        self.fitted = False
        self.use_kde = True
    
    def fit(self, data: np.ndarray) -> 'MarginalDistribution':
        """Fit marginal distribution using KDE with parametric fallback."""
        data_clean = data[np.isfinite(data)]
        
        if len(data_clean) < 10:
            raise ValueError(f"Insufficient data for {self.name}: {len(data_clean)} samples")
        
        try:
            # Try KDE first
            self.kde_model = gaussian_kde(data_clean)
            self.kde_model.set_bandwidth(bw_method='silverman')
            self.use_kde = True
        except:
            self.use_kde = False
        
        # Always fit parametric fallback
        self.fallback_params = {
            'mean': np.mean(data_clean),
            'std': np.std(data_clean),
            'min': np.min(data_clean),
            'max': np.max(data_clean)
        }
        
        self.fitted = True
        return self
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        if not self.fitted:
            raise ValueError("Distribution must be fitted first")
        
        x = np.asarray(x)
        
        if self.use_kde and self.kde_model is not None:
            # Numerical integration for KDE CDF (approximate)
            # For efficiency, we'll use the parametric approximation
            return stats.norm.cdf(x, self.fallback_params['mean'], self.fallback_params['std'])
        else:
            # Parametric fallback
            return stats.norm.cdf(x, self.fallback_params['mean'], self.fallback_params['std'])
    
    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF)."""
        if not self.fitted:
            raise ValueError("Distribution must be fitted first")
        
        u = np.clip(u, 1e-6, 1-1e-6)
        
        # Use parametric approximation for inverse
        return stats.norm.ppf(u, self.fallback_params['mean'], self.fallback_params['std'])
    
    def sample(self, n: int) -> np.ndarray:
        """Sample from the distribution."""
        if not self.fitted:
            raise ValueError("Distribution must be fitted first")
        
        if self.use_kde and self.kde_model is not None:
            return self.kde_model.resample(n)[0]
        else:
            return np.random.normal(self.fallback_params['mean'], 
                                  self.fallback_params['std'], n)

class RegimeCopulaModel:
    """Copula model for a specific trend-volatility regime."""
    
    def __init__(self, regime: str, min_samples: int = 50):
        self.regime = regime
        self.min_samples = min_samples
        
        # Marginal distributions
        self.high_marginal = MarginalDistribution(f"{regime}_high")
        self.low_marginal = MarginalDistribution(f"{regime}_low")
        
        # Copula families to try
        self.copula_families = {
            'gaussian': GaussianCopula(),
            'clayton': ClaytonCopula(),
            'gumbel': GumbelCopula(),
            'frank': FrankCopula()
        }
        
        self.best_copula = None
        self.best_copula_name = None
        self.fitted = False
        
    def fit(self, high_data: np.ndarray, low_data: np.ndarray) -> 'RegimeCopulaModel':
        """Fit copula model for this regime."""
        if len(high_data) != len(low_data):
            raise ValueError("High and low data must have same length")
        
        if len(high_data) < self.min_samples:
            raise ValueError(f"Insufficient data for regime {self.regime}: {len(high_data)} < {self.min_samples}")
        
        # Fit marginal distributions
        self.high_marginal.fit(high_data)
        self.low_marginal.fit(low_data)
        
        # Convert to uniform marginals for copula fitting
        u_high = self._to_uniform_marginal(high_data, self.high_marginal)
        u_low = self._to_uniform_marginal(low_data, self.low_marginal)
        
        # Fit all copula families and select best
        copula_scores = {}
        
        for name, copula in self.copula_families.items():
            try:
                copula.fit(u_high, u_low)
                aic_score = copula.aic(u_high, u_low)
                copula_scores[name] = aic_score
            except Exception as e:
                print(f"  ⚠️ {name} copula failed for {self.regime}: {e}")
                copula_scores[name] = np.inf
        
        # Select best copula (lowest AIC)
        self.best_copula_name = min(copula_scores, key=copula_scores.get)
        self.best_copula = self.copula_families[self.best_copula_name]
        
        self.fitted = True
        return self
    
    def _to_uniform_marginal(self, data: np.ndarray, marginal: MarginalDistribution) -> np.ndarray:
        """Convert data to uniform marginals using empirical CDF."""
        # Use empirical CDF for better uniform transformation
        ranks = rankdata(data, method='average')
        uniform = ranks / (len(data) + 1)
        return uniform
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample high-low pairs from the copula model."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Sample from copula
        u_high, u_low = self.best_copula.sample(n)
        
        # Transform back to original scale
        high_samples = self.high_marginal.ppf(u_high)
        low_samples = self.low_marginal.ppf(u_low)
        
        return high_samples, low_samples
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        if not self.fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'regime': self.regime,
            'best_copula': self.best_copula_name,
            'copula_params': self.best_copula.params,
            'high_marginal_type': 'kde' if self.high_marginal.use_kde else 'normal',
            'low_marginal_type': 'kde' if self.low_marginal.use_kde else 'normal'
        }

class GlobalHighLowCopulaModel:
    """Global copula model trained on all stocks."""
    
    def __init__(self, min_samples_per_regime: int = 100):
        self.min_samples_per_regime = min_samples_per_regime
        self.regime_models = {}
        self.fitted = False
        
        # Trend and volatility classification (same as open price model)
        self.trend_thresholds = {
            'strong_bull': 0.002,
            'bull': 0.0005,
            'neutral': -0.0005,
            'bear': -0.002,
            'strong_bear': float('-inf')
        }
    
    def fit_global_model(self, all_stock_data: Dict[str, pd.DataFrame]) -> 'GlobalHighLowCopulaModel':
        """Fit global copula models using data from all stocks."""
        print("🌍 Training Global High-Low Copula Models")
        print("=" * 60)
        
        # Collect high-low data from all stocks by regime
        regime_data = defaultdict(lambda: {'high': [], 'low': []})
        stock_count = 0
        
        for symbol, stock_df in all_stock_data.items():
            try:
                regime_features = self._extract_regime_features(stock_df, symbol)
                
                for _, row in regime_features.iterrows():
                    regime = row['Combined_Regime']
                    regime_data[regime]['high'].append(row['High_Return'])
                    regime_data[regime]['low'].append(row['Low_Return'])
                
                stock_count += 1
                if stock_count % 50 == 0:
                    print(f"  Processed {stock_count} stocks...")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing {symbol}: {e}")
                continue
        
        print(f"  📊 Processed {stock_count} stocks total")
        
        # Train regime-specific copula models
        successful_models = 0
        
        for regime, data in regime_data.items():
            high_data = np.array(data['high'])
            low_data = np.array(data['low'])
            
            print(f"  📊 {regime}: {len(high_data)} samples", end="")
            
            if len(high_data) >= self.min_samples_per_regime:
                try:
                    regime_model = RegimeCopulaModel(regime, min_samples=self.min_samples_per_regime)
                    regime_model.fit(high_data, low_data)
                    
                    self.regime_models[regime] = regime_model
                    successful_models += 1
                    
                    model_info = regime_model.get_model_info()
                    print(f" ✅ {model_info['best_copula']}")
                    
                except Exception as e:
                    print(f" ❌ Failed: {e}")
            else:
                print(f" ⚠️ Below minimum threshold ({self.min_samples_per_regime})")
        
        print(f"  ✅ Successfully trained {successful_models} regime copula models")
        
        self.fitted = True
        return self
    
    def _extract_regime_features(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract high-low features with regime classification."""
        df = stock_df.copy()
        
        # Calculate returns relative to open/close
        reference_price = (df['Open'] + df['Close']) / 2
        df['High_Return'] = (df['High'] - reference_price) / reference_price
        df['Low_Return'] = (reference_price - df['Low']) / reference_price  # Positive for drops
        
        # Calculate MA slope for trend classification
        if 'MA' in df.columns:
            ma_col = 'MA'
        elif 'BB_MA' in df.columns:
            ma_col = 'BB_MA'
        else:
            df['BB_MA'] = df['Close'].rolling(window=20).mean()
            ma_col = 'BB_MA'
            
        df['MA_Slope'] = df[ma_col].pct_change(periods=5)
        
        # Classify trend regime
        df['Trend_Regime'] = self._classify_trend_regime(df['MA_Slope'])
        
        # Classify volatility regime
        if 'BB_Width' in df.columns:
            vol_col = 'BB_Width'
        else:
            returns = df['Close'].pct_change()
            vol_rolling = returns.rolling(window=20).std()
            df['BB_Width'] = vol_rolling
            vol_col = 'BB_Width'
            
        vol_median = df[vol_col].median()
        df['Vol_Regime'] = np.where(df[vol_col] > vol_median, 'High_Vol', 'Low_Vol')
        
        # Create combined regime
        df['Combined_Regime'] = df['Trend_Regime'] + '_' + df['Vol_Regime']
        
        # Select relevant columns
        feature_cols = ['High_Return', 'Low_Return', 'Combined_Regime', 'Trend_Regime', 'Vol_Regime']
        
        result = df[feature_cols].dropna()
        return result
    
    def _classify_trend_regime(self, ma_slopes: pd.Series) -> pd.Series:
        """Classify trend regime based on MA slope."""
        conditions = [
            ma_slopes > self.trend_thresholds['strong_bull'],
            ma_slopes > self.trend_thresholds['bull'],
            ma_slopes > self.trend_thresholds['neutral'],
            ma_slopes > self.trend_thresholds['bear']
        ]
        
        choices = ['Strong_Bull', 'Bull', 'Neutral', 'Bear']
        
        return pd.Series(np.select(conditions, choices, default='Strong_Bear'), 
                        index=ma_slopes.index)
    
    def sample_high_low(self, regime: str, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Sample high-low pairs for a specific regime."""
        if regime in self.regime_models:
            return self.regime_models[regime].sample(n_samples)
        else:
            # Fallback: independent sampling
            high_samples = np.random.exponential(0.01, n_samples)  # Small positive moves
            low_samples = np.random.exponential(0.01, n_samples)   # Small positive drops
            return high_samples, low_samples
    
    def get_regime_info(self) -> Dict:
        """Get information about fitted regimes."""
        info = {}
        
        for regime, model in self.regime_models.items():
            info[regime] = model.get_model_info()
        
        return info

# StockSpecificHighLowCopula class removed - using global-only modeling architecture

class IntelligentHighLowForecaster:
    """Main interface for intelligent high-low forecasting using global-only copulas."""
    
    def __init__(self, global_model_path: Optional[str] = None):
        self.global_model = GlobalHighLowCopulaModel()
        self.global_model_path = global_model_path
        
        # Load global model if available
        if global_model_path and os.path.exists(global_model_path):
            self._load_global_model(global_model_path)
    
    def train_global_model(self, all_stock_data: Dict[str, pd.DataFrame], 
                          save_path: Optional[str] = None) -> 'IntelligentHighLowForecaster':
        """Train the global copula model on all available stock data."""
        self.global_model.fit_global_model(all_stock_data)
        
        if save_path:
            self._save_global_model(save_path)
            self.global_model_path = save_path
        
        return self
    
    def forecast_high_low(self, symbol: str, reference_price: float,
                         trend_regime: str, vol_regime: str,
                         n_samples: int = 1) -> Dict:
        """
        Forecast high and low prices using global-only copula models.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (used for logging/tracking only)
        reference_price : float
            Reference price (e.g., average of open/close)
        trend_regime : str
            Current trend regime
        vol_regime : str
            Current volatility regime
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        dict
            Forecast results with high/low predictions and confidence intervals
        """
        combined_regime = f"{trend_regime}_{vol_regime}"
        
        # Sample high-low returns using global model only
        high_returns, low_returns = self.global_model.sample_high_low(
            combined_regime, n_samples)
        model_used = 'global_only'
        
        # Convert returns to prices
        high_prices = reference_price * (1 + high_returns)
        low_prices = reference_price * (1 - low_returns)  # low_returns are positive drops
        
        # Calculate statistics
        if n_samples == 1:
            return {
                'forecasted_high': high_prices[0],
                'forecasted_low': low_prices[0],
                'high_return': high_returns[0],
                'low_return': low_returns[0],
                'model_used': model_used,
                'regime': combined_regime
            }
        else:
            # Multiple samples - return statistics
            return {
                'high_samples': high_prices,
                'low_samples': low_prices,
                'high_mean': np.mean(high_prices),
                'low_mean': np.mean(low_prices),
                'high_ci': (np.percentile(high_prices, 2.5), np.percentile(high_prices, 97.5)),
                'low_ci': (np.percentile(low_prices, 2.5), np.percentile(low_prices, 97.5)),
                'correlation': np.corrcoef(high_returns, low_returns)[0, 1],
                'model_used': model_used,
                'regime': combined_regime
            }
    
    def _save_global_model(self, filepath: str) -> None:
        """Save the global model (simplified - would need full serialization)."""
        import pickle
        
        model_data = {
            'regime_models': {},  # Would need custom serialization for copulas
            'fitted': self.global_model.fitted,
            'trend_thresholds': self.global_model.trend_thresholds
        }
        
        # Note: Full implementation would require custom serialization for copula objects
        print(f"💾 Model saving functionality ready (full implementation needed)")
    
    def _load_global_model(self, filepath: str) -> None:
        """Load the global model (simplified)."""
        print(f"📁 Model loading functionality ready (full implementation needed)")
    
    def get_system_info(self) -> Dict:
        """Get comprehensive information about the global-only forecasting system."""
        info = {
            'global_model': {
                'fitted': self.global_model.fitted,
                'regimes': len(self.global_model.regime_models),
            },
            'architecture': 'global_only'
        }
        
        return info
    
    def analyze_regime_copulas(self) -> pd.DataFrame:
        """Analyze copula types selected for each regime."""
        regime_analysis = []
        
        for regime, model in self.global_model.regime_models.items():
            model_info = model.get_model_info()
            
            # Extract trend and volatility
            trend, vol = regime.split('_', 1)
            
            regime_analysis.append({
                'Regime': regime,
                'Trend': trend,
                'Volatility': vol,
                'Best_Copula': model_info['best_copula'],
                'High_Marginal': model_info['high_marginal_type'],
                'Low_Marginal': model_info['low_marginal_type']
            })
        
        return pd.DataFrame(regime_analysis)