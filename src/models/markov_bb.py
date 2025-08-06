import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
from scipy import stats
from sklearn.cluster import KMeans

class BollingerBandMarkovModel:
    """
    Markov Chain model for Bollinger Band position transitions.
    
    This model discretizes the Bollinger Band position space and models
    transitions between different regions as a Markov chain.
    """
    
    def __init__(self, n_states: int = 5):
        """
        Initialize the Markov model.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states for BB position
            Default 5: Extreme Low, Low, Middle, High, Extreme High
        """
        self.n_states = n_states
        self.state_labels = self._create_state_labels()
        self.transition_matrix = None
        self.state_boundaries = None
        self.fitted = False
    
    def sample_bb_position(self, state: int) -> float:
        """
        Sample a BB_Position from a normal distribution centered around the state mean.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")

        mean = self.state_means[state]
        std = self.state_stds[state]
        return np.random.normal(loc=mean, scale=std)

    def _create_state_labels(self) -> List[str]:
        """Create descriptive labels for states."""
        if self.n_states == 3:
            return ['Lower_Band', 'Middle', 'Upper_Band']
        elif self.n_states == 5:
            return ['Extreme_Low', 'Low', 'Middle', 'High', 'Extreme_High']
        elif self.n_states == 7:
            return ['Extreme_Low', 'Very_Low', 'Low', 'Middle', 'High', 'Very_High', 'Extreme_High']
        else:
            return [f'State_{i}' for i in range(self.n_states)]
    
    def _discretize_bb_position(self, bb_position: pd.Series) -> np.ndarray:
        """
        Discretize BB position into states.
        
        Parameters
        ----------
        bb_position : pd.Series
            Bollinger Band position (-1 to 1)
            
        Returns
        -------
        np.ndarray
            Array of state indices
        """
        # Define state boundaries based on quantiles or fixed ranges
        if self.state_boundaries is None:
            if self.n_states == 3:
                # Simple 3-state model: below BB, between BB, above BB
                boundaries = [-0.8, 0.8]
            elif self.n_states == 5:
                # 5-state model with extreme regions
                boundaries = [-0.9, -0.3, 0.3, 0.9]
            else:
                # Use quantiles for other numbers of states
                boundaries = [bb_position.quantile(i/self.n_states) for i in range(1, self.n_states)]
            
            self.state_boundaries = boundaries
        
        # Discretize
        states = np.digitize(bb_position, self.state_boundaries)
        
        # Ensure states are in valid range [0, n_states-1]
        states = np.clip(states, 0, self.n_states - 1)
        
        return states
    
    def fit(self, bb_position: pd.Series, 
            bb_width: Optional[pd.Series] = None, 
            global_prior: Optional[np.ndarray] = None, 
            alpha: float = 5.0) -> 'BollingerBandMarkovModel':
        """
        Fit the Markov model to Bollinger Band position data.
        
        Parameters
        ----------
        bb_position : pd.Series
            Bollinger Band position time series
        bb_width : pd.Series, optional
            Bollinger Band width (for regime-dependent modeling)
            
        Returns
        -------
        self
        """
        bb_position_clean = bb_position.dropna()
        
        # Ensure we have minimum data for state calculation  
        if len(bb_position_clean) < 2:
            raise ValueError(f"Insufficient data: need at least 2 observations, got {len(bb_position_clean)}")
        
        states = self._discretize_bb_position(bb_position_clean)

        self.transition_matrix = self._calculate_transition_matrix(states, global_prior, alpha)
        self.state_sequence = states

        # Calculate steady-state probabilities
        self.steady_state = self._calculate_steady_state()
        
        # Calculate state statistics
        self.state_stats = self._calculate_state_statistics(bb_position_clean, states)
        
        # If BB width provided, calculate regime-dependent statistics  
        if bb_width is not None:
            self.regime_stats = self._calculate_regime_statistics(bb_position_clean, bb_width.dropna(), states)
        
        self.fitted = True

        # Store per-state mean and std for sampling
        self.state_means = []
        self.state_stds = []

        for i in range(self.n_states):
            state_values = bb_position_clean[self.state_sequence == i]
            if len(state_values) > 1:
                self.state_means.append(state_values.mean())
                self.state_stds.append(state_values.std())
            else:
                # Fallback based on state boundaries
                if self.state_boundaries is not None and len(self.state_boundaries) >= self.n_states:
                    # Use state boundaries for fallback
                    if i == 0:
                        # First state: use boundary as upper bound
                        upper = self.state_boundaries[0]
                        lower = bb_position_clean.min()
                        midpoint = 0.5 * (lower + upper)
                    elif i == self.n_states - 1:
                        # Last state: use boundary as lower bound
                        lower = self.state_boundaries[-1]
                        upper = bb_position_clean.max()
                        midpoint = 0.5 * (lower + upper)
                    else:
                        # Middle states: use adjacent boundaries
                        lower = self.state_boundaries[i-1]
                        upper = self.state_boundaries[i]
                        midpoint = 0.5 * (lower + upper)
                    
                    self.state_means.append(midpoint)
                    self.state_stds.append(abs(upper - lower) / 4)  # Rough guess
                else:
                    # Ultimate fallback: distribute states evenly across data range
                    data_min = bb_position_clean.min()
                    data_max = bb_position_clean.max()
                    data_range = data_max - data_min
                    
                    # Place state i at position i/(n_states-1) across the range
                    if self.n_states > 1:
                        midpoint = data_min + (data_range * i / (self.n_states - 1))
                    else:
                        midpoint = (data_min + data_max) / 2
                    
                    self.state_means.append(midpoint)
                    self.state_stds.append(data_range / (4 * self.n_states))  # Conservative std

        return self
    
    def _calculate_transition_matrix(self, states: np.ndarray, 
                                 global_prior: Optional[np.ndarray] = None, 
                                 alpha: float = 5.0) -> np.ndarray:
        """Calculate the transition probability matrix with Dirichlet prior (Bayesian approach)."""
        transition_counts = np.zeros((self.n_states, self.n_states))

        # Count observed transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1

        # Apply Bayesian approach with Dirichlet prior
        if global_prior is not None:
            # Add prior knowledge weighted by alpha
            transition_counts += alpha * global_prior
        else:
            # Default uninformative prior (uniform)
            transition_counts += alpha / self.n_states

        # Normalize to probabilities (Bayesian posterior)
        transition_matrix = np.zeros_like(transition_counts)
        for i in range(self.n_states):
            row_sum = transition_counts[i, :].sum()
            if row_sum > 0:
                transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # Fallback to uniform distribution (should not happen with proper prior)
                transition_matrix[i, :] = 1.0 / self.n_states

        return transition_matrix
    
    def _calculate_steady_state(self) -> np.ndarray:
        """Calculate steady-state probabilities."""
        # Solve: π = πP where π is the steady state vector
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        
        # Find the eigenvector corresponding to eigenvalue 1
        steady_state_idx = np.argmin(np.abs(eigenvals - 1.0))
        steady_state = np.real(eigenvecs[:, steady_state_idx])
        
        # Normalize to probabilities
        steady_state = np.abs(steady_state) / np.sum(np.abs(steady_state))
        
        return steady_state
    
    def _calculate_state_statistics(self, bb_position: pd.Series, states: np.ndarray) -> Dict:
        """Calculate statistics for each state."""
        state_stats = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            if np.any(state_mask):
                state_positions = bb_position.iloc[state_mask]
                state_stats[self.state_labels[state]] = {
                    'count': len(state_positions),
                    'frequency': len(state_positions) / len(states),
                    'mean_position': state_positions.mean(),
                    'std_position': state_positions.std(),
                    'min_position': state_positions.min(),
                    'max_position': state_positions.max()
                }
            else:
                state_stats[self.state_labels[state]] = {
                    'count': 0,
                    'frequency': 0,
                    'mean_position': np.nan,
                    'std_position': np.nan,
                    'min_position': np.nan,
                    'max_position': np.nan
                }
        
        return state_stats
    
    def _calculate_regime_statistics(self, bb_position: pd.Series, bb_width: pd.Series, states: np.ndarray) -> Dict:
        """Calculate regime-dependent statistics."""
        # Define volatility regimes based on BB width
        width_median = bb_width.median()
        low_vol_mask = bb_width < width_median
        high_vol_mask = bb_width >= width_median
        
        regime_stats = {}
        
        for regime, mask in [('Low_Vol', low_vol_mask), ('High_Vol', high_vol_mask)]:
            regime_transitions = np.zeros((self.n_states, self.n_states))
            
            # Calculate transitions within this regime
            regime_indices = np.where(mask)[0]
            for i in range(len(regime_indices) - 1):
                if regime_indices[i+1] == regime_indices[i] + 1:  # Consecutive days
                    current_state = states[regime_indices[i]]
                    next_state = states[regime_indices[i+1]]
                    regime_transitions[current_state, next_state] += 1
            
            # Normalize to probabilities
            regime_matrix = np.zeros_like(regime_transitions)
            for i in range(self.n_states):
                row_sum = regime_transitions[i, :].sum()
                if row_sum > 0:
                    regime_matrix[i, :] = regime_transitions[i, :] / row_sum
            
            regime_stats[regime] = {
                'transition_matrix': regime_matrix,
                'mean_bb_width': bb_width[mask].mean(),
                'state_frequencies': [(states[mask] == i).mean() for i in range(self.n_states)]
            }
        
        return regime_stats
    
    def predict_next_state_probs(self, current_state: int, n_steps: int = 1) -> np.ndarray:
        """
        Predict probability distribution of states after n steps.
        
        Parameters
        ----------
        current_state : int
            Current state index
        n_steps : int
            Number of steps ahead to predict
            
        Returns
        -------
        np.ndarray
            Probability distribution over states
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Start with current state
        state_prob = np.zeros(self.n_states)
        state_prob[current_state] = 1.0
        
        # Apply transition matrix n_steps times
        for _ in range(n_steps):
            state_prob = state_prob @ self.transition_matrix
        
        return state_prob
    
    def simulate_path(self, current_state: int, n_steps: int = 20, n_simulations: int = 1000) -> Dict:
        """
        Simulate future paths using Monte Carlo.
        
        Parameters
        ----------
        current_state : int
            Starting state
        n_steps : int
            Number of steps to simulate
        n_simulations : int
            Number of simulation paths
            
        Returns
        -------
        dict
            Simulation results including paths and statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        paths = np.zeros((n_simulations, n_steps))
        
        for sim in range(n_simulations):
            path = [current_state]
            current = current_state
            
            for step in range(n_steps - 1):
                # Get transition probabilities from current state
                probs = self.transition_matrix[current, :]
                
                # Sample next state
                next_state = np.random.choice(self.n_states, p=probs)
                path.append(next_state)
                current = next_state
            
            paths[sim, :] = path
        
        # Calculate statistics
        state_percentiles = {}
        for step in range(n_steps):
            step_states = paths[:, step]
            state_percentiles[step] = {
                'mean': np.mean(step_states),
                'std': np.std(step_states),
                'percentiles': np.percentile(step_states, [10, 25, 50, 75, 90]),
                'state_probs': [(step_states == i).mean() for i in range(self.n_states)]
            }
        
        return {
            'paths': paths,
            'state_percentiles': state_percentiles,
            'final_state_distribution': [(paths[:, -1] == i).mean() for i in range(self.n_states)]
        }
    
    def get_state_from_bb_position(self, bb_position: float) -> int:
        """Convert BB position to state index."""
        if self.state_boundaries is None:
            raise ValueError("Model must be fitted first")
        
        state = np.digitize([bb_position], self.state_boundaries)[0]
        return np.clip(state, 0, self.n_states - 1)
    
    def get_bb_position_from_state(self, state: int) -> Tuple[float, float]:
        """Get BB position range for a given state."""
        if self.state_boundaries is None:
            raise ValueError("Model must be fitted first")
        
        if state == 0:
            return -1.0, self.state_boundaries[0]
        elif state == self.n_states - 1:
            return self.state_boundaries[-1], 1.0
        else:
            return self.state_boundaries[state-1], self.state_boundaries[state]
    
    def plot_transition_matrix(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot the transition matrix as a heatmap."""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.transition_matrix, 
                    annot=True, 
                    fmt='.3f',
                    xticklabels=self.state_labels,
                    yticklabels=self.state_labels,
                    cmap='Blues',
                    cbar_kws={'label': 'Transition Probability'})
        
        plt.title('Bollinger Band State Transition Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Next State', fontsize=12)
        plt.ylabel('Current State', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_steady_state(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot steady-state probabilities."""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=figsize)
        bars = plt.bar(self.state_labels, self.steady_state, alpha=0.7, color='skyblue')
        
        # Add value labels on bars
        for bar, prob in zip(bars, self.steady_state):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Steady-State Probabilities', fontsize=14, fontweight='bold')
        plt.xlabel('Bollinger Band State', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_simulation_results(self, simulation_results: Dict, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot Monte Carlo simulation results."""
        paths = simulation_results['paths']
        state_percentiles = simulation_results['state_percentiles']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        
        # Plot sample paths
        n_sample_paths = min(50, len(paths))
        sample_indices = np.random.choice(len(paths), n_sample_paths, replace=False)
        
        for i in sample_indices:
            ax1.plot(paths[i, :], alpha=0.1, color='blue')
        
        # Plot mean path
        mean_path = [state_percentiles[i]['mean'] for i in range(len(state_percentiles))]
        ax1.plot(mean_path, color='red', linewidth=3, label='Mean Path')
        
        ax1.set_title('Simulated State Paths', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence bands
        steps = list(range(len(state_percentiles)))
        percentiles_10 = [state_percentiles[i]['percentiles'][0] for i in steps]
        percentiles_90 = [state_percentiles[i]['percentiles'][4] for i in steps]
        percentiles_25 = [state_percentiles[i]['percentiles'][1] for i in steps]
        percentiles_75 = [state_percentiles[i]['percentiles'][3] for i in steps]
        
        ax2.fill_between(steps, percentiles_10, percentiles_90, alpha=0.3, color='blue', label='10-90th percentile')
        ax2.fill_between(steps, percentiles_25, percentiles_75, alpha=0.5, color='blue', label='25-75th percentile')
        ax2.plot(steps, mean_path, color='red', linewidth=2, label='Mean')
        
        ax2.set_title('State Evolution Confidence Bands', fontsize=12, fontweight='bold')
        ax2.set_ylabel('State Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot final state distribution
        final_dist = simulation_results['final_state_distribution']
        bars = ax3.bar(self.state_labels, final_dist, alpha=0.7, color='green')
        
        for bar, prob in zip(bars, final_dist):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Final State Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('State')
        ax3.set_ylabel('Probability')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        summary = {
            'model_info': {
                'n_states': self.n_states,
                'state_labels': self.state_labels,
                'state_boundaries': self.state_boundaries
            },
            'transition_matrix': self.transition_matrix,
            'steady_state': self.steady_state,
            'state_statistics': self.state_stats
        }
        
        if hasattr(self, 'regime_stats'):
            summary['regime_statistics'] = self.regime_stats
        
        return summary

class MultiStockBBMarkovModel:
    """
    Multi-stock Bollinger Band Markov model with Bayesian priors.
    
    Features:
    - Separate global priors for each trend state (strong_up, up, ranging, down, strong_down)
    - Stock-specific models that use trend-specific global priors
    - Bayesian updating when stock-specific data is available
    """
    
    def __init__(self, n_states: int = 5, 
                 parabolic_up_thresh: float = 0.05,
                 strong_up_thresh: float = 0.025,
                 up_thresh: float = 0.008,
                 down_thresh: float = -0.008,
                 strong_down_thresh: float = -0.025,
                 parabolic_down_thresh: float = -0.05):
        self.n_states = n_states
        self.parabolic_up_thresh = parabolic_up_thresh
        self.strong_up_thresh = strong_up_thresh
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.strong_down_thresh = strong_down_thresh
        self.parabolic_down_thresh = parabolic_down_thresh
        
        # Trend names for consistent reference (7 trend states)
        self.trend_names = ['parabolic_up', 'strong_up', 'up', 'ranging', 'down', 'strong_down', 'parabolic_down']
        
        # Global priors for each trend state
        self.global_trend_priors = {}
        self.global_trend_models = {}
        
        # Initialize global models for each trend
        for trend in self.trend_names:
            self.global_trend_models[trend] = BollingerBandMarkovModel(n_states)
        
        # Create a unified global model for fallback (combines all trends)
        self.global_model = BollingerBandMarkovModel(n_states)
        
        # Stock-specific models
        self.stock_models = {}
        
        # Fitted status
        self.fitted = False
        
    def classify_trend(self, ma_series: pd.Series) -> pd.Series:
        """
        Enhanced trend classification with 7 states:
        Parabolic_up, Strong_up, Up, Ranging, Down, Strong_down, Parabolic_down
        """
        # Calculate slope of moving average
        slope = ma_series.pct_change(periods=5)  # 5-day slope
        
        trend = pd.Series(index=ma_series.index, dtype='object')
        
        # Classify into 7 trend states based on thresholds
        trend[slope > self.parabolic_up_thresh] = 'parabolic_up'
        trend[(slope > self.strong_up_thresh) & (slope <= self.parabolic_up_thresh)] = 'strong_up'
        trend[(slope > self.up_thresh) & (slope <= self.strong_up_thresh)] = 'up'
        trend[(slope >= self.down_thresh) & (slope <= self.up_thresh)] = 'ranging'
        trend[(slope >= self.strong_down_thresh) & (slope < self.down_thresh)] = 'down'
        trend[(slope >= self.parabolic_down_thresh) & (slope < self.strong_down_thresh)] = 'strong_down'
        trend[slope < self.parabolic_down_thresh] = 'parabolic_down'
        
        return trend.ffill()
    
    def fit_global_prior(self, all_stock_data: dict, alpha_global: float = 1.0):
        """
        Fit separate global priors for each trend state using ALL stocks.
        
        Parameters:
        -----------
        all_stock_data : dict
            Dictionary with stock symbols as keys and DataFrames with 
            'BB_Position' and 'MA' columns as values
        alpha_global : float
            Concentration parameter for global prior
        """
        print("🌍 Learning TREND-SPECIFIC global priors from all stocks...")
        
        # Collect data for each trend across all stocks
        trend_data_combined = {trend: [] for trend in self.trend_names}
        total_observations = 0
        
        for symbol, stock_data in all_stock_data.items():
            if 'BB_Position' not in stock_data.columns or 'MA' not in stock_data.columns:
                print(f"  ⚠️ Skipping {symbol}: missing required columns")
                continue
            
            # Add trend classification
            stock_data_with_trend = stock_data.copy()
            stock_data_with_trend['Trend'] = self.classify_trend(stock_data_with_trend['MA'])
            
            # Collect BB positions for each trend
            for trend in self.trend_names:
                trend_subset = stock_data_with_trend[stock_data_with_trend['Trend'] == trend]
                if len(trend_subset) > 0:
                    bb_positions = trend_subset['BB_Position'].dropna()
                    if len(bb_positions) > 0:
                        trend_data_combined[trend].append(bb_positions)
            
            total_observations += len(stock_data_with_trend)
            print(f"  📊 Processed {symbol}: {len(stock_data_with_trend)} observations")
        
        # Fit global prior for each trend
        print(f"\n🎯 Fitting trend-specific global priors...")
        trend_obs_summary = {}
        
        for trend in self.trend_names:
            if trend_data_combined[trend]:
                # Combine all stocks' data for this trend
                combined_bb_positions = pd.concat(trend_data_combined[trend], ignore_index=True)
                n_obs = len(combined_bb_positions)
                
                if n_obs >= 10:  # Minimum observations for reliable prior
                    try:
                        # Fit global model for this trend
                        self.global_trend_models[trend].fit(combined_bb_positions, alpha=alpha_global)
                        self.global_trend_priors[trend] = self.global_trend_models[trend].transition_matrix
                        
                        trend_obs_summary[trend] = n_obs
                        print(f"  ✅ {trend}: {n_obs} global observations → prior fitted")
                    except Exception as e:
                        print(f"  ❌ {trend}: {n_obs} observations → failed ({e})")
                        # Create default uniform prior
                        uniform_prior = np.ones((self.n_states, self.n_states)) / self.n_states
                        self.global_trend_priors[trend] = uniform_prior
                        trend_obs_summary[trend] = f"{n_obs} (uniform prior)"
                else:
                    print(f"  ⚠️ {trend}: {n_obs} observations → insufficient data, using uniform prior")
                    # Create default uniform prior
                    uniform_prior = np.ones((self.n_states, self.n_states)) / self.n_states
                    self.global_trend_priors[trend] = uniform_prior
                    trend_obs_summary[trend] = f"{n_obs} (uniform prior)"
            else:
                print(f"  ⚠️ {trend}: 0 observations → using uniform prior")
                uniform_prior = np.ones((self.n_states, self.n_states)) / self.n_states
                self.global_trend_priors[trend] = uniform_prior
                trend_obs_summary[trend] = "0 (uniform prior)"
        
        # Add sparse bucket logging - find 5 sparsest BB-position + trend buckets
        print(f"\n📊 Analyzing BB-position + trend bucket sparsity...")
        bucket_counts = []
        for trend in self.trend_names:
            if trend_data_combined[trend]:
                combined_bb_positions = pd.concat(trend_data_combined[trend], ignore_index=True)
                # Count positions for each BB state within this trend
                bb_counts = combined_bb_positions.value_counts()
                for bb_pos, count in bb_counts.items():
                    bucket_counts.append((f"{trend}_BB{bb_pos}", count))
            else:
                # Add empty trend buckets
                for bb_pos in range(1, self.n_states + 1):
                    bucket_counts.append((f"{trend}_BB{bb_pos}", 0))
        
        # Sort by count and show 5 sparsest buckets
        bucket_counts.sort(key=lambda x: x[1])
        sparse_buckets = bucket_counts[:5]
        print(f"🔍 5 sparsest BB-position + trend buckets:")
        for bucket, count in sparse_buckets:
            print(f"  📉 {bucket}: {count} samples")
        
        # Also fit unified global model (for fallback)
        all_bb_positions = []
        for trend_data_list in trend_data_combined.values():
            all_bb_positions.extend(trend_data_list)
        
        if all_bb_positions:
            unified_bb_positions = pd.concat(all_bb_positions, ignore_index=True)
            self.global_model.fit(unified_bb_positions, alpha=alpha_global)
            print(f"  🌍 Unified global model: {len(unified_bb_positions)} total observations")
        
        print(f"\n✅ Trend-specific global priors learned!")
        print(f"📊 Prior Summary:")
        for trend, obs_info in trend_obs_summary.items():
            print(f"  🎯 {trend}: {obs_info}")
        
        return self.global_trend_priors
    
    def fit_stock_models(self, all_stock_data: dict, alpha_stock: float = 5.0):
        """
        Fit individual stock models using trend-specific global priors.
        
        Parameters:
        -----------
        all_stock_data : dict
            Dictionary with stock symbols as keys and DataFrames as values
        alpha_stock : float
            Weight given to global prior vs. stock-specific data
        """
        if not self.global_trend_priors:
            raise ValueError("Must fit global priors first using fit_global_prior()")
        
        print(f"🏢 Fitting individual stock models with trend-specific priors (α={alpha_stock})...")
        
        for symbol, stock_data in all_stock_data.items():
            if 'BB_Position' not in stock_data.columns or 'MA' not in stock_data.columns:
                print(f"⚠️  Skipping {symbol}: missing required columns")
                continue
            
            print(f"🔄 Fitting model for {symbol}...")
            
            # Add trend classification
            stock_data_with_trend = stock_data.copy()
            stock_data_with_trend['Trend'] = self.classify_trend(stock_data_with_trend['MA'])
            
            # Create trend-specific models for this stock
            stock_trend_models = {}
            for trend in self.trend_names:
                stock_trend_models[trend] = BollingerBandMarkovModel(self.n_states)
            
            # Fit each trend model with its specific global prior
            fitted_trends = []
            skipped_trends = []
            
            for trend_name in self.trend_names:
                trend_data = stock_data_with_trend[stock_data_with_trend['Trend'] == trend_name]
                
                if len(trend_data) > 0:
                    bb_positions = trend_data['BB_Position'].dropna()
                    
                    if len(bb_positions) >= 10:  # Sufficient data for Bayesian updating
                        try:
                            # Use trend-specific global prior for Bayesian updating
                            trend_global_prior = self.global_trend_priors[trend_name]
                            stock_trend_models[trend_name].fit(
                                bb_positions, 
                                global_prior=trend_global_prior, 
                                alpha=alpha_stock
                            )
                            fitted_trends.append(trend_name)
                            print(f"  ✅ {trend_name}: {len(bb_positions)} observations (Bayesian update)")
                        except Exception as e:
                            skipped_trends.append((trend_name, len(bb_positions)))
                            print(f"  ❌ {trend_name}: {len(bb_positions)} observations (failed: {e})")
                            continue
                    elif len(bb_positions) > 0:
                        # Insufficient data - use pure global prior for this trend
                        try:
                            trend_global_prior = self.global_trend_priors[trend_name]
                            # Create a model that just uses the global prior
                            stock_trend_models[trend_name].transition_matrix = trend_global_prior.copy()
                            stock_trend_models[trend_name].state_labels = self.global_trend_models[trend_name].state_labels
                            stock_trend_models[trend_name].state_boundaries = self.global_trend_models[trend_name].state_boundaries
                            stock_trend_models[trend_name].fitted = True
                            
                            # Copy other necessary attributes
                            if hasattr(self.global_trend_models[trend_name], 'steady_state'):
                                stock_trend_models[trend_name].steady_state = self.global_trend_models[trend_name].steady_state
                            if hasattr(self.global_trend_models[trend_name], 'state_stats'):
                                stock_trend_models[trend_name].state_stats = self.global_trend_models[trend_name].state_stats
                            
                            fitted_trends.append(trend_name)
                            print(f"  🌍 {trend_name}: {len(bb_positions)} observations (using global prior)")
                        except Exception as e:
                            skipped_trends.append((trend_name, len(bb_positions)))
                            print(f"  ❌ {trend_name}: {len(bb_positions)} observations (global prior failed: {e})")
                            continue
                    else:
                        skipped_trends.append((trend_name, 0))
                        print(f"  ⚠️  {trend_name}: 0 observations (skipped - no data)")
                else:
                    skipped_trends.append((trend_name, 0))
                    print(f"  ⚠️  {trend_name}: 0 observations (skipped - no data)")
            
            self.stock_models[symbol] = {
                'models': stock_trend_models,
                'data': stock_data_with_trend,
                'fitted_trends': fitted_trends,
                'skipped_trends': skipped_trends
            }
            
            success_msg = f"✅ {symbol} model complete: {len(fitted_trends)} trend states fitted"
            if skipped_trends:
                skip_msg = f", {len(skipped_trends)} skipped"
                success_msg += skip_msg
            print(success_msg)
        
        self.fitted = True
        print(f"🎉 All stock models fitted! Total stocks: {len(self.stock_models)}")
    
    def get_stock_model(self, symbol: str, trend: str = None) -> BollingerBandMarkovModel:
        """Get the appropriate model for a stock and trend."""
        if symbol not in self.stock_models:
            # Fallback to trend-specific global model if trend specified
            if trend and trend in self.global_trend_models:
                return self.global_trend_models[trend]
            # Otherwise fallback to unified global model
            return self.global_model
        
        if trend is None:
            # Return the unified global model if no trend specified
            return self.global_model
        
        stock_info = self.stock_models[symbol]
        if trend in stock_info['models'] and stock_info['models'][trend].fitted:
            # Return stock-specific model for this trend
            return stock_info['models'][trend]
        else:
            # Fallback to trend-specific global model
            if trend in self.global_trend_models and self.global_trend_models[trend].fitted:
                return self.global_trend_models[trend]
            # Final fallback to unified global model
            return self.global_model
    
    def detect_current_trend(self, symbol: str) -> str:
        """Detect current trend for a specific stock."""
        if symbol not in self.stock_models:
            return 'ranging'  # Default
        
        stock_data = self.stock_models[symbol]['data']
        current_trend = stock_data['Trend'].iloc[-1]
        return current_trend
    
    def get_stock_forecast(self, symbol: str, forecast_days: int = 5, n_simulations: int = 1000):
        """
        Generate forecast for a specific stock using its trend-aware model.
        """
        if symbol not in self.stock_models:
            raise ValueError(f"No model available for {symbol}")
        
        stock_info = self.stock_models[symbol]
        stock_data = stock_info['data']
        
        # Get current state
        current_bb_pos = stock_data['BB_Position'].iloc[-1]
        current_trend = stock_data['Trend'].iloc[-1]
        
        # Get appropriate model
        model = self.get_stock_model(symbol, current_trend)
        current_state = model.get_state_from_bb_position(current_bb_pos)
        
        # Generate forecasts
        state_probs = model.predict_next_state_probs(current_state, forecast_days)
        simulation_results = model.simulate_path(current_state, forecast_days, n_simulations)
        
        return {
            'symbol': symbol,
            'current_trend': current_trend,
            'current_state': current_state,
            'current_bb_position': current_bb_pos,
            'state_probabilities': state_probs,
            'simulation_results': simulation_results,
            'model': model
        }
    
    def get_model_summary(self):
        """Get comprehensive summary of the multi-stock model."""
        summary = {
            'global_models': {
                'unified_global': {
                    'fitted': self.global_model.fitted,
                    'n_states': self.n_states,
                    'transition_matrix': self.global_model.transition_matrix if self.global_model.fitted else None
                },
                'trend_specific_globals': {}
            },
            'stock_models': {}
        }
        
        # Add trend-specific global model info
        for trend in self.trend_names:
            if trend in self.global_trend_models:
                summary['global_models']['trend_specific_globals'][trend] = {
                    'fitted': self.global_trend_models[trend].fitted,
                    'transition_matrix': self.global_trend_priors.get(trend, None)
                }
        
        for symbol, stock_info in self.stock_models.items():
            summary['stock_models'][symbol] = {
                'fitted_trends': stock_info['fitted_trends'],
                'skipped_trends': stock_info.get('skipped_trends', []),
                'total_observations': len(stock_info['data']),
                'current_trend': stock_info['data']['Trend'].iloc[-1] if len(stock_info['data']) > 0 else None
            }
        
        return summary
    
    def plot_multi_stock_comparison(self, symbols: list = None, figsize: tuple = (25, 15)):
        """
        Plot comparison of transition matrices across multiple stocks.
        """
        if symbols is None:
            symbols = list(self.stock_models.keys())[:4]  # Limit to 4 stocks for readability with 7 trends
        
        n_symbols = len(symbols)
        n_trends = 7  # parabolic_up, strong_up, up, ranging, down, strong_down, parabolic_down
        
        fig, axes = plt.subplots(n_symbols, n_trends, figsize=figsize)
        
        if n_symbols == 1:
            axes = axes.reshape(1, -1)
        
        trend_names = ['parabolic_up', 'strong_up', 'up', 'ranging', 'down', 'strong_down', 'parabolic_down']
        
        for i, symbol in enumerate(symbols):
            if symbol not in self.stock_models:
                continue
                
            stock_info = self.stock_models[symbol]
            
            for j, trend in enumerate(trend_names):
                ax = axes[i, j]
                
                if trend in stock_info['models'] and stock_info['models'][trend].fitted:
                    model = stock_info['models'][trend]
                    sns.heatmap(model.transition_matrix, 
                              annot=True, fmt='.2f', 
                              xticklabels=model.state_labels,
                              yticklabels=model.state_labels,
                              cmap='Blues', ax=ax, cbar=False)
                    ax.set_title(f'{symbol} - {trend.title()}', fontsize=10)
                else:
                    # Show trend-specific global prior if trend model not fitted
                    if trend in self.global_trend_priors:
                        global_prior_matrix = self.global_trend_priors[trend]
                        global_state_labels = self.global_trend_models[trend].state_labels
                    else:
                        # Fallback to unified global prior
                        global_prior_matrix = self.global_model.transition_matrix if self.global_model.fitted else np.ones((self.n_states, self.n_states)) / self.n_states
                        global_state_labels = self.global_model.state_labels
                    
                    sns.heatmap(global_prior_matrix, 
                              annot=True, fmt='.2f',
                              xticklabels=global_state_labels,
                              yticklabels=global_state_labels,
                              cmap='Greys', ax=ax, cbar=False, alpha=0.5)
                    ax.set_title(f'{symbol} - {trend.title()} (Global Prior)', fontsize=10)
                
                if i == n_symbols - 1:
                    ax.set_xlabel('Next State')
                if j == 0:
                    ax.set_ylabel('Current State')
        
        plt.tight_layout()
        plt.suptitle('Multi-Stock Trend-Specific Transition Matrices', fontsize=16, y=1.02)
        plt.show()
    
    def plot_trend_distribution(self, figsize: tuple = (15, 10)):
        """
        Plot the distribution of trends across all stocks.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Collect trend data from all stocks
        all_trends = []
        trend_by_stock = {}
        
        for symbol, stock_info in self.stock_models.items():
            stock_trends = stock_info['data']['Trend'].value_counts()
            trend_by_stock[symbol] = stock_trends
            all_trends.extend(stock_info['data']['Trend'].tolist())
        
        # Overall trend distribution
        trend_counts = pd.Series(all_trends).value_counts()
        trend_counts.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
        ax1.set_title('Overall Trend Distribution Across All Stocks', fontweight='bold')
        ax1.set_xlabel('Trend State')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Trend distribution by stock (stacked bar)
        trend_df = pd.DataFrame(trend_by_stock).fillna(0).T
        trend_df.plot(kind='bar', stacked=True, ax=ax2, 
                     color=['darkred', 'red', 'orange', 'gray', 'lightblue', 'green', 'darkgreen'])
        ax2.set_title('Trend Distribution by Stock', fontweight='bold')
        ax2.set_xlabel('Stock Symbol')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Trend', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()


class TrendAwareBBMarkovWrapper:
    def __init__(self, n_states: int = 5, slope_window: int = 5,
                 up_thresh: float = 0.05, down_thresh: float = -0.05):
        self.models = {
            'up': BollingerBandMarkovModel(n_states),
            'down': BollingerBandMarkovModel(n_states),
            'sideways': BollingerBandMarkovModel(n_states),
        }
        self.global_model = BollingerBandMarkovModel(n_states)
        self.fitted = False
        self.slope_window = slope_window
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh

    def sample_bb_position(self, state: int, trend: str) -> float:
        """
        Sample a BB_Position value from the given state and trend.
        Falls back to global model if trend model not available.
        """
        model = self.models.get(trend, self.global_model)
        return model.sample_bb_position(state)
    
    def classify_trend(self, ma_series: pd.Series) -> pd.Series:
        """
        Classify trend as 'up', 'down', or 'sideways' based on slope of MA.
        """
        slope = ma_series.diff(self.slope_window)
        trend = pd.Series(index=ma_series.index, dtype='object')
        trend[slope > self.up_thresh] = 'up'
        trend[slope < self.down_thresh] = 'down'
        trend[(slope >= self.down_thresh) & (slope <= self.up_thresh)] = 'sideways'
        return trend.ffill()
    
    
    def fit(self, bb_data: pd.DataFrame, alpha: float = 5.0):
        """
        Fit the wrapper using BB data with columns: ['BB_Position', 'BB_Width', 'MA']
        """
        if not {'BB_Position', 'MA'}.issubset(bb_data.columns):
            raise ValueError("Input DataFrame must include 'BB_Position' and 'MA' columns")

        # Assign trend to each row
        bb_data = bb_data.copy()
        bb_data['Trend'] = self.classify_trend(bb_data['MA'])

        # Fit global model
        self.global_model.fit(bb_data['BB_Position'])
        global_prior = self.global_model.transition_matrix
        # Group data by trend and fit each model
        for trend, group in bb_data.groupby('Trend'):
            if trend not in self.models:
                continue
            pos = group['BB_Position']
            
            # Fit using Bayesian approach with Dirichlet prior - works with any amount of data
            print(f"🔄 Fitting {trend} trend model with {len(pos.dropna())} observations (Bayesian approach)")
            self.models[trend].fit(pos, global_prior=global_prior, alpha=alpha)
            print(f"✅ Fitted {trend} trend model successfully")

        self.fitted = True

    def detect_trend(self, ma_series: pd.Series) -> str:
        """
        Detect current trend from MA series (use latest slope).
        """
        slope = ma_series.diff(self.slope_window).iloc[-1]
        if slope > self.up_thresh:
            return 'up'
        elif slope < self.down_thresh:
            return 'down'
        else:
            return 'sideways'

    def get_model(self, trend: str) -> BollingerBandMarkovModel:
        if not self.fitted:
            raise ValueError("Must call .fit() first")
        return self.models.get(trend, self.global_model)
    
    def get_state(self, bb_position: float, trend: str = None) -> int:
        """Get the current state for a given BB position."""
        if trend is None:
            model = self.global_model
        else:
            model = self.models.get(trend, self.global_model)
        
        if not model.fitted:
            # Fallback to global model if trend-specific model isn't fitted
            model = self.global_model
            if not model.fitted:
                raise ValueError("No fitted model available")
            
        # Use the model's discretization method
        return model._discretize_bb_position(pd.Series([bb_position]))[0]
    
    def sample_states(self, n: int, current_state: int, trend: str) -> np.ndarray:
        """Sample future states using the trend-specific model."""
        model = self.models.get(trend, self.global_model)
        
        if not model.fitted:
            # Fallback to global model if trend-specific model isn't fitted
            model = self.global_model
            if not model.fitted:
                raise ValueError("No fitted model available")
        
        # Simple state sampling using transition matrix
        states = [current_state]
        for _ in range(n - 1):
            current = states[-1]
            probs = model.transition_matrix[current, :]
            next_state = np.random.choice(model.n_states, p=probs)
            states.append(next_state)
        
        return np.array(states)