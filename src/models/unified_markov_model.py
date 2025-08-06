"""
Unified Markov Model using Centralized Regime Configuration

This model demonstrates how Markov models should integrate with the centralized
regime configuration system. It can model transitions between:
- Trend regimes (strong_bull, bull, sideways, bear, strong_bear)
- Volatility regimes (low, medium, high)  
- Combined regimes (trend_volatility combinations)
- Bollinger Band states

Uses the centralized regime configuration from config.regime_config.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from scipy import stats
import sys
import os

# Import centralized regime configuration and classifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.regime_config import REGIME_CONFIG, get_all_combined_regimes
from models.regime_classifier import REGIME_CLASSIFIER, classify_stock_regimes


class UnifiedMarkovModel:
    """
    Markov Chain model that uses centralized regime configuration.
    
    Can model transitions between:
    - Trend regimes only
    - Volatility regimes only  
    - Combined trend-volatility regimes
    - Custom states (e.g., BB positions)
    """
    
    def __init__(self, 
                 model_type: str = 'combined',
                 custom_states: Optional[List[str]] = None):
        """
        Initialize the unified Markov model.
        
        Parameters
        ----------
        model_type : str
            Type of regime modeling:
            - 'trend': Use only trend regimes from config
            - 'volatility': Use only volatility regimes from config
            - 'combined': Use combined trend_volatility regimes from config
            - 'custom': Use custom states provided
        custom_states : List[str], optional
            Custom state names (required if model_type='custom')
        """
        self.model_type = model_type
        
        # Set up states based on model type
        if model_type == 'trend':
            self.states = REGIME_CONFIG.trend.get_all_labels()
        elif model_type == 'volatility':
            self.states = REGIME_CONFIG.volatility.get_all_labels()
        elif model_type == 'combined':
            self.states = get_all_combined_regimes()
        elif model_type == 'custom':
            if custom_states is None:
                raise ValueError("custom_states must be provided when model_type='custom'")
            self.states = custom_states.copy()
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        self.n_states = len(self.states)
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}
        
        # Model components
        self.transition_matrix = None
        self.stationary_distribution = None
        self.state_stats = {}  # Statistics for each state
        self.fitted = False
        
        print(f"Initialized {model_type} Markov model with {self.n_states} states:")
        print(f"States: {self.states}")
    
    def fit(self, stock_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> 'UnifiedMarkovModel':
        """
        Fit the Markov model on stock data.
        
        Parameters
        ----------
        stock_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Stock data with OHLC and technical indicators, or dictionary of such DataFrames
        """
        print(f"🔧 Fitting {self.model_type} Markov model...")
        
        # Convert single DataFrame to dict format
        if isinstance(stock_data, pd.DataFrame):
            stock_data = {'STOCK': stock_data}
        
        # Collect regime sequences from all stocks
        all_regime_sequences = []
        total_observations = 0
        
        for symbol, df in stock_data.items():
            regime_sequence = self._extract_regime_sequence(df, symbol)
            if len(regime_sequence) > 0:
                all_regime_sequences.append(regime_sequence)
                total_observations += len(regime_sequence)
        
        if not all_regime_sequences:
            raise ValueError("No valid regime sequences found in data")
        
        print(f"  📊 Collected {total_observations} regime observations from {len(all_regime_sequences)} stocks")
        
        # Combine all sequences
        combined_sequence = pd.concat(all_regime_sequences, ignore_index=True)
        
        # Estimate transition matrix
        self._estimate_transition_matrix(combined_sequence)
        
        # Calculate state statistics
        self._calculate_state_statistics(combined_sequence)
        
        # Calculate stationary distribution
        self._calculate_stationary_distribution()
        
        self.fitted = True
        print(f"✅ Markov model fitting complete!")
        
        return self
    
    def _extract_regime_sequence(self, stock_df: pd.DataFrame, symbol: str) -> pd.Series:
        """Extract regime sequence from stock data based on model type."""
        
        if self.model_type == 'custom':
            # For custom models, expect the regime column to already exist
            if 'regime' in stock_df.columns:
                return stock_df['regime'].dropna()
            else:
                print(f"  ⚠️ Warning: No 'regime' column found for {symbol}")
                return pd.Series(dtype=object)
        
        # For trend/volatility/combined models, classify regimes using centralized classifier
        try:
            df_with_regimes = classify_stock_regimes(stock_df)
            
            if self.model_type == 'trend':
                return df_with_regimes['trend_regime'].dropna()
            elif self.model_type == 'volatility':
                return df_with_regimes['vol_regime'].dropna() 
            elif self.model_type == 'combined':
                return df_with_regimes['combined_regime'].dropna()
                
        except Exception as e:
            print(f"  ⚠️ Error processing {symbol}: {str(e)[:50]}")
            return pd.Series(dtype=object)
    
    def _estimate_transition_matrix(self, regime_sequence: pd.Series):
        """Estimate transition matrix from regime sequence."""
        
        # Initialize transition count matrix
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(regime_sequence) - 1):
            current_state = regime_sequence.iloc[i]
            next_state = regime_sequence.iloc[i + 1]
            
            if current_state in self.state_to_idx and next_state in self.state_to_idx:
                current_idx = self.state_to_idx[current_state]
                next_idx = self.state_to_idx[next_state]
                transition_counts[current_idx, next_idx] += 1
        
        # Convert counts to probabilities
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = transition_counts[i, :].sum()
            if row_sum > 0:
                self.transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # If no transitions from this state, assume uniform distribution
                self.transition_matrix[i, :] = 1.0 / self.n_states
        
        print(f"  📈 Estimated transition matrix from {len(regime_sequence)} observations")
    
    def _calculate_state_statistics(self, regime_sequence: pd.Series):
        """Calculate statistics for each state."""
        
        state_counts = regime_sequence.value_counts()
        total_count = len(regime_sequence)
        
        for state in self.states:
            count = state_counts.get(state, 0)
            self.state_stats[state] = {
                'count': count,
                'frequency': count / total_count if total_count > 0 else 0,
                'transitions_in': 0,  # Will be calculated from transition matrix
                'transitions_out': 0  # Will be calculated from transition matrix
            }
        
        # Calculate transition statistics
        for i, state in enumerate(self.states):
            transitions_out = self.transition_matrix[i, :].sum()
            transitions_in = self.transition_matrix[:, i].sum()
            
            self.state_stats[state]['transitions_out'] = transitions_out
            self.state_stats[state]['transitions_in'] = transitions_in
        
        # Add sparse bucket diagnostics
        self._log_sparse_buckets()
    
    def _log_sparse_buckets(self):
        """Log the 5 least populated (trend, BB-position) buckets for diagnostic purposes."""
        
        # Only log for combined or custom models that might have trend-BB combinations
        if self.model_type not in ['combined', 'custom']:
            return
        
        # Create list of (state, count) tuples
        state_counts = [(state, stats['count']) for state, stats in self.state_stats.items()]
        
        # Sort by count (ascending) to get sparsest buckets first
        state_counts.sort(key=lambda x: x[1])
        
        # Get the 5 sparsest buckets
        sparse_buckets = state_counts[:5]
        
        print(f"  📊 SPARSE BUCKET DIAGNOSTIC: 5 least populated buckets:")
        for i, (state, count) in enumerate(sparse_buckets):
            if count == 0:
                print(f"     {i+1}. {state}: {count} samples ❌ (EMPTY)")
            elif count < 10:
                print(f"     {i+1}. {state}: {count} samples ⚠️ (LOW)")
            else:
                print(f"     {i+1}. {state}: {count} samples")
    
    def _calculate_stationary_distribution(self):
        """Calculate stationary distribution of the Markov chain."""
        
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            
            # Find the eigenvector corresponding to eigenvalue 1
            stationary_idx = np.argmax(np.real(eigenvalues))
            stationary_vector = np.real(eigenvectors[:, stationary_idx])
            
            # Normalize to get probabilities
            self.stationary_distribution = stationary_vector / stationary_vector.sum()
            
        except Exception as e:
            print(f"  ⚠️ Could not calculate stationary distribution: {e}")
            # Fallback to uniform distribution
            self.stationary_distribution = np.ones(self.n_states) / self.n_states
    
    def predict_next_state(self, current_state: str, n_steps: int = 1) -> Dict[str, float]:
        """
        Predict next state probabilities.
        
        Parameters
        ----------
        current_state : str
            Current regime state
        n_steps : int
            Number of steps ahead to predict
            
        Returns
        -------
        Dict[str, float]
            Probability distribution over next states
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if current_state not in self.state_to_idx:
            raise ValueError(f"Unknown state: {current_state}")
        
        current_idx = self.state_to_idx[current_state]
        
        # Calculate n-step transition probabilities
        if n_steps == 1:
            next_probs = self.transition_matrix[current_idx, :]
        else:
            # Matrix power for n-step transitions
            n_step_matrix = np.linalg.matrix_power(self.transition_matrix, n_steps)
            next_probs = n_step_matrix[current_idx, :]
        
        # Convert to dictionary
        return {self.idx_to_state[i]: prob for i, prob in enumerate(next_probs)}
    
    def sample_sequence(self, initial_state: str, n_steps: int) -> List[str]:
        """
        Sample a sequence of states from the Markov chain.
        
        Parameters
        ----------
        initial_state : str
            Starting state
        n_steps : int
            Number of steps to simulate
            
        Returns
        -------
        List[str]
            Sequence of sampled states
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if initial_state not in self.state_to_idx:
            raise ValueError(f"Unknown state: {initial_state}")
        
        sequence = [initial_state]
        current_idx = self.state_to_idx[initial_state]
        
        for _ in range(n_steps):
            # Sample next state based on transition probabilities
            next_idx = np.random.choice(self.n_states, p=self.transition_matrix[current_idx, :])
            next_state = self.idx_to_state[next_idx]
            sequence.append(next_state)
            current_idx = next_idx
        
        return sequence
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        
        if not self.fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'model_type': self.model_type,
            'n_states': self.n_states,
            'states': self.states,
            'state_statistics': self.state_stats,
            'stationary_distribution': {
                self.idx_to_state[i]: prob 
                for i, prob in enumerate(self.stationary_distribution)
            },
            'transition_matrix_shape': self.transition_matrix.shape,
            'regime_config_info': REGIME_CLASSIFIER.get_regime_info()
        }
    
    def plot_transition_matrix(self, figsize: Tuple[int, int] = (10, 8)):
        """Plot the transition matrix as a heatmap."""
        
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.transition_matrix,
            xticklabels=self.states,
            yticklabels=self.states,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Transition Probability'}
        )
        plt.title(f'{self.model_type.title()} Regime Transition Matrix')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()
        plt.show()


# Convenience functions for creating specific model types
def create_trend_markov_model() -> UnifiedMarkovModel:
    """Create a Markov model for trend regimes only."""
    return UnifiedMarkovModel(model_type='trend')


def create_volatility_markov_model() -> UnifiedMarkovModel:
    """Create a Markov model for volatility regimes only."""
    return UnifiedMarkovModel(model_type='volatility')


def create_combined_markov_model() -> UnifiedMarkovModel:
    """Create a Markov model for combined trend-volatility regimes."""
    return UnifiedMarkovModel(model_type='combined')


def create_bb_markov_model() -> UnifiedMarkovModel:
    """Create a Markov model for Bollinger Band states."""
    bb_states = ['Extreme_Low', 'Low', 'Middle', 'High', 'Extreme_High']
    return UnifiedMarkovModel(model_type='custom', custom_states=bb_states)


# Export key components
__all__ = [
    'UnifiedMarkovModel',
    'create_trend_markov_model',
    'create_volatility_markov_model', 
    'create_combined_markov_model',
    'create_bb_markov_model'
]