import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
from scipy import stats
from sklearn.preprocessing import LabelEncoder

class GridMarkovModel:
    """
    7x3 Grid Markov Model for Bollinger Bands and Price Trends.
    
    This model creates a 21-state space (7 trend states x 3 BB states) where:
    - 7 Trend States: parabolic_up, strong_up, up, ranging, down, strong_down, parabolic_down
    - 3 BB States: Low_BB, Med_BB, High_BB
    - Equal sampling across all 21 combinations
    """
    
    def __init__(self, 
                 parabolic_up_thresh: float = 0.05,
                 strong_up_thresh: float = 0.025,
                 up_thresh: float = 0.008,
                 down_thresh: float = -0.008,
                 strong_down_thresh: float = -0.025,
                 parabolic_down_thresh: float = -0.05):
        
        # Trend classification thresholds
        self.parabolic_up_thresh = parabolic_up_thresh
        self.strong_up_thresh = strong_up_thresh
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.strong_down_thresh = strong_down_thresh
        self.parabolic_down_thresh = parabolic_down_thresh
        
        # Define the 7x3 grid
        self.trend_states = ['parabolic_up', 'strong_up', 'up', 'ranging', 'down', 'strong_down', 'parabolic_down']
        self.bb_states = ['Low_BB', 'Med_BB', 'High_BB']
        
        # Create 21 combined states
        self.grid_states = []
        self.state_to_index = {}
        self.index_to_state = {}
        
        index = 0
        for trend in self.trend_states:
            for bb in self.bb_states:
                combined_state = f"{trend}_{bb}"
                self.grid_states.append(combined_state)
                self.state_to_index[combined_state] = index
                self.index_to_state[index] = combined_state
                index += 1
        
        self.n_states = len(self.grid_states)  # 21 states
        
        # Model components
        self.transition_matrix = None
        self.steady_state = None
        self.fitted = False
        
        # For equal sampling
        self.min_samples_per_state = None
        self.sampled_data = None
        
    def classify_trend(self, ma_series: pd.Series) -> pd.Series:
        """Classify price trend based on MA slope."""
        slope = ma_series.pct_change(periods=5)  # 5-day slope
        
        trend = pd.Series(index=ma_series.index, dtype='object')
        
        # 7-state classification
        trend[slope > self.parabolic_up_thresh] = 'parabolic_up'
        trend[(slope > self.strong_up_thresh) & (slope <= self.parabolic_up_thresh)] = 'strong_up'
        trend[(slope > self.up_thresh) & (slope <= self.strong_up_thresh)] = 'up'
        trend[(slope >= self.down_thresh) & (slope <= self.up_thresh)] = 'ranging'
        trend[(slope >= self.strong_down_thresh) & (slope < self.down_thresh)] = 'down'
        trend[(slope >= self.parabolic_down_thresh) & (slope < self.strong_down_thresh)] = 'strong_down'
        trend[slope < self.parabolic_down_thresh] = 'parabolic_down'
        
        return trend.ffill()
    
    def classify_bb_state(self, bb_position: pd.Series) -> pd.Series:
        """Classify BB position into 3 states with equal sampling."""
        bb_state = pd.Series(index=bb_position.index, dtype='object')
        
        # Use terciles for equal distribution
        low_threshold = bb_position.quantile(0.33)
        high_threshold = bb_position.quantile(0.67)
        
        bb_state[bb_position <= low_threshold] = 'Low_BB'
        bb_state[(bb_position > low_threshold) & (bb_position <= high_threshold)] = 'Med_BB'
        bb_state[bb_position > high_threshold] = 'High_BB'
        
        return bb_state
    
    def create_combined_states(self, trend_series: pd.Series, bb_series: pd.Series) -> pd.Series:
        """Create combined 7x3 grid states."""
        combined = trend_series.astype(str) + "_" + bb_series.astype(str)
        return combined
    
    def balance_sampling(self, data: pd.DataFrame, target_samples: int = None) -> pd.DataFrame:
        """
        Ensure EXACTLY equal sampling across all 21 grid states.
        
        Every combination (trend × BB state) gets the exact same number of samples.
        E.g., parabolic_up × high_bb = ranging × low_bb = same sample count
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with 'Combined_State' column
        target_samples : int
            Target number of samples per state. If None, uses minimum available across ALL 21 states.
        
        Returns:
        --------
        pd.DataFrame
            Perfectly balanced dataset with identical samples per state
        """
        print("⚖️ Ensuring EXACTLY EQUAL sampling across all 21 grid combinations...")
        
        # Count samples per state for ALL 21 possible combinations
        state_counts = data['Combined_State'].value_counts().reindex(self.grid_states, fill_value=0)
        
        print(f"📊 Initial state distribution analysis:")
        print(f"  Total data points: {len(data):,}")
        print(f"  Available states: {(state_counts > 0).sum()}/21")
        print(f"  Min samples per state: {state_counts.min()}")
        print(f"  Max samples per state: {state_counts.max()}")
        
        # Determine target samples - must be achievable by ALL states
        if target_samples is None:
            # Use the minimum count across all 21 states (including zero counts)
            min_available = state_counts.min()
            if min_available == 0:
                # Some states have no data - use a conservative approach
                non_zero_counts = state_counts[state_counts > 0]
                if len(non_zero_counts) > 0:
                    target_samples = min(non_zero_counts.min(), max(50, int(non_zero_counts.quantile(0.1))))
                else:
                    target_samples = 50  # Fallback minimum
            else:
                target_samples = min_available
            
            print(f"📊 Auto-selected target: {target_samples} samples per state")
        else:
            print(f"📊 Using specified target: {target_samples} samples per state")
        
        self.min_samples_per_state = target_samples
        
        # Ensure exactly equal sampling across all 21 combinations
        balanced_data = []
        states_with_insufficient_data = []
        states_perfectly_balanced = []
        
        print(f"\n🎯 Sampling exactly {target_samples} samples from each grid state:")
        
        for state in self.grid_states:
            state_data = data[data['Combined_State'] == state]
            available_samples = len(state_data)
            
            if available_samples >= target_samples:
                # Perfect case: enough data for exact sampling
                sampled = state_data.sample(n=target_samples, random_state=42)
                balanced_data.append(sampled)
                states_perfectly_balanced.append(state)
                print(f"✅ {state}: {target_samples}/{available_samples} samples (perfectly balanced)")
                
            elif available_samples > 0:
                # Insufficient data: use all available + synthetic/resampling
                if available_samples >= target_samples // 2:
                    # Oversample to reach target
                    n_resample = target_samples - available_samples
                    resampled = state_data.sample(n=n_resample, replace=True, random_state=42)
                    combined = pd.concat([state_data, resampled], ignore_index=True)
                    balanced_data.append(combined)
                    states_with_insufficient_data.append((state, available_samples, 'oversampled'))
                    print(f"🔄 {state}: {available_samples}→{target_samples} samples (oversampled)")
                else:
                    # Very little data: use all available (breaks perfect balance)
                    balanced_data.append(state_data)
                    states_with_insufficient_data.append((state, available_samples, 'insufficient'))
                    print(f"⚠️ {state}: only {available_samples}/{target_samples} samples (insufficient data)")
            else:
                # No data at all for this state
                states_with_insufficient_data.append((state, 0, 'no_data'))
                print(f"❌ {state}: 0/{target_samples} samples (no data available)")
        
        # Combine all balanced data
        if balanced_data:
            balanced_df = pd.concat(balanced_data, ignore_index=True)
        else:
            balanced_df = pd.DataFrame()
        
        # Final verification of balance
        if not balanced_df.empty:
            final_counts = balanced_df['Combined_State'].value_counts()
            perfect_balance = (final_counts == target_samples).all()
            
            print(f"\n📊 FINAL BALANCE VERIFICATION:")
            print(f"  Total balanced samples: {len(balanced_df):,}")
            print(f"  States perfectly balanced: {len(states_perfectly_balanced)}/21")
            print(f"  States with issues: {len(states_with_insufficient_data)}/21")
            print(f"  Perfect equal sampling achieved: {'✅ YES' if perfect_balance else '⚠️ NO'}")
            
            if not perfect_balance:
                print(f"  Sample count range: {final_counts.min()}-{final_counts.max()} per state")
                
                # Show states that don't have perfect balance
                imperfect_states = final_counts[final_counts != target_samples]
                if len(imperfect_states) > 0:
                    print(f"  Imperfect states:")
                    for state, count in imperfect_states.items():
                        print(f"    {state}: {count} samples")
            
            # Show summary by trend and BB state
            # Extract trend and BB components from combined states
            # Combined states are like "parabolic_up_High_BB", need to split on pattern
            def extract_components(combined_state):
                """Extract trend and BB components from combined state name."""
                # Find the BB component (ends with _BB)
                for bb_state in self.bb_states:
                    if combined_state.endswith(bb_state):
                        # Remove BB component to get trend
                        trend_part = combined_state[:-len(bb_state)-1]  # -1 for underscore
                        return trend_part, bb_state
                return combined_state, "Unknown"
            
            # Apply extraction
            components = balanced_df['Combined_State'].apply(extract_components)
            balanced_df['Trend_Component'] = [comp[0] for comp in components]
            balanced_df['BB_Component'] = [comp[1] for comp in components]
            
            trend_totals = balanced_df['Trend_Component'].value_counts()
            bb_totals = balanced_df['BB_Component'].value_counts()
            
            print(f"\n📈 BALANCE BY COMPONENTS:")
            print(f"  Trend State Totals: {dict(trend_totals)}")
            print(f"  BB State Totals: {dict(bb_totals)}")
            
        else:
            print(f"\n❌ No balanced data generated - check input data quality")
        
        return balanced_df
    
    def fit(self, all_stock_data: Dict[str, pd.DataFrame], 
            target_samples_per_state: int = None,
            alpha_prior: float = 1.0) -> 'GridMarkovModel':
        """
        Fit the 7x3 grid Markov model with equal sampling.
        
        Parameters:
        -----------
        all_stock_data : dict
            Dictionary with stock data containing 'BB_Position', 'MA', etc.
        target_samples_per_state : int
            Target number of samples per grid state
        alpha_prior : float
            Dirichlet prior concentration parameter
        """
        print("🚀 FITTING 7x3 GRID MARKOV MODEL")
        print("=" * 60)
        
        print(f"📊 Grid Configuration:")
        print(f"  Trend States: {len(self.trend_states)} ({self.trend_states})")
        print(f"  BB States: {len(self.bb_states)} ({self.bb_states})")
        print(f"  Total Grid States: {self.n_states}")
        
        # Combine all stock data
        combined_data = []
        
        for symbol, stock_data in all_stock_data.items():
            if 'BB_Position' not in stock_data.columns or 'MA' not in stock_data.columns:
                print(f"⚠️ Skipping {symbol}: missing required columns")
                continue
            
            # Classify trends and BB states
            trends = self.classify_trend(stock_data['MA'])
            bb_states = self.classify_bb_state(stock_data['BB_Position'])
            combined_states = self.create_combined_states(trends, bb_states)
            
            # Create dataframe with all required info
            stock_df = pd.DataFrame({
                'Symbol': symbol,
                'Date': stock_data.index,
                'BB_Position': stock_data['BB_Position'],
                'MA': stock_data['MA'],
                'Trend_State': trends,
                'BB_State': bb_states,
                'Combined_State': combined_states
            }).dropna()
            
            combined_data.append(stock_df)
            print(f"✅ {symbol}: {len(stock_df)} observations")
        
        # Combine all data
        all_data = pd.concat(combined_data, ignore_index=True)
        print(f"\n📊 Total combined data: {len(all_data)} observations")
        
        # Balance sampling across grid states
        self.sampled_data = self.balance_sampling(all_data, target_samples_per_state)
        
        # Create transition matrix from balanced data
        self._fit_transition_matrix(alpha_prior)
        
        self.fitted = True
        print(f"\n✅ 7x3 Grid Markov Model fitted successfully!")
        
        return self
    
    def _fit_transition_matrix(self, alpha_prior: float):
        """Fit transition matrix from balanced data."""
        print(f"\n🔄 Computing transition matrix with α={alpha_prior}...")
        
        # Convert combined states to indices
        state_indices = self.sampled_data['Combined_State'].map(self.state_to_index)
        
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Group by symbol to get proper transitions
        for symbol in self.sampled_data['Symbol'].unique():
            symbol_data = self.sampled_data[self.sampled_data['Symbol'] == symbol].sort_values('Date')
            symbol_states = symbol_data['Combined_State'].map(self.state_to_index).values
            
            # Count transitions within this symbol
            for i in range(len(symbol_states) - 1):
                current_state = symbol_states[i]
                next_state = symbol_states[i + 1]
                transition_counts[current_state, next_state] += 1
        
        # Add Dirichlet prior
        transition_counts += alpha_prior
        
        # Normalize to get probabilities
        self.transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        
        # Calculate steady state
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        steady_idx = np.argmax(eigenvals.real)
        self.steady_state = eigenvecs[:, steady_idx].real
        self.steady_state = self.steady_state / self.steady_state.sum()
        
        print(f"✅ Transition matrix computed: {self.transition_matrix.shape}")
        print(f"📊 Non-zero transitions: {np.count_nonzero(self.transition_matrix)}")
    
    def get_state_index(self, trend: str, bb_state: str) -> int:
        """Get state index for a trend-BB combination."""
        combined_state = f"{trend}_{bb_state}"
        return self.state_to_index.get(combined_state, 0)
    
    def get_current_state(self, bb_position: float, ma_slope: float) -> Tuple[int, str]:
        """Get current state from BB position and MA slope."""
        # Classify trend
        if ma_slope > self.parabolic_up_thresh:
            trend = 'parabolic_up'
        elif ma_slope > self.strong_up_thresh:
            trend = 'strong_up'
        elif ma_slope > self.up_thresh:
            trend = 'up'
        elif ma_slope >= self.down_thresh:
            trend = 'ranging'
        elif ma_slope >= self.strong_down_thresh:
            trend = 'down'
        elif ma_slope >= self.parabolic_down_thresh:
            trend = 'strong_down'
        else:
            trend = 'parabolic_down'
        
        # Classify BB state (simplified for real-time use)
        if bb_position <= -0.5:
            bb_state = 'Low_BB'
        elif bb_position >= 0.5:
            bb_state = 'High_BB'
        else:
            bb_state = 'Med_BB'
        
        state_index = self.get_state_index(trend, bb_state)
        combined_state = f"{trend}_{bb_state}"
        
        return state_index, combined_state
    
    def forecast_states(self, current_state: int, n_steps: int = 20) -> np.ndarray:
        """Forecast future states using transition matrix."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        states = [current_state]
        current = current_state
        
        for _ in range(n_steps):
            # Sample next state from transition probabilities
            probs = self.transition_matrix[current]
            next_state = np.random.choice(self.n_states, p=probs)
            states.append(next_state)
            current = next_state
        
        return np.array(states[1:])  # Return forecasted states (exclude current)
    
    def plot_grid_heatmap(self, figsize: Tuple[int, int] = (15, 8)):
        """Plot heatmap of steady-state probabilities on 7x3 grid."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Reshape steady state into 7x3 grid
        grid_probs = self.steady_state.reshape(len(self.trend_states), len(self.bb_states))
        
        plt.figure(figsize=figsize)
        sns.heatmap(grid_probs, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=self.bb_states,
                   yticklabels=self.trend_states,
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Steady-State Probability'})
        
        plt.title('7x3 Grid Markov Model - Steady State Probabilities', fontsize=14, fontweight='bold')
        plt.xlabel('Bollinger Band State', fontsize=12)
        plt.ylabel('Trend State', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_sampling_grid_heatmap(self, figsize: Tuple[int, int] = (15, 8)):
        """Plot heatmap showing exact sample counts for each grid combination."""
        if not self.fitted or self.sampled_data is None:
            raise ValueError("Model must be fitted with sampling data first")
        
        # Count samples per grid state
        sample_counts = self.sampled_data['Combined_State'].value_counts().reindex(self.grid_states, fill_value=0)
        
        # Reshape into 7x3 grid for visualization
        grid_samples = np.zeros((len(self.trend_states), len(self.bb_states)))
        
        for i, trend in enumerate(self.trend_states):
            for j, bb_state in enumerate(self.bb_states):
                combined_state = f"{trend}_{bb_state}"
                grid_samples[i, j] = sample_counts.get(combined_state, 0)
        
        plt.figure(figsize=figsize)
        sns.heatmap(grid_samples, 
                   annot=True, 
                   fmt='g',  # Integer format
                   xticklabels=self.bb_states,
                   yticklabels=self.trend_states,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Sample Count'})
        
        plt.title('7x3 Grid Sampling Balance - Exact Sample Counts per Combination', fontsize=14, fontweight='bold')
        plt.xlabel('Bollinger Band State', fontsize=12)
        plt.ylabel('Trend State', fontsize=12)
        
        # Add perfect balance indicator
        unique_counts = np.unique(grid_samples)
        if len(unique_counts) == 1:
            plt.suptitle(f'✅ PERFECT BALANCE: {int(unique_counts[0])} samples per combination', fontsize=12, color='green')
        else:
            plt.suptitle(f'⚠️ Range: {int(grid_samples.min())}-{int(grid_samples.max())} samples per combination', fontsize=12, color='orange')
        
        plt.tight_layout()
        plt.show()
        
        # Also print the exact grid
        print(f"\n📊 EXACT SAMPLE COUNT GRID:")
        print("=" * 80)
        header_label = "Trend \\ BB State"
        print(f"{header_label:<15} {'Low_BB':<8} {'Med_BB':<8} {'High_BB':<8} {'Total':<8}")
        print("-" * 80)
        
        for i, trend in enumerate(self.trend_states):
            row_counts = [int(grid_samples[i, j]) for j in range(len(self.bb_states))]
            row_total = sum(row_counts)
            print(f"{trend:<15} {row_counts[0]:<8} {row_counts[1]:<8} {row_counts[2]:<8} {row_total:<8}")
        
        # Column totals
        col_totals = [int(grid_samples[:, j].sum()) for j in range(len(self.bb_states))]
        grand_total = sum(col_totals)
        print("-" * 80)
        print(f"{'Total':<15} {col_totals[0]:<8} {col_totals[1]:<8} {col_totals[2]:<8} {grand_total:<8}")
        
        # Balance verification
        all_counts = grid_samples.flatten()
        unique_counts = np.unique(all_counts)
        
        if len(unique_counts) == 1:
            print(f"\n✅ PERFECT BALANCE ACHIEVED: {int(unique_counts[0])} samples per combination")
        else:
            print(f"\n⚠️ IMPERFECT BALANCE:")
            print(f"   Min samples: {int(all_counts.min())}")
            print(f"   Max samples: {int(all_counts.max())}")
            print(f"   Range: {int(all_counts.max() - all_counts.min())}")
            balance_ratio = all_counts.min() / all_counts.max() if all_counts.max() > 0 else 0
            print(f"   Balance ratio: {balance_ratio:.3f}")
            
        return grid_samples
    
    def plot_transition_heatmap(self, figsize: Tuple[int, int] = (20, 16)):
        """Plot full 21x21 transition matrix heatmap."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.transition_matrix, 
                   annot=False,  # Too many states for annotations
                   xticklabels=self.grid_states,
                   yticklabels=self.grid_states,
                   cmap='Blues',
                   cbar_kws={'label': 'Transition Probability'})
        
        plt.title('21x21 Grid State Transition Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Next State', fontsize=12)
        plt.ylabel('Current State', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Analyze sampling balance
        sampling_summary = {}
        if self.sampled_data is not None:
            state_counts = self.sampled_data['Combined_State'].value_counts()
            sampling_summary = {
                'total_samples': len(self.sampled_data),
                'samples_per_state': state_counts.to_dict(),
                'min_samples': state_counts.min(),
                'max_samples': state_counts.max(),
                'target_samples': self.min_samples_per_state
            }
        
        return {
            'model_type': '7x3_Grid_Markov',
            'n_states': self.n_states,
            'trend_states': self.trend_states,
            'bb_states': self.bb_states,
            'grid_states': self.grid_states,
            'sampling_summary': sampling_summary,
            'transition_matrix_shape': self.transition_matrix.shape,
            'steady_state_shape': self.steady_state.shape,
            'fitted': self.fitted
        }

class MultiStockGridMarkovModel:
    """
    Multi-stock version of the 7x3 Grid Markov Model.
    Supports both global and stock-specific models.
    """
    
    def __init__(self, **kwargs):
        self.global_model = GridMarkovModel(**kwargs)
        self.stock_models = {}
        self.fitted = False
    
    def fit_global_model(self, all_stock_data: Dict[str, pd.DataFrame], **kwargs):
        """Fit global model using all stock data."""
        print("🌍 FITTING GLOBAL 7x3 GRID MARKOV MODEL")
        print("=" * 60)
        
        self.global_model.fit(all_stock_data, **kwargs)
        self.fitted = True
        
        return self.global_model
    
    def fit_stock_model(self, symbol: str, stock_data: pd.DataFrame, **kwargs):
        """Fit stock-specific model."""
        print(f"🏢 FITTING STOCK-SPECIFIC MODEL: {symbol}")
        print("=" * 40)
        
        stock_model = GridMarkovModel(
            parabolic_up_thresh=self.global_model.parabolic_up_thresh,
            strong_up_thresh=self.global_model.strong_up_thresh,
            up_thresh=self.global_model.up_thresh,
            down_thresh=self.global_model.down_thresh,
            strong_down_thresh=self.global_model.strong_down_thresh,
            parabolic_down_thresh=self.global_model.parabolic_down_thresh
        )
        
        stock_model.fit({symbol: stock_data}, **kwargs)
        self.stock_models[symbol] = stock_model
        
        return stock_model
    
    def get_model(self, symbol: str = None) -> GridMarkovModel:
        """Get model for specific stock or global model."""
        if symbol and symbol in self.stock_models:
            return self.stock_models[symbol]
        return self.global_model
    
    def fit_etf_model(self, etf_symbol: str, etf_data: pd.DataFrame, 
                      target_samples_per_state: int = None, 
                      alpha_prior: float = 2.0) -> GridMarkovModel:
        """
        Fit ETF-specific model using global stock priors.
        
        This creates an ETF-specific model that leverages the global stock model's
        knowledge without contaminating the stock-based global priors.
        
        Parameters:
        -----------
        etf_symbol : str
            ETF symbol (e.g., 'SPY', 'QQQ', 'IWM')
        etf_data : pd.DataFrame
            ETF OHLC data with BB_Position, MA, etc.
        target_samples_per_state : int
            Target samples per grid state for ETF
        alpha_prior : float
            Bayesian prior weight (higher = more global influence)
            
        Returns:
        --------
        GridMarkovModel
            Trained ETF-specific model
        """
        print(f"🏦 FITTING ETF-SPECIFIC MODEL: {etf_symbol}")
        print("=" * 50)
        
        if not self.fitted:
            raise ValueError("Global model must be fitted first to provide priors for ETF model")
        
        # Create new ETF-specific model with same parameters as global
        etf_model = GridMarkovModel(
            parabolic_up_thresh=self.global_model.parabolic_up_thresh,
            strong_up_thresh=self.global_model.strong_up_thresh,
            up_thresh=self.global_model.up_thresh,
            down_thresh=self.global_model.down_thresh,
            strong_down_thresh=self.global_model.strong_down_thresh,
            parabolic_down_thresh=self.global_model.parabolic_down_thresh
        )
        
        print(f"📊 ETF Model Configuration:")
        print(f"  Using global stock priors as Bayesian baseline")
        print(f"  Alpha prior weight: {alpha_prior} (higher = more global influence)")
        print(f"  ETF data points: {len(etf_data):,}")
        
        # Fit ETF model with ETF data only
        etf_model.fit({etf_symbol: etf_data}, 
                      target_samples_per_state=target_samples_per_state,
                      alpha_prior=alpha_prior)
        
        # Now apply global stock priors as Bayesian priors for the ETF model
        if etf_model.fitted and self.global_model.fitted:
            print(f"\n🔄 Applying global stock priors to ETF transition matrix...")
            
            # Blend ETF transitions with global stock priors
            etf_transitions = etf_model.transition_matrix.copy()
            global_transitions = self.global_model.transition_matrix.copy()
            
            # Bayesian update: blend transitions with prior weight
            blended_transitions = (etf_transitions + alpha_prior * global_transitions) / (1 + alpha_prior)
            etf_model.transition_matrix = blended_transitions
            
            # Recompute steady state with blended transitions
            eigenvals, eigenvecs = np.linalg.eig(blended_transitions.T)
            steady_idx = np.argmax(eigenvals.real)
            etf_model.steady_state = eigenvecs[:, steady_idx].real
            etf_model.steady_state = etf_model.steady_state / etf_model.steady_state.sum()
            
            print(f"✅ Global priors successfully applied to {etf_symbol} model")
            print(f"🎯 Transition matrix shape: {etf_model.transition_matrix.shape}")
            
            # Store in ETF models dictionary
            if not hasattr(self, 'etf_models'):
                self.etf_models = {}
            self.etf_models[etf_symbol] = etf_model
            
            print(f"\n📈 ETF Model Benefits:")
            print(f"  🎲 Leverages global stock market knowledge")
            print(f"  📊 Tailored to {etf_symbol}-specific patterns")
            print(f"  ⚖️ Balanced between ETF data and stock priors")
            print(f"  🔄 Does not contaminate global stock model")
            
        else:
            print(f"⚠️ Warning: Could not apply global priors - models not properly fitted")
        
        return etf_model