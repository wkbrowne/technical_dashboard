#!/usr/bin/env python3
"""
Markov Transition Matrix Visualization Tool
===========================================

This script visualizes the Markov transition matrices for different trend regimes
as heatmaps with proper titles and colorbars.

Usage:
    python visualize_markov_matrices.py

The script will:
1. Load a trained global Markov model
2. Display transition matrices for 5-10 representative trend regimes
3. Show heatmaps with proper titles and colorbars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Import our models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.markov_bb import MultiStockBBMarkovModel, TrendAwareBBMarkovWrapper
from models.training_pipeline_api import TrainingPipelineAPI


class MarkovMatrixVisualizer:
    """Visualize Markov transition matrices for different trend regimes."""
    
    def __init__(self, cache_dir: str = "demo_cache"):
        self.cache_dir = Path(cache_dir)
        self.global_markov_model = None
        self.trend_regimes = []
        
    def load_trained_model(self) -> bool:
        """Load a trained global Markov model from cache."""
        cache_file = self.cache_dir / "stage1_global_markov.pkl"
        
        if not cache_file.exists():
            print(f"❌ No trained model found at {cache_file}")
            print("   Please run the training pipeline first:")
            print("   python demo_training_pipeline_api.py")
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.global_markov_model = cached_data['model']
            print(f"✅ Loaded global Markov model from {cache_file}")
            
            # Extract trend regimes
            if hasattr(self.global_markov_model, 'global_trend_models'):
                self.trend_regimes = list(self.global_markov_model.global_trend_models.keys())
                print(f"📊 Found {len(self.trend_regimes)} trend regimes: {self.trend_regimes}")
            else:
                print("⚠️ Model doesn't have trend-specific components")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def visualize_transition_matrices(self, max_regimes: int = 10, figsize: tuple = (20, 15)):
        """
        Visualize transition matrices for different trend regimes.
        
        Parameters
        ----------
        max_regimes : int
            Maximum number of regimes to display
        figsize : tuple
            Figure size for the plot
        """
        if not self.global_markov_model:
            print("❌ No model loaded. Call load_trained_model() first.")
            return
        
        # Select regimes to display (up to max_regimes)
        selected_regimes = self.trend_regimes[:max_regimes]
        n_regimes = len(selected_regimes)
        
        if n_regimes == 0:
            print("❌ No trend regimes found to visualize")
            return
        
        # Calculate grid size
        n_cols = min(3, n_regimes)
        n_rows = (n_regimes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_regimes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        print(f"🎨 Creating transition matrix heatmaps for {n_regimes} trend regimes...")
        
        for i, regime in enumerate(selected_regimes):
            ax = axes[i]
            
            # Get transition matrix for this regime
            if regime in self.global_markov_model.global_trend_priors:
                transition_matrix = self.global_markov_model.global_trend_priors[regime]
                
                # Get state labels (BB positions)
                if hasattr(self.global_markov_model, 'global_trend_models') and regime in self.global_markov_model.global_trend_models:
                    state_labels = self.global_markov_model.global_trend_models[regime].state_labels
                else:
                    # Default BB position labels
                    n_states = transition_matrix.shape[0]
                    state_labels = [f"BB{i+1}" for i in range(n_states)]
                
                # Create heatmap
                sns.heatmap(
                    transition_matrix, 
                    annot=True, 
                    fmt='.3f',
                    xticklabels=state_labels,
                    yticklabels=state_labels,
                    cmap='Blues',
                    ax=ax,
                    cbar=True,
                    square=True,
                    linewidths=0.5
                )
                
                ax.set_title(f'{regime.replace("_", " ").title()} Trend Regime\\nTransition Matrix', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Next BB State', fontsize=10)
                ax.set_ylabel('Current BB State', fontsize=10)
                
            else:
                # No data for this regime
                ax.text(0.5, 0.5, f'No Data\\n{regime}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{regime} (No Data)', fontsize=12)
        
        # Hide unused subplots
        for i in range(n_regimes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Markov Transition Matrices by Trend Regime', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        output_file = "markov_transition_matrices.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization to {output_file}")
        
        plt.show()
    
    def print_matrix_statistics(self):
        """Print statistics about the transition matrices."""
        if not self.global_markov_model:
            print("❌ No model loaded.")
            return
        
        print("\\n📊 MARKOV TRANSITION MATRIX STATISTICS")
        print("=" * 50)
        
        for regime in self.trend_regimes:
            if regime in self.global_markov_model.global_trend_priors:
                matrix = self.global_markov_model.global_trend_priors[regime]
                
                # Calculate statistics
                diagonal_sum = np.trace(matrix)  # Persistence probability
                max_transition = np.max(matrix)
                min_transition = np.min(matrix)
                entropy = -np.sum(matrix * np.log(matrix + 1e-10))  # Information entropy
                
                print(f"\\n🔍 {regime.replace('_', ' ').title()}:")
                print(f"   Persistence (diagonal sum): {diagonal_sum:.3f}")
                print(f"   Max transition probability: {max_transition:.3f}")
                print(f"   Min transition probability: {min_transition:.3f}")
                print(f"   Matrix entropy: {entropy:.3f}")
            else:
                print(f"\\n⚠️ {regime}: No transition matrix available")


def main():
    """Main function to run the visualization."""
    print("🎨 Markov Transition Matrix Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = MarkovMatrixVisualizer()
    
    # Load trained model
    if not visualizer.load_trained_model():
        print("\\n❌ Failed to load trained model. Please run training first.")
        return
    
    # Create visualizations
    print("\\n🎨 Creating transition matrix visualizations...")
    visualizer.visualize_transition_matrices(max_regimes=8)
    
    # Print statistics
    visualizer.print_matrix_statistics()
    
    print("\\n✅ Visualization complete!")


if __name__ == "__main__":
    main()