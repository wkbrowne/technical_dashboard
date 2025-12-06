"""
Feature visualization utilities for the technical dashboard.

This module provides tools for:
- Visualizing features for single stocks
- Feature distribution analysis
- Feature coverage (NaN analysis)
- Correlation matrices
- Interactive exploration with plotly

Both matplotlib (static) and plotly (interactive) backends are supported.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureVisualizer:
    """Feature visualization with matplotlib and plotly support.

    Usage:
        >>> df = pd.read_parquet('artifacts/features_daily.parquet')
        >>> viz = FeatureVisualizer(df)
        >>> viz.plot_single_stock('AAPL', ['rsi_14', 'macd_histogram'])
        >>> viz.interactive_stock_explorer('MSFT')  # plotly version
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        backend: str = 'matplotlib'
    ):
        """Initialize visualizer with features DataFrame.

        Args:
            features_df: DataFrame with columns ['symbol', 'date', ...features]
            backend: 'matplotlib' for static plots, 'plotly' for interactive
        """
        self.df = features_df.copy()
        self.backend = backend

        # Ensure proper types
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Get feature columns (exclude metadata)
        self.metadata_cols = {'symbol', 'date', 'week_end', 'month_end'}
        self.feature_cols = [
            col for col in self.df.columns
            if col not in self.metadata_cols
        ]

        logger.info(f"Initialized with {len(self.df)} rows, {len(self.feature_cols)} features")

    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return sorted(self.df['symbol'].unique().tolist())

    def get_features(self, prefix: Optional[str] = None) -> List[str]:
        """Get list of feature columns.

        Args:
            prefix: Optional prefix to filter (e.g., 'w_' for weekly)

        Returns:
            List of feature column names
        """
        if prefix:
            return [col for col in self.feature_cols if col.startswith(prefix)]
        return self.feature_cols

    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get data for a single stock.

        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Filtered DataFrame for the symbol
        """
        stock_data = self.df[self.df['symbol'] == symbol].copy()

        if start_date:
            stock_data = stock_data[stock_data['date'] >= start_date]
        if end_date:
            stock_data = stock_data[stock_data['date'] <= end_date]

        return stock_data.sort_values('date')

    # =========================================================================
    # Static Matplotlib Plots
    # =========================================================================

    def plot_single_stock(
        self,
        symbol: str,
        features: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        include_price: bool = True
    ):
        """Plot features for a single stock using matplotlib.

        Args:
            symbol: Stock symbol
            features: List of feature column names to plot
            start_date: Optional start date
            end_date: Optional end date
            figsize: Figure size
            include_price: Whether to include price chart

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        stock_data = self.get_stock_data(symbol, start_date, end_date)

        if stock_data.empty:
            logger.warning(f"No data for {symbol}")
            return None

        # Determine number of subplots
        n_plots = len(features) + (1 if include_price else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Price chart (if requested and available)
        if include_price:
            price_col = 'close' if 'close' in stock_data.columns else 'adjclose'
            if price_col in stock_data.columns:
                axes[plot_idx].plot(
                    stock_data['date'],
                    stock_data[price_col],
                    label=price_col.title(),
                    color='black',
                    linewidth=1
                )
                axes[plot_idx].set_ylabel('Price')
                axes[plot_idx].legend(loc='upper left')
                axes[plot_idx].set_title(f'{symbol} - Feature Analysis')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # Feature charts
        for feature in features:
            if feature in stock_data.columns:
                axes[plot_idx].plot(
                    stock_data['date'],
                    stock_data[feature],
                    label=feature
                )
                axes[plot_idx].set_ylabel(feature)
                axes[plot_idx].legend(loc='upper left')
                axes[plot_idx].grid(True, alpha=0.3)

                # Add zero line for centered features
                if any(x in feature.lower() for x in ['macd', 'ret', 'slope', 'dist']):
                    axes[plot_idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

                # Add thresholds for RSI
                if 'rsi' in feature.lower():
                    axes[plot_idx].axhline(y=70, color='red', linestyle='--', alpha=0.5)
                    axes[plot_idx].axhline(y=30, color='green', linestyle='--', alpha=0.5)
            else:
                axes[plot_idx].text(
                    0.5, 0.5,
                    f'{feature} not found',
                    transform=axes[plot_idx].transAxes,
                    ha='center', va='center'
                )
            plot_idx += 1

        plt.tight_layout()
        return fig

    def plot_feature_distribution(
        self,
        feature: str,
        by_symbol: bool = False,
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Plot distribution of a feature.

        Args:
            feature: Feature column name
            by_symbol: If True, show separate distributions per symbol
            bins: Number of histogram bins
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        if feature not in self.df.columns:
            logger.warning(f"Feature {feature} not found")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        if by_symbol:
            # Overlay distributions for each symbol
            symbols = self.get_symbols()[:10]  # Limit to 10 symbols
            for symbol in symbols:
                data = self.df[self.df['symbol'] == symbol][feature].dropna()
                ax.hist(data, bins=bins, alpha=0.5, label=symbol, density=True)
            ax.legend()
        else:
            # Single distribution
            data = self.df[feature].dropna()
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)

        ax.set_xlabel(feature)
        ax.set_ylabel('Density' if by_symbol else 'Count')
        ax.set_title(f'Distribution of {feature}')
        ax.grid(True, alpha=0.3)

        return fig

    def plot_correlation_matrix(
        self,
        features: Optional[List[str]] = None,
        method: str = 'spearman',
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot correlation matrix between features.

        Args:
            features: List of features to include (default: all)
            method: Correlation method ('pearson' or 'spearman')
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if features is None:
            # Use top N features by variance
            feature_vars = self.df[self.feature_cols].var().sort_values(ascending=False)
            features = feature_vars.head(20).index.tolist()

        # Compute correlation
        corr = self.df[features].corr(method=method)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title(f'Feature Correlation ({method})')

        plt.tight_layout()
        return fig

    def validate_feature_coverage(self) -> pd.DataFrame:
        """Check feature coverage (NaN percentages) by symbol.

        Returns:
            DataFrame with columns [symbol, feature, nan_pct, valid_count]
        """
        coverage = []
        for symbol in self.get_symbols():
            symbol_data = self.df[self.df['symbol'] == symbol]
            for col in self.feature_cols:
                nan_count = symbol_data[col].isna().sum()
                valid_count = (~symbol_data[col].isna()).sum()
                nan_pct = (nan_count / len(symbol_data)) * 100 if len(symbol_data) > 0 else 100

                coverage.append({
                    'symbol': symbol,
                    'feature': col,
                    'nan_pct': nan_pct,
                    'valid_count': valid_count,
                    'total_count': len(symbol_data)
                })

        return pd.DataFrame(coverage)

    def coverage_summary(self) -> pd.DataFrame:
        """Get summary of feature coverage across all symbols.

        Returns:
            DataFrame with mean/max NaN % per feature
        """
        coverage = self.validate_feature_coverage()
        summary = coverage.groupby('feature')['nan_pct'].agg(['mean', 'max', 'min']).reset_index()
        summary.columns = ['feature', 'nan_pct_mean', 'nan_pct_max', 'nan_pct_min']
        return summary.sort_values('nan_pct_mean', ascending=False)

    # =========================================================================
    # Interactive Plotly Plots
    # =========================================================================

    def interactive_stock_explorer(
        self,
        symbol: str,
        features: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        height: int = 800
    ):
        """Interactive stock explorer using plotly.

        Args:
            symbol: Stock symbol
            features: List of features (default: common features)
            start_date: Optional start date
            end_date: Optional end date
            height: Plot height in pixels

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return None

        stock_data = self.get_stock_data(symbol, start_date, end_date)
        if stock_data.empty:
            logger.warning(f"No data for {symbol}")
            return None

        # Default features if not specified
        if features is None:
            # Select common features that exist
            candidates = ['rsi_14', 'macd_histogram', 'vol_regime', 'trend_score_granular']
            features = [f for f in candidates if f in stock_data.columns][:4]

        n_plots = len(features) + 1  # +1 for price

        fig = make_subplots(
            rows=n_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f'{symbol} Price'] + features
        )

        # Price chart
        price_col = 'close' if 'close' in stock_data.columns else 'adjclose'
        if price_col in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data['date'],
                    y=stock_data[price_col],
                    name='Price',
                    line=dict(color='black', width=1)
                ),
                row=1, col=1
            )

        # Feature charts
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, feature in enumerate(features):
            if feature in stock_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data['date'],
                        y=stock_data[feature],
                        name=feature,
                        line=dict(color=colors[i % len(colors)], width=1)
                    ),
                    row=i+2, col=1
                )

                # Add zero line for centered features
                if any(x in feature.lower() for x in ['macd', 'ret', 'slope']):
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i+2, col=1)

        fig.update_layout(
            height=height,
            title_text=f'{symbol} - Interactive Feature Explorer',
            showlegend=True,
            hovermode='x unified'
        )

        return fig

    def interactive_feature_scatter(
        self,
        feature_x: str,
        feature_y: str,
        color_by: Optional[str] = None,
        symbol: Optional[str] = None
    ):
        """Interactive scatter plot of two features.

        Args:
            feature_x: Feature for x-axis
            feature_y: Feature for y-axis
            color_by: Optional feature for color coding
            symbol: Optional symbol filter

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.express as px
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return None

        data = self.df if symbol is None else self.df[self.df['symbol'] == symbol]

        if feature_x not in data.columns or feature_y not in data.columns:
            logger.warning(f"Features not found: {feature_x}, {feature_y}")
            return None

        fig = px.scatter(
            data,
            x=feature_x,
            y=feature_y,
            color=color_by,
            hover_data=['symbol', 'date'],
            title=f'{feature_x} vs {feature_y}'
        )

        return fig

    def interactive_coverage_heatmap(self):
        """Interactive heatmap of feature coverage by symbol.

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.express as px
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return None

        coverage = self.validate_feature_coverage()

        # Pivot for heatmap
        pivot = coverage.pivot(index='symbol', columns='feature', values='nan_pct')

        fig = px.imshow(
            pivot,
            labels=dict(x="Feature", y="Symbol", color="NaN %"),
            title="Feature Coverage (NaN %)",
            color_continuous_scale="RdYlGn_r",  # Red = high NaN
            aspect="auto"
        )

        return fig


def quick_feature_check(
    df: pd.DataFrame,
    symbol: str,
    features: Optional[List[str]] = None
) -> None:
    """Quick utility to check feature values for a symbol.

    Args:
        df: Features DataFrame
        symbol: Symbol to check
        features: Optional list of features (default: show stats for all)
    """
    stock_data = df[df['symbol'] == symbol]

    if stock_data.empty:
        print(f"No data for {symbol}")
        return

    if features is None:
        # Show summary stats for top features by variance
        numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
        features = [c for c in numeric_cols if c not in ['date']][:10]

    print(f"\n{symbol} Feature Summary ({len(stock_data)} rows)")
    print("=" * 60)

    for feature in features:
        if feature not in stock_data.columns:
            print(f"{feature}: NOT FOUND")
            continue

        series = stock_data[feature]
        nan_pct = series.isna().mean() * 100

        if series.dropna().empty:
            print(f"{feature}: ALL NaN")
            continue

        print(f"{feature}:")
        print(f"  min={series.min():.4f}, max={series.max():.4f}, "
              f"mean={series.mean():.4f}, std={series.std():.4f}")
        print(f"  NaN: {nan_pct:.1f}%")
