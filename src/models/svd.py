import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def rolling_svd_factors(X, dates, assets, window_size=180, n_components=10):
    """
    Perform rolling SVD decomposition on a matrix of time-series data.

    Parameters
    ----------
    X : np.ndarray
        Time x Assets matrix of preprocessed returns or log prices
    dates : pd.Index
        Time index corresponding to rows in X
    assets : list or pd.Index
        Asset names corresponding to columns in X
    window_size : int
        Number of days to include in each rolling window
    n_components : int
        Number of principal components to extract

    Returns
    -------
    loadings_df : pd.DataFrame
        Asset loadings for each component, multi-indexed by (date, asset)
    components_df : pd.DataFrame
        Component values per date
    explained_var_df : pd.DataFrame
        Variance explained by each component per date
    """

    X = X.values if isinstance(X, pd.DataFrame) else X
    T = X.shape[0]
    loadings = []
    component_ts = []
    var_explained = []
    prev_U = None

    for t in range(window_size, T):
        window_data = X[t - window_size:t, :]
        row_dates = dates[t - window_size:t]

        valid_mask = ~np.isnan(window_data).all(axis=0)
        valid_assets = np.array(assets)[valid_mask]
        valid_data = window_data[:, valid_mask]

        if valid_data.shape[1] < n_components:
            continue

        # SVD: time x asset (so transpose)
        U, s, Vt = np.linalg.svd(valid_data.T, full_matrices=False)

        # Sign correction
        if prev_U is not None:
            for i in range(n_components):
                sign = np.sign(np.dot(U[:, i], prev_U[:, i]))
                if sign == -1:
                    U[:, i] *= -1
                    Vt[i, :] *= -1
        prev_U = U.copy()

        u_full = np.full((len(assets), n_components), np.nan)
        u_full[valid_mask] = U[:, :n_components]

        loadings.append(pd.DataFrame(u_full, index=assets, columns=[f'PC{i+1}' for i in range(n_components)])
                        .assign(date=dates[t]))

        V = Vt[:n_components, -1]
        component_ts.append(pd.Series(V, index=[f'PC{i+1}' for i in range(n_components)], name=dates[t]))

        var = (s**2) / np.sum(s**2)
        var_explained.append(pd.Series(var[:n_components], index=[f'PC{i+1}' for i in range(n_components)], name=dates[t]))

    loadings_df = pd.concat(loadings)
    loadings_df.index.name = 'asset'
    loadings_df = loadings_df.set_index('date', append=True).swaplevel()
    components_df = pd.DataFrame(component_ts)
    explained_var_df = pd.DataFrame(var_explained)

    return loadings_df, components_df, explained_var_df


def plot_explained_variance(explained_var_df):
    """Plot explained variance of each component over time."""
    explained_var_df.plot(title="Explained Variance by Component")
    plt.grid()
    plt.show()


def plot_component_series(components_df, components='PC1'):
    """
    Plot one or more SVD components over time.

    Parameters
    ----------
    components_df : pd.DataFrame
        DataFrame where each column is a principal component over time
    components : str or list of str
        Component(s) to plot (e.g., 'PC1' or ['PC1', 'PC2', 'PC3'])
    """
    if isinstance(components, str):
        components = [components]

    components_df[components].plot(title=f"SVD Component Series: {', '.join(components)}")
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Component Value")
    plt.tight_layout()
    plt.show()

def plot_asset_loading(loadings_df, asset_name, component='PC1'):
    """Plot loading of an asset on a single principal component over time."""
    if loadings_df.index.names != ['date', 'asset']:
        loadings_df.index.names = ['date', 'asset']
    loadings_df.xs(asset_name, level='asset')[[component]].plot(title=f"{asset_name} Loading on {component}")
    plt.grid()
    plt.show()


def plot_latest_heatmap(loadings_df, component_range=slice(0, 5)):
    """Plot heatmap of loadings on selected components at the most recent date."""
    latest_date = loadings_df.index.get_level_values('date').max()
    latest = loadings_df.xs(latest_date, level='date')
    sns.heatmap(latest.iloc[:, component_range], cmap='coolwarm', center=0)
    plt.title(f"Asset Loadings on Top Components ({latest_date.date()})")
    plt.show()

def plot_cumulative_explained_variance(explained_var_df, max_components=10):
    """
    Plot the cumulative explained variance over time for top N components.

    Parameters
    ----------
    explained_var_df : pd.DataFrame
        DataFrame with component variance by date (columns = PC1, PC2, ..., rows = dates)
    max_components : int
        Maximum number of components to include in the cumulative sum
    """
    # Ensure only up to N components
    cols = [f'PC{i+1}' for i in range(min(max_components, explained_var_df.shape[1]))]
    cumulative = explained_var_df[cols].cumsum(axis=1)

    plt.figure(figsize=(10, 6))
    for comp in cumulative.columns:
        plt.plot(cumulative.index, cumulative[comp], label=comp)

    plt.title(f'Cumulative Explained Variance (Top {len(cols)} Components)')
    plt.ylabel('Cumulative Variance Explained')
    plt.xlabel('Date')
    plt.grid(True)
    plt.legend(title='Component')
    plt.tight_layout()
    plt.show()