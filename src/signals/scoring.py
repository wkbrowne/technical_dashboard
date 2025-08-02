import numpy as np
import pandas as pd

def generate_signals_from_forecast(loadings_df, forecast_deltas, date, n_assets=10):
    """
    Generate long and short signals based on forecasted component moves.

    Parameters
    ----------
    loadings_df : pd.DataFrame
        MultiIndex (date, asset), component loadings
    forecast_deltas : dict
        Forecasted delta for each PC
    date : datetime
        Date to extract loadings from
    n_assets : int
        Number of assets to long and short

    Returns
    -------
    tuple
        longs : list of str
            Top N assets to long
        shorts : list of str
            Bottom N assets to short
        signal_series : pd.Series
            Raw signal values for all assets
    """
    factor_vector = np.array([forecast_deltas.get(f'PC{i+1}', 0) for i in range(len(next(iter(forecast_deltas))))])
    loadings = loadings_df.xs(date, level='date')
    signal = loadings.values @ factor_vector
    signal_series = pd.Series(signal, index=loadings.index)

    ranked = signal_series.sort_values(ascending=False)
    return ranked.head(n_assets).index.tolist(), ranked.tail(n_assets).index.tolist(), signal_series
