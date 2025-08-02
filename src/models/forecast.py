import numpy as np
import pandas as pd
import warnings
from pmdarima import auto_arima


def forecast_components_arima(components_df, horizon=10, delta_method='sum'):
    """
    Fit auto_arima to each principal component and forecast forward.

    Parameters
    ----------
    components_df : pd.DataFrame
        Principal component time series (columns = PC1, PC2, ...)
    horizon : int
        Number of steps to forecast
    delta_method : str
        Method to calculate forecast delta: 'point', 'mean', 'trend', or 'sum'

    Returns
    -------
    pd.DataFrame
        Forecast metadata including delta, model order, AIC, and forecast path
    """
    results = []

    for col in components_df.columns:
        series = components_df[col].dropna()
        delta = np.nan
        model_order = (None, None, None)
        aic = np.nan
        forecast_path = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)

            try:
                model = auto_arima(series, seasonal=False, suppress_warnings=True, error_action="ignore")
                forecast = model.predict(n_periods=horizon)

                if delta_method == 'point':
                    delta = forecast[-1] - series.iloc[-1]
                elif delta_method == 'mean':
                    delta = forecast.mean() - series.iloc[-1]
                elif delta_method == 'trend':
                    delta = forecast[-1] - forecast[0]
                elif delta_method == 'sum':
                    delta = forecast.sum()
                else:
                    raise ValueError("Invalid delta_method. Choose from 'point', 'mean', 'trend', or 'sum'.")

                model_order = model.order
                aic = model.aic()
                forecast_path = forecast

            except Exception as e:
                print(f"Auto-ARIMA failed on {col}: {e}")

        results.append({
            'component': col,
            'forecast_delta': delta,
            'order': model_order,
            'aic': aic,
            'forecast_path': forecast_path
        })

    return pd.DataFrame(results).set_index('component')