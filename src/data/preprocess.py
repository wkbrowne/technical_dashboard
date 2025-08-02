import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler

def preprocess_price_matrix(close_prices: pd.DataFrame,
                             method='log_return',
                             rolling_window=5,
                             drop_thresh=0.8,
                             scale_axis=1,
                             interpolate=True,
                             return_mask=False,
                             lookback_days=20,
                             winsorize_span=20):
    """
    Preprocesses price data for SVD modeling by smoothing, normalizing, cleaning, and applying EWM-based winsorization.

    Parameters
    ----------
    close_prices : pd.DataFrame
        DataFrame of raw price data (rows=dates, columns=assets)
    method : str
        'log_return' or 'log_price'
    rolling_window : int
        Rolling window for smoothing log returns
    drop_thresh : float
        Threshold for acceptable missingness per row/column
    scale_axis : int
        Axis for standard scaling (1=features/assets, 0=samples/dates)
    interpolate : bool
        Whether to interpolate missing data
    return_mask : bool
        If True, returns mask of interpolated values
    lookback_days : int
        Minimum lookback window required for asset inclusion
    winsorize_span : int
        Span parameter for exponentially weighted rolling winsorization

    Returns
    -------
    tuple or pd.DataFrame
        pre_scaled_df : pd.DataFrame of scaled (and optionally winsorized) values
        mask (optional) : pd.DataFrame of boolean interpolation mask
    """
    data = close_prices.copy()
    if interpolate:
        all_bdays = pd.date_range(start=data.index.min(), end=data.index.max(), freq=BDay())
        data = data.reindex(index=all_bdays)

    first_valid_row = data.index.min() + BDay(lookback_days)
    valid_assets = data.columns[data.loc[:first_valid_row].notna().any(axis=0)]
    dropped_assets = data.columns.difference(valid_assets)
    if len(dropped_assets) > 0:
        print("Dropped assets due to lookback_days requirement:")
        print(dropped_assets.tolist())

    data = data[valid_assets]

    def fill_leading_nans(df):
        def fill_col(col):
            first_valid_idx = col.first_valid_index()
            if first_valid_idx is not None:
                col.loc[:first_valid_idx] = col.loc[first_valid_idx]
            return col
        return df.apply(fill_col, axis=0)

    data = fill_leading_nans(data)
    interp_mask = data.isna()
    data = data.interpolate(method='linear', axis=0)

    row_thresh = int(drop_thresh * data.shape[1])
    data = data.dropna(axis=0, thresh=row_thresh)

    col_thresh = int(drop_thresh * data.shape[0])
    data = data.dropna(axis=1, thresh=col_thresh)

    if method == 'log_return':
        log_returns = np.log(data / data.shift(1))
        smoothed = log_returns.ewm(span=rolling_window, adjust=False).mean()
        pre_data = smoothed.dropna()
    elif method == 'log_price':
        log_prices = np.log(data)
        pre_data = log_prices.dropna()
    else:
        raise ValueError("method must be either 'log_return' or 'log_price'")

    # EWM-based adaptive winsorization
    def adaptive_winsorize(series):
        ewm_mean = series.ewm(span=winsorize_span, adjust=False).mean()
        ewm_std = series.ewm(span=winsorize_span, adjust=False).std()
        upper = ewm_mean + 3 * ewm_std
        lower = ewm_mean - 3 * ewm_std
        return np.clip(series, lower, upper)

    winsorized_data = pre_data.apply(adaptive_winsorize)
    winsorized_mask = winsorized_data != pre_data
    total_winsorized = winsorized_mask.sum().sum()
    print(f"Total values winsorized: {total_winsorized:,}")

    # Scale
    scaler = StandardScaler()
    pre_scaled = scaler.fit_transform(winsorized_data.values) if scale_axis == 1 else scaler.fit_transform(winsorized_data.values.T).T
    pre_scaled_df = pd.DataFrame(pre_scaled, index=pre_data.index, columns=pre_data.columns)

    if return_mask:
        mask = interp_mask.loc[pre_data.index, pre_data.columns]
        return pre_scaled_df, mask
    else:
        return pre_scaled_df
