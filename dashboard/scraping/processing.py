from scraping.scrape import fetch_all_data, fetch_ticker_data, fetch_sector_data
import numpy as np
import pandas_ta as ta
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

COLUMNS = ['Earnings', 'Date', 'Est. Date', 'AfterH', 'Sector', 'Market Cap', 'atr_fraction', 'perfT', 'perfH', 'perfY', 'perfYtd',
           'Volume', 'Price', 'Change', 'RSI', 'perfW', 'perfM', 'perfQ', 'ATR', 'MACDh', 'n_bullish_candle', 'n_bearish_candle',
           'bullish_candle', 'bearish_candle', 'stop_loss', 'profit_take', 'position_shares']


def compute_technicals(hist):
    """
    Compute technical indicators on the historical market data.
    """
    macd = ta.macd(hist['Close'])
    hist = pd.concat([hist, macd], axis=1)
    hist = hist.dropna()
    return hist


def save_ticker_data(tickers, period="3mo"):
    """
    Fetch and save historical market data for a list of tickers.
    """
    ticker_data_map = {}
    for ticker in tickers:
        hist = fetch_ticker_data(ticker, period)
        ticker_data_map[ticker] = hist
    return ticker_data_map


def process_ticker_data(ticker_data_map):
    """
    Process technical indicators for each ticker's historical data.
    """
    processed_data_map = {}
    for ticker, hist in ticker_data_map.items():
        hist.dropna(inplace=True)
        if hist.shape[0] < 3:
            continue
        df = compute_technicals(hist)

        # Add candle pattern information
        cdl_df = hist.ta.cdl_pattern(name="all")
        cdl_df['Columns_with_100'] = cdl_df.apply(lambda x: find_columns_with_value(x, 100), axis=1)
        cdl_df['Columns_with_neg100'] = cdl_df.apply(lambda x: find_columns_with_value(x, -100), axis=1)
        df['bullish_candle'] = cdl_df['Columns_with_100']
        df['bearish_candle'] = cdl_df['Columns_with_neg100']
        processed_data_map[ticker] = df
    return processed_data_map


def fetch_and_merge_data():
    """
    Fetch all data, merge, and compute necessary metrics.
    """
    all_data = fetch_all_data()
    yf_tickers = [tick for tick in all_data.index]

    # Fetch historical market data and process technical indicators
    historical_data_map = save_ticker_data(yf_tickers)
    processed_data_map = process_ticker_data(historical_data_map)

    PORTFOLIO_SIZE = 100000
    R_FRACTION = 0.01
    R_DOLLAR = PORTFOLIO_SIZE * R_FRACTION
    ATR_MULTIPLE = 1
    PROFIT_MULTIPLE = 3

    all_data['bullish_candle'] = np.nan
    all_data['bearish_candle'] = np.nan
    all_data['n_bullish_candle'] = 0
    all_data['n_bearish_candle'] = 0

    for ticker, frame in processed_data_map.items():
        all_data.loc[ticker, 'MACDh'] = frame['MACDh_12_26_9'].values[-1]
        all_data.at[ticker, 'bullish_candle'] = str(frame.loc[frame.index[-1], 'bullish_candle'])
        all_data.at[ticker, 'bearish_candle'] = str(frame.loc[frame.index[-1], 'bearish_candle'])
        all_data.loc[ticker, 'n_bullish_candle'] = len(frame.loc[frame.index[-1], 'bullish_candle'])
        all_data.loc[ticker, 'n_bearish_candle'] = len(frame.loc[frame.index[-1], 'bearish_candle'])

        price = float(all_data.loc[ticker, 'Price'])
        atr_increment = float(all_data.loc[ticker, 'ATR']) * ATR_MULTIPLE
        stop_loss = price - atr_increment
        profit_take = price + PROFIT_MULTIPLE * atr_increment
        all_data.loc[ticker, 'stop_loss'] = stop_loss
        all_data.loc[ticker, 'profit_take'] = profit_take
        all_data.loc[ticker, 'position_shares'] = np.round(R_DOLLAR / atr_increment / 10) * 10
        all_data.loc[ticker, 'atr_fraction'] = atr_increment / price

    sector_df = fetch_sector_data()
    all_data = process_earnings_data(all_data)
    all_data = all_data.reset_index().merge(sector_df[['label', 'perfT', 'perfW', 'perfM', 'perfQ', 'perfH', 'perfY', 'perfYtd']],
                                            left_on='Sector', right_on='label', how='left')
    all_data = all_data.set_index('Ticker')
    all_data = all_data[COLUMNS]

    FIXED_VECTOR = [0.4, 0.3, 0.3, 5]
    all_data['score'] = all_data[['perfW', 'perfM', 'perfQ', 'MACDh']].apply(lambda row: sum(row * FIXED_VECTOR), axis=1)
    return all_data, sector_df


def process_earnings_data(df):
    df[['Date', 'AfterH']] = df['Earnings'].str.split('/', expand=True)
    df['Date'] = df['Date'].replace('-', pd.NaT)
    df['AfterH'] = df['AfterH'].replace('-', False)
    current_year = datetime.now().year
    df['Date'] = pd.to_datetime(df['Date'] + f' {current_year}', format='%b %d %Y', errors='coerce')
    df['AfterH'] = df['AfterH'].apply(lambda x: x.lower() == 'a' if pd.notna(x) else False)
    df['Date_Display'] = df['Date'].dt.strftime('%b %d')
    df['Est. Date'] = df['Date'].apply(estimate_earnings_date).dt.strftime('%m-%d')
    df['Date'] = df['Date'].dt.strftime('%m-%d')
    return df


def find_columns_with_value(row, value):
    return [col.replace("CDL_", "", 1) for col, val in row.items() if val == value]


def estimate_earnings_date(date):
    if pd.isna(date):
        return pd.NaT
    current_date = datetime.now()
    if date > current_date:
        return date
    else:
        return date + relativedelta(months=3)