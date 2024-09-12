import streamlit as st
import pandas as pd
from scraping.scrape import get_market_sentiment
from scraping.processing import fetch_and_merge_data
import colorizers.colorizer as color
from colorizers.colorizer import (
    colorize_bullish_count,
    colorize_bearish_count,
    colorize_performance,
    colorize_sentiment,
    colorize_macd_histogram,
    colorize_rsi,
    colorize_sector_table
)


# Apply the colorizer function to each column using style.map
def style_dataframe(df):
    styled_df = df.style
    for col in df.columns[1:]:  # Skip the 'label' column
        styled_df = styled_df.applymap(lambda val: colorize_sector_table(val, col), subset=[col])
    return styled_df


# Function to apply styles to the sentiment DataFrame
def apply_sentiment_styles(df):
    styled_df = df.style.apply(lambda col: col.apply(colorize_sentiment, args=(col.name,)))
    return styled_df


# Streamlit app setup
st.set_page_config(layout="wide")

# Fetch market sentiment and display it
st.title("Market Sentiment Overview")
sentiment_dict = get_market_sentiment()
sentiment_df = pd.DataFrame(sentiment_dict, index=["Percent"], columns=sentiment_dict.keys())
st.session_state['sentiment_df'] = apply_sentiment_styles(sentiment_df)
st.write(st.session_state['sentiment_df'])

# Fetch all stock and sector data, and display sector data
st.title("Sector Sentiment Overview")
all_data, sector_df = fetch_and_merge_data()

# Apply the styling to sector_df and store in session state
st.session_state['sector_data'] = style_dataframe(sector_df)
st.table(st.session_state['sector_data'])

# Dashboard for uptrending stocks
st.title("Strong Uptrending Stocks Dashboard")

# Initialize or refresh the all_data
if 'all_data' not in st.session_state or st.button('Refresh Data'):
    st.session_state['all_data'] = all_data

# Apply styling to the stock data
styled_df = st.session_state.all_data.style \
    .applymap(colorize_macd_histogram, subset=['MACDh'])\
    .applymap(colorize_rsi, subset=['RSI'])\
    .applymap(lambda val: colorize_performance(val, color.max_weekly_perf, color.min_weekly_perf), subset=['perfW']) \
    .applymap(lambda val: colorize_performance(val, color.max_monthly_perf, color.min_monthly_perf), subset=['perfM']) \
    .applymap(lambda val: colorize_performance(val, color.max_quarterly_perf, color.min_quarterly_perf), subset=['perfQ'])\
    .applymap(colorize_bullish_count, subset=['n_bullish_candle'])\
    .applymap(colorize_bearish_count, subset=['n_bearish_candle'])

# Display the styled DataFrame
st.write(styled_df)