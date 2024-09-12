import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import yfinance as yf
from datetime import datetime

SCREENER_URL = "https://finviz.com/screener.ashx?v=152&f=fa_epsyoy_o20,geo_usa,ta_rsi_nob60,ta_sma20_pa,ta_sma200_pa,ta_sma50_pa&ft=3&c=1,2,3,4,6,17,49,59,68,67,65,66"
SECTOR_URL = "https://finviz.com/groups.ashx"


def fetch_data(url):
    """
    Fetch HTML content from a given URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    return response.text


def extract_table_values(table):
    """
    Extract table values from a BeautifulSoup table object.
    """
    columns = [header.text.strip() for header in table.find_all('th')]
    data = []
    rows = table.find_all('tr', {'class': 'styled-row is-hoverable is-bordered is-rounded is-striped has-color-text'})
    for row in rows:
        cols = row.find_all('td')
        if cols:
            row_data = {columns[i]: cols[i].text.strip() for i in range(len(cols))}
            data.append(row_data)
    df = pd.DataFrame(data)
    if 'No.' in df.columns:
        df = df.drop(columns=['No.'])
    if 'Ticker' in df.columns:
        df = df.set_index('Ticker')
    return df


def get_pagination_urls(base_url):
    """
    Get all pagination URLs from the initial page.
    """
    html = fetch_data(base_url)
    soup = BeautifulSoup(html, 'html.parser')
    pagination_select = soup.find('select', {'class': 'pages-combo fv-select', 'id': 'pageSelect'})
    
    page_dict = {}
    if pagination_select:
        options = pagination_select.find_all('option')
        for option in options:
            page_number = option.text.strip()
            page_value = option['value']
            page_dict[page_number] = page_value
            
    return [f"{base_url}&r={value}" for value in page_dict.values()]


def fetch_all_data():
    """
    Fetch and concatenate data from all pagination URLs.
    """
    all_data = pd.DataFrame()
    urls = get_pagination_urls(SCREENER_URL)
    
    for url in urls:
        html = fetch_data(url)
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', {'class': 'styled-table-new is-rounded is-tabular-nums w-full screener_table'})
        if table:
            df = extract_table_values(table)
            all_data = pd.concat([all_data, df], ignore_index=False)
    
    return all_data


def fetch_ticker_data(ticker, period="3mo"):
    """
    Fetch historical market data for a given ticker from Yahoo Finance.
    """
    ticker_data = yf.Ticker(ticker)
    hist = ticker_data.history(period=period)
    return hist


def fetch_sector_data():
    """
    Fetch sector performance data from Finviz.
    """
    html = fetch_data(SECTOR_URL)
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the JavaScript object containing the sector performance data
    script_tag = soup.find('script', string=re.compile(r'var rows ='))
    script_content = script_tag.string if script_tag else ""

    # Extract JSON-like data from the JavaScript object
    json_data_match = re.search(r'var rows = (\[.*\]);', script_content)
    if not json_data_match:
        return pd.DataFrame()  # Return an empty DataFrame if no data found
    
    json_data = json_data_match.group(1)

    # Convert the JSON-like string to a Python list of dictionaries
    sector_data = json.loads(json_data)
    sector_df = pd.DataFrame(sector_data)

    # Drop unnecessary columns
    sector_df = sector_df.drop(columns=['ticker', 'screenerUrl', 'group'], errors='ignore')
    
    return sector_df


def get_market_sentiment():
    """
    Fetch market sentiment from the Finviz homepage.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    url = "https://finviz.com/"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    sentiment_dict = {}

    market_divs = soup.find_all('div', {'data-boxover': True})
    
    for div in market_divs:
        if 'Advancing / Declining' in div.get('data-boxover', ''):
            advancing_div = div.find('div', {'class': 'market-stats_labels_left'})
            declining_div = div.find('div', {'class': 'market-stats_labels_right'})

            advancing_text = advancing_div.find_all('p')[1].text.strip()
            declining_text = declining_div.find_all('p')[1].text.strip()

            advancing_percentage, advancing_number = advancing_text.split()
            declining_number, declining_percentage = declining_text.split()

            advancing_number = advancing_number.strip("()")
            declining_number = declining_number.strip("()")
            advancing_percentage = advancing_percentage.strip("%")
            declining_percentage = declining_percentage.strip("%")
            
            sentiment_dict["advancing"] = advancing_percentage
            sentiment_dict["declining"] = declining_percentage

        elif 'New High / New Low' in div.get('data-boxover', ''):
            new_high_div = div.find('div', {'class': 'market-stats_labels_left'})
            new_low_div = div.find('div', {'class': 'market-stats_labels_right'})

            new_high_text = new_high_div.find_all('p')[1].text.strip()
            new_low_text = new_low_div.find_all('p')[1].text.strip()

            new_high_percentage, new_high_number = new_high_text.split()
            new_low_number, new_low_percentage = new_low_text.split()

            new_high_number = new_high_number.strip("()")
            new_low_number = new_low_number.strip("()")
            new_high_percentage = new_high_percentage.strip("%")
            new_low_percentage = new_low_percentage.strip("%")
            
            sentiment_dict["new_high"] = new_high_percentage
            sentiment_dict["new_low"] = new_low_percentage

        elif 'Above SMA50' in div.get('data-boxover', ''):
            above_sma50_div = div.find('div', {'class': 'market-stats_labels_left'})
            above_sma50_text = above_sma50_div.find_all('p')[1].text.strip()
            above_sma50_percentage, above_sma50_number = above_sma50_text.split()
            above_sma50_percentage = above_sma50_percentage.strip('%')
            above_sma50_number = above_sma50_number.strip('()')

            below_sma50_div = div.find('div', {'class': 'market-stats_labels_right'})
            below_sma50_text = below_sma50_div.find_all('p')[1].text.strip()
            below_sma50_number, below_sma50_percentage = below_sma50_text.split()
            below_sma50_percentage = below_sma50_percentage.strip('%')
            below_sma50_number = below_sma50_number.strip('()')

            sentiment_dict["above_sma50"] = above_sma50_percentage
            sentiment_dict["below_sma50"] = below_sma50_percentage

        elif 'Above SMA200' in div.get('data-boxover', ''):
            above_sma200_div = div.find('div', {'class': 'market-stats_labels_left'})
            above_sma200_text = above_sma200_div.find_all('p')[1].text.strip()
            above_sma200_percentage, above_sma200_number = above_sma200_text.split()
            above_sma200_percentage = above_sma200_percentage.strip('%')
            above_sma200_number = above_sma200_number.strip('()')

            below_sma200_div = div.find('div', {'class': 'market-stats_labels_right'})
            below_sma200_text = below_sma200_div.find_all('p')[1].text.strip()
            below_sma200_number, below_sma200_percentage = below_sma200_text.split()
            below_sma200_percentage = below_sma200_percentage.strip('%')
            below_sma200_number = below_sma200_number.strip('()')
            
            sentiment_dict["above_sma200"] = above_sma200_percentage
            sentiment_dict["below_sma200"] = below_sma200_percentage

    return sentiment_dict