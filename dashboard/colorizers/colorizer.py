# colorizers/colorizer.py

def colorize_bullish_count(val):
    if val > 0:
        return 'background-color: lightgreen'
    else:
        return 'background-color: lightcoral'

def colorize_bearish_count(val):
    if val > 0:
        return 'background-color: lightcoral'
    else:
        return 'background-color: lightgreen'

def colorize_performance(val, max_val, min_val):
    """
    Apply gradient color based on performance value.
    """
    try:
        val = float(val)
    except ValueError:
        return ''
    
    # Normalize value to a range [0, 1]
    norm_val = (val - min_val) / (max_val - min_val)
    norm_val = max(0, min(norm_val, 1))  # Ensure norm_val is within [0, 1]
    
    # Calculate the color intensity
    red_component = int((1 - norm_val) * 255)
    green_component = int(norm_val * 255)
    
    return f'background-color: rgba({red_component}, {green_component}, 0, 0.6)'
# Function to colorize based on sentiment values
# Define colorizer function
# Define colorizer function
def colorize_sentiment(value, metric):
    value = float(value)  # Convert to float for comparison
    if metric in ["advancing", "new_high", "above_sma50", "above_sma200"]:
        if value > 60:
            return 'background-color: lightgreen;'
        elif value < 40:
            return 'background-color: lightcoral;'
    elif metric in ["declining", "new_low", "below_sma50", "below_sma200"]:
        if value > 60:
            return 'background-color: lightcoral;'
        elif value < 40:
            return 'background-color: lightgreen;'
    return ''  # No color for neutral values

# Function to colorize based on sentiment values
# Define colorizer function
# Define colorizer function
def colorize_sentiment(value, metric):
    value = float(value)  # Convert to float for comparison
    if metric in ["advancing", "new_high", "above_sma50", "above_sma200"]:
        if value > 60:
            return 'background-color: lightgreen;'
        elif value < 40:
            return 'background-color: lightcoral;'
    elif metric in ["declining", "new_low", "below_sma50", "below_sma200"]:
        if value > 60:
            return 'background-color: lightcoral;'
        elif value < 40:
            return 'background-color: lightgreen;'
    return ''  # No color for neutral values

def colorize_macd_histogram(val):
    """
    Apply color based on MACD histogram value.
    """
    try:
        val = float(val)
    except ValueError:
        return ''
    
    if val > 0:
        # Gradient from light green to dark green
        color = f'rgba(0, {min(int(val * 255), 255)}, 0, 0.6)'
    elif val < 0:
        # Gradient from light red to dark red
        color = f'rgba({min(int(-val * 255), 255)}, 0, 0, 0.6)'
    else:
        # Neutral color for zero crossing
        color = 'gray'
    
    return f'background-color: {color}'


def colorize_rsi(val):
    """
    Apply gradient color based on RSI value.
    """
    try:
        val = float(val)
    except ValueError:
        return ''
    
    if val <= 30:
        # Green for oversold conditions (RSI <= 30)
        return 'background-color: rgba(0, 255, 0, 0.6)'  # Green
    elif 30 < val < 70:
        # Yellow to orange gradient for neutral conditions (RSI between 31 and 69)
        green_component = int(255 * (70 - val) / 40)  # Max green at RSI 31, min at 69
        red_component = 255 - green_component
        return f'background-color: rgba(255, {green_component}, 0, 0.6)'
    else:
        # Red for overbought conditions (RSI >= 70)
        return 'background-color: rgba(255, 0, 0, 0.6)'  # Red

# Custom max and min values for each time frame
max_daily_perf = 1   # max expected positive weekly performance in %
min_daily_perf = -1  # max expected negative weekly performance in %
max_weekly_perf = 1   # max expected positive weekly performance in %
min_weekly_perf = -1  # max expected negative weekly performance in %
max_monthly_perf = 3 # max expected positive monthly performance in %
min_monthly_perf = -3 # max expected negative monthly performance in %
max_quarterly_perf = 7 # max expected positive quarterly performance in %
min_quarterly_perf = -7 # max expected negative quarterly performance in %
max_lt_perf = 20 # max expected positive quarterly performance in %
min_lt_perf = -20 # max expected negative quarterly performance in %

# Define the colorization ranges for each column
color_ranges = {
    "operfT": (-1, 1),
    "perfW": (-3, 3),
    "perfM": (-5, 5),
    "perfQ": (-10, 10),
    "perfH": (-10, 10),
    "perfY": (-10, 10),
    "perfYtd": (-10, 10),
}
# Function to colorize each cell based on its value and the column range
def colorize_sector_table(val, col_name):
    min_val, max_val = color_ranges[col_name]
    try:
        val = float(val)
    except ValueError:
        return ''
    
    # Normalize value to a range [0, 1]
    norm_val = (val - min_val) / (max_val - min_val)
    norm_val = max(0, min(norm_val, 1))  # Ensure norm_val is within [0, 1]
    
    # Calculate the color intensity
    red_component = int((1 - norm_val) * 255)
    green_component = int(norm_val * 255)
    
    # Return the color as a hex string
    return f'background-color: rgb({red_component}, {green_component}, 0)'