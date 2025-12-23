"""
Momentum Trading Dashboard - Streamlit App

Features:
1. Ticker Analysis - Enter a ticker, see probability, target, stop, SHAP waterfall
2. Top Candidates - Browse highest probability stocks with filtering
3. Model Monitoring - Rolling AUC, feature importance stability, calibration
4. Alerts - Automated health checks with threshold-based warnings

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Momentum Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING (cached)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the production model."""
    model_file = Path('artifacts/models/production_model.pkl')
    metadata_file = Path('artifacts/models/model_metadata.json')

    if not model_file.exists():
        return None, None

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    return model, metadata


@st.cache_data(ttl=300)  # 5 minute cache
def load_features():
    """Load feature data."""
    features_file = Path('artifacts/features_daily.parquet')
    if not features_file.exists():
        return None
    return pd.read_parquet(features_file)


def get_data_info(features_df: pd.DataFrame, model_features: list) -> dict:
    """Get data date and missing feature information.

    Args:
        features_df: Feature DataFrame with 'date' column
        model_features: List of features the model expects

    Returns:
        Dictionary with data_date, days_stale, missing_features, available_features, symbols_count
    """
    info = {
        'data_date': None,
        'days_stale': 0,
        'missing_features': [],
        'available_features': [],
        'coverage_pct': 100.0,
        'symbols_count': 0,
    }

    if features_df is None:
        return info

    # Get latest date with VALID prediction data (non-null close prices)
    # This is what actually gets used for predictions
    if 'date' in features_df.columns and 'close' in features_df.columns:
        features_df = features_df.copy()
        features_df['date'] = pd.to_datetime(features_df['date'])

        # Filter to rows with valid close prices
        valid_data = features_df[features_df['close'].notna()]

        if len(valid_data) > 0:
            # Find the latest date that has substantial data (>100 symbols)
            date_counts = valid_data.groupby('date').size()
            substantial_dates = date_counts[date_counts > 100]

            if len(substantial_dates) > 0:
                max_date = substantial_dates.index.max()
                info['data_date'] = max_date
                info['symbols_count'] = int(substantial_dates.loc[max_date])

                # Calculate staleness (days since last data)
                today = pd.Timestamp.now().normalize()
                info['days_stale'] = (today - max_date).days

    # Check which model features are available
    data_columns = set(features_df.columns)
    info['available_features'] = [f for f in model_features if f in data_columns]
    info['missing_features'] = [f for f in model_features if f not in data_columns]

    if model_features:
        info['coverage_pct'] = len(info['available_features']) / len(model_features) * 100

    return info


@st.cache_data(ttl=300)
def load_targets():
    """Load target data."""
    targets_file = Path('artifacts/targets_triple_barrier.parquet')
    if not targets_file.exists():
        return None
    df = pd.read_parquet(targets_file)
    if 't0' in df.columns:
        df = df.rename(columns={'t0': 'date'})
    return df


@st.cache_data(ttl=300)
def load_predictions():
    """Load latest predictions."""
    pred_file = Path('artifacts/predictions/predictions_latest.parquet')
    if not pred_file.exists():
        return None
    return pd.read_parquet(pred_file)


@st.cache_data
def load_feature_importance():
    """Load feature importance."""
    imp_file = Path('artifacts/models/feature_importance.csv')
    if not imp_file.exists():
        return None
    return pd.read_csv(imp_file)


def get_shap_values(model, X, feature_names):
    """Calculate SHAP values for a single prediction."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, shap_values is a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        return shap_values, explainer.expected_value
    except ImportError:
        st.warning("SHAP not installed. Run: pip install shap")
        return None, None
    except Exception as e:
        st.error(f"SHAP error: {e}")
        return None, None


# =============================================================================
# TICKER ANALYSIS PAGE
# =============================================================================

def render_ticker_analysis():
    """Render the ticker analysis page."""
    st.header("🎯 Ticker Analysis")

    # Load model first (fast)
    model, metadata = load_model()

    if model is None:
        st.error("Model not found. Run `python run_training.py` first.")
        return

    feature_names = metadata.get('features', [])

    # Load heavy data with spinner
    with st.spinner("Loading feature data..."):
        features_df = load_features()
        predictions_df = load_predictions()

    if features_df is None:
        st.error("Features not found. Run feature computation first.")
        return

    # Get data info (date, missing features)
    data_info = get_data_info(features_df, feature_names)

    # Display data status bar
    if data_info['data_date']:
        date_str = data_info['data_date'].strftime('%Y-%m-%d')
        stale_days = data_info['days_stale']
        symbols_count = data_info['symbols_count']

        if stale_days <= 1:
            st.success(f"📅 **Data Date:** {date_str} ({symbols_count:,} symbols)")
        elif stale_days <= 3:
            st.warning(f"📅 **Data Date:** {date_str} ({stale_days} days old, {symbols_count:,} symbols)")
        else:
            st.error(f"📅 **Data Date:** {date_str} ({stale_days} days stale, {symbols_count:,} symbols) - run `python -m src.cli.compute` to refresh")

    # Show missing features warning if any
    if data_info['missing_features']:
        n_missing = len(data_info['missing_features'])
        with st.expander(f"⚠️ {n_missing} missing features ({data_info['coverage_pct']:.0f}% coverage)", expanded=False):
            st.write("The following features expected by the model are missing from the data:")
            st.code('\n'.join(data_info['missing_features']))
            st.caption("Missing features are filled with 0. Run feature pipeline to compute them.")

    # Get available symbols
    available_symbols = sorted(features_df['symbol'].unique())

    # Input section - just ticker (ATR multiples are fixed from training)
    ticker = st.text_input(
        "Enter Ticker Symbol",
        value="AAPL",
        max_chars=10,
        help="Enter a stock ticker symbol"
    ).upper().strip()

    # Fixed barrier parameters from training (displayed for reference)
    # These match the triple barrier targets the model was trained on
    UP_MULT = 3.0  # Target = entry + 3 * ATR
    DN_MULT = 1.5  # Stop = entry - 1.5 * ATR

    if ticker not in available_symbols:
        st.warning(f"Symbol '{ticker}' not found in data. Available: {len(available_symbols)} symbols")

        # Show suggestions
        matches = [s for s in available_symbols if ticker in s][:10]
        if matches:
            st.info(f"Did you mean: {', '.join(matches)}")
        return

    # Get latest data for ticker with valid close price
    ticker_data = features_df[features_df['symbol'] == ticker].copy()
    if 'date' in ticker_data.columns:
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        ticker_data = ticker_data.sort_values('date')

    # Filter to rows with valid close price
    valid_data = ticker_data[ticker_data['close'].notna()]
    if len(valid_data) == 0:
        st.error(f"No valid price data for {ticker}")
        return

    latest = valid_data.iloc[-1:].copy()
    latest_date = latest['date'].iloc[0] if 'date' in latest.columns else "N/A"

    # Check if data is stale
    max_date = ticker_data['date'].max() if 'date' in ticker_data.columns else None
    if max_date and latest_date < max_date:
        days_stale = (max_date - latest_date).days
        st.warning(f"⚠️ Data is {days_stale} days stale. Latest valid: {latest_date.date()}")

    # Calculate prediction
    available_features = [f for f in feature_names if f in latest.columns]
    X = latest[available_features].copy()
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)

    probability = model.predict_proba(X.values)[:, 1][0]

    # Calculate target and stop using fixed training parameters
    close_price = latest['close'].iloc[0]
    # atr_percent is already in decimal form (e.g., 0.02 = 2%)
    atr_pct = latest['atr_percent'].iloc[0] if 'atr_percent' in latest.columns and pd.notna(latest['atr_percent'].iloc[0]) else 0.02

    target_price = close_price * (1 + UP_MULT * atr_pct)
    stop_price = close_price * (1 - DN_MULT * atr_pct)
    target_pct = (target_price / close_price - 1) * 100
    stop_pct = (1 - stop_price / close_price) * 100
    reward_risk = target_pct / max(stop_pct, 0.1)

    # Break-even probability: minimum probability needed to be profitable
    # Formula: BE = Risk / (Reward + Risk) = 1 / (1 + R/R ratio)
    break_even_prob = 1 / (1 + reward_risk)

    # Display main metrics
    st.markdown("---")

    # Probability gauge
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        prob_color = "🟢" if probability > 0.6 else "🟡" if probability > 0.4 else "🔴"
        st.metric(
            label=f"{prob_color} Hit Probability",
            value=f"{probability:.1%}",
            delta=f"{'Bullish' if probability > 0.5 else 'Bearish'}"
        )

    with col2:
        st.metric(
            label="📈 Target Price",
            value=f"${target_price:.2f}",
            delta=f"+{target_pct:.1f}%"
        )

    with col3:
        st.metric(
            label="🛑 Stop Loss",
            value=f"${stop_price:.2f}",
            delta=f"-{stop_pct:.1f}%"
        )

    with col4:
        st.metric(
            label="⚖️ Reward/Risk",
            value=f"{reward_risk:.1f}x",
            delta=f"{'Good' if reward_risk > 2 else 'Fair' if reward_risk > 1.5 else 'Poor'}"
        )

    with col5:
        # Edge = probability - break_even (positive = profitable)
        edge = probability - break_even_prob
        edge_color = "🟢" if edge > 0.10 else "🟡" if edge > 0 else "🔴"
        st.metric(
            label=f"{edge_color} Break-Even",
            value=f"{break_even_prob:.1%}",
            delta=f"Edge: {edge:+.1%}",
            help="Minimum probability needed to break even. Edge = Predicted - Break-even."
        )

    # Price info
    st.markdown(f"**Current Price:** ${close_price:.2f} | **Date:** {latest_date} | **ATR%:** {atr_pct*100:.2f}%")
    st.caption(f"Barriers: Target = Entry + {UP_MULT}×ATR, Stop = Entry - {DN_MULT}×ATR (fixed from training)")

    # Multi-timeframe alignment indicator
    st.markdown("---")
    st.subheader("📊 Timeframe Alignment")

    # Define indicator pairs to compare (daily, weekly, interpretation)
    alignment_indicators = [
        ('rsi_14', 'w_rsi_14', 'RSI', 'momentum'),
        ('trend_score_sign', 'w_trend_score_sign', 'Trend', 'trend'),
        ('pct_slope_ma_20', 'w_pct_slope_ma_20', 'Slope 20', 'slope'),
        ('vol_regime', 'w_vol_regime', 'Vol Regime', 'volatility'),
        ('pct_dist_ma_20', 'w_pct_dist_ma_20', 'Dist MA20', 'distance'),
    ]

    def get_alignment_status(daily_val, weekly_val, indicator_type):
        """Determine alignment status between daily and weekly values."""
        if pd.isna(daily_val) or pd.isna(weekly_val):
            return "⚪", "N/A", "gray"

        if indicator_type == 'momentum':  # RSI: >50 bullish, <50 bearish
            d_bull = daily_val > 50
            w_bull = weekly_val > 50
        elif indicator_type == 'trend':  # Trend score: >0 bullish
            d_bull = daily_val > 0
            w_bull = weekly_val > 0
        elif indicator_type == 'slope':  # Slope: >0 bullish
            d_bull = daily_val > 0
            w_bull = weekly_val > 0
        elif indicator_type == 'volatility':  # Vol regime: <0.5 low vol (favorable)
            d_bull = daily_val < 0.5
            w_bull = weekly_val < 0.5
        elif indicator_type == 'distance':  # Distance: >0 above MA (bullish)
            d_bull = daily_val > 0
            w_bull = weekly_val > 0
        else:
            d_bull = daily_val > 0
            w_bull = weekly_val > 0

        if d_bull and w_bull:
            return "🟢", "Aligned ↑", "#00cc96"
        elif not d_bull and not w_bull:
            return "🔴", "Aligned ↓", "#ef553b"
        else:
            return "🟡", "Mixed", "#ffa500"

    # Build alignment display
    alignment_cols = st.columns(len(alignment_indicators))

    aligned_bullish = 0
    aligned_bearish = 0
    mixed = 0

    for i, (d_col, w_col, label, ind_type) in enumerate(alignment_indicators):
        d_val = latest[d_col].iloc[0] if d_col in latest.columns else np.nan
        w_val = latest[w_col].iloc[0] if w_col in latest.columns else np.nan

        icon, status, color = get_alignment_status(d_val, w_val, ind_type)

        # Count for summary
        if status == "Aligned ↑":
            aligned_bullish += 1
        elif status == "Aligned ↓":
            aligned_bearish += 1
        elif status == "Mixed":
            mixed += 1

        with alignment_cols[i]:
            # Format values based on type
            if ind_type == 'momentum':
                d_str = f"{d_val:.0f}" if pd.notna(d_val) else "—"
                w_str = f"{w_val:.0f}" if pd.notna(w_val) else "—"
            elif ind_type in ('slope', 'distance'):
                d_str = f"{d_val:+.1%}" if pd.notna(d_val) else "—"
                w_str = f"{w_val:+.1%}" if pd.notna(w_val) else "—"
            elif ind_type == 'volatility':
                d_str = f"{d_val:.2f}" if pd.notna(d_val) else "—"
                w_str = f"{w_val:.2f}" if pd.notna(w_val) else "—"
            else:
                d_str = f"{d_val:+.2f}" if pd.notna(d_val) else "—"
                w_str = f"{w_val:+.2f}" if pd.notna(w_val) else "—"

            st.markdown(f"""
            <div style="text-align: center; padding: 8px; border-radius: 8px; background-color: {color}22; border: 1px solid {color};">
                <div style="font-size: 1.5em;">{icon}</div>
                <div style="font-weight: bold; color: {color};">{label}</div>
                <div style="font-size: 0.85em; color: #666;">D: {d_str} | W: {w_str}</div>
                <div style="font-size: 0.75em; color: {color};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    # Overall alignment summary
    total_indicators = aligned_bullish + aligned_bearish + mixed
    if total_indicators > 0:
        if aligned_bullish >= 4:
            overall = "🟢 **Strong Bullish Alignment** - Daily and Weekly confirm uptrend"
        elif aligned_bearish >= 4:
            overall = "🔴 **Strong Bearish Alignment** - Daily and Weekly confirm downtrend"
        elif aligned_bullish >= 3:
            overall = "🟢 **Bullish Bias** - Majority of indicators aligned up"
        elif aligned_bearish >= 3:
            overall = "🔴 **Bearish Bias** - Majority of indicators aligned down"
        elif mixed >= 3:
            overall = "🟡 **Mixed Signals** - Daily and Weekly diverging, proceed with caution"
        else:
            overall = "🟡 **Neutral** - No clear alignment"

        st.markdown(f"\n{overall}")

    # Position Planner
    st.markdown("---")
    st.subheader("💰 Position Planner")

    # Get settings from session state
    account_size = st.session_state.get('account_size', 100000)
    max_risk_pct = st.session_state.get('max_risk_pct', 2.0) / 100
    num_entries = st.session_state.get('num_entries', 3)

    # Calculate Kelly-based position sizing
    # Kelly % = Edge / (Risk per unit)
    # Half-Kelly is more conservative and commonly used
    if edge > 0:
        kelly_pct = edge / (stop_pct / 100) if stop_pct > 0 else 0
        half_kelly_pct = kelly_pct / 2
    else:
        kelly_pct = 0
        half_kelly_pct = 0

    # Cap at max risk
    suggested_risk_pct = min(half_kelly_pct, max_risk_pct)

    # Calculate position size based on risk
    max_risk_dollars = account_size * max_risk_pct
    risk_per_share = close_price * (stop_pct / 100)
    max_shares = int(max_risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    max_position_value = max_shares * close_price

    # Define entry levels based on MA distances
    ma20_val = latest['ma_20'].iloc[0] if 'ma_20' in latest.columns else close_price * 0.98
    ma50_val = latest['ma_50'].iloc[0] if 'ma_50' in latest.columns else close_price * 0.95

    # Entry allocation based on number of entries and alignment
    if num_entries == 1:
        entry_allocations = [1.0]
        entry_prices = [close_price]
        entry_labels = ["Current"]
    elif num_entries == 2:
        # If bullish aligned, more at current; if mixed, more at pullback
        if aligned_bullish >= 3:
            entry_allocations = [0.60, 0.40]
        else:
            entry_allocations = [0.40, 0.60]
        entry_prices = [close_price, ma20_val]
        entry_labels = ["Current", "MA20 Pullback"]
    else:  # 3 entries
        if aligned_bullish >= 4:
            entry_allocations = [0.50, 0.30, 0.20]
        elif aligned_bullish >= 3:
            entry_allocations = [0.40, 0.35, 0.25]
        else:
            entry_allocations = [0.30, 0.40, 0.30]
        entry_prices = [close_price, ma20_val, ma50_val]
        entry_labels = ["Current", "MA20 Pullback", "MA50 Pullback"]

    # Display position sizing summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kelly_color = "🟢" if half_kelly_pct > max_risk_pct else "🟡" if half_kelly_pct > 0 else "🔴"
        st.metric(
            label=f"{kelly_color} Half-Kelly",
            value=f"{half_kelly_pct:.1%}",
            delta=f"Full: {kelly_pct:.1%}",
            help="Kelly Criterion suggests optimal bet size. Half-Kelly reduces variance."
        )

    with col2:
        st.metric(
            label="💵 Max Risk",
            value=f"${max_risk_dollars:,.0f}",
            delta=f"{max_risk_pct:.1%} of account",
            help="Maximum dollar amount at risk based on your settings"
        )

    with col3:
        st.metric(
            label="📊 Max Shares",
            value=f"{max_shares:,}",
            delta=f"${max_position_value:,.0f} value",
            help="Maximum shares based on stop loss distance"
        )

    with col4:
        # Confidence level based on edge and alignment
        if edge > 0.10 and aligned_bullish >= 4:
            confidence = "High"
            conf_color = "🟢"
        elif edge > 0.05 and aligned_bullish >= 3:
            confidence = "Medium"
            conf_color = "🟡"
        elif edge > 0:
            confidence = "Low"
            conf_color = "🟠"
        else:
            confidence = "None"
            conf_color = "🔴"
        st.metric(
            label=f"{conf_color} Confidence",
            value=confidence,
            delta=f"Edge: {edge:+.1%}",
            help="Based on model edge and timeframe alignment"
        )

    # Entry plan table
    st.markdown("#### Entry Plan")

    entry_data = []
    total_shares = 0
    total_cost = 0

    for i, (label, price, alloc) in enumerate(zip(entry_labels, entry_prices, entry_allocations)):
        shares_at_level = int(max_shares * alloc)
        cost_at_level = shares_at_level * price
        pct_from_current = (price / close_price - 1) * 100
        risk_at_level = shares_at_level * risk_per_share

        total_shares += shares_at_level
        total_cost += cost_at_level

        entry_data.append({
            "Level": f"{i+1}. {label}",
            "Price": f"${price:.2f}",
            "vs Current": f"{pct_from_current:+.1f}%",
            "Allocation": f"{alloc:.0%}",
            "Shares": f"{shares_at_level:,}",
            "Cost": f"${cost_at_level:,.0f}",
            "Risk $": f"${risk_at_level:,.0f}",
        })

    # Add totals row
    avg_entry = total_cost / total_shares if total_shares > 0 else close_price
    total_risk = total_shares * risk_per_share

    entry_data.append({
        "Level": "**TOTAL**",
        "Price": f"**${avg_entry:.2f}**",
        "vs Current": f"**{(avg_entry/close_price-1)*100:+.1f}%**",
        "Allocation": "**100%**",
        "Shares": f"**{total_shares:,}**",
        "Cost": f"**${total_cost:,.0f}**",
        "Risk $": f"**${total_risk:,.0f}**",
    })

    entry_df = pd.DataFrame(entry_data)
    st.dataframe(entry_df, use_container_width=True, hide_index=True)

    # Effective R/R if all entries fill
    if total_shares > 0:
        effective_target = avg_entry * (1 + UP_MULT * atr_pct)
        effective_stop = avg_entry * (1 - DN_MULT * atr_pct)
        effective_target_pct = (effective_target / avg_entry - 1) * 100
        effective_stop_pct = (1 - effective_stop / avg_entry) * 100
        effective_rr = effective_target_pct / effective_stop_pct if effective_stop_pct > 0 else 0

        st.markdown(f"""
        **If all entries fill:**
        Avg Entry: ${avg_entry:.2f} | Target: ${effective_target:.2f} (+{effective_target_pct:.1f}%) |
        Stop: ${effective_stop:.2f} (-{effective_stop_pct:.1f}%) | R/R: {effective_rr:.1f}x
        """)

    # Warning if negative edge
    if edge <= 0:
        st.warning("⚠️ **Negative or zero edge** - Model suggests this trade may not be profitable. Consider passing or waiting for better setup.")

    # SHAP Waterfall
    st.markdown("---")
    st.subheader("🔍 Feature Contribution (SHAP)")

    with st.spinner("Calculating SHAP values..."):
        shap_values, expected_value = get_shap_values(model, X.values, feature_names)

    if shap_values is not None:
        # Handle expected_value being a list
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        # Create waterfall data
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[0],
            'feature_value': X.values[0]
        })
        shap_df['abs_shap'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=False)

        # Top N features
        top_n = st.slider("Show top N features", 5, 30, 15)
        top_shap = shap_df.head(top_n).copy()

        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(top_shap),
            y=top_shap['feature'].tolist()[::-1],
            x=top_shap['shap_value'].tolist()[::-1],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef553b"}},
            increasing={"marker": {"color": "#00cc96"}},
            text=[f"{v:.3f}" for v in top_shap['shap_value'].tolist()[::-1]],
            textposition="outside"
        ))

        fig.update_layout(
            title=f"Feature Contributions to Prediction (Base: {base_value:.3f})",
            xaxis_title="SHAP Value (impact on probability)",
            yaxis_title="Feature",
            height=max(400, top_n * 25),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature value table
        with st.expander("📊 Feature Values"):
            display_df = top_shap[['feature', 'feature_value', 'shap_value']].copy()
            display_df.columns = ['Feature', 'Value', 'SHAP Impact']
            st.dataframe(display_df, use_container_width=True)

    # Historical chart
    st.markdown("---")
    st.subheader("📈 Price History")

    if len(ticker_data) > 0 and 'close' in ticker_data.columns:
        # Price chart with target/stop lines
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ticker_data['date'],
            y=ticker_data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add target line
        fig.add_hline(y=target_price, line_dash="dash", line_color="green",
                      annotation_text=f"Target: ${target_price:.2f}")

        # Add stop line
        fig.add_hline(y=stop_price, line_dash="dash", line_color="red",
                      annotation_text=f"Stop: ${stop_price:.2f}")

        # Add current price line
        fig.add_hline(y=close_price, line_dash="dot", line_color="gray",
                      annotation_text=f"Current: ${close_price:.2f}")

        fig.update_layout(
            title=f"{ticker} - Last {len(ticker_data)} Days",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TOP CANDIDATES PAGE
# =============================================================================

def render_top_candidates():
    """Render the top candidates page."""
    # Load model first (fast)
    model, metadata = load_model()

    if model is None:
        st.header("🏆 Top Momentum Candidates")
        st.error("Model not found. Run training first.")
        return

    feature_names = metadata.get('features', [])

    # Load heavy data with spinner
    with st.spinner("Loading data..."):
        features_df = load_features()
        predictions_df = load_predictions()

    if features_df is None:
        st.header("🏆 Top Momentum Candidates")
        st.error("Features not found. Run feature computation first.")
        return

    # Get data info (date, missing features)
    data_info = get_data_info(features_df, feature_names)

    # Display header with entry date
    if data_info['data_date']:
        entry_date_str = data_info['data_date'].strftime('%Y-%m-%d')
        st.header(f"🏆 Top Momentum Candidates — Entry: {entry_date_str}")
    else:
        st.header("🏆 Top Momentum Candidates")

    # Display data status bar (compact version for this page)
    col_date, col_missing = st.columns([2, 3])
    with col_date:
        if data_info['data_date']:
            date_str = data_info['data_date'].strftime('%Y-%m-%d')
            stale_days = data_info['days_stale']
            symbols_count = data_info['symbols_count']
            if stale_days <= 1:
                st.success(f"📅 Data: {date_str} ({symbols_count:,} symbols)")
            elif stale_days <= 3:
                st.warning(f"📅 Data: {date_str} ({stale_days}d old, {symbols_count:,} symbols)")
            else:
                st.error(f"📅 Data: {date_str} ({stale_days}d stale, {symbols_count:,} symbols)")

    with col_missing:
        if data_info['missing_features']:
            n_missing = len(data_info['missing_features'])
            with st.expander(f"⚠️ {n_missing} missing features"):
                st.code('\n'.join(data_info['missing_features']))

    # Generate predictions if not available
    if predictions_df is None:
        st.info("Generating predictions...")

        # Get latest date
        features_df['date'] = pd.to_datetime(features_df['date'])
        latest_date = features_df['date'].max()
        latest_df = features_df[features_df['date'] == latest_date].copy()

        # Prepare features
        available_features = [f for f in feature_names if f in latest_df.columns]
        X = latest_df[available_features].copy()
        for f in feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)

        # Predict
        probabilities = model.predict_proba(X.values)[:, 1]
        latest_df['probability'] = probabilities

        # Calculate targets/stops
        atr_pct = latest_df.get('atr_percent', 2.0) / 100
        latest_df['target_price'] = latest_df['close'] * (1 + 3.0 * atr_pct)
        latest_df['stop_price'] = latest_df['close'] * (1 - 1.5 * atr_pct)
        latest_df['target_pct'] = (latest_df['target_price'] / latest_df['close'] - 1) * 100
        latest_df['stop_pct'] = (1 - latest_df['stop_price'] / latest_df['close']) * 100
        latest_df['reward_risk'] = latest_df['target_pct'] / latest_df['stop_pct'].clip(lower=0.1)

        predictions_df = latest_df

    # Calculate break-even and edge for all predictions
    predictions_df['break_even'] = 1 / (1 + predictions_df['reward_risk'])
    predictions_df['edge'] = predictions_df['probability'] - predictions_df['break_even']

    # Filters
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        min_edge = st.slider("Min Edge", -0.2, 0.5, 0.0, 0.05,
                             help="Edge = Probability - Break-even. Positive edge = expected profit.")

    with col2:
        min_target_pct = st.slider("Min Target %", 0.0, 20.0, 5.0, 1.0,
                                   help="Minimum target return percentage")

    with col3:
        min_rr = st.slider("Min Reward/Risk", 0.0, 5.0, 1.5, 0.25)

    with col4:
        top_n = st.slider("Show Top N", 10, 200, 50, 10)

    with col5:
        sort_by = st.selectbox("Sort By", ["edge", "probability", "reward_risk", "target_pct"])

    # Filter data
    filtered = predictions_df[
        (predictions_df['edge'] >= min_edge) &
        (predictions_df['target_pct'] >= min_target_pct) &
        (predictions_df.get('reward_risk', 999) >= min_rr)
    ].copy()

    filtered = filtered.nlargest(top_n, sort_by)

    st.markdown(f"**Showing {len(filtered)} candidates** (filtered from {len(predictions_df)} total)")

    # Display table
    display_cols = ['symbol', 'probability', 'break_even', 'edge', 'close', 'target_price', 'stop_price',
                    'target_pct', 'stop_pct', 'reward_risk']
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Add sector if available
    if 'sector' in filtered.columns:
        display_cols.insert(1, 'sector')

    display_df = filtered[display_cols].copy()

    # Format columns
    if 'probability' in display_df.columns:
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
    if 'break_even' in display_df.columns:
        display_df['break_even'] = display_df['break_even'].apply(lambda x: f"{x:.1%}")
    if 'edge' in display_df.columns:
        display_df['edge'] = display_df['edge'].apply(lambda x: f"{x:+.1%}")
    if 'close' in display_df.columns:
        display_df['close'] = display_df['close'].apply(lambda x: f"${x:.2f}")
    if 'target_price' in display_df.columns:
        display_df['target_price'] = display_df['target_price'].apply(lambda x: f"${x:.2f}")
    if 'stop_price' in display_df.columns:
        display_df['stop_price'] = display_df['stop_price'].apply(lambda x: f"${x:.2f}")
    if 'target_pct' in display_df.columns:
        display_df['target_pct'] = display_df['target_pct'].apply(lambda x: f"+{x:.1f}%")
    if 'stop_pct' in display_df.columns:
        display_df['stop_pct'] = display_df['stop_pct'].apply(lambda x: f"-{x:.1f}%")
    if 'reward_risk' in display_df.columns:
        display_df['reward_risk'] = display_df['reward_risk'].apply(lambda x: f"{x:.1f}x")

    st.dataframe(display_df, use_container_width=True, height=600)

    # Distribution plot
    st.markdown("---")
    st.subheader("📊 Edge Distribution")

    fig = px.histogram(
        predictions_df,
        x='edge',
        nbins=50,
        title="Distribution of Edge (Probability - Break-even) Across Universe"
    )
    fig.add_vline(x=min_edge, line_dash="dash", line_color="red",
                  annotation_text=f"Filter: {min_edge:+.0%}")
    fig.add_vline(x=0, line_dash="dot", line_color="gray",
                  annotation_text="Break-even")
    fig.update_layout(xaxis_title="Edge (Probability - Break-even)", yaxis_title="Count")

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MODEL MONITORING PAGE
# =============================================================================

def render_monitoring():
    """Render the model monitoring page."""
    st.header("📊 Model Monitoring")

    # Load model first (fast)
    model, metadata = load_model()
    feature_importance = load_feature_importance()

    if model is None:
        st.error("Model not found.")
        return

    # Model info (show immediately, before heavy data load)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Features", metadata.get('n_features', 'N/A'))
    with col2:
        st.metric("Training AUC", f"{metadata.get('train_auc', 0):.4f}")
    with col3:
        training_date = metadata.get('training_date', 'N/A')
        if training_date != 'N/A':
            training_date = training_date[:10]
        st.metric("Last Trained", training_date)

    # Feature importance
    st.markdown("---")
    st.subheader("🔧 Feature Importance")

    if feature_importance is not None:
        top_n = st.slider("Show top N features", 10, 52, 20, key="imp_slider")
        top_features = feature_importance.head(top_n)

        fig = px.bar(
            top_features,
            x='importance_pct',
            y='feature',
            orientation='h',
            title=f"Top {top_n} Features by Importance"
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Importance (%)",
            yaxis_title="Feature",
            height=max(400, top_n * 20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Historical performance (if we have predictions with outcomes)
    st.markdown("---")
    st.subheader("📈 Historical Performance")

    # Load heavy data for historical analysis with spinner
    with st.spinner("Loading historical data (cached after first load)..."):
        features_df = load_features()
        targets_df = load_targets()

    if targets_df is not None and features_df is not None:
        feature_names = metadata.get('features', [])

        # Merge features with targets for backtesting
        targets_df['date'] = pd.to_datetime(targets_df['date'])
        features_df['date'] = pd.to_datetime(features_df['date'])

        merged = features_df.merge(
            targets_df[['symbol', 'date', 'hit']],
            on=['symbol', 'date'],
            how='inner'
        )

        # Exclude neutral outcomes
        merged = merged[merged['hit'] != 0].copy()
        merged['target'] = (merged['hit'] == 1).astype(int)

        if len(merged) > 0:
            # Calculate rolling AUC by date
            merged = merged.sort_values('date')

            # Group by date and calculate daily predictions
            available_features = [f for f in feature_names if f in merged.columns]

            if len(available_features) > 0:
                X = merged[available_features].fillna(0).replace([np.inf, -np.inf], 0)

                # Add missing features
                for f in feature_names:
                    if f not in X.columns:
                        X[f] = 0
                X = X[feature_names]

                merged['pred_prob'] = model.predict_proba(X.values)[:, 1]

                # Rolling AUC
                from sklearn.metrics import roc_auc_score

                dates = merged['date'].unique()
                dates = sorted(dates)

                rolling_window = 63  # ~3 months
                rolling_metrics = []

                for i, end_date in enumerate(dates):
                    if i < rolling_window:
                        continue

                    start_date = dates[i - rolling_window]
                    window_data = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]

                    if len(window_data) > 100 and window_data['target'].nunique() == 2:
                        try:
                            auc = roc_auc_score(window_data['target'], window_data['pred_prob'])
                            rolling_metrics.append({
                                'date': end_date,
                                'auc': auc,
                                'n_samples': len(window_data)
                            })
                        except:
                            pass

                if rolling_metrics:
                    metrics_df = pd.DataFrame(rolling_metrics)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metrics_df['date'],
                        y=metrics_df['auc'],
                        mode='lines',
                        name='Rolling AUC (63-day)',
                        line=dict(color='#1f77b4', width=2)
                    ))

                    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                                  annotation_text="Random (0.5)")
                    fig.add_hline(y=metadata.get('train_auc', 0.69), line_dash="dash",
                                  line_color="green", annotation_text="Training AUC")

                    fig.update_layout(
                        title="Rolling 63-Day AUC",
                        xaxis_title="Date",
                        yaxis_title="AUC",
                        height=400,
                        yaxis_range=[0.4, 0.8]
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current AUC", f"{metrics_df['auc'].iloc[-1]:.4f}")
                    with col2:
                        st.metric("Mean AUC", f"{metrics_df['auc'].mean():.4f}")
                    with col3:
                        st.metric("Min AUC", f"{metrics_df['auc'].min():.4f}")

    # Calibration
    st.markdown("---")
    st.subheader("🎯 Prediction Calibration")

    st.info("Calibration analysis requires historical predictions with realized outcomes. "
            "Generate predictions daily and track outcomes to populate this section.")


# =============================================================================
# ALERTS PAGE
# =============================================================================

# Alert thresholds (configurable)
ALERT_THRESHOLDS = {
    'auc_critical': 0.55,           # AUC below this is critical
    'auc_warning': 0.60,            # AUC below this is warning
    'auc_consecutive_days': 5,      # Days below threshold to trigger
    'importance_correlation': 0.80,  # Feature importance correlation threshold
    'prediction_shift_std': 2.0,    # Std deviations for prediction distribution shift
    'model_age_warning_days': 14,   # Days since training for warning
    'model_age_critical_days': 30,  # Days since training for critical
}


def check_model_health(metadata, rolling_metrics_df=None, baseline_importance=None, current_importance=None, predictions_df=None):
    """
    Check model health and return list of alerts.

    Returns list of dicts: {'level': 'critical'|'warning'|'info', 'title': str, 'message': str, 'metric': value}
    """
    alerts = []

    # 1. Check model age
    training_date_str = metadata.get('training_date')
    if training_date_str:
        try:
            training_date = datetime.fromisoformat(training_date_str.replace('Z', '+00:00'))
            if training_date.tzinfo:
                training_date = training_date.replace(tzinfo=None)
            days_since_training = (datetime.now() - training_date).days

            if days_since_training >= ALERT_THRESHOLDS['model_age_critical_days']:
                alerts.append({
                    'level': 'critical',
                    'title': 'Model Stale',
                    'message': f'Model was trained {days_since_training} days ago. Retrain immediately.',
                    'metric': days_since_training,
                    'threshold': ALERT_THRESHOLDS['model_age_critical_days']
                })
            elif days_since_training >= ALERT_THRESHOLDS['model_age_warning_days']:
                alerts.append({
                    'level': 'warning',
                    'title': 'Model Aging',
                    'message': f'Model was trained {days_since_training} days ago. Consider retraining soon.',
                    'metric': days_since_training,
                    'threshold': ALERT_THRESHOLDS['model_age_warning_days']
                })
            else:
                alerts.append({
                    'level': 'ok',
                    'title': 'Model Fresh',
                    'message': f'Model trained {days_since_training} days ago.',
                    'metric': days_since_training,
                    'threshold': ALERT_THRESHOLDS['model_age_warning_days']
                })
        except:
            alerts.append({
                'level': 'warning',
                'title': 'Unknown Training Date',
                'message': 'Could not parse model training date.',
                'metric': None,
                'threshold': None
            })

    # 2. Check rolling AUC
    if rolling_metrics_df is not None and len(rolling_metrics_df) > 0:
        recent_auc = rolling_metrics_df.tail(ALERT_THRESHOLDS['auc_consecutive_days'])

        if len(recent_auc) >= ALERT_THRESHOLDS['auc_consecutive_days']:
            # Check for consecutive days below critical threshold
            critical_days = (recent_auc['auc'] < ALERT_THRESHOLDS['auc_critical']).sum()
            warning_days = (recent_auc['auc'] < ALERT_THRESHOLDS['auc_warning']).sum()
            current_auc = recent_auc['auc'].iloc[-1]

            if critical_days >= ALERT_THRESHOLDS['auc_consecutive_days']:
                alerts.append({
                    'level': 'critical',
                    'title': 'AUC Critical',
                    'message': f'Rolling AUC has been below {ALERT_THRESHOLDS["auc_critical"]} for {critical_days} consecutive days. Current: {current_auc:.4f}',
                    'metric': current_auc,
                    'threshold': ALERT_THRESHOLDS['auc_critical']
                })
            elif warning_days >= ALERT_THRESHOLDS['auc_consecutive_days']:
                alerts.append({
                    'level': 'warning',
                    'title': 'AUC Degraded',
                    'message': f'Rolling AUC below {ALERT_THRESHOLDS["auc_warning"]} for {warning_days} days. Current: {current_auc:.4f}',
                    'metric': current_auc,
                    'threshold': ALERT_THRESHOLDS['auc_warning']
                })
            elif current_auc < ALERT_THRESHOLDS['auc_warning']:
                alerts.append({
                    'level': 'warning',
                    'title': 'AUC Below Target',
                    'message': f'Current rolling AUC ({current_auc:.4f}) is below target ({ALERT_THRESHOLDS["auc_warning"]}).',
                    'metric': current_auc,
                    'threshold': ALERT_THRESHOLDS['auc_warning']
                })
            else:
                alerts.append({
                    'level': 'ok',
                    'title': 'AUC Healthy',
                    'message': f'Rolling AUC: {current_auc:.4f}',
                    'metric': current_auc,
                    'threshold': ALERT_THRESHOLDS['auc_warning']
                })

    # 3. Check feature importance stability
    if baseline_importance is not None and current_importance is not None:
        try:
            # Merge on feature name and calculate correlation
            merged = baseline_importance.merge(
                current_importance,
                on='feature',
                suffixes=('_baseline', '_current')
            )
            if len(merged) > 5:
                correlation = merged['importance_baseline'].corr(merged['importance_current'])

                if correlation < ALERT_THRESHOLDS['importance_correlation']:
                    alerts.append({
                        'level': 'warning',
                        'title': 'Feature Importance Drift',
                        'message': f'Feature importance correlation ({correlation:.3f}) below threshold ({ALERT_THRESHOLDS["importance_correlation"]}). Model behavior may have changed.',
                        'metric': correlation,
                        'threshold': ALERT_THRESHOLDS['importance_correlation']
                    })
                else:
                    alerts.append({
                        'level': 'ok',
                        'title': 'Feature Importance Stable',
                        'message': f'Correlation with baseline: {correlation:.3f}',
                        'metric': correlation,
                        'threshold': ALERT_THRESHOLDS['importance_correlation']
                    })
        except Exception as e:
            pass

    # 4. Check prediction distribution
    if predictions_df is not None and 'probability' in predictions_df.columns:
        current_mean = predictions_df['probability'].mean()
        current_std = predictions_df['probability'].std()

        # Expected distribution (from training or historical)
        expected_mean = 0.5  # Roughly balanced predictions expected
        expected_std = 0.15  # Typical spread

        # Check for distribution shift
        mean_shift = abs(current_mean - expected_mean) / expected_std

        if mean_shift > ALERT_THRESHOLDS['prediction_shift_std']:
            direction = "bullish" if current_mean > expected_mean else "bearish"
            alerts.append({
                'level': 'warning',
                'title': 'Prediction Distribution Shift',
                'message': f'Predictions are unusually {direction}. Mean: {current_mean:.3f} (expected ~{expected_mean:.3f})',
                'metric': current_mean,
                'threshold': expected_mean
            })
        else:
            alerts.append({
                'level': 'ok',
                'title': 'Predictions Normal',
                'message': f'Mean probability: {current_mean:.3f}, Std: {current_std:.3f}',
                'metric': current_mean,
                'threshold': expected_mean
            })

        # Check for extreme concentration
        high_conviction = (predictions_df['probability'] > 0.7).mean()
        if high_conviction > 0.3:  # More than 30% very bullish
            alerts.append({
                'level': 'info',
                'title': 'High Conviction Concentration',
                'message': f'{high_conviction:.1%} of predictions above 70% probability.',
                'metric': high_conviction,
                'threshold': 0.3
            })

    return alerts


@st.cache_data(ttl=3600)  # 1 hour cache for heavy computation
def compute_rolling_auc_metrics(_model, features_df, targets_df, feature_names):
    """Compute rolling AUC metrics (cached to avoid blocking UI)."""
    from sklearn.metrics import roc_auc_score

    if targets_df is None or features_df is None:
        return None

    targets_df = targets_df.copy()
    features_df = features_df.copy()
    targets_df['date'] = pd.to_datetime(targets_df['date'])
    features_df['date'] = pd.to_datetime(features_df['date'])

    merged = features_df.merge(
        targets_df[['symbol', 'date', 'hit']],
        on=['symbol', 'date'],
        how='inner'
    )
    merged = merged[merged['hit'] != 0].copy()

    if len(merged) == 0:
        return None

    merged['target'] = (merged['hit'] == 1).astype(int)
    merged = merged.sort_values('date')

    available_features = [f for f in feature_names if f in merged.columns]
    if len(available_features) == 0:
        return None

    X = merged[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[feature_names]

    merged['pred_prob'] = _model.predict_proba(X.values)[:, 1]

    dates = sorted(merged['date'].unique())
    rolling_window = 63
    rolling_metrics = []

    for i, end_date in enumerate(dates):
        if i < rolling_window:
            continue
        start_date = dates[i - rolling_window]
        window_data = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]

        if len(window_data) > 100 and window_data['target'].nunique() == 2:
            try:
                auc = roc_auc_score(window_data['target'], window_data['pred_prob'])
                rolling_metrics.append({'date': end_date, 'auc': auc})
            except:
                pass

    if rolling_metrics:
        return pd.DataFrame(rolling_metrics)
    return None


def render_alerts():
    """Render the alerts page."""
    st.header("🚨 Model Health Alerts")

    st.markdown("""
    This page monitors model health and generates alerts when thresholds are breached.
    Check this page regularly or before making trading decisions.
    """)

    # Load all required data
    model, metadata = load_model()
    features_df = load_features()
    targets_df = load_targets()
    predictions_df = load_predictions()
    feature_importance = load_feature_importance()

    if model is None:
        st.error("Model not found. Run `python run_training.py` first.")
        return

    # Calculate rolling metrics for AUC check (cached)
    feature_names = metadata.get('features', [])

    with st.spinner("Computing rolling AUC metrics (cached after first load)..."):
        rolling_metrics_df = compute_rolling_auc_metrics(
            model, features_df, targets_df, feature_names
        )

    # Run health checks
    alerts = check_model_health(
        metadata=metadata,
        rolling_metrics_df=rolling_metrics_df,
        baseline_importance=feature_importance,  # Using current as baseline for now
        current_importance=feature_importance,
        predictions_df=predictions_df
    )

    # Display alert summary
    st.markdown("---")

    critical_count = sum(1 for a in alerts if a['level'] == 'critical')
    warning_count = sum(1 for a in alerts if a['level'] == 'warning')
    ok_count = sum(1 for a in alerts if a['level'] == 'ok')

    # Overall status
    if critical_count > 0:
        st.error(f"🔴 **CRITICAL**: {critical_count} critical issue(s) detected")
    elif warning_count > 0:
        st.warning(f"🟡 **WARNING**: {warning_count} warning(s) detected")
    else:
        st.success(f"🟢 **HEALTHY**: All checks passed")

    # Status summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Critical", critical_count, delta=None)
    with col2:
        st.metric("Warnings", warning_count, delta=None)
    with col3:
        st.metric("OK", ok_count, delta=None)
    with col4:
        st.metric("Total Checks", len(alerts), delta=None)

    # Display individual alerts
    st.markdown("---")
    st.subheader("Alert Details")

    # Critical alerts first
    for alert in sorted(alerts, key=lambda x: {'critical': 0, 'warning': 1, 'info': 2, 'ok': 3}[x['level']]):
        if alert['level'] == 'critical':
            with st.container():
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #ff4b4b; background-color: #ffebeb; margin: 5px 0;">
                    <strong>🔴 CRITICAL: {alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        elif alert['level'] == 'warning':
            with st.container():
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #ffa500; background-color: #fff4e5; margin: 5px 0;">
                    <strong>🟡 WARNING: {alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        elif alert['level'] == 'info':
            with st.container():
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #0068c9; background-color: #e6f3ff; margin: 5px 0;">
                    <strong>ℹ️ INFO: {alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        else:  # ok
            with st.container():
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid #00cc96; background-color: #e6fff5; margin: 5px 0;">
                    <strong>✅ OK: {alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)

    # Threshold configuration (collapsible)
    st.markdown("---")
    with st.expander("⚙️ Alert Thresholds Configuration"):
        st.markdown("Current thresholds (edit in code to change):")

        threshold_df = pd.DataFrame([
            {"Threshold": "AUC Critical", "Value": ALERT_THRESHOLDS['auc_critical'], "Description": "AUC below this triggers critical alert"},
            {"Threshold": "AUC Warning", "Value": ALERT_THRESHOLDS['auc_warning'], "Description": "AUC below this triggers warning"},
            {"Threshold": "Consecutive Days", "Value": ALERT_THRESHOLDS['auc_consecutive_days'], "Description": "Days below threshold before alert"},
            {"Threshold": "Importance Correlation", "Value": ALERT_THRESHOLDS['importance_correlation'], "Description": "Feature importance correlation threshold"},
            {"Threshold": "Prediction Shift (Std)", "Value": ALERT_THRESHOLDS['prediction_shift_std'], "Description": "Std deviations for distribution shift"},
            {"Threshold": "Model Age Warning", "Value": f"{ALERT_THRESHOLDS['model_age_warning_days']} days", "Description": "Days until model age warning"},
            {"Threshold": "Model Age Critical", "Value": f"{ALERT_THRESHOLDS['model_age_critical_days']} days", "Description": "Days until model age critical"},
        ])
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    # Recommended actions
    st.markdown("---")
    st.subheader("📋 Recommended Actions")

    if critical_count > 0:
        st.markdown("""
        **Immediate actions required:**
        1. Check data quality - verify recent data is complete and accurate
        2. Re-run hyperparameter optimization: `python run_model_tuning.py --n-trials 200`
        3. If still degraded, re-run feature selection: `python run_feature_selection.py`
        4. Review market conditions for regime change
        """)
    elif warning_count > 0:
        st.markdown("""
        **Suggested actions:**
        1. Monitor closely over the next few days
        2. Consider retraining the model: `python run_training.py`
        3. Check if market regime has changed significantly
        """)
    else:
        st.markdown("""
        **Model is healthy!** Regular maintenance:
        - Retrain weekly: `python run_training.py`
        - Generate daily predictions: `python run_predict.py`
        - Review this page before trading decisions
        """)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("📈 Momentum Dashboard")

    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Ticker Analysis", "🏆 Top Candidates", "📊 Model Monitoring", "🚨 Alerts"]
    )

    st.sidebar.markdown("---")

    # Model status
    model, metadata = load_model()
    if model is not None:
        st.sidebar.success("✅ Model loaded")
        st.sidebar.caption(f"Features: {metadata.get('n_features', 'N/A')}")
        st.sidebar.caption(f"AUC: {metadata.get('train_auc', 'N/A'):.4f}")

        # Quick health check for sidebar indicator
        training_date_str = metadata.get('training_date')
        if training_date_str:
            try:
                training_date = datetime.fromisoformat(training_date_str.replace('Z', '+00:00'))
                if training_date.tzinfo:
                    training_date = training_date.replace(tzinfo=None)
                days_old = (datetime.now() - training_date).days
                if days_old >= ALERT_THRESHOLDS['model_age_critical_days']:
                    st.sidebar.error(f"🔴 Model {days_old}d old!")
                elif days_old >= ALERT_THRESHOLDS['model_age_warning_days']:
                    st.sidebar.warning(f"🟡 Model {days_old}d old")
            except:
                pass
    else:
        st.sidebar.error("❌ No model found")
        st.sidebar.caption("Run: python run_training.py")

    st.sidebar.markdown("---")

    # Position sizing settings (stored in session state)
    with st.sidebar.expander("💰 Position Sizing", expanded=False):
        if 'account_size' not in st.session_state:
            st.session_state.account_size = 100000
        if 'max_risk_pct' not in st.session_state:
            st.session_state.max_risk_pct = 2.0
        if 'num_entries' not in st.session_state:
            st.session_state.num_entries = 3

        st.session_state.account_size = st.number_input(
            "Account Size ($)",
            min_value=1000,
            max_value=10000000,
            value=st.session_state.account_size,
            step=10000,
            help="Total account value for position sizing"
        )
        st.session_state.max_risk_pct = st.slider(
            "Max Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.max_risk_pct,
            step=0.5,
            help="Maximum % of account to risk on a single trade"
        )
        st.session_state.num_entries = st.selectbox(
            "Entry Levels",
            options=[1, 2, 3],
            index=st.session_state.num_entries - 1,
            help="Number of entry levels for scaling in"
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Render selected page
    if page == "🎯 Ticker Analysis":
        render_ticker_analysis()
    elif page == "🏆 Top Candidates":
        render_top_candidates()
    elif page == "📊 Model Monitoring":
        render_monitoring()
    elif page == "🚨 Alerts":
        render_alerts()


if __name__ == "__main__":
    main()
