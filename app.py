"""
Momentum Trading Dashboard - Streamlit App

Features:
1. Ticker Analysis - Enter a ticker, see probability, target, stop, SHAP waterfall
2. Top Candidates - Browse highest probability stocks with filtering
3. Model Monitoring - Rolling AUC, feature importance stability, calibration

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

    # Load data
    model, metadata = load_model()
    features_df = load_features()
    predictions_df = load_predictions()

    if model is None:
        st.error("Model not found. Run `python run_training.py` first.")
        return

    if features_df is None:
        st.error("Features not found. Run feature computation first.")
        return

    feature_names = metadata.get('features', [])

    # Get available symbols
    available_symbols = sorted(features_df['symbol'].unique())

    # Input section
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Ticker input with autocomplete
        ticker = st.text_input(
            "Enter Ticker Symbol",
            value="AAPL",
            max_chars=10,
            help="Enter a stock ticker symbol"
        ).upper().strip()

    with col2:
        # ATR multiplier for target
        up_mult = st.number_input("Target ATR Multiple", value=3.0, min_value=1.0, max_value=10.0, step=0.5)

    with col3:
        # ATR multiplier for stop
        dn_mult = st.number_input("Stop ATR Multiple", value=1.5, min_value=0.5, max_value=5.0, step=0.25)

    if ticker not in available_symbols:
        st.warning(f"Symbol '{ticker}' not found in data. Available: {len(available_symbols)} symbols")

        # Show suggestions
        matches = [s for s in available_symbols if ticker in s][:10]
        if matches:
            st.info(f"Did you mean: {', '.join(matches)}")
        return

    # Get latest data for ticker
    ticker_data = features_df[features_df['symbol'] == ticker].copy()
    if 'date' in ticker_data.columns:
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        ticker_data = ticker_data.sort_values('date')

    latest = ticker_data.iloc[-1:].copy()
    latest_date = latest['date'].iloc[0] if 'date' in latest.columns else "N/A"

    # Calculate prediction
    available_features = [f for f in feature_names if f in latest.columns]
    X = latest[available_features].copy()
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)

    probability = model.predict_proba(X.values)[:, 1][0]

    # Calculate target and stop
    close_price = latest['close'].iloc[0]
    atr_pct = latest.get('atr_percent', pd.Series([2.0])).iloc[0] / 100

    target_price = close_price * (1 + up_mult * atr_pct)
    stop_price = close_price * (1 - dn_mult * atr_pct)
    target_pct = (target_price / close_price - 1) * 100
    stop_pct = (1 - stop_price / close_price) * 100
    reward_risk = target_pct / max(stop_pct, 0.1)

    # Display main metrics
    st.markdown("---")

    # Probability gauge
    col1, col2, col3, col4 = st.columns(4)

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

    # Price info
    st.markdown(f"**Current Price:** ${close_price:.2f} | **Date:** {latest_date} | **ATR%:** {atr_pct*100:.2f}%")

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
    st.header("🏆 Top Momentum Candidates")

    model, metadata = load_model()
    features_df = load_features()
    predictions_df = load_predictions()

    if model is None or features_df is None:
        st.error("Model or features not found. Run training first.")
        return

    feature_names = metadata.get('features', [])

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

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        min_prob = st.slider("Min Probability", 0.0, 1.0, 0.5, 0.05)

    with col2:
        min_rr = st.slider("Min Reward/Risk", 0.0, 5.0, 1.5, 0.25)

    with col3:
        top_n = st.slider("Show Top N", 10, 200, 50, 10)

    with col4:
        sort_by = st.selectbox("Sort By", ["probability", "reward_risk", "target_pct"])

    # Filter data
    filtered = predictions_df[
        (predictions_df['probability'] >= min_prob) &
        (predictions_df.get('reward_risk', 999) >= min_rr)
    ].copy()

    filtered = filtered.nlargest(top_n, sort_by)

    st.markdown(f"**Showing {len(filtered)} candidates** (filtered from {len(predictions_df)} total)")

    # Display table
    display_cols = ['symbol', 'probability', 'close', 'target_price', 'stop_price',
                    'target_pct', 'stop_pct', 'reward_risk']
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Add sector if available
    if 'sector' in filtered.columns:
        display_cols.insert(1, 'sector')

    display_df = filtered[display_cols].copy()

    # Format columns
    if 'probability' in display_df.columns:
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
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
    st.subheader("📊 Probability Distribution")

    fig = px.histogram(
        predictions_df,
        x='probability',
        nbins=50,
        title="Distribution of Hit Probabilities Across Universe"
    )
    fig.add_vline(x=min_prob, line_dash="dash", line_color="red",
                  annotation_text=f"Filter: {min_prob:.0%}")
    fig.update_layout(xaxis_title="Probability", yaxis_title="Count")

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MODEL MONITORING PAGE
# =============================================================================

def render_monitoring():
    """Render the model monitoring page."""
    st.header("📊 Model Monitoring")

    model, metadata = load_model()
    features_df = load_features()
    targets_df = load_targets()
    feature_importance = load_feature_importance()

    if model is None:
        st.error("Model not found.")
        return

    # Model info
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
# MAIN APP
# =============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("📈 Momentum Dashboard")

    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Ticker Analysis", "🏆 Top Candidates", "📊 Model Monitoring"]
    )

    st.sidebar.markdown("---")

    # Model status
    model, metadata = load_model()
    if model is not None:
        st.sidebar.success("✅ Model loaded")
        st.sidebar.caption(f"Features: {metadata.get('n_features', 'N/A')}")
        st.sidebar.caption(f"AUC: {metadata.get('train_auc', 'N/A'):.4f}")
    else:
        st.sidebar.error("❌ No model found")
        st.sidebar.caption("Run: python run_training.py")

    st.sidebar.markdown("---")
    st.sidebar.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Render selected page
    if page == "🎯 Ticker Analysis":
        render_ticker_analysis()
    elif page == "🏆 Top Candidates":
        render_top_candidates()
    elif page == "📊 Model Monitoring":
        render_monitoring()


if __name__ == "__main__":
    main()
