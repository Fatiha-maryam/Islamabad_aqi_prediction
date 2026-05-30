"""
Islamabad AQI Prediction Dashboard
Professional dark-themed Streamlit dashboard
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# DARK THEME CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    .stApp { background-color: #0a0e1a; color: #e0e6f0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    .dashboard-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #1e2d4a;
        margin-bottom: 2rem;
    }
    .dashboard-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #0080ff, #7b2fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3px;
        margin: 0;
    }
    .dashboard-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #4a6fa5;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    .aqi-card {
        background: linear-gradient(145deg, #0d1526, #111c33);
        border: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .aqi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #7b2fff);
    }
    .aqi-card-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        color: #4a6fa5;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .aqi-card-sublabel {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.7rem;
        color: #2a4a6a;
        letter-spacing: 2px;
        margin-bottom: 0.3rem;
    }
    .aqi-card-value {
        font-family: 'Orbitron', monospace;
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.2rem 0;
    }
    .aqi-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 50px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 0.4rem;
    }
    .alert-box {
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.95rem;
        border-left-width: 4px;
        border-left-style: solid;
    }
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        color: #00d4ff;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e2d4a;
        margin-bottom: 1.2rem;
    }
    .divider { border: none; border-top: 1px solid #1e2d4a; margin: 1.8rem 0; }
    .no-fi-card {
        background: linear-gradient(145deg, #0d1526, #111c33);
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 2rem 1rem;
        text-align: center;
        height: 320px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# AQI HELPER
# ============================================
def get_aqi_info(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#888888", "Unknown", "Data unavailable", "#88888822", "#88888844"
    val = float(val)
    if val <= 50:
        return "#00e676", "Good", "Air quality is satisfactory. Safe for all activities.", "#00e67622", "#00e67644"
    elif val <= 100:
        return "#ffee58", "Moderate", "Acceptable. Sensitive individuals should limit prolonged outdoor exertion.", "#ffee5822", "#ffee5844"
    elif val <= 150:
        return "#ffa726", "Unhealthy for Sensitive Groups", "Children, elderly & people with respiratory conditions should reduce outdoor activity.", "#ffa72622", "#ffa72644"
    elif val <= 200:
        return "#ef5350", "Unhealthy", "Everyone may experience health effects. Limit outdoor activities.", "#ef535022", "#ef535044"
    elif val <= 300:
        return "#ab47bc", "Very Unhealthy", "Health alert! Everyone should avoid prolonged outdoor exertion.", "#ab47bc22", "#ab47bc44"
    else:
        return "#b71c1c", "Hazardous", "Health emergency. Avoid all outdoor activities.", "#b71c1c22", "#b71c1c44"

# ============================================
# DATA LOADING
# ============================================
@st.cache_resource
def get_mongo_collection():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        st.error("MONGODB_URI not set!")
        return None
    client = MongoClient(mongo_uri)
    return client["aqi_db"]["aqi_features"]

@st.cache_data(ttl=300)
def load_latest_data():
    collection = get_mongo_collection()
    if collection is None:
        return None
    return collection.find_one(sort=[("datetime", -1)], projection={"_id": 0})

@st.cache_data(ttl=300)
def load_recent_trend(days=7):
    collection = get_mongo_collection()
    if collection is None:
        return None
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    cursor = collection.find(
        {"datetime": {"$gte": cutoff}},
        {"_id": 0, "datetime": 1, "lag1": 1}
    ).sort("datetime", 1)
    df = pd.DataFrame(list(cursor))
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={'lag1': 'aqi'})
    return df

@st.cache_data(ttl=3600)
def load_metrics():
    try:
        return pd.read_csv("models/evaluation_metrics.csv")
    except:
        return None

@st.cache_resource
def load_models():
    models = {}
    for horizon in ['24h', '48h', '72h']:
        try:
            with open(f"models/best_model_{horizon}.pkl", 'rb') as f:
                models[horizon] = pickle.load(f)
        except:
            models[horizon] = None
    return models

# ============================================
# PREDICTION
# ============================================
def make_predictions(models, latest_data):
    if latest_data is None:
        return {h: None for h in ['24h', '48h', '72h']}

    FEATURE_COLS = [
        'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'lag48', 'lag72',
        'aqi_ma6', 'aqi_ma12', 'aqi_ma24', 'aqi_std12',
        'hour', 'day_of_week', 'month',
        'pm2_5', 'pm10', 'o3', 'temperature', 'wind_speed', 'rain_code'
    ]

    predictions = {}
    for horizon, model_data in models.items():
        try:
            if model_data is None:
                predictions[horizon] = None
                continue
            feature_row = pd.DataFrame([{col: latest_data.get(col, 0) for col in FEATURE_COLS}])
            pred = model_data['model'].predict(feature_row)[0]
            predictions[horizon] = max(0, round(float(pred), 1))
        except:
            predictions[horizon] = None
    return predictions

# ============================================
# PLOTLY DARK CONFIG
# ============================================
PLOTLY_DARK = dict(
    plot_bgcolor='#0a0e1a', paper_bgcolor='#0a0e1a',
    font=dict(color='#c0cce0', family='Rajdhani'),
    xaxis=dict(gridcolor='#1e2d4a', linecolor='#1e2d4a'),
    yaxis=dict(gridcolor='#1e2d4a', linecolor='#1e2d4a'),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ============================================
# MAIN
# ============================================
def main():

    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🌫️ ISLAMABAD AQI</h1>
        <p class="dashboard-subtitle">Real-Time Air Quality Intelligence & 72-Hour Forecast</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading live data..."):
        latest_data = load_latest_data()
        models      = load_models()
        predictions = make_predictions(models, latest_data)
        trend_df    = load_recent_trend(days=7)
        metrics_df  = load_metrics()

    # ── Row 1: All 4 AQI Cards Equal Size ──────────────────
    col1, col2, col3, col4 = st.columns(4)

    current_aqi = latest_data.get('lag1') if latest_data else None

    cards = [
        (col1, current_aqi,            "CURRENT AQI", "Islamabad"),
        (col2, predictions.get('24h'), "TOMORROW",    "+24 Hours"),
        (col3, predictions.get('48h'), "DAY 2",       "+48 Hours"),
        (col4, predictions.get('72h'), "DAY 3",       "+72 Hours"),
    ]

    for col, val, label, sublabel in cards:
        color, category, _, bg, border = get_aqi_info(val)
        with col:
            st.markdown(f"""
            <div class="aqi-card">
                <div class="aqi-card-label">{label}</div>
                <div class="aqi-card-sublabel">{sublabel}</div>
                <div class="aqi-card-value" style="color:{color};">
                    {int(val) if val is not None else '—'}
                </div>
                <span class="aqi-badge" style="background:{bg};color:{color};border:1px solid {border};">
                    {category}
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Row 2: Alerts ───────────────────────────────────────
    st.markdown('<div class="section-header">⚠ Health Alerts & Recommendations</div>', unsafe_allow_html=True)

    alert_items = [
        ("Current AQI",       current_aqi),
        ("Tomorrow (+24h)",   predictions.get('24h')),
        ("Day 2 (+48h)",      predictions.get('48h')),
        ("Day 3 (+72h)",      predictions.get('72h')),
    ]

    for label, val in alert_items:
        if val is None:
            continue
        color, category, advice, bg, border = get_aqi_info(val)
        st.markdown(f"""
        <div class="alert-box" style="background:{bg};border-left-color:{color};border-color:{border};">
            <span style="color:{color};font-weight:600;font-family:'Orbitron',monospace;font-size:0.82rem;">
                {label} — AQI {int(val)} — {category}
            </span><br>
            <span style="color:#8899aa;font-size:0.88rem;">{advice}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Row 3: Trend Chart + Metrics ────────────────────────
    col_chart, col_metrics = st.columns([1.6, 1])

    with col_chart:
        st.markdown('<div class="section-header">📈 AQI Trend — Last 7 Days</div>', unsafe_allow_html=True)

        if trend_df is not None and not trend_df.empty and 'aqi' in trend_df.columns:
            fig = go.Figure()

            # Zone fills
            for y0, y1, fill in [
                (0,   50,  'rgba(0,230,118,0.04)'),
                (50,  100, 'rgba(255,238,88,0.04)'),
                (100, 150, 'rgba(255,167,38,0.04)'),
                (150, 200, 'rgba(239,83,80,0.04)'),
                (200, 300, 'rgba(171,71,188,0.04)'),
            ]:
                fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0)

            fig.add_trace(go.Scatter(
                x=trend_df['datetime'], y=trend_df['aqi'],
                mode='lines',
                line=dict(color='#00d4ff', width=2),
                fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
                name='AQI',
                hovertemplate='<b>%{x}</b><br>AQI: %{y}<extra></extra>'
            ))

            # Prediction markers
            if latest_data:
                last_dt = pd.to_datetime(latest_data['datetime'])
                for horizon, hours in [('24h', 24), ('48h', 48), ('72h', 72)]:
                    pred_val = predictions.get(horizon)
                    if pred_val:
                        p_color = get_aqi_info(pred_val)[0]
                        fig.add_trace(go.Scatter(
                            x=[last_dt + timedelta(hours=hours)],
                            y=[pred_val],
                            mode='markers+text',
                            marker=dict(color=p_color, size=12, symbol='diamond'),
                            text=[f'+{hours}h: {int(pred_val)}'],
                            textposition='top center',
                            textfont=dict(color=p_color, size=11),
                            name=f'Pred {horizon}',
                        ))

            # Day labels for x-axis
            today    = pd.Timestamp.now().normalize()
            tickvals = pd.date_range(
                start=trend_df['datetime'].min(),
                end=trend_df['datetime'].max(),
                freq='D'
            )
            ticktext = []
            for d in tickvals:
                delta = (d.normalize() - today).days
                if delta == 0:
                    ticktext.append("Today")
                elif delta == 1:
                    ticktext.append("Tomorrow")
                elif delta == -1:
                    ticktext.append("Yesterday")
                elif delta > 1:
                    ticktext.append(f"Day +{delta}")
                else:
                    ticktext.append(f"Day {delta}")

            fig.update_layout(
               **PLOTLY_DARK,
               height=330,
               showlegend=False,
                yaxis_title='US AQI',
                 hovermode='x unified',
                )
            fig.update_xaxes(
               gridcolor='#1e2d4a',
               linecolor='#1e2d4a',
               tickvals=tickvals,
               ticktext=ticktext,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trend data not available yet.")

    with col_metrics:
        st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

        if metrics_df is not None:
            # Best model per horizon — all 3
            best_rows = []
            for horizon in ['24h', '48h', '72h']:
                h_df = metrics_df[metrics_df['horizon'] == horizon]
                if not h_df.empty:
                    best = h_df.loc[h_df['mae'].idxmin()].copy()
                    best_rows.append(best)

            if best_rows:
                display_df = pd.DataFrame(best_rows)[['horizon', 'model', 'mae', 'rmse', 'r2']]
                display_df.columns = ['Horizon', 'Best Model', 'MAE', 'RMSE', 'R²']
                display_df['MAE']  = display_df['MAE'].round(2)
                display_df['RMSE'] = display_df['RMSE'].round(2)
                display_df['R²']   = display_df['R²'].round(3)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Horizon":    st.column_config.TextColumn("Horizon"),
                        "Best Model": st.column_config.TextColumn("Best Model"),
                        "MAE":        st.column_config.NumberColumn("MAE",  format="%.2f"),
                        "RMSE":       st.column_config.NumberColumn("RMSE", format="%.2f"),
                        "R²":         st.column_config.NumberColumn("R²",   format="%.3f"),
                    }
                )

            # MAE comparison bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            fig_m = px.bar(
                metrics_df, x='model', y='mae', color='horizon',
                barmode='group',
                color_discrete_map={'24h': '#00d4ff', '48h': '#7b2fff', '72h': '#ff6b6b'},
                labels={'mae': 'MAE', 'model': ''},
            )
            fig_m.update_layout(
                **PLOTLY_DARK, height=230,
                xaxis_tickangle=-30,
                legend=dict(bgcolor='#0d1526', bordercolor='#1e2d4a', font=dict(size=10)),
                title=dict(text='MAE — All Models Comparison', font=dict(size=11, color='#4a6fa5'), x=0.5)
            )
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("Metrics not available yet.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Row 4: Feature Importance ────────────────────────────
    st.markdown('<div class="section-header">🔬 Feature Importance Analysis</div>', unsafe_allow_html=True)

    fi_cols = st.columns(3)

    for i, horizon in enumerate(['24h', '48h', '72h']):
        with fi_cols[i]:
            model_data  = models.get(horizon)
            model_obj   = model_data['model'] if model_data else None
            model_name  = model_data['model_name'] if model_data else 'N/A'
            feat_cols   = model_data.get('feature_cols', []) if model_data else []
            importances = None

            if model_obj and hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
            elif model_obj and hasattr(model_obj, 'estimators_'):
                for est in model_obj.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        importances = est.feature_importances_
                        model_name  = f"{model_name} (via {type(est).__name__})"
                        break

            if importances is not None and len(feat_cols) > 0:
                fi_df = pd.DataFrame({
                    'feature':    feat_cols,
                    'importance': importances
                }).sort_values('importance', ascending=True).tail(10)

                fig_fi = go.Figure(go.Bar(
                    x=fi_df['importance'], y=fi_df['feature'],
                    orientation='h',
                    marker=dict(
                        color=fi_df['importance'],
                        colorscale=[[0, '#1e2d4a'], [0.5, '#0080ff'], [1, '#00d4ff']],
                        showscale=False
                    ),
                ))
                fig_fi.update_layout(
                    **PLOTLY_DARK, height=320,
                    title=dict(
                        text=f'{horizon} — {model_name}',
                        font=dict(size=11, color='#4a6fa5'), x=0.5
                    ),
                    xaxis_title='Importance', yaxis_title='',
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="no-fi-card">
                    <div style="color:#00d4ff;font-family:'Orbitron',monospace;font-size:0.9rem;letter-spacing:2px;">{horizon}</div>
                    <div style="color:#4a6fa5;font-size:1rem;margin-top:0.8rem;font-family:'Rajdhani';">{model_name}</div>
                    <div style="color:#2a4a6a;font-size:0.82rem;margin-top:0.5rem;font-family:'Rajdhani';">
                        Feature importances not available<br>for this model type
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;font-family:'Rajdhani',sans-serif;color:#FFFFFF;font-size:0.8rem;padding:1rem 0;">
        ISLAMABAD AQI FORECAST SYSTEM · Data: Open-Meteo API · Updated every 5 hours ·
        Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()