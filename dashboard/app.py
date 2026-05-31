"""
Islamabad AQI Prediction Dashboard
Light-Dark Theme with Yellow/Gold Accents – Professional & Modern
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

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
# LIGHT‑DARK THEME (Dark Slate + Gold Accents)
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&family=Space+Mono:wght@400;700&display=swap');

    .stApp { background-color: #1e2a3a; color: #e2e8f0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    .dashboard-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #fbbf24;
        margin-bottom: 2rem;
    }
    .dashboard-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        margin: 0;
    }
    .dashboard-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #94a3b8;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    .aqi-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 1.5rem 1rem;
        text-align: center;
        height: 210px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .aqi-card:hover { transform: translateY(-4px); border-color: #fbbf24; }
    .aqi-card-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #94a3b8;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .aqi-card-sublabel {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #64748b;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .aqi-card-value {
        font-family: 'Space Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.2rem 0;
    }
    .aqi-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 50px;
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 0.4rem;
    }
    .alert-box {
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        border-left-width: 4px;
        border-left-style: solid;
        background-color: #0f172a;
        backdrop-filter: blur(2px);
        border: 1px solid #334155;
    }
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #fbbf24;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
        margin-bottom: 1.2rem;
    }
    .divider { border: none; border-top: 1px solid #334155; margin: 1.8rem 0; }
    .no-fi-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 1rem;
        text-align: center;
        height: 320px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid #334155;
    }
    .stDataFrame th {
        background-color: #1e2a3a;
        color: #fbbf24;
        font-family: 'Inter', sans-serif;
    }
    .stDataFrame td {
        color: #cbd5e1;
    }
    /* Buttons & interactive elements */
    .stButton > button {
        background-color: #fbbf24;
        color: #0f172a;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #f59e0b;
        color: #0f172a;
    }
    /* Metric boxes (if any) */
    [data-testid="stMetricValue"] {
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# AQI HELPER
# ============================================
def get_aqi_info(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#94a3b8", "Unknown", "Data unavailable", "#1e293b", "#334155"
    val = float(val)
    if val <= 50:
        return "#10b981", "Good", "Air quality is satisfactory.", "#064e3b", "#10b98166"
    elif val <= 100:
        return "#f59e0b", "Moderate", "Acceptable. Sensitive individuals should limit outdoor exertion.", "#451a03", "#f59e0b66"
    elif val <= 150:
        return "#f97316", "Unhealthy for Sensitive Groups", "Children, elderly & people with respiratory conditions reduce outdoor activity.", "#431407", "#f9731666"
    elif val <= 200:
        return "#ef4444", "Unhealthy", "Everyone may experience health effects. Limit outdoor activities.", "#7f1d1d", "#ef444466"
    elif val <= 300:
        return "#a855f7", "Very Unhealthy", "Health alert! Avoid prolonged outdoor exertion.", "#3b0764", "#a855f766"
    else:
        return "#ec4899", "Hazardous", "Health emergency. Avoid all outdoor activities.", "#4c0519", "#ec489966"

def get_forecast_date(base_date, hours_ahead):
    return (base_date + timedelta(hours=hours_ahead)).strftime("%b %d, %Y")

# ============================================
# MLFLOW SETUP
# ============================================
def setup_mlflow():
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "Fatiha-maryam")
    token = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if not token:
        st.warning("MLflow token not set. Metrics may not load.")
        return
    dagshub.auth.add_app_token(token=token)
    dagshub.init(repo_owner=username, repo_name="Islamabad_aqi_prediction", mlflow=True)

# ============================================
# MONGODB LOADING
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

# ============================================
# METRICS FROM REGISTRY (best model per horizon)
# ============================================
@st.cache_data(ttl=3600)
def load_metrics_for_registered_models():
    try:
        client = MlflowClient()
        records = []
        for horizon in ['24h', '48h', '72h']:
            model_name = f"aqi_model_{horizon}"
            try:
                latest_version = client.get_latest_versions(model_name, stages=["None", "Production", "Staging"])
                if not latest_version:
                    continue
                version = latest_version[0]
                run_id = version.run_id
                run = client.get_run(run_id)
                params = run.data.params
                metrics = run.data.metrics
                records.append({
                    "horizon": horizon,
                    "model": params.get("model_name", "Unknown"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                })
            except Exception as e:
                print(f"Could not fetch metrics for {model_name}: {e}")
        return pd.DataFrame(records) if records else None
    except Exception as e:
        st.warning(f"Could not load registry metrics: {e}")
        return None

# ============================================
# ALL MODELS METRICS (for comparison chart)
# ============================================
@st.cache_data(ttl=3600)
def load_all_models_metrics():
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Islamabad_AQI_Prediction")
        if not experiment:
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100
        )
        records = []
        for run in runs:
            params = run.data.params
            metrics = run.data.metrics
            model_name = params.get("model_name")
            horizon = params.get("horizon")
            mae = metrics.get("mae")
            if model_name and horizon and mae is not None:
                records.append({
                    "model": model_name,
                    "horizon": horizon,
                    "mae": mae,
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                    "run_id": run.info.run_id
                })
        if not records:
            return None
        df = pd.DataFrame(records)
        df = df.sort_values("run_id", ascending=False).drop_duplicates(subset=["model","horizon"])
        return df
    except Exception as e:
        st.warning(f"Could not load all metrics: {e}")
        return None

# ============================================
# LOAD MODELS FROM REGISTRY
# ============================================
@st.cache_resource
def load_models():
    setup_mlflow()
    models = {}
    for horizon in ['24h', '48h', '72h']:
        try:
            model_uri = f"models:/aqi_model_{horizon}/latest"
            loaded = mlflow.sklearn.load_model(model_uri)
            class_name = type(loaded).__name__
            if 'Stacking' in class_name:
                friendly = 'StackingRegressor'
            elif 'XGB' in class_name:
                friendly = 'XGBoost'
            elif 'LGBM' in class_name:
                friendly = 'LightGBM'
            elif 'CatBoost' in class_name:
                friendly = 'CatBoost'
            elif 'RandomForest' in class_name:
                friendly = 'RandomForest'
            else:
                friendly = class_name
            models[horizon] = {
                'model': loaded,
                'model_name': friendly,
                'horizon': horizon,
                'feature_cols': FEATURE_COLS
            }
        except Exception as e:
            print(f"Failed to load aqi_model_{horizon}: {e}")
            models[horizon] = None
    return models

# ============================================
# PREDICTION
# ============================================
def make_predictions(models, latest_data):
    if latest_data is None:
        return {h: None for h in ['24h', '48h', '72h']}
    feature_cols = FEATURE_COLS
    predictions = {}
    for horizon, model_data in models.items():
        if model_data is None:
            predictions[horizon] = None
            continue
        try:
            feature_row = {col: latest_data.get(col, 0) for col in feature_cols}
            input_df = pd.DataFrame([feature_row])
            pred = model_data['model'].predict(input_df)[0]
            predictions[horizon] = max(0, round(float(pred), 1))
        except Exception:
            predictions[horizon] = None
    return predictions

# ============================================
# PLOTLY CONFIG (dark background, gold accents)
# ============================================
PLOTLY_DARK = dict(
    plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
    font=dict(color='#e2e8f0', family='Inter'),
    xaxis=dict(gridcolor='#334155', linecolor='#475569', title_font=dict(color='#fbbf24'), tickfont=dict(color='#cbd5e1')),
    yaxis=dict(gridcolor='#334155', linecolor='#475569', title_font=dict(color='#fbbf24'), tickfont=dict(color='#cbd5e1')),
    legend=dict(font=dict(color='#e2e8f0'), bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155'),
    margin=dict(l=20, r=20, t=40, b=20),
)

FEATURE_COLS = [
    'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'lag48', 'lag72',
    'aqi_ma6', 'aqi_ma12', 'aqi_ma24', 'aqi_std12',
    'aqi_trend_3h', 'aqi_trend_6h', 'aqi_trend_24h',
    'aqi_min_24h', 'aqi_max_24h', 'aqi_range_24h',
    'pm2_5', 'pm10', 'pm25_lag24', 'pm25_ma12',
    'hour_sin', 'hour_cos',
    'season', 'is_rush_hour', 'is_smog_season',
    'day_of_week', 'o3', 'no2', 'co',
    'temperature', 'humidity', 'wind_speed', 'rain_code'
]

# ============================================
# MAIN
# ============================================
def main():
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🌫️ ISLAMABAD AQI</h1>
        <p class="dashboard-subtitle">Real-Time Air Quality Intelligence & 72-Hour Forecast</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading live data..."):
        latest_data = load_latest_data()
        models = load_models()
        predictions = make_predictions(models, latest_data)
        trend_df = load_recent_trend(days=7)
        registered_metrics_df = load_metrics_for_registered_models()
        all_metrics_df = load_all_models_metrics()

    # --- AQI Cards ---
    if latest_data:
        latest_dt = pd.to_datetime(latest_data['datetime'])
    else:
        latest_dt = datetime.now()
    current_aqi = latest_data.get('lag1') if latest_data else None

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, current_aqi, "CURRENT AQI", latest_dt.strftime("%b %d, %Y")),
        (col2, predictions.get('24h'), "TOMORROW", get_forecast_date(latest_dt, 24)),
        (col3, predictions.get('48h'), "DAY 2", get_forecast_date(latest_dt, 48)),
        (col4, predictions.get('72h'), "DAY 3", get_forecast_date(latest_dt, 72)),
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

    # --- Health Alerts ---
    st.markdown('<div class="section-header">⚠ Health Alerts & Recommendations</div>', unsafe_allow_html=True)
    alert_items = [
        ("Current AQI", current_aqi),
        ("Tomorrow (+24h)", predictions.get('24h')),
        ("Day 2 (+48h)", predictions.get('48h')),
        ("Day 3 (+72h)", predictions.get('72h')),
    ]
    for label, val in alert_items:
        if val is None:
            continue
        color, category, advice, bg, border = get_aqi_info(val)
        st.markdown(f"""
        <div class="alert-box" style="background:{bg};border-left-color:{color};border-color:{border};">
            <span style="color:#fbbf24;font-weight:600;font-family:'Inter',sans-serif;font-size:0.82rem;">
                {label} — AQI {int(val)} — {category}
            </span><br>
            <span style="color:#cbd5e1;font-size:0.88rem;">{advice}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- Trend Chart + Best Models Table ---
    col_chart, col_metrics = st.columns([1.6, 1])
    with col_chart:
        st.markdown('<div class="section-header">📈 AQI Trend — Last 7 Days</div>', unsafe_allow_html=True)
        if trend_df is not None and not trend_df.empty:
            fig = go.Figure()
            for y0, y1, fill in [
                (0, 50, 'rgba(16,185,129,0.05)'),
                (50, 100, 'rgba(245,158,11,0.05)'),
                (100, 150, 'rgba(249,115,22,0.05)'),
                (150, 200, 'rgba(239,68,68,0.05)'),
                (200, 300, 'rgba(168,85,247,0.05)'),
            ]:
                fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0)
            fig.add_trace(go.Scatter(
                x=trend_df['datetime'], y=trend_df['aqi'],
                mode='lines',
                line=dict(color='#fbbf24', width=2),
                fill='tozeroy', fillcolor='rgba(251,191,36,0.05)',
                name='AQI',
                hovertemplate='<b>%{x}</b><br>AQI: %{y}<extra></extra>'
            ))
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
            today = pd.Timestamp.now().normalize()
            tickvals = pd.date_range(start=trend_df['datetime'].min(), end=trend_df['datetime'].max(), freq='D')
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
            fig.update_layout(**PLOTLY_DARK, height=330, showlegend=False, yaxis_title='US AQI', hovermode='x unified')
            fig.update_xaxes(gridcolor='#334155', linecolor='#475569', tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trend data not available yet.")

    with col_metrics:
        st.markdown('<div class="section-header">🏆 Best Models (Registry)</div>', unsafe_allow_html=True)
        if registered_metrics_df is not None and not registered_metrics_df.empty:
            display_df = registered_metrics_df[['horizon', 'model', 'mae', 'rmse', 'r2']].round(2)
            display_df.columns = ['Horizon', 'Best Model', 'MAE', 'RMSE', 'R²']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Registry metrics not available. Run training pipeline first.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- MAE Comparison (All Models) ---
    st.markdown('<div class="section-header">📊 MAE Comparison — All Models by Horizon</div>', unsafe_allow_html=True)
    if all_metrics_df is not None and not all_metrics_df.empty:
        fig_all = px.bar(
            all_metrics_df,
            x='model',
            y='mae',
            color='horizon',
            barmode='group',
            color_discrete_map={'24h': '#fbbf24', '48h': '#f59e0b', '72h': '#d97706'},
            labels={'mae': 'Mean Absolute Error (MAE)', 'model': 'Model', 'horizon': 'Horizon'},
            title=''
        )
        fig_all.update_layout(**PLOTLY_DARK, height=400, legend_title_text='Horizon')
        fig_all.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.info("Model metrics not available yet. Run training pipeline to populate.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- Feature Importance ---
    st.markdown('<div class="section-header">🔬 Feature Importance Analysis</div>', unsafe_allow_html=True)
    fi_cols = st.columns(3)
    for i, horizon in enumerate(['24h', '48h', '72h']):
        with fi_cols[i]:
            model_data = models.get(horizon)
            model_obj = model_data['model'] if model_data else None
            actual_name = model_data['model_name'] if model_data else 'N/A'
            feat_cols = FEATURE_COLS if model_obj else []
            importances = None
            if model_obj and hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
            elif model_obj and hasattr(model_obj, 'estimators_'):
                for est in model_obj.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        importances = est.feature_importances_
                        actual_name = f"{actual_name} (via {type(est).__name__})"
                        break
            if importances is not None and len(feat_cols) > 0:
                fi_df = pd.DataFrame({'feature': feat_cols, 'importance': importances})
                fi_df = fi_df.sort_values('importance', ascending=True).tail(10)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df['importance'], y=fi_df['feature'],
                    orientation='h',
                    marker=dict(color=fi_df['importance'], colorscale=[[0, '#334155'], [0.5, '#fbbf24'], [1, '#f59e0b']], showscale=False)
                ))
                fig_fi.update_layout(**PLOTLY_DARK, height=320,
                                     title=dict(text=f'{horizon} — {actual_name}', font=dict(size=11, color='#fbbf24'), x=0.5),
                                     xaxis_title='Importance', yaxis_title='')
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="no-fi-card">
                    <div style="color:#fbbf24;font-family:'Inter',sans-serif;font-size:0.9rem;letter-spacing:2px;">{horizon}</div>
                    <div style="color:#cbd5e1;font-size:1rem;margin-top:0.8rem;">{actual_name}</div>
                    <div style="color:#64748b;font-size:0.82rem;margin-top:0.5rem;">Feature importances not available</div>
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;font-family:'Inter',sans-serif;color:#94a3b8;font-size:0.8rem;padding:1rem 0;">
        ISLAMABAD AQI FORECAST SYSTEM · Data: Open-Meteo API · Updated every 5 hours ·
        Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()