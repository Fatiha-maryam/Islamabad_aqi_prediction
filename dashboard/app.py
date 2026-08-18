"""
Islamabad AQI Prediction Dashboard
Modern air-quality monitoring interface for Islamabad.

Plan A Architecture:
- Reads latest features from: islamabad_aqi.aqi_features_v2 (no target columns)
- Features updated every 5 hours by feature_pipeline
- Predictions: Uses trained CatBoost models via model.predict()
- Models updated weekly (Sunday) with fresh training data
- No target columns expected in feature database (by design)
"""

import base64
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import dagshub
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import streamlit as st
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from pymongo import MongoClient


def utc_to_local(utc_str):
    utc_time = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    utc_time = utc_time.replace(tzinfo=pytz.UTC)
    return utc_time.astimezone(pytz.timezone("Asia/Karachi"))


load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def get_local_image_data_url(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


BACKGROUND_IMAGE = get_local_image_data_url(Path(__file__).resolve().parent / "image" / "islamabad.jpg")

st.markdown(
    f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

        .stApp {{
            background: linear-gradient(rgba(7,10,18,0.58), rgba(7,10,18,0.72)),
                        url('{BACKGROUND_IMAGE}') center / cover no-repeat fixed;
            color: #e2e8f0;
        }}

        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
        .block-container {{ padding: 2rem 2.2rem 3rem 2.2rem; max-width: 1500px; }}

        .hero-section {{
            position: relative;
            min-height: 72vh;
            display: flex;
            align-items: flex-end;
            padding: 2.5rem 2rem 2rem 2rem;
            border-radius: 28px;
            overflow: hidden;
            margin-bottom: 1.5rem;
            background: linear-gradient(180deg, rgba(8,11,17,0.18), rgba(8,11,17,0.68)),
                        url('{BACKGROUND_IMAGE}') center / cover no-repeat;
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
            border: 1px solid rgba(255,255,255,0.15);
        }}
        .hero-overlay {{
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(7,11,17,0.72), rgba(7,11,17,0.18));
        }}
        .hero-content {{
            position: relative;
            z-index: 1;
            max-width: 780px;
        }}
        .dashboard-title {{
            font-family: 'Inter', sans-serif;
            font-size: clamp(2.5rem, 4vw, 5rem);
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #f8fafc;
            margin: 0;
            line-height: 0.95;
            text-transform: uppercase;
        }}
        .dashboard-subtitle {{
            font-size: 0.8rem;
            color: #e2e8f0;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 0.7rem 0 0 0;
            opacity: 0.9;
        }}
        .hero-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1rem;
        }}
        .badge-pill {{
            background: rgba(15, 23, 42, 0.52);
            border: 1px solid rgba(255,255,255,0.16);
            color: #f8fafc;
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            backdrop-filter: blur(6px);
        }}

        .aqi-card {{
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(30, 41, 59, 0.82));
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 22px;
            padding: 1.4rem 1rem 1.1rem 1rem;
            text-align: center;
            min-height: 220px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 18px 34px rgba(2, 6, 23, 0.28);
            backdrop-filter: blur(8px);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .aqi-card:hover {{ transform: translateY(-3px); border-color: rgba(251,191,36,0.75); }}
        .aqi-card-label {{
            color: #94a3b8;
            font-size: 0.74rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }}
        .aqi-card-sublabel {{
            color: #64748b;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}
        .aqi-card-value {{
            font-family: 'Space Mono', monospace;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1;
            margin: 0.3rem 0 0.5rem 0;
        }}
        .aqi-badge {{
            display: inline-block;
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}

        .panel {{
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            box-shadow: 0 10px 18px rgba(2, 6, 23, 0.22);
            padding: 1rem 1.1rem;
            margin-top: 1.2rem;
        }}
        .section-title {{
            font-size: 0.8rem;
            color: #fbbf24;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
            font-weight: 700;
        }}

        .metric-box {{
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            height: 100%;
        }}
        .metric-label {{
            color: #94a3b8;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }}
        .metric-value {{
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: 0.3rem;
            color: #f8fafc;
        }}
        .metric-unit {{
            font-size: 0.7rem;
            color: #94a3b8;
            margin-left: 0.35rem;
        }}
        .meter-wrap {{
            margin-top: 0.7rem;
            background: rgba(51,65,85,0.7);
            border-radius: 999px;
            overflow: hidden;
            height: 10px;
            position: relative;
        }}
        .meter-fill {{
            height: 100%;
            border-radius: 999px;
            box-shadow: inset 0 0 10px rgba(255,255,255,0.2);
        }}
        .meter-label-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.5rem;
            color: #cbd5e1;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .suggestion-box {{
            background: rgba(15, 23, 42, 0.8);
            border-left: 4px solid #fbbf24;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #e2e8f0;
        }}
        .suggestion-box strong {{ color: #fbbf24; }}
        .pollutant-cell {{
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            padding: 1rem 0.7rem;
            min-height: 110px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            box-shadow: 0 10px 18px rgba(2, 6, 23, 0.18);
        }}

        .feature-image-card {{
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.7rem;
            min-height: 260px;
        }}

        .stDataFrame {{ background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; }}
        .stDataFrame th {{ background-color: rgba(15, 23, 42, 0.95); color: #fbbf24; }}
        .stDataFrame td {{ color: #e2e8f0; }}
        .stMetric {{ background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 12px; padding: 0.6rem 0.8rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================
# AQI HELPER
# ============================================
def get_aqi_info(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#94a3b8", "Unknown", "Data unavailable", "rgba(148,163,184,0.15)", "#94a3b8"
    val = float(val)
    if val <= 50:
        return "#10b981", "Good", "Air quality is satisfactory.", "rgba(16,185,129,0.12)", "#10b981"
    elif val <= 100:
        return "#f59e0b", "Moderate", "Sensitive people should reduce prolonged outdoor activity.", "rgba(245,158,11,0.12)", "#f59e0b"
    elif val <= 150:
        return "#f97316", "Sensitive", "Children, elderly, and people with breathing conditions should reduce exertion.", "rgba(249,115,22,0.12)", "#f97316"
    elif val <= 200:
        return "#ef4444", "Unhealthy", "Avoid prolonged or intense outdoor activity.", "rgba(239,68,68,0.12)", "#ef4444"
    elif val <= 300:
        return "#a855f7", "Very Unhealthy", "Health alert. Minimize outdoor exposure.", "rgba(168,85,247,0.12)", "#a855f7"
    else:
        return "#ec4899", "Hazardous", "Health emergency. Avoid outdoor exposure.", "rgba(236,72,153,0.12)", "#ec4899"


def get_forecast_date(base_date, hours_ahead):
    return (base_date + timedelta(hours=hours_ahead)).strftime("%b %d, %Y")


def guidance_for_band(aqi: int):
    if aqi <= 50:
        return [
            {"icon": "ti-mood-smile", "label": "Air quality is good today"},
            {"icon": "ti-mask", "label": "Wear a mask outdoors"},
            {"icon": "ti-wind", "label": "Open windows briefly"},
            {"icon": "ti-droplet", "label": "Stay hydrated outdoors"},
        ]
    elif aqi <= 100:
        return [
            {"icon": "ti-alert-circle", "label": "Sensitive groups: light activity only"},
            {"icon": "ti-mask", "label": "Wear a mask outdoors"},
            {"icon": "ti-run", "label": "Skip the 6–9am jog"},
            {"icon": "ti-window", "label": "Keep windows closed"},
        ]
    elif aqi <= 150:
        return [
            {"icon": "ti-mask", "label": "Wear a mask outdoors"},
            {"icon": "ti-run", "label": "Skip the 6–9am jog"},
            {"icon": "ti-window", "label": "Keep windows closed"},
            {"icon": "ti-air-conditioning", "label": "Run purifier indoors"},
        ]
    else:
        return [
            {"icon": "ti-mask", "label": "Wear a mask outdoors"},
            {"icon": "ti-run", "label": "Skip strenuous activity"},
            {"icon": "ti-window", "label": "Keep windows closed"},
            {"icon": "ti-home", "label": "Stay indoors if possible"},
        ]


def build_health_actions(current_aqi, predictions):
    current_aqi = int(float(current_aqi)) if current_aqi is not None else 0
    actions = guidance_for_band(current_aqi)
    if current_aqi >= 100 or any(v is not None and float(v) >= 100 for v in predictions.values()):
        actions.append({"icon": "ti-user-heart", "label": "Extra caution for children, elderly, or breathing conditions"})
    else:
        actions.append({"icon": "ti-heart", "label": "Watch symptoms if you have asthma or allergies"})
    return actions


def meter_color_for_value(value):
    if value is None:
        return "#94a3b8"
    value = float(value)
    if value <= 50:
        return "#10b981"
    elif value <= 100:
        return "#f59e0b"
    elif value <= 150:
        return "#f97316"
    elif value <= 200:
        return "#ef4444"
    elif value <= 300:
        return "#a855f7"
    return "#ec4899"


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
# MONGODB LOADING (Plan A: Read from aqi_features_v2)
# ============================================
@st.cache_resource
def get_mongo_collection():
    """Connect to MongoDB and get aqi_features_v2 collection (features only, no targets)"""
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        st.error("MONGODB_URI not set!")
        return None
    client = MongoClient(mongo_uri)
    return client["islamabad_aqi"]["aqi_features_v2"]

@st.cache_data(ttl=300)
def load_latest_data():
    collection = get_mongo_collection()
    if collection is None:
        return None
    doc= collection.find_one(sort=[("datetime", -1)], projection={"_id": 0})
    print("Latest doc keys:", doc.keys() if doc else "None")
    return doc
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
#======================
#       Load Models
#======================
@st.cache_resource
def load_models():
    print(" load_models() called")
    setup_mlflow()
    models = {}
    for horizon in ['24h', '48h', '72h']:
        try:
            model_uri = f"models:/aqi_model_{horizon}/latest"
            print(f" Loading {horizon} from {model_uri}")
            wrapper = mlflow.pyfunc.load_model(model_uri)
            print(f" Loaded {horizon} – type: {type(wrapper)}")
            models[horizon] = {
                'model': wrapper,
                'model_name': 'pyfunc_wrapper',
                'horizon': horizon,
                'feature_cols': FEATURE_COLS
            }
        except Exception as e:
            print(f" {horizon} failed: {e}")
            models[horizon] = None
    print(f" Models loaded: {len([m for m in models.values() if m is not None])} of 3")
    return models
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


@st.cache_data(ttl=3600)
def load_best_model_metrics():
    return load_metrics_for_registered_models()


@st.cache_data(ttl=3600)
def load_feature_importance_files():
    output = {}
    models_dir = Path(__file__).resolve().parent.parent / "models"
    for horizon in ["24h", "48h", "72h"]:
        path = models_dir / f"feature_importance_{horizon}.png"
        if path.exists():
            output[horizon] = str(path)
    return output


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

            pred_result = model_data['model'].predict(input_df)
            arr = np.asarray(pred_result).ravel() if pred_result is not None else np.array([])
            pred = float(arr[0]) if len(arr) > 0 else None

            if pred is not None:
                predictions[horizon] = max(0, round(float(pred), 1))
            else:
                predictions[horizon] = None

        except Exception as e:
            print(f" {horizon} prediction error: {e}")
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
    with st.spinner("Loading live data..."):
        latest_data = load_latest_data()
        models = load_models()
        predictions = make_predictions(models, latest_data)
        trend_df = load_recent_trend(days=7)

    if latest_data:
        latest_dt = pd.to_datetime(latest_data["datetime"])
    else:
        latest_dt = datetime.now()
    current_aqi = float(latest_data.get("lag1")) if latest_data and latest_data.get("lag1") is not None else None

    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="dashboard-title">Islamabad AQI</div>
                <div class="dashboard-subtitle">Real-time air quality intelligence & 72-hour forecast</div>
                <div class="hero-meta">
                    <span class="badge-pill">{date}</span>
                    <span class="badge-pill">{time}</span>
                    <span class="badge-pill">Live feature stream</span>
                </div>
            </div>
        </div>
        """.format(date=datetime.now().strftime("%b %d, %Y"), time=datetime.now().strftime("%H:%M PKT")),
        unsafe_allow_html=True,
    )

    cards = [
        ("Current AQI", current_aqi, latest_dt.strftime("%b %d, %Y")),
        ("Tomorrow", predictions.get("24h"), get_forecast_date(latest_dt, 24)),
        ("Day 2", predictions.get("48h"), get_forecast_date(latest_dt, 48)),
        ("Day 3", predictions.get("72h"), get_forecast_date(latest_dt, 72)),
    ]

    cols = st.columns(4)
    for col, (label, val, sublabel) in zip(cols, cards):
        color, category, _, bg, border = get_aqi_info(val)
        with col:
            st.markdown(
                f"""
                <div class="aqi-card">
                    <div class="aqi-card-label">{label}</div>
                    <div class="aqi-card-sublabel">{sublabel}</div>
                    <div class="aqi-card-value" style="color:{color};">{int(val) if val is not None else '—'}</div>
                    <span class="aqi-badge" style="background:{bg};color:{color};border:1px solid {border};">{category}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Live conditions</div>", unsafe_allow_html=True)
    env_items = [
        ("Temperature", latest_data.get("temperature") if latest_data else None, "°C", 0, 40),
        ("Humidity", latest_data.get("humidity") if latest_data else None, "%", 0, 100),
        ("Wind Speed", latest_data.get("wind_speed") if latest_data else None, "m/s", 0, 20),
        ("Ozone", latest_data.get("o3") if latest_data else None, "ppb", 0, 200),
        ("PM2.5", latest_data.get("pm2_5") if latest_data else None, "µg/m³", 0, 200),
        ("PM10", latest_data.get("pm10") if latest_data else None, "µg/m³", 0, 300),
    ]

    metric_cols = st.columns(6)
    for col, (name, value, unit, min_val, max_val) in zip(metric_cols, env_items):
        with col:
            display_value = "—"
            meter_value = 0
            meter_color = "#94a3b8"
            if value is not None:
                value_f = float(value)
                display_value = f"{round(value_f, 1)} {unit}"
                meter_value = max(0, min(100, ((value_f - min_val) / max(1, max_val - min_val)) * 100))
                meter_color = meter_color_for_value(value_f)
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{name}</div>
                    <div class="metric-value">{display_value}</div>
                    <div class="meter-wrap">
                        <div class="meter-fill" style="width:{meter_value}%; background:{meter_color};"></div>
                    </div>
                    <div class="meter-label-row"><span>Low</span><span>High</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Seven-day AQI trend</div>", unsafe_allow_html=True)
    if trend_df is not None and not trend_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend_df["datetime"], y=trend_df["aqi"],
                mode="lines",
                line=dict(color="#fbbf24", width=3),
                name="AQI",
                fill="tozeroy",
                fillcolor="rgba(251,191,36,0.15)",
            )
        )
        if latest_data:
            for horizon, hours in [("24h", 24), ("48h", 48), ("72h", 72)]:
                pred_val = predictions.get(horizon)
                if pred_val is not None:
                    pcolor = get_aqi_info(pred_val)[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[latest_dt + timedelta(hours=hours)],
                            y=[pred_val],
                            mode="markers",
                            marker=dict(size=14, color=pcolor, line=dict(width=2, color="#f8fafc")),
                            name=f"{horizon} forecast",
                        )
                    )
        fig.add_hrect(y0=0, y1=50, fillcolor="rgba(16,185,129,0.08)", line_width=0)
        fig.add_hrect(y0=50, y1=100, fillcolor="rgba(245,158,11,0.08)", line_width=0)
        fig.add_hrect(y0=100, y1=150, fillcolor="rgba(249,115,22,0.08)", line_width=0)
        fig.add_hrect(y0=150, y1=200, fillcolor="rgba(239,68,68,0.08)", line_width=0)
        fig.add_hrect(y0=200, y1=300, fillcolor="rgba(168,85,247,0.08)", line_width=0)
        layout = dict(PLOTLY_DARK)
        layout.update({
            "height": 420,
            "hovermode": "x unified",
            "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            "margin": {"l": 18, "r": 18, "t": 20, "b": 18},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        })
        fig.update_layout(layout)
        fig.update_xaxes(tickformat="%b %d", title_text="Date")
        fig.update_yaxes(title_text="AQI", range=[0, max(150, float(trend_df['aqi'].max()) * 1.2)])
        fig.add_annotation(
            text="Last 7 days AQI pattern",
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#f8fafc", size=12),
            align="center",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Trend chart is not available yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Health guidance</div>", unsafe_allow_html=True)
    action_tiles = build_health_actions(current_aqi, predictions)
    cols = st.columns(len(action_tiles))
    for col, tile in zip(cols, action_tiles):
        with col:
            st.markdown(
                f"""
                <div class="pollutant-cell" style="text-align:center;">
                    <i class="ti {tile['icon']}" style="font-size:24px; color:#fbbf24;"></i>
                    <p style="font-size:13px; margin:8px 0 0; color:#e2e8f0; line-height:1.4;">{tile['label']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align:center; color:#cbd5e1; font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase; margin-top:1.4rem; opacity:0.8;'>
            Islamabad Air Quality Intelligence • Updated from live feature stream
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()