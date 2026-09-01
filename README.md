# Islamabad AQI Forecasting System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://islamabad-aqi-forecast.streamlit.app/)  
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Tracking-orange)](https://dagshub.com/Fatiha-maryam/Islamabad_aqi_prediction)  
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated-blue)](https://github.com/Fatiha-maryam/Islamabad_aqi_prediction/actions)

> **End‑to‑end serverless pipeline** that forecasts Air Quality Index (AQI) for the next 24h, 48h and 72h for Islamabad, Pakistan.  
> Data is fetched hourly, features are engineered automatically, models are retrained weekly, and a live dashboard displays the latest forecasts.

---

##  Live Demo

 [**https://islamabad-aqi-forecast.streamlit.app/**](https://islamabad-aqi-forecast.streamlit.app/)

---

##  Overview

| Component | Technology | Automation |
|-----------|------------|------------|
| **Data source** | Open‑Meteo Air Quality & Weather API | – |
| **Feature Pipeline** | Python + MongoDB | Every 5 hours (GitHub Actions) |
| **Training Data Prep** | Python + MongoDB | Weekly (Sunday 1:00 AM UTC) |
| **Training Pipeline** | Python + MLflow (DagsHub) | Weekly (Sunday 2:00 AM UTC) |
| **Model Registry** | DagsHub MLflow | Stores latest model versions |
| **Dashboard** | Streamlit | Serves live predictions |

---

##  Architecture (Plan A)

This project follows a **clean separation** between live features and training data:

| Collection | Purpose | Contains |
|------------|---------|----------|
| `aqi_features_v2` | Live feature store | Features only (no targets) |
| `aqi_training_data` | Training dataset | Features + targets (complete) |

**Why this matters:**  
- The latest row is never "incomplete" because no targets are expected in the live store.  
- Training uses a separate, complete dataset with targets.  
- Dashboard reads features only and uses `model.predict()` for forecasting.  
- This prevents data leakage and ensures clean ML pipelines.

---

##  Models & Performance

For each horizon (`24h`, `48h`, `72h`) we train and compare:

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- StackingRegressor

**Best model per horizon (latest run):**

| Horizon | Best Model | MAE | RMSE | R² |
|--------|-----------|-----|------|-----|
| 24h | CatBoost | 11.63 | 15.28 | 0.759 |
| 48h | CatBoost | 16.01 | 20.76 | 0.558 |
| 72h | CatBoost | 17.53 | 23.24 | 0.446 |

> Metrics are updated weekly after each training run.

---

##  Repository Structure

```bash
├── .github/workflows/               # GitHub Actions CI/CD
│   ├── feature-pipeline.yml         # every 5 hours
│   ├── training_data_preparation.yml # weekly (Sunday 1 AM)
│   └── training-pipeline.yml        # weekly (Sunday 2 AM)
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── src/
│   ├── feature_pipeline.py          # fetch, clean, feature engineering → MongoDB
│   ├── training_data_preparation.py # compute targets → aqi_training_data
│   └── training_pipeline.py         # train models → register in MLflow
├── models/                          # local .pkl backups (pushed to GitHub)
│   ├── best_model_24h.pkl
│   ├── best_model_48h.pkl
│   └── best_model_72h.pkl
├── requirements.txt                 # all Python dependencies
└── README.md                        # project documentation
```

---

##  Features Engineered

For each hour, the pipeline creates:

| Feature Category | Features Created | Purpose |
|------------------|------------------|---------|
| **Lags** | `lag1` … `lag72` | Captures past AQI values (1h to 72h) |
| **Rolling Averages** | `aqi_ma6`, `aqi_ma12`, `aqi_ma24` | Smooths out short-term noise |
| **Rolling Std Dev** | `aqi_std12` | Captures volatility and instability |
| **Trends** | `aqi_trend_3h`, `aqi_trend_6h`, `aqi_trend_24h` | Rate of change in AQI |
| **Daily Stats** | `aqi_min_24h`, `aqi_max_24h`, `aqi_range_24h` | Daily extremes and variability |
| **Time Features** | `hour`, `day_of_week`, `month`, `hour_sin`, `hour_cos` | Cyclical time encoding for seasonality |
| **Season** | `season` (winter, spring, summer, autumn) | Captures seasonal pollution patterns |
| **Flags** | `is_rush_hour`, `is_smog_season` | Binary indicators for high-risk periods |
| **PM2.5 Features** | `pm25_lag24`, `pm25_ma12` | Lag and rolling averages for fine particles |
| **Weather** | `temperature`, `humidity`, `wind_speed`, `rain_code` | Meteorological conditions affecting dispersion |

All features are stored in **MongoDB Atlas** (one row per hour) and served to the training pipeline.

---

##  How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Fatiha-maryam/Islamabad_aqi_prediction.git
cd Islamabad_aqi_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
MONGODB_URI=your_mongodb_connection_string
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# 4. Run the dashboard
streamlit run dashboard/app.py
```

---

##  Automated Pipelines (GitHub Actions)

| Pipeline | Schedule | Purpose |
|----------|----------|---------|
| **Feature Pipeline** | Every 5 hours (`0 */5 * * *`) | Fetches API data → engineers features → stores in `aqi_features_v2` |
| **Training Data Prep** | Weekly (Sunday 1:00 AM UTC) | Computes targets from `aqi_features_v2` → updates `aqi_training_data` |
| **Training Pipeline** | Weekly (Sunday 2:00 AM UTC) | Loads `aqi_training_data` → trains models → registers best in DagsHub MLflow |

All pipelines can also be triggered manually via the GitHub Actions UI.

---

##  Dashboard Features

- **Current AQI** – most recent hourly reading
- **24h / 48h / 72h forecasts** – using the latest registered models
- **Health alerts** – colour‑coded AQI categories and recommendations
- **7‑day trend chart** – historical AQI with prediction markers

---

##  Author

**Fatiha Maryam**  
[GitHub](https://github.com/Fatiha-maryam) · [LinkedIn](https://www.linkedin.com/in/fatiha-maryam)

---

##  License

This project is for internship / academic evaluation. All rights reserved.
