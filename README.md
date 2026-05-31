# Islamabad_aqi_prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://islamabad-aqi-prediction.streamlit.app/) 
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Tracking-orange)](https://dagshub.com/Fatiha-maryam/Islamabad_aqi_prediction)

> **End‑to‑end serverless pipeline** that forecasts Air Quality Index (AQI) for the next 24h, 48h and 72h for Islamabad, Pakistan.  
> Data is fetched hourly, features are engineered automatically, models are retrained daily, and a live dashboard displays the latest forecasts.

---

##  Live Demo

 [**https://islamabad-aqi-forecast.streamlit.app**](https://islamabad-aqi-prediction.streamlit.app/)

---

##  Overview

| Component | Technology | Automation |
|-----------|------------|------------|
| **Data source** | Open‑Meteo Air Quality & Weather API | – |
| **Feature pipeline** | Python + MongoDB | Every 5 hours (GitHub Actions) |
| **Training pipeline** | Python + MLflow (DagsHub) | Daily at 02:00 UTC (GitHub Actions) |
| **Model registry** | DagsHub MLflow | Stores latest model versions |
| **Dashboard** | Streamlit | Serves live predictions |

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
| 24h     | CatBoost  | 10.65 | 14.30 | 0.817 |
| 48h     | CatBoost  | 14.20 | 19.04 | 0.674 |
| 72h     | StackingRegressor | 16.81 | 21.73 | 0.574 |

> Metrics are updated daily after each training run.


##  Repository Structure

```bash
├── .github/workflows/          # GitHub Actions CI/CD
│   ├── feature-pipeline.yml    # every 5h
│   └── training-pipeline.yml   # daily at 2am
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── src/
│   ├── feature_pipeline.py     # fetch, clean, feature engineering, store to MongoDB
│   └── training_pipeline.py    # load data, train models, register best in MLflow
├── models/                     # local backups (ignored by git)
├── requirements.txt            # all Python dependencies
└── README.md 
```


##  Features Engineered

For each hour, the pipeline creates:

- **Lag features** (`lag1` … `lag72`)
- **Rolling averages** (`aqi_ma6`, `aqi_ma12`, `aqi_ma24`)
- **Rolling standard deviation** (`aqi_std12`)
- **Trends** (`aqi_trend_3h`, `aqi_trend_6h`, `aqi_trend_24h`)
- **Daily min / max / range** (`aqi_min_24h`, `aqi_max_24h`, `aqi_range_24h`)
- **Time features** (hour, day_of_week, month, cyclic encoding)
- **Season** (winter, spring, summer, autumn)
- **Rush hour & smog season flags**
- **PM2.5** lag and rolling averages
- **Weather**: temperature, humidity, wind speed, rain code

All features are stored in **MongoDB Atlas** (one row per hour) and served to the training pipeline.

##  How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Fatiha-maryam/Islamabad_aqi_prediction.git
cd Islamabad_aqi_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (create .env file)
MONGODB_URI=your_mongodb_uri
MLFLOW_TRACKING_USERNAME=Fatiha-maryam
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# 4. Run the dashboard
streamlit run dashboard/app.py 
```


##  Automated Pipelines (GitHub Actions)

- **Feature pipeline**: runs every 5 hours (schedule `0 */5 * * *`). Fetches the last 4 days of data, engineers features, and inserts new rows into MongoDB.
- **Training pipeline**: runs daily at 02:00 UTC. Loads all historical features, trains models, selects the best per horizon, and registers them in DagsHub MLflow.

Both can also be triggered manually via the GitHub Actions UI.

##  Dashboard Features

- **Current AQI** – most recent hourly reading
- **24h / 48h / 72h forecasts** – using the latest registered models
- **Health alerts** – colour‑coded AQI categories and recommendations
- **7‑day trend chart** – historical AQI with prediction markers
- **Model performance table** – best model per horizon (MAE, RMSE, R²)
- **MAE comparison bar chart** – all models side‑by‑side
- **Feature importance** – bar plots for each horizon

##  Notes

- The dashboard loads the **latest model version** (`latest`) from DagsHub MLflow, so it always uses the most recent retrained model.
- MongoDB stores all historical features – no data is lost.
- All secrets (MongoDB URI, DagsHub token) are stored as GitHub Actions secrets and Streamlit Cloud secrets – never hard‑coded.

##  Author

**Fatiha Maryam**  
[GitHub](https://github.com/Fatiha-maryam) · [LinkedIn](https://www.linkedin.com/in/fatiha-maryam)

## License

This project is for internship / academic evaluation. All rights reserved.