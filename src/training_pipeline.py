"""
Training Pipeline: Load Features → Train Models → Select Best → Save
Location: Islamabad AQI Prediction
Runs: Once daily via GitHub Actions
Horizons: 24h, 48h, 72h
Models: XGBoost, LightGBM, CatBoost, RandomForest, StackingRegressor
Selection: Best model per horizon based on MAE, RMSE, R² (majority wins)
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()


# ============================================
# CONFIGURATION
# ============================================
FEATURE_COLS = [
    'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'lag48', 'lag72',
    'aqi_ma6', 'aqi_ma12', 'aqi_ma24', 'aqi_std12',
    'hour', 'day_of_week', 'month',
    'pm2_5', 'pm10', 'o3', 'temperature', 'wind_speed', 'rain_code'
]

HORIZONS = {
    '24h': 'target_h24',
    '48h': 'target_h48',
    '72h': 'target_h72',
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================
# MONGODB — LOAD FEATURES
# ============================================
def load_features_from_mongodb():
    """Load all features from MongoDB into a DataFrame"""

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable not set!")

    print("  Connecting to MongoDB...")
    client     = MongoClient(mongo_uri)
    db         = client["aqi_db"]
    collection = db["aqi_features"]

    cursor = collection.find({}, {"_id": 0})
    df     = pd.DataFrame(list(cursor))

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"  Loaded {len(df)} rows from MongoDB")
    print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    return df

# ============================================
# TRAIN / TEST SPLIT
# ============================================
def split_data(df, test_size=0.2):
    """Chronological train/test split — no shuffling for time series"""

    split_idx = int(len(df) * (1 - test_size))
    train     = df.iloc[:split_idx].copy()
    test      = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train)} rows | {train['datetime'].min().date()} → {train['datetime'].max().date()}")
    print(f"  Test : {len(test)} rows  | {test['datetime'].min().date()} → {test['datetime'].max().date()}")
    return train, test

# ============================================
# DEFINE MODELS
# ============================================
def get_models():
    """Define all models to train"""

    base_estimators = [
        ('xgb', xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbosity=0
        )),
        ('lgb', lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbose=-1
        )),
        ('rf', RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
    ]

    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=200, learning_rate=0.05,
            depth=6, random_seed=42, verbose=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'StackingRegressor': StackingRegressor(
            estimators=base_estimators,
            final_estimator=LinearRegression(),
            cv=5
        ),
    }
    return models

# ============================================
# EVALUATE MODEL
# ============================================
def evaluate_model(y_true, y_pred, model_name, horizon):
    """Calculate MAE, RMSE, R² metrics"""

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"    {model_name:20s} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | R²: {r2:6.3f}")

    return {
        'model':   model_name,
        'horizon': horizon,
        'mae':     round(mae, 4),
        'rmse':    round(rmse, 4),
        'r2':      round(r2, 4),
    }

# ============================================
# SELECT BEST MODEL — MAJORITY VOTE
# ============================================
def select_best_model(results_df):
    """
    Select best model using majority vote across 3 metrics:

    Rules:
    - Lowest MAE  → 1 point
    - Lowest RMSE → 1 point
    - Highest R²  → 1 point

    If one model wins 2 or 3 out of 3 → selected
    If tie in points → lowest MAE breaks the tie
    If R² is negative for all models → ignore R², decide by MAE + RMSE only
    """

    scores = {model: 0 for model in results_df['model']}

    # Rule 1: Lowest MAE wins
    best_mae_model = results_df.loc[results_df['mae'].idxmin(), 'model']
    scores[best_mae_model] += 1

    # Rule 2: Lowest RMSE wins
    best_rmse_model = results_df.loc[results_df['rmse'].idxmin(), 'model']
    scores[best_rmse_model] += 1

    # Rule 3: Highest R² wins — only if at least one model has R² > 0
    if results_df['r2'].max() > 0:
        best_r2_model = results_df.loc[results_df['r2'].idxmax(), 'model']
        scores[best_r2_model] += 1
        print(f"\n    Metric Winners:")
        print(f"       Lowest MAE  → {best_mae_model}")
        print(f"       Lowest RMSE → {best_rmse_model}")
        print(f"       Highest R²  → {best_r2_model}")
    else:
        # All R² negative — skip R² metric
        print(f"\n    Metric Winners:")
        print(f"       Lowest MAE  → {best_mae_model}")
        print(f"       Lowest RMSE → {best_rmse_model}")
        print(f"       Highest R²  → skipped (all negative)")

    print(f"\n   Scores: {scores}")

    # Find max score
    max_score  = max(scores.values())
    top_models = [m for m, s in scores.items() if s == max_score]

    if len(top_models) == 1:
        # Clear winner
        best_name = top_models[0]
    else:
        # Tie — break by lowest MAE
        print(f"  Tie between: {top_models} — breaking by lowest MAE")
        tied_df   = results_df[results_df['model'].isin(top_models)]
        best_name = tied_df.loc[tied_df['mae'].idxmin(), 'model']

    best_metrics = results_df[results_df['model'] == best_name].iloc[0]
    print(f"\n  BEST MODEL: {best_name} "
          f"(Score: {max_score}/3 | MAE: {best_metrics['mae']} | "
          f"RMSE: {best_metrics['rmse']} | R²: {best_metrics['r2']})")

    return best_name

# ============================================
# SAVE FEATURE IMPORTANCE PLOT
# ============================================
def save_feature_importance(model, model_name, horizon):
    """Save feature importance plot for tree-based models"""

    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print(f"  {model_name} has no feature importances — skipping plot")
            return

        fi_df = pd.DataFrame({
            'feature':    FEATURE_COLS,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi_df['feature'], fi_df['importance'], color='steelblue')
        ax.set_title(f'Feature Importance — {model_name} ({horizon})', fontsize=14)
        ax.set_xlabel('Importance')
        plt.tight_layout()

        path = f"{MODELS_DIR}/feature_importance_{horizon}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f" Saved feature importance → {path}")

    except Exception as e:
        print(f"  Could not save feature importance: {e}")

# ============================================
# TRAIN ALL MODELS FOR ONE HORIZON
# ============================================
def train_horizon(train, test, horizon_name, target_col):
    """Train all models for one horizon, select best by majority vote"""

    print(f"\n  {'─'*55}")
    print(f" Horizon: {horizon_name}")
    print(f"  {'─'*55}")

    X_train = train[FEATURE_COLS]
    y_train = train[target_col]
    X_test  = test[FEATURE_COLS]
    y_test  = test[target_col]

    models  = get_models()
    results = []
    trained = {}

    for model_name, model in models.items():
        try:
            print(f"\n  Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred, model_name, horizon_name)
            results.append(metrics)
            trained[model_name] = model

        except Exception as e:
            print(f" {model_name} failed: {e}")

    # Select best model by majority vote
    results_df = pd.DataFrame(results)
    best_name  = select_best_model(results_df)
    best_model = trained[best_name]

    # Save feature importance for best model
    save_feature_importance(best_model, best_name, horizon_name)

    return best_model, best_name, results_df

# ============================================
# SAVE MODEL — OVERWRITES PREVIOUS DAILY
# ============================================
def save_model(model, horizon_name, model_name):
    """
    Save best model as .pkl — overwrites previous model.
    Every daily run replaces old model with new best model.
    """

    path = f"{MODELS_DIR}/best_model_{horizon_name}.pkl"

    with open(path, 'wb') as f:
        pickle.dump({
            'model':        model,
            'model_name':   model_name,
            'horizon':      horizon_name,
            'feature_cols': FEATURE_COLS,
            'trained_at':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, f)

    print(f"Saved → {path}  (overwrites previous)")
    return path

# ============================================
# MAIN PIPELINE
# ============================================
def run_training_pipeline():
    """Main training pipeline — runs once daily"""

    print("\n" + "="*60)
    print("TRAINING PIPELINE — ISLAMABAD AQI")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load features from MongoDB
    print("\n[1/4] Loading features from MongoDB...")
    df = load_features_from_mongodb()

    # Step 2: Train/test split
    print("\n[2/4] Splitting data (80% train / 20% test)...")
    train, test = split_data(df, test_size=0.2)

    # Step 3: Train models for each horizon
    print("\n[3/4] Training models for all horizons...")
    all_metrics = []
    best_models = {}

    for horizon_name, target_col in HORIZONS.items():
        best_model, best_name, results_df = train_horizon(
            train, test, horizon_name, target_col
        )
        all_metrics.append(results_df)
        best_models[horizon_name] = (best_model, best_name)

    # Step 4: Save everything
    print("\n[4/4] Saving models and metrics...")

    for horizon_name, (model, name) in best_models.items():
        save_model(model, horizon_name, name)

    # Save all metrics to CSV — overwrites previous
    metrics_df   = pd.concat(all_metrics, ignore_index=True)
    metrics_path = f"{MODELS_DIR}/evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}  (overwrites previous)")

    # Final summary table
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n{'Horizon':>10} | {'Best Model':>20} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8}")
    print("-"*65)

    for horizon_name in HORIZONS:
        _, name  = best_models[horizon_name]
        h_df     = metrics_df[
            (metrics_df['horizon'] == horizon_name) &
            (metrics_df['model']   == name)
        ].iloc[0]
        print(f"{horizon_name:>10} | {name:>20} | "
              f"{h_df['mae']:>8.2f} | {h_df['rmse']:>8.2f} | {h_df['r2']:>8.3f}")

    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    return best_models, metrics_df


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    run_training_pipeline()