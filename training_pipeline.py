"""
Training Pipeline — AQI Predictor
===================================
Step 2 of the AQI Predictor project.

What this script does:
1. Loads accumulated (features, target) data from the Feature Store
   (Hopsworks if configured, otherwise the local data/features.csv).
2. Builds the REAL prediction target: AQI 3 days (72 hourly readings)
   ahead of each row.
3. Trains and evaluates FOUR models:
   - Ridge Regression (statistical baseline)
   - Random Forest Regressor
   - MLP Regressor (scikit-learn neural network)
   - TensorFlow/Keras deep neural network
4. Evaluates all with RMSE, MAE, and R^2 on a held-out time-based test split.
5. Picks the best model by RMSE and saves it (+ metadata) to Hopsworks
   Model Registry if configured, else a local `models/` folder.
6. Prints Random Forest feature importances.

Usage:
  python training_pipeline.py
"""

import os
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

load_dotenv()

CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

DATA_DIR = Path(__file__).parent / "data"
LOCAL_CSV = DATA_DIR / "features.csv"
MODELS_DIR = Path(__file__).parent / "models"

HORIZON_HOURS = 72
MIN_ROWS_REQUIRED = HORIZON_HOURS + 48

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3", "no",
    "temp_c", "humidity", "pressure", "wind_speed", "wind_deg", "clouds_pct",
    "hour", "day", "month", "day_of_week", "aqi_change_rate",
]
TARGET_COLUMN = "aqi_us"


def load_data() -> pd.DataFrame:
    """Load historical features from Hopsworks if configured, else local CSV."""
    if HOPSWORKS_API_KEY:
        try:
            import hopsworks
            cert_dir = Path(__file__).parent / "hopsworks_certs"
            cert_dir.mkdir(exist_ok=True)
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT,
                host=HOPSWORKS_HOST,
                port=443,
                cert_folder=str(cert_dir),
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features", version=1)
            df = fg.read()
            print(f"Loaded {len(df)} rows from Hopsworks feature group 'aqi_features'.")
            return df
        except Exception as e:
            print(f"[warn] Could not load from Hopsworks ({e}); falling back to local CSV.")

    if not LOCAL_CSV.exists():
        sys.exit(f"ERROR: No data found at {LOCAL_CSV}. Run feature_pipeline.py first (and let it run hourly for a while).")

    df = pd.read_csv(LOCAL_CSV)
    df = df[df["city"] == CITY_NAME]
    print(f"Loaded {len(df)} rows from local CSV for city={CITY_NAME}.")
    return df


def build_forecast_target(df: pd.DataFrame) -> pd.DataFrame:
    """Shift aqi_us backwards by HORIZON_HOURS rows to create a proper 3-day-ahead target."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_aqi_3d"] = df[TARGET_COLUMN].shift(-HORIZON_HOURS)
    df = df.dropna(subset=["target_aqi_3d"])
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by time order (not randomly) -- essential for time series data."""
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["target_aqi_3d"]
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    y_test = test_df["target_aqi_3d"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = {}

    # --- Ridge Regression baseline ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    results["ridge_regression"] = evaluate(y_test, ridge.predict(X_test_scaled))
    models["ridge_regression"] = ridge

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate(y_test, rf.predict(X_test))
    models["random_forest"] = rf

    # --- Small neural network (scikit-learn MLP) ---
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
    mlp.fit(X_train_scaled, y_train)
    results["mlp_neural_net"] = evaluate(y_test, mlp.predict(X_test_scaled))
    models["mlp_neural_net"] = mlp

    # --- Deep learning model (TensorFlow/Keras) ---
    try:
        import tensorflow as tf

        tf.random.set_seed(42)
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        tf_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        tf_model.fit(
            X_train_scaled, y_train,
            validation_split=0.15,
            epochs=100,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0,
        )
        tf_preds = tf_model.predict(X_test_scaled, verbose=0).flatten()
        results["tensorflow_nn"] = evaluate(y_test, tf_preds)
        models["tensorflow_nn"] = tf_model
    except ImportError:
        print("[info] TensorFlow not installed -- skipping the deep learning model. "
              "Install with `pip install tensorflow` to include it.")

    return results, models, scaler


def print_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print("\nFeature importance (Random Forest):")
    for name, score in ranked[:10]:
        print(f"  {name:20s} {score:.4f}")


def _save_model_files(model, scaler, model_name: str):
    """
    Save a trained model to MODELS_DIR, handling both scikit-learn models
    (via joblib) and TensorFlow/Keras models (via their native .keras
    format, since joblib cannot serialize Keras models).
    """
    MODELS_DIR.mkdir(exist_ok=True)
    is_keras = hasattr(model, "save") and "keras" in str(type(model)).lower()

    if is_keras:
        model.save(MODELS_DIR / "best_model.keras")
        joblib.dump({"scaler": scaler, "features": FEATURE_COLUMNS}, MODELS_DIR / "best_model_meta.joblib")
        return "keras"
    else:
        joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, MODELS_DIR / "best_model.joblib")
        return "sklearn"


def store_model_locally(model, scaler, model_name: str, metrics: dict, all_results: dict):
    model_format = _save_model_files(model, scaler, model_name)

    metadata = {
        "selected_model": model_name,
        "selected_model_metrics": metrics,
        "all_model_results": all_results,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "feature_columns": FEATURE_COLUMNS,
        "model_format": model_format,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model ('{model_name}', format={model_format}) to {MODELS_DIR}")
    print(f"Saved metrics/metadata to {MODELS_DIR / 'metrics.json'}")


def store_model_to_hopsworks(model, scaler, model_name: str, metrics: dict):
    """Push the best model to the Hopsworks Model Registry."""
    import hopsworks

    cert_dir = Path(__file__).parent / "hopsworks_certs"
    cert_dir.mkdir(exist_ok=True)
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        cert_folder=str(cert_dir),
    )
    mr = project.get_model_registry()

    model_format = _save_model_files(model, scaler, model_name)
    metadata = {
        "selected_model": model_name,
        "selected_model_metrics": metrics,
        "horizon_hours": HORIZON_HOURS,
        "feature_columns": FEATURE_COLUMNS,
        "model_format": model_format,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    hw_model = mr.python.create_model(
        name="aqi_forecast_model",
        metrics=metrics,
        description=f"AQI 3-day forecast model ({model_name}, format={model_format})",
    )
    hw_model.save(str(MODELS_DIR))
    print(f"Uploaded model to Hopsworks Model Registry as 'aqi_forecast_model'.")


def main():
    df = load_data()
    df = build_forecast_target(df)

    if len(df) < MIN_ROWS_REQUIRED:
        sys.exit(
            f"Not enough data yet to train: have {len(df)} usable rows after building the "
            f"3-day-ahead target, need at least {MIN_ROWS_REQUIRED}.\n"
            f"This means you need roughly {MIN_ROWS_REQUIRED + HORIZON_HOURS} total hourly "
            f"readings collected -- let feature_pipeline.py / the GitHub Action keep running "
            f"and try again in a few days. This is expected, not an error in your code."
        )

    train_df, test_df = time_based_split(df)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    results, models, scaler = train_and_evaluate(train_df, test_df)

    print("\nModel comparison (on held-out, time-based test split):")
    for name, metrics in results.items():
        print(f"  {name:20s} RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (lowest RMSE)")

    if best_name == "random_forest":
        print_feature_importance(best_model, FEATURE_COLUMNS)

    if HOPSWORKS_API_KEY:
        try:
            store_model_to_hopsworks(best_model, scaler, best_name, results[best_name])
            return
        except Exception as e:
            print(f"[warn] Hopsworks model store failed ({e}); falling back to local storage.")

    store_model_locally(best_model, scaler, best_name, results[best_name], results)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE ON SHAP: once you have a trained model and want proper SHAP-based
# explanations, install shap separately (`pip install shap`) and run:
#
#   import shap
#   explainer = shap.TreeExplainer(best_model)
#   shap_values = explainer.shap_values(X_test)
#   shap.summary_plot(shap_values, X_test)
# ---------------------------------------------------------------------------