"""
Training Pipeline — AQI Predictor
===================================
Step 2 of the AQI Predictor project.

What this script does:
1. Loads accumulated (features, target) data from the Feature Store
   (Hopsworks if configured, otherwise the local data/features.csv).
2. Builds THREE separate prediction targets, matching the project's
   "next 3 days" forecast requirement:
     - Day 1 ahead  (24 hourly readings ahead)
     - Day 2 ahead  (48 hourly readings ahead)
     - Day 3 ahead  (72 hourly readings ahead)
   Training one model per horizon gives an actual day-by-day 3-day
   forecast (like a weather app), rather than a single point estimate
   for "3 days from now".
3. For each horizon, trains and evaluates FOUR models: Ridge Regression,
   Random Forest, an MLP (scikit-learn), and a TensorFlow/Keras deep
   neural network -- then keeps whichever performs best (lowest RMSE).
4. Evaluates with RMSE, MAE, and R^2 on a held-out time-based test split.
5. Saves each horizon's best model (+ metadata) to Hopsworks Model
   Registry if configured (as three separate registered models), else to
   a local `models/day1/`, `models/day2/`, `models/day3/` folder each.
6. Prints Random Forest feature importances for any horizon where RF won.

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

# One model per horizon, so the dashboard can show an actual day-by-day
# 3-day forecast rather than a single point estimate.
HORIZONS = {
    "day1": 24,
    "day2": 48,
    "day3": 72,
}
MIN_EXTRA_ROWS = 48  # need at least this many usable rows beyond the horizon shift to train+test meaningfully

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


def build_forecast_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Shift aqi_us backwards by horizon_hours rows to create the target for this horizon."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.copy()
    df["target"] = df[TARGET_COLUMN].shift(-horizon_hours)
    df = df.dropna(subset=["target"])
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by time order (not randomly) -- essential for time series data."""
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    y_test = test_df["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    results["ridge_regression"] = evaluate(y_test, ridge.predict(X_test_scaled))
    models["ridge_regression"] = ridge

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate(y_test, rf.predict(X_test))
    models["random_forest"] = rf

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
    mlp.fit(X_train_scaled, y_train)
    results["mlp_neural_net"] = evaluate(y_test, mlp.predict(X_test_scaled))
    models["mlp_neural_net"] = mlp

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
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        tf_model.fit(
            X_train_scaled, y_train,
            validation_split=0.15, epochs=100, batch_size=16,
            callbacks=[early_stop], verbose=0,
        )
        tf_preds = tf_model.predict(X_test_scaled, verbose=0).flatten()
        results["tensorflow_nn"] = evaluate(y_test, tf_preds)
        models["tensorflow_nn"] = tf_model
    except ImportError:
        print("[info] TensorFlow not installed -- skipping the deep learning model.")

    return results, models, scaler


def print_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print("  Feature importance (Random Forest):")
    for name, score in ranked[:8]:
        print(f"    {name:20s} {score:.4f}")


def _save_model_files(model, scaler, target_dir: Path):
    """Save a trained model, handling both scikit-learn (joblib) and Keras (.keras) formats."""
    target_dir.mkdir(parents=True, exist_ok=True)
    is_keras = hasattr(model, "save") and "keras" in str(type(model)).lower()

    if is_keras:
        model.save(target_dir / "best_model.keras")
        joblib.dump({"scaler": scaler, "features": FEATURE_COLUMNS}, target_dir / "best_model_meta.joblib")
        return "keras"
    else:
        joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, target_dir / "best_model.joblib")
        return "sklearn"


def train_one_horizon(df: pd.DataFrame, horizon_name: str, horizon_hours: int):
    """Train, evaluate, and save the best model for a single forecast horizon."""
    print(f"\n{'=' * 60}\nHorizon: {horizon_name} (+{horizon_hours}h)\n{'=' * 60}")

    horizon_df = build_forecast_target(df, horizon_hours)
    min_rows_required = horizon_hours + MIN_EXTRA_ROWS

    if len(horizon_df) < min_rows_required:
        print(
            f"  Not enough data yet: have {len(horizon_df)} usable rows, need {min_rows_required}. "
            f"Skipping this horizon for now -- it will train automatically once enough hourly "
            f"data has accumulated."
        )
        return None

    train_df, test_df = time_based_split(horizon_df)
    print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    results, models, scaler = train_and_evaluate(train_df, test_df)

    print("  Model comparison (held-out, time-based test split):")
    for name, metrics in results.items():
        print(f"    {name:20s} RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = models[best_name]
    print(f"  Best model for {horizon_name}: {best_name} (lowest RMSE)")

    if best_name == "random_forest":
        print_feature_importance(best_model, FEATURE_COLUMNS)

    return {
        "horizon_name": horizon_name,
        "horizon_hours": horizon_hours,
        "model": best_model,
        "scaler": scaler,
        "model_name": best_name,
        "metrics": results[best_name],
        "all_results": results,
    }


def store_locally(horizon_results: dict):
    for horizon_name, result in horizon_results.items():
        if result is None:
            continue
        target_dir = MODELS_DIR / horizon_name
        model_format = _save_model_files(result["model"], result["scaler"], target_dir)

        metadata = {
            "horizon_name": horizon_name,
            "horizon_hours": result["horizon_hours"],
            "selected_model": result["model_name"],
            "selected_model_metrics": result["metrics"],
            "all_model_results": result["all_results"],
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "feature_columns": FEATURE_COLUMNS,
            "model_format": model_format,
        }
        with open(target_dir / "metrics.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {horizon_name} model ('{result['model_name']}', format={model_format}) to {target_dir}")


def store_to_hopsworks(horizon_results: dict):
    """Push each horizon's best model to the Hopsworks Model Registry as a separate registered model."""
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

    for horizon_name, result in horizon_results.items():
        if result is None:
            continue
        target_dir = MODELS_DIR / horizon_name
        model_format = _save_model_files(result["model"], result["scaler"], target_dir)

        metadata = {
            "horizon_name": horizon_name,
            "horizon_hours": result["horizon_hours"],
            "selected_model": result["model_name"],
            "selected_model_metrics": result["metrics"],
            "feature_columns": FEATURE_COLUMNS,
            "model_format": model_format,
        }
        with open(target_dir / "metrics.json", "w") as f:
            json.dump(metadata, f, indent=2)

        hw_model = mr.python.create_model(
            name=f"aqi_forecast_model_{horizon_name}",
            metrics=result["metrics"],
            description=f"AQI forecast model for {horizon_name} (+{result['horizon_hours']}h), "
                         f"model={result['model_name']}, format={model_format}",
        )
        hw_model.save(str(target_dir))
        print(f"Uploaded {horizon_name} model to Hopsworks Model Registry as 'aqi_forecast_model_{horizon_name}'.")


def main():
    df = load_data()

    horizon_results = {}
    for horizon_name, horizon_hours in HORIZONS.items():
        horizon_results[horizon_name] = train_one_horizon(df, horizon_name, horizon_hours)

    if all(v is None for v in horizon_results.values()):
        sys.exit(
            "\nNo horizon had enough data to train yet. Let feature_pipeline.py / the "
            "GitHub Action keep running hourly and try again in a few days. This is "
            "expected, not an error in your code."
        )

    if HOPSWORKS_API_KEY:
        try:
            store_to_hopsworks(horizon_results)
            return
        except Exception as e:
            print(f"[warn] Hopsworks model store failed ({e}); falling back to local storage.")

    store_locally(horizon_results)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE ON SHAP: once you have trained models and want proper SHAP-based
# explanations, install shap separately (`pip install shap`) and run, e.g.:
#
#   import shap
#   explainer = shap.TreeExplainer(best_model)   # for a Random Forest horizon
#   shap_values = explainer.shap_values(X_test)
#   shap.summary_plot(shap_values, X_test)
# ---------------------------------------------------------------------------