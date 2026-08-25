"""
API — AQI Predictor (FastAPI)
================================
Run:
  uvicorn api:app --reload
"""

import os
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()

CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

DATA_DIR = Path(__file__).parent / "data"
LOCAL_CSV = DATA_DIR / "features.csv"
MODELS_DIR = Path(__file__).parent / "models"

HORIZONS = {"day1": 24, "day2": 48, "day3": 72}
HAZARD_THRESHOLD = 150

app = FastAPI(title="AQI Predictor API", version="1.0.0")


def categorize_aqi(aqi: float) -> str:
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"


def add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must match training_pipeline.py's feature engineering exactly."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_lag_24h"] = df["aqi_us"].shift(24)
    df["aqi_rolling_24h_mean"] = df["aqi_us"].rolling(window=24, min_periods=1).mean()
    return df


def _get_hopsworks_project():
    import hopsworks
    cert_dir = Path(__file__).parent / "hopsworks_certs"
    cert_dir.mkdir(exist_ok=True)
    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST, port=443, cert_folder=str(cert_dir),
    )


def load_latest_features() -> Optional[pd.Series]:
    if HOPSWORKS_API_KEY:
        try:
            project = _get_hopsworks_project()
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features", version=1)
            df = fg.read()
            df = df[df["city"] == CITY_NAME].sort_values("timestamp")
            if not df.empty:
                df = add_persistence_features(df)
                return df.iloc[-1]
        except Exception:
            pass

    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
        df = df[df["city"] == CITY_NAME].sort_values("timestamp")
        if not df.empty:
            df = add_persistence_features(df)
            return df.iloc[-1]
    return None


def load_horizon_model(horizon_name: str):
    if HOPSWORKS_API_KEY:
        try:
            project = _get_hopsworks_project()
            mr = project.get_model_registry()
            candidates = mr.get_models(name=f"aqi_forecast_model_{horizon_name}")
            hw_model = max(candidates, key=lambda m: m.version)
            model_dir = Path(hw_model.download())
            return _load_bundle_from_dir(model_dir)
        except Exception:
            pass

    horizon_dir = MODELS_DIR / horizon_name
    if horizon_dir.exists():
        return _load_bundle_from_dir(horizon_dir)
    return None, None


def _load_bundle_from_dir(model_dir: Path):
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None, None
    metadata = json.loads(metrics_path.read_text())
    model_format = metadata.get("model_format", "sklearn")

    if model_format == "keras":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_dir / "best_model.keras")
        meta = joblib.load(model_dir / "best_model_meta.joblib")
        bundle = {"model": model, "scaler": meta["scaler"], "features": meta["features"]}
    else:
        bundle = joblib.load(model_dir / "best_model.joblib")

    return bundle, metadata


def predict_forecast(bundle: dict, latest_row: pd.Series) -> float:
    model = bundle["model"]
    scaler = bundle.get("scaler")
    features = bundle["features"]

    X = latest_row[features].fillna(0).to_frame().T

    model_type = type(model).__name__
    needs_scaling = scaler is not None and (
        model_type in ("Ridge", "MLPRegressor") or "keras" in str(type(model)).lower()
    )
    if needs_scaling:
        X = scaler.transform(X)

    preds = model.predict(X)
    return float(np.asarray(preds).flatten()[0])


@app.get("/")
def root():
    return {"message": "AQI Predictor API. See /docs.", "endpoints": ["/forecast", "/health"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def get_forecast():
    latest = load_latest_features()
    if latest is None:
        raise HTTPException(status_code=503, detail="No feature data available yet.")

    current_aqi = float(latest["aqi_us"])
    response = {
        "city": CITY_NAME,
        "current": {"aqi": round(current_aqi, 1), "category": categorize_aqi(current_aqi), "timestamp": str(latest["timestamp"])},
        "forecast": {},
    }

    worst_case = current_aqi
    for horizon_name, hours in HORIZONS.items():
        bundle, metadata = load_horizon_model(horizon_name)
        if bundle is None:
            response["forecast"][horizon_name] = {"status": "not_trained_yet", "horizon_hours": hours}
            continue

        forecast_aqi = predict_forecast(bundle, latest)
        worst_case = max(worst_case, forecast_aqi)
        response["forecast"][horizon_name] = {
            "horizon_hours": hours,
            "aqi": round(forecast_aqi, 1),
            "category": categorize_aqi(forecast_aqi),
            "model": metadata.get("selected_model") if metadata else None,
            "rmse": round(metadata["selected_model_metrics"]["rmse"], 2) if metadata else None,
        }

    response["hazard_alert"] = worst_case >= HAZARD_THRESHOLD
    return response