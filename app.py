"""
Web App — AQI Predictor Dashboard
====================================
Step 4 of the AQI Predictor project.

What this app does:
1. Loads THREE trained models -- one each for Day 1, Day 2, and Day 3
   ahead -- from the Model Registry (Hopsworks if configured, else the
   local models/day1, models/day2, models/day3 folders that
   training_pipeline.py produced). Handles both scikit-learn models
   (joblib) and TensorFlow/Keras models (.keras format).
2. Loads the latest features (Hopsworks feature store or local CSV).
3. Computes an actual day-by-day 3-day AQI forecast (like a weather app),
   matching the project's "predict AQI in the next 3 days" requirement.
4. Shows a dashboard with:
   - Current AQI + category
   - Day 1 / Day 2 / Day 3 forecast cards
   - Historical AQI trend chart
   - Hazard alert banner if current or any forecasted day is unhealthy+
   - SHAP-based feature importance (works for any of the 4 model types)

Run:
  streamlit run app.py
"""

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

DATA_DIR = Path(__file__).parent / "data"
LOCAL_CSV = DATA_DIR / "features.csv"
MODELS_DIR = Path(__file__).parent / "models"

HORIZONS = {"day1": ("Day 1", 24), "day2": ("Day 2", 48), "day3": ("Day 3", 72)}

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3", "no",
    "temp_c", "humidity", "pressure", "wind_speed", "wind_deg", "clouds_pct",
    "hour", "day", "month", "day_of_week", "aqi_change_rate",
]

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]
HAZARD_THRESHOLD = 150


def categorize_aqi(aqi: float):
    """Categorize an AQI value using standard US EPA breakpoints."""
    if aqi <= 50:
        return AQI_CATEGORIES[0][2], AQI_CATEGORIES[0][3]
    elif aqi <= 100:
        return AQI_CATEGORIES[1][2], AQI_CATEGORIES[1][3]
    elif aqi <= 150:
        return AQI_CATEGORIES[2][2], AQI_CATEGORIES[2][3]
    elif aqi <= 200:
        return AQI_CATEGORIES[3][2], AQI_CATEGORIES[3][3]
    elif aqi <= 300:
        return AQI_CATEGORIES[4][2], AQI_CATEGORIES[4][3]
    else:
        return AQI_CATEGORIES[5][2], AQI_CATEGORIES[5][3]


def _get_hopsworks_project():
    import hopsworks
    cert_dir = Path(__file__).parent / "hopsworks_certs"
    cert_dir.mkdir(exist_ok=True)
    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        cert_folder=str(cert_dir),
    )


@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
    if HOPSWORKS_API_KEY:
        try:
            project = _get_hopsworks_project()
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features", version=1)
            df = fg.read()
            return df[df["city"] == CITY_NAME].sort_values("timestamp")
        except Exception as e:
            st.warning(f"Could not load from Hopsworks ({e}); using local CSV instead.")

    if not LOCAL_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOCAL_CSV)
    return df[df["city"] == CITY_NAME].sort_values("timestamp")


def _load_bundle_from_dir(model_dir: Path):
    """Load a saved model bundle, handling both sklearn (joblib) and Keras formats."""
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


@st.cache_resource
def load_all_models():
    """Load all three horizon models (day1/day2/day3), whichever are available."""
    results = {}

    if HOPSWORKS_API_KEY:
        try:
            project = _get_hopsworks_project()
            mr = project.get_model_registry()
            for horizon_name in HORIZONS:
                try:
                    hw_model = mr.get_model(f"aqi_forecast_model_{horizon_name}")
                    model_dir = Path(hw_model.download())
                    results[horizon_name] = _load_bundle_from_dir(model_dir)
                except Exception:
                    results[horizon_name] = (None, None)
            if any(b is not None for b, _ in results.values()):
                return results
        except Exception as e:
            st.warning(f"Could not load models from Hopsworks ({e}); using local models instead.")

    results = {}
    for horizon_name in HORIZONS:
        horizon_dir = MODELS_DIR / horizon_name
        if horizon_dir.exists():
            results[horizon_name] = _load_bundle_from_dir(horizon_dir)
        else:
            results[horizon_name] = (None, None)
    return results


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


def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="\U0001F32B\uFE0F", layout="wide")
    st.title(f"\U0001F32B\uFE0F AQI Predictor \u2014 {CITY_NAME}")
    st.caption("Day-by-day 3-day air quality forecast, powered by a serverless ML pipeline.")

    df = load_history()
    if df.empty:
        st.error(
            "No feature data found yet. Make sure feature_pipeline.py (or the "
            "hourly GitHub Action) has run at least once."
        )
        st.stop()

    latest = df.iloc[-1]
    current_aqi = float(latest["aqi_us"])
    current_label, current_color = categorize_aqi(current_aqi)

    all_models = load_all_models()

    st.subheader("Current AQI")
    st.markdown(
        f"<h1 style='color:{current_color}'>{current_aqi:.0f}</h1>"
        f"<p style='color:{current_color}; font-weight:bold'>{current_label}</p>",
        unsafe_allow_html=True,
    )
    st.caption(f"Last updated: {latest['timestamp']}")

    st.divider()
    st.subheader("3-Day Forecast")

    forecast_values = {}
    cols = st.columns(3)
    for col, (horizon_key, (day_label, hours)) in zip(cols, HORIZONS.items()):
        bundle, metadata = all_models.get(horizon_key, (None, None))
        with col:
            st.markdown(f"**{day_label}** (+{hours}h)")
            if bundle is None:
                st.info("Not trained yet \u2014 needs more data.")
            else:
                forecast_aqi = predict_forecast(bundle, latest)
                forecast_values[horizon_key] = forecast_aqi
                label, color = categorize_aqi(forecast_aqi)
                st.markdown(
                    f"<h2 style='color:{color}'>{forecast_aqi:.0f}</h2>"
                    f"<p style='color:{color}; font-weight:bold'>{label}</p>",
                    unsafe_allow_html=True,
                )
                if metadata:
                    st.caption(
                        f"{metadata['selected_model']} "
                        f"(RMSE={metadata['selected_model_metrics']['rmse']:.1f})"
                    )

    worst_case = max([current_aqi] + list(forecast_values.values())) if forecast_values else current_aqi
    if worst_case >= HAZARD_THRESHOLD:
        st.error(
            f"\u26A0\uFE0F Hazard alert: AQI is expected to reach "
            f"'{categorize_aqi(worst_case)[0]}' levels within the next 3 days. "
            "Consider limiting outdoor activity, especially for sensitive groups."
        )

    st.divider()

    st.subheader("Historical AQI Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi_us"],
        mode="lines+markers", name="AQI", line=dict(color="#1f77b4"),
    ))
    for lo, hi, label, color in AQI_CATEGORIES:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.08, line_width=0)
    fig.update_layout(xaxis_title="Time", yaxis_title="AQI (US)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    for horizon_key, (day_label, _) in HORIZONS.items():
        bundle, metadata = all_models.get(horizon_key, (None, None))
        shap_importance = (metadata or {}).get("shap_importance")
        if shap_importance and bundle is not None:
            st.subheader(f"What's driving the {day_label} forecast? (SHAP feature importance)")
            st.caption(f"Model: {metadata['selected_model']}")
            importances = pd.Series(shap_importance).sort_values(ascending=False).head(10)
            fig2 = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h"))
            fig2.update_layout(height=350, xaxis_title="Mean |SHAP value|", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View raw feature data"):
        st.dataframe(df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()