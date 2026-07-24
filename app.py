"""
Web App — AQI Predictor Dashboard
====================================
Step 4 of the AQI Predictor project.

What this app does:
1. Loads the trained model (+ scaler + feature list) from the Model
   Registry (Hopsworks if configured, else the local models/ folder).
2. Loads the latest features (Hopsworks feature store or local CSV).
3. Computes a 3-day-ahead AQI prediction from the most recent reading.
4. Shows a dashboard with current AQI, 3-day forecast, historical trend,
   hazard alerts, and feature importance.

Run:
  streamlit run app.py
"""

import os
import json
from pathlib import Path

import joblib
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
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"


@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
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
            return df[df["city"] == CITY_NAME].sort_values("timestamp")
        except Exception as e:
            st.warning(f"Could not load from Hopsworks ({e}); using local CSV instead.")

    if not LOCAL_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOCAL_CSV)
    return df[df["city"] == CITY_NAME].sort_values("timestamp")


@st.cache_resource
def load_model():
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
            mr = project.get_model_registry()
            hw_model = mr.get_model("aqi_forecast_model")
            model_dir = hw_model.download()
            return joblib.load(Path(model_dir) / "best_model.joblib"), None
        except Exception as e:
            st.warning(f"Could not load model from Hopsworks ({e}); using local model instead.")

    model_path = MODELS_DIR / "best_model.joblib"
    metrics_path = MODELS_DIR / "metrics.json"
    if not model_path.exists():
        return None, None

    bundle = joblib.load(model_path)
    metadata = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return bundle, metadata


def predict_forecast(bundle: dict, latest_row: pd.Series) -> float:
    model = bundle["model"]
    scaler = bundle.get("scaler")
    features = bundle["features"]

    X = latest_row[features].fillna(0).to_frame().T

    model_type = type(model).__name__
    if scaler is not None and model_type in ("Ridge", "MLPRegressor"):
        X = scaler.transform(X)

    return float(model.predict(X)[0])


def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")
    st.title(f"🌫️ AQI Predictor — {CITY_NAME}")
    st.caption("3-day air quality forecast, powered by a serverless ML pipeline.")

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

    bundle, metadata = load_model()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current AQI")
        st.markdown(
            f"<h1 style='color:{current_color}'>{current_aqi:.0f}</h1>"
            f"<p style='color:{current_color}; font-weight:bold'>{current_label}</p>",
            unsafe_allow_html=True,
        )
        st.caption(f"Last updated: {latest['timestamp']}")

    with col2:
        st.subheader("3-Day Forecast")
        if bundle is None:
            st.info(
                "Model not trained yet. Run training_pipeline.py once enough "
                "hourly data has accumulated (see README)."
            )
        else:
            forecast_aqi = predict_forecast(bundle, latest)
            forecast_label, forecast_color = categorize_aqi(forecast_aqi)
            st.markdown(
                f"<h1 style='color:{forecast_color}'>{forecast_aqi:.0f}</h1>"
                f"<p style='color:{forecast_color}; font-weight:bold'>{forecast_label}</p>",
                unsafe_allow_html=True,
            )
            if metadata:
                st.caption(
                    f"Model: {metadata['selected_model']} "
                    f"(RMSE={metadata['selected_model_metrics']['rmse']:.1f}, "
                    f"R²={metadata['selected_model_metrics']['r2']:.2f})"
                )

            worst_case = max(current_aqi, forecast_aqi)
            if worst_case >= HAZARD_THRESHOLD:
                st.error(
                    f"⚠️ Hazard alert: AQI is expected to reach '{categorize_aqi(worst_case)[0]}' "
                    "levels. Consider limiting outdoor activity, especially for sensitive groups."
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

    if metadata and metadata.get("selected_model") == "random_forest" and bundle is not None:
        st.subheader("What's driving this forecast? (Feature importance)")
        rf = bundle["model"]
        importances = pd.Series(rf.feature_importances_, index=bundle["features"]).sort_values(ascending=False).head(10)
        fig2 = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h"))
        fig2.update_layout(height=350, xaxis_title="Importance", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View raw feature data"):
        st.dataframe(df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()