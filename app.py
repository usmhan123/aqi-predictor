"""
Web App — AQI Predictor Dashboard
====================================
Step 4 of the AQI Predictor project.

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

AQI_CATEGORIES = [
    (0, 50, "Good", "#4ADE9A"),
    (51, 100, "Moderate", "#F5C24D"),
    (101, 150, "Unhealthy for Sensitive Groups", "#F5934D"),
    (151, 200, "Unhealthy", "#F0576B"),
    (201, 300, "Very Unhealthy", "#B564E8"),
    (301, 500, "Hazardous", "#E23D6B"),
]
HAZARD_THRESHOLD = 150

BG = "#0A0E17"
SURFACE = "rgba(255,255,255,0.035)"
BORDER = "rgba(255,255,255,0.09)"
TEXT = "#E8ECF4"
TEXT_DIM = "#8B93A7"


def categorize_aqi(aqi: float):
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


def add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must match training_pipeline.py's feature engineering exactly."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_lag_24h"] = df["aqi_us"].shift(24)
    df["aqi_rolling_24h_mean"] = df["aqi_us"].rolling(window=24, min_periods=1).mean()
    return df


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {BG}; color: {TEXT}; }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .hero-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.6rem; letter-spacing: -0.02em; margin-bottom: 0.1rem; color: {TEXT}; }}
    .hero-sub {{ font-family: 'Inter', sans-serif; color: {TEXT_DIM}; font-size: 1rem; margin-bottom: 2rem; }}
    .section-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: {TEXT_DIM}; margin-bottom: 0.6rem; }}
    .aqi-hero {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.6rem; }}
    .aqi-hero-number {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 4.2rem; line-height: 1; margin-bottom: 0.4rem; }}
    .aqi-pill {{ display: inline-block; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.85rem; padding: 0.35rem 0.9rem; border-radius: 999px; margin-bottom: 0.7rem; }}
    .aqi-caption {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {TEXT_DIM}; }}
    .scale-wrap {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.4rem 1.6rem 2.2rem 1.6rem; margin-bottom: 1.8rem; position: relative; }}
    .scale-bar {{ height: 10px; border-radius: 999px; background: linear-gradient(90deg, #4ADE9A 0%, #4ADE9A 10%, #F5C24D 10%, #F5C24D 20%, #F5934D 20%, #F5934D 30%, #F0576B 30%, #F0576B 40%, #B564E8 40%, #B564E8 60%, #E23D6B 60%, #E23D6B 100%); position: relative; margin-top: 0.4rem; }}
    .scale-marker {{ position: absolute; top: -8px; width: 2px; height: 26px; background: {TEXT}; transform: translateX(-1px); }}
    .scale-marker-label {{ position: absolute; top: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {TEXT_DIM}; transform: translateX(-50%); white-space: nowrap; }}
    .scale-ticks {{ display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {TEXT_DIM}; margin-top: 0.5rem; }}
    .forecast-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.3rem 1.2rem; text-align: left; height: 100%; }}
    .forecast-day {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: {TEXT_DIM}; margin-bottom: 0.5rem; }}
    .forecast-number {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.2rem; line-height: 1; margin-bottom: 0.5rem; }}
    .forecast-model {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {TEXT_DIM}; margin-top: 0.6rem; }}
    .not-trained {{ font-family: 'Inter', sans-serif; color: {TEXT_DIM}; font-size: 0.85rem; padding: 1.5rem 0; }}
    .hazard-banner {{ background: rgba(226,61,107,0.12); border: 1px solid rgba(226,61,107,0.4); border-radius: 14px; padding: 1rem 1.3rem; color: #FFB3C6; font-family: 'Inter', sans-serif; font-size: 0.92rem; margin-bottom: 1.6rem; }}
    </style>
    """, unsafe_allow_html=True)


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


# ttl=1800 (30 min): without this, Streamlit caches the loaded models
# forever until the app process restarts, so newly-retrained models
# uploaded to Hopsworks would never be picked up by a long-running
# deployed app. This was flagged in review and is fixed here.
@st.cache_resource(ttl=1800)
def load_all_models():
    results = {}

    if HOPSWORKS_API_KEY:
        try:
            project = _get_hopsworks_project()
            mr = project.get_model_registry()
            for horizon_name in HORIZONS:
                try:
                    candidates = mr.get_models(name=f"aqi_forecast_model_{horizon_name}")
                    hw_model = max(candidates, key=lambda m: m.version)
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


def render_scale_bar(current_aqi: float, forecast_values: dict):
    def pos(v):
        return max(0, min(100, v / 500 * 100))

    markers_html = f'<div class="scale-marker" style="left:{pos(current_aqi)}%;"></div>'
    markers_html += f'<div class="scale-marker-label" style="left:{pos(current_aqi)}%;">NOW &middot; {current_aqi:.0f}</div>'

    for key in HORIZONS:
        if key in forecast_values:
            v = forecast_values[key]
            markers_html += f'<div class="scale-marker" style="left:{pos(v)}%; opacity:0.55;"></div>'

    st.markdown(f"""
    <div class="scale-wrap">
        <div class="section-label">AQI Scale — 0 to 500</div>
        <div class="scale-bar">{markers_html}</div>
        <div class="scale-ticks">
            <span>0</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300</span><span>500</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="\U0001F32B\uFE0F", layout="wide")
    inject_css()

    st.markdown(f'<div class="hero-title">AQI Predictor \u2014 {CITY_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Day-by-day 3-day air quality forecast, powered by a serverless ML pipeline.</div>', unsafe_allow_html=True)

    df = load_history()
    if df.empty:
        st.error("No feature data found yet.")
        st.stop()

    df = add_persistence_features(df)
    latest = df.iloc[-1]
    current_aqi = float(latest["aqi_us"])
    current_label, current_color = categorize_aqi(current_aqi)

    all_models = load_all_models()

    st.markdown(f"""
    <div class="aqi-hero">
        <div class="section-label">Current AQI</div>
        <div class="aqi-hero-number" style="color:{current_color};">{current_aqi:.0f}</div>
        <div class="aqi-pill" style="background:{current_color}22; color:{current_color}; border:1px solid {current_color}55;">{current_label}</div>
        <div class="aqi-caption">Last updated {latest['timestamp']}</div>
    </div>
    """, unsafe_allow_html=True)

    forecast_values = {}
    for horizon_key in HORIZONS:
        bundle, metadata = all_models.get(horizon_key, (None, None))
        if bundle is not None:
            forecast_values[horizon_key] = predict_forecast(bundle, latest)

    render_scale_bar(current_aqi, forecast_values)

    st.markdown('<div class="section-label">3-Day Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (horizon_key, (day_label, hours)) in zip(cols, HORIZONS.items()):
        bundle, metadata = all_models.get(horizon_key, (None, None))
        with col:
            if bundle is None:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-day">{day_label} &middot; +{hours}h</div>
                    <div class="not-trained">Not trained yet — needs more data.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                forecast_aqi = forecast_values[horizon_key]
                label, color = categorize_aqi(forecast_aqi)
                model_caption = ""
                if metadata:
                    model_caption = f"{metadata['selected_model']} &middot; RMSE {metadata['selected_model_metrics']['rmse']:.1f}"
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-day">{day_label} &middot; +{hours}h</div>
                    <div class="forecast-number" style="color:{color};">{forecast_aqi:.0f}</div>
                    <div class="aqi-pill" style="background:{color}22; color:{color}; border:1px solid {color}55;">{label}</div>
                    <div class="forecast-model">{model_caption}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    worst_case = max([current_aqi] + list(forecast_values.values())) if forecast_values else current_aqi
    if worst_case >= HAZARD_THRESHOLD:
        worst_label, _ = categorize_aqi(worst_case)
        st.markdown(f"""
        <div class="hazard-banner">
            \u26A0\uFE0F <strong>Hazard alert:</strong> AQI is expected to reach '{worst_label}' levels
            within the next 3 days. Consider limiting outdoor activity, especially for sensitive groups.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Historical AQI Trend</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi_us"],
        mode="lines", name="AQI", line=dict(color="#7DD3C0", width=2),
    ))
    for lo, hi, label, color in AQI_CATEGORIES:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.06, line_width=0)
    fig.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_DIM),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Time"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI (US)"),
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    for horizon_key, (day_label, _) in HORIZONS.items():
        bundle, metadata = all_models.get(horizon_key, (None, None))
        shap_importance = (metadata or {}).get("shap_importance")
        if shap_importance and bundle is not None:
            st.markdown(f'<div class="section-label">What\'s driving the {day_label} forecast (SHAP)</div>', unsafe_allow_html=True)
            importances = pd.Series(shap_importance).sort_values(ascending=False).head(8)
            fig2 = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker=dict(color="#7DD3C0")))
            fig2.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_DIM),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Mean |SHAP value|"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=10, l=10, r=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View raw feature data"):
        st.dataframe(df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()