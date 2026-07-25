"""
Exploratory Data Analysis — AQI Predictor
============================================
Analyzes the accumulated feature data to surface trends, patterns, and
relationships that motivate the modeling choices in training_pipeline.py.
Produces a set of publication-quality PNG charts (for the final report)
plus a printed summary of key statistics.

Loads data from Hopsworks Feature Store if configured, else the local
data/features.csv that feature_pipeline.py has been building up.

Usage:
  python eda_analysis.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

DATA_DIR = Path(__file__).parent / "data"
LOCAL_CSV = DATA_DIR / "features.csv"
OUTPUT_DIR = Path(__file__).parent / "eda_outputs"

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy (Sensitive)", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


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
        raise SystemExit(f"No data found at {LOCAL_CSV}. Run feature_pipeline.py first.")

    df = pd.read_csv(LOCAL_CSV)
    print(f"Loaded {len(df)} rows from local CSV.")
    return df


def categorize_aqi(aqi: float) -> str:
    """Categorize an AQI value using standard US EPA breakpoints."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"City: {df['city'].iloc[0] if 'city' in df.columns else CITY_NAME}")
    print(f"Total readings: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(f"\nAQI statistics:")
    print(df["aqi_us"].describe().round(2))
    print(f"\nAQI category breakdown:")
    print(df["aqi_category"].value_counts())


def plot_aqi_trend(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamp"], df["aqi_us"], color="#1f77b4", linewidth=1.2)
    for lo, hi, label, color in AQI_CATEGORIES:
        ax.axhspan(lo, hi, color=color, alpha=0.08)
    ax.set_title(f"AQI Trend Over Time — {CITY_NAME}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("AQI (US)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.savefig(out / "01_aqi_trend.png")
    plt.close(fig)


def plot_aqi_category_distribution(df: pd.DataFrame, out: Path):
    counts = df["aqi_category"].value_counts()
    colors = {label: color for _, _, label, color in AQI_CATEGORIES}
    order = [label for _, _, label, _ in AQI_CATEGORIES if label in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, [counts[l] for l in order], color=[colors[l] for l in order], edgecolor="black")
    ax.set_title("Distribution of AQI Categories", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of readings")
    plt.xticks(rotation=20, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom")
    fig.savefig(out / "02_aqi_category_distribution.png")
    plt.close(fig)


def plot_hourly_pattern(df: pd.DataFrame, out: Path):
    hourly = df.groupby("hour")["aqi_us"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["mean"], marker="o", color="#d62728")
    ax.fill_between(hourly["hour"], hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=0.2, color="#d62728")
    ax.set_title("Average AQI by Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average AQI (± std dev)")
    ax.set_xticks(range(0, 24, 2))
    fig.savefig(out / "03_hourly_pattern.png")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out: Path):
    numeric_cols = [
        "aqi_us", "pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3", "no",
        "temp_c", "humidity", "pressure", "wind_speed", "clouds_pct",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.savefig(out / "04_correlation_heatmap.png")
    plt.close(fig)


def plot_pollutant_trends(df: pd.DataFrame, out: Path):
    pollutants = ["pm2_5", "pm10", "o3", "no2"]
    available = [p for p in pollutants if p in df.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, pollutant in zip(axes, available):
        ax.plot(df["timestamp"], df[pollutant], linewidth=1)
        ax.set_ylabel(pollutant.upper().replace("_", "."))
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Pollutant Concentration Trends", fontsize=14, fontweight="bold")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.savefig(out / "05_pollutant_trends.png")
    plt.close(fig)


def plot_weather_vs_aqi(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(df["temp_c"], df["aqi_us"], alpha=0.4, s=15, color="#ff7f0e")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("AQI (US)")
    axes[0].set_title("Temperature vs AQI")

    axes[1].scatter(df["wind_speed"], df["aqi_us"], alpha=0.4, s=15, color="#2ca02c")
    axes[1].set_xlabel("Wind speed (m/s)")
    axes[1].set_ylabel("AQI (US)")
    axes[1].set_title("Wind Speed vs AQI")

    fig.suptitle("Weather Variables vs AQI", fontsize=14, fontweight="bold")
    fig.savefig(out / "06_weather_vs_aqi.png")
    plt.close(fig)


def plot_aqi_change_rate_distribution(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["aqi_change_rate"].dropna(), bins=30, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Hour-to-Hour AQI Change Rate", fontsize=14, fontweight="bold")
    ax.set_xlabel("AQI change rate")
    ax.set_ylabel("Frequency")
    fig.savefig(out / "07_aqi_change_rate_distribution.png")
    plt.close(fig)


def main():
    df = load_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_category"] = df["aqi_us"].apply(categorize_aqi)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print_summary(df)

    if len(df) < 5:
        print(
            "\n[note] Only a few rows available so far -- charts will look sparse. "
            "They'll become much more informative as feature_pipeline.py keeps "
            "running hourly and more data accumulates. Re-run this script anytime."
        )

    print("\nGenerating charts...")
    plot_aqi_trend(df, OUTPUT_DIR)
    plot_aqi_category_distribution(df, OUTPUT_DIR)
    if len(df) >= 24:
        plot_hourly_pattern(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_pollutant_trends(df, OUTPUT_DIR)
    plot_weather_vs_aqi(df, OUTPUT_DIR)
    plot_aqi_change_rate_distribution(df, OUTPUT_DIR)

    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
    print("Use these directly in your final report.")


if __name__ == "__main__":
    main()