"""
Feature Pipeline — AQI Predictor
==================================
Step 1 of the AQI Predictor project.

Run modes:
  python feature_pipeline.py                # fetch "now" and store one row
  python feature_pipeline.py --backfill 30   # backfill last 30 days

BACKFILL NOTE: historical weather for backfilled rows is fetched from
Open-Meteo's free historical weather archive (no API key needed) and
matched to each pollution reading's actual timestamp.
"""

import os
import sys
import argparse
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
LAT = float(os.getenv("LAT", "33.6844"))
LON = float(os.getenv("LON", "73.0479"))
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

DATA_DIR = Path(__file__).parent / "data"
LOCAL_CSV = DATA_DIR / "features.csv"

AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_current_pollution():
    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(AIR_POLLUTION_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["list"][0]


def fetch_pollution_history(start_unix, end_unix):
    params = {
        "lat": LAT, "lon": LON,
        "start": start_unix, "end": end_unix,
        "appid": OPENWEATHER_API_KEY,
    }
    resp = requests.get(AIR_POLLUTION_HISTORY_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("list", [])


def fetch_current_weather():
    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(WEATHER_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_historical_weather_lookup(start_date: dt.date, end_date: dt.date) -> dict:
    """
    Fetch real historical weather for a date range from Open-Meteo's free
    archive API (no key required), keyed by hour-truncated UTC datetime,
    shaped like an OpenWeather /weather response.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    lookup = {}
    for i, t in enumerate(hourly["time"]):
        ts = dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc)
        lookup[ts] = {
            "main": {
                "temp": hourly["temperature_2m"][i],
                "humidity": hourly["relative_humidity_2m"][i],
                "pressure": hourly["surface_pressure"][i],
            },
            "wind": {
                "speed": hourly["wind_speed_10m"][i],
                "deg": hourly["wind_direction_10m"][i],
            },
            "clouds": {"all": hourly["cloud_cover"][i]},
        }
    return lookup


def nearest_weather(lookup: dict, timestamp: dt.datetime, fallback: dict) -> dict:
    """Weather for the closest hour to `timestamp`; falls back to `fallback`
    if the lookup is empty or the timestamp is too far outside it."""
    if not lookup:
        return fallback
    hour_ts = timestamp.replace(minute=0, second=0, microsecond=0)
    if hour_ts in lookup:
        return lookup[hour_ts]
    nearest_key = min(lookup.keys(), key=lambda k: abs((k - hour_ts).total_seconds()))
    if abs((nearest_key - hour_ts).total_seconds()) > 3 * 3600:
        return fallback
    return lookup[nearest_key]


def openweather_aqi_to_us_aqi(components: dict) -> float:
    pm25 = components.get("pm2_5", 0)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo, 1)
    return 500.0


def build_feature_row(pollution: dict, weather: dict, timestamp: dt.datetime) -> dict:
    components = pollution["components"]
    aqi_us = openweather_aqi_to_us_aqi(components)

    row = {
        "timestamp": timestamp.isoformat(),
        "city": CITY_NAME,
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "nh3": components.get("nh3"),
        "no": components.get("no"),
        "temp_c": weather.get("main", {}).get("temp"),
        "humidity": weather.get("main", {}).get("humidity"),
        "pressure": weather.get("main", {}).get("pressure"),
        "wind_speed": weather.get("wind", {}).get("speed"),
        "wind_deg": weather.get("wind", {}).get("deg"),
        "clouds_pct": weather.get("clouds", {}).get("all"),
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        "day_of_week": timestamp.weekday(),
        "aqi_us": aqi_us,
    }
    return row


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_change_rate"] = df["aqi_us"].diff().fillna(0)
    return df


def store_to_hopsworks(df: pd.DataFrame):
    import hopsworks

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    float_columns = [
        "pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3", "no",
        "temp_c", "humidity", "pressure", "wind_speed", "wind_deg",
        "clouds_pct", "aqi_us", "aqi_change_rate",
    ]
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype("float64")

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

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI + weather features for AQI forecasting",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"Inserted {len(df)} row(s) into Hopsworks feature group 'aqi_features'.")


def store_locally(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    if LOCAL_CSV.exists():
        existing = pd.read_csv(LOCAL_CSV)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["city", "timestamp"], keep="last")
    else:
        combined = df
    combined.to_csv(LOCAL_CSV, index=False)
    print(f"Stored {len(df)} row(s) locally. Total rows now: {len(combined)} -> {LOCAL_CSV}")


def store_features(df: pd.DataFrame):
    if HOPSWORKS_API_KEY:
        try:
            store_to_hopsworks(df)
            return
        except Exception as e:
            print(f"[warn] Hopsworks store failed ({e}); falling back to local CSV.")
    store_locally(df)


def run_current():
    if not OPENWEATHER_API_KEY:
        sys.exit("ERROR: OPENWEATHER_API_KEY not set. Copy .env.example to .env and fill it in.")

    now = dt.datetime.now(dt.timezone.utc)
    pollution = fetch_current_pollution()
    weather = fetch_current_weather()
    row = build_feature_row(pollution, weather, now)
    df = pd.DataFrame([row])
    df = add_derived_features_with_history(df)
    store_features(df)


def add_derived_features_with_history(new_df: pd.DataFrame) -> pd.DataFrame:
    if LOCAL_CSV.exists():
        history = pd.read_csv(LOCAL_CSV)
        history = history[history["city"] == CITY_NAME]
        if not history.empty:
            last_aqi = history.sort_values("timestamp").iloc[-1]["aqi_us"]
            new_df["aqi_change_rate"] = new_df["aqi_us"] - last_aqi
            return new_df
    new_df["aqi_change_rate"] = 0
    return new_df


def run_backfill(days: int):
    if not OPENWEATHER_API_KEY:
        sys.exit("ERROR: OPENWEATHER_API_KEY not set. Copy .env.example to .env and fill it in.")

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    print(f"Backfilling pollution history from {start} to {end} ...")

    try:
        history_list = fetch_pollution_history(int(start.timestamp()), int(end.timestamp()))
    except requests.HTTPError as e:
        sys.exit(
            "Historical air-pollution request failed "
            f"({e}). Your OpenWeather plan may not include history access."
        )

    if not history_list:
        sys.exit("No historical data returned. Try a smaller --backfill window or check your plan.")

    print("Fetching matching historical weather from Open-Meteo (free, per-timestamp)...")
    try:
        weather_lookup = fetch_historical_weather_lookup(start.date(), end.date())
        print(f"  Got {len(weather_lookup)} hourly weather points.")
    except Exception as e:
        print(f"[warn] Historical weather fetch failed ({e}); falling back to current weather for all rows.")
        weather_lookup = {}

    weather_fallback = fetch_current_weather()

    rows = []
    for entry in history_list:
        ts = dt.datetime.fromtimestamp(entry["dt"], tz=dt.timezone.utc)
        weather = nearest_weather(weather_lookup, ts, weather_fallback)
        rows.append(build_feature_row(entry, weather, ts))

    df = pd.DataFrame(rows)
    df = add_derived_features(df)
    store_features(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Predictor feature pipeline")
    parser.add_argument("--backfill", type=int, default=0, help="Number of past days to backfill")
    args = parser.parse_args()

    if args.backfill > 0:
        run_backfill(args.backfill)
    else:
        run_current()