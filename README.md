# Pearls AQI Predictor

A system that provides a 3-day Air Quality Index (AQI) forecast for Wah Cantt. The entire pipeline is serverless, data is automatically collected, features are generated, models are trained, and a live dashboard displays the AQI.

**Live Dashboard:** https://aqi-predictor-lqhhpcnuwkbnl9qf39qzu8.streamlit.app
**GitHub Repo:** https://github.com/usmhan123/aqi-predictor

## What the Project Does

This project predicts the AQI for the next 3 days (Day 1, Day 2, and Day 3) similar to a weather app, but instead of showing a single number, it displays separate predictions for each of the three days.

## How It Works

1. **Feature Pipeline** Collects pollution and weather data from OpenWeather every hour, generates features (such as hour, day, month, and AQI change rate), and stores them in the Hopsworks Feature Store.
2. **Training Pipeline** Runs daily, retrieves the data, trains 4 models (Ridge Regression, Random Forest, MLP, and TensorFlow), selects the best model based on RMSE, and saves it to the Hopsworks Model Registry. This process is done separately for Day 1, Day 2, and Day 3.
3. **Dashboard** Built with Streamlit. It displays the current AQI, the 3-day forecast, and uses SHAP to show which factors have the greatest impact on the forecast.
4. **API** A small FastAPI backend is also included, which provides the same data in JSON format.
5. **Automation** GitHub Actions automatically runs the feature pipeline every hour and the training pipeline every day.

## Tech Stack

* Python, Scikit-learn, TensorFlow
* Hopsworks (Feature Store + Model Registry)
* GitHub Actions (automation)
* Streamlit + FastAPI (web app)
* OpenWeather + Open-Meteo (data sources)
* SHAP (feature importance)

## Problems Encountered and How They Were Fixed

* **Weather backfill issue:** When historical data was backfilled, the current weather was being used for every historical row instead of the actual weather at that timestamp. This was fixed Open-Meteo is now used to match the correct weather data with each timestamp.
* **SHAP was not showing on the dashboard:** The reason was that the dashboard was loading an older model version that did not include SHAP. This was fixed the dashboard now always loads the latest model version.

## Current Limitations (Honestly)

* The data is still being collected (~250–480 rows so far), so the accuracy (R²) of some models is still low. As more data becomes available, the models should improve.
* The Hopsworks free monthly budget ran out once due to the compute limit. Therefore, the code is designed so that if Hopsworks is unavailable, it automatically falls back to local files the dashboard does not crash.
* FastAPI is used instead of Flask (both are allowed in the requirements).
* TensorFlow is not included in the deployed dashboard because TensorFlow is not currently available for Python 3.14 however, TensorFlow is used during training on the local machine.

## How to Run (Setup)

```bash
pip install -r requirements.txt
cp .env.example .env
python feature_pipeline.py --backfill 15
python feature_pipeline.py
python training_pipeline.py
streamlit run app.py
uvicorn api:app --reload
```

## Files

* `feature_pipeline.py` Collects and saves data
* `training_pipeline.py` Trains the models
* `app.py` Streamlit dashboard
* `api.py` FastAPI backend
* `eda_analysis.py` Generates data charts
* `.github/workflows/` Automation workflow files
