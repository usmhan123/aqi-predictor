# Pearls AQI Predictor

A machine-learning system that predicts the **Air Quality Index (AQI) for the next 3 days in Wah Cantt**. The project automatically collects pollution and weather data, generates features, trains multiple forecasting models, selects the best model for each forecast horizon, and displays the predictions through a live Streamlit dashboard.

## Live Project

**Live Dashboard:**
https://aqi-predictor-lqhhpcnuwkbnl9qf39qzu8.streamlit.app

**GitHub Repository:**
https://github.com/usmhan123/aqi-predictor

---

## What the Project Does

The system works similarly to a weather forecasting application, but instead of predicting temperature or rainfall, it predicts **AQI for three future horizons**:

* **Day 1:** approximately 24 hours ahead
* **Day 2:** approximately 48 hours ahead
* **Day 3:** approximately 72 hours ahead

Each forecast horizon has its own trained model. The best-performing model is automatically selected based on RMSE using a held-out, time-based test set.

---

## System Architecture

The project consists of five main parts:

### 1. Feature Pipeline

The feature pipeline collects:

* PM2.5
* PM10
* O₃
* NO₂
* SO₂
* CO
* NH₃
* NO
* Temperature
* Humidity
* Atmospheric pressure
* Wind speed
* Wind direction
* Cloud cover
* Hour
* Day
* Month
* Day of week
* AQI change rate

Pollution data is collected from **OpenWeather**, while historical weather data for backfilled timestamps is obtained from **Open-Meteo**.

The pipeline runs automatically every hour using GitHub Actions.

Historical data can also be backfilled in chunks. The backfill process matches weather information to the actual historical timestamp instead of incorrectly using the current weather for old records.

The pipeline supports local-only backfills so that large historical data operations do not unnecessarily consume the Hopsworks compute budget.

---

## 2. Historical Dataset

Initially, the project had only around **480 rows of data**, which resulted in weak forecasting performance.

After further development and mentor feedback, a **one-year historical backfill** was performed, increasing the dataset to approximately **8,900 rows**.

This provided a much larger training dataset and allowed a more meaningful evaluation of the forecasting models.

---

## 3. Model Training

The training pipeline creates three separate forecasting targets:

```text
Day 1 → 24 hours ahead
Day 2 → 48 hours ahead
Day 3 → 72 hours ahead
```

For each horizon, four models are evaluated:

1. Ridge Regression
2. Random Forest Regressor
3. Scikit-learn MLP Neural Network
4. TensorFlow/Keras Neural Network

The models are evaluated using:

* **RMSE**
* **MAE**
* **R²**

A **time-based train/test split** is used rather than a random split because this is a time-series forecasting problem.

The model with the lowest RMSE on the held-out test data is selected as the best model for that forecast horizon.

The training pipeline also calculates **SHAP feature importance** for the selected model so that the dashboard can explain which features are contributing most strongly to the prediction.

---

## 4. Model Storage

The selected models are stored in the **Hopsworks Model Registry** when Hopsworks is available.

The system also maintains local model files as a fallback.

If Hopsworks is unavailable or its free-tier compute budget has been exhausted, the pipeline falls back to local storage instead of causing the application to fail.

This makes the project more resilient during development and deployment.

---

## 5. Dashboard

The dashboard is built using **Streamlit**.

It displays:

* Current AQI
* Day 1 AQI forecast
* Day 2 AQI forecast
* Day 3 AQI forecast
* AQI hazard/warning information
* SHAP-based feature importance
* Forecast explanations

The dashboard loads the latest available model version.

A caching issue was also identified during development: the dashboard was previously keeping an old model in memory after retraining. This was fixed by adding a limited cache lifetime so that newly trained models can be loaded without requiring a permanent application restart.

---

## 6. API

A small **FastAPI** backend is also included.

It provides the AQI prediction information in JSON format and exposes the same underlying forecast information used by the dashboard.

FastAPI was used instead of Flask; both satisfy the project requirements.

---

# Results

## Results with Approximately 480 Rows

When the project initially had approximately 480 rows:

| Forecast |       R² |
| -------- | -------: |
| Day 1    | Negative |
| Day 2    |    ~0.14 |
| Day 3    | Negative |

The limited amount of historical data made it difficult for the models to learn reliable relationships, particularly for longer forecast horizons.

---

## Results After One-Year Backfill (~8,900 Rows)

After increasing the dataset to approximately 8,900 rows:

| Forecast |                  R² |
| -------- | ------------------: |
| Day 1    |      **~0.23–0.26** |
| Day 2    | **~−0.08 to −0.13** |
| Day 3    | **~−0.44 to −0.53** |

The most important result is that **Day 1 forecasting improved significantly after increasing the amount of historical data**.

However, Day 2 and Day 3 remained weak, with negative R² values. Therefore, these longer-horizon predictions should not currently be considered highly reliable.

This limitation is intentionally reported rather than hidden.

---

# Attempt to Improve Day 2 and Day 3

Following mentor feedback, an additional experiment was performed using **AQI persistence features**.

Two new features were added:

* AQI approximately 24 hours earlier (`aqi_lag_24h`)
* 24-hour rolling mean AQI (`aqi_rolling_24h_mean`)

The reasoning was that recent AQI behavior might provide useful information for predicting AQI 48 or 72 hours into the future.

However, the experiment **did not improve the forecasting results**.

In fact, Day 1 R² decreased slightly from approximately **0.262 to 0.232** in the tested run, while Day 2 and Day 3 also failed to show meaningful improvement.

SHAP analysis indicated that **PM2.5 remained one of the strongest predictive features**, while the newly added persistence features did not provide enough additional information to substantially improve the forecasts.

This experiment is included as part of the final report because it demonstrates that an attempted improvement was tested and evaluated rather than simply assuming that it would work.

---

# Problems Encountered and Fixes

## 1. Incorrect Historical Weather During Backfill

### Problem

During the initial historical backfill, the current weather conditions were being used for historical pollution records.

This meant that an old pollution observation could incorrectly receive today's weather information.

### Solution

Historical weather is now retrieved from the **Open-Meteo archive API** and matched to the actual timestamp of each historical pollution observation.

The backfill process also handles large ranges in smaller chunks so that progress is saved incrementally.

---

## 2. SHAP Not Appearing on the Dashboard

### Problem

The dashboard was loading an older model version that did not contain the SHAP information required for feature explanations.

### Solution

The dashboard was changed to load the latest available model rather than relying on an older fixed model version.

SHAP information is generated during model training and stored with the model metadata.

---

## 3. Dashboard Not Updating After Retraining

### Problem

The dashboard cached models without a suitable expiration time.

As a result, a newly trained model was not necessarily loaded immediately after retraining.

### Solution

A **30-minute cache lifetime** was introduced so that the dashboard periodically refreshes the model and can display newer forecasts.

---

## 4. Hopsworks Compute Budget

### Problem

The Hopsworks free-tier compute budget was exhausted during development because the pipeline was run frequently while testing and retraining models.

### Solution

The system was designed with a local fallback.

The feature pipeline supports:

```bash
python feature_pipeline.py --backfill 365 --local-only
```

This allows historical data to be collected without continuously writing to Hopsworks.

The locally stored data can later be pushed to Hopsworks using the project's push-local functionality.

If Hopsworks fails during normal operation, the application automatically falls back to local files instead of crashing.

---

# Current Limitations

The project is functional, but several limitations remain.

### 1. Day 2 and Day 3 Accuracy

The biggest limitation is the performance of the longer forecast horizons.

Day 2 and Day 3 currently have negative R² values. This means the current models are not yet sufficiently reliable for accurate long-term AQI forecasting.

More historical data alone may not solve this problem. A substantially different forecasting strategy, additional predictive variables, or a different target formulation may be required.

### 2. AQI Forecasting Difficulty

AQI can change rapidly because of factors such as pollution emissions, wind conditions, atmospheric mixing, weather changes, and other local environmental effects.

Predicting 48–72 hours ahead is therefore considerably more difficult than predicting the next 24 hours.

### 3. Hopsworks Free Tier

The free Hopsworks environment has limited compute resources. Heavy experimentation and frequent training can consume the available monthly budget.

The local fallback architecture reduces the impact of this limitation but does not remove the underlying resource constraint.

### 4. TensorFlow Deployment

TensorFlow is used during model training when available.

The deployed Streamlit environment currently uses the models that are compatible with its Python/runtime environment. TensorFlow is not required for the dashboard to continue operating when a compatible saved model is available.

---

# Automation

GitHub Actions is used to automate the system.

### Hourly

The feature pipeline runs automatically and collects the latest pollution and weather information.

### Daily

The training pipeline runs automatically, evaluates the available models, selects the best model for each forecast horizon, calculates SHAP feature importance, and stores the resulting models.

This creates an automated end-to-end workflow:

```text
OpenWeather / Open-Meteo
          ↓
   Feature Pipeline
          ↓
    Feature Storage
          ↓
    Model Training
          ↓
 Model Selection by RMSE
          ↓
 Hopsworks / Local Models
          ↓
 Streamlit Dashboard
          ↓
     3-Day AQI Forecast
```

---

# Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Hopsworks**
* **GitHub Actions**
* **Streamlit**
* **FastAPI**
* **OpenWeather API**
* **Open-Meteo API**
* **SHAP**
* **Git / GitHub**

---

# Project Files

```text
aqi-predictor/
│
├── feature_pipeline.py
├── training_pipeline.py
├── app.py
├── api.py
├── eda_analysis.py
├── requirements.txt
│
├── data/
├── models/
├── eda_outputs/
│
└── .github/
    └── workflows/
```

### Main Files

**`feature_pipeline.py`**
Collects pollution/weather data, generates features, performs historical backfills, and stores the data.

**`training_pipeline.py`**
Creates Day 1/Day 2/Day 3 targets, trains the candidate models, evaluates them, selects the best model, and generates SHAP explanations.

**`app.py`**
Streamlit dashboard displaying the current AQI and 3-day forecast.

**`api.py`**
FastAPI backend providing prediction information as JSON.

**`eda_analysis.py`**
Generates exploratory data analysis outputs and charts.

**`.github/workflows/`**
Contains the automation workflows for hourly data collection and daily model training.

---

# How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Create the environment configuration:

```bash
cp .env.example .env
```

Run a historical backfill:

```bash
python feature_pipeline.py --backfill 15
```

Or perform a larger local-only backfill:

```bash
python feature_pipeline.py --backfill 365 --local-only
```

Collect the latest data:

```bash
python feature_pipeline.py
```

Train the models:

```bash
python training_pipeline.py
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Run the FastAPI backend:

```bash
uvicorn api:app --reload
```

---

# Conclusion

Pearls AQI Predictor demonstrates a complete end-to-end machine-learning forecasting pipeline rather than only a standalone prediction model.

The project includes automated data collection, historical backfilling, feature engineering, multiple model architectures, time-based evaluation, automatic model selection, model storage, SHAP explainability, a live dashboard, an API, and automated execution through GitHub Actions.

The increase from approximately **480 rows to around 8,900 rows** produced a clear improvement in Day 1 forecasting performance, with R² increasing to approximately **0.23–0.26**.

At the same time, the experiments showed that simply adding AQI persistence features was not enough to solve the Day 2 and Day 3 forecasting problem. These limitations are an important part of the project's findings and provide clear directions for future work.

The current system is therefore best viewed as a **working AQI forecasting prototype**, with Day 1 showing useful predictive signal while Day 2 and Day 3 require further research and improvement.
