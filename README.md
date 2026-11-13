# Taxi Fare Prediction System

A beginner-friendly Machine Learning project that predicts taxi fare from
**trip distance (km)** and **trip duration (minutes)**. Change the inputs, click
**Predict Fare**, and get a new prediction from the trained model.

## Quick start (easiest)

You only need **Python 3.9 or newer** installed (https://www.python.org/downloads/ - on Windows tick "Add Python to PATH").

- **Windows:** double-click `run.bat`
- **Mac / Linux:** open a terminal in this folder and run `bash run.sh`

The launcher creates a virtual environment, installs the libraries, trains the model
the first time, and opens the website in your browser (http://localhost:8501).
Press `Ctrl + C` in the terminal window to stop it.

## Manual start

```
pip install -r requirements.txt
python train_model.py        # optional: the website trains the model automatically on first launch
streamlit run app.py
```

## Using the website

1. **Predict Fare** tab: enter Trip Distance and Trip Duration, click **Predict Fare**.
2. Change either value and click again for a new prediction. The change from the previous fare is shown, and all predictions are listed in the history table.
3. **Fare Explorer** tab: charts showing how the predicted fare changes with distance and with duration.
4. **Data Insights** tab: the analysis charts created during training.
5. The sidebar shows the model's accuracy (R2, MAE, RMSE).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit website |
| `train_model.py` | Cleans data, analyses it, trains and evaluates the model, saves it |
| `taxi_fare.csv` | Sample dataset (synthetic, with a few messy rows to demonstrate cleaning) |
| `requirements.txt` | Python libraries needed |
| `run.bat` / `run.sh` | One-click launchers for Windows / Mac and Linux |
| `.streamlit/config.toml` | Website colours and theme |
| `models/` | Created by training; holds `taxi_fare_model.pkl` and `metrics.json` |
| `plots/` | Created by training; holds the analysis charts |

## How it works

- Data cleaning removes missing, duplicate and invalid (zero/negative) rows.
- A **Linear Regression** model learns the link between distance, duration and fare.
- The model is evaluated with MAE, RMSE and R2 on data it has not seen, then saved with Joblib.
- Every prediction comes from `model.predict(...)`. No hand-written fare formula is used.

## Using your own data

Replace `taxi_fare.csv` with a real dataset that has the columns
`trip_distance_km`, `trip_duration_min`, `fare_amount`, delete the `models` folder,
and start the app again (or run `python train_model.py`).

## Troubleshooting

- **"python is not recognized"** - install Python and tick "Add Python to PATH".
- **Port already in use** - run `streamlit run app.py --server.port 8502`.
- **Model looks wrong after changing the data** - delete the `models` folder and restart.
