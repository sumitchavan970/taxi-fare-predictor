"""
app.py
------
Streamlit website for the Taxi Fare Prediction System.

The client enters Trip Distance and Trip Duration, clicks "Predict Fare",
and the trained Machine Learning model (models/taxi_fare_model.pkl)
returns the predicted fare. No manual fare formula is used here.

The model is trained automatically on the first launch if it does not
exist yet. Run:  streamlit run app.py
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "models" / "taxi_fare_model.pkl"
METRICS_FILE = BASE_DIR / "models" / "metrics.json"
PLOTS_DIR = BASE_DIR / "plots"

# Column names must match the ones used during training
FEATURES = ["trip_distance_km", "trip_duration_min"]

st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕", layout="wide")

# ---------------------------------------------------------------
# Styling (taxi yellow + black)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .banner {
        background: linear-gradient(120deg, #111827 0%, #1f2937 60%, #374151 100%);
        border-radius: 16px; padding: 28px 32px; margin-bottom: 18px;
        border-left: 10px solid #F5B800;
    }
    .banner h1 { color: #F5B800; margin: 0; font-size: 2.2rem; }
    .banner p  { color: #E5E7EB; margin: 6px 0 0 0; font-size: 1.05rem; }
    .fare-card {
        background: #FFF8E1; border: 2px solid #F5B800; border-radius: 16px;
        padding: 22px; text-align: center; margin-top: 8px;
    }
    .fare-card .label { color: #6B7280; font-size: 0.95rem; letter-spacing: 1px; }
    .fare-card .amount { color: #111827; font-size: 3.2rem; font-weight: 800; }
    .fare-card .detail { color: #374151; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banner">
      <h1>🚕 Taxi Fare Predictor</h1>
      <p>Enter your trip distance and duration. A trained Machine Learning model predicts the fare instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Model loading (trains automatically on the very first launch)
# ---------------------------------------------------------------
def ensure_model_exists():
    """If no trained model is found, run train_model.py once."""
    if MODEL_FILE.exists():
        return
    with st.spinner("First launch: training the model (takes a few seconds)..."):
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "train_model.py")],
            cwd=str(BASE_DIR), capture_output=True, text=True,
        )
    if result.returncode != 0 or not MODEL_FILE.exists():
        st.error("Training failed. Details below:")
        st.code(result.stderr or result.stdout)
        st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def load_metrics():
    """Read the scores saved by train_model.py (empty dict if missing)."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {}


ensure_model_exists()
model = load_model()
metrics = load_metrics()

# Remember predictions while the page is open
if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None


def predict_fare(distance_km, duration_min):
    """Ask the trained ML model for a fare."""
    trip = pd.DataFrame([[distance_km, duration_min]], columns=FEATURES)
    return float(model.predict(trip)[0])


# ---------------------------------------------------------------
# Sidebar: model information
# ---------------------------------------------------------------
with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** Linear Regression")
    st.write("**Inputs:** trip distance (km) and trip duration (minutes)")
    st.write("**Output:** predicted fare amount")
    if metrics:
        st.subheader("Accuracy on test data")
        st.metric("R² score", f"{metrics['r2']:.3f}")
        st.metric("Mean absolute error (MAE)", f"{metrics['mae']:.2f}")
        st.metric("Root mean squared error (RMSE)", f"{metrics['rmse']:.2f}")
        st.caption(f"Trained on {metrics['rows_used']:,} cleaned trips.")

# ---------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------
tab_predict, tab_explore, tab_data = st.tabs(
    ["🧮 Predict Fare", "📈 Fare Explorer", "📊 Data Insights"]
)

# ----- Tab 1: prediction -----
with tab_predict:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Trip details")
        distance = st.number_input(
            "Trip Distance (km)", min_value=0.5, max_value=200.0, value=5.0, step=0.5
        )
        duration = st.number_input(
            "Trip Duration (minutes)", min_value=1.0, max_value=480.0, value=15.0, step=1.0
        )
        # Streamlit re-runs the script on every click, so the values
        # above are always the latest ones the user typed.
        if st.button("Predict Fare", type="primary"):
            fare = predict_fare(distance, duration)
            previous = st.session_state.last["fare"] if st.session_state.last else None
            st.session_state.last = {
                "distance": distance, "duration": duration,
                "fare": fare, "previous": previous,
            }
            st.session_state.history.append({
                "Distance (km)": distance,
                "Duration (min)": duration,
                "Predicted Fare": round(fare, 2),
            })
        st.caption("Change the distance or duration and click Predict Fare again for a new estimate.")

    with right:
        st.subheader("Predicted fare")
        last = st.session_state.last
        if last is None:
            st.info("Enter trip details and click **Predict Fare**.")
        else:
            st.markdown(
                f"""
                <div class="fare-card">
                  <div class="label">ESTIMATED FARE</div>
                  <div class="amount">{last['fare']:,.2f}</div>
                  <div class="detail">for {last['distance']:g} km in {last['duration']:g} minutes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            delta = None
            if last["previous"] is not None:
                delta = f"{last['fare'] - last['previous']:+.2f} vs previous prediction"
            c1, c2, c3 = st.columns(3)
            c1.metric("Fare", f"{last['fare']:,.2f}", delta=delta, delta_color="off")
            c2.metric("Fare per km", f"{last['fare'] / last['distance']:.2f}")
            c3.metric("Average speed", f"{last['distance'] / (last['duration'] / 60):.1f} km/h")

            # Warn if the inputs are far outside the data the model learned from
            if metrics:
                d_lo, d_hi = metrics["distance_range"]
                t_lo, t_hi = metrics["duration_range"]
                if not (d_lo <= last["distance"] <= d_hi and t_lo <= last["duration"] <= t_hi):
                    st.warning(
                        f"These values are outside the training data "
                        f"({d_lo:g}-{d_hi:g} km, {t_lo:g}-{t_hi:g} min), "
                        "so the estimate may be less reliable."
                    )

    # Prediction history
    if st.session_state.history:
        st.divider()
        st.subheader("Prediction history")
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]))
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last = None
            st.rerun()

# ----- Tab 2: explore how fare changes -----
with tab_explore:
    st.subheader("How does the predicted fare change?")
    st.write("Both charts are drawn by asking the trained model for many predictions.")

    ex_left, ex_right = st.columns(2)
    with ex_left:
        fixed_duration = st.slider("Keep duration fixed at (minutes)", 5, 90, 20)
        dist_range = pd.DataFrame({"trip_distance_km": [float(d) for d in range(1, 41)]})
        dist_range["trip_duration_min"] = float(fixed_duration)
        dist_range["Predicted Fare"] = model.predict(dist_range[FEATURES])
        st.caption("Predicted fare as distance increases")
        st.line_chart(dist_range.set_index("trip_distance_km")[["Predicted Fare"]])

    with ex_right:
        fixed_distance = st.slider("Keep distance fixed at (km)", 1, 40, 10)
        dur_range = pd.DataFrame({"trip_duration_min": [float(t) for t in range(5, 91, 5)]})
        dur_range["trip_distance_km"] = float(fixed_distance)
        dur_range["Predicted Fare"] = model.predict(dur_range[FEATURES])
        st.caption("Predicted fare as duration increases")
        st.line_chart(dur_range.set_index("trip_duration_min")[["Predicted Fare"]])

# ----- Tab 3: charts saved by train_model.py -----
with tab_data:
    st.subheader("Data analysis charts")
    charts = [
        ("distance_vs_fare.png", "Distance vs Fare"),
        ("duration_vs_fare.png", "Duration vs Fare"),
        ("fare_distribution.png", "Fare Distribution"),
        ("actual_vs_predicted.png", "Actual vs Predicted (test data)"),
    ]
    found = [(PLOTS_DIR / name, title) for name, title in charts if (PLOTS_DIR / name).exists()]
    if not found:
        st.info("Charts are created when you run train_model.py.")
    else:
        cols = st.columns(2)
        for i, (path, title) in enumerate(found):
            with cols[i % 2]:
                st.image(str(path), caption=title)

st.caption("Taxi Fare Prediction System - predictions come from a trained Machine Learning model.")
