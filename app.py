"""
Taxi Fare Predictor - minimal, light-themed Streamlit app.

What it does
------------
1. Loads the trained Machine Learning model saved with Joblib (models/ folder).
2. Lets the user enter Trip Distance (km) and Trip Duration (minutes).
3. On "Predict fare", the TRAINED MODEL predicts the fare (no manual formula).
4. Shows the fare, fare per km, average speed, change from the last prediction,
   and a history of recent predictions.

Place this file next to your `models/` folder and `.streamlit/config.toml`.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Settings - edit these if needed
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "models"          # folder where train_model.py saved the model
MODEL_EXTENSIONS = (".joblib", ".pkl", ".sav")

# Accuracy shown in the "About the model" box (taken from your training results)
MODEL_INFO = {
    "Algorithm": "Linear Regression",
    "R² score": "0.990",
    "MAE": "1.16",
    "RMSE": "1.42",
    "Trained on": "1,475 cleaned trips",
}

# ----------------------------------------------------------------------------
# Model helpers (no Streamlit code here)
# ----------------------------------------------------------------------------
def find_model_file():
    """Return the first saved model file inside the models/ folder."""
    if not MODEL_DIR.exists():
        return None
    for path in sorted(MODEL_DIR.iterdir()):
        if path.suffix.lower() in MODEL_EXTENSIONS:
            return path
    return None


def unwrap_model(obj):
    """If the saved file is a dict (e.g. {'model': ...}), pick the object that can predict."""
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            if hasattr(value, "predict"):
                return value
    raise ValueError("The saved file does not contain a model with a predict() method.")


def build_input(model, distance_km, duration_min):
    """Arrange the two inputs the way the model expects them."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:                       # model was trained on a plain array
        return np.array([[distance_km, duration_min]], dtype=float)

    row = {}
    for name in names:
        lower = str(name).lower()
        if "dist" in lower:
            row[name] = distance_km
        elif "dur" in lower or "time" in lower or "min" in lower:
            row[name] = duration_min
    if len(row) != len(names):              # unknown column names: fall back to order
        row = dict(zip(names, [distance_km, duration_min]))
    return pd.DataFrame([row], columns=list(names))


def predict_fare(model, distance_km, duration_min):
    """Ask the trained model for the fare. This is the ONLY place a fare is calculated."""
    value = model.predict(build_input(model, distance_km, duration_min))
    return max(0.0, float(np.ravel(value)[0]))


@st.cache_resource(show_spinner=False)
def load_model():
    path = find_model_file()
    if path is None:
        raise FileNotFoundError(
            "No saved model found in the 'models' folder. Run `python train_model.py` first."
        )
    return unwrap_model(joblib.load(path))


# --- end of helpers (used by tests) ---

# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #FAF8F4; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 720px; padding-top: 3rem; padding-bottom: 4rem; }

/* Header */
.brand { display:flex; align-items:center; gap:.6rem; margin-bottom:.35rem; }
.brand .dot { width:10px; height:10px; border-radius:50%; background:#E8B923; }
.brand span { font-size:.8rem; letter-spacing:.18em; text-transform:uppercase; color:#8A8F98; font-weight:500; }
h1.title { font-size:2.3rem; font-weight:600; letter-spacing:-.02em; margin:0 0 .4rem 0; color:#1F2933; }
p.subtitle { color:#7B8089; font-size:1rem; margin:0 0 2rem 0; font-weight:300; }

/* Cards */
.card { background:#FFFFFF; border:1px solid #EEE9E0; border-radius:18px; padding:1.4rem 1.5rem; }
.label { font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#9AA0A8; font-weight:500; }

/* Result */
.result { text-align:center; padding:2rem 1.5rem 1.6rem 1.5rem; margin-top:1rem; }
.result .amount { font-size:4rem; font-weight:600; letter-spacing:-.03em; color:#1F2933; line-height:1.1; margin:.4rem 0 .3rem 0; }
.result .trip { color:#7B8089; font-size:.95rem; font-weight:300; }
.result.empty .amount { color:#D3CEC4; }
.chips { display:flex; gap:.75rem; margin-top:.9rem; }
.chip { flex:1; background:#FFFFFF; border:1px solid #EEE9E0; border-radius:14px; padding:.9rem 1rem; }
.chip .value { font-size:1.35rem; font-weight:600; color:#1F2933; margin-top:.25rem; }
.chip .value small { font-size:.75rem; font-weight:400; color:#9AA0A8; margin-left:.2rem; }
.up { color:#C2553F !important; } .down { color:#2F7D5B !important; }

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"] { border-radius:12px !important; }
div[data-baseweb="input"] { background:#FFFFFF !important; border:1px solid #E7E2D8 !important; }
.stNumberInput label p { font-size:.78rem !important; color:#7B8089 !important; font-weight:500 !important; }
div[data-testid="stForm"] { background:#FFFFFF; border:1px solid #EEE9E0; border-radius:18px; padding:1.4rem 1.5rem 1.2rem 1.5rem; }

/* Buttons */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width:100%; border-radius:12px; height:3rem; font-weight:500; letter-spacing:.02em;
    border:1px solid #1F2933; background:#1F2933; color:#FFFFFF; transition:all .15s ease;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background:#3A4652; border-color:#3A4652; color:#FFFFFF;
}
.stButton > button[kind="secondary"] { background:transparent; color:#7B8089; border:1px solid #E7E2D8; height:2.4rem; }
.stButton > button[kind="secondary"]:hover { background:#F1EEE8; color:#1F2933; border-color:#DDD7CA; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:1.6rem; border-bottom:1px solid #EEE9E0; }
.stTabs [data-baseweb="tab"] { background:transparent; padding:.6rem 0; color:#9AA0A8; font-weight:500; }
.stTabs [aria-selected="true"] { color:#1F2933 !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#E8B923 !important; height:2px; }

/* Misc */
hr { border-color:#EEE9E0 !important; }
div[data-testid="stDataFrame"] { border:1px solid #EEE9E0; border-radius:14px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid #EEE9E0; border-radius:14px; background:#FFFFFF; }
.foot { text-align:center; color:#B3B0A8; font-size:.78rem; margin-top:3rem; font-weight:300; }
@media (max-width: 640px) {
    h1.title { font-size:1.8rem; } .result .amount { font-size:3.2rem; }
    .chips { flex-direction:column; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load the model
# ----------------------------------------------------------------------------
try:
    model = load_model()
except Exception as error:  # show a friendly message instead of a crash
    st.error(f"Could not load the model: {error}")
    st.stop()

# Session memory: last result and history survive each button click
if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None


def clear_history():
    st.session_state.history = []
    st.session_state.last = None


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand"><div class="dot"></div><span>Machine Learning</span></div>
    <h1 class="title">Taxi Fare Predictor</h1>
    <p class="subtitle">Enter your trip details. A trained model estimates the fare instantly.</p>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_about = st.tabs(["Predict", "Insights", "About"])

# ----------------------------------------------------------------------------
# Tab 1 - Predict
# ----------------------------------------------------------------------------
with tab_predict:
    with st.form("trip_form"):
        col1, col2 = st.columns(2)
        distance = col1.number_input("Trip distance (km)", min_value=0.1, max_value=500.0,
                                     value=7.0, step=0.5, format="%.1f")
        duration = col2.number_input("Trip duration (minutes)", min_value=1.0, max_value=600.0,
                                     value=17.0, step=1.0, format="%.0f")
        submitted = st.form_submit_button("Predict fare")

    if submitted:
        fare = predict_fare(model, distance, duration)       # prediction from the ML model
        previous = st.session_state.last["fare"] if st.session_state.last else None
        st.session_state.last = {
            "distance": distance, "duration": duration, "fare": fare, "previous": previous,
        }
        st.session_state.history.insert(0, {
            "Distance (km)": round(distance, 1),
            "Duration (min)": round(duration),
            "Predicted fare": round(fare, 2),
        })
        st.session_state.history = st.session_state.history[:10]

    last = st.session_state.last
    if last is None:
        st.markdown(
            """
            <div class="card result empty">
              <div class="label">Estimated fare</div>
              <div class="amount">--.--</div>
              <div class="trip">Press “Predict fare” to see your estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        fare_per_km = last["fare"] / last["distance"]
        speed = last["distance"] / (last["duration"] / 60)
        if last["previous"] is None:
            change_html = '<div class="value">–</div>'
        else:
            diff = last["fare"] - last["previous"]
            arrow, css = ("↑", "up") if diff > 0 else (("↓", "down") if diff < 0 else ("→", ""))
            change_html = f'<div class="value {css}">{arrow} {abs(diff):.2f}</div>'

        st.markdown(
            f"""
            <div class="card result">
              <div class="label">Estimated fare</div>
              <div class="amount">{last['fare']:.2f}</div>
              <div class="trip">for {last['distance']:g} km in {last['duration']:g} minutes</div>
            </div>
            <div class="chips">
              <div class="chip"><div class="label">Fare per km</div>
                <div class="value">{fare_per_km:.2f}</div></div>
              <div class="chip"><div class="label">Average speed</div>
                <div class="value">{speed:.1f}<small>km/h</small></div></div>
              <div class="chip"><div class="label">vs previous</div>{change_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="label" style="margin-bottom:.6rem">Recent predictions</div>',
                    unsafe_allow_html=True)
        table = pd.DataFrame(st.session_state.history)
        try:                                   # newer Streamlit versions
            st.dataframe(table, hide_index=True, width="stretch")
        except TypeError:                      # older Streamlit versions
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.button("Clear history", on_click=clear_history, type="secondary")

# ----------------------------------------------------------------------------
# Tab 2 - Insights (curves drawn from the trained model's own predictions)
# ----------------------------------------------------------------------------
with tab_insights:
    base_d = st.session_state.last["distance"] if st.session_state.last else 7.0
    base_t = st.session_state.last["duration"] if st.session_state.last else 17.0

    st.markdown('<div class="label" style="margin:1rem 0 .4rem 0">Fare vs distance</div>',
                unsafe_allow_html=True)
    st.caption(f"Duration fixed at {base_t:g} minutes")
    dists = np.linspace(1, 20, 40)
    curve_d = pd.DataFrame(
        {"Predicted fare": [predict_fare(model, d, base_t) for d in dists]},
        index=pd.Index(np.round(dists, 1), name="Distance (km)"),
    )
    st.line_chart(curve_d)

    st.markdown('<div class="label" style="margin:1.4rem 0 .4rem 0">Fare vs duration</div>',
                unsafe_allow_html=True)
    st.caption(f"Distance fixed at {base_d:g} km")
    times = np.linspace(5, 60, 40)
    curve_t = pd.DataFrame(
        {"Predicted fare": [predict_fare(model, base_d, t) for t in times]},
        index=pd.Index(np.round(times, 0), name="Duration (min)"),
    )
    st.line_chart(curve_t)
    st.caption("Both lines are straight because the model is Linear Regression.")

# ----------------------------------------------------------------------------
# Tab 3 - About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:.7rem 0;"
        f"border-bottom:1px solid #F1EEE8'><span style='color:#7B8089'>{k}</span>"
        f"<span style='font-weight:500'>{v}</span></div>"
        for k, v in MODEL_INFO.items()
    )
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7B8089;font-weight:300;margin-top:1rem'>Inputs: trip distance and "
        "trip duration. Output: predicted fare amount. Every prediction comes from the "
        "trained model, not from a hand-written formula.</p>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">Taxi Fare Prediction System</div>', unsafe_allow_html=True)
