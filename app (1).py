"""
Taxi Fare Prediction System - Streamlit web app
------------------------------------------------
Users enter Trip Distance (km) and Trip Duration (minutes), click
"Predict Fare", and the trained Machine Learning model returns the fare.

NOTE: There is NO hand-written fare formula in this file.
Every fare shown comes from model.predict().
"""

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Settings (easy to change)
# ---------------------------------------------------------------------------
CURRENCY = "₹"  # change to "₹" if your dataset is in rupees
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "taxi_fare.csv"

# Theme colours (used in charts; the page CSS below uses the same palette)
YELLOW = "#FFC20E"
MUTED = "#9AA4B8"
TEXT = "#F5F7FA"

st.set_page_config(
    page_title="Taxi Fare Predictor",
    page_icon="🚖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom theme: fonts, colours, cards, animations
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --y: #FFC20E;
    --amber: #FF9F1C;
    --card: rgba(255, 255, 255, 0.045);
    --line: rgba(255, 255, 255, 0.09);
    --text: #F5F7FA;
    --muted: #9AA4B8;
}

/* ---------- page background ---------- */
.stApp {
    background:
        radial-gradient(900px 500px at 10% -5%, rgba(255, 194, 14, 0.13), transparent 60%),
        radial-gradient(800px 500px at 100% 10%, rgba(99, 102, 241, 0.16), transparent 60%),
        linear-gradient(180deg, #0A0E1A 0%, #0D1324 100%);
    background-attachment: fixed;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 2rem; padding-bottom: 3rem; }

/* ---------- fonts ---------- */
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
.hero-title, .sec-title, .fare-amount, .stat-value, .step b, .kpi-value {
    font-family: 'Outfit', sans-serif;
}

/* ---------- hero banner ---------- */
.hero {
    position: relative; overflow: hidden; margin-bottom: 1.6rem;
    border-radius: 24px; border: 1px solid var(--line);
    background: linear-gradient(135deg, #111936 0%, #0F1530 55%, #1B1F3F 100%);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
}
.hero::before {
    content: ""; position: absolute; right: -80px; top: -80px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 194, 14, 0.35), transparent 65%);
}
.hero-strip {
    height: 12px;
    background: repeating-conic-gradient(#FFC20E 0% 25%, #0A0E1A 0% 50%) 0 0 / 24px 24px;
}
.hero-body { position: relative; padding: 2rem 2rem 1.3rem; }
.hero-badge {
    display: inline-block; margin-bottom: 0.9rem; padding: 0.3rem 0.85rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em; color: var(--y);
    background: rgba(255, 194, 14, 0.1); border: 1px solid rgba(255, 194, 14, 0.35);
    border-radius: 999px;
}
.hero-title { font-size: 2.9rem; font-weight: 800; line-height: 1.1; margin-bottom: 0.6rem; color: var(--text); }
.hero-title .accent {
    background: linear-gradient(90deg, #FFE27A, #FFC20E, #FF9F1C);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--muted); font-size: 1.02rem; max-width: 620px; margin-bottom: 1rem; }
.hero-chips span {
    display: inline-block; margin: 0 0.4rem 0.4rem 0; padding: 0.3rem 0.8rem;
    font-size: 0.82rem; color: var(--text);
    background: rgba(255, 255, 255, 0.06); border: 1px solid var(--line); border-radius: 999px;
}
.road { position: relative; height: 46px; background: rgba(0, 0, 0, 0.28); border-top: 1px solid var(--line); overflow: hidden; }
.road::before {
    content: ""; position: absolute; left: 0; right: 0; top: 60%; height: 3px;
    background: repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.35) 0 26px, transparent 26px 52px);
    animation: dash 1.2s linear infinite;
}
.cab { position: absolute; top: 4px; font-size: 1.8rem; transform: scaleX(-1); animation: drive 9s linear infinite; }
@keyframes drive { from { left: -60px; } to { left: 100%; } }
@keyframes dash { from { background-position: 0 0; } to { background-position: -52px 0; } }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }

/* ---------- tabs as pills ---------- */
[data-baseweb="tab-list"] {
    gap: 6px; padding: 6px; width: fit-content; max-width: 100%; overflow-x: auto;
    background: var(--card); border: 1px solid var(--line); border-radius: 16px;
}
button[data-baseweb="tab"] {
    height: auto; padding: 0.55rem 1.2rem; border-radius: 12px; background: transparent;
    transition: all 0.2s ease;
}
button[data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
button[data-baseweb="tab"]:hover p { color: var(--text); }
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #FFC20E, #FF9F1C);
    box-shadow: 0 6px 18px rgba(255, 170, 0, 0.35);
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #14120A !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

/* ---------- section titles ---------- */
.sec-title { font-size: 1.45rem; font-weight: 700; color: var(--text); margin: 1.2rem 0 0.15rem; }
.sec-sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.9rem; }

/* ---------- input card ---------- */
[data-testid="stForm"] {
    background: var(--card); border: 1px solid var(--line); border-radius: 20px;
    padding: 1.4rem 1.4rem 1.2rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}
[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600; font-size: 0.95rem; }
div[data-baseweb="input"] {
    background: rgba(255, 255, 255, 0.06) !important; border: 1px solid var(--line) !important;
    border-radius: 12px !important; overflow: hidden;
}
div[data-baseweb="base-input"] { background: transparent !important; }
div[data-baseweb="input"]:focus-within {
    border-color: var(--y) !important; box-shadow: 0 0 0 3px rgba(255, 194, 14, 0.2) !important;
}
[data-testid="stNumberInput"] input {
    color: var(--text) !important; background: transparent !important;
    font-weight: 600; font-size: 1.05rem;
}
[data-testid="stNumberInput"] button {
    background: rgba(255, 255, 255, 0.06) !important; color: var(--y) !important; border: none !important;
}
[data-testid="stNumberInput"] button:hover { background: rgba(255, 194, 14, 0.2) !important; }

/* ---------- predict button ---------- */
[data-testid="stFormSubmitButton"] button,
[data-testid="stForm"] button[kind="secondaryFormSubmit"],
[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    width: 100%; padding: 0.8rem 1rem; border: none; border-radius: 14px;
    background: linear-gradient(135deg, #FFC20E, #FF9F1C);
    box-shadow: 0 8px 24px rgba(255, 170, 0, 0.35); transition: all 0.2s ease;
}
[data-testid="stFormSubmitButton"] button p,
[data-testid="stForm"] button[kind="secondaryFormSubmit"] p,
[data-testid="stForm"] button[kind="primaryFormSubmit"] p {
    color: #14120A !important; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.08rem;
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover,
[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-2px); filter: brightness(1.06);
    box-shadow: 0 12px 30px rgba(255, 170, 0, 0.5);
}

/* ---------- sliders ---------- */
div[role="slider"] { background-color: var(--y) !important; box-shadow: 0 0 0 4px rgba(255, 194, 14, 0.25); }

/* ---------- result card ---------- */
.result {
    position: relative; overflow: hidden; border-radius: 22px; padding: 1.6rem; animation: rise 0.5s ease;
    background: linear-gradient(145deg, rgba(255, 194, 14, 0.16), rgba(255, 159, 28, 0.05) 45%, rgba(99, 102, 241, 0.10));
    border: 1px solid rgba(255, 194, 14, 0.35);
    box-shadow: 0 20px 60px rgba(255, 170, 0, 0.12);
}
.result-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.18em; color: var(--y); }
.fare-amount {
    font-size: 4rem; font-weight: 800; line-height: 1.05; margin: 0.3rem 0 0.4rem;
    background: linear-gradient(90deg, #FFE27A, #FFC20E, #FF9F1C);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.trip-line { color: var(--muted); font-size: 0.95rem; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; margin-top: 1.3rem; }
.stat {
    padding: 0.8rem 0.5rem; text-align: center; border-radius: 14px;
    background: rgba(255, 255, 255, 0.05); border: 1px solid var(--line);
}
.stat-value { font-size: 1.15rem; font-weight: 700; color: var(--text); }
.stat-label { font-size: 0.72rem; color: var(--muted); margin-top: 0.15rem; }

.placeholder {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 290px; padding: 1.5rem; text-align: center; border-radius: 22px;
    border: 2px dashed rgba(255, 194, 14, 0.3); background: rgba(255, 194, 14, 0.04);
}
.placeholder .big { font-size: 3rem; margin-bottom: 0.4rem; }
.placeholder .t { font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 600; color: var(--text); }
.placeholder .s { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }

/* ---------- how it works ---------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.9rem; margin-top: 1.8rem; }
.step {
    display: flex; gap: 0.8rem; align-items: center; padding: 0.9rem 1rem;
    background: var(--card); border: 1px solid var(--line); border-radius: 16px;
}
.step-n {
    flex: 0 0 auto; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;
    font-family: 'Outfit', sans-serif; font-weight: 800; color: #14120A; border-radius: 50%;
    background: linear-gradient(135deg, #FFC20E, #FF9F1C);
}
.step b { display: block; font-size: 0.98rem; color: var(--text); }
.step span { font-size: 0.8rem; color: var(--muted); }

/* ---------- data insight cards ---------- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.9rem; margin: 0.5rem 0 1rem; }
.kpi { padding: 1rem; background: var(--card); border: 1px solid var(--line); border-radius: 16px; }
.kpi-icon { font-size: 1.3rem; }
.kpi-value { font-size: 1.55rem; font-weight: 700; color: var(--y); margin-top: 0.2rem; }
.kpi-label { font-size: 0.78rem; color: var(--muted); }

[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line) !important; border-radius: 14px; }
.foot { text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 2.2rem; }
.foot b { color: var(--y); }

/* ---------- small screens ---------- */
@media (max-width: 640px) {
    .hero-title { font-size: 2rem; }
    .hero-body { padding: 1.4rem 1.2rem 1rem; }
    .fare-amount { font-size: 3rem; }
    .stats { gap: 0.5rem; }
}
"""


def html(block: str) -> None:
    """Render an HTML/CSS snippet. Lines are trimmed so Markdown never treats them as code."""
    cleaned = " ".join(line.strip() for line in block.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


html("<style>" + CSS + "</style>")


# ---------------------------------------------------------------------------
# Loading the trained model and the dataset
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the model saved by train_model.py (any .pkl / .joblib file in models/)."""
    files = sorted(list(MODEL_DIR.glob("*.pkl")) + list(MODEL_DIR.glob("*.joblib")))
    if not files:
        return None
    # Prefer a file that looks like the model if several files exist
    preferred = [f for f in files if "model" in f.name.lower() or "fare" in f.name.lower()]
    obj = joblib.load((preferred or files)[0])
    if isinstance(obj, dict):  # in case the model was saved inside a dictionary
        obj = obj.get("model", obj)
    return obj


@st.cache_data
def load_data():
    """Load and lightly clean the CSV (only used for the charts in Data Insights)."""
    if not DATA_FILE.exists():
        return None
    df = pd.read_csv(DATA_FILE)

    def find(exact, contains):
        for col in df.columns:
            if col.lower() in exact:
                return col
        for col in df.columns:
            if any(key in col.lower() for key in contains):
                return col
        return None

    dist = find(["trip_distance", "distance"], ["distance"])
    dur = find(["trip_duration", "duration"], ["duration", "time"])
    fare = find(["fare_amount", "fare"], ["fare"])
    if not (dist and dur and fare):
        return None

    out = df[[dist, dur, fare]].copy()
    out.columns = ["distance", "duration", "fare"]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna().drop_duplicates()
    return out[(out > 0).all(axis=1)].reset_index(drop=True)


def predict_many(model, distances, durations):
    """Ask the trained ML model for fares. Works for one trip or many trips."""
    distances = np.asarray(distances, dtype=float)
    durations = np.asarray(durations, dtype=float)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        # Build a table whose column names match the ones used during training
        data = {n: (distances if "dist" in str(n).lower() else durations) for n in names}
        features = pd.DataFrame(data, columns=list(names))
    else:
        features = np.column_stack([distances, durations])
    return np.maximum(np.asarray(model.predict(features), dtype=float), 0.0)  # a fare is never below 0


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def section(title: str, sub: str = "") -> None:
    html(f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>')


def kpi(icon: str, value: str, label: str) -> str:
    return (
        f'<div class="kpi"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'
    )


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Give every chart the same dark / yellow look."""
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=TEXT, labelFontSize=12, titleFontSize=12,
            gridColor="rgba(255,255,255,0.07)", domainColor="rgba(255,255,255,0.18)",
            tickColor="rgba(255,255,255,0.18)",
        )
    )


def show_chart(chart: alt.Chart) -> None:
    st.altair_chart(style_chart(chart), theme=None)


def fare_curve(df: pd.DataFrame, x_title: str) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X("x:Q", title=x_title),
        y=alt.Y("fare:Q", title=f"Predicted fare ({CURRENCY})"),
    )
    area = base.mark_area(opacity=0.18, color=YELLOW)
    line = base.mark_line(color=YELLOW, strokeWidth=3)
    return (area + line).properties(height=300, width="container")


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
html("""
<div class="hero">
  <div class="hero-strip"></div>
  <div class="hero-body">
    <div class="hero-badge">ML-POWERED &middot; INSTANT ESTIMATES</div>
    <div class="hero-title">🚖 Taxi Fare <span class="accent">Predictor</span></div>
    <div class="hero-sub">Enter your trip distance and duration. A trained Machine Learning model predicts the fare instantly.</div>
    <div class="hero-chips"><span>📏 Distance</span><span>⏱️ Duration</span><span>🤖 Trained regression model</span></div>
  </div>
  <div class="road"><div class="cab">🚕</div></div>
</div>
""")

model = load_model()
if model is None:
    st.error("No trained model found in the 'models/' folder. Run `python train_model.py` first.")
    st.stop()

tab_predict, tab_explore, tab_data = st.tabs(["🚖 Predict Fare", "📈 Fare Explorer", "📊 Data Insights"])

# ---------------------------------------------------------------------------
# Tab 1: Predict Fare
# ---------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns(2, gap="large")

    with left:
        section("Trip details", "Tell us about your ride")
        with st.form("trip_form"):
            distance = st.number_input(
                "📏 Trip Distance (km)", min_value=0.1, max_value=500.0, value=5.0, step=0.5, format="%.2f"
            )
            duration = st.number_input(
                "⏱️ Trip Duration (minutes)", min_value=1.0, max_value=600.0, value=15.0, step=1.0, format="%.2f"
            )
            submitted = st.form_submit_button("🚖  Predict Fare")

        # Every click uses the CURRENT values typed above and the trained model
        if submitted:
            fare = float(predict_many(model, [distance], [duration])[0])
            st.session_state["result"] = {"fare": fare, "distance": distance, "duration": duration}

        html('<div class="sec-sub" style="margin-top:0.8rem;">Change the distance or duration and click Predict Fare again for a new estimate.</div>')

    with right:
        section("Predicted fare", "Straight from the Machine Learning model")
        result = st.session_state.get("result")
        if result is None:
            html("""
            <div class="placeholder">
              <div class="big">🚕</div>
              <div class="t">Your fare will appear here</div>
              <div class="s">Enter trip details and click <b>Predict Fare</b>.</div>
            </div>
            """)
        else:
            f, d, t = result["fare"], result["distance"], result["duration"]
            html(f"""
            <div class="result">
              <div class="result-label">ESTIMATED FARE</div>
              <div class="fare-amount">{CURRENCY}{f:,.2f}</div>
              <div class="trip-line">for a {d:,.1f} km trip lasting {t:,.0f} minutes</div>
              <div class="stats">
                <div class="stat"><div class="stat-value">{CURRENCY}{f / d:,.2f}</div><div class="stat-label">per km</div></div>
                <div class="stat"><div class="stat-value">{CURRENCY}{f / t:,.2f}</div><div class="stat-label">per minute</div></div>
                <div class="stat"><div class="stat-value">{d / (t / 60):,.0f} km/h</div><div class="stat-label">avg speed</div></div>
              </div>
            </div>
            """)

    html("""
    <div class="steps">
      <div class="step"><div class="step-n">1</div><div><b>Enter your trip</b><span>Distance and duration</span></div></div>
      <div class="step"><div class="step-n">2</div><div><b>Model predicts</b><span>Trained regression model</span></div></div>
      <div class="step"><div class="step-n">3</div><div><b>See your fare</b><span>Change inputs and repeat</span></div></div>
    </div>
    """)

# ---------------------------------------------------------------------------
# Tab 2: Fare Explorer (how the model's fare changes with each input)
# ---------------------------------------------------------------------------
with tab_explore:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section("Fare vs distance", "Duration is kept fixed")
        fixed_duration = st.slider("Trip duration (minutes)", 5, 120, 20)
        xs = np.linspace(0.5, 30, 60)
        ys = predict_many(model, xs, np.full_like(xs, fixed_duration))
        show_chart(fare_curve(pd.DataFrame({"x": xs, "fare": ys}), "Trip distance (km)"))

    with col_b:
        section("Fare vs duration", "Distance is kept fixed")
        fixed_distance = st.slider("Trip distance (km)", 1, 50, 8)
        xs = np.linspace(2, 90, 60)
        ys = predict_many(model, np.full_like(xs, fixed_distance), xs)
        show_chart(fare_curve(pd.DataFrame({"x": xs, "fare": ys}), "Trip duration (minutes)"))

    html('<div class="sec-sub">Every point on these curves is predicted by the trained ML model.</div>')

# ---------------------------------------------------------------------------
# Tab 3: Data Insights (basic analysis of the training dataset)
# ---------------------------------------------------------------------------
with tab_data:
    data = load_data()
    if data is None:
        st.info("The dataset (taxi_fare.csv) could not be loaded, so charts are not available.")
    else:
        section("Dataset at a glance", "After removing missing, duplicate and invalid rows")
        html(
            '<div class="kpis">'
            + kpi("🧾", f"{len(data):,}", "clean trips")
            + kpi("💵", f"{CURRENCY}{data['fare'].mean():,.2f}", "average fare")
            + kpi("📏", f"{data['distance'].mean():,.1f} km", "average distance")
            + kpi("⏱️", f"{data['duration'].mean():,.1f} min", "average duration")
            + "</div>"
        )

        section("Fare distribution", "How fares are spread across all trips")
        hist = (
            alt.Chart(data)
            .mark_bar(color=YELLOW, opacity=0.9, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("fare:Q", bin=alt.Bin(maxbins=25), title=f"Fare ({CURRENCY})"),
                y=alt.Y("count()", title="Number of trips"),
            )
            .properties(height=280, width="container")
        )
        show_chart(hist)

        sample = data.sample(min(len(data), 1500), random_state=42)
        col_c, col_d = st.columns(2, gap="large")
        with col_c:
            section("Distance vs fare", "Longer trips cost more")
            sc1 = (
                alt.Chart(sample)
                .mark_circle(size=45, opacity=0.55, color=YELLOW)
                .encode(x=alt.X("distance:Q", title="Distance (km)"), y=alt.Y("fare:Q", title=f"Fare ({CURRENCY})"))
                .properties(height=280, width="container")
            )
            show_chart(sc1)
        with col_d:
            section("Duration vs fare", "Longer rides cost more")
            sc2 = (
                alt.Chart(sample)
                .mark_circle(size=45, opacity=0.55, color="#818CF8")
                .encode(x=alt.X("duration:Q", title="Duration (minutes)"), y=alt.Y("fare:Q", title=f"Fare ({CURRENCY})"))
                .properties(height=280, width="container")
            )
            show_chart(sc2)

        with st.expander("🔎 Preview the cleaned data"):
            st.dataframe(data.head(50), hide_index=True)

html('<div class="foot">🚖 <b>Taxi Fare Prediction System</b> &middot; predictions come from a trained Machine Learning model.</div>')
