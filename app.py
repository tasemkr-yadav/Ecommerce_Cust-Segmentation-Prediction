import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* Header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: #161a23;
    border: 1px solid #1f2535;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}

/* Result boxes */
.result-churn {
    background: linear-gradient(135deg, #2d1b1b, #1a0f0f);
    border: 1px solid #7f1d1d;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-no-churn {
    background: linear-gradient(135deg, #0f2d1b, #0a1f13);
    border: 1px solid #14532d;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-sub {
    font-size: 0.9rem;
    color: #9ca3af;
}

/* Risk badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-high   { background: #7f1d1d; color: #fca5a5; }
.badge-medium { background: #78350f; color: #fcd34d; }
.badge-low    { background: #14532d; color: #86efac; }

/* Metric tiles */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-tile {
    flex: 1;
    background: #1a1f2e;
    border: 1px solid #1f2535;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 2px;
}

/* Slider labels */
.input-label {
    font-size: 0.82rem;
    color: #9ca3af;
    font-weight: 500;
    margin-bottom: 2px;
    letter-spacing: 0.3px;
}

/* Divider */
hr { border-color: #1f2535; }

/* Button */
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1d4ed8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 1px solid #1f2535;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_model.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Helper functions ───────────────────────────────────────────────────────────
def assign_risk_category(churn_probability):
    if churn_probability >= 70:  return ("High Risk",   "badge-high")
    if churn_probability >= 40:  return ("Medium Risk", "badge-medium")
    return ("Low Risk", "badge-low")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Frequency &amp; Monetary-based churn prediction · Random Forest model</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️  `churn_prediction_model.pkl` not found. Make sure it's in the same folder as this app.")
    st.stop()


# ── Sidebar — about ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
This app uses a **Random Forest Classifier** trained on RFM (Recency, Frequency, Monetary) data
from an e-commerce transaction dataset.

**Features used:**
- `Frequency` — number of purchases
- `Value` — total spend (£)

**Why not Recency?** The churn label itself is defined as "no purchase in the last 60 days" —
i.e. directly from Recency. Using Recency (or any score derived from it) as a model *input* would
leak the label into the features, artificially inflating accuracy. This model only uses
`Frequency` and `Value`, which are independent of how churn is defined.

**Churn definition:**  
A customer is considered churned if their last purchase was **more than 60 days ago**.
    """)


# ── Input section ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="input-label">PURCHASE FREQUENCY</p>', unsafe_allow_html=True)
    frequency = st.number_input(
        "Number of orders", min_value=1, max_value=1000, value=25,
        help="Total number of invoices/orders placed by this customer",
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<p class="input-label">MONETARY VALUE (£ total spend)</p>', unsafe_allow_html=True)
    value = st.number_input(
        "Total spend", min_value=0.0, max_value=100000.0, value=850.0, step=50.0,
        help="Total revenue generated by this customer",
        label_visibility="collapsed"
    )

st.markdown("</div>", unsafe_allow_html=True)


# ── Predict ────────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict Churn")

if predict_btn:
    input_data = pd.DataFrame([[frequency, value]], columns=["Frequency", "Value"])
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    churn_prob    = round(probability[1] * 100, 1)
    no_churn_prob = round(probability[0] * 100, 1)

    risk_label, risk_class = assign_risk_category(churn_prob)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Prediction Result")

    # Main result box
    if prediction == 1:
        st.markdown(f"""
        <div class="result-churn">
            <div class="result-label" style="color:#f87171;">⚠️  Likely to Churn</div>
            <div class="result-sub">Churn probability: <strong style="color:#f87171">{churn_prob}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-no-churn">
            <div class="result-label" style="color:#4ade80;">✅  Not Likely to Churn</div>
            <div class="result-sub">Retention probability: <strong style="color:#4ade80">{no_churn_prob}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Metric tiles
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-value">{churn_prob}%</div>
            <div class="metric-label">Churn Probability</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{no_churn_prob}%</div>
            <div class="metric-label">Retention Probability</div>
        </div>
        <div class="metric-tile">
            <div style="margin-top:6px"><span class="badge {risk_class}">{risk_label}</span></div>
            <div class="metric-label">Churn Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation — tied to risk tier (derived directly from the model's own probability
    # output) rather than a separate RFM business-segment taxonomy, which has been removed.
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Recommended Action")

    recommendations = {
        "High Risk":   "🚨 Immediate win-back: personal outreach, exclusive discounts.",
        "Medium Risk": "🔔 Launch a re-engagement campaign before they disengage further.",
        "Low Risk":    "🏆 Low churn risk — consider loyalty perks to keep them engaged.",
    }
    st.info(recommendations.get(risk_label, "Monitor this customer closely."))


# ── Batch prediction ───────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### 📂 Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV with columns: `Frequency`, `Value`")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        required = {"Frequency", "Value"}

        if not required.issubset(batch_df.columns):
            st.error(f"CSV must contain these columns: {required}")
        else:
            preds  = model.predict(batch_df[["Frequency", "Value"]])
            probas = model.predict_proba(batch_df[["Frequency", "Value"]])[:, 1]

            batch_df["Churn_Prediction"]  = preds
            batch_df["Churn_Probability"] = (probas * 100).round(1).astype(str) + "%"
            batch_df["Churn_Label"]       = batch_df["Churn_Prediction"].map({1: "⚠️ Churn", 0: "✅ Retain"})

            st.success(f"Processed {len(batch_df)} customers.")
            st.dataframe(batch_df, use_container_width=True)

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download Results", csv_out, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
