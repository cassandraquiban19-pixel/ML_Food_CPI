import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from prediction_utils import prepare_prediction_input

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CPI Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --forest: #24563A;
        --sage: #79966D;
        --cream: #F6F8F1;
        --white: #FFFFFF;
        --gold: #D9B86C;
        --ink: #174A2B;
        --muted: #66806D;
        --border: #DDE6D8;
    }

    .stApp { background: var(--cream); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(246, 248, 241, 0.96); }
    [data-testid="stSidebar"] {
        background: #EDF2E8;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label { color: var(--muted); }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { font-weight: 800; color: var(--forest); }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,.58); border: 1px solid transparent;
        border-radius: 12px; padding: 8px 10px; transition: all .15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { border-color: #c9dac1; background: #fff; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size: 1.15rem; }

    h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.025em; }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 750 !important; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: #355C42; }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    .hero {
        background: linear-gradient(118deg, #ffffff 0%, #f4f8ef 62%, #e7efdf 100%);
        border: 1px solid #d9e5d2;
        border-left: 7px solid var(--forest);
        border-radius: 24px;
        padding: 34px 36px;
        margin: 4px 0 28px 0;
        box-shadow: 0 12px 32px rgba(36, 86, 58, 0.09);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "🌾";
        position: absolute;
        right: 34px;
        top: 18px;
        font-size: 5.2rem;
        opacity: 0.18;
        transform: rotate(-12deg);
        pointer-events: none;
    }
    .hero-content { position: relative; z-index: 1; max-width: 780px; }
    .hero-pill {
        display: inline-flex; align-items: center; gap: 7px;
        background: #e6efdf; color: #24563a; border: 1px solid #d1e1c8;
        border-radius: 999px; padding: 6px 11px; margin-bottom: 13px;
        font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;
    }
    .hero-kicker {
        color: var(--sage);
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero h1 { margin: 0 0 9px 0; font-size: clamp(1.65rem, 3vw, 2.35rem); line-height: 1.18; }
    .hero p { margin: 0; color: var(--muted); font-size: 1rem; line-height: 1.65; }

    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8faf5);
        border: 1px solid var(--border); border-radius: 18px;
        padding: 20px 21px; min-height: 120px;
        box-shadow: 0 7px 22px rgba(36, 86, 58, 0.065);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(36, 86, 58, 0.10); }
    .metric-label { text-transform: uppercase; letter-spacing: .055em; }
    .metric-value { font-variant-numeric: tabular-nums; }
    .metric-label { color: var(--muted); font-size: 0.82rem; font-weight: 700; margin-bottom: 8px; }
    .metric-value { color: var(--forest); font-size: 1.7rem; font-weight: 850; line-height: 1.15; }
    .metric-note { color: var(--muted); font-size: 0.78rem; margin-top: 8px; }
    .section-card {
        background: rgba(255,255,255,.94); border: 1px solid var(--border); border-radius: 18px;
        padding: 22px 24px; margin: 8px 0 18px 0;
        box-shadow: 0 7px 22px rgba(36, 86, 58, 0.045);
    }
    .section-eyebrow { color: #6f8b62; font-size: .74rem; font-weight: 850; letter-spacing: .12em; text-transform: uppercase; }
    .stSelectbox, .stNumberInput { margin-bottom: .3rem; }
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { align-items: stretch; }
    [data-testid="stForm"] { border: 1px solid var(--border); border-radius: 16px; padding: 18px; }
    [data-testid="stPlotlyChart"], [data-testid="stPyplotChart"] { background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 8px; }
    .result-banner { border-radius: 16px; padding: 15px 18px; background: #eaf2e5; border: 1px solid #d4e3cc; color: #24563a; margin: 12px 0; }
    .small-muted { font-size: .86rem; color: #66806d; }
    .workflow-card {
        background: rgba(255,255,255,.94); border: 1px solid #dce6d5;
        border-radius: 20px; padding: 22px 24px 18px 24px; margin: 10px 0 18px 0;
        box-shadow: 0 9px 26px rgba(36, 86, 58, .055);
    }
    .workflow-heading { display:flex; align-items:center; gap:14px; margin-bottom:18px; }
    .step-number {
        display:flex; align-items:center; justify-content:center; flex:0 0 48px; height:48px;
        border-radius:15px; color:#fff; background:linear-gradient(145deg,#2c6244,#17452e);
        box-shadow:0 0 0 5px #e5edde; font-weight:850; font-size:1.05rem;
    }
    .workflow-title { font-size:1.35rem; line-height:1.2; font-weight:800; color:#123d2c; margin:0; }
    .workflow-subtitle { color:#6a8175; font-size:.91rem; margin-top:4px; }
    .info-panel { background:linear-gradient(135deg,#f7f9f2,#fffdf6); border:1px solid #e1e8d8; border-radius:16px; padding:16px 18px; }
    .info-panel-gold { background:linear-gradient(135deg,#fffaf0,#fffdf8); border-color:#f0e4c7; }
    .field-hint { color:#71867a; font-size:.82rem; margin-top:-7px; margin-bottom:10px; }
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background:#f7f9f8 !important; border-color:#d7e1dc !important; min-height:44px;
    }
    [data-testid="stNumberInput"] button { border-radius:8px !important; }
    .workflow-card h3 { margin-top:0; }
    @media (max-width: 900px) {
        .workflow-card { padding:17px; }
        .workflow-title { font-size:1.15rem; }
        .step-number { flex-basis:42px; height:42px; }
    }
    div[data-testid="stDataFrame"] { background: white; }

    .stButton > button, [data-testid="stBaseButton-primary"] {
        background: var(--forest) !important; border: 1px solid var(--forest) !important;
        color: var(--white) !important; border-radius: 11px !important;
        font-weight: 750 !important; min-height: 2.75rem;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover, [data-testid="stBaseButton-primary"]:hover {
        background: #193F2A !important; border-color: #193F2A !important; color: var(--white) !important;
    }
    .stButton > button:focus-visible { outline: 3px solid var(--gold) !important; outline-offset: 2px; }
    [data-baseweb="select"] > div, input, textarea {
        border-radius: 10px !important;
    }
    input:focus, textarea:focus, [data-baseweb="select"]:focus-within {
        border-color: var(--sage) !important; box-shadow: 0 0 0 1px var(--sage) !important;
    }
    [data-testid="stMetric"] {
        background: var(--white); border: 1px solid var(--border);
        border-radius: 14px; padding: 14px 16px;
    }
    [data-testid="stAlert"] { border-radius: 12px; border-width: 1px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
    }
    hr { border-color: var(--border); }
    a { color: var(--forest) !important; text-decoration-color: var(--sage) !important; }
    a:hover { color: var(--sage) !important; }

    @media (max-width: 768px) {
        .hero { padding: 23px 20px; border-radius: 17px; margin-bottom: 20px; }
        .hero:after { right: 12px; top: 8px; font-size: 3rem; }
        .section-card { padding: 16px; }
        .metric-card { padding: 15px; min-height: 100px; }
        .metric-value { font-size: 1.45rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { transition: none !important; animation: none !important; }
    }
    
/* Fix primary button text contrast */
.stButton > button[kind="primary"],
.stFormSubmitButton > button,
button[kind="primary"] {
    background-color: #24563A !important;
    color: #FFFFFF !important;
    border: 1px solid #24563A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Keep text and icons readable */
.stButton > button[kind="primary"] *,
.stFormSubmitButton > button *,
button[kind="primary"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Hover effect */
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover,
button[kind="primary"]:hover {
    background-color: #1B442D !important;
    border-color: #1B442D !important;
    color: #FFFFFF !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Paths and data loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "MLDST_Final_Cleaned.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_FILES = {
    "Linear Regression": "linear_regression_pipeline.pkl",
    "Decision Tree": "decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
}


@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_model(path):
    return joblib.load(path)


data = load_data(DATA_PATH)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <div class="hero-pill">🌱 SDG 2 · Zero Hunger</div>
            <div class="hero-kicker">Machine Learning · Regression · Philippines</div>
            <h1>The Rising Cost of Food</h1>
            <p>Explore regional food price changes and estimate the next month's Consumer Price Index (CPI). Turn historical patterns into clearer, more informed insights.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown("## 🌾 CPI Forecasting")
st.sidebar.caption("REGIONAL FOOD-PRICE INTELLIGENCE")
st.sidebar.markdown("<div style='height:1px;background:#d7e2d0;margin:10px 0 14px 0'></div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate",
    ["Forecast CPI", "Model Evaluation", "About the Project"],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption("Predictions are estimates based on historical CPI patterns, not official price announcements.")

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def get_unique_values(column, fallback):
    if data.empty or column not in data.columns:
        return fallback
    values = data[column].dropna().astype(str).unique().tolist()
    return sorted(values) if values else fallback


def get_default_model_path(model_filename):
    return os.path.join(MODELS_DIR, model_filename)


def load_metrics_file(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def find_col(df, candidates):
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


# ---------------------------------------------------------
# Philippines seasonal climate information (descriptive only)
# ---------------------------------------------------------
def get_climate_info(month):
    """Return broad Philippine seasonal context; this is not a live weather forecast."""
    if month in [12, 1, 2]:
        return (
            "Cool Dry Season",
            "December–February is generally part of the cool dry season, commonly associated "
            "with the northeast monsoon (Amihan).",
        )
    elif month in [3, 4, 5]:
        return (
            "Hot Dry Season",
            "March–May is generally the hot dry season, often the warmest part of the year.",
        )
    elif month in [6, 7, 8]:
        return (
            "Rainy Season",
            "June–August generally falls within the rainy season and is commonly influenced "
            "by the southwest monsoon (Habagat).",
        )
    else:
        return (
            "Rainy Season / Transition",
            "September–November is generally within the rainy season, with seasonal transition "
            "conditions toward the end of the period.",
        )


# ---------------------------------------------------------
# Page: Forecast CPI
# ---------------------------------------------------------
if page == "Forecast CPI":
    st.markdown(
        "<h2 style='margin-bottom:2px;'>Forecast next month's CPI</h2>"
        "<p style='color:#6a8175;margin-top:0;'>Choose a food series, enter recent CPI history, and select a model to estimate the next period.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-eyebrow'>Forecast workspace · SDG 2: Zero Hunger</div>", unsafe_allow_html=True)

    if data.empty:
        st.warning(
            "The dataset was not found. Make sure the file is located at "
            "`data/MLDST_Final_Cleaned.csv` inside your project folder."
        )

    left, right = st.columns([1.18, 0.92], gap="large")

    with left:
        # Step 1: Food series
        st.markdown(
            "<div class='workflow-card'><div class='workflow-heading'>"
            "<div class='step-number'>01</div><div><div class='workflow-title'>Choose your food series</div>"
            "<div class='workflow-subtitle'>Select the region and food commodity you want to forecast.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        locations = get_unique_values("Geolocation", ["National"])
        location = st.selectbox("📍 Region / Geolocation", locations)

        commodities = data.copy()
        if not commodities.empty and "Geolocation" in commodities.columns:
            commodities = commodities[commodities["Geolocation"].astype(str) == str(location)]

        if not commodities.empty and {"Commodity_Code", "Commodity_Name"}.issubset(commodities.columns):
            commodity_options = (
                commodities[["Commodity_Code", "Commodity_Name"]]
                .dropna().drop_duplicates().sort_values("Commodity_Name")
            )
            option_labels = commodity_options["Commodity_Name"].astype(str).tolist()
            label_to_row = {str(row.Commodity_Name): row for row in commodity_options.itertuples(index=False)}
        else:
            commodity_options = pd.DataFrame([{"Commodity_Code": "DEFAULT", "Commodity_Name": "Selected commodity"}])
            option_labels = ["Selected commodity"]
            label_to_row = {option_labels[0]: commodity_options.iloc[0]}

        selected_label = st.selectbox("🍴 Food commodity", option_labels)
        selected_row = label_to_row[selected_label]
        commodity_code = str(selected_row.Commodity_Code)
        commodity_name = str(selected_row.Commodity_Name)
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 2: History and forecast period
        st.markdown(
            "<div class='workflow-card'><div class='workflow-heading'>"
            "<div class='step-number'>02</div><div><div class='workflow-title'>Enter recent CPI history</div>"
            "<div class='workflow-subtitle'>Provide the forecast period and latest available CPI values.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        date_col1, date_col2 = st.columns(2, gap="medium")
        today_year = int(pd.Timestamp.today().year)
        with date_col1:
            year = st.number_input("🗓️ Forecast year", min_value=2000, max_value=2100, value=today_year, step=1)
        with date_col2:
            month_name = st.selectbox(
                "🗓️ Forecast month", list(range(1, 13)),
                format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
                index=(int(pd.Timestamp.today().month) % 12),
            )

        forecast_month_text = pd.Timestamp(2000, int(month_name), 1).strftime("%B")
        st.markdown(
            f"<div class='result-banner'><b>Forecast period: {forecast_month_text} {int(year)}</b><br>"
            "Use the most recent available CPI observations before this month.</div>",
            unsafe_allow_html=True,
        )
        climate_label, _ = get_climate_info(int(month_name))
        st.caption(f"🌦️ Seasonal context: {climate_label}. General context only; not a live weather forecast.")

        cpi_col1, cpi_col2 = st.columns(2, gap="medium")
        with cpi_col1:
            st.markdown("**▥ Latest known CPI**")
            st.caption("Most recent month before the forecast")
            current_cpi = st.number_input("Latest known CPI", min_value=0.01, value=150.0, step=0.1, format="%.3f", label_visibility="collapsed")
            st.markdown("**▥ CPI one month earlier**")
            st.caption("One month before the latest value")
            cpi_lag_1 = st.number_input("CPI one month earlier", min_value=0.01, value=148.0, step=0.1, format="%.3f", label_visibility="collapsed")
        with cpi_col2:
            st.markdown("**▥ CPI two months earlier**")
            st.caption("Two months before the latest value")
            cpi_lag_2 = st.number_input("CPI two months earlier", min_value=0.01, value=147.0, step=0.1, format="%.3f", label_visibility="collapsed")
            st.markdown("**▥ CPI three months earlier**")
            st.caption("Three months before the latest value")
            cpi_lag_3 = st.number_input("CPI three months earlier", min_value=0.01, value=146.0, step=0.1, format="%.3f", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            "<div class='workflow-card'><div class='workflow-heading'>"
            "<div class='step-number'>03</div><div><div class='workflow-title'>Choose your model</div>"
            "<div class='workflow-subtitle'>Select a prediction model to estimate the next period.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        model_name = st.selectbox("⚙️ Regression model", list(MODEL_FILES.keys()))
        model_descriptions = {
            "Linear Regression": "Models the relationship between CPI and time, region, commodity, and recent CPI history.",
            "Decision Tree": "Learns decision rules from historical CPI patterns and input features.",
            "Random Forest": "Combines multiple decision trees to estimate CPI from historical patterns.",
        }
        recommended = " · Recommended based on project evaluation" if model_name == "Linear Regression" else ""
        st.markdown(
            f"<div class='info-panel'><div style='display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;'>"
            f"<b>About {model_name}</b><span style='background:#f8edcf;color:#77591d;border:1px solid #ead7a1;border-radius:99px;padding:4px 10px;font-size:.78rem;font-weight:700;'>"
            f"{'★ Recommended' if model_name == 'Linear Regression' else 'Model option'}</span></div>"
            f"<p style='margin:10px 0 0;color:#536d61;'>{model_descriptions[model_name]}{recommended}</p>"
            "<hr style='border-color:#dce5d5;margin:15px 0;'>"
            "<b>Key inputs used</b><div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;color:#536d61;font-size:.9rem;'>"
            "<span>✓ Region and commodity</span><span>✓ CPI changes and rolling statistics</span>"
            "<span>✓ Month and year</span><span>✓ Seasonal features</span><span>✓ Recent CPI values</span></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div><div class='info-panel info-panel-gold'>"
                    "<b style='font-size:1.05rem;'>Before you forecast</b>"
                    "<p style='color:#6a6655;margin:8px 0 12px;'>Make sure your inputs are correct for a more reliable estimate.</p>"
                    "<p style='margin:6px 0;'>① Use CPI observations from the same region and commodity, ordered from oldest to newest.</p>"
                    "<p style='margin:6px 0 0;'>② The latest known CPI should be the most recent observation available before the forecast month.</p>"
                    "</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("↗  Generate CPI forecast", type="primary", use_container_width=True)
        st.caption("Predictions are estimates, not official PSA forecasts or direct peso-price predictions.")

    if predict_clicked:
        cpi_values = [float(current_cpi), float(cpi_lag_1), float(cpi_lag_2), float(cpi_lag_3)]
        if any(value <= 0 for value in cpi_values):
            st.error("All CPI values must be greater than zero.")
        elif not 50 <= float(current_cpi) <= 300:
            st.error("The latest CPI is outside the provisional safety range (50–300). Please verify it against the source data.")
        elif max(cpi_values) / min(cpi_values) > 2:
            st.error("The entered CPI history varies too widely. Please verify that all values belong to the same region and commodity.")
        else:
            model_path = get_default_model_path(MODEL_FILES[model_name])
            if not os.path.exists(model_path):
                st.error(f"Model file not found: `models/{MODEL_FILES[model_name]}`. Make sure the model files are in the project's `models` folder.")
            else:
                try:
                    model = load_model(model_path)
                    input_df = prepare_prediction_input(
                        geolocation=location, commodity_code=commodity_code, commodity_name=commodity_name,
                        year=int(year), month=int(month_name), current_cpi=float(current_cpi),
                        cpi_lag_1=float(cpi_lag_1), cpi_lag_2=float(cpi_lag_2), cpi_lag_3=float(cpi_lag_3),
                    )
                    prediction = float(np.asarray(model.predict(input_df)).reshape(-1)[0])
                    if not np.isfinite(prediction) or prediction <= 0:
                        st.error("The model returned a non-positive or invalid CPI prediction. Please verify the inputs or try another model.")
                    else:
                        change = prediction - float(current_cpi)
                        pct_change = (change / float(current_cpi)) * 100
                        forecast_month_label = pd.Timestamp(int(year), int(month_name), 1).strftime("%B %Y")
                        st.markdown("---")
                        st.subheader(f"Forecast result · {forecast_month_label}")
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted CPI</div><div class="metric-value">{prediction:.2f}</div><div class="metric-note">Estimated index points</div></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Change from latest CPI</div><div class="metric-value">{change:+.2f}</div><div class="metric-note">CPI index points</div></div>', unsafe_allow_html=True)
                        with m3:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Relative change</div><div class="metric-value">{pct_change:+.2f}%</div><div class="metric-note">Compared with latest known CPI</div></div>', unsafe_allow_html=True)
                        if change > 0.01:
                            st.warning(f"The model estimates that CPI may increase by {change:.2f} points ({pct_change:.2f}%) compared with the latest known CPI.")
                        elif change < -0.01:
                            st.info(f"The model estimates that CPI may decrease by {abs(change):.2f} points ({abs(pct_change):.2f}%) compared with the latest known CPI.")
                        else:
                            st.success("The model estimates little to no change from the latest known CPI.")
                        with st.expander("View the model input features"):
                            st.dataframe(input_df, width="stretch")
                        st.caption(f"Model used: {model_name}. This is an estimate, not an official PSA forecast or a direct prediction of a specific peso price.")
                except Exception as exc:
                    st.error(f"Could not generate the forecast: {exc}")
                    st.info("Check that prediction_utils.py matches the model's training features and that the saved model files are compatible.")

# ---------------------------------------------------------
# Page: Model Evaluation
# ---------------------------------------------------------
elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    st.write("Compare the candidate regression models using the saved validation and held-out test results.")
    st.markdown("<div class='section-eyebrow'>Evidence · comparison · model selection</div>", unsafe_allow_html=True)

    final_metrics = load_metrics_file("final_model_comparison.csv")
    validation_metrics = load_metrics_file("validation_results.csv")

    if final_metrics.empty and validation_metrics.empty:
        st.info(
            "Evaluation CSV files were not found. Add `final_model_comparison.csv` and/or "
            "`validation_results.csv` to the `results` folder to show your measured results here."
        )
    else:
        if not final_metrics.empty:
            st.markdown("### Final held-out test results")
            st.dataframe(final_metrics, use_container_width=True, hide_index=True)
        if not validation_metrics.empty:
            st.markdown("### Time-aware validation results")
            st.dataframe(validation_metrics, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="section-card">
            <strong>How to read the metrics</strong>
            <ul>
                <li><b>MAE:</b> average absolute prediction error, in CPI index points. Lower is better.</li>
                <li><b>RMSE:</b> gives larger errors more weight, in CPI index points. Lower is better.</li>
                <li><b>R²:</b> indicates how much variation is explained by the model relative to a baseline. Higher is generally better.</li>
            </ul>
            <p style="color:#5f7467;margin-bottom:0;">Use validation results for model selection and keep the held-out test results as the final evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Page: About the Project
# ---------------------------------------------------------
else:
    st.subheader("About this project")
    st.markdown("<div class='section-eyebrow'>Purpose · methodology · responsible use</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-card">
            <h3>The Rising Cost of Food: Exploring Regional Food Price Changes in the Philippines</h3>
            <p>
            This machine-learning project uses historical Consumer Price Index (CPI) observations
            to estimate the following month's CPI for a selected region and food commodity.
            </p>
            <p><b>Problem:</b> Food-price changes can affect household budgets and local planning.</p>
            <p><b>Approach:</b> Regression models learn patterns from past CPI values and time-related features.</p>
            <p><b>Intended users:</b> Students, researchers, and people exploring regional food-price trends.</p>
            <p><b>Responsible use:</b> Results are estimates, not official statistics or guaranteed future outcomes.
            Data quality, unusual economic events, supply disruptions, and policy changes may affect real prices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Project files")
    st.code(
        "ML_CPI_Project_Final/\n"
        "├── app.py\n"
        "├── prediction_utils.py\n"
        "├── data/\n"
        "│   └── MLDST_Final_Cleaned.csv\n"
        "├── models/\n"
        "│   ├── linear_regression_pipeline.pkl\n"
        "│   ├── decision_tree_pipeline.pkl\n"
        "│   └── random_forest_pipeline.pkl\n"
        "└── results/\n"
        "    ├── final_model_comparison.csv\n"
        "    └── validation_results.csv",
        language="text",
    )

st.markdown("---")
st.caption("CPI Forecasting System • Academic machine-learning project • For learning and exploratory analysis")
