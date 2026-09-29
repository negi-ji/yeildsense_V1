import os
import requests
import joblib
import streamlit as st
import pandas as pd


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="YieldSense",
    page_icon="🌾",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_URL = os.getenv("MODEL_URL")


@st.cache_resource
def load_model():

    if not MODEL_URL:
        st.error(
            "MODEL_URL is not configured. "
            "Please add your model URL in Render Environment Variables."
        )
        st.stop()

    model_path = "/tmp/yieldsense_random_forest.pkl"

    if not os.path.exists(model_path):

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(model_path, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

    return joblib.load(model_path)


model = load_model()


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/yield_df.csv"
)


# ==========================================
# CLEAN / RENAME COLUMNS
# ==========================================

if "Unnamed: 0" in df.columns:

    df = df.drop(
        columns=["Unnamed: 0"]
    )


df = df.rename(
    columns={
        "hg/ha_yield": "yield",
        "average_rain_fall_mm_per_year": "rainfall",
        "pesticides_tonnes": "pesticides",
        "avg_temp": "temperature"
    }
)


# ==========================================
# TITLE
# ==========================================

st.title("🌾 YieldSense")

st.subheader(
    "Precision Agriculture Yield Forecaster"
)

st.write(
    "Predict crop yield using historical "
    "agricultural and weather data."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "Farm Information"
)


area = st.sidebar.selectbox(
    "Country / Area",
    sorted(df["Area"].unique())
)


crop = st.sidebar.selectbox(
    "Crop",
    sorted(df["Item"].unique())
)


year = st.sidebar.number_input(
    "Year",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()) + 10,
    value=int(df["Year"].max())
)


rainfall = st.sidebar.number_input(
    "Rainfall (mm/year)",
    min_value=0.0,
    max_value=float(df["rainfall"].max()),
    value=float(df["rainfall"].mean())
)


pesticides = st.sidebar.number_input(
    "Pesticides (tonnes)",
    min_value=0.0,
    max_value=float(df["pesticides"].max()),
    value=float(df["pesticides"].mean())
)


temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=float(df["temperature"].min()),
    max_value=float(df["temperature"].max()),
    value=float(df["temperature"].mean())
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

predict_button = st.button(
    "🌱 Predict Crop Yield",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    input_data = pd.DataFrame(
        {
            "Area": [area],
            "Item": [crop],
            "Year": [year],
            "rainfall": [rainfall],
            "pesticides": [pesticides],
            "temperature": [temperature]
        }
    )


    # ======================================
    # MAKE PREDICTION
    # ======================================

    prediction = model.predict(
        input_data
    )[0]


    # ======================================
    # RESULT
    # ======================================

    st.success(
        "Prediction completed!"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Yield",
            f"{prediction:,.0f} hg/ha"
        )


    with col2:

        st.metric(
            "Crop",
            crop
        )


    with col3:

        st.metric(
            "Area",
            area
        )


    # ======================================
    # CONVERT TO KG / HA
    # ======================================

    yield_kg = prediction / 100

    yield_tonnes = prediction / 100000


    st.subheader(
        "Yield Conversion"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Yield (kg/ha)",
            f"{yield_kg:,.2f}"
        )


    with col2:

        st.metric(
            "Yield (tonnes/ha)",
            f"{yield_tonnes:,.2f}"
        )


    # ======================================
    # INPUT SUMMARY
    # ======================================

    st.subheader(
        "Prediction Inputs"
    )


    st.dataframe(
        input_data,
        use_container_width=True
    )


# ==========================================
# DATASET INFORMATION
# ==========================================

st.divider()

st.subheader(
    "About the Dataset"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Records",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Countries / Areas",
        df["Area"].nunique()
    )


with col3:

    st.metric(
        "Crops",
        df["Item"].nunique()
    )
