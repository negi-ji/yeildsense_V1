import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="YieldSense",
    page_icon="🌾",
    layout="wide"
)

MODEL_REPO = "bhgugvgvtyuvyctrctcjuy/yieldsense-random-forest"
MODEL_FILE = "models/yieldsense_random_forest.pkl"


@st.cache_resource
def load_model():

@st.cache_resource
def load_model():
     model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    return joblib.load(model_path)


model = load_model()

df = pd.read_csv("data/yield_df.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.rename(columns={
    "hg/ha_yield": "yield",
    "average_rain_fall_mm_per_year": "rainfall",
    "pesticides_tonnes": "pesticides",
    "avg_temp": "temperature"
})

st.title("🌾 YieldSense")
st.subheader("Precision Agriculture Yield Forecaster")

st.write(
    "Predict crop yield using historical agricultural and weather data."
)

st.sidebar.header("Farm Information")

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

predict_button = st.button(
    "🌱 Predict Crop Yield",
    use_container_width=True
)

if predict_button:

    input_data = pd.DataFrame({
        "Area": [area],
        "Item": [crop],
        "Year": [year],
        "rainfall": [rainfall],
        "pesticides": [pesticides],
        "temperature": [temperature]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Yield",
            f"{prediction:,.0f} hg/ha"
        )

    with col2:
        st.metric("Crop", crop)

    with col3:
        st.metric("Area", area)

    yield_kg = prediction / 100
    yield_tonnes = prediction / 100000

    st.subheader("Yield Conversion")

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

    st.subheader("Prediction Inputs")

    st.dataframe(
        input_data,
        use_container_width=True
    )

st.divider()

st.subheader("About the Dataset")

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
