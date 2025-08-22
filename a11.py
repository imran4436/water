import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load Model and Scaler ---
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please run the training script first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- UI Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .st-emotion-cache-10trblm {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Application Header ---
st.title("💧 Water Potability Prediction")
st.write(
    "Enter the water quality parameters below to predict if the water is safe for consumption. "
    "This app uses a RandomForest model to make predictions."
)

# --- Input Form ---
st.header("Enter Water Quality Metrics")

# Create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input('pH Level', min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="pH of the water (0-14)")
    hardness = st.number_input('Hardness', min_value=0.0, value=150.0, help="Water hardness in mg/L")
    solids = st.number_input('Solids (TDS)', min_value=0.0, value=20000.0, help="Total dissolved solids in ppm")
    chloramines = st.number_input('Chloramines', min_value=0.0, value=7.0, help="Chloramines concentration in ppm")
    sulfate = st.number_input('Sulfate', min_value=0.0, value=330.0, help="Sulfate concentration in mg/L")

with col2:
    conductivity = st.number_input('Conductivity', min_value=0.0, value=400.0, help="Electrical conductivity of water in μS/cm")
    organic_carbon = st.number_input('Organic Carbon', min_value=0.0, value=14.0, help="Total organic carbon in ppm")
    trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=66.0, help="Trihalomethanes concentration in μg/L")
    turbidity = st.number_input('Turbidity', min_value=0.0, value=4.0, help="Measure of light-scattering properties of water in NTU")

# --- Prediction Logic ---
if st.button('Predict Potability'):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity]
    })

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display the result
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.success("✅ The water is **Potable** (Safe for consumption).")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error("❌ The water is **Not Potable** (Not safe for consumption).")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

# --- Sidebar Information ---
st.sidebar.header("About")
st.sidebar.info(
    "This web application is designed to predict water potability based on "
    "physicochemical properties. The underlying model is a RandomForest Classifier "
    "trained on a balanced dataset to ensure fair predictions."
)
st.sidebar.header("Attribute Information")
st.sidebar.markdown("""
- **pH:** pH of water.
- **Hardness:** Hardness is mainly caused by calcium and magnesium salts.
- **Solids (TDS):** Total dissolved solids.
- **Chloramines:** A disinfectant for water.
- **Sulfate:** A naturally occurring substance.
- **Conductivity:** Electrical conductivity of water.
- **Organic Carbon:** Total organic carbon.
- **Trihalomethanes:** Chemicals that may be found in water treated with chlorine.
- **Turbidity:** A measure of the cloudiness of water.
""")
import pickle
import streamlit as st
import numpy as np

# Load model & scaler
try:
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except FileNotFoundError:
    st.error("❌ Model or scaler files not found. Please run the training script first.")
    st.stop()

st.title("💧 Water Potability Prediction App")

# User input
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=350.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=450.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=70.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

if st.button("Predict Potability"):
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.success("✅ This water is Potable (Safe to drink)")
    else:
        st.error("🚱 This water is Not Potable (Unsafe to drink)")



