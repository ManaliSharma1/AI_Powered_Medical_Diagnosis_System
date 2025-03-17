import streamlit as st
import pickle
import numpy as np


# Change Name & Logo
st.set_page_config(page_title="Disease Prediction", page_icon="👨‍⚕️")


# Hiding Streamlit add-ons
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Adding Background Image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://i.ibb.co/SwGGkBPS/medical-diagnosis-bg.png") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Title
st.markdown(
    """
    <h1 style='text-align: center; color: white; text-shadow: 2px 2px 5px yellow;'>🩺 AI-Powered Medical Diagnosis System 🩺</h1>
    """,
    unsafe_allow_html=True
)


# Set Sidebar Layout
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            width: 300px !important;  /* Adjust sidebar width */
        }
        div[role="radiogroup"] > label {
            display: block;
            padding: 15px;
            font-size: 20px;  /* Increase font size */
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar Title
st.sidebar.markdown(
    "<h2 style='text-align: center;'>🔍 Select Disease to Predict</h2>",
    unsafe_allow_html=True,
)

# Disease Selection
diseases = ["Diabetes", "Heart Disease", "Hyperthyroid", "Lung Cancer", "Parkinsons"]
icons = ["🩸", "❤️", "🦋", "🫁", "🧠"]

selected_disease = st.sidebar.radio(
    "Select Disease:",  # Provide a label
    options=diseases, 
    format_func=lambda x: f"{icons[diseases.index(x)]} {x}", 
    key="disease_selector",
    label_visibility="collapsed"  # Hides the label while keeping it accessible
)


# Load Models
import os

# Load Models
def load_model_scaler(model_name):
    model_path = os.path.join("Models", f"{model_name}_model.pkl")
    scaler_path = os.path.join("Models", f"{model_name}_scaler.pkl")
    
    with open(model_path, "rb") as model_file, open(scaler_path, "rb") as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    
    return model, scaler

# Dictionary to Store Models
models = {}
diseases = ["diabetes", "heart", "hyperthyroid", "lung_cancer", "parkinson"]

for disease in diseases:
    models[disease] = load_model_scaler(disease)

# Access models like this:
diabetes_model, diabetes_scaler = models["diabetes"]
heart_model, heart_scaler = models["heart"]
hyperthyroid_model, hyperthyroid_scaler = models["hyperthyroid"]
lung_cancer_model, lung_cancer_scaler = models["lung_cancer"]
parkinson_model, parkinson_scaler = models["parkinson"]


# Function for Prediction
def predict_disease(model, scaler, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    if input_array.shape[1] != scaler.n_features_in_:
        st.error(f"Incorrect number of features. Expected {scaler.n_features_in_}, but got {input_array.shape[1]}.")
        return "Error"
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    return "Positive" if prediction[0] == 1 else "Negative"


# Input Fields
st.markdown(f"<h2 style='color: white;'>{selected_disease} Prediction</h2>", unsafe_allow_html=True)

def input_field(label, min_value, max_value, step):
    default_value = min_value if isinstance(min_value, (int, float)) else 0
    st.markdown(f"<p style='color: white; font-weight: bold; margin-bottom: 0px;'>{label}</p>", unsafe_allow_html=True)
    return st.number_input(label, min_value=min_value, max_value=max_value, step=step, value=default_value, key=label, label_visibility="collapsed")

if selected_disease == "Diabetes":
    input_data = [
        input_field("Pregnancies", 0, 20, 1),
        input_field("Glucose Level", 0, 300, 1),
        input_field("Blood Pressure", 0, 180, 1),
        input_field("Skin Thickness", 0, 100, 1),
        input_field("Insulin", 0, 900, 1),
        input_field("BMI", 0.0, 50.0, 0.1),
        input_field("Diabetes Pedigree Function", 0.0, 2.5, 0.01),
        input_field("Age", 0, 100, 1)
    ]
    model, scaler = diabetes_model, diabetes_scaler

elif selected_disease == "Heart Disease":
    st.markdown("<p style='color: white; font-weight: bold;'>Sex</p>", unsafe_allow_html=True)
    Sex = 1 if st.selectbox("", options=["Male", "Female"]) == "Male" else 0

    input_data = [
        input_field("Age", 0, 100, 1),
        Sex,
        input_field("Chest Pain Type", 0, 3, 1),
        input_field("Resting Blood Pressure", 0, 200, 1),
        input_field("Cholesterol", 0, 600, 1),
        input_field("Fasting Blood Sugar > 120 mg/dl", 0, 1, 1),
        input_field("Resting ECG", 0, 2, 1),
        input_field("Max Heart Rate Achieved", 0, 250, 1),
        input_field("Exercise Induced Angina", 0, 1, 1),
        input_field("Oldpeak ST Depression", 0.0, 6.2, 0.1),
        input_field("Slope of ST Segment", 0, 2, 1),
        input_field("Number of Major Vessels", 0, 4, 1),
        input_field("Thalassemia", 0, 3, 1)
    ]
    model, scaler = heart_model, heart_scaler

elif selected_disease == "Hyperthyroid":
    st.markdown("<p style='color: white; font-weight: bold;'>Sex</p>", unsafe_allow_html=True)
    Sex = 1 if st.selectbox("", options=["Male", "Female"]) == "Male" else 0
    input_data = [
        input_field("Age", 0, 100, 1),
        Sex,
        input_field("On Thyroxine", 0, 1, 1),
        input_field("TSH", 0.0, 100.0, 0.1),
        input_field("T3 Measured", 0, 1, 1),
        input_field("T3", 0.0, 30.0, 0.1),
        input_field("TT4", 0.0, 300.0, 0.1)
    ]
    model, scaler = hyperthyroid_model, hyperthyroid_scaler

elif selected_disease == "Lung Cancer":
    st.markdown("<p style='color: white; font-weight: bold;'>GENDER</p>", unsafe_allow_html=True)
    GENDER = 1 if st.selectbox("", options=["Male", "Female"]) == "Male" else 0
    input_data = [
        GENDER,
        input_field("AGE", 0, 100, 1),
        input_field("SMOKING", 0, 2, 1),
        input_field("YELLOW_FINGERS", 0, 2, 1),
        input_field("ANXIETY", 0, 2, 1),
        input_field("PEER_PRESSURE", 0, 2, 1),
        input_field("CHRONIC DISEASE", 0, 2, 1),
        input_field("FATIGUE", 0, 2, 1),
        input_field("ALLERGY", 0, 2, 1),
        input_field("WHEEZING", 0, 2, 1),
        input_field("ALCOHOL CONSUMING", 0, 2, 1),
        input_field("COUGHING", 0, 2, 1),
        input_field("SHORTNESS OF BREATH", 0, 2, 1),
        input_field("SWALLOWING DIFFICULTY", 0, 2, 1),
        input_field("CHEST PAIN", 0, 2, 1)
    ]
    model, scaler = lung_cancer_model, lung_cancer_scaler

elif selected_disease == "Parkinsons":
    input_data = [
        input_field(feature, 0.0, 650.0, 0.1) for feature in [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]
    ]
    model, scaler = parkinson_model, parkinson_scaler

    
# Predict Button
if st.button(f"Predict {selected_disease}"):
    result = predict_disease(model, scaler, input_data)
    color = "darkgreen" if result == "Positive" else "red"
    st.markdown(f"<h3 style='background-color: {color}; color: white; padding: 10px; text-align: center; border-radius: 5px;'>{result}</h3>", unsafe_allow_html=True)
