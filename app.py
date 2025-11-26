import streamlit as st
import pandas as pd
import pickle
import joblib
import os

st.set_page_config(page_title="Azithromycin Decision Tool", layout="centered")

st.title("🩺 Azithromycin Benefit Prediction")

# Load pickle model
# def load_model():
#     model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
#     return joblib.load(model_path) 
model = joblib.load("model.pkl")

# model = load_model()


# Input form
with st.form("patient_form"):
    age_months = st.number_input("Age (months)", 1, 60, 12)
    muac = st.number_input("MUAC (cm)", 9.0, 14.0, 12.0)
    local_resistance_index = st.slider("Local Resistance Index", 0.1, 0.9, 0.4)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dehydration_grade = st.selectbox("Dehydration Grade", ["None", "Mild", "Moderate", "Severe"])
    prior_antibiotic_use = st.selectbox("Prior Antibiotic Use", [0, 1])
    fever = st.selectbox("Fever", [0, 1])
    vomiting = st.selectbox("Vomiting", [0, 1])
    blood_in_stool = st.selectbox("Blood in Stool", [0, 1])
    diarrhea_duration_baseline = st.number_input("Baseline Diarrhea Duration (days)", 1, 10, 3)
    treatment_given = st.selectbox("Azithromycin Already Given", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "age_months": age_months,
        "sex": sex,
        "dehydration_grade": dehydration_grade,
        "muac": muac,
        "prior_antibiotic_use": prior_antibiotic_use,
        "local_resistance_index": local_resistance_index,
        "fever": fever,
        "vomiting": vomiting,
        "blood_in_stool": blood_in_stool,
        "diarrhea_duration_baseline": diarrhea_duration_baseline,
        "treatment_given": treatment_given
    }])

    # Prediction
    prob = model.predict_proba(input_df)[0][1]

    st.metric("Predicted benefit probability", f"{prob*100:.2f}%")

    if prob < 0.30:
        st.warning("Low chance of benefit — avoid unnecessary antibiotics.")
    elif prob < 0.60:
        st.info("Moderate chance of benefit — use clinical judgment.")
    else:
        st.success("High probability of benefit — azithromycin likely helpful.")
