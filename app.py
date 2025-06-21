import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Setup
st.set_page_config(page_title="VisionCare AI", layout="centered")

# Load model
model = joblib.load("vision_model.pkl")

# Title
st.markdown("<h1 style='text-align: center; color: navy;'>🧠 VisionCare AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI-powered Prediction of Early Vision Issues</h4>", unsafe_allow_html=True)

st.divider()

# Sidebar: Patient Details
st.sidebar.header("👤 Patient Profile")
age = st.sidebar.slider("Age", 18, 90, 35)
diabetes = st.sidebar.radio("Diabetes", ["No", "Yes"])
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 90, 180, 120)
screen_time = st.sidebar.slider("Avg Screen Time (hrs/day)", 0, 12, 5)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 300, 180)
hba1c = st.sidebar.slider("HbA1c (%)", 4.0, 14.0, 6.5)

# Symptoms and Lifestyle
st.subheader("🩺 Symptoms & Lifestyle Factors")
col1, col2 = st.columns(2)
with col1:
    blurred = st.checkbox("Blurred Vision")
    eye_pain = st.checkbox("Eye Pain")
    family_history = st.checkbox("Family History of Eye Disease")
    night_vision = st.checkbox("Night Vision Difficulty")
with col2:
    headache = st.checkbox("Frequent Headaches")
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    tear = st.selectbox("Tear Production", ["Low", "Normal", "Excessive"])
    glasses = st.radio("Wears Glasses?", ["Yes", "No"])

vision_rating = st.select_slider("👁️ Vision Sharpness", ["Excellent", "Good", "Average", "Poor"])
occupation = st.selectbox("Occupation Type", ["Screen-based", "Non-screen-based"])
blue_light = st.selectbox("Blue Light Exposure", ["Low", "Moderate", "High"])

# Predict Button
if st.button("🔍 Predict Vision Risk"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Diabetes": 1 if diabetes == "Yes" else 0,
        "Blood_Pressure": bp,
        "Screen_Time": screen_time,
        "Blurred_Vision": int(blurred),
        "Eye_Pain": int(eye_pain),
        "Headache": int(headache),
        "Cholesterol": cholesterol,
        "HbA1c": hba1c,
        "Smoking_Status_Former": 1 if smoking == "Former" else 0,
        "Smoking_Status_Never": 1 if smoking == "Never" else 0,
        "Family_History": int(family_history),
        "Vision_Sharpness_Good": 1 if vision_rating == "Good" else 0,
        "Vision_Sharpness_Average": 1 if vision_rating == "Average" else 0,
        "Vision_Sharpness_Poor": 1 if vision_rating == "Poor" else 0,
        "Occupation_Type_Screen-based": 1 if occupation == "Screen-based" else 0,
        "Blue_Light_Exposure_Moderate": 1 if blue_light == "Moderate" else 0,
        "Blue_Light_Exposure_High": 1 if blue_light == "High" else 0,
        "Night_Vision_Difficulty": int(night_vision),
        "Tear_Production_Normal": 1 if tear == "Normal" else 0,
        "Tear_Production_Excessive": 1 if tear == "Excessive" else 0,
        "Wears_Glasses_Yes": 1 if glasses == "Yes" else 0
    }])

    # Align with model training columns
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    # Prediction
    prediction = model.predict(input_df)
    confidence = model.predict_proba(input_df)[0][1] * 100
    result_label = "⚠️ High Risk of Vision Issue" if prediction[0] == 1 else "✅ Low Risk"

    # Display Result
    st.subheader("🔬 Prediction Result")
    st.markdown(f"<div style='background-color: #ffe6e6; padding: 10px; border-radius: 10px;'>"
                f"<strong>{result_label}</strong> (Confidence: {confidence:.2f}%)</div>", unsafe_allow_html=True)

    # Confidence Gauge
    st.subheader("📊 Confidence Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Risk Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson" if confidence > 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#c8f7c5"},
                {'range': [50, 100], 'color': "#f5b7b1"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""<br><hr>
<small>⚠️ This tool is for educational/clinical support. Please consult a licensed ophthalmologist for medical decisions.</small>""", unsafe_allow_html=True)
