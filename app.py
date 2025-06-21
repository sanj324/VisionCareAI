
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="VisionCare AI", layout="centered")
model = joblib.load("vision_model.pkl")

st.markdown("""
    <h1 style='text-align: center; color: navy;'>🧠 VisionCare AI</h1>
    <h4 style='text-align: center; color: gray;'>AI-powered Prediction of Early Vision Issues</h4>
""", unsafe_allow_html=True)
st.divider()

# Sidebar: Patient Info
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

# Prediction logic
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

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)
    confidence = model.predict_proba(input_df)[0][1] * 100
    result_label = "⚠️ High Risk of Vision Issue" if prediction[0] == 1 else "✅ Low Risk"
    recommendation = (
        "🔴 Immediate consultation with an ophthalmologist recommended. Reduce screen time, track symptoms."
        if prediction[0] == 1 else
        "🟢 Maintain healthy vision habits and schedule annual checkups."
    )

    st.markdown("### 🔬 Prediction Result")
    st.markdown(f"""
        <div style='background-color: #ffe6e6; padding: 10px; border-radius: 10px; font-size: 18px;'>
        <strong>{result_label}</strong> (Confidence: {confidence:.2f}%)
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Confidence Gauge")
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

    st.markdown("### 🧾 Clinical Recommendation")
    st.success(recommendation)

    def generate_pdf(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "VisionCare AI Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        for key, value in data.items():
            pdf.cell(0, 10, f"{key}: {value}", ln=True)
        pdf.output("VisionCare_Report.pdf")

    report_data = {
        "Age": age,
        "Diabetes": diabetes,
        "Blood Pressure": f"{bp} mm Hg",
        "Screen Time": f"{screen_time} hrs",
        "Cholesterol": f"{cholesterol} mg/dL",
        "HbA1c": f"{hba1c}%",
        "Smoking": smoking,
        "Wears Glasses": glasses,
        "Prediction": result_label,
        "Confidence": f"{confidence:.2f}%",
        "Recommendation": recommendation
    }

    generate_pdf(report_data)
    with open("VisionCare_Report.pdf", "rb") as f:
        st.download_button("📄 Download VisionCare Report (PDF)", f, file_name="VisionCare_Report.pdf")

st.markdown("""<br><hr>
<small>⚠️ This tool is for educational/clinical support. Please consult a licensed ophthalmologist for medical decisions.</small>""", unsafe_allow_html=True)
