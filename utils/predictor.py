import pandas as pd
import joblib

model = joblib.load("model/vision_model.pkl")

def predict_vision_risk(input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction[0], probability[0][1]
