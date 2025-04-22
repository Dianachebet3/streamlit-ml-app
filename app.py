import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("ML Classifier App")

# Example input fields - modify as per your features
feature_1 = st.number_input("Enter Feature 1", value=0.0)
feature_2 = st.number_input("Enter Feature 2", value=0.0)
feature_3 = st.selectbox("Choose Feature 3", options=["Option1", "Option2"])

# Mapping categorical options to numbers
feature_3_encoded = {"Option1": 0, "Option2": 1}[feature_3]

if st.button("Predict"):
    # Create dataframe with the inputs
    input_data = pd.DataFrame([[feature_1, feature_2, feature_3_encoded]],
                              columns=["feature_1", "feature_2", "feature_3"])

    # Scale inputs
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    st.success(f"Prediction: {prediction[0]}")
