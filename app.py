import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pickle
from model import load_data, train_random_forest


model2= joblib.load("model.pkl")
# # Load data
# df = load_data()

# # Train or load model
# #model = train_model()

# # Streamlit app
# st.title("Prostate Cancer Prediction App")
# st.write("Dataset preview:")
# st.write(df.head())



st.title("Prostate Cancer Diagnosis Prediction App")

with st.spinner("Loading data and training model..."):
    X, y = load_data()
    model, scaler, accuracy, report = train_random_forest(X, y)

st.success(f"Model trained! Accuracy on test set: {accuracy:.2f}")

# Get feature names
feature_names = list(X.columns)

# Build input form dynamically
st.header("Enter Patient Details")
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

# Prediction button
if st.button("Predict Diagnosis"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model2.predict(input_scaled)
    st.subheader(f"Prediction: {prediction[0]}")
