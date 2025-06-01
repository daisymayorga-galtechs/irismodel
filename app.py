import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Saved Model, Scaler, and Label Encoder ---
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')

    # Define feature names for the input DataFrame
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

except FileNotFoundError:
    st.error("Error: Model, scaler, or label encoder files not found.")
    st.info("Please ensure 'logistic_regression_model.joblib', 'standard_scaler.joblib', and 'label_encoder.joblib' are in the same directory.")
    st.stop() # Stop the app if files are missing

# --- 2. Streamlit Application Title and Description ---
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the measurements of an Iris flower to predict its species!")
st.markdown("---")

# --- 3. User Input Widgets ---
st.header("Flower Measurements (cm)")

sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.markdown("---")

# --- 4. Prediction Button and Logic ---
if st.button("Predict Species"):
    # Collect input into a DataFrame
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=feature_names) # Use the corrected feature_names here

    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction_encoded = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)

    # Decode the numerical prediction back to the original species name
    predicted_species = label_encoder.inverse_transform(prediction_encoded)[0]

    # Display Results
    st.header("Prediction Results:")
    st.success(f"The predicted Iris species is: **{predicted_species}**")

    st.subheader("Prediction Probabilities:")
    proba_df = pd.DataFrame({
        'Species': label_encoder.classes_,
        'Probability': prediction_proba[0]
    }).sort_values(by='Probability', ascending=False)
    st.dataframe(proba_df.style.format({'Probability': "{:.2%}"}))

    st.balloons()