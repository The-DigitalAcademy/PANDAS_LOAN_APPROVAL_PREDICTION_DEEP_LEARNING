import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model
model = tf.keras.models.load_model('manoko4.h5')

# Load the StandardScaler fitted to your training data
scaler = joblib.load('scaler.pkl')  # Replace with the path to your scaler file

st.title("Loan Approval Prediction")

# Sample input fields corresponding to the columns in your X_train dataset
st.title("Loan Approval Prediction")

no_of_dependents = st.number_input("Number of Dependents:", min_value=0, step=1.0)
income_annum = st.number_input("Annual Income:", min_value=0.0)
loan_amount = st.number_input("Loan Amount:", min_value=0.0)
loan_term = st.number_input("Loan Term (in months):", min_value=0.0)
cibil_score = st.number_input("CIBIL Score:", min_value=0.0)
residential_assets_value = st.number_input("Residential Assets Value:", min_value=0.0)
commercial_assets_value = st.number_input("Commercial Assets Value:", min_value=0.0)
luxury_assets_value = st.number_input("Luxury Assets Value:", min_value=0.0)
bank_asset_value = st.number_input("Bank Asset Value:", min_value=0.0)

# Make predictions when a button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([no_of_dependents, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value])

    # Standardize the input data using the loaded scaler
    input_data = scaler.transform(input_data.reshape(1, -1))

    # Use the loaded model to make predictions
    prediction = model.predict(input_data)

    # Predicted class (0 or 1)
    loan_approval_class = np.argmax(prediction)

    # Display the prediction class and probability as a percentage
    st.write(f"Loan Approval Class: {loan_approval_class}")
    st.write(f"Loan Approval Probability: {prediction[0, loan_approval_class] * 100:.2f}%")
