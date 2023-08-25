import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load the class prediction model
model1 = tf.keras.models.load_model('model1.h5')

# Load the probability prediction model
model2 = tf.keras.models.load_model('model2.h5')

# Load the scaler
scaler = joblib.load('scaler3.pkl')

# Sample input fields corresponding to the columns in your X_train dataset
st.title("Loan Approval Prediction")

no_of_dependents = st.number_input("Number of Dependents:")
income_annum = st.number_input("Annual Income:")
loan_amount = st.number_input("Loan Amount:")
loan_term = st.number_input("Loan Term (in months):")
cibil_score = st.number_input("CIBIL Score:")
residential_assets_value = st.number_input("Residential Assets Value:")
commercial_assets_value = st.number_input("Commercial Assets Value:")
luxury_assets_value = st.number_input("Luxury Assets Value:")
bank_asset_value = st.number_input("Bank Asset Value:")

# Radio button to select the prediction model
selected_model = st.radio("Select Prediction Model", ["Class Prediction", "Probability Prediction"])

# Make predictions when a button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([no_of_dependents, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value])

    # Standardize the input data using the loaded scaler
    input_data = scaler.transform(input_data.reshape(1, -1))

    # Use the selected model to make predictions
    if selected_model == "Class Prediction":
        prediction = model1.predict(input_data)
        loan_approval_class = int(round(prediction[0, 0]))
        st.write(f"Loan Approval Class: {loan_approval_class}")
    elif selected_model == "Probability Prediction":
        prediction = model2.predict(input_data)
        st.write(f"Loan Approval Probability: {prediction[0, 0] * 100:.2f}%")
