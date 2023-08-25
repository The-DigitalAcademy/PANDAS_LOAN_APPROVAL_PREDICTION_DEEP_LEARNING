import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('manoko3.h5')

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

# Make predictions when a button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([no_of_dependents, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value])

    # Standardize the input data using the same scaler used during training
    # Load the scaler (replace with the path to your scaler file)
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)  # Fit the scaler on your training data
    input_data = scaler.transform(input_data.reshape(1, -1))

    # Use the loaded model to make predictions
    prediction = model.predict(input_data)

    # Determine the class (0 or 1) based on the prediction
    loan_approval_class = 1 if prediction[0, 0] >= 0.5 else 0

    # Map the class to "not approved" or "approved"
    approval_status = "approved" if loan_approval_class == 1 else "not approved"

    # Display the loan approval status
    st.write(f"Loan Approval Status: {approval_status}")
