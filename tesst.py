import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    st.title("Loan Approval Prediction")

    # Load your trained model (replace with the path to your model file)
    model = tf.keras.models.load_model('final3.h5')

    # Load your scaler (replace with the path to your scaler file)
    scaler = joblib.load('last_scaler.pkl')

    # Sample input fields corresponding to the columns in your training data
    no_of_dependents = st.number_input("Number of Dependents", value=0)
    income_annum = st.number_input("Annual Income", value=200)
    loan_amount = st.number_input("Loan Amount", value=32)
    loan_term = st.number_input("Loan Term (in months)", value=30)
    cibil_score = st.number_input("CIBIL Score", value=744)

    # Make predictions when a button is clicked
    if st.button("Predict"):
        # Create input data array
        input_data = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score]])

        # Standardize the input data using the loaded scaler
        input_data = scaler.transform(input_data)

        # Use the loaded model to make predictions
        prediction = model.predict(input_data)

        # Print the predicted class
        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            st.write("Prediction: Rejected (Class 0)")
        else:
            st.write("Prediction: Approved (Class 1)")

if __name__ == '__main__':
    main()
