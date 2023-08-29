import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('final4.h5')

# Load your scaler (replace with the path to your scaler file)
scaler = joblib.load('second_last_scaler.pkl')

# Define column names in the same order as your training data
columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']

def main():
    st.title("Loan Approval Prediction")

    # Create a DataFrame from the input variables
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = [0] * len(columns)

    # Create input fields for each column
    for column in columns:
        input_df[column] = st.number_input(f"{column.replace('_', ' ').title()}", value=input_df[column].values[0])

    if st.button("Predict"):
        try:
            with st.spinner("Predicting..."):
                # Standardize the input data using the loaded scaler
                input_data = input_df.values
                input_data = scaler.transform(input_data)

                # Use the loaded model to make predictions
                prediction = model.predict(input_data)

                # Display prediction result
                predicted_class = np.argmax(prediction)
                result = "likely to be approved" if prediction >= 0.5 else "not likely to be approved"

                st.success(f"Prediction: The client is {result}.")
                st.write(f"Probability of Approval: {prediction[0]:.2%}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("Clear"):
        # Clear input fields and results
        st.experimental_set_query_params()

if __name__ == '__main__':
    main()
