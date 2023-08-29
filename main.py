import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('final3.h5')

# Load your scaler (replace with the path to your scaler file)
scaler = joblib.load('last_scaler.pkl')

# Define column names in the same order as your training data
columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']
# Add more column names as needed to match your training data

def main():
    # Sample input fields corresponding to the columns in your training data
    st.title("Loan Approval Prediction")

    # Create a DataFrame from the input variables
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = [0] * len(columns)  # Initialize with zeros, you can replace these with your desired default values

    # Create input fields for each column
    for column in columns:
        input_df[column] = st.number_input(f"{column.replace('_', ' ').title()}", value=input_df[column].values[0])

    # Make predictions when a button is clicked
    if st.button("Predict"):
        # Standardize the input data using the loaded scaler
        input_data = input_df.values  # Convert DataFrame to array
        input_data = scaler.transform(input_data)

        # Use the loaded model to make predictions
        prediction = model.predict(input_data)

        # Print the predicted class
        predicted_class = np.argmax(prediction)
        # if predicted_class == 0:
        #     st.write("Prediction: Rejected (Class 0)")
        # else:
        #     st.write("Prediction: Approved (Class 1)")

if __name__ == '__main__':
    main()
