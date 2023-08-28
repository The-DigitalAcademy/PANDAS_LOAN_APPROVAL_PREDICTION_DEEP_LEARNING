import streamlit as st
import tensorflow as tf
import numpy as np
import altair as alt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to load the scaler

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('final.h5')

# Load your training data
# Replace 'your_training_data.csv' with the path to your training data CSV file
training_data = pd.read_excel('loan_approval_dataset.xlsx')

# Define your training features (X_train) and target (y_train)
X_train = training_data.drop(['loan_status'], axis=1)
y_train = training_data['loan_status']

# Load the scaler (replace with the path to your scaler file)
#scaler = joblib.load('scaler3.pkl')

def main():
    # Sample input fields corresponding to the columns in your training data
    st.title("Loan Approval Prediction")

    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
    # ...

    # Make predictions when a button is clicked
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = np.array([no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                                residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value])

        # Standardize the input data using the loaded scaler
        input_data = scaler.transform(input_data.reshape(1, -1))

        # Use the loaded model to make predictions
        prediction = model.predict(input_data)

        # Create a bar chart to visualize the prediction using Altair
        chart_data = pd.DataFrame({'Loan Status': ['Rejected (0)', 'Approved (1)'], 'Probability': prediction[0]})
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Loan Status',
            y='Probability'
        ).properties(
            width=400,
            height=300
        )
        st.altair_chart(chart)

        # Print the predicted class
        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            st.write("Prediction: Rejected (Class 0)")
        else:
            st.write("Prediction: Approved (Class 1)")

if __name__ == '__main__':
    main()
