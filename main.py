import streamlit as st
import tensorflow as tf
import numpy as np
import altair as alt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to load the scaler

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('final2.h5')

# Load your scaler (replace with the path to your scaler file)
scaler = joblib.load('final_scaler.pkl')

def main():
    # Sample input fields corresponding to the columns in your training data
    st.title("Loan Approval Prediction")

    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
    income_annum = st.number_input("Annual Income", min_value=0.0, max_value=1000000.0, step=1000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, max_value=1000000.0, step=1000.0)
    loan_term = st.number_input("Loan Term (in months)", min_value=1, max_value=360, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=1000, step=1)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0.0, max_value=1000000.0, step=1000.0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0.0, max_value=1000000.0, step=1000.0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0.0, max_value=1000000.0, step=1000.0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0.0, max_value=1000000.0, step=1000.0)
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

# Make sure prediction has the shape (1, num_classes)
      assert prediction.shape == (1, 2)  # Assuming 2 classes: 'Rejected (0)' and 'Approved (1)'

# Access the prediction for the first example in the batch
      prediction_proba = prediction[0]

# Define the labels for the chart
     labels = ['Rejected (0)', 'Approved (1)']

# Make sure labels and probabilities have the same length
     assert len(labels) == len(prediction_proba)


        # Create a DataFrame with 'Loan Status' and 'Probability' columns
        chart_data = pd.DataFrame({'Loan Status': labels, 'Probability': prediction_proba})

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
