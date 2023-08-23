import streamlit as st
import tensorflow as tf
import numpy as np


#model = tf.keras.models.load_model("loan.h5")  # Replace with the path to your model file
model = tf.keras.models.load_model('manoko3.h5')


st.title("Loan Approval Prediction")

import streamlit as st
import numpy as np

# Sample input fields corresponding to the columns in your X_train dataset
st.title("Loan Approval Prediction")

#education=["Not Graduate", "Graduate"]
#self_employed=["No", "Yes"]
# Fit and transform categorical variables
#education_encoded = education_encoder.fit_transform([education])
#self_employed_encoded = self_employed_encoder.fit_transform([self_employed])


no_of_dependents =st.number_input("Number of Dependents:")
#education = st.selectbox("Education", ["Not Graduate", "Graduate"])
#self_employed = st.selectbox("Self Employed", ["No", "Yes"])
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
    # Prepare the input data for prediction (ensure it matches your model's input format)
    input_data = np.array([no_of_dependents, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value])
 

    # Preprocess input_data if needed (e.g., scaling, feature engineering)
    input_data = input_data.reshape(1, -1)

    # Use the loaded model to make predictions
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Loan Approval Probability: {prediction[0, 0]:.2%}")
