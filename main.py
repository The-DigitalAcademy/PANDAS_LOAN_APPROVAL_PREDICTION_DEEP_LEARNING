import streamlit as st
import tensorflow as tf
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('final.h5')

# Sample input fields corresponding to the columns in your X_train dataset
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

# Make predictions when a button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                            residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value])

    # Standardize the input data using the same scaler used during training
    # Load the scaler (replace with the path to your scaler file)
    scaler = StandardScaler()
   #scaler = scaler.fit(X_train)  # Fit the scaler on your training data
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
