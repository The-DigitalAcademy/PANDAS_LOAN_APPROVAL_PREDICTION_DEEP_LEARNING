import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('final4.h5')

# Load your scaler (replace with the path to your scaler file)
scaler = joblib.load('second_last_scaler.pkl')

# Define column names in the same order as your training data
columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']
# Add more column names as needed to match your training data

# Define a function to display the "Meet the Team" page
def meet_the_team():
    st.title("Meet the Team")
    
    # Add team members with their pictures and descriptions
    team_members = [
        {"name": "Manoko Langa", "position": "Data Scientist", "image": "manoko.jpeg", "description": "Manoko is a data scientist with expertise in machine learning and data analysis."},
        {"name": "Thando Vilakazi", "position": "Web Developer", "image": "thando.jpeg", "description": "Thando is a web developer responsible for creating the Streamlit app."},
        # Add more team members as needed
    ]
    
    # Create a container for the team members
    team_container = st.container()
    
    # Create a CSS class for horizontal layout
    st.markdown(
        """
        <style>
        .horizontal-layout {
            display: flex;
            flex-direction: row;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    for member in team_members:
        with team_container:
            st.image(member['image'], caption=member['name'], use_column_width=True)
            st.markdown(f"**{member['name']}**")
            st.write(f"**Position**: {member['position']}")
            st.write(member['description'])

# Define a function to display the "Overview" page
def project_overview():
    st.title("Project Overview")

    
    st.subheading("This project is aimed at predicting loan approval using machine learning.")

    
    st.write("It uses a deep learning model to predict whether a loan application is likely to be approved or not.")
    
    st.write("Please navigate to other pages for more details about the team and predictions.")

# Set page configuration and title
st.title("Loan Approval Prediction")

# Sidebar
with st.sidebar:
    # Add options in the sidebar to navigate to different pages
    page_selection = st.selectbox("Navigation", 
                                   ["Project Overview", "Loan Approval Prediction", "Meet the Team"])

# Main content
if page_selection == "Loan Approval Prediction":
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
        result = "likely to be approved" if prediction >= 0.5 else "not likely to be approved"
        
        st.write(f"Prediction: {prediction[0]}")
        st.write(f"The client is {result}.")
elif page_selection == "Meet the Team":
    meet_the_team()
elif page_selection == "Project Overview":
    project_overview()
