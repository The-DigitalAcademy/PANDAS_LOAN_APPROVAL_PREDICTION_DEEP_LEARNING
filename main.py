
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
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
    #st.title("Meet the Team")
    
    # Add team members with their pictures and descriptions
    team_members = [
        {"name": "John Doe", "position": "Data Scientist", "image": "manoko.jpeg", "description": "John is a data scientist with expertise in machine learning and data analysis."},
        # Add more team members as needed
    ]
    
    for member in team_members:
        st.write(f"## {member['name']}")
        st.image(member['image'], caption=member['name'], use_column_width=True)
        st.write(f"**Position**: {member['position']}")
        st.write(member['description'])

# Set page configuration and title
st.title("Loan Approval Prediction")

# Sidebar
with st.sidebar:
    # Add an option in the sidebar to navigate to the "Meet the Team" page
    page_selection = st.selectbox("Navigation", 
                                   ["Loan Approval Prediction", "Meet the Team"])
    if page_selection == "Meet the Team":
        meet_the_team()

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

if __name__ == '__main__':
    main()
