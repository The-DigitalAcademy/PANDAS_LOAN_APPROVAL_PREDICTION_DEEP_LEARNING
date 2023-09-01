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

# Resize all images to a consistent size
def resize_image(image_path, size=(150, 150)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image

# Add a CSS style to set the background image for the entire application
st.markdown(
    """
    <style>
    body {
        background-image: url('https://github.com/The-DigitalAcademy/PANDAS_LOAN_APPROVAL_PREDICTION_DEEP_LEARNING/blob/main/business.jpeg'); /* Replace 'background.jpeg' with your image file path */
        background-size: cover;
    }
    .team-member {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .team-member img {
        width: 150px; /* Set the desired width for the images */
        height: 150px;
        margin-bottom: 10px;
        border-radius: 50%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define a function to display the "Meet the Team" page
def meet_the_team():
    st.title("Meet the Team")
    
    # Add team members with their pictures and descriptions
    team_members = [
        {"name": "Sibongile Mokoena", "position": "Junior Data Scientist", "image": "sbosha.jpeg", "description": "Sibongile is a data scientist with expertise in machine learning and data analysis."},
        {"name": "Manoko Langa", "position": "Web Developer", "image": "manoko.jpeg", "description": "Manoko is a web developer responsible for creating the Streamlit app."},
        {"name": "Zandile Mdiniso", "position": "Data Scientist", "image": "zand.jpeg", "description": "Similar to Manoko, Zandile is a data scientist with expertise in data analysis and machine learning."},
        {"name": "Thando Vilakazi", "position": "Business Analyst", "image": "thando.jpeg", "description": "Thando is a business analyst responsible for the valuable insights extracted in this project."},
    ]

    # Create columns for images and descriptions
    columns = st.columns(len(team_members))
    for i, member in enumerate(team_members):
        with columns[i]:
            st.image(resize_image(member['image']), caption=member['name'], use_column_width=True)
            st.write(f"**{member['name']}**")
            st.write(f"**Position**: {member['position']}")
            st.write(member['description'])

# Define a function to display the "Overview" page
def project_overview():
    
    st.title("Project Overview")
    
    st.subheader("Predicting clients whose loans are most likely to be approved.")
    
    st.image('business.jpeg')
    
    st.write("This project is aimed at predicting whether a loan application will be approved or not using deep learning.")
    
    st.write("It uses a deep learning model to predict whether a loan application is likely to be approved or not.")

    st.write("The model is trained on historical loan application data to make predictions based on various features such as annual income, loan amount, credit score, and more.")

    
    st.write("Please navigate to other pages for more details about the team and predictions.")

# Define a function to display the "Analysis" page
def data_analysis():
    st.title("Data Analysis")
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Load your EDA visuals
    image1 = Image.open('loan-amount.jpeg')
    image2 = Image.open('average-credit-score.jpeg')
    image3 = Image.open('average-annual-income.jpeg')
    
    st.image(image1, caption='Credit Score versus Loan Amount', use_column_width=True)
    st.image(image2, caption='Average Credit Score and Credit Score Status', use_column_width=True)
    st.image(image2, caption='Credit Status vs Average Annual Income', use_column_width=True)
    
    st.write("Here are some exploratory data analysis (EDA) visuals from your dataset.")
    st.write("You can add more visuals and descriptions as needed.")

# Sidebar
with st.sidebar:
    # Add options in the sidebar to navigate to different pages
    page_selection = st.selectbox("Navigation", 
                                   ["Project Overview", "Loan Approval Prediction", "Meet the Team", "Data Analysis", "Contact Us"])

# Main content
if page_selection == "Loan Approval Prediction":
    
    st.title("Loan Approval Prediction App")
    
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
elif page_selection == "Data Analysis":
    data_analysis()
elif page_selection == "Contact Us":
        st.title('Contact Us!')
        st.markdown("Have a question or want to get in touch with us? Please fill out the form below with your email "
                    "address, and we'll get back to you as soon as possible. We value your privacy and assure you "
                    "that your information will be kept confidential.")
        st.markdown("By submitting this form, you consent to receiving email communications from us regarding your "
                    "inquiry. We may use the email address you provide to respond to your message and provide any "
                    "necessary assistance or information.")
        with st.form("Email Form"):
            subject = st.text_input(label='Subject', placeholder='Please enter subject of your email')
            fullname = st.text_input(label='Full Name', placeholder='Please enter your full name')
            email = st.text_input(label='Email Address', placeholder='Please enter your email address')
            text = st.text_area(label='Email Text', placeholder='Please enter your text here')
            uploaded_file = st.file_uploader("Attachment")
            submit_res = st.form_submit_button("Send")
        st.markdown("Thank you for reaching out to us. We appreciate your interest in our loan approval web "
                    "application and look forward to connecting with you soon")
