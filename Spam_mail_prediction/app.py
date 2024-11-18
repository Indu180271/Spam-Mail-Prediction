
import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer (this should match the one used during training)
vectorizer = joblib.load('vectorizer.pkl')

# Title of the app
st.title("Spam Mail Prediction")

# Input text box for user to input the email message
user_input = st.text_area("Enter the email message here", height = 200)

if st.button("Classify"):
    if user_input:
        # Transform the input using the saved vectorizer
        input_vector = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vector)

        # Map prediction to the category
        result = "Spam" if prediction[0] == 1 else "Ham"

        # Display the result
        st.write(f"The email message is classified as: {result}")
    else:
        st.write("Please enter an email message to classify")

