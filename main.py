import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load("random_forest_pipeline.joblib")

# Streamlit UI
st.title("Fraud Detection App")

# User input fields
reference_number = st.text_input("Reference Number:")
control_number = st.text_input("Control Number:")
financial_institution_number = st.text_input("Financial Institution Number:")
deposit_business_date = st.date_input("Deposit Business Date:")

# Button to trigger prediction
if st.button("Predict Fraud"):
    # Make prediction
    prediction = model.predict([[reference_number, control_number, financial_institution_number, deposit_business_date]])
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("Fraudulent transaction detected!")
    else:
        st.write("No fraud detected.")

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
