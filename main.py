import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('fraud_detection_model.joblib')

# Define the app
st.title('Fraud Detection App')
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(rgb(173, 216, 230), rgb(25, 25, 112));
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Example data
data = {
    ' Reference Number': [39909909],
    'Control-Number': [202000000000000.0],
    'Financial-Institution-Number': [10],
    'Deposit-Business-Date': ['4/3/2023'],
    'Financial-Institution-Business-Date': ['4/3/2023'],
    'Financial-Institution-Transaction-Date-Date': ['4/3/2023'],
    'Financial-Institution-Transaction-Type-Code-': ['DEP'],
    'Financial-Institution-Transaction-Amount': [41317.17],
    'Authorization-Number': ['100030'],
    'Transaction-Amount-Deviation': [0],
    'Fraud': [0]
}

# Create DataFrame from example data
df = pd.DataFrame(data)

# Check if the target column 'Fraud' exists in the DataFrame
if 'Fraud' in df.columns:
    # Drop the target column 'Fraud'
    X_train = df.drop(columns=['Fraud'])
else:
    # If 'Fraud' column doesn't exist, use the entire DataFrame
    X_train = df.copy()

# Define the data types of features (replace with the actual data types)
column_data_types = {
    ' Reference Number': 'int64',
    'Control-Number': 'float64',
    'Financial-Institution-Number': 'int64',
    'Deposit-Business-Date': 'datetime64[ns]',
    'Financial-Institution-Business-Date': 'datetime64[ns]',
    'Financial-Institution-Transaction-Date-Date': 'datetime64[ns]',
    'Financial-Institution-Transaction-Type-Code-': 'object',
    'Financial-Institution-Transaction-Amount': 'float64',
    'Authorization-Number': 'object',
    'Transaction-Amount-Deviation': 'int64'
}

# Create input fields based on the defined data types
input_data = {}
for column, dtype in column_data_types.items():
    if dtype == 'object':
        input_value = st.text_input(column, value='Enter value')
    elif dtype == 'float64':
        input_value = st.number_input(column, value=0.0)
    elif dtype == 'int64':
        input_value = st.number_input(column, value=0)

    # Store input values in a dictionary
    input_data[column] = input_value

# Make predictions
if st.button('Predict'):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.write('Prediction: Fraudulent transaction')
    else:
        st.write('Prediction: No fraud')

