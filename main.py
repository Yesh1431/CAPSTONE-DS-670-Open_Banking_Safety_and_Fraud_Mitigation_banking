import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import requests

# Function to load data from Google Drive
@st.cache
def load_data(url):
    response = requests.get(url)
    data = pd.read_csv(url)
    return data

# Streamlit UI
st.title("Google Drive Data Loader")

# User input field for Google Drive shareable link
drive_link = st.text_input("Enter the shareable link to your Google Drive CSV file:")

# Button to trigger data loading and display
if st.button("Load Data"):
    if drive_link:
        try:
            # Load data from Google Drive
            data = load_data(drive_link)
            
            # Display the loaded data
            st.write(data.head())
        except Exception as e:
            st.error("Error loading data. Please check the provided link and try again.")
    else:
        st.warning("Please enter a valid shareable link to your Google Drive CSV file.")



# Preprocess the data
def preprocess_data(data):
    # Preprocessing steps, replace with your own
    # For example: imputation, scaling, encoding categorical variables
    numeric_features = ['Control-Number', 'Financial-Institution-Number']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)])
    preprocessed_data = preprocessor.fit_transform(data)
    return preprocessed_data

# Train your model
def train_model(data):
    X = data.drop(columns=['Fraud'])  # Assuming 'Fraud' is the target variable
    y = data['Fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Define prediction function
def predict_fraud(model, input_data):
    # Example prediction logic, replace with your own model
    prediction = model.predict(input_data)
    return prediction

# Load data
data_file = 'your_data.csv'  # Replace with your CSV file path
data = load_data(data_file)

# Preprocess data
preprocessed_data = preprocess_data(data)

# Train model
model = train_model(data)

# Streamlit UI
st.title("Fraud Detection App")

# User input fields
reference_number = st.text_input("Reference Number:")
control_number = st.text_input("Control Number:")
financial_institution_number = st.text_input("Financial Institution Number:")
deposit_business_date = st.date_input("Deposit Business Date:", value=datetime.today())

# Button to trigger prediction
if st.button("Predict Fraud"):
    # Convert user input into a format compatible with the model
    user_input = pd.DataFrame({
        'Control-Number': [control_number],
        'Financial-Institution-Number': [financial_institution_number],
    })
    # Preprocess user input
    preprocessed_input = preprocess_data(user_input)
    
    # Make prediction
    prediction = predict_fraud(model, preprocessed_input)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("Fraudulent transaction detected!")
    else:
        st.write("No fraud detected.")
