import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Function to load data
def load_data(csv_file):
    return pd.read_csv(csv_file)

# Preprocess the data
def preprocess_data(data):
    numeric_features = ['Control-Number', 'Financial-Institution-Number']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)])
    preprocessed_data = preprocessor.fit_transform(data)
    return preprocessed_data

# Train the model
def train_model(data):
    X = data.drop(columns=['Fraud'])  # Assuming 'Fraud' is the target variable
    y = data['Fraud']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Load data
csv_file_path = st.file_uploader("Upload CSV", type=["csv"])
if csv_file_path is not None:
    data = load_data(csv_file_path)
    st.write("Data loaded successfully!")
else:
    st.write("Please upload a CSV file.")

# Preprocess data and train model
if st.button("Train Model"):
    if 'data' in locals():
        preprocessed_data = preprocess_data(data)
        model = train_model(preprocessed_data)
        st.write("Model trained successfully!")

# Input fields for prediction
if 'model' in locals():
    st.title("Fraud Detection App")
    reference_number = st.text_input("Reference Number:")
    control_number = st.text_input("Control Number:")
    financial_institution_number = st.text_input("Financial Institution Number:")
    deposit_business_date = st.date_input("Deposit Business Date:", value=datetime.today())

    # Button to trigger prediction
    if st.button("Predict Fraud"):
        user_input = pd.DataFrame({
            'Control-Number': [control_number],
            'Financial-Institution-Number': [financial_institution_number],
        })
        preprocessed_input = preprocess_data(user_input)
        prediction = model.predict(preprocessed_input)
        if prediction[0] == 1:
            st.write("Fraudulent transaction detected!")
        else:
            st.write("No fraud detected.")
