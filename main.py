import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
# Replace this with your dataset path
data_path = "path_to_your_dataset.csv"
df = pd.read_csv(data_path)

# Split data into features and target variable
X = df.drop(columns=["Fraud"])  # Assuming "Fraud" is the target variable
y = df["Fraud"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy}")

# Streamlit UI
st.title("Fraud Detection App")

# User input fields
reference_number = st.text_input("Reference Number:")
control_number = st.text_input("Control Number:")
financial_institution_number = st.text_input("Financial Institution Number:")
deposit_business_date = st.date_input("Deposit Business Date:")

# Button to trigger prediction
if st.button("Predict Fraud"):
    # Convert user input into a format compatible with the model
    user_input = [[reference_number, control_number, financial_institution_number, deposit_business_date]]
    
    # Make prediction
    prediction = model.predict(user_input)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("Fraudulent transaction detected!")
    else:
        st.write("No fraud detected.")
