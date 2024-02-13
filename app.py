from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask("Fraud Detection")

# Load the trained Random Forest pipeline
with open("random_forest_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    reference_number = request.form['reference_number']
    control_number = request.form['control_number']
    financial_institution_number = request.form['financial_institution_number']
    deposit_business_date = request.form['deposit_business_date']

    # Create a DataFrame from the extracted data
    data = pd.DataFrame({
        'Reference Number': [reference_number],
        'Control-Number': [control_number],
        'Financial-Institution-Number': [financial_institution_number],
        'Deposit-Business-Date': [deposit_business_date]
        # Add other columns as needed
    })

    # Make predictions using the loaded pipeline
    prediction = pipeline.predict(data)

    # Convert prediction to human-readable format
    prediction_label = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'

    return jsonify({'prediction': prediction_label})

if __name__ == "__main__":
    app.run()

