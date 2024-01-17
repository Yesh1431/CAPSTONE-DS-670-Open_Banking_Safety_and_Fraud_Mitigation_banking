def preprocess_data(data):
  preprocessed_data = preprocessor.transform(data)
  return preprocessed_data

def predict_fraud(data):
    # Preprocess the data
    processed_data = preprocess_data(data)

    # Make predictions using the loaded pipeline
    predicted_labels = loaded_pipeline.predict(processed_data)

    return predicted_labels
def main():
    st.title("Fraud Detection Streamlit App")

    # Sample input form
    st.sidebar.header("Input Features")
    control_number = st.sidebar.number_input("Control Number", min_value=0)
    financial_institution_number = st.sidebar.number_input("Financial Institution Number", min_value=0)
    financial_institution_transaction_amount = st.sidebar.number_input("Financial Institution Transaction Amount", min_value=0.0)
    transaction_amount_deviation = st.sidebar.number_input("Transaction Amount Deviation", min_value=0.0)
    transaction_day = st.sidebar.number_input("Transaction Day", min_value=1, max_value=31)
    transaction_month = st.sidebar.number_input("Transaction Month", min_value=1, max_value=12)
    transaction_year = st.sidebar.number_input("Transaction Year", min_value=2022, max_value=2025)

    # Create a DataFrame with the input features
    input_data = {
        'Control-Number': [control_number],
        'Financial-Institution-Number': [financial_institution_number],
        'Financial-Institution-Transaction-Amount': [financial_institution_transaction_amount],
        'Transaction-Amount-Deviation': [transaction_amount_deviation],
        'Transaction-Day': [transaction_day],
        'Transaction-Month': [transaction_month],
        'Transaction-Year': [transaction_year],
    }
    input_df = pd.DataFrame(input_data)

    # Predict button
    if st.sidebar.button("Predict Fraud"):
        # Get predictions
        predictions = predict_fraud(input_df)

        # Display the results
        st.subheader("Prediction Results")
        st.write("Fraudulent Transaction:", predictions[0])

if __name__ == "__main__":
    main()

