import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, preprocess_data
from fraud_model import train_model, evaluate_model
import numpy as np

# Set the page title and layout for Streamlit
st.set_page_config(page_title="AI-Powered Fraud Detection", layout="wide")

# Title for the web app
st.title("🔐 Guarding Transactions with AI-Powered Credit Card Fraud Detection")

# Load the data
df = load_data()

# Display raw data preview in an expandable section
with st.expander("📊 Raw Data Preview"):
    st.write(df.head())

# Preprocess the data (Splitting and scaling)
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Check for class distribution in training data (for debugging purposes)
st.write("Class distribution in y_train:")
st.write(y_train.value_counts())

# Check if both classes are present in the training data
if len(np.unique(y_train)) < 2:
    st.error("Training data contains only one class! Ensure your data is balanced for both classes.")
else:
    # Train the model
    model, _, _, _, _ = train_model(X_train, y_train, model_type='logistic')

    # Evaluate the model
    report, matrix, roc_auc = evaluate_model(model, X_test, y_test)

    # Display model performance metrics
    st.subheader("📈 Model Performance")
    st.write(f"ROC AUC Score: **{roc_auc:.4f}**")

    # Display confusion matrix
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax)
    st.pyplot(fig)

    # Display classification report
    st.write("Classification Report:")
    st.json(report)

    # Real-Time Transaction Prediction Section
    st.subheader("🚀 Real-Time Transaction Prediction")

    feature_input = []
    for col in df.columns[:-1]:  # All columns except for 'Class'
        val = st.number_input(f"Enter {col}", value=0.0)
        feature_input.append(val)

    if st.button("Predict Transaction Type"):
        input_data = scaler.transform([feature_input])  # Transform input using scaler
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of class 1 (fraud)

        if prediction == 1:
            st.error(f"⚠️ This transaction is likely FRAUDULENT (Probability: {proba:.2f})")
        else:
            st.success(f"✅ This transaction appears legitimate (Probability of Fraud: {proba:.2f})")
