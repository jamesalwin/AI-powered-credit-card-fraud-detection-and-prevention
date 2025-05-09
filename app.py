import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, preprocess_data
from fraud_model import train_model, evaluate_model
import numpy as np

st.set_page_config(page_title="AI-Powered Fraud Detection", layout="wide")
st.title("🔐 Guarding Transactions with AI-Powered Credit Card Fraud Detection")

df = load_data()

with st.expander("📊 Raw Data Preview"):
    st.write(df.head())

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Check class balance
class_counts = y_train.value_counts()
st.write("Class distribution in y_train:")
st.write(class_counts)

if len(class_counts) < 2:
    st.error("Training data contains only one class! Ensure your data is balanced for both classes.")
else:
    model = train_model(X_train, y_train, model_type='logistic')
    report, matrix, roc_auc = evaluate_model(model, X_test, y_test)

    st.subheader("📈 Model Performance")
    st.write(f"ROC AUC Score: **{roc_auc:.4f}**")
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax)
    st.pyplot(fig)

    st.write("Classification Report:")
    st.json(report)

    st.subheader("🚀 Real-Time Transaction Prediction")
    feature_input = []
    for col in df.columns[:-1]:
        val = st.number_input(f"Enter {col}", value=0.0)
        feature_input.append(val)

    if st.button("Predict Transaction Type"):
        input_data = scaler.transform([feature_input])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"⚠️ This transaction is likely FRAUDULENT (Probability: {proba:.2f})")
        else:
            st.success(f"✅ This transaction appears legitimate (Probability of Fraud: {proba:.2f})")
