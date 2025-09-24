# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model
model = joblib.load('model/churn_model.pkl')

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📉 Telco Customer Churn Predictor")

st.markdown("Fill in the customer details to predict churn probability.")

# --- Input Form ---
gender = st.selectbox("Gender", ['Female', 'Male'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ['No', 'Yes'])
Dependents = st.selectbox("Has Dependents", ['No', 'Yes'])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ['No', 'Yes'])
MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes'])
OnlineBackup = st.selectbox("Online Backup", ['No', 'Yes'])
DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes'])
TechSupport = st.selectbox("Tech Support", ['No', 'Yes'])
StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes'])
StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['No', 'Yes'])
PaymentMethod = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check',
    'Bank transfer (automatic)', 'Credit card (automatic)'
])
MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

# --- Encoders (same as used during training) ---
encoder = {
    'Yes': 1, 'No': 0,
    'Female': 0, 'Male': 1,
    'DSL': 0, 'Fiber optic': 1, 'No': 2,
    'Month-to-month': 0, 'One year': 1, 'Two year': 2,
    'Electronic check': 0, 'Mailed check': 1,
    'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
}

if st.button("🚀 Predict Churn Probability"):
    # Create input DataFrame
    input_data = {
        'gender': encoder[gender],
        'SeniorCitizen': SeniorCitizen,
        'Partner': encoder[Partner],
        'Dependents': encoder[Dependents],
        'tenure': tenure,
        'PhoneService': encoder[PhoneService],
        'MultipleLines': encoder[MultipleLines],
        'InternetService': encoder[InternetService],
        'OnlineSecurity': encoder[OnlineSecurity],
        'OnlineBackup': encoder[OnlineBackup],
        'DeviceProtection': encoder[DeviceProtection],
        'TechSupport': encoder[TechSupport],
        'StreamingTV': encoder[StreamingTV],
        'StreamingMovies': encoder[StreamingMovies],
        'Contract': encoder[Contract],
        'PaperlessBilling': encoder[PaperlessBilling],
        'PaymentMethod': encoder[PaymentMethod],
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_data])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Churn Probability: **{prob * 100:.2f}%**")

    # Gauge plot
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prob > 0.5 else "green"},
            'steps': [
                {'range': [0, 40], 'color': 'lightgreen'},
                {'range': [40, 70], 'color': 'orange'},
                {'range': [70, 100], 'color': 'tomato'}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

