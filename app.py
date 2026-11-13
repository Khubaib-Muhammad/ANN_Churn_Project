import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter a customer's details and click **Predict** to estimate churn probability.")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

with st.form("customer_form"):
    st.subheader("Customer information")

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=38, step=1)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
    with col2:
        balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Column names and order must match the training data exactly.
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary
    }])

    customer_processed = preprocessor.transform(customer)
    probability = float(model.predict(customer_processed, verbose=0)[0][0])

    st.metric("Estimated churn probability", f"{probability * 100:.2f}%")
    if probability >= 0.5:
        st.error("Verdict: This customer is likely to LEAVE.")
    else:
        st.success("Verdict: This customer is likely to STAY.")

    st.caption("This is a model estimate, not a guarantee of a customer's future behaviour.")
