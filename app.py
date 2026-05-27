import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🏠 Home Loan Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])

gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0

input_data = np.array([[gender, married, income, loan_amount, credit_history]])

if st.button("Predict"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
