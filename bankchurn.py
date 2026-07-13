
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("Best_ANN_Tuned.keras")

scaler = joblib.load("featurescaler.pkl")

st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")

st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details to predict churn status.")

# Inputs
credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 250000.0, 50000.0)

gender = st.selectbox("Gender", ["Female", "Male"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

if geography == "France":
    geo = 0
elif geography == "Germany":
    geo = 1
else:
    geo =2

# Encoding
gender_male = 1 if gender == "Male" else 0

# Prediction
if st.button("Predict Churn"):

    data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
        gender_male,
        geo
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0][0]

    st.subheader(f"Churn Probability: {prediction:.2%}")

    if prediction > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")
