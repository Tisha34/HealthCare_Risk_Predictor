import streamlit as st
import pickle
import numpy as np

# loading model
model = pickle.load(open('rfc_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Health Risk Prediction System")

st.write("Enter patient details below")

# input fields
age = st.number_input("Age")
bmi = st.number_input("BMI")
bp = st.number_input("Blood Pressure")
chol = st.number_input("Cholesterol")
glucose = st.number_input("Glucose Level")
heart = st.number_input("Heart Rate")
sleep = st.number_input("Sleep Hours")
exercise = st.number_input("Exercise Hours")
water = st.number_input("Water Intake")
stress = st.number_input("Stress Level")

# prediction button
if st.button("Predict"):

    features = np.array([[
        age,
        bmi,
        bp,
        chol,
        glucose,
        heart,
        sleep,
        exercise,
        water,
        stress
    ]])
    

    features = scaler.transform(features)
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("High Health Risk Detected")
    else:
        st.success("Low Health Risk")