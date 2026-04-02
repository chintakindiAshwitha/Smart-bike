import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚲 Smart Bike Usage Predictor")

# Inputs
hr = st.slider("Hour (0-23)", 0, 23)
temp = st.slider("Temperature", 0.0, 1.0)
weather = st.selectbox("Weather", [1, 2, 3, 4])

# Predict
if st.button("Predict"):
    prediction = model.predict([[hr, temp, weather]])

    if prediction[0] == 1:
        st.success("✅ Good time to rent a bike")
    else:
        st.error("❌ Avoid renting now")