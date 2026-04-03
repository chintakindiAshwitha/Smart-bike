import streamlit as st
import pickle

# Load models
bike_model = pickle.load(open("bike_model.pkl", "rb"))
traffic_model = pickle.load(open("traffic.pkl", "rb"))

# Title
st.title("🚲 Smart Bike Usage Predictor")

# Inputs
hour = st.selectbox("Select Hour", list(range(24)))

temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)

weather_option = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])

# Convert weather to numeric
weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}
weather = weather_map[weather_option]

# Predict button
if st.button("Predict"):

    # Step 1: Predict Traffic Automatically
    predicted_traffic = traffic_model.predict([[hour, weather]])[0]

    traffic_labels = ["Low", "Medium", "High"]

    st.info(f"🚦 Predicted Traffic: {traffic_labels[predicted_traffic]}")

    # Step 2: Predict Bike Usage
    prediction = bike_model.predict([[hour, temp, weather, predicted_traffic]])

    # Output
    if prediction[0] == 1:
        st.success("✅ Good time to rent a bike")
    else:
        st.error("❌ Avoid renting now")