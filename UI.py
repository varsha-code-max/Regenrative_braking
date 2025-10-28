# app.py
import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="EV Regenerative Braking Dashboard", layout="centered")

# --- Load model
MODEL_PATH = "models/regen_model.pkl"
if os.path.exists(MODEL_PATH):
    model = pickle.load(MODEL_PATH)
else:
    st.error("Model not found! Please run train_model.py first.")
    st.stop()

st.title("⚡ EV Regenerative Energy Recovery Predictor")
st.markdown("### Predict how much energy is recovered using regenerative braking")

# --- Inputs
km_travelled = st.number_input("🚗 Kilometers Travelled", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
no_of_brakes = st.number_input("🛞 Number of Times Brake Applied", min_value=1, max_value=1000, value=100, step=1)

if st.button("🔋 Calculate Energy Recovery"):
    X_input = np.array([[km_travelled, no_of_brakes]])
    energy_recovered = model.predict(X_input)[0]

    # --- Estimate battery range extension (assume 1 kWh = 6 km range)
    battery_range_extended = energy_recovered * 6

    # --- Efficiency advice
    if no_of_brakes > (km_travelled * 3):
        tip = "⚠️ Too many braking events — drive smoothly to maximize energy recovery."
    elif no_of_brakes < (km_travelled * 0.5):
        tip = "✅ Great! You’re driving efficiently with fewer brakes."
    else:
        tip = "💡 Moderate braking pattern — consider coasting gently before braking."

    # --- Display results
    st.success(f"🔋 **Energy Recovered:** {energy_recovered:.2f} kWh")
    st.info(f"🔧 **Battery Range Extended:** ≈ {battery_range_extended:.2f} km")
    st.write("### 🚦 Driving Efficiency Tip:")
    st.write(tip)
