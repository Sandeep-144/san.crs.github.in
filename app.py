import streamlit as st
import numpy as np
import pickle

# Load models and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model/scaler files: {e}")
    model = None
    sc = None
    ms = None

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

# Streamlit UI
st.set_page_config(page_title="Crop Recommendation System 🌱", layout="centered")
st.title("🌾 Crop Recommendation System")
st.markdown("Enter the required agricultural parameters below:")

with st.form("crop_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen", min_value=0.0, format="%.2f")
    with col2:
        P = st.number_input("Phosphorus", min_value=0.0, format="%.2f")
    with col3:
        K = st.number_input("Potassium", min_value=0.0, format="%.2f")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        temp = st.number_input("Temperature (°C)", format="%.2f")
    with col5:
        humidity = st.number_input("Humidity (%)", format="%.2f")
    with col6:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, format="%.2f")
    
    rainfall = st.number_input("Rainfall (mm)", format="%.2f")

    submit = st.form_submit_button("🚜 Recommend Crop")

if submit:
    if not model or not sc or not ms:
        st.error("Model or scaler files are missing.")
    else:
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop = crop_dict.get(prediction[0], "Unknown crop")
        st.success(f"✅ **Recommended Crop:** {crop}")

        with st.expander("📊 View Input Parameters"):
            st.write(f"Nitrogen: {N}")
            st.write(f"Phosphorus: {P}")
            st.write(f"Potassium: {K}")
            st.write(f"Temperature: {temp} °C")
            st.write(f"Humidity: {humidity} %")
            st.write(f"pH: {ph}")
            st.write(f"Rainfall: {rainfall} mm")

# Footer
st.markdown(
    "<hr><center>Developed by <b>Sandeep</b> &copy; 2024 | Helping farmers make better decisions</center>",
    unsafe_allow_html=True
)