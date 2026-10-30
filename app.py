import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Air Quality Risk Predictor", layout="wide")

@st.cache_resource
def load_files():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_files()

st.title("☁️ Air Quality Risk Predictor Dashboard")
st.markdown("Adjust the meteorological parameters below to evaluate real-time pollution risk using our optimized Gradient Boosting model.")

col1, col2 = st.columns(2, gap="large")

with col1:
    um003 = st.slider("Particle Count (UM003)", 1000, 5000, 2680)
    humidity = st.slider("Relative Humidity (%)", 40.0, 100.0, 76.6)
    hour = st.slider("Hour of Day", 0, 23, 8)
    temp = st.slider("Temperature (°C)", 20.0, 45.0, 25.2)
    
    predict_btn = st.button("Predict Risk Level", use_container_width=True)

with col2:
    if predict_btn:
        # Create input dataframe matching training features
        input_df = pd.DataFrame(
            [[um003, humidity, hour, temp]], 
            columns=['UM003', 'Relative_Humidity', 'Hour', 'Temp']
        )
        
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Get predictions
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        
        st.markdown("**Predicted Risk Category**")
        st.info(f"**{pred}**")
        
        st.markdown("**📊 Class Confidence Probabilities**")
        
        for class_name, prob in zip(model.classes_, probs):
            st.markdown(f"**{class_name}**")
            st.progress(float(prob), text=f"{int(prob * 100)}%")