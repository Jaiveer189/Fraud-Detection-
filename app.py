import streamlit as st
import pickle
import os 
import gdown
import numpy as np

MODEL_PATH = "model.pkl"

# download model if not exists
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=18fQ3pGilBp47D6SauiLuQ-8Ub4W2wM0I"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        st.error("Model download failed. Check link or permissions.")
        st.stop()

# load model & scaler
model = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Fraud Detection System")

features = []
for i in range(30):
    val = st.number_input(f"Feature {i}")
    features.append(val)

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("Fraud Transaction 🚨")
    else:
        st.success("Legit Transaction ✅")
