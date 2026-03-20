import streamlit as st
import pickle
import os 
import gdown
import numpy as np

if not os.path.exists("model.pkl"):
    url = "https://drive.google.com/file/d/18fQ3pGilBp47D6SauiLuQ-8Ub4W2wM0I/view?usp=drive_link"
    gdown.download(url, "model.pkl", quiet=False)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Fraud Detection System")

features = []
for i in range(30):  # dataset has 30 features
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