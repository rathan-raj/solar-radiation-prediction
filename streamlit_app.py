import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# App title
st.title("🔆 Solar Energy Prediction App")
st.write("Upload input data to get solar predictions!")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", data.head())

    try:
        # Make predictions
        predictions = model.predict(data)
        data["Prediction"] = predictions
        st.write("### Predictions", data)
        
        # Option to download results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv, "solar_predictions.csv", "text/csv")
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")

