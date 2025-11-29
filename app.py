# =====================================================
#  Content Monetization Modeler — Streamlit App
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Content Monetization Modeler", layout="centered")
st.title(" YouTube Ad Revenue Prediction App")

# =====================================================
# Load the trained model
# =====================================================
MODEL_PATH = "best_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(" Model file not found! Please run train.py first to generate 'best_model.joblib'.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success(" Model loaded successfully!")

# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header(" Enter Video Details:")

views = st.sidebar.number_input("Total Views", min_value=0, value=10000, step=100)
likes = st.sidebar.number_input("Total Likes", min_value=0, value=500, step=10)
comments = st.sidebar.number_input("Total Comments", min_value=0, value=50, step=5)
watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=0.0, value=3000.0, step=50.0)
video_length_minutes = st.sidebar.number_input("Video Length (minutes)", min_value=0.1, value=10.0, step=0.5)
subscribers = st.sidebar.number_input("Subscribers", min_value=0, value=100000, step=1000)
category = st.sidebar.selectbox("Category", ["Education", "Entertainment", "Gaming", "Technology", "Other"])
device = st.sidebar.selectbox("Device", ["Mobile", "Desktop", "Tablet", "Other"])
country = st.sidebar.selectbox("Country", ["US", "IN", "GB", "CA", "Other"])

upload_month = st.sidebar.slider("Upload Month", 1, 12, 6)
upload_dayofweek = st.sidebar.slider("Upload Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

# =====================================================
# Prepare input features
# =====================================================
input_data = pd.DataFrame([{
    "views": views,
    "likes": likes,
    "comments": comments,
    "watch_time_minutes": watch_time_minutes,
    "video_length_minutes": video_length_minutes,
    "subscribers": subscribers,
    "category": category,
    "device": device,
    "country": country,
    "upload_month": upload_month,
    "upload_dayofweek": upload_dayofweek
}])

# =====================================================
# Feature Engineering (same as training)
# =====================================================
# Engagement Rate
input_data["engagement_rate"] = (input_data["likes"] + input_data["comments"]) / input_data["views"].replace(0, np.nan)
input_data["engagement_rate"] = input_data["engagement_rate"].fillna(0)

# Display processed input
st.write("###  Processed Input Data")
st.dataframe(input_data)

# =====================================================
# Prediction
# =====================================================
if st.button(" Predict Ad Revenue"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f" Estimated Ad Revenue: **${prediction:,.2f} USD**")
    except Exception as e:
        st.error(f" Prediction failed: {e}")

st.markdown("---")
st.caption("Developed by **PRAJITH VISHWA** | Project: Content Monetization Modeler")
