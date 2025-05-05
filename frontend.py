import streamlit as st
import pandas as pd
import numpy as np
import cv2
import backend

# Streamlit UI
st.set_page_config(page_title="Urban Snap Admin Panel", page_icon="🏢", layout="wide")
st.title("Urban Snap Admin Panel 🏢")
st.subheader("Monitor Complaints and Object Detection")

# Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.sidebar.header("Admin Login 🔐")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if backend.authenticate_user(username, password):
            st.session_state.authenticated = True
            st.sidebar.success("Login Successful ✅")
        else:
            st.sidebar.error("Invalid Credentials ❌")
else:
    st.sidebar.success("Logged in as Admin ✅")
    complaints = backend.fetch_complaints()
    predictions = backend.fetch_predictions()
    
    # Display complaints
    st.subheader("📢 Complaints Overview")
    if complaints:
        df_complaints = pd.DataFrame(complaints)
        df_complaints = df_complaints.drop(columns=['_id'])
        st.dataframe(df_complaints)
    else:
        st.write("No complaints found.")
    
    # Display predictions
    st.subheader("📸 Object Detection Results")
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        df_predictions = df_predictions.drop(columns=['_id'])
        st.dataframe(df_predictions)
    else:
        st.write("No object detection data found.")
    
    # Object detection test
    st.subheader("🛠️ Test Object Detection")
    uploaded_file = st.file_uploader("Upload an image for pothole detection", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        detected_image, num_detections, avg_confidence, model_output = backend.detect_potholes(image)
        
        if detected_image is not None:
            st.image(detected_image, caption=f"Detected Potholes: {num_detections} potholes detected | Avg. Confidence: {avg_confidence:.2f}", channels="BGR")
        else:
            st.write("No potholes detected in the image.")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
