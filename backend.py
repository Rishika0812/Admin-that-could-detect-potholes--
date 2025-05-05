# backend.py
import pandas as pd
import cv2
import numpy as np
import requests
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import gc
from dotenv import load_dotenv
from ultralytics import YOLO
import streamlit as st

# Load environment variables
load_dotenv()

# MongoDB connection details
CONNECTION_URL = os.getenv("MONGODB_CONNECTION_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_COMPLAINTS = os.getenv("COLLECTION_NAME")
COLLECTION_PREDICTIONS = os.getenv("COLLECTION_NAME_1")
CSV_FILE_PATH = "admin_credentials.csv"  # CSV file storing user credentials
MODEL_PATH = "yolov8n_pothole_finetuned_20241230_175522_Multiclass_Annotation2.pt"  # Path to YOLO model

# Load YOLO model
model = YOLO(MODEL_PATH)
model.to('cpu')  # Force using CPU

def get_mongo_client():
    try:
        client = MongoClient(CONNECTION_URL)
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None
    finally:
        gc.collect()

def authenticate_user(username, password):
    df = pd.read_csv(CSV_FILE_PATH)
    if ((df['username'] == username) & (df['password'] == password)).any():
        return True
    return False

def fetch_complaints():
    client = get_mongo_client()
    if not client:
        return []
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_COMPLAINTS]
        complaints = list(collection.find())
        return complaints
    except Exception as e:
        print(f"Error retrieving complaints: {e}")
        return []
    finally:
        client.close()
        gc.collect()

def fetch_predictions():
    client = get_mongo_client()
    if not client:
        return []
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_PREDICTIONS]
        predictions = list(collection.find())
        return predictions
    except Exception as e:
        print(f"Error retrieving predictions: {e}")
        return []
    finally:
        client.close()
        gc.collect()

def login_ui():
    st.title("Urban Snap Admin Panel 🏢")
    st.subheader("Admin Login 🔐")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.success("Login Successful ✅")
            else:
                st.error("Invalid Credentials ❌")


def get_image_url(document_id):
    client = get_mongo_client()
    if not client:
        return None
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_COMPLAINTS]
        document = collection.find_one({"_id": ObjectId(document_id)})
        if document and "image" in document:
            return document["image"]
        else:
            print("No image URL found for the given ID.")
            return None
    except Exception as e:
        print(f"Error retrieving image URL: {e}")
        return None
    finally:
        client.close()
        gc.collect()

def download_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        else:
            print("Failed to download image.")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def detect_potholes(image):
    model.to('cpu')  # Ensure model runs on CPU
    results = model.predict(image)
    
    # Initialize variables for number of detections and average confidence
    num_detections = 0
    avg_confidence = 0
    model_output = []
    
    if results:  # Ensure results are not empty
        if results[0].boxes:  # Ensure boxes are present
            num_detections = len(results[0].boxes)
            avg_confidence = np.mean([box.conf[0].item() for box in results[0].boxes]) if results[0].boxes else 0
            model_output = [f"{results[0].names[int(box.cls[0])]}: {box.conf[0]:.2f}" for box in results[0].boxes]
        
        orig_img = results[0].orig_img  # Original image in RGB format
        orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            
            # Draw bounding box and label
            cv2.rectangle(orig_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(orig_img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        orig_img_bgr = None  # No results, return None image

    return orig_img_bgr, num_detections, avg_confidence, model_output
