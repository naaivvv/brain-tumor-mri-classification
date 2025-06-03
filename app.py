import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import os

# --- Page Configuration ---
st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="centered")

# --- Define Class Mappings ---
class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
inv_class_mappings = {v: k for k, v in class_mappings.items()}
class_emojis = {
    'Glioma': "🧠🔴",
    'Meninigioma': "🧠🟡",
    'Notumor': "✅🧠",
    'Pituitary': "🧠🟣"
}
class_colors = {
    'Glioma': "#FF4B4B",
    'Meninigioma': "#FFD700",
    'Notumor': "#4CAF50",
    'Pituitary': "#9370DB"
}
image_dim = (168, 168)

# --- Load Model ---
model_path = 'model.keras'

@st.cache_resource
def load_brain_tumor_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model file exists.")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_brain_tumor_model(model_path)
if model is None:
    st.stop()

# --- Image Preprocessing ---
def load_and_preprocess_image(uploaded_file, image_shape=(168, 168)):
    try:
        img = image.load_img(uploaded_file, target_size=image_shape, color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- App Title ---
st.title("🧠 Brain MRI Tumor Classifier")
st.markdown("Upload a Brain MRI image to classify the type of tumor or verify if it's healthy.")

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼️ Uploaded Image", use_container_width=True)
    img_array = load_and_preprocess_image(uploaded_file, image_shape=image_dim)

    if img_array is not None:
        st.subheader("🔎 Classification Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for percent in range(0, 101, 10):
            time.sleep(0.05)
            progress_bar.progress(percent)
            status_text.text(f"Processing: {percent}%")

        predictions = model.predict(img_array)
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        predicted_class = inv_class_mappings[predicted_label_index]
        confidence_scores = predictions[0]

        # --- Prediction Output ---
        st.markdown(f"""
        <div style='text-align: center; font-size: 1.5em; font-weight: bold; color: {class_colors[predicted_class]};'>
            Prediction: {class_emojis[predicted_class]} <br> {predicted_class}
        </div>
        """, unsafe_allow_html=True)

        # --- Confidence Scores ---
        st.subheader("📊 Confidence Scores")
        for class_name, score in zip(class_mappings.keys(), confidence_scores):
            bar_percent = int(score * 100)
            st.markdown(f"""
                <div style='margin-bottom: 8px;'>
                    <b>{class_emojis[class_name]} {class_name}:</b> {bar_percent:.2f}%
                    <div style='background-color: #eee; border-radius: 4px; height: 15px;'>
                        <div style='width: {bar_percent}%; background-color: {class_colors[class_name]}; height: 100%; border-radius: 4px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- Footer and Credits ---
st.markdown("<br><hr style='margin-top: 30px; margin-bottom: 30px;'><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; color: #4A4A4A; font-size: 0.9em;'>
        <p>Developed by: <b>Edwin P. Bayog Jr.</b><br>
        <i>BSCpE 3-A</i></p>
        <p style='margin-top: 5px;'>Course: <b>CpETE1 Embedded System 1 - Realtime Systems</b></p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)
