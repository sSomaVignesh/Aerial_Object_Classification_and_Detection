import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# ---------------------------------------------------------
# Google Drive Model Download
# ---------------------------------------------------------
MODEL_PATH = "best_custom_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/file/d/1u7TP6tVNHRhY9gb4mlRxtY-tTBO6eMBq/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
IMG_SIZE = 224

# ---------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------
def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_image(image):
    x = preprocess_image(image)
    prob = model.predict(x)[0][0]

    label = "Drone" if prob >= 0.5 else "Bird"
    confidence = prob if prob >= 0.5 else (1 - prob)

    return label, float(confidence)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Aerial Object Classifier", layout="centered")

st.title("🛩️ Aerial Object Classifier (Bird vs Drone)")
st.write("Upload an image and the model will classify it as **Bird** or **Drone**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        label, confidence = predict_image(img)

    st.success(f"### Prediction: **{label}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

st.write("---")
st.caption("Aerial Object Classification App — Powered by TensorFlow + Streamlit + Google Drive")
