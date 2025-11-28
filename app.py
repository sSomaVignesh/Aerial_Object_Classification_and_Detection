import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import tflite_runtime.interpreter as tflite

# ---------------------------------------------------------
# Google Drive TFLite Model Download
# ---------------------------------------------------------
MODEL_PATH = "best_custom_model.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/file/d/1T-V6J_X0gH3qjGaIMngNGcgKUhpzruLa/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    model_path = download_model()
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
IMG_SIZE = 224

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_image(image):
    img_array = preprocess_image(image)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "Drone" if prediction >= 0.5 else "Bird"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)

    return label, float(confidence)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Aerial Object Classifier", layout="centered")

st.title("🛩️ Aerial Object Classification (Bird vs Drone)")
st.write("Upload an image and the model will classify it as **Bird** or **Drone**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        label, confidence = predict_image(img)

    st.success(f"### Prediction: **{label}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

st.write("---")
st.caption("Aerial Object Classification App — Powered by TFLite + Streamlit + Google Drive")
