"""
CIFAR-10 CNN Classifier - Streamlit App
Upload an image or pick a sample image to get a prediction with confidence.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/final_model.keras"
IMG_SIZE = 96
PREVIEW_SIZE = 512
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SAMPLE_IMAGES_DIR = "sample_images"

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="wide")


# -----------------------------
# Load model (cached so it only loads once per session)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
def prepare_image(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image into a model-ready batch of shape (1, IMG_SIZE, IMG_SIZE, 3)."""
    img = pil_image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)  # must match preprocessing used during training
    return np.expand_dims(arr, axis=0)


def predict(pil_image: Image.Image):
    batch = prepare_image(pil_image)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# -----------------------------
# UI
# -----------------------------
st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Upload your own image, or pick one of the sample images below, "
    "to see the model's predicted class and confidence."
)

st.subheader("1. Choose image(s)")

tab_upload, tab_sample = st.tabs(["Upload images", "Use sample images"])

uploaded_images = []
sample_images = []


with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload one or more JPG/PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="uploaded_files",
    )
    if uploaded_files:
        uploaded_images = []
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                img.load()
                uploaded_images.append(img.copy())
            except Exception as e:
                st.error(f"Could not read {uploaded_file.name}: {e}")
    else:
        uploaded_images = []

with tab_sample:
    if os.path.isdir(SAMPLE_IMAGES_DIR):
        sample_files = sorted(f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if sample_files:
            st.caption(f"Preview the sample images and select the ones you want to predict.")
            cols = st.columns(5)
            selected_sample_files = []

            for idx, file_name in enumerate(sample_files):
                image_path = os.path.join(SAMPLE_IMAGES_DIR, file_name)
                try:
                    preview_image = Image.open(image_path)
                    preview_image.load()
                except Exception as e:
                    st.error(f"Could not read sample image {file_name}: {e}")
                    continue

                with cols[idx % 5]:
                    preview_resized = preview_image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
                    st.image(preview_resized, caption=file_name, use_container_width=True)
                    is_selected = st.checkbox(
                        "Select",
                        key=f"sample_{file_name}",
                        value=st.session_state.get(f"sample_{file_name}", False),
                    )
                    if is_selected:
                        selected_sample_files.append((file_name, preview_image.copy()))

            sample_images = [img for _, img in selected_sample_files]
        else:
            st.info(f"No sample images found in '{SAMPLE_IMAGES_DIR}/'.")
    else:
        st.info(f"Sample images folder '{SAMPLE_IMAGES_DIR}/' not found.")

selected_images = uploaded_images + sample_images

# -----------------------------
# Prediction display
# -----------------------------

if selected_images:
    st.subheader("2. Preview & Prediction")
    st.caption(f"{len(selected_images)} image(s) selected for prediction")

    predict_button = st.button("Predict selected images", use_container_width=True, type="primary")

    if predict_button:
        results = []
        for idx, selected_image in enumerate(selected_images):
            st.divider()
            col1, col2 = st.columns([1, 1])

            with col1:
                preview_resized = selected_image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
                st.image(preview_resized, caption=f"Image {idx + 1}", use_container_width=True)

            with col2:
                with st.spinner(f"Predicting image {idx + 1}..."):
                    pred_class, confidence, all_probs = predict(selected_image)

                st.metric(label="Predicted class", value=pred_class.capitalize())
                st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
                results.append((pred_class, confidence, all_probs))

                probs_df = pd.DataFrame({
                    "class": CLASS_NAMES,
                    "probability": all_probs
                }).sort_values("probability", ascending=False).reset_index(drop=True)

                st.bar_chart(probs_df.set_index("class")["probability"], horizontal=True)

        st.session_state.last_prediction_results = results

else:
    st.info("Upload one or more images or select sample images to get predictions.")
