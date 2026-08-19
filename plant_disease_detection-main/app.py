import streamlit as st
import gdown
import os
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

MODEL_PATH = 'plant_model_v2.keras'
FILE_ID = '1pX1SXx2YHQQHkbCZhmp4nn_CYkk2hETu'
URL = f'https://drive.google.com/uc?id={FILE_ID}'
MIN_VALID_SIZE_BYTES = 1_000_000


def download_model():
    st.write("Downloading model... this takes 1 min first time")
    gdown.download(URL, MODEL_PATH, quiet=False)


def is_valid_model(path):
    """Reject missing, incomplete, or non-Keras model downloads."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_VALID_SIZE_BYTES:
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    return header[:8] == b"\x89HDF\r\n\x1a\n" or header[:4] == b"PK\x03\x04"


# If a previous run left behind a bad/partial download, remove it and retry
if os.path.exists(MODEL_PATH) and not is_valid_model(MODEL_PATH):
    os.remove(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    download_model()

if not is_valid_model(MODEL_PATH):
    st.error(
        "Model download failed or returned an invalid file "
        "(this usually means the Google Drive file isn't shared as "
        "'Anyone with the link', or Drive served a warning page instead "
        "of the file). Check the Drive sharing settings and redeploy."
    )
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Pepper__bell___Bacterial_spot', ...]  
st.title("🌱 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect its condition using a CNN model.")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        resized_image = image.resize((128, 128))

        image_array = np.array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )