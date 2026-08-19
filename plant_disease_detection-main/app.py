import streamlit as st
import os
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_model_v2.keras')

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at {MODEL_PATH}. Make sure "
        "plant_model_v2.keras is committed to the repo alongside app.py."
    )
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Pepper__bell___Bacterial_spot', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # keep your full existing list here
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