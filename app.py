import streamlit as st
    import gdown
    import os
    import tensorflow as tf
    from PIL import Image
    import numpy as np

    MODEL_PATH = 'plant_model.h5'
    FILE_ID = '1pX1SXx2YHQQHkbCZhmp4nn_CYkk2hETu'
    URL = f'https://drive.google.com/uc?id={FILE_ID}'

    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... this takes 1 min first time")
        gdown.download(URL, MODEL_PATH, quiet=False)
        
    model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Pepper__bell___Bacterial_spot', ...]  
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

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
