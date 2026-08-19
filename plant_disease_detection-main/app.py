import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

st.title("🌿 Plant Disease Detector")

@st.cache_resource
def load_model():
    try:
        model = torch.load('models/plant_model.pth', map_location='cpu') # CHANGE PATH
        model.eval()
        return model
    except:
        st.error("Model file not found or corrupted. Upload model.pth to /models folder")
        st.stop()

model = load_model()

# CLASSES - CHANGE THESE TO YOUR DISEASE CLASSES
CLASSES = ['Healthy', 'Powdery Mildew', 'Leaf Spot', 'Rust'] 

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        out = model(img_t)
        pred = CLASSES[torch.argmax(out)]
    
    st.success(f"Prediction: {pred}")