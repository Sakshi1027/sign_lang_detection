import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="centered")

# --- Sidebar ---
st.sidebar.title("How to Use")
st.sidebar.markdown("""
- **Single Image**: Upload a single image of a hand sign to get the predicted letter.
- **Sequence of Frames**: Upload multiple images (frames) to get a predicted word or sentence (letters shown in a single line).
- **Model**: Trained on 26 letters (A-Z, except J & Z), plus 'del', 'nothing', 'space'.
""")
st.sidebar.info("For best results, use clear, centered images of your hand sign on a plain background.")

# --- Main Title ---
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>Sign Language Recognition 🤟</h1>", unsafe_allow_html=True)
st.write('Upload an image or a sequence of frames to detect sign language gestures. The model will predict the corresponding letter, word, or sentence.')

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sign_language_model.keras')

model = load_model()

# --- Class Labels ---
class_labels = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']] + ['del', 'nothing', 'space']

# --- Helper Functions ---
def preprocess_image(img):
    img = img.convert('L')  # Grayscale
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)  # (64, 64, 1)
    img = np.expand_dims(img, axis=0)   # (1, 64, 64, 1)
    return img

def predict(img):
    processed = preprocess_image(img)
    preds = model.predict(processed)
    idx = np.argmax(preds)
    return class_labels[idx], preds[0][idx]

# --- Tabs for Single Image and Sequence of Frames ---
tab1, tab2 = st.tabs(["Single Image", "Sequence of Frames"])

with tab1:
    st.markdown("<h3 style='color: #F97B4F;'>Upload a Single Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload a single image', type=['jpg', 'jpeg', 'png'], key='single')
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', width=250)
        label, conf = predict(img)
        st.success(f'Prediction: {label} (Confidence: {conf:.2f})')

with tab2:
    st.markdown("<h3 style='color: #4FF9A0;'>Upload a Sequence of Frames</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader('Upload multiple images (frames)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key='sequence')
    if uploaded_files:
        cols = st.columns(min(5, len(uploaded_files)))
        predicted_letters = []
        for idx, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file)
            with cols[idx % len(cols)]:
                st.image(img, width=120)
            label, conf = predict(img)
            predicted_letters.append(label)
        st.markdown("<div style='margin-top: 20px; font-size: 1.3em; color: #333;'><b>Predicted Sequence:</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; background: #e6f7ff; border-radius: 8px; font-size: 1.5em; color: #4F8BF9; letter-spacing: 0.15em;'><b>{''.join(predicted_letters)}</b></div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
---
<div style='text-align: center; color: #888;'>
    Made with ❤️ using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True) 