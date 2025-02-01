
import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gzip
import pickle

title_html = """
    <style>
        .main-title { font-size: 36px; font-weight: bold; color: #2E86C1; text-align: center; }
        .description { font-size: 18px; color: #444; text-align: center; margin-bottom: 20px; }
        .footer { font-size: 14px; color: #777; text-align: center; margin-top: 50px; }
        .highlight { color: #E74C3C; font-weight: bold; }
    </style>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Crear un directorio para guardar imágenes si no existe
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_model():
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image_array = img_to_array(image) / 255.0
    return image_array.reshape(1, -1)

st.markdown('<div class="main-title">Clasificación de Dígitos MNIST</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Sube una imagen y la clasificaremos usando un <span class="highlight">KNeighborsClassifier</span> preentrenado.</div>',
    unsafe_allow_html=True
)

st.markdown("### Hiperparámetros del Modelo")
st.write("""
- **Modelo:** KNeighborsClassifier
- **n_neighbors:** 4 (número de vecinos a considerar)
- **p:** 3 (parámetro de distancia de Minkowski)
- **Escalador:** Ninguno (sin normalización adicional)
""")

uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
if uploaded_file:
    st.subheader("Vista previa de la imagen subida")
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen original", use_container_width=True)
    with col2:
        st.image(preprocessed_image.reshape(28, 28), caption="Imagen preprocesada", use_container_width=True)
    
    file_path = save_image(uploaded_file)
    st.success(f"Imagen guardada en: `{file_path}`")

    mnist_classes = {i: str(i) for i in range(10)}
    if st.button("Clasificar imagen"):
        with st.spinner("Cargando modelo y clasificando..."):
            model = load_model()
            prediction = model.predict(preprocessed_image)
            predicted_class = mnist_classes[prediction[0]]
            st.success(f"La imagen fue clasificada como: **{predicted_class}**")

st.markdown('<div class="footer">Página creada por <b>Yulisa Ortiz Giraldo</b> | © 2025</div>', unsafe_allow_html=True)
