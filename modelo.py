import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st

MODEL_PATH = "modelo/modelo_pragmatic.keras"

@st.cache_resource
def carregar_modelo():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

def prever_jogo_em_frame(modelo, frame_path):
    if modelo is None:
        return None
    try:
        img = image.load_img(frame_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        prob = modelo.predict(x)[0][0]
        return "pragmaticplay" if prob >= 0.5 else None
    except Exception as e:
        print(f"Erro ao prever com modelo ML: {e}")
        return None
