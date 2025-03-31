from datetime import datetime, timedelta
import streamlit as st
import os
import pandas as pd
import logging
from tensorflow.keras.models import load_model

# ------------------------------
# LOGGING CONFIG
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# ------------------------------
# IMPORTS DE FUNÇÕES
# ------------------------------
from ml_utils import (
    match_template_from_image,
    capturar_frame_ffmpeg_imageio,
    prever_jogo_em_frame
)

# ------------------------------
# CONFIGURAÇÕES INICIAIS
# ------------------------------
st.set_page_config(page_title="Monitor Cassino PP - Detecção", layout="wide")

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "seu_client_id_aqui")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN", "seu_token_aqui")
HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'
STREAMERS_FILE = "streamers.txt"
TEMPLATES_DIR = "templates/"
MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_pragmatic.keras")

if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

def carregar_streamers():
    if not os.path.exists(STREAMERS_FILE):
        with open(STREAMERS_FILE, "w", encoding="utf-8") as f:
            f.write("jukes\n")
    with open(STREAMERS_FILE, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]

STREAMERS_INTERESSE = carregar_streamers()

@st.cache_resource
def carregar_modelo():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.warning("Modelo de ML ainda não treinado. Usando detecção por template.", icon="⚠️")
        return None

if "modelo_ml" not in st.session_state and os.path.exists(MODEL_PATH):
    st.session_state["modelo_ml"] = load_model(MODEL_PATH)

# ------------------------------
# INTERFACE STREAMLIT
# ------------------------------
st.markdown("""
    <style>
        body { background-color: white; color: black; }
        .stApp { background-color: white; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='background-color:white; padding:10px; display:flex; align-items:center;'>
        <img src='https://findfaircasinos.com/gfx/uploads/620_620_kr/716_Pragmatic%20play%20logo.png' style='height:60px; margin-right:20px;'>
        <h1 style='color:black; margin:0;'>Monitor Cassino Pragmatic Play</h1>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.subheader("🎯 Filtros")
streamers_input = st.sidebar.text_input("Streamers (separados por vírgula)")
data_inicio = st.sidebar.date_input("Data de início", value=datetime.today() - timedelta(days=7))
data_fim = st.sidebar.date_input("Data de fim", value=datetime.today())
url_custom = st.sidebar.text_input("URL .m3u8 personalizada")

# TESTE DE SEGUNDO EXATO
st.sidebar.subheader("⏱️ Testar segundo exato")
segundo_alvo = st.sidebar.number_input("Segundo do vídeo", min_value=0, max_value=100000, value=0, step=1)

if st.sidebar.button("🎯 Capturar frame no tempo exato") and url_custom:
    st.markdown("### 🎯 Teste de frame em tempo específico")
    frame_path = "frame_exato.jpg"
    if capturar_frame_ffmpeg_imageio(url_custom, frame_path, skip_seconds=segundo_alvo):
        modelo = st.session_state.get("modelo_ml")
        jogo = prever_jogo_em_frame(frame_path, modelo)
        st.image(frame_path, caption=f"🕒 Frame em {segundo_alvo} segundos", use_column_width=True)
        if jogo:
            st.success(f"🎰 Jogo detectado: {jogo}")
        else:
            st.warning("❌ Nenhum jogo detectado nesse ponto.")
    else:
        st.error("⚠️ Não foi possível capturar o frame. Verifique a URL.")

# ------------------------------
# TREINAR MODELO (Atualizado com MobileNetV2 + class_weight)
# ------------------------------
if st.sidebar.button("🚀 Treinar modelo agora"):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from collections import Counter

    st.sidebar.write("🔧 Iniciando treinamento com MobileNetV2...")

    dataset_dir = "dataset"
    img_height, img_width = 224, 224
    batch_size = 32

    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    counter = Counter(train_gen.classes)
    total = float(sum(counter.values()))
    class_weight = {cls: total / count for cls, count in counter.items()}

    model.fit(train_gen, validation_data=val_gen, epochs=5, class_weight=class_weight)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(MODEL_PATH)

    if os.path.exists(MODEL_PATH):
        st.sidebar.success("✅ Modelo treinado e salvo com sucesso!")
        st.sidebar.write(f"📁 Caminho: {MODEL_PATH}")
        st.rerun()
    else:
        st.sidebar.error("❌ Erro ao salvar modelo.")
