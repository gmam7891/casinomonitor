from datetime import datetime, timedelta
import streamlit as st
import os
import pandas as pd
import logging
import requests
import tensorflow as tf
import time
import re
from tensorflow.keras.models import load_model

from ml_utils import (
    match_template_from_image,
    capturar_frame_ffmpeg_imageio,
    prever_jogo_em_frame,
    verificar_jogo_em_live,
    varrer_url_customizada,
    varrer_vods_com_template
)

# ------------------ CONFIGURAÇÕES INICIAIS ------------------
st.set_page_config(page_title="Monitor Cassino PP", layout="wide")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
st.markdown("""
    <div style='background-color:white; padding:10px; display:flex; align-items:center;'>
        <img src='https://findfaircasinos.com/gfx/uploads/620_620_kr/716_Pragmatic%20play%20logo.png' style='height:60px; margin-right:20px;'>
        <h1 style='color:black; margin:0;'>Monitor Cassino Pragmatic Play</h1>
    </div>
    """, unsafe_allow_html=True)

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "gp762nuuoqcoxypju8c569th9wz7q5")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN", "moila7dw5ejlk3eja6ne08arw0oexs")
HEADERS_TWITCH = {'Client-ID': CLIENT_ID, 'Authorization': f'Bearer {ACCESS_TOKEN}'}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'
STREAMERS_FILE = "streamers.txt"
MODEL_PATH = "modelo/modelo_pragmatic.keras"
TEMPLATES_DIR = "templates/"

# ------------------ FUNÇÕES AUXILIARES ------------------
def carregar_streamers():
    if not os.path.exists(STREAMERS_FILE):
        with open(STREAMERS_FILE, "w") as f:
            f.write("jukes\n")
    with open(STREAMERS_FILE, "r") as f:
        return [l.strip() for l in f if l.strip()]

def extrair_segundos_da_url_vod(url):
    match = re.search(r"[?&]t=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", url)
    if not match: return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h * 3600 + m * 60 + s

def converter_duracao_para_segundos(dur_str):
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", dur_str)
    if not match: return 0
    h, m, s = match.groups(default="0")
    return int(h)*3600 + int(m)*60 + int(s)

def obter_user_id(login):
    resp = requests.get(BASE_URL_TWITCH + f"users?login={login}", headers=HEADERS_TWITCH)
    data = resp.json().get("data", [])
    return data[0]["id"] if data else None

def buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, streamers):
    todos_vods = []
    for login in streamers:
        user_id = obter_user_id(login)
        if not user_id: continue
        resp = requests.get(BASE_URL_TWITCH + f"videos?user_id={user_id}&type=archive&first=100", headers=HEADERS_TWITCH)
        vods = resp.json().get("data", [])
        for vod in vods:
            created_at = datetime.fromisoformat(vod["created_at"].replace("Z", "+00:00"))
            if dt_inicio <= created_at <= dt_fim:
                todos_vods.append({
                    "streamer": login,
                    "titulo": vod["title"],
                    "url": vod["url"],
                    "data": created_at,
                    "duração_raw": vod["duration"],
                    "duração_segundos": converter_duracao_para_segundos(vod["duration"]),
                    "id_vod": vod["id"]
                })
    return todos_vods

def formatar_datas_br(df, coluna="timestamp"):
    if coluna in df.columns:
        df[coluna] = pd.to_datetime(df[coluna]).dt.strftime("%d/%m/%Y %H:%M:%S")
    return df

def sugerir_novos_streamers(game_name="Slots"):
    sugestoes = []
    try:
        response = requests.get(BASE_URL_TWITCH + f"streams?first=100", headers=HEADERS_TWITCH)
        data = response.json().get("data", [])
        atuais = set(STREAMERS_INTERESSE)
        for stream in data:
            if game_name.lower() in stream.get("game_name", "").lower():
                login = stream.get("user_login")
                if login and login not in atuais:
                    sugestoes.append(login)
    except Exception as e:
        logging.error(f"Erro ao buscar streamers: {e}")
    return sugestoes

# ------------------ STREAMLIT UI ------------------
STREAMERS_INTERESSE = carregar_streamers()
if "modelo_ml" not in st.session_state:
    if os.path.exists(MODEL_PATH):
        st.session_state["modelo_ml"] = load_model(MODEL_PATH)
    else:
        st.warning("Modelo não encontrado. Será usada detecção por template.")

st.title("🎰 Monitor Cassino Pragmatic Play")
st.sidebar.header("🎯 Filtros")
streamers_input = st.sidebar.text_input("Streamers (separados por vírgula)")
data_inicio = st.sidebar.date_input("Data de início", value=datetime.today() - timedelta(days=7))
data_fim = st.sidebar.date_input("Data de fim", value=datetime.today())
url_custom = st.sidebar.text_input("URL personalizada (VOD .m3u8 ou com ?t=...)")
segundo_alvo = st.sidebar.number_input("Segundo para captura manual", min_value=0, max_value=99999, value=0)

# ------------------ CAPTURAR FRAME EXATO ------------------
if st.sidebar.button("🎯 Capturar frame no segundo exato") and url_custom:
    frame_path = "frame_manual.jpg"
    if capturar_frame_ffmpeg_imageio(url_custom, frame_path, skip_seconds=segundo_alvo):
        st.image(frame_path, caption=f"Frame em {segundo_alvo}s", use_column_width=True)
        resultado = prever_jogo_em_frame(frame_path, st.session_state.get("modelo_ml"))
        if resultado:
            st.success(f"🧠 Jogo detectado: {resultado}")
        else:
            st.warning("❌ Nenhum jogo detectado.")
    else:
        st.error("Erro ao capturar frame.")

# ------------------ TREINAR MODELO ------------------
if st.sidebar.button("🚀 Treinar modelo agora"):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from collections import Counter

    st.info("Iniciando treinamento...")
    datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    train_gen = datagen.flow_from_directory("dataset", target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
    val_gen = datagen.flow_from_directory("dataset", target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    class_weight = {0: 1.0, 1: 1.0}
    model.fit(train_gen, validation_data=val_gen, epochs=5, class_weight=class_weight)
    model.save(MODEL_PATH)
    st.success("Modelo treinado e salvo com sucesso!")
    st.rerun()

# ------------------ BOTÕES DE ANÁLISE ------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Verificar lives agora"):
        resultados = []
        for streamer in STREAMERS_INTERESSE:
            res = verificar_jogo_em_live(streamer, HEADERS_TWITCH, BASE_URL_TWITCH)
            if res:
                jogo, categoria = res
                resultados.append({"streamer": streamer, "jogo_detectado": jogo, "categoria": categoria, "timestamp": datetime.now()})
        st.session_state['dados_lives'] = resultados

with col2:
    if st.button("📺 Verificar VODs no período"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        vods = buscar_vods_twitch_por_periodo(dt_ini, dt_fim, STREAMERS_INTERESSE)
        st.session_state['dados_vods'] = vods

with col3:
    if st.button("🌐 Varredura na URL personalizada") and url_custom:
        tempo_inicial = extrair_segundos_da_url_vod(url_custom)
        st.info(f"Iniciando varredura a partir de {tempo_inicial}s")
        inicio = time.time()
        resultado_url = varrer_url_customizada(url_custom, st, st.session_state, prever_jogo_em_frame, skip_inicial=tempo_inicial)
        duracao = time.time() - inicio
        st.success(f"✅ Varredura concluída em {duracao:.2f}s")
        st.session_state['dados_url'] = resultado_url

with col4:
    if st.button("🖼️ Varrer VODs com imagem"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        resultados = varrer_vods_com_template(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['dados_vods_template'] = resultados

# ------------------ RESULTADOS ------------------
if 'dados_url' in st.session_state:
    st.markdown("### 🎰 Resultados da VOD personalizada")
    for res in st.session_state['dados_url']:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(res["frame"], caption=f"{res['segundo']}s", use_column_width=True)
        with col2:
            st.success(f"🎯 Jogo detectado: `{res['jogo_detectado']}`")

if 'dados_vods' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_vods'])
    df = formatar_datas_br(df, "data")
    st.markdown("### 🎞️ VODs encontrados")
    st.dataframe(df)

# ------------------ SUGERIR STREAMERS ------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔎 Buscar novos streamers"):
    novos = sugerir_novos_streamers()
    if novos:
        st.success("Novos possíveis streamers:")
        for s in novos:
            st.write(f"- {s}")
    else:
        st.warning("Nenhum novo streamer encontrado.")
