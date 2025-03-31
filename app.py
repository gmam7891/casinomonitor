# -------------------------
# app.py
# -------------------------

from datetime import datetime, timedelta
import streamlit as st
import os
import pandas as pd
import logging
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model

from ml_utils import (
    match_template_from_image,
    capturar_frame_ffmpeg_imageio,
    prever_jogo_em_frame,
    verificar_jogo_em_live,
    varrer_url_customizada,
    varrer_vods_com_template
)

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
# CONFIGS
# ------------------------------
st.set_page_config(page_title="Monitor Cassino PP - Detecção", layout="wide")

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "gp762nuuoqcoxypju8c569th9wz7q5")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN", "moila7dw5ejlk3eja6ne08arw0oexs")
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

if "modelo_ml" not in st.session_state:
    st.session_state["modelo_ml"] = carregar_modelo()

# ------------------------------
# FUNÇÕES DE BUSCA DE VODs
# ------------------------------

def obter_user_id(login, headers):
    url = f"https://api.twitch.tv/helix/users?login={login}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    if data.get("data"):
        return data["data"][0]["id"]
    return None

def converter_duracao_para_segundos(dur_str):
    import re
    h, m, s = 0, 0, 0
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", dur_str)
    if match:
        h, m, s = match.groups(default="0")
    return int(h) * 3600 + int(m) * 60 + int(s)

def buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers):
    todos_vods = []
    for login in streamers:
        user_id = obter_user_id(login, headers)
        if not user_id:
            logging.warning(f"❌ User ID não encontrado para streamer: {login}")
            continue

        url = f"{base_url}videos?user_id={user_id}&type=archive&first=100"
        try:
            resp = requests.get(url, headers=headers)
            vods = resp.json().get("data", [])

            for vod in vods:
                created_at = datetime.fromisoformat(vod["created_at"].replace("Z", "+00:00"))
                if not (dt_inicio <= created_at <= dt_fim):
                    continue

                dur = converter_duracao_para_segundos(vod["duration"])

                todos_vods.append({
                    "streamer": login,
                    "titulo": vod["title"],
                    "url": vod["url"],
                    "data": created_at,
                    "duração_segundos": dur,
                    "duração_raw": vod["duration"],
                    "id_vod": vod["id"]
                })

        except Exception as e:
            logging.error(f"Erro ao buscar VODs para {login}: {e}")

    return todos_vods

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

streamers_filtrados = [s.strip().lower() for s in streamers_input.split(",") if s.strip()] if streamers_input else []

# Ações
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Verificar lives agora"):
        resultados = []
        for streamer in STREAMERS_INTERESSE:
            resultado = verificar_jogo_em_live(streamer, HEADERS_TWITCH, BASE_URL_TWITCH)
            if resultado:
                jogo, categoria = resultado
                resultados.append({
                    "streamer": streamer,
                    "jogo_detectado": jogo,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fonte": "Live",
                    "categoria": categoria
                })
        st.session_state['dados_lives'] = resultados

with col2:
    if st.button("📺 Verificar VODs no período"):
        dt_inicio = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        vod_resultados = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['dados_vods'] = vod_resultados

with col3:
    if st.button("🌐 Rodar varredura na URL personalizada") and url_custom:
        resultado_url = varrer_url_customizada(url_custom, st, st.session_state, prever_jogo_em_frame)
        st.session_state['dados_url'] = resultado_url

with col4:
    if st.button("🖼️ Varrer VODs com detecção de imagem"):
        dt_inicio = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        resultados = varrer_vods_com_template(dt_inicio, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['dados_vods_template'] = resultados

# Exibir VODs encontrados
def formatar_datas_br(df, coluna="timestamp"):
    if coluna in df.columns:
        df[coluna] = pd.to_datetime(df[coluna]).dt.strftime("%d/%m/%Y %H:%M:%S")
    return df

if 'dados_vods' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_vods'])
    df = formatar_datas_br(df, coluna="data")
    if streamers_filtrados:
        df = df[df['streamer'].str.lower().isin(streamers_filtrados)]
    st.markdown("### 🎞️ VODs encontrados no período")
    for _, row in df.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"📺 **Streamer:** `{row['streamer']}`")
            st.markdown(f"🕒 **Data:** {row['data']}")
            st.markdown(f"⏱️ **Duração:** {row['duração_raw']}")
        with col2:
            st.markdown(f"**{row['titulo']}**")
            st.markdown(f"[🔗 Assistir VOD]({row['url']})")
    st.dataframe(df, use_container_width=True)
