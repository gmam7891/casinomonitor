from datetime import datetime, timedelta
import streamlit as st
import os
import pandas as pd
import logging
import requests
import re
import gdown
from tensorflow.keras.models import load_model

# Instalação segura do OpenCV
try:
    import cv2
except ImportError:
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "opencv-python-headless"])
        import cv2
    except Exception as e:
        st.error(f"❌ Falha ao instalar OpenCV automaticamente: {e}")
        st.stop()

from ml_training import treinar_modelo
from ml_utils import (
    match_template_from_image,
    capturar_frame_ffmpeg_imageio,
    prever_jogo_em_frame,
    verificar_jogo_em_live,
    varrer_url_customizada,
    varrer_vods_com_template,
    buscar_vods_twitch_por_periodo
)

# ------------------ CONFIGURAÇÕES INICIAIS ------------------

st.set_page_config(page_title="Monitor Cassino PP", layout="wide")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

st.markdown("""
    <div style='background-color:white; padding:10px; display:flex; align-items:center;'>
        <img src='https://findfaircasinos.com/gfx/uploads/620_620_kr/716_Pragmatic%20play%20logo.png' 
             style='height:60px; margin-right:20px;'>
        <h1 style='color:black; margin:0;'>Monitor Cassino Pragmatic Play</h1>
    </div>
    """, unsafe_allow_html=True)

# Variáveis
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "gp762nuuoqcoxypju8c569th9wz7q5")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN", "moila7dw5ejlk3eja6ne08arw0oexs")
HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'
STREAMERS_FILE = "streamers.txt"
MODEL_PATH = "modelo/modelo_pragmatic.keras"
MODEL_URL = "https://drive.google.com/uc?id=1i_zEMwUkTfu9L5HGNdrIs4OPCTN6Q8Zr"

if "modelo_ml" not in st.session_state:
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Baixando modelo do Google Drive...")
        os.makedirs("modelo", exist_ok=True)
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Modelo baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar modelo: {e}")
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state["modelo_ml"] = load_model(MODEL_PATH)
            st.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")

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

def formatar_datas_br(df, coluna="timestamp"):
    if coluna in df.columns:
        df[coluna] = pd.to_datetime(df[coluna]).dt.strftime("%d/%m/%Y %H:%M:%S")
    return df

def buscar_resumo_vods(dt_inicio, dt_fim, headers, base_url, streamers):
    resumo = []
    vods = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers)
    for vod in vods:
        resumo.append({
            "streamer": vod["streamer"],
            "data": vod["data"],
            "duração (min)": round(vod["duração_segundos"] / 60, 1),
            "visualizações": vod.get("view_count", "N/A"),
            "url": vod["url"]
        })
    return resumo

# ------------------ STREAMLIT UI ------------------

STREAMERS_INTERESSE = carregar_streamers()

st.sidebar.header("🎯 Filtros")
data_inicio = st.sidebar.date_input("Data de início", value=datetime.today() - timedelta(days=7))
data_fim = st.sidebar.date_input("Data de fim", value=datetime.today())

col1, col2, col3, col4 = st.columns(4)

with col2:
    if st.button("📅 Verificar VODs (resumo)"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())

        resumo = buscar_resumo_vods(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['vods_resumo'] = resumo
        st.success(f"✅ {len(resumo)} VODs encontradas no período.")

if 'vods_resumo' in st.session_state and st.session_state['vods_resumo']:
    st.markdown("### 🗓️ Resumo de VODs do período")
    df = pd.DataFrame(st.session_state['vods_resumo'])
    df["data"] = pd.to_datetime(df["data"]).dt.strftime("%d/%m/%Y %H:%M")
    df["link"] = df["url"].apply(lambda x: f"[Abrir VOD]({x})")
    df = df.drop(columns=["url"])
    st.write(df.to_markdown(index=False), unsafe_allow_html=True)
