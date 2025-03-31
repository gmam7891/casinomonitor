# ======================= IMPORTS ==========================
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
    varrer_url_customizada,  # você pode apagar esse do ml_utils se estiver usando o novo
    varrer_vods_com_template
)

# ======================= LOGGING ==========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# ======================= CONFIGS ==========================
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

# ======================= FUNÇÕES AUXILIARES ==========================

def extrair_segundos_da_url_vod(vod_url):
    match = re.search(r"[?&]t=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", vod_url)
    if not match:
        return 0
    h = int(match.group(1)) if match.group(1) else 0
    m = int(match.group(2)) if match.group(2) else 0
    s = int(match.group(3)) if match.group(3) else 0
    return h * 3600 + m * 60 + s

def converter_duracao_para_segundos(dur_str):
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", dur_str)
    if match:
        h, m, s = match.groups(default="0")
        return int(h) * 3600 + int(m) * 60 + int(s)
    return 0

def obter_user_id(login, headers):
    url = f"https://api.twitch.tv/helix/users?login={login}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    if data.get("data"):
        return data["data"][0]["id"]
    return None

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

def varrer_url_customizada(url_m3u8, st, session_state, prever_jogo_em_frame, skip_inicial=0):
    tempo_atual = skip_inicial
    intervalo = 30
    resultados = []

    while True:
        frame_path = f"frame_{tempo_atual}.jpg"
        sucesso = capturar_frame_ffmpeg_imageio(url_m3u8, frame_path, skip_seconds=tempo_atual)

        if not sucesso:
            break

        jogo = prever_jogo_em_frame(frame_path, session_state.get("modelo_ml"))
        if jogo:
            resultados.append({
                "segundo": tempo_atual,
                "frame": frame_path,
                "jogo_detectado": jogo
            })

        tempo_atual += intervalo

    return resultados

def formatar_datas_br(df, coluna="timestamp"):
    if coluna in df.columns:
        df[coluna] = pd.to_datetime(df[coluna]).dt.strftime("%d/%m/%Y %H:%M:%S")
    return df

def sugerir_novos_streamers(game_name="Slots"):
    sugestoes = []
    try:
        response = requests.get(BASE_URL_TWITCH + f'streams?game_name={game_name}&first=100', headers=HEADERS_TWITCH)
        data = response.json().get("data", [])
        atuais = set(STREAMERS_INTERESSE)
        for stream in data:
            login = stream.get("user_login")
            if login and login not in atuais:
                sugestoes.append(login)
    except Exception as e:
        logging.error(f"Erro ao buscar novos streamers: {e}")
    return sugestoes

# ======================= INTERFACE STREAMLIT ==========================

st.markdown("""<style>body { background-color: white; color: black; }</style>""", unsafe_allow_html=True)
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
url_custom = st.sidebar.text_input("URL da VOD personalizada (com ?t=)")

streamers_filtrados = [s.strip().lower() for s in streamers_input.split(",") if s.strip()] if streamers_input else []

# Varredura personalizada
if st.sidebar.button("🌐 Rodar varredura na URL personalizada") and url_custom:
    tempo_inicial = extrair_segundos_da_url_vod(url_custom)
    st.info(f"⏱️ Iniciando varredura a partir de {tempo_inicial} segundos da VOD.")

    inicio = time.time()
    resultado_url = varrer_url_customizada(url_custom, st, st.session_state, prever_jogo_em_frame, skip_inicial=tempo_inicial)
    fim = time.time()
    duracao = fim - inicio

    st.session_state['dados_url'] = resultado_url
    st.success(f"✅ Varredura concluída em {duracao:.2f} segundos.")

# Mostrar resultados da varredura
if 'dados_url' in st.session_state:
    resultados = st.session_state['dados_url']
    if resultados:
        st.markdown("### 🎰 Resultados da VOD")
        for res in resultados:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(res["frame"], caption=f"{res['segundo']}s", use_column_width=True)
            with col2:
                st.success(f"🎯 Jogo detectado: `{res['jogo_detectado']}` no segundo `{res['segundo']}`")
    else:
        st.warning("Nenhum jogo foi detectado durante a varredura.")

# Buscar novos streamers
st.sidebar.markdown("---")
if st.sidebar.button("🔎 Buscar novos streamers"):
    novos = sugerir_novos_streamers()
    if novos:
        st.success(f"Encontrados {len(novos)} novos possíveis streamers:")
        for nome in novos:
            st.write(f"- {nome}")
    else:
        st.warning("Nenhum novo streamer encontrado no momento.")
