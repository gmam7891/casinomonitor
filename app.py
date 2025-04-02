from datetime import datetime, timedelta
import streamlit as st
import os
import pandas as pd
import logging
import requests
import tensorflow as tf
import re
import gdown
from tensorflow.keras.models import load_model

# OpenCV seguro para ambiente sem GUI
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

# Importações internas
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

# Configuração geral
st.set_page_config(page_title="Monitor Cassino PP", layout="wide")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Cabeçalho
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
HEADERS_TWITCH = {'Client-ID': CLIENT_ID, 'Authorization': f'Bearer {ACCESS_TOKEN}'}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'
STREAMERS_FILE = "streamers.txt"
MODEL_PATH = "modelo/modelo_pragmatic.keras"
MODEL_URL = "https://drive.google.com/uc?id=1i_zEMwUkTfu9L5HGNdrIs4OPCTN6Q8Zr"

# Carregar modelo
if "modelo_ml" not in st.session_state:
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Baixando modelo...")
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
# ------------------ Funções auxiliares ------------------

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
    return h * 3600 + m * 60 + int(s)

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

# ------------------ Interface Streamlit ------------------

STREAMERS_INTERESSE = carregar_streamers()

st.sidebar.header("🎯 Filtros")
data_inicio = st.sidebar.date_input("Data de início", value=datetime.today() - timedelta(days=7))
data_fim = st.sidebar.date_input("Data de fim", value=datetime.today())
url_custom = st.sidebar.text_input("URL personalizada (VOD .m3u8 ou com ?t=...)")
segundo_alvo = st.sidebar.number_input("Segundo para captura manual", min_value=0, max_value=99999, value=0)

# Captura manual
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

# Botão de treinar modelo
if st.sidebar.button("🚀 Treinar modelo agora"):
    sucesso = treinar_modelo(st)
    if sucesso:
        st.rerun()

# Botões principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Verificar lives agora"):
        resultados = []
        for streamer in STREAMERS_INTERESSE:
            res = verificar_jogo_em_live(streamer, HEADERS_TWITCH, BASE_URL_TWITCH)
            if res:
                jogo, categoria = res
                resultados.append({
                    "streamer": streamer,
                    "jogo_detectado": jogo,
                    "categoria": categoria,
                    "timestamp": datetime.now()
                })
        st.session_state['dados_lives'] = resultados

with col2:
    if st.button("📅 Verificar VODs (resumo)"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        resumo = buscar_resumo_vods(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['vods_resumo'] = resumo
        st.success(f"✅ {len(resumo)} VODs encontradas no período.")
# ------------------ Abas principais ------------------

import plotly.express as px

abas = st.tabs(["Resultados", "Ranking de Jogos", "Timeline", "Resumo de VODs"])

# Aba 1 - Resultados
with abas[0]:
    if 'dados_url' in st.session_state:
        st.markdown("### 🎰 Resultados da VOD personalizada")
        for res in st.session_state['dados_url']:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(res["frame"], caption=f"{res['segundo']}s", use_column_width=True)
            with col2:
                st.success(f"🎯 Jogo detectado: `{res['jogo_detectado']}`")

    if 'dados_vods_template' in st.session_state:
        st.markdown("### 🖼️ Resultados por Template")
        for res in st.session_state['dados_vods_template']:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(res["frame"], caption=f"{res['segundo']}s", use_column_width=True)
            with col2:
                st.write(f"🎥 Streamer: `{res['streamer']}`")
                st.write(f"🧩 Jogo detectado: `{res['jogo_detectado']}`")
                st.write(f"⏱ Tempo: {res['segundo']}s")
                st.write(f"🔗 [Ver VOD]({res['url']})")

# Aba 2 - Ranking de Jogos
with abas[1]:
    def exibir_ranking_jogos(dados):
        if not dados:
            st.info("Nenhum jogo detectado ainda.")
            return

        df = pd.DataFrame(dados)
        if 'jogo_detectado' not in df.columns:
            st.warning("⚠️ Coluna 'jogo_detectado' não encontrada.")
            return

        ranking = df['jogo_detectado'].value_counts().reset_index()
        ranking.columns = ['Jogo', 'Aparições']

        st.markdown("### 🏆 Ranking de Jogos Detectados")
        st.dataframe(ranking, use_container_width=True)

        fig = px.bar(ranking, x='Jogo', y='Aparições', text='Aparições', color='Jogo', title="Top Jogos")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    dados_para_ranking = []
    if 'dados_url' in st.session_state:
        dados_para_ranking += st.session_state['dados_url']
    if 'dados_vods_template' in st.session_state:
        dados_para_ranking += st.session_state['dados_vods_template']
    
    exibir_ranking_jogos(dados_para_ranking)

# Aba 3 - Timeline
with abas[2]:
    def exibir_timeline_jogos(dados):
        if not dados:
            st.info("Nenhum dado disponível para exibir a timeline.")
            return

        df = pd.DataFrame(dados)
        if 'segundo' not in df.columns or 'jogo_detectado' not in df.columns:
            st.warning("⚠️ Dados incompletos para a timeline.")
            return

        if 'streamer' not in df.columns:
            df['streamer'] = 'Desconhecido'

        fig = px.scatter(
            df,
            x="segundo",
            y="jogo_detectado",
            color="streamer",
            hover_data=["streamer", "segundo", "url"] if 'url' in df.columns else ["streamer", "segundo"],
            title="🕒 Timeline de Jogos Detectados na VOD",
            labels={"segundo": "Tempo (s)", "jogo_detectado": "Jogo"}
        )
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    dados_timeline = []
    if 'dados_url' in st.session_state:
        dados_timeline += st.session_state['dados_url']
    if 'dados_vods_template' in st.session_state:
        dados_timeline += st.session_state['dados_vods_template']
    if 'dados_lives' in st.session_state:
        dados_timeline += st.session_state['dados_lives']

    exibir_timeline_jogos(dados_timeline)

# Aba 4 - Resumo de VODs (sem varredura de frames)
with abas[3]:
    st.markdown("### 📂 Resumo de VODs no período selecionado")
    if 'vods_resumo' in st.session_state and st.session_state['vods_resumo']:
        df = pd.DataFrame(st.session_state['vods_resumo'])
        df["data"] = pd.to_datetime(df["data"]).dt.strftime("%d/%m/%Y %H:%M")
        df["link"] = df["url"].apply(lambda x: f"[Abrir VOD]({x})")
        df = df.drop(columns=["url"])
        st.write(df.to_markdown(index=False), unsafe_allow_html=True)
    else:
        st.info("📭 Nenhum dado carregado. Clique em **'Verificar VODs (resumo)'**.")
