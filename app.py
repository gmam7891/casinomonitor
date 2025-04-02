from datetime import datetime, timedelta
import streamlit as st
import os
import pandas as pd
import logging
import requests
import tensorflow as tf
import time
import re
import gdown
from tensorflow.keras.models import load_model

# ✅ Instalação segura do OpenCV em ambiente sem GUI
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
    buscar_vods_twitch_por_periodo,
    varrer_vods_simples
)


# ------------------ CONFIGURAÇÕES INICIAIS ------------------

import logging
import os
import streamlit as st

# Configuração do layout da página Streamlit
st.set_page_config(page_title="Monitor Cassino PP", layout="wide")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

# Cabeçalho com logo da Pragmatic Play e título do app
st.markdown("""
    <div style='background-color:white; padding:10px; display:flex; align-items:center;'>
        <img src='https://findfaircasinos.com/gfx/uploads/620_620_kr/716_Pragmatic%20play%20logo.png' 
             style='height:60px; margin-right:20px;'>
        <h1 style='color:black; margin:0;'>Monitor Cassino Pragmatic Play</h1>
    </div>
    """, unsafe_allow_html=True)

# Variáveis de ambiente e caminhos principais
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "gp762nuuoqcoxypju8c569th9wz7q5")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN", "moila7dw5ejlk3eja6ne08arw0oexs")

HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}

BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'
STREAMERS_FILE = "streamers.txt"
TEMPLATES_DIR = "templates/"
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
            st.warning("⚠️ Usando detecção por template.")
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state["modelo_ml"] = load_model(MODEL_PATH)
            st.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            st.warning("⚠️ Usando detecção por template.")

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
from ml_training import treinar_modelo

if st.sidebar.button("🚀 Treinar modelo agora"):
    sucesso = treinar_modelo(st)
    if sucesso:
        st.rerun()

col1, col2, col3, col4, col5 = st.columns(5)

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

    # ✅ Curva de aprendizado como "popup"
    if "curva_fig" in st.session_state:
        with st.expander("📈 Ver curva de aprendizado recente", expanded=True):
            st.pyplot(st.session_state["curva_fig"])

with col2:
    if st.button("📺 Verificar VODs no período"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        vods = buscar_vods_twitch_por_periodo(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['dados_vods'] = vods

with col3:
    if st.button("🌐 Varredura na URL personalizada") and url_custom:
        tempo_inicial = extrair_segundos_da_url_vod(url_custom)

        # 🧠 NOVO: Configuração da varredura por tempo total
        tempo_total = 10000   # tempo total em segundos que deseja analisar
        intervalo = 1      # intervalo entre capturas (em segundos)
        max_frames = tempo_total // intervalo  # calcula automaticamente

        st.info(f"📡 Iniciando varredura de {tempo_total}s a partir de {tempo_inicial}s (a cada {intervalo}s)")

        inicio = time.time()

        resultado_url = varrer_url_customizada(
            url_custom,
            st,
            st.session_state,
            prever_jogo_em_frame,
            skip_inicial=tempo_inicial,
            intervalo=intervalo,
            max_frames=max_frames
        )

        duracao = time.time() - inicio
        st.success(f"✅ Varredura concluída em {duracao:.2f}s")
        st.session_state['dados_url'] = resultado_url

with col4:
    if st.button("🖼️ Varrer VODs com imagem"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        resultados = varrer_vods_com_template(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['dados_vods_template'] = resultados

with col5:
    if st.button("🧾 Verificar VODs completas"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        resultados = varrer_vods_simples(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE)
        st.session_state['vods_completas'] = resultados
        st.success("✅ VODs completas varridas com sucesso.")

import plotly.express as px
import pandas as pd

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


abas = st.tabs(["Resultados", "Ranking de Jogos", "Timeline"])

import shutil

def varrer_e_salvar_para_treinamento(url, modelo=None, intervalo=30, max_frames=100, skip_inicial=0):
    st.markdown("### 📥 Coleta de dados para treino")

    os.makedirs("dataset/pragmatic", exist_ok=True)
    os.makedirs("dataset/outros", exist_ok=True)

    tempo = skip_inicial
    salvos = {"pragmatic": 0, "outros": 0}

    for i in range(max_frames):
        frame_path = f"frame_treino_{tempo}.jpg"
        sucesso = capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=tempo)

        if not sucesso:
            st.warning(f"❌ Falha ao capturar frame em {tempo}s")
            break

        jogo = prever_jogo_em_frame(frame_path, modelo)

        if jogo:
            destino = f"dataset/pragmatic/frame_{tempo}.jpg"
            salvos["pragmatic"] += 1
        else:
            destino = f"dataset/outros/frame_{tempo}.jpg"
            salvos["outros"] += 1

        shutil.move(frame_path, destino)
        st.write(f"✔️ Frame salvo em `{destino}`")

        tempo += intervalo

    st.success(f"✅ Coleta finalizada: {salvos['pragmatic']} frames positivos, {salvos['outros']} negativos.")

with abas[0]:
    # Aqui você mostra resultados das varreduras (frames, tabelas, etc.)
    # Ex: dados_url, dados_vods, dados_lives...
    ...

with abas[1]:
    dados_para_ranking = []
    
    if 'dados_url' in st.session_state:
        dados_para_ranking += st.session_state['dados_url']
    if 'dados_vods_template' in st.session_state:
        dados_para_ranking += st.session_state['dados_vods_template']

with abas[2]:
    dados_timeline = []
    if 'dados_url' in st.session_state:
        dados_timeline += st.session_state['dados_url']
    if 'dados_vods_template' in st.session_state:
        dados_timeline += st.session_state['dados_vods_template']
    if 'dados_lives' in st.session_state:
        dados_timeline += st.session_state['dados_lives']

    exibir_timeline_jogos(dados_timeline)


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

if 'vods_completas' in st.session_state:
    st.markdown("### 📂 Resultados da Varrredura de VODs Completas")
    for res in st.session_state['vods_completas']:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(res["frame"], caption=f"{res['segundo']}s", use_column_width=True)
        with col2:
            st.write(f"🎥 Streamer: `{res['streamer']}`")
            st.write(f"⏱ Tempo: `{res['segundo']}s`")
            st.write(f"🔗 URL: {res['url']}")

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
def exibir_ranking_jogos(dados):
    if not dados:
        st.info("Nenhum jogo detectado ainda.")
        return

    # Montar DataFrame
    df = pd.DataFrame(dados)

    # Extrair só o nome do jogo
    if 'jogo_detectado' not in df.columns:
        st.warning("⚠️ Coluna 'jogo_detectado' não encontrada nos dados.")
        return

    ranking = df['jogo_detectado'].value_counts().reset_index()
    ranking.columns = ['Jogo', 'Aparições']

    st.markdown("### 🏆 Ranking de Jogos Detectados")
    st.dataframe(ranking, use_container_width=True)

    # Gráfico de barras (opcional)
    import plotly.express as px
    fig = px.bar(ranking, x='Jogo', y='Aparições', text='Aparições', color='Jogo', title="Top Jogos")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
