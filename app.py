# ==============================================
# Monitor Twitch Global — app.py
# ==============================================

import os
import time
import subprocess
import threading
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- VARIÁVEIS DE AMBIENTE --------------------
load_dotenv()
TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

# -------------------- IMPORTA SUAS FUNÇÕES --------------------
from ml_utils import (
    prever_jogo_em_frame,
    obter_url_m3u8_twitch,
    varrer_vods_com_modelo,
    analisar_por_periodo
)
from storage import salvar_deteccao
from cluster_processor import clusterizar_streamers

# -------------------- SUA FUNÇÃO ORIGINAL --------------------
def verificar_jogo_em_live(streamer, headers):
    """
    Sua lógica real aqui — ex:
    verificar se está ao vivo, categoria, etc.
    """
    # EXEMPLO:
    # return ("Nome do Jogo", "Categoria", 123)
    return None

# -------------------- SUPORTE: OBTER ACCESS TOKEN --------------------
def obter_access_token(client_id, client_secret):
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    resp = requests.post(url, data=data, timeout=15)
    resp.raise_for_status()
    return resp.json().get("access_token")

ACCESS_TOKEN = obter_access_token(TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET)
HEADERS_TWITCH = {
    "Client-ID": TWITCH_CLIENT_ID,
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

# -------------------- SUPORTE: LISTAR STREAMERS GLOBAIS --------------------
def listar_streamers_globais(headers, idioma):
    url = "https://api.twitch.tv/helix/streams"
    params = {"first": 100, "language": idioma}
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    br, pt, outros = [], [], []
    for stream in data['data']:
        lang = stream.get('broadcaster_language', '')
        titulo = stream['title'].lower()
        user = stream['user_login']

        if idioma == "pt":
            if lang == 'pt-br' or "br" in titulo:
                br.append(user)
            elif lang == 'pt-pt':
                pt.append(user)
            else:
                outros.append(user)
        else:
            outros.append(user)
    return br, pt, outros

# -------------------- SUPORTE: VARREDURA INDIVIDUAL --------------------
def varrer_streamer(streamer):
    print(f"🔍 Varredura: {streamer}")

    res = verificar_jogo_em_live(streamer, HEADERS_TWITCH)
    if res:
        jogo, categoria, viewers = res
        print(f"🎮 {streamer} jogando {jogo} ({viewers} viewers)")
        resultados = [{
            "Streamer": streamer,
            "Jogo": jogo,
            "Categoria": categoria,
            "Viewers": viewers,
            "Data": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }]
        salvar_deteccao("lives", resultados)

# -------------------- SUPORTE: LOOP GLOBAL --------------------
def varredura_global():
    idiomas = ["en", "pt", "es", "fr", "de"]
    todos_streamers = {"br": [], "pt": [], "outros": []}

    for idioma in idiomas:
        br, pt, outros = listar_streamers_globais(HEADERS_TWITCH, idioma)
        todos_streamers["br"].extend(br)
        todos_streamers["pt"].extend(pt)
        todos_streamers["outros"].extend(outros)

    for key in todos_streamers:
        todos_streamers[key] = list(set(todos_streamers[key]))

    print(f"🌍 Resumo: BR={len(todos_streamers['br'])} | PT={len(todos_streamers['pt'])} | Outros={len(todos_streamers['outros'])}")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for grupo in todos_streamers.values():
            for streamer in grupo:
                futures.append(executor.submit(varrer_streamer, streamer))
        for f in as_completed(futures):
            f.result()

# -------------------- LOOP AUTOMÁTICO --------------------
def varredura_automatica():
    while True:
        print(f"🚀 Iniciando varredura global: {datetime.now()}")
        varredura_global()
        time.sleep(300)

t = threading.Thread(target=varredura_automatica, daemon=True)
t.start()

# -------------------- STREAMLIT INTERFACE --------------------
@st.cache_data
def carregar_dados_semanais():
    caminho = "output/dados_semana.csv"
    if not os.path.exists(caminho):
        colunas = ["Data", "Streamer", "Jogo", "Tipo", "Canal", "Viewers",
                   "Screenshot", "Título", "Duração", "Link", "Clip"]
        return pd.DataFrame(columns=colunas)

    df = pd.read_csv(caminho)
    df['Data'] = pd.to_datetime(df['Data'])
    df['Semana'] = df['Data'].dt.isocalendar().week
    df['Ano'] = df['Data'].dt.isocalendar().year
    ano, semana, _ = date.today().isocalendar()
    return df[(df['Ano'] == ano) & (df['Semana'] == semana)]

st.title("Monitor Twitch Global 🌍")
st.header("📈 Detecções da Semana")
st.dataframe(carregar_dados_semanais())
