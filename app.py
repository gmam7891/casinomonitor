import sys
import os
sys.path.append(os.path.dirname(__file__))

from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import datetime
import logging
import requests
from dotenv import load_dotenv
load_dotenv()

print("TWITCH_CLIENT_ID:", os.getenv("TWITCH_CLIENT_ID"))
print("TWITCH_CLIENT_SECRET:", os.getenv("TWITCH_CLIENT_SECRET"))

import tensorflow as tf
import time
import re
import gdown
import subprocess
from tensorflow.keras.models import load_model
from storage import salvar_deteccao

from ml_utils import (
    extrair_segundos_da_url_vod,
    buscar_vods_por_streamer_e_periodo,
    analisar_por_periodo
)


# ---------------- OBTER ACCESS TOKEN DA TWITCH ----------------
def obter_access_token(client_id, client_secret):
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    try:
        resp = requests.post(url, data=data)
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        st.error("Erro ao obter access_token:")
        st.code(str(e))
        st.stop()

# ---------------- OpenCV em ambiente headless ----------------
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

# ---------------- Importar módulos internos ----------------
from ml_training import treinar_modelo
from ml_utils import (
    match_template_from_image,
    capturar_frame_ffmpeg_imageio,
    prever_jogo_em_frame,
    verificar_jogo_em_live,
    varrer_url_customizada,
    varrer_vods_com_modelo,
    buscar_vods_twitch_por_periodo,
    buscar_vods_por_streamer_e_periodo
)

# ---------------- CONFIGURAÇÃO GERAL ----------------
st.set_page_config(page_title="Monitor Cassino PP", layout="wide")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# ---------------- CABEÇALHO ----------------
st.markdown("""
<div style='background-color:white; padding:10px; display:flex; align-items:center;'>
    <img src='https://findfaircasinos.com/gfx/uploads/620_620_kr/716_Pragmatic%20play%20logo.png' 
         style='height:60px; margin-right:20px;'>
    <h1 style='color:black; margin:0;'>Monitor Cassino Pragmatic Play</h1>
</div>
""", unsafe_allow_html=True)

# ---------------- VARIÁVEIS ----------------
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")
ACCESS_TOKEN = obter_access_token(CLIENT_ID, CLIENT_SECRET)

HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'

MODEL_PATH = "modelo/modelo_pragmatic.keras"
MODEL_URL = "https://drive.google.com/uc?id=1i_zEMwUkTfu9L5HGNdrIs4OPCTN6Q8Zr"

# ---------------- FILTRO DE STREAMERS PT ----------------
def filtrar_streamers_pt(streamers):
    """Filtra a lista mantendo apenas streamers com idioma 'pt' (português)."""
    streamers_pt = []
    ignorados = []
    for s in streamers:
        try:
            url = f"{BASE_URL_TWITCH}users?login={s}"
            resp = requests.get(url, headers=HEADERS_TWITCH)
            data = resp.json().get("data", [])
            if data and data[0].get("broadcaster_language") == "pt":
                streamers_pt.append(s)
            else:
                ignorados.append(s)
        except Exception as e:
            logging.warning(f"Erro ao verificar idioma de {s}: {e}")
            ignorados.append(s)

    if ignorados:
        st.sidebar.warning("Alguns streamers foram ignorados por não estarem em PT:")
        for i in ignorados:
            st.sidebar.text(f"❌ {i}")

    return streamers_pt


# ---------------- MODELO ML ----------------
import os
from tensorflow.keras.models import load_model
import gdown

# Caminho absoluto para o modelo na pasta raiz
MODEL_DIR = os.path.join(os.path.dirname(__file__), "modelo")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_pragmatic.keras")
MODEL_URL = "https://drive.google.com/uc?id=1i_zEMwUkTfu9L5HGNdrIs4OPCTN6Q8Zr"

if "modelo_ml" not in st.session_state:
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Baixando modelo...")
        os.makedirs(MODEL_DIR, exist_ok=True)
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

# ---------------- FUNÇÕES AUXILIARES ----------------
import os
import subprocess
import logging
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 📡 Extrai link m3u8 de um VOD da Twitch via streamlink
def obter_url_m3u8_twitch(vod_url):
    """
    Usa o streamlink para extrair a URL .m3u8 de um VOD da Twitch.
    Ex: https://www.twitch.tv/videos/2426101798
    """
    try:
        result = subprocess.run(
            ["streamlink", "--stream-url", vod_url, "best"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            st.error(f"❌ Erro ao rodar streamlink:\n{result.stderr}")
            return None
    except Exception as e:
        st.error(f"❌ Erro ao obter URL m3u8: {e}")
        return None

# ⚡️ Captura múltiplos frames paralelamente a partir de URLs
def capturar_frames_paralelamente(vod_urls, segundo_alvo):
    """Captura frames de múltiplos VODs em paralelo."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for url in vod_urls:
            futures.append(executor.submit(capturar_frame_ffmpeg_imageio, url, "frame.jpg", skip_seconds=segundo_alvo))
        resultados = [future.result() for future in futures]
    return resultados

# ⚙️ Varredura paralela para URL personalizada com modelo ML
def processar_frame(m3u8_url, tempo, modelo, nome_jogo_previsto=None):
    frame = capturar_frame_ffmpeg_imageio(m3u8_url, segundo=tempo)

    if frame is not None:
        previsao = modelo(frame)
        if previsao and previsao["jogo"]:
            print(f"[{tempo}s] 🎰 Jogo detectado: {previsao['jogo']}")
            return {
                "segundo": tempo,
                "jogo": previsao["jogo"],
                "confianca": previsao["confianca"]
            }
    else:
        print(f"[ERRO] Frame não capturado no segundo {tempo}")
    return None

    tempos = [skip_inicial + i * intervalo for i in range(max_frames)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(processar_frame, tempo) for tempo in tempos]
        for future in futures:
            res = future.result()
            if res:
                resultados.append(res)

    session_state["dados_url"] = resultados
    return resultados

# 📂 Diretórios e arquivos fixos
STREAMERS_FILE = "streamers.txt"
DADOS_DIR = "dados"
os.makedirs(DADOS_DIR, exist_ok=True)

# 📄 Lê streamers fixos do arquivo local
def carregar_streamers():
    """Lê os streamers fixos do arquivo streamers.txt"""
    if not os.path.exists(STREAMERS_FILE):
        with open(STREAMERS_FILE, "w") as f:
            f.write("jukes\n")  # streamer padrão inicial
    with open(STREAMERS_FILE, "r") as f:
        return [l.strip() for l in f if l.strip()]

# 💾 Salva detecções em CSV
def salvar_deteccao(tipo, dados):
    """Salva dados detectados no diretório /dados como CSV"""
    nome_arquivo = f"{DADOS_DIR}/{tipo}.csv"
    df_novo = pd.DataFrame(dados)
    df_novo["data_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(nome_arquivo):
        df_existente = pd.read_csv(nome_arquivo)
        df = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df = df_novo

    df.to_csv(nome_arquivo, index=False)

# 🧽 Filtra streamers apenas com idioma português
def filtrar_streamers_pt(streamers):
    """Filtra a lista mantendo apenas streamers com idioma 'pt' (português)."""
    streamers_pt = []
    ignorados = []

    for s in streamers:
        try:
            url = f"{BASE_URL_TWITCH}users?login={s}"
            resp = requests.get(url, headers=HEADERS_TWITCH)
            data = resp.json().get("data", [])
            if data and data[0].get("broadcaster_language") == "pt":
                streamers_pt.append(s)
            else:
                ignorados.append(s)
        except Exception as e:
            logging.warning(f"Erro ao verificar idioma de {s}: {e}")
            ignorados.append(s)

    if ignorados:
        st.sidebar.warning("Alguns streamers foram ignorados por não estarem em PT:")
        for i in ignorados:
            st.sidebar.text(f"❌ {i}")

    return streamers_pt


# ---------------- FUNÇÃO: calcular minutos únicos com jogo por streamer ----------------
def calcular_minutos_por_streamer(dados, nome_jogo="pragmatic"):
    """
    Retorna um dicionário com {streamer: minutos únicos com jogo detectado}
    """
    minutos_por_streamer = {}

    for d in dados:
        if "jogo_detectado" not in d or "segundo" not in d or "streamer" not in d:
            continue
        if nome_jogo.lower() in d["jogo_detectado"].lower():
            minuto = d["segundo"] // 60
            streamer = d["streamer"]
            if streamer not in minutos_por_streamer:
                minutos_por_streamer[streamer] = set()
            minutos_por_streamer[streamer].add(minuto)

    return {s: len(mins) for s, mins in minutos_por_streamer.items()}

# ---------------- CARREGAR E FILTRAR STREAMERS FIXOS ----------------
STREAMERS_INTERESSE = carregar_streamers()
TODOS_STREAMERS = STREAMERS_INTERESSE

# ------------------ SIDEBAR REFACTORED ------------------
with st.sidebar.expander("🎯 Filtros de Data e URL"):
    data_inicio = st.date_input("Data de início", value=datetime.today() - timedelta(days=7))
    data_fim = st.date_input("Data de fim", value=datetime.today())
    url_custom = st.text_input("URL personalizada (VOD .m3u8 ou com ?t=...)")
    segundo_alvo = st.number_input("Segundo para captura manual", min_value=0, max_value=99999, value=0)

with st.sidebar.expander("🔧 Utilitários Twitch"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Testar conexão"):
            test_url = "https://api.twitch.tv/helix/streams?first=1"
            resp = requests.get(test_url, headers=HEADERS_TWITCH)
            st.write("Status:", resp.status_code)
            try:
                st.json(resp.json())
            except Exception as e:
                st.error(f"Erro ao converter resposta: {e}")
    with col2:
        if st.button("🎲 Testar categoria"):
            nome_categoria = "Virtual Casino"
            url = f"{BASE_URL_TWITCH}games?name={nome_categoria}"
            resp = requests.get(url, headers=HEADERS_TWITCH)
            st.write("🔁 Status:", resp.status_code)
            st.json(resp.json())

with st.sidebar.expander("🧠 Modelo de Detecção"):
    if "modelo_ml" in st.session_state:
        st.success("✅ Modelo ML carregado")
    else:
        st.warning("⚠️ Modelo não carregado ainda")

    if st.button("🚀 Treinar modelo agora"):
        with st.spinner("Treinando modelo..."):
            sucesso, modelo = treinar_modelo(st)
        if sucesso:
            modelo.save(MODEL_PATH)
            st.session_state["modelo_ml"] = modelo
            st.success("✅ Modelo treinado e salvo com sucesso!")
        else:
            st.warning("⚠️ Falha no treinamento do modelo.")

with st.sidebar.expander("🎯 Análise de VOD / Período"):
    streamer_escolhido = st.selectbox("👤 Escolha o streamer", carregar_streamers())
    tipo_analise = st.radio("Tipo de análise", ["VOD específica (URL)", "Por período"])
    vod_url_individual = st.text_input("📺 URL da VOD", placeholder="https://www.twitch.tv/videos/...")

    intervalo = st.number_input("⏱️ Intervalo entre frames (s)", min_value=10, max_value=600, value=60)
    max_frames = st.number_input("🖼️ Total de frames a capturar", min_value=5, max_value=100, value=30)

    if tipo_analise == "VOD específica (URL)":
        if st.button("🎯 Analisar esta VOD"):
            if vod_url_individual:
                with st.spinner("🔍 Analisando VOD..."):
                    m3u8_url = obter_url_m3u8_twitch(vod_url_individual)
                    if m3u8_url:
                        tempo_inicial = extrair_segundos_da_url_vod(vod_url_individual)
                        resultado = varrer_url_customizada_paralela(
                            m3u8_url, st, st.session_state, prever_jogo_em_frame,
                            skip_inicial=tempo_inicial, intervalo=100, max_frames=300
                        )
                        if resultado:
                            for r in resultado:
                                r["streamer"] = streamer_escolhido
                            salvar_deteccao("url", resultado)
                            st.success("✅ Análise concluída e salva com sucesso!")
                        else:
                            st.warning("⚠️ Nenhum jogo detectado na VOD.")
                    else:
                        st.error("❌ Não foi possível extrair a URL .m3u8.")
            else:
                st.warning("⚠️ Forneça a URL da VOD para análise.")

    elif tipo_analise == "Por período":
        data_inicio = st.date_input("📅 Data de início", value=datetime.today() - timedelta(days=7))
        data_fim = st.date_input("📅 Data de fim", value=datetime.today())

        if st.button("📅 Analisar por Período"):
    with st.spinner(f"🔎 Buscando VODs do streamer {streamer_escolhido} por período..."):
        vods = buscar_vods_por_streamer_e_periodo(
            streamer_escolhido, data_inicio, data_fim,
            HEADERS_TWITCH, BASE_URL_TWITCH
        )
        if not vods:
            st.warning("⚠️ Nenhuma VOD encontrada nesse período.")
        else:
            resultados = analisar_por_periodo(
                streamer_escolhido, vods,
                st, st.session_state,
                prever_jogo_em_frame, varrer_url_customizada_paralela,
                obter_url_m3u8_twitch
            )
            if resultados:
                salvar_deteccao("periodo", resultados)
                st.success("✅ Análise por período concluída e salva!")
            else:
                st.warning("⚠️ Nenhuma detecção relevante encontrada.")


    def buscar_vods_por_streamer_e_periodo(streamer, data_inicio, data_fim):
        try:
            df = pd.read_csv("vods.csv", parse_dates=["data"])
        except FileNotFoundError:
            st.error("❌ Arquivo 'vods.csv' não encontrado.")
            return []
    
        data_inicio = pd.to_datetime(data_inicio)
        data_fim = pd.to_datetime(data_fim)
    
        df_filtrado = df[
            (df["streamer"] == streamer) &
            (df["data"] >= data_inicio) &
            (df["data"] <= data_fim)
        ]
    
        return df_filtrado.to_dict(orient="records")


# ------------------ EXIBIÇÃO DE RESULTADOS (MELHORADA) ------------------
if 'dados_url' in st.session_state:
    st.markdown("### 🎰 Resultados da VOD personalizada")
    for res in st.session_state['dados_url']:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(res["frame"], caption=f"{res['segundo']}s", use_column_width=True)
        with col2:
            st.markdown(f"**🎯 Jogo detectado:** {res['jogo_detectado']}")
            st.markdown(f"🧠 **Confiança:** {res['confianca']:.2%}")

    st.success(f"Total de detecções: {len(st.session_state['dados_url'])}")


# ------------------ BOTÕES PRINCIPAIS ------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Verificar lives agora"):
        resultados = []

        for streamer in TODOS_STREAMERS:
            res = verificar_jogo_em_live(streamer, HEADERS_TWITCH, BASE_URL_TWITCH)
            if res and len(res) == 3:
                jogo, categoria, viewers = res
                resultados.append({
                    "streamer": streamer,
                    "jogo_detectado": jogo,
                    "categoria": categoria,
                    "viewers": viewers,
                    "timestamp": datetime.now()
                })

        if resultados:
            salvar_deteccao("lives", resultados)
            st.success(f"{len(resultados)} detecções salvas com sucesso!")
        else:
            st.info("Nenhum jogo detectado ao vivo.")


with col2:
    if st.button("📺 Verificar VODs no período"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        vods = buscar_vods_twitch_por_periodo(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, TODOS_STREAMERS)
        if vods:
            salvar_deteccao("vods", vods)
            st.success(f"{len(vods)} VODs salvos com sucesso!")
        else:
            st.info("Nenhuma VOD encontrada no período.")


from ml_utils import varrer_vods_com_modelo  # no topo se ainda não estiver

with col3:
    if st.button("🖼️ Varrer VODs com imagem"):
        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        resultados = varrer_vods_com_modelo(
            dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, TODOS_STREAMERS,
            st.session_state, prever_jogo_em_frame
        )
        if resultados:
            salvar_deteccao("template", resultados)
            st.success(f"{len(resultados)} jogos detectados com ML.")
        else:
            st.info("Nenhum jogo detectado com ML.")


with col4:
    if st.button("🌐 Varredura na URL personalizada") and url_custom:
        if ".m3u8" not in url_custom:
            m3u8_url = obter_url_m3u8_twitch(url_custom)
            if not m3u8_url:
                st.error("❌ Não foi possível obter o link .m3u8 a partir do VOD.")
                st.stop()
        else:
            m3u8_url = url_custom

        tempo_inicial = extrair_segundos_da_url_vod(url_custom)
        tempo_total = 720
        intervalo = 120
        max_frames = tempo_total // intervalo

        inicio = time.time()
        resultado_url = varrer_url_customizada_paralela(
            m3u8_url,
            st,
            st.session_state,
            prever_jogo_em_frame,
            skip_inicial=tempo_inicial,
            intervalo=intervalo,
            max_frames=max_frames
        )
        duracao = time.time() - inicio

        if resultado_url:
            salvar_deteccao("url", resultado_url)
            st.success(f"✅ Varredura concluída e salva em {duracao:.2f}s")
        else:
            st.warning("⚠️ Nenhum jogo detectado na URL.")

# ---------------- ABAS PRINCIPAIS ----------------
import plotly.express as px
from storage import carregar_historico


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

abas = st.tabs([
    "📊 Detecções", 
    "🏆 Ranking", 
    "🕒 Timeline", 
    "📺 VODs", 
    "📁 Histórico", 
    "📈 Dashboards", 
    "🖼️ Dataset", 
    "🎯 Streamer Focus"
])

# ------------------ ABA 0: Detecções ------------------
with abas[0]:
    st.subheader("🧠 Detecções recentes")
    if 'dados_url' in st.session_state:
        st.markdown("#### 🎰 VOD personalizada")
        for res in st.session_state['dados_url']:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(res["frame"], caption=f"{res['segundo']}s", use_container_width=True)
            with col2:
                st.markdown(f"**Jogo:** {res['jogo_detectado']}")
                st.markdown(f"**Confiança:** {res['confianca']:.2%}")

    if 'dados_vods_template' in st.session_state:
        st.markdown("#### 🖼️ Por Template")
        for res in st.session_state['dados_vods_template']:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(res["frame"], caption=f"{res['segundo']}s", use_container_width=True)
            with col2:
                st.write(f"**Streamer:** {res['streamer']}")
                st.write(f"**Jogo:** {res['jogo_detectado']}")
                st.write(f"**Tempo:** {res['segundo']}s")
                st.write(f"🔗 [Ver VOD]({res['url']})")

# ------------------ ABA 1: Ranking ------------------
with abas[1]:
    from collections import Counter

    dados_para_ranking = []
    if 'dados_url' in st.session_state:
        dados_para_ranking += st.session_state['dados_url']
    if 'dados_vods_template' in st.session_state:
        dados_para_ranking += st.session_state['dados_vods_template']

    st.subheader("🏆 Jogos mais detectados")
    if dados_para_ranking:
        df = pd.DataFrame(dados_para_ranking)
        ranking = df['jogo_detectado'].value_counts().reset_index()
        ranking.columns = ['Jogo', 'Aparições']
        st.dataframe(ranking, use_container_width=True)
        fig = px.bar(ranking, x='Jogo', y='Aparições', text='Aparições', title="Ranking de Jogos")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum dado disponível.")

# ------------------ ABA 2: Timeline ------------------
with abas[2]:
    st.subheader("🕒 Linha do Tempo de Detecção")
    dados_timeline = []
    if 'dados_url' in st.session_state:
        dados_timeline += st.session_state['dados_url']
    if 'dados_vods_template' in st.session_state:
        dados_timeline += st.session_state['dados_vods_template']
    if 'dados_lives' in st.session_state:
        dados_timeline += st.session_state['dados_lives']

    if dados_timeline:
        df = pd.DataFrame(dados_timeline)
        if 'segundo' in df.columns and 'jogo_detectado' in df.columns:
            if 'streamer' not in df.columns:
                df['streamer'] = 'Desconhecido'
            fig = px.scatter(df, x="segundo", y="jogo_detectado", color="streamer",
                             title="Timeline de Detecções",
                             hover_data=["streamer", "segundo", "url"] if 'url' in df.columns else ["streamer", "segundo"])
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados incompletos.")
    else:
        st.info("Nenhuma detecção disponível.")

# ------------------ ABA 3: VODs ------------------
with abas[3]:
    st.subheader("📺 VODs Resumidas")

    col1, _ = st.columns([1, 3])
    with col1:
        if st.button("🔄 Carregar resumo de VODs"):
            from datetime import datetime, timedelta
            dt_ini = datetime.today() - timedelta(days=7)
            dt_fim = datetime.today()
            resumo = buscar_resumo_vods(dt_ini, dt_fim, HEADERS_TWITCH, BASE_URL_TWITCH, TODOS_STREAMERS)
            st.session_state['vods_resumo'] = resumo

    if 'vods_resumo' in st.session_state and st.session_state['vods_resumo']:
        df = pd.DataFrame(st.session_state['vods_resumo'])
        df["data"] = pd.to_datetime(df["data"]).dt.strftime("%d/%m/%Y %H:%M")
        df["link"] = df["url"].apply(lambda x: f"[Abrir VOD]({x})")
        df = df.drop(columns=["url"])
        df = df.sort_values(by="duração (min)", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="vods_resumo.csv", mime="text/csv")
    else:
        st.info("Nenhuma VOD carregada.")

# ------------------ ABA 5: Dashboards ------------------
with abas[5]:
    st.subheader("📈 Painéis de Detecção")
    dados_template = carregar_historico("template")
    dados_url = carregar_historico("url")
    dados_lives = carregar_historico("lives")
    df_geral = pd.concat([dados_template, dados_url, dados_lives], ignore_index=True)

    if df_geral.empty:
        st.info("📭 Nenhum dado disponível para análise. Execute uma varredura primeiro.")
    else:
        st.write("✅ Dados carregados para análise.")
        
        # --- Gráfico 1: Share of Voice ---
        st.markdown("### 🥧 Share of Voice (Distribuição dos Jogos Detectados)")

        if "jogo_detectado" in df_geral.columns:
            ranking = df_geral["jogo_detectado"].value_counts().reset_index()
            ranking.columns = ["Jogo", "Aparições"]

            fig1 = px.pie(
                ranking,
                names="Jogo",
                values="Aparições",
                title="Distribuição dos Jogos Detectados"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # --- Gráfico 2: Detecções por Streamer ---
        st.markdown("### 🧍‍♂️ Comparativo: Total de Detecções por Streamer")

        if "streamer" in df_geral.columns and "jogo_detectado" in df_geral.columns:
            comparativo = df_geral.groupby("streamer")["jogo_detectado"].count().reset_index()
            comparativo.columns = ["Streamer", "Total de Detecções"]
            comparativo = comparativo.sort_values(by="Total de Detecções", ascending=False)

            fig2 = px.bar(
                comparativo,
                x="Streamer",
                y="Total de Detecções",
                title="🎯 Total de Jogos Detectados por Streamer",
                text_auto=True
            )
            st.plotly_chart(fig2, use_container_width=True)

        # --- Gráfico 3: Evolução Temporal ---
        st.markdown("### 📈 Evolução Temporal das Detecções")
        
        if "data_hora" in df_geral.columns and "jogo_detectado" in df_geral.columns:
            # ✅ Conversão segura para datetime
            df_geral["data_hora"] = pd.to_datetime(df_geral["data_hora"], errors="coerce")
        
            evolucao = (
                df_geral.groupby([pd.Grouper(key="data_hora", freq="D"), "jogo_detectado"])
                .size()
                .reset_index(name="Detecções")
            )
        
            fig3 = px.line(
                evolucao,
                x="data_hora",
                y="Detecções",
                color="jogo_detectado",
                title="📅 Detecções por Jogo ao Longo do Tempo"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Dados temporais insuficientes para gerar evolução.")


        # --- Gráfico 4: Tempo Médio por Jogo ---
        st.markdown("### ⏱ Tempo Médio de Detecção por Jogo")

        if "jogo_detectado" in df_geral.columns and "segundo" in df_geral.columns:
            media_tempo = df_geral.groupby("jogo_detectado")["segundo"].mean().reset_index()
            media_tempo.columns = ["Jogo", "Tempo Médio (s)"]
            media_tempo = media_tempo.sort_values(by="Tempo Médio (s)", ascending=False)

            fig4 = px.bar(
                media_tempo,
                x="Jogo",
                y="Tempo Médio (s)",
                text_auto=".2f",
                title="⏱ Tempo Médio de Detecção por Jogo"
            )
            st.plotly_chart(fig4, use_container_width=True)


        # --- Gráfico 5: Top Streamers por Jogo ---
        st.markdown("### 🧍‍♂️🎮 Streamers com mais detecções por Jogo")

        if "jogo_detectado" in df_geral.columns and "streamer" in df_geral.columns:
            top_streamers_jogo = (
            df_geral.groupby(["jogo_detectado", "streamer"])
            .size()
            .reset_index(name="Detecções")
            )

            fig5 = px.bar(
            top_streamers_jogo,
            x="jogo_detectado",
            y="Detecções",
            color="streamer",
            title="Top Streamers por Jogo Detectado",
            barmode="group"
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Não há dados suficientes para exibir Top Streamers por Jogo.")

# --- Gráfico 6: Distribuição por Dia da Semana ---
st.markdown("### 📆 Detecções por Dia da Semana")

if "data_hora" in df_geral.columns and "jogo_detectado" in df_geral.columns:
    # Criar coluna de dia da semana sem uso de locale
    dias_semana = {
        0: 'segunda-feira',
        1: 'terça-feira',
        2: 'quarta-feira',
        3: 'quinta-feira',
        4: 'sexta-feira',
        5: 'sábado',
        6: 'domingo'
    }
    df_geral["dia_semana"] = df_geral["data_hora"].dt.dayofweek.map(dias_semana)

    distrib_dia = df_geral["dia_semana"].value_counts().reindex([
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo"
    ]).fillna(0).reset_index()

    distrib_dia.columns = ["Dia", "Detecções"]

    fig6 = px.bar(
        distrib_dia,
        x="Dia",
        y="Detecções",
        title="📆 Total de Detecções por Dia da Semana",
        text_auto=True
    )
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("Dados temporais insuficientes para gerar distribuição semanal.")

# --- Gráfico 7: Mapa de Calor Jogo x Dia da Semana ---
st.markdown("### 🔥 Mapa de Calor: Jogos por Dia da Semana")

if "data_hora" in df_geral.columns and "jogo_detectado" in df_geral.columns:
    # Reaproveita df_geral["dia_semana"] já criado
    matriz = (
        df_geral.groupby(["jogo_detectado", "dia_semana"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[
            "segunda-feira", "terça-feira", "quarta-feira",
            "quinta-feira", "sexta-feira", "sábado", "domingo"
        ], fill_value=0)
    )

    fig7 = px.imshow(
        matriz,
        labels=dict(x="Dia da Semana", y="Jogo", color="Detecções"),
        aspect="auto",
        color_continuous_scale="Oranges",
        title="🔥 Frequência de Jogos por Dia da Semana"
    )
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.info("Dados temporais insuficientes para gerar mapa de calor.")

            # --- Gráfico 8: Tendência de Crescimento por Jogo ---
        
st.markdown("### 📈 Tendência de Crescimento por Jogo (Média Móvel 3 dias)")

if "data_hora" in df_geral.columns and "jogo_detectado" in df_geral.columns:
    tendencia = (
        df_geral.groupby([pd.Grouper(key="data_hora", freq="D"), "jogo_detectado"])
        .size()
        .reset_index(name="Detecções")
    )

    # Aplica média móvel de 3 dias por jogo
    tendencia["MediaMovel"] = (
        tendencia.groupby("jogo_detectado")["Detecções"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    fig8 = px.line(
        tendencia,
        x="data_hora",
        y="MediaMovel",
        color="jogo_detectado",
        title="📈 Tendência de Detecção dos Jogos (Média Móvel)"
    )
    st.plotly_chart(fig8, use_container_width=True)
else:
    st.info("Dados temporais insuficientes para gerar tendência.")

   # --- Gráfico 9: Média de Viewers por Jogo ---
st.markdown("### 👀 Média de Viewers por Jogo Detectado")

if "jogo_detectado" in df_geral.columns and "viewers" in df_geral.columns:
    media_viewers = df_geral.groupby("jogo_detectado")["viewers"].mean().reset_index()
    media_viewers.columns = ["Jogo", "Viewers Médios"]
    media_viewers = media_viewers.sort_values(by="Viewers Médios", ascending=False)

    fig9 = px.bar(
        media_viewers,
        x="Jogo",
        y="Viewers Médios",
        text_auto=".0f",
        title="👀 Audiência Média por Jogo Detectado"
    )
    st.plotly_chart(fig9, use_container_width=True)
else:
    st.info("Nenhum dado com número de viewers disponível ainda.")

# --- Gráfico 10: Média de Viewers por Streamer ---
st.markdown("### 🎥 Streamers com Maior Audiência Média")

if "streamer" in df_geral.columns and "viewers" in df_geral.columns:
    media_streamers = df_geral.groupby("streamer")["viewers"].mean().reset_index()
    media_streamers.columns = ["Streamer", "Viewers Médios"]
    media_streamers = media_streamers.sort_values(by="Viewers Médios", ascending=False)

    fig10 = px.bar(
        media_streamers,
        x="Streamer",
        y="Viewers Médios",
        text_auto=".0f",
        title="🎥 Audiência Média por Streamer"
    )
    st.plotly_chart(fig10, use_container_width=True)
else:
    st.info("Nenhum dado de viewers por streamer disponível.")

# --- Gráfico 11: Evolução dos Viewers ao Longo do Tempo ---
st.markdown("### ⏱️ Evolução dos Viewers nas Detecções")

if "data_hora" in df_geral.columns and "viewers" in df_geral.columns:
    df_viewers = df_geral.copy()
    df_viewers["data_hora"] = pd.to_datetime(df_viewers["data_hora"])
    evolucao_viewers = (
        df_viewers.groupby(pd.Grouper(key="data_hora", freq="D"))["viewers"].mean().reset_index()
    )

    fig11 = px.line(
        evolucao_viewers,
        x="data_hora",
        y="viewers",
        title="⏱️ Audiência Média ao Longo do Tempo"
    )
    st.plotly_chart(fig11, use_container_width=True)
else:
    st.info("Sem dados temporais suficientes para mostrar evolução de viewers.")

# --- Gráfico 12: Pico de Audiência por Streamer ---
st.markdown("### 🔝 Pico de Audiência por Streamer")

if "streamer" in df_geral.columns and "viewers" in df_geral.columns:
    pico_streamers = df_geral.groupby("streamer")["viewers"].max().reset_index()
    pico_streamers.columns = ["Streamer", "Pico de Viewers"]
    pico_streamers = pico_streamers.sort_values(by="Pico de Viewers", ascending=False)

    fig12 = px.bar(
        pico_streamers,
        x="Streamer",
        y="Pico de Viewers",
        text_auto=True,
        title="🔝 Maior Número de Viewers por Streamer"
    )
    st.plotly_chart(fig12, use_container_width=True)
else:
    st.info("Não há dados de pico de audiência.")

# ------------------ SUGERIR NOVOS STREAMERS ------------------
def sugerir_novos_streamers():
    sugestoes = []
    categorias_alvo = ["Slots", "Virtual Casino"]

    try:
        response = requests.get(
            f"{BASE_URL_TWITCH}streams?first=100",
            headers=HEADERS_TWITCH
        )
        data = response.json().get("data", [])
        atuais = set(STREAMERS_INTERESSE)

        for stream in data:
            game_name = stream.get("game_name", "").lower()
            login = stream.get("user_login")
            if any(cat.lower() in game_name for cat in categorias_alvo):
                if login and login not in atuais:
                    sugestoes.append(login)
    except Exception as e:
        logging.error(f"Erro ao buscar streamers: {e}")

    return sugestoes


st.sidebar.markdown("---")
if st.sidebar.button("🔎 Sugerir novos streamers PT-BR"):
    novos = sugerir_novos_streamers()
    if novos:
        st.success("Sugestões de novos streamers (idioma PT):")
        for s in novos:
            st.write(f"- {s}")
    else:
        st.warning("Nenhum novo streamer encontrado.")


# ------------------ Teste manual de resposta da Twitch ------------------
if st.sidebar.button("🔬 Testar busca de streams"):
    test_url = "https://api.twitch.tv/helix/streams?first=20"
    resp = requests.get(test_url, headers=HEADERS_TWITCH)
    st.sidebar.write("🔁 Status:", resp.status_code)
    st.sidebar.json(resp.json())

def buscar_vods_por_streamer_e_periodo(streamer, data_inicio, data_fim):
    try:
        df = pd.read_csv("vods.csv", parse_dates=["data"])
    except FileNotFoundError:
        st.error("❌ Arquivo 'vods.csv' não encontrado.")
        return []

    data_inicio = pd.to_datetime(data_inicio)
    data_fim = pd.to_datetime(data_fim)

    df_filtrado = df[
        (df["streamer"] == streamer) &
        (df["data"] >= data_inicio) &
        (df["data"] <= data_fim)
    ]

    return df_filtrado.to_dict(orient="records")


def analisar_por_periodo(streamer, vods):
    resultados_finais = []

    for vod in vods:
        m3u8_url = obter_url_m3u8_twitch(vod["url"])
        if not m3u8_url:
            continue

        resultado = varrer_url_customizada_paralela(
            m3u8_url, st, st.session_state, prever_jogo_em_frame,
            skip_inicial=0, intervalo=20, max_frames=10
        )

        if resultado:
            for r in resultado:
                r["streamer"] = streamer
            resultados_finais.extend(resultado)

    return resultados_finais
