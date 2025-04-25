import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from ml_utils_corrigido import processar_frame, prever_jogo_em_frame, capturar_frame_ffmpeg_imageio
from concurrent.futures import ThreadPoolExecutor
import streamlink
import os

def obter_url_m3u8_twitch(url_vod):
    try:
        stream = streamlink.streams(url_vod)
        if "best" in stream:
            return stream["best"].url
    except Exception as e:
        print(f"[ERRO] Não foi possível obter a .m3u8: {e}")
    return None

def varrer_url_customizada_paralela(m3u8_url, st, session_state, skip_inicial=0, intervalo=20, max_frames=10):
    tempos = [skip_inicial + i * intervalo for i in range(max_frames)]
    resultados = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(processar_frame, m3u8_url, tempo, session_state)
            for tempo in tempos
        ]
        for future in futures:
            res = future.result()
            if res:
                resultados.append(res)
    session_state["dados_url"] = resultados
    return resultados

def salvar_deteccao(tipo, resultados):
    df = pd.DataFrame(resultados)
    data_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    caminho = f"deteccoes_{tipo}_{data_str}.csv"
    df.to_csv(caminho, index=False)
    return caminho

st.set_page_config(layout="wide")
st.title("🎰 Monitoramento de Jogos - Pragmatic Play")

with st.sidebar.expander("🎯 Análise de VOD / Período"):
    streamer_escolhido = st.text_input("👤 Nome do streamer")
    tipo_analise = st.radio("Tipo de análise", ["VOD específica (URL)", "Por período"])
    vod_url_individual = st.text_input("📺 URL da VOD", placeholder="https://www.twitch.tv/videos/...")
    intervalo = st.slider("⏱️ Intervalo entre frames (segundos)", 10, 600, 60)
    max_frames = st.slider("📸 Total de frames a capturar", 1, 50, 10)

    if tipo_analise == "VOD específica (URL)":
        if st.button("🎯 Analisar esta VOD"):
            if vod_url_individual:
                m3u8_url = obter_url_m3u8_twitch(vod_url_individual)
                if m3u8_url:
                    st.info(f"🔗 .m3u8: {m3u8_url}")
                    with st.spinner("🔍 Analisando..."):
                        resultado = varrer_url_customizada_paralela(
                            m3u8_url, st, st.session_state,
                            skip_inicial=0, intervalo=intervalo, max_frames=max_frames
                        )
                        if resultado:
                            salvar_deteccao("url", resultado)
                            st.success("✅ Análise concluída e salva com sucesso!")
                            df = pd.DataFrame(resultado)
                            st.dataframe(df[["segundo", "jogo_detectado", "confianca"]])
                        else:
                            st.warning("⚠️ Nenhuma detecção realizada.")
                else:
                    st.error("❌ Não foi possível extrair a URL .m3u8.")
            else:
                st.warning("⚠️ Forneça a URL da VOD para análise.")

    elif tipo_analise == "Por período":
        data_inicio = st.date_input("📅 Data de início", value=datetime.today() - timedelta(days=7))
        data_fim = st.date_input("📅 Data de fim", value=datetime.today())

        if st.button("📅 Analisar por Período"):
            try:
                df_vods = pd.read_csv("vods.csv", parse_dates=["data"])
                df_filtrado = df_vods[
                    (df_vods["streamer"] == streamer_escolhido) &
                    (df_vods["data"] >= pd.to_datetime(data_inicio)) &
                    (df_vods["data"] <= pd.to_datetime(data_fim))
                ]
                vods = df_filtrado.to_dict(orient="records")
                if not vods:
                    st.warning("⚠️ Nenhuma VOD encontrada para esse período.")
                else:
                    resultados = []
                    with st.spinner("🔍 Analisando VODs por período..."):
                        for vod in vods:
                            m3u8_url = obter_url_m3u8_twitch(vod["url"])
                            if not m3u8_url:
                                continue
                            res = varrer_url_customizada_paralela(
                                m3u8_url, st, st.session_state,
                                skip_inicial=0, intervalo=intervalo, max_frames=max_frames
                            )
                            for r in res:
                                r["streamer"] = streamer_escolhido
                                r["vod_url"] = vod["url"]
                            resultados.extend(res)

                    if resultados:
                        salvar_deteccao("periodo", resultados)
                        df = pd.DataFrame(resultados)
                        st.success(f"✅ {len(resultados)} frames analisados no período.")
                        st.dataframe(df[["streamer", "segundo", "jogo_detectado", "confianca", "vod_url"]])
                    else:
                        st.warning("⚠️ Nenhuma detecção relevante encontrada.")
            except FileNotFoundError:
                st.error("❌ Arquivo vods.csv não encontrado.")

st.markdown("---")
st.markdown("🧠 Modelo carregado automaticamente no início via `ml_utils_corrigido.py`")
