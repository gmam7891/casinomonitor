import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from util import create_dirs, capturar_frame_ffmpeg, match_template_from_image
from modelo import carregar_modelo, prever_jogo_em_frame
from processamento import verificar_jogo_em_live

# Config inicial
st.set_page_config(page_title="Monitor Cassino PP", layout="wide")
create_dirs()

# Streamers
if not os.path.exists("streamers.txt"):
    with open("streamers.txt", "w") as f:
        f.write("jukes\n")

with open("streamers.txt", "r") as f:
    STREAMERS = [s.strip() for s in f if s.strip()]

# Título
st.markdown("<h1 style='color:#F68B2A;'>🎰 Monitor Cassino Pragmatic Play</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Parâmetros")
data_inicio = st.sidebar.date_input("Data início", datetime.today() - timedelta(days=7))
data_fim = st.sidebar.date_input("Data fim", datetime.today())
url_custom = st.sidebar.text_input("URL .m3u8 personalizada")

modelo = carregar_modelo()

# 🔍 Função ajustada para varredura personalizada
def varrer_url_customizada(url, duracao_analise=30, intervalo_frames=1):
    resultados = []
    total_frames = duracao_analise // intervalo_frames
    progresso = st.progress(0, text="🔎 Iniciando varredura...")

    for i in range(int(total_frames)):
        skip = i * intervalo_frames
        frame_path = f"custom_frame_{i}.jpg"
        progresso.progress(i / total_frames, text=f"⏱️ Analisando segundo {skip}...")

        if capturar_frame_ffmpeg(url, frame_path, skip_seconds=skip):
            jogo = prever_jogo_em_frame(modelo, frame_path) or match_template_from_image(frame_path)
            if jogo:
                resultados.append({
                    "jogo_detectado": jogo,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fonte": f"URL personalizada (segundo {skip})"
                })
                st.image(frame_path, caption=f"✅ Frame detectado no segundo {skip}", use_column_width=True)
                break
            else:
                os.remove(frame_path)

    progresso.empty()

    if not resultados:
        st.warning("❌ Nenhuma detecção foi feita na URL.")
    else:
        st.success("🎯 Jogo detectado com sucesso!")

    return resultados

# Colunas de botões
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Verificar lives agora"):
        resultados = []
        for streamer in STREAMERS:
            res = verificar_jogo_em_live(streamer)
            if res:
                jogo, categoria = res
                resultados.append({
                    "streamer": streamer,
                    "jogo_detectado": jogo,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fonte": "Live",
                    "categoria": categoria
                })
        if resultados:
            st.session_state['dados_lives'] = resultados
        else:
            st.warning("Nenhuma detecção em lives.")

with col2:
    st.info("🚧 Em breve: varredura de VODs com template ou ML.")

with col3:
    if st.button("🌐 Rodar varredura na URL personalizada") and url_custom:
        resultado_url = varrer_url_customizada(url_custom)
        if resultado_url:
            st.session_state['dados_url'] = resultado_url

# Exibição dos resultados
if 'dados_lives' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_lives'])
    st.markdown("### 📺 Detecções em Lives")
    st.dataframe(df, use_container_width=True)

if 'dados_url' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_url'])
    st.markdown("### 🌐 Detecção via URL personalizada")
    st.dataframe(df, use_container_width=True)

if not any(k in st.session_state for k in ['dados_lives', 'dados_url']):
    st.info("ℹ️ Nenhuma detecção ainda. Execute uma das ações acima para iniciar.")
