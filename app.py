from datetime import datetime, timedelta
import requests
import streamlit as st
st.set_page_config(page_title="Monitor Cassino PP - Detecção", layout="wide")
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import subprocess



# ------------------------------
# CONFIGURAÇÕES INICIAIS
# ------------------------------
CLIENT_ID = 'gp762nuuoqcoxypju8c569th9wz7q5'
ACCESS_TOKEN = 'moila7dw5ejlk3eja6ne08arw0oexs'
HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'
STREAMERS_FILE = "streamers.txt"
TEMPLATES_DIR = "templates/"

if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

def carregar_streamers():
    if not os.path.exists(STREAMERS_FILE):
        with open(STREAMERS_FILE, "w", encoding="utf-8") as f:
            f.write("jukes\n")
    with open(STREAMERS_FILE, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]

STREAMERS_INTERESSE = carregar_streamers()

def match_template_from_image(image_path, template_path="templates/pragmaticplay.png"):
    try:
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_path, 0)
        if template is not None:
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print(f"Similaridade máxima: {max_val:.3f}")
            if max_val >= 0.7:
                return "pragmaticplay"
            else:
                print("Logo não encontrado no frame. Similaridade abaixo do limiar.")
        else:
            print("Template não foi carregado corretamente.")
    except Exception as e:
        print(f"Erro no template matching: {e}")
    return None

def get_stream_m3u8_url(user_login):
    return f"https://usher.ttvnw.net/api/channel/hls/{user_login}.m3u8"

def capturar_frame_ffmpeg_imageio(m3u8_url, output_path="frame.jpg", skip_seconds=10):
    try:
        width, height = 1280, 720
        cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(skip_seconds),
        "-i", m3u8_url,
            "-vf", f"scale={width}:{height}",
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"Erro ao capturar frame: {e}")
        return None

def varrer_url_customizada(url, duracao_analise=86400, intervalo_frames=1):
    resultados = []
    total_frames = duracao_analise // intervalo_frames
    progresso = st.progress(0, text="🔎 Iniciando varredura...")
    modelo = st.session_state.get("modelo_ml", None)

    for i in range(int(total_frames)):
        skip = i * intervalo_frames
        frame_path = f"custom_frame_{i}.jpg"
        progresso.progress(i / total_frames, text=f"⏱️ Analisando segundo {skip}...")

        if capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=skip):
            jogo = prever_jogo_em_frame(frame_path)
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


def verificar_jogo_em_live(streamer):
    try:
        user_response = requests.get(BASE_URL_TWITCH + f'users?login={streamer}', headers=HEADERS_TWITCH)
        user_data = user_response.json().get('data', [])
        if not user_data:
            return None
        user_id = user_data[0]['id']
        stream_response = requests.get(BASE_URL_TWITCH + f'streams?user_id={user_id}', headers=HEADERS_TWITCH)
        stream_data = stream_response.json().get('data', [])
        if not stream_data:
            return None

        game_id = stream_data[0].get('game_id')
        game_name = "Desconhecida"
        if game_id:
            game_response = requests.get(BASE_URL_TWITCH + f'games?id={game_id}', headers=HEADERS_TWITCH)
            game_data = game_response.json().get("data", [])
            if game_data:
                game_name = game_data[0]['name']

        m3u8_url = get_stream_m3u8_url(streamer)
        temp_frame = f"{streamer}_frame.jpg"
        if capturar_frame_ffmpeg_imageio(m3u8_url, temp_frame):
            jogo = match_template_from_image(temp_frame)
            os.remove(temp_frame)
            return jogo, game_name
    except Exception as e:
        print(f"Erro ao verificar live de {streamer}: {e}")
    return None

def buscar_vods_twitch_por_periodo(data_inicio, data_fim):
    resultados = []
    for streamer in STREAMERS_INTERESSE:
        try:
            user_response = requests.get(BASE_URL_TWITCH + f'users?login={streamer}', headers=HEADERS_TWITCH)
            user_data = user_response.json().get('data', [])
            if not user_data:
                continue
            user_id = user_data[0]['id']
            vod_response = requests.get(BASE_URL_TWITCH + f'videos?user_id={user_id}&type=archive&first=20', headers=HEADERS_TWITCH)
            vods = vod_response.json().get('data', [])

            for vod in vods:
                created_at = datetime.strptime(vod['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                if not (data_inicio <= created_at <= data_fim):
                    continue
                resultados.append({
                    "streamer": streamer,
                    "jogo_detectado": "-",
                    "timestamp": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "fonte": "Twitch VOD",
                    "categoria": vod.get("game_name", "Desconhecida"),
                    "url": vod['url']
                })
        except Exception as e:
            print(f"Erro ao buscar VODs: {e}")
    return resultados

def varrer_vods_com_template(data_inicio, data_fim):
    resultados = []
    for streamer in STREAMERS_INTERESSE:
        try:
            user_response = requests.get(BASE_URL_TWITCH + f'users?login={streamer}', headers=HEADERS_TWITCH)
            user_data = user_response.json().get('data', [])
            if not user_data:
                continue
            user_id = user_data[0]['id']
            vod_response = requests.get(BASE_URL_TWITCH + f'videos?user_id={user_id}&type=archive&first=20', headers=HEADERS_TWITCH)
            vods = vod_response.json().get('data', [])

            for vod in vods:
                created_at = datetime.strptime(vod['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                if not (data_inicio <= created_at <= data_fim):
                    continue

                vod_url = vod['url']
                vod_id = vod_url.split('/')[-1]
                m3u8_url = f"https://vod-secure.twitch.tv/{vod_id}/chunked/index-dvr.m3u8"

                frame_path = f"vod_frame_{vod_id}.jpg"
                if capturar_frame_ffmpeg_imageio(m3u8_url, frame_path):
                    jogo = match_template_from_image(frame_path)
                    os.remove(frame_path)
                    if jogo:
                        resultados.append({
                            "streamer": streamer,
                            "jogo_detectado": jogo,
                            "timestamp": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "fonte": "VOD",
                            "categoria": vod.get("game_name", "Desconhecida"),
                            "url": vod_url
                        })
        except Exception as e:
            print(f"Erro ao buscar e varrer VODs: {e}")
    return resultados

# ------------------------------
# MACHINE LEARNING E SUGESTÃO DE NOVOS STREAMERS
# ------------------------------

def sugerir_novos_streamers(game_name="Slots"):
    sugestoes = []
    try:
        response = requests.get(BASE_URL_TWITCH + f'streams?game_name={game_name}&first=50', headers=HEADERS_TWITCH)
        data = response.json().get("data", [])
        atuais = set(STREAMERS_INTERESSE)
        for stream in data:
            login = stream.get("user_login")
            if login and login not in atuais:
                sugestoes.append(login)
    except Exception as e:
        print(f"Erro ao buscar novos streamers: {e}")
    return sugestoes

# ------------------------------
# MODELO DE MACHINE LEARNING (CNN SIMPLES)
# ------------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_pragmatic.keras")

@st.cache_resource
def carregar_modelo():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.warning("Modelo de ML ainda não treinado. Usando detecção por template.", icon="⚠️")
        return None

# modelo_ml será carregado dinamicamente com session_state

if "modelo_ml" not in st.session_state and os.path.exists(MODEL_PATH):
    st.session_state["modelo_ml"] = load_model(MODEL_PATH)

def prever_jogo_em_frame(frame_path):
    modelo = st.session_state.get("modelo_ml", None)
    if modelo is None:
        return match_template_from_image(frame_path)  # fallback
    try:
        img = image.load_img(frame_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        pred = modelo.predict(x)[0][0]
        print(f"Probabilidade modelo ML: {pred:.3f}")
        return "pragmaticplay" if pred >= 0.5 else None
    except Exception as e:
        print(f"Erro ao prever com modelo ML: {e}")
        return None
# ------------------------------
# ⚠️ REMOVIDO para resolver erro: set_page_config deve ser o primeiro comando Streamlit

st.markdown(
    """
    <style>
        body {
            background-color: white;
            color: black;
        }
        .stApp {
            background-color: white;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style='background-color:white; padding:10px; display:flex; align-items:center;'>
        <img src='https://findfaircasinos.com/gfx/uploads/620_620_kr/716_Pragmatic%20play%20logo.png' style='height:60px; margin-right:20px;'>
        <h1 style='color:black; margin:0;'>Monitor Cassino Pragmatic Play</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("🎯 Filtros")
streamers_input = st.sidebar.text_input("Streamers (separados por vírgula)")
data_inicio = st.sidebar.date_input("Data de início", value=datetime.today() - timedelta(days=7))
data_fim = st.sidebar.date_input("Data de fim", value=datetime.today())
url_custom = st.sidebar.text_input("URL .m3u8 personalizada")
# Teste de detecção em tempo exato via input
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Testar segundo exato")
segundo_alvo = st.sidebar.number_input("Segundo do vídeo", min_value=0, max_value=100000, value=0, step=1)

if st.sidebar.button("🎯 Capturar frame no tempo exato") and url_custom:
    st.markdown("### 🎯 Teste de frame em tempo específico")
    frame_path = "frame_exato.jpg"
    if capturar_frame_ffmpeg_imageio(url_custom, frame_path, skip_seconds=segundo_alvo):
        jogo = prever_jogo_em_frame(frame_path)
        st.image(frame_path, caption=f"🕒 Frame em {segundo_alvo} segundos", use_column_width=True)
        if jogo:
            st.success(f"🎰 Jogo detectado: {jogo}")
        else:
            st.warning("❌ Nenhum jogo detectado nesse ponto.")
    else:
        st.error("⚠️ Não foi possível capturar o frame. Verifique a URL.")

if st.sidebar.button("🚀 Treinar modelo agora"):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models

    st.sidebar.write("Iniciando treinamento...")

    dataset_dir = "dataset"
    img_height, img_width = 224, 224
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=5)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        st.sidebar.success("✅ Modelo treinado e salvo com sucesso como 'modelo_pragmatic.keras'")
        st.sidebar.write(f"📁 Caminho: {MODEL_PATH}")
        st.rerun()
    else:
        st.sidebar.error("❌ Modelo NÃO foi salvo! Verifique permissões ou erros no ambiente.")

streamers_filtrados = [s.strip().lower() for s in streamers_input.split(",") if s.strip()] if streamers_input else []

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Verificar lives agora"):
        resultados = []
        for streamer in STREAMERS_INTERESSE:
            resultado_live = verificar_jogo_em_live(streamer)
            if resultado_live:
                jogo, categoria = resultado_live
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
        vod_resultados = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim)
        if vod_resultados:
            st.session_state['dados_vods'] = vod_resultados

with col3:
    if st.button("🌐 Rodar varredura na URL personalizada") and url_custom:
        resultado_url = varrer_url_customizada(url_custom)
        if resultado_url:
            st.session_state['dados_url'] = resultado_url
            st.image("custom_frame_0.jpg", caption="Frame analisado", use_column_width=True)

with col4:
    if st.button("🖼️ Varrer VODs com detecção de imagem"):
        dt_inicio = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        st.session_state['dados_vods_template'] = varrer_vods_com_template(dt_inicio, dt_fim)

if 'dados_lives' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_lives'])
    if streamers_filtrados and 'streamer' in df.columns:
        df = df[df['streamer'].str.lower().isin(streamers_filtrados)]
    st.markdown("<h3 style='color:#F68B2A;'>Detecções em Lives</h3>", unsafe_allow_html=True)
    for col in ['categoria']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"🎯 {x}")
    st.dataframe(df, use_container_width=True)

if 'dados_vods' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_vods'])
    if streamers_filtrados and 'streamer' in df.columns:
        df = df[df['streamer'].str.lower().isin(streamers_filtrados)]
    st.markdown("<h3 style='color:#F68B2A;'>Detecções em VODs</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

if 'dados_vods_template' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_vods_template'])
    st.markdown("<h3 style='color:#F68B2A;'>Detecções por imagem nas VODs</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

if 'dados_url' in st.session_state:
    df = pd.DataFrame(st.session_state['dados_url'])
    st.markdown("<h3 style='color:#F68B2A;'>Detecção em URL personalizada</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

if not any(k in st.session_state for k in ['dados_lives', 'dados_vods', 'dados_url', 'dados_vods_template']):
    st.info("Nenhuma detecção encontrada.")

# Treinamento do modelo pelo Streamlit



# Sugestão de novos streamers

if st.sidebar.button("🔎 Buscar novos streamers"):
    novos = sugerir_novos_streamers()
    if novos:
        st.success(f"Encontrados {len(novos)} novos possíveis streamers:")
        for nome in novos:
            st.write(f"- {nome}")
    else:
        st.warning("Nenhum novo streamer encontrado no momento.")
