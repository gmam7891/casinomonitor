import cv2
import os
import numpy as np
import logging
import subprocess
from datetime import datetime
import requests
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf

# ------------------------------
# DETECÇÃO POR TEMPLATE
# ------------------------------
def match_template_from_image(image_path, template_path="templates/pragmaticplay.png"):
    try:
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_path, 0)
        if template is not None:
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            logging.info(f"Similaridade máxima: {max_val:.3f}")
            if max_val >= 0.7:
                return "pragmaticplay"
    except Exception as e:
        logging.error(f"Erro no template matching: {e}")
    return None

# ------------------------------
# CAPTURA DE FRAME COM FFMPEG
# ------------------------------
def capturar_frame_ffmpeg_imageio(m3u8_url, output_path="frame.jpg", skip_seconds=10):
    try:
        width, height = 1280, 720
        cmd = [
            "ffmpeg", "-y", "-ss", str(skip_seconds), "-i", m3u8_url,
            "-vf", f"scale={width}:{height}",
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        logging.error(f"Erro ao capturar frame: {e}")
        return None

# ------------------------------
# PREVER JOGO EM UM FRAME
# ------------------------------
def prever_jogo_em_frame(frame_path, model):
    try:
        img = keras_image.load_img(frame_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        pred = model.predict(x)[0][0]
        logging.info(f"Probabilidade modelo ML: {pred:.3f}")
        return "pragmaticplay" if pred >= 0.5 else None
    except Exception as e:
        logging.error(f"Erro ao prever com modelo ML: {e}")
        return None

# ------------------------------
# DETECÇÃO EM LIVE
# ------------------------------
def verificar_jogo_em_live(streamer, HEADERS_TWITCH, BASE_URL_TWITCH):
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

        m3u8_url = f"https://usher.ttvnw.net/api/channel/hls/{streamer}.m3u8"
        temp_frame = f"{streamer}_frame.jpg"
        if capturar_frame_ffmpeg_imageio(m3u8_url, temp_frame):
            jogo = match_template_from_image(temp_frame)
            os.remove(temp_frame)
            return jogo, game_name
    except Exception as e:
        logging.error(f"Erro ao verificar live de {streamer}: {e}")
    return None

# ------------------------------
# BUSCAR VODs POR PERÍODO
# ------------------------------
def buscar_vods_twitch_por_periodo(data_inicio, data_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE):
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
            logging.error(f"Erro ao buscar VODs: {e}")
    return resultados

# ------------------------------
# VARRER VODs COM TEMPLATE
# ------------------------------
def varrer_vods_com_template(data_inicio, data_fim, HEADERS_TWITCH, BASE_URL_TWITCH, STREAMERS_INTERESSE):
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
            logging.error(f"Erro ao buscar e varrer VODs: {e}")
    return resultados

# ------------------------------
# VARRER URL PERSONALIZADA
# ------------------------------
def varrer_url_customizada(url, st, st_session_state, prever_jogo_em_frame_fn, duracao_analise=30, intervalo_frames=1):
    resultados = []
    total_frames = duracao_analise // intervalo_frames
    progresso = st.progress(0, text="🔎 Iniciando varredura...")

    for i in range(int(total_frames)):
        skip = i * intervalo_frames
        frame_path = f"custom_frame_{i}.jpg"
        progresso.progress(i / total_frames, text=f"⏱️ Analisando segundo {skip}...")

        if capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=skip):
            jogo = prever_jogo_em_frame_fn(frame_path, st_session_state.get("modelo_ml"))
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
