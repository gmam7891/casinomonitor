import os
import re
import logging
import traceback
from datetime import datetime, timezone
from collections import Counter

import cv2
import requests
import numpy as np
from PIL import Image
import subprocess
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import models
import matplotlib.pyplot as plt


def capturar_frame_ffmpeg_imageio(url, output_path, skip_seconds=0):
    try:
        cmd = [
            "ffmpeg",
            "-ss", str(skip_seconds),
            "-i", url,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
            "-y"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ffmpeg erro] {result.stderr}")
            return False
        return os.path.exists(output_path)
    except Exception as e:
        print(f"[Erro] capturar_frame_ffmpeg_imageio: {e}")
        return False


def prever_jogo_em_frame(path, modelo):
    img = keras_image.load_img(path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = modelo.predict(img_array)[0][0]
    classe = "Classe 1" if prediction >= 0.5 else "Classe 0"
    confianca = prediction if prediction >= 0.5 else 1 - prediction

    return classe, float(confianca)


def verificar_jogo_em_live(streamer, headers, base_url):
    try:
        user_resp = requests.get(f"{base_url}users?login={streamer}", headers=headers)
        user_data = user_resp.json().get("data", [])
        if not user_data:
            return None

        user_id = user_data[0]["id"]
        stream_resp = requests.get(f"{base_url}streams?user_id={user_id}", headers=headers)
        stream_data = stream_resp.json().get("data", [])
        if not stream_data:
            return None

        m3u8_url = f"https://usher.ttvnw.net/api/channel/hls/{streamer}.m3u8"
        temp_path = f"live_frame_{streamer}.jpg"

        if capturar_frame_ffmpeg_imageio(m3u8_url, temp_path, skip_seconds=5):
            classe, confianca = prever_jogo_em_frame(temp_path, modelo=None)
            categoria = stream_data[0].get("game_name", "")
            return f"{classe} ({confianca:.2%})", categoria

        return None
    except Exception as e:
        print(f"[Erro] verificar_jogo_em_live: {e}")
        return None


def varrer_url_customizada(url, st, session_state, prever_func, skip_inicial=0, intervalo=1000, max_frames=100000):
    resultados = []
    tempo_atual = skip_inicial

    st.write(f"🎬 Iniciando varredura personalizada em: {url}")
    st.write(f"🔁 Intervalo: {intervalo}s | Máximo de frames: {max_frames}")

    for i in range(max_frames):
        frame_path = f"frame_{tempo_atual}.jpg"
        st.write(f"📸 Tentando capturar frame no segundo {tempo_atual}...")

        if not capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=tempo_atual):
            st.warning(f"❌ Falha ao capturar frame no segundo {tempo_atual}. Interrompendo varredura.")
            break

        classe, confianca = prever_func(frame_path, session_state.get("modelo_ml"))
        st.write(f"🧠 {tempo_atual}s → {classe} (confiança: {confianca:.2%})")

        if confianca > 0.5:
            st.success(f"🎯 Jogo detectado: {classe} no segundo {tempo_atual} (confiança: {confianca:.2%})")
            resultados.append({
                "segundo": tempo_atual,
                "frame": frame_path,
                "jogo_detectado": classe
            })
        else:
            st.info(f"⏭️ Nenhum jogo com confiança suficiente ({confianca:.2%}) no segundo {tempo_atual}")

        tempo_atual += intervalo

    st.write(f"✅ Varredura finalizada. Total de jogos detectados: {len(resultados)}")
    return resultados


def buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers):
    todos_vods = []

    if dt_inicio.tzinfo is None:
        dt_inicio = dt_inicio.replace(tzinfo=timezone.utc)
    if dt_fim.tzinfo is None:
        dt_fim = dt_fim.replace(tzinfo=timezone.utc)

    for login in streamers:
        user_id = obter_user_id(login, headers)
        if not user_id:
            logging.warning(f"❌ User ID não encontrado para streamer: {login}")
            continue

        try:
            resp = requests.get(f"{base_url}videos?user_id={user_id}&type=archive&first=100", headers=headers)
            vods = resp.json().get("data", [])

            for vod in vods:
                created_at = vod.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

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
                    "id_vod": vod["id"],
                    "view_count": vod.get("view_count", 0)
                })

        except Exception as e:
            logging.error(f"Erro ao buscar VODs para {login}: {e}")

    return todos_vods


def obter_user_id(login, headers):
    url = f"https://api.twitch.tv/helix/users?login={login}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    return data["data"][0]["id"] if data.get("data") else None


def converter_duracao_para_segundos(dur_str):
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", dur_str)
    if not match:
        return 0
    h, m, s = match.groups(default="0")
    return int(h) * 3600 + int(m) * 60 + int(s)
