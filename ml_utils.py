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
import imageio_ffmpeg as ffmpeg
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import models
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import streamlit as st  # necessário para chamadas como st.write, st.warning
from concurrent.futures import ThreadPoolExecutor


def extrair_segundos_da_url_vod(url):
    match = re.search(r"[?&]t=(\d+h)?(\d+m)?(\d+s)?", url)
    if not match:
        return 0
    horas = int(match.group(1)[:-1]) if match.group(1) else 0
    minutos = int(match.group(2)[:-1]) if match.group(2) else 0
    segundos = int(match.group(3)[:-1]) if match.group(3) else 0
    return horas * 3600 + minutos * 60 + segundos


def obter_user_id(login, headers):
    url = f"https://api.twitch.tv/helix/users?login={login}"
    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        return data["data"][0]["id"] if data.get("data") else None
    except Exception as e:
        logging.error(f"Erro ao obter user_id para {login}: {e}")
        return None


def buscar_vods_por_streamer_e_periodo(streamer, data_inicio, data_fim, headers, base_url):
    todos_vods = []

    if not isinstance(data_inicio, datetime):
        data_inicio = pd.to_datetime(data_inicio)
    if not isinstance(data_fim, datetime):
        data_fim = pd.to_datetime(data_fim)

    if data_inicio.tzinfo is None:
        data_inicio = data_inicio.replace(tzinfo=timezone.utc)
    if data_fim.tzinfo is None:
        data_fim = data_fim.replace(tzinfo=timezone.utc)

    user_id = obter_user_id(streamer, headers)
    if not user_id:
        logging.warning(f"Streamer {streamer} não encontrado na API da Twitch.")
        return []

    try:
        url = f"{base_url}videos?user_id={user_id}&type=archive&first=100"
        resp = requests.get(url, headers=headers)
        vods = resp.json().get("data", [])

        for vod in vods:
            created_at = datetime.fromisoformat(vod["created_at"].replace("Z", "+00:00"))
            if not (data_inicio <= created_at <= data_fim):
                continue

            dur = converter_duracao_para_segundos(vod["duration"])

            todos_vods.append({
                "streamer": streamer,
                "titulo": vod["title"],
                "url": vod["url"],
                "data": created_at,
                "duração_segundos": dur,
                "duração_raw": vod["duration"],
                "id_vod": vod["id"],
                "view_count": vod.get("view_count", 0)
            })

    except Exception as e:
        logging.error(f"Erro ao buscar VODs para {streamer}: {e}")

    return todos_vods


def analisar_por_periodo(streamer, vods, st, session_state, prever_jogo_em_frame, varrer_url_customizada_paralela, obter_url_m3u8_twitch):
    st.write("🛠️ Rodando análise por período")
    st.write("🔎 VODs recebidas:", vods)
    resultados_finais = []

    for vod in vods:
        m3u8_url = obter_url_m3u8_twitch(vod["url"])
        if not m3u8_url:
            continue

        resultado = varrer_url_customizada_paralela(
            m3u8_url, st, session_state, prever_jogo_em_frame,
            skip_inicial=0, intervalo=120, max_frames=6
        )

        if resultado:
            for r in resultado:
                r["streamer"] = streamer
            resultados_finais.extend(resultado)

    return resultados_finais


def prever_jogo_em_frame(image_path, modelo=None, threshold=0.4):
    try:
        if modelo is None:
            resultado = match_template_from_image(image_path)
            return resultado, None

        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = mobilenet_v2.preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        y_pred = modelo.predict(x)[0][0]
        resultado = "Pragmatic Play (ML)" if y_pred > threshold else None
        return resultado, float(y_pred)
    except Exception as e:
        print(f"[Erro] prever_jogo_em_frame: {e}")
        return None, None


def match_template_from_image(image_path, templates_dir="templates/", threshold=0.8):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for template_file in os.listdir(templates_dir):
            template_path = os.path.join(templates_dir, template_file)
            template = cv2.imread(template_path, 0)
            if template is None:
                continue

            res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val >= threshold:
                return os.path.splitext(template_file)[0]

        return None
    except Exception as e:
        print(f"[Erro] match_template_from_image: {e}")
        return None


def capturar_frame_ffmpeg_imageio(m3u8_url, output_path, skip_seconds=0):
    try:
        output_options = [
            "-ss", str(skip_seconds),
            "-i", m3u8_url,
            "-frames:v", "1",
            "-q:v", "2",
            output_path
        ]

        ffmpeg_binary = ffmpeg.get_ffmpeg_exe()
        comando = [ffmpeg_binary] + output_options

        result = subprocess.run(comando, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Erro capturando frame: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"[Erro] capturar_frame_ffmpeg_imageio: {e}")
        return False


def verificar_jogo_em_live(streamer, headers, base_url):
    try:
        url = f"{base_url}streams?user_login={streamer}"
        resp = requests.get(url, headers=headers)
        data = resp.json().get("data", [])
        if data:
            stream = data[0]
            game_name = stream.get("game_name", "")
            category = stream.get("game_id", "")
            viewers = stream.get("viewer_count", 0)
            return game_name, category, viewers
        else:
            return None
    except Exception as e:
        print(f"[Erro] verificar_jogo_em_live: {e}")
        return None


def varrer_url_customizada(m3u8_url, st, session_state, prever_jogo_fn, skip_inicial=0, intervalo=60, max_frames=5):
    resultados = []
    for i in range(max_frames):
        tempo = skip_inicial + i * intervalo
        frame_path = f"frame_{tempo}.jpg"
        sucesso = capturar_frame_ffmpeg_imageio(m3u8_url, frame_path, skip_seconds=tempo)
        if sucesso:
            resultado, confianca = prever_jogo_fn(frame_path, session_state.get("modelo_ml"))
            if resultado:
                resultados.append({
                    "segundo": tempo,
                    "frame": frame_path,
                    "jogo_detectado": resultado,
                    "confianca": confianca
                })
    return resultados


def varrer_vods_com_modelo(dt_inicio, dt_fim, headers, base_url, streamers, session_state, prever_jogo_fn):
    resultados = []
    for streamer in streamers:
        vods = buscar_vods_por_streamer_e_periodo(streamer, dt_inicio, dt_fim, headers, base_url)
        for vod in vods:
            m3u8_url = obter_url_m3u8_twitch(vod["url"])
            if not m3u8_url:
                continue
            res = varrer_url_customizada(m3u8_url, st, session_state, prever_jogo_fn)
            for r in res:
                r["streamer"] = streamer
                r["url"] = vod["url"]
            resultados.extend(res)
    return resultados


def buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers):
    todos_vods = []
    for streamer in streamers:
        vods = buscar_vods_por_streamer_e_periodo(streamer, dt_inicio, dt_fim, headers, base_url)
        todos_vods.extend(vods)
    return todos_vods


def converter_duracao_para_segundos(duracao_str):
    h, m, s = 0, 0, 0
    if 'h' in duracao_str:
        h = int(duracao_str.split('h')[0])
        duracao_str = duracao_str.split('h')[1]
    if 'm' in duracao_str:
        m = int(duracao_str.split('m')[0])
        duracao_str = duracao_str.split('m')[1]
    if 's' in duracao_str:
        s = int(duracao_str.split('s')[0])
    return h * 3600 + m * 60 + s


def obter_url_m3u8_twitch(vod_url):
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
