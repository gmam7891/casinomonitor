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
            return match_template_from_image(image_path)

        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = mobilenet_v2.preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        y_pred = modelo.predict(x)[0][0]
        print(f"🧠 Confiança da predição: {y_pred:.4f} | Threshold: {threshold} | Resultado: {'✅' if y_pred > threshold else '❌'}")

        return "Pragmatic Play (ML)" if y_pred > threshold else None
    except Exception as e:
        print(f"[Erro] prever_jogo_em_frame: {e}")
        return None

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

def capturar_frame_ffmpeg_imageio(url, output_path, skip_seconds=0):
    try:
        cmd = ["ffmpeg", "-i", url, "-ss", str(skip_seconds), "-frames:v", "1", "-q:v", "2", "-y", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ffmpeg erro:", result.stderr)
            return False
        return os.path.exists(output_path)
    except Exception as e:
        print(f"[Erro] capturar_frame_ffmpeg_imageio: {e}")
        return False

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
        frame_path = f"live_frame_{streamer}.jpg"

        if capturar_frame_ffmpeg_imageio(m3u8_url, frame_path, skip_seconds=5):
            jogo = prever_jogo_em_frame(frame_path)
            return jogo, stream_data[0].get("game_name"), stream_data[0].get("viewer_count")

        return None
    except Exception as e:
        print(f"[Erro] verificar_jogo_em_live: {e}")
        return None

def varrer_url_customizada(url, st, session_state, prever_func, skip_inicial=0, intervalo=1000, max_frames=10000):
    resultados = []
    tempo_atual = skip_inicial

    for _ in range(max_frames):
        frame_path = f"frame_{tempo_atual}.jpg"

        if not capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=tempo_atual):
            st.error(f"❌ Falha ao capturar frame no segundo {tempo_atual}.")
            break

        jogo = prever_func(frame_path, session_state.get("modelo_ml"))
        if jogo:
            resultados.append({
                "segundo": tempo_atual,
                "frame": frame_path,
                "jogo_detectado": jogo
            })

        tempo_atual += intervalo

    return resultados

def varrer_vods_com_modelo(dt_inicio, dt_fim, headers, base_url, streamers, session_state, prever_fn, intervalo=60):
    resultados = []
    vods = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers)

    modelo = session_state.get("modelo_ml")
    if modelo is None:
        print("❌ Modelo ML não carregado na sessão.")
        return []

    for vod in vods:
        dur = vod["duração_segundos"]
        url = vod["url"]
        for segundo in range(0, dur, intervalo):
            frame_path = f"vod_{vod['id_vod']}_{segundo}.jpg"
            if capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=segundo):
                jogo = prever_fn(frame_path, modelo)
                if jogo:
                    resultados.append({
                        "streamer": vod["streamer"],
                        "segundo": segundo,
                        "frame": frame_path,
                        "jogo_detectado": jogo,
                        "url": url
                    })
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
