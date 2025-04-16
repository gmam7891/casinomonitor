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
