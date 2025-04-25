import numpy as np
import tensorflow as tf
import re
import streamlink
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess

# 🎯 Captura um frame da VOD no segundo especificado
def capturar_frame_ffmpeg_imageio(m3u8_url, segundo):
    ffmpeg_path = get_ffmpeg_exe()
    comando = [
        ffmpeg_path,
        "-ss", str(segundo),
        "-i", m3u8_url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]
    try:
        processo = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if processo.stdout:
            frame = np.frombuffer(processo.stdout, np.uint8)
            frame = frame.reshape((720, 1280, 3))  # ajuste se necessário
            return frame
    except Exception as e:
        print(f"[FFMPEG ERRO] Frame {segundo}s – {e}")
        return None

# 🧠 Realiza predição com o modelo carregado
def prever_jogo_em_frame(frame):
    modelo = tf.keras.models.load_model("modelo.h5")
    if modelo is None:
        print("[ERRO] Modelo não carregado.")
        return None

    frame = tf.image.resize(frame, (224, 224))
    frame = tf.cast(frame, tf.float32) / 255.0
    frame = tf.expand_dims(frame, axis=0)

    pred = modelo.predict(frame)[0]
    print(f"[DEBUG] Predição: {pred[0]}")
    return {
        "jogo_detectado": "pragmaticplay" if pred[0] > 0.5 else "outros",
        "confianca": float(pred[0])
    }

# 🔄 Processa um frame em tempo X e retorna a predição
def processar_frame(m3u8_url, tempo, session_state):
    frame = capturar_frame_ffmpeg_imageio(m3u8_url, segundo=tempo)
    if frame is not None:
        previsao = prever_jogo_em_frame(frame)
        if previsao:
            print(f"[{tempo}s] 🎰 Jogo detectado: {previsao['jogo_detectado']}")
            return {
                "segundo": tempo,
                **previsao,
                "frame": frame
            }
    else:
        print(f"[ERRO] Frame não capturado no segundo {tempo}")
    return None

# 🔗 Extrai a URL .m3u8 de uma VOD da Twitch
def obter_url_m3u8_twitch(url_vod):
    try:
        streams = streamlink.streams(url_vod)
        if "best" in streams:
            return streams["best"].url
        else:
            print("[ERRO] Não foi possível encontrar a qualidade 'best'")
    except Exception as e:
        print(f"[ERRO] Falha ao obter .m3u8 da VOD: {e}")
    return None

# ⏱️ Extrai os segundos do timestamp ?t= em uma VOD da Twitch
def extrair_segundos_da_url_vod(url):
    match = re.search(r"t=(\d+)h(\d+)m(\d+)s", url)
    if match:
        horas, minutos, segundos = map(int, match.groups())
        return horas * 3600 + minutos * 60 + segundos
    return 0

# 🔁 Varre múltiplas VODs com o modelo carregado
def varrer_vods_com_modelo(vods, st, session_state, prever_jogo_em_frame, capturar_frame, intervalo=20, max_frames=10):
    resultados = []
    for vod in vods:
        m3u8_url = obter_url_m3u8_twitch(vod["url"])
        if not m3u8_url:
            continue

        for i in range(max_frames):
            tempo = i * intervalo
            frame = capturar_frame(m3u8_url, segundo=tempo)
            if frame is None:
                continue

            resultado = prever_jogo_em_frame(frame)
            if resultado:
                resultado["streamer"] = vod.get("streamer", "")
                resultado["vod_url"] = vod["url"]
                resultado["segundo"] = tempo
                resultados.append(resultado)

    session_state["dados_periodo"] = resultados
    return resultados

import requests
from datetime import datetime

def buscar_vods_twitch_por_periodo(data_inicio, data_fim, headers, base_url, streamers):
    resultado = []

    for streamer in streamers:
        user_url = f"{base_url}/users?login={streamer}"
        resp_user = requests.get(user_url, headers=headers)
        if resp_user.status_code != 200:
            print(f"[ERRO] Não foi possível obter ID do streamer: {streamer}")
            continue

        user_data = resp_user.json()["data"]
        if not user_data:
            continue

        user_id = user_data[0]["id"]

        videos_url = f"{base_url}/videos?user_id={user_id}&type=archive&first=100"
        resp_videos = requests.get(videos_url, headers=headers)
        if resp_videos.status_code != 200:
            print(f"[ERRO] Não foi possível obter VODs de {streamer}")
            continue

        videos_data = resp_videos.json()["data"]

        for vod in videos_data:
            vod_created_at = datetime.fromisoformat(vod["created_at"].replace('Z', '+00:00'))

            if data_inicio <= vod_created_at <= data_fim:
                resultado.append({
                    "streamer": streamer,
                    "url": f"https://www.twitch.tv/videos/{vod['id']}",
                    "data": vod_created_at
                })

    return resultado

