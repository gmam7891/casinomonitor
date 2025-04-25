import numpy as np
import tensorflow as tf
import re
import streamlink
import requests
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
from datetime import datetime

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
            frame = frame.reshape((720, 1280, 3))  # ajuste conforme sua VOD
            return frame
    except Exception as e:
        print(f"[FFMPEG ERRO] Frame {segundo}s – {e}")
        return None

# 🧠 Predição com modelo carregado
def prever_jogo_em_frame(frame):
    modelo = tf.keras.models.load_model("modelo.h5")
    if modelo is None:
        print("[ERRO] Modelo não carregado.")
        return None

    frame = tf.image.resize(frame, (224, 224))
    frame = tf.cast(frame, tf.float32) / 255.0
    frame = tf.expand_dims(frame, axis=0)

    pred = modelo.predict(frame)[0]
    return {
        "jogo_detectado": "pragmaticplay" if pred[0] > 0.5 else "outros",
        "confianca": float(pred[0])
    }

# 🔄 Processa um frame de um ponto do vídeo
def processar_frame(m3u8_url, tempo):
    frame = capturar_frame_ffmpeg_imageio(m3u8_url, segundo=tempo)
    if frame is not None:
        previsao = prever_jogo_em_frame(frame)
        if previsao:
            return {
                "segundo": tempo,
                **previsao
            }
    return None

# 🔗 Pega a .m3u8 a partir da VOD
def obter_url_m3u8_twitch(url_vod):
    try:
        streams = streamlink.streams(url_vod)
        if "best" in streams:
            return streams["best"].url
    except Exception as e:
        print(f"[ERRO] .m3u8: {e}")
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

# 📚 Busca VODs de vários streamers por período
def buscar_vods_twitch_por_periodo(data_inicio, data_fim, headers, base_url, streamers):
    resultado = []

    for streamer in streamers:
        user_url = f"{base_url}/users?login={streamer}"
        resp_user = requests.get(user_url, headers=headers)
        if resp_user.status_code != 200:
            print(f"[ERRO] Não foi possível obter ID do streamer: {streamer}")
            continue

        user_data = resp_user.json().get("data", [])
        if not user_data:
            continue

        user_id = user_data[0]["id"]

        videos_url = f"{base_url}/videos?user_id={user_id}&type=archive&first=100"
        resp_videos = requests.get(videos_url, headers=headers)
        if resp_videos.status_code != 200:
            print(f"[ERRO] Não foi possível obter VODs de {streamer}")
            continue

        videos_data = resp_videos.json().get("data", [])
        for vod in videos_data:
            vod_created_at = datetime.fromisoformat(vod["created_at"].replace('Z', '+00:00'))

            if data_inicio <= vod_created_at <= data_fim:
                resultado.append({
                    "streamer": streamer,
                    "url": f"https://www.twitch.tv/videos/{vod['id']}",
                    "data": vod_created_at
                })

    return resultado

# 🔍 Verifica se o streamer está ao vivo e o título da live
def verificar_jogo_em_live(streamer, headers, base_url):
    try:
        url = f"{base_url}/streams?user_login={streamer}"
        resp = requests.get(url, headers=headers)
        data = resp.json()

        if "data" in data and len(data["data"]) > 0:
            titulo = data["data"][0].get("title", "").lower()
            return {
                "ao_vivo": True,
                "titulo": titulo
            }
        return {"ao_vivo": False, "titulo": ""}
    except Exception as e:
        print(f"[ERRO] verificar_jogo_em_live: {e}")
        return {"ao_vivo": False, "titulo": ""}

# 📚 Busca VODs de UM streamer por período
def buscar_vods_por_streamer_e_periodo(streamer, data_inicio, data_fim, headers, base_url):
    resultado = []

    user_url = f"{base_url}/users?login={streamer}"
    resp_user = requests.get(user_url, headers=headers)
    if resp_user.status_code != 200:
        print(f"[ERRO] Não foi possível obter ID do streamer: {streamer}")
        return resultado

    user_data = resp_user.json().get("data", [])
    if not user_data:
        return resultado

    user_id = user_data[0]["id"]
    videos_url = f"{base_url}/videos?user_id={user_id}&type=archive&first=100"
    resp_videos = requests.get(videos_url, headers=headers)
    if resp_videos.status_code != 200:
        print(f"[ERRO] Não foi possível obter VODs do streamer: {streamer}")
        return resultado

    videos_data = resp_videos.json().get("data", [])
    for vod in videos_data:
        vod_created_at = datetime.fromisoformat(vod["created_at"].replace('Z', '+00:00'))

        if data_inicio <= vod_created_at <= data_fim:
            resultado.append({
                "streamer": streamer,
                "url": f"https://www.twitch.tv/videos/{vod['id']}",
                "data": vod_created_at
            })

    return resultado

def match_template_from_image(*args, **kwargs):
    print("[DEBUG] match_template_from_image chamado (mock)")
    return None

def varrer_url_customizada(*args, **kwargs):
    print("[DEBUG] varrer_url_customizada chamado (mock)")
    return []
