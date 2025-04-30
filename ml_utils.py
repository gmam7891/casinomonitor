import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image

TEMPLATES_DIR = "templates"

def carregar_templates():
    templates = {}
    for filename in os.listdir(TEMPLATES_DIR):
        if filename.endswith(".png"):
            nome_jogo = filename.replace(".png", "").lower()
            path = os.path.join(TEMPLATES_DIR, filename)
            templates[nome_jogo] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates

def match_template_from_image(frame, templates, threshold=0.7):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for jogo, template in templates.items():
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return jogo
    return None

def capturar_frame_ffmpeg_imageio(m3u8_url, tempo_seg):
    import imageio_ffmpeg as ffmpeg
    import subprocess
    import imageio
    import io

    try:
        command = [
            "ffmpeg",
            "-ss", str(tempo_seg),
            "-i", m3u8_url,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        image_bytes = result.stdout
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"Erro ao capturar frame com ffmpeg: {e}")
        return None

def prever_jogo_em_frame(frame, model):
    try:
        imagem = cv2.resize(frame, (224, 224))
        imagem = imagem / 255.0
        imagem = np.expand_dims(imagem, axis=0)
        pred = model.predict(imagem)
        classe_idx = np.argmax(pred)
        return model.class_names[classe_idx]
    except Exception as e:
        print(f"Erro na predição do jogo: {e}")
        return None

def carregar_modelo(path="modelo/modelo_pragmatic.keras"):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def buscar_vods_por_streamer_e_periodo(streamer, inicio, fim):
    return []

def varrer_url_customizada(m3u8_url, intervalo=60, max_frames=10, templates=None):
    resultados = []
    for i in range(max_frames):
        tempo = i * intervalo
        frame = capturar_frame_ffmpeg_imageio(m3u8_url, tempo)
        if frame is not None:
            jogo = match_template_from_image(frame, templates)
            if jogo:
                resultados.append({
                    "tempo": tempo,
                    "jogo_detectado": jogo
                })
    return resultados

def varrer_vods_com_modelo(m3u8_url, intervalo=60, max_frames=10, model=None):
    resultados = []
    for i in range(max_frames):
        tempo = i * intervalo
        frame = capturar_frame_ffmpeg_imageio(m3u8_url, tempo)
        if frame is not None:
            jogo = prever_jogo_em_frame(frame, model)
            if jogo:
                resultados.append({
                    "tempo": tempo,
                    "jogo_detectado": jogo
                })
    return resultados

def varrer_url_combinada(m3u8_url, modelo, templates, intervalo=60, max_frames=60, skip_inicial=0):
    resultados = []
    for i in range(max_frames):
        tempo = skip_inicial + i * intervalo
        frame = capturar_frame_ffmpeg_imageio(m3u8_url, tempo)
        if frame is None:
            continue

        jogo = prever_jogo_em_frame(frame, modelo)
        if not jogo:
            jogo = match_template_from_image(frame, templates)

        if jogo:
            resultados.append({
                "segundo": tempo,
                "jogo_detectado": jogo,
                "frame": frame
            })

    return resultados
