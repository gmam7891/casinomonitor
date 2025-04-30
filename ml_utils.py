import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image

# Caminho onde os templates estão armazenados
TEMPLATES_DIR = "templates"

# Carrega todos os templates .png do diretório
def carregar_templates():
    templates = {}
    for filename in os.listdir(TEMPLATES_DIR):
        if filename.endswith(".png"):
            nome_jogo = filename.replace(".png", "").lower()
            path = os.path.join(TEMPLATES_DIR, filename)
            templates[nome_jogo] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates

# Verifica se algum template bate com o frame
def match_template_from_image(frame, templates, threshold=0.7):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for jogo, template in templates.items():
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return jogo
    return None

# Usa FFmpeg (via imageio-ffmpeg) para capturar frame
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

# Modelo de IA para prever o jogo a partir de um frame
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

# Carrega modelo treinado (Keras)
def carregar_modelo(path="modelo/modelo_pragmatic.keras"):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

# Busca VODs da Twitch por streamer e período
def buscar_vods_por_streamer_e_periodo(streamer, inicio, fim):
    # Isso depende de como você está integrando com a Twitch API
    # Suponha que já está implementado em outro lugar
    return []

# Varrer um VOD customizado
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

# Detecção com modelo de IA
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
