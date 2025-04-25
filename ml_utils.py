import tensorflow as tf
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess

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
            # Ajuste a resolução conforme sua entrada real — aqui está como 1280x720
            frame = frame.reshape((720, 1280, 3))
            return frame
    except Exception as e:
        print(f"[FFMPEG ERRO] Frame {segundo}s – {e}")
        return None

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
