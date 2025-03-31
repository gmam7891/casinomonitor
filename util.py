import cv2
import os
import numpy as np
import logging
import subprocess
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf

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
