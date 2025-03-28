import os
import subprocess
import shutil
from datetime import datetime
import cv2

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def create_dirs():
    for d in ["templates", "modelo", "dataset"]:
        os.makedirs(d, exist_ok=True)

def capturar_frame_ffmpeg(m3u8_url, output_path="frame.jpg", skip_seconds=10):
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(skip_seconds), "-i", m3u8_url,
            "-vf", "scale=1280:720", "-vframes", "1", "-q:v", "2", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        log(f"Erro ao capturar frame: {e}")
        return None

def match_template_from_image(image_path, template_path="templates/pragmaticplay.png", limiar=0.7):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_path, 0)
        if template is None:
            return None
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return "pragmaticplay" if max_val >= limiar else None
    except Exception as e:
        log(f"Erro no template matching: {e}")
        return None
