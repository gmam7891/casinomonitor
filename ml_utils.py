import cv2
import numpy as np
from PIL import Image
import imageio_ffmpeg as ffmpeg
import os
import requests
from datetime import datetime

def capturar_frame_ffmpeg_imageio(url, output_path, skip_seconds=0):
    try:
        w, h = 1280, 720
        cmd = [
            "-ss", str(skip_seconds),
            "-i", url,
            "-frames:v", "1",
            "-vf", f"scale={w}:{h}",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", "-"
        ]
        process = ffmpeg.read_frames(cmd, size=(w, h))
        frame = next(process)

        image = Image.fromarray(frame)
        image.save(output_path)
        return True
    except Exception as e:
        print(f"[Erro] capturar_frame_ffmpeg_imageio: {e}")
        return False

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

def prever_jogo_em_frame(image_path, modelo=None):
    try:
        if modelo is None:
            return match_template_from_image(image_path)

        from tensorflow.keras.preprocessing import image as keras_image
        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        y_pred = modelo.predict(x)[0][0]
        return "Pragmatic Play (ML)" if y_pred > 0.5 else None
    except Exception as e:
        print(f"[Erro] prever_jogo_em_frame: {e}")
        return None

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
        temp_path = f"live_frame_{streamer}.jpg"
        if capturar_frame_ffmpeg_imageio(m3u8_url, temp_path, skip_seconds=5):
            jogo = prever_jogo_em_frame(temp_path)
            categoria = stream_data[0].get("game_name", "")
            return (jogo, categoria)
        return None
    except Exception as e:
        print(f"[Erro] verificar_jogo_em_live: {e}")
        return None

def varrer_url_customizada(url, st, session_state, prever_func, skip_inicial=0, intervalo=1000, max_frames=10000):
    resultados = []
    tempo_atual = skip_inicial

    for _ in range(max_frames):
        frame_path = f"frame_{tempo_atual}.jpg"
        sucesso = capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=tempo_atual)
        if not sucesso:
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

def varrer_vods_com_template(dt_inicio, dt_fim, headers, base_url, streamers, intervalo=60):
    from app import buscar_vods_twitch_por_periodo  # cuidado com circular import se mover código
    resultados = []
    vods = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, streamers)
    for vod in vods:
        dur = vod["duração_segundos"]
        url = vod["url"]
        for segundo in range(0, dur, intervalo):
            frame_path = f"vod_{vod['id_vod']}_{segundo}.jpg"
            if capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=segundo):
                jogo = match_template_from_image(frame_path)
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

    for login in streamers:
        user_id = obter_user_id(login, headers)
        if not user_id:
            logging.warning(f"❌ User ID não encontrado para streamer: {login}")
            continue

        url = f"{base_url}videos?user_id={user_id}&type=archive&first=100"
        try:
            resp = requests.get(url, headers=headers)
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
                    "id_vod": vod["id"]
                })

        except Exception as e:
            logging.error(f"Erro ao buscar VODs para {login}: {e}")

    return todos_vods
