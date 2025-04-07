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


def capturar_frame_ffmpeg_imageio(url, output_path, skip_seconds=0):
    try:
        cmd = [
            "ffmpeg",
            "-i", url,              # <- primeiro o input
            "-ss", str(skip_seconds),  # <- depois o seek (para .m3u8 é mais seguro)
            "-frames:v", "1",
            "-q:v", "2",
            "-y",
            output_path
        ]

        print(f"[FFMPEG] Rodando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ ffmpeg erro:", result.stderr)
            return False

        if not os.path.exists(output_path):
            print("❌ Frame não foi salvo.")
            return False

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


def prever_jogo_em_frame(image_path, modelo=None, threshold=0.4):
    try:
        if modelo is None:
            return match_template_from_image(image_path)

        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = mobilenet_v2.preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        y_pred = modelo.predict(x)[0][0]
        print(f"🧠 Confiança da predição: {y_pred:.4f} | Frame: {image_path}")

        return "Pragmatic Play (ML)" if y_pred > threshold else None
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

        stream_info = stream_data[0]  # <- separa pra facilitar
        m3u8_url = f"https://usher.ttvnw.net/api/channel/hls/{streamer}.m3u8"
        temp_path = f"live_frame_{streamer}.jpg"

        if capturar_frame_ffmpeg_imageio(m3u8_url, temp_path, skip_seconds=5):
            jogo = prever_jogo_em_frame(temp_path)
            categoria = stream_info.get("game_name", "")
            viewers = stream_info.get("viewer_count", 0)
            return jogo, categoria, viewers

        return None
    except Exception as e:
        print(f"[Erro] verificar_jogo_em_live: {e}")
        return None


def varrer_url_customizada(url, st, session_state, prever_func, skip_inicial=0, intervalo=1000, max_frames=10000):
    resultados = []
    tempo_atual = skip_inicial

    st.write(f"🎬 Iniciando varredura personalizada em: {url}")
    st.write(f"🔁 Intervalo: {intervalo}s | Máximo de frames: {max_frames}")

    for i in range(max_frames):
        frame_path = f"frame_{tempo_atual}.jpg"
        st.write(f"📸 Tentando capturar frame no segundo {tempo_atual}...")

        if not capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=tempo_atual):
            st.warning(f"❌ Falha ao capturar frame no segundo {tempo_atual}. Interrompendo varredura.")
            break

        jogo = prever_func(frame_path, session_state.get("modelo_ml"))
        if jogo:
            st.success(f"🎯 Jogo detectado: {jogo} no segundo {tempo_atual}")
            resultados.append({
                "segundo": tempo_atual,
                "frame": frame_path,
                "jogo_detectado": jogo
            })
        else:
            st.info(f"⏭️ Nenhum jogo detectado no segundo {tempo_atual}")

        tempo_atual += intervalo

    st.write(f"✅ Varredura finalizada. Total de jogos detectados: {len(resultados)}")
    return resultados



def varrer_vods_com_template(dt_inicio, dt_fim, headers, base_url, streamers, intervalo=60):
    resultados = []
    vods = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers)

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

    # Certificar-se de que as datas têm timezone
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


def varrer_vods_simples(dt_inicio, dt_fim, headers, base_url, streamers, intervalo=60):
    resultados = []
    vods = buscar_vods_twitch_por_periodo(dt_inicio, dt_fim, headers, base_url, streamers)

    for vod in vods:
        dur = vod["duração_segundos"]
        url = vod["url"]
        for segundo in range(0, dur, intervalo):
            frame_path = f"vod_completo_{vod['id_vod']}_{segundo}.jpg"
            sucesso = capturar_frame_ffmpeg_imageio(url, frame_path, skip_seconds=segundo)
            if sucesso:
                resultados.append({
                    "streamer": vod["streamer"],
                    "segundo": segundo,
                    "frame": frame_path,
                    "url": url
                })
    return resultados


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


def treinar_modelo(st, base_path="dataset", model_path="modelo/modelo_pragmatic.keras", epochs=5):
    try:
        st.markdown("### 🔄 Iniciando treinamento do modelo...")

        subdirs = os.listdir(base_path)
        if not subdirs or len(subdirs) < 2:
            st.error("❌ O diretório 'dataset/' deve conter pelo menos 2 subpastas com classes diferentes.")
            return

        st.info(f"📁 Classes detectadas: `{', '.join(subdirs)}`")

        datagen = ImageDataGenerator(
            validation_split=0.2,
            preprocessing_function=mobilenet_v2.preprocess_input
        )

        train_gen = datagen.flow_from_directory(
            base_path, target_size=(224, 224), batch_size=32,
            class_mode='binary', subset='training', shuffle=True
        )

        val_gen = datagen.flow_from_directory(
            base_path, target_size=(224, 224), batch_size=32,
            class_mode='binary', subset='validation', shuffle=False
        )

        class_counts = Counter(train_gen.classes)
        st.write("📊 Distribuição das classes no treino:", dict(class_counts))

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        total = sum(class_counts.values())
        class_weight = {
            0: total / (2.0 * class_counts[0]),
            1: total / (2.0 * class_counts[1])
        }

        st.write("⚖️ Pesos de classe aplicados:", class_weight)

        st.markdown("### ⏳ Treinando modelo...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weight,
            verbose=1
        )

        model.save(model_path)
        st.success("✅ Modelo treinado e salvo com sucesso!")

        st.markdown("### 📊 Curvas de Aprendizado")
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(history.history['loss'], label='Treino')
        axs[0].plot(history.history['val_loss'], label='Validação')
        axs[0].set_title('Loss por Época')
        axs[0].set_xlabel('Época')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(history.history['accuracy'], label='Treino')
        axs[1].plot(history.history['val_accuracy'], label='Validação')
        axs[1].set_title('Acurácia por Época')
        axs[1].set_xlabel('Época')
        axs[1].set_ylabel('Acurácia')
        axs[1].legend()

        st.pyplot(fig)
        return True

    except Exception:
        st.error("❌ Erro durante o treinamento:")
        st.code(traceback.format_exc())
        return False
