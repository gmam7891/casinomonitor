from twitch_api import get_user_data, get_stream_data, get_vods, get_game_name
from util import capturar_frame_ffmpeg, match_template_from_image, log
from modelo import carregar_modelo, prever_jogo_em_frame
from datetime import datetime
import os

modelo = carregar_modelo()

def verificar_jogo_em_live(streamer):
    try:
        user = get_user_data(streamer)
        if not user:
            return None
        stream = get_stream_data(user['id'])
        if not stream:
            return None
        m3u8_url = f"https://usher.ttvnw.net/api/channel/hls/{streamer}.m3u8"
        frame_path = f"temp_{streamer}.jpg"
        if capturar_frame_ffmpeg(m3u8_url, frame_path):
            jogo = prever_jogo_em_frame(modelo, frame_path) or match_template_from_image(frame_path)
            os.remove(frame_path)
            return jogo, get_game_name(stream['game_id'])
    except Exception as e:
        log(f"Erro: {e}")
    return None
