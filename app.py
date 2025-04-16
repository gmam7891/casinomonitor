import sys
import os
sys.path.append(os.path.dirname(__file__))

from datetime import datetime, timedelta, timezone
import streamlit as st
import pandas as pd
import logging
import requests
from dotenv import load_dotenv
load_dotenv()

from storage import salvar_deteccao

# ---------------- OBTER ACCESS TOKEN DA TWITCH ----------------
def obter_access_token(client_id, client_secret):
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    try:
        resp = requests.post(url, data=data)
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        st.error("Erro ao obter access_token:")
        st.code(str(e))
        st.stop()

# ---------------- CONFIG ----------------
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")
ACCESS_TOKEN = obter_access_token(CLIENT_ID, CLIENT_SECRET)

HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}
BASE_URL_TWITCH = 'https://api.twitch.tv/helix/'

# ---------------- OBTÉM USER ID ----------------
def obter_user_id(login):
    url = f"{BASE_URL_TWITCH}users?login={login}"
    resp = requests.get(url, headers=HEADERS_TWITCH)
    data = resp.json()
    return data["data"][0]["id"] if data.get("data") else None

# ---------------- CONVERTE DURAÇÃO ----------------
def converter_duracao_para_segundos(dur_str):
    import re
    match = re.match(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", dur_str)
    if not match:
        return 0
    h, m, s = match.groups(default="0")
    return int(h) * 3600 + int(m) * 60 + int(s)

# ---------------- BUSCA VODS DIRETO API ----------------
def buscar_vods_por_streamer_e_periodo(streamer, data_inicio, data_fim):
    todos_vods = []

    if data_inicio.tzinfo is None:
        data_inicio = data_inicio.replace(tzinfo=timezone.utc)
    if data_fim.tzinfo is None:
        data_fim = data_fim.replace(tzinfo=timezone.utc)

    user_id = obter_user_id(streamer)
    if not user_id:
        st.warning(f"Streamer {streamer} não encontrado na API da Twitch.")
        return []

    try:
        url = f"{BASE_URL_TWITCH}videos?user_id={user_id}&type=archive&first=100"
        resp = requests.get(url, headers=HEADERS_TWITCH)
        vods = resp.json().get("data", [])

        for vod in vods:
            created_at = vod.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

            if not (data_inicio <= created_at <= data_fim):
                continue

            dur = converter_duracao_para_segundos(vod["duration"])

            todos_vods.append({
                "streamer": streamer,
                "titulo": vod["title"],
                "url": vod["url"],
                "data": created_at,
                "duração_segundos": dur,
                "duração_raw": vod["duration"],
                "id_vod": vod["id"],
                "view_count": vod.get("view_count", 0)
            })

    except Exception as e:
        logging.error(f"Erro ao buscar VODs para {streamer}: {e}")

    return todos_vods

# ---------------- ANALISAR POR PERÍODO ----------------
def analisar_por_periodo(streamer, vods, st, session_state, prever_jogo_fn, varrer_fn, obter_m3u8_fn):
    st.write("🛠️ Rodando análise por período")
    st.write("🔎 VODs recebidas:", vods)
    resultados_finais = []

    for vod in vods:
        m3u8_url = obter_m3u8_fn(vod["url"])
        if not m3u8_url:
            continue

        resultado = varrer_fn(
            m3u8_url, st, session_state, prever_jogo_fn,
            skip_inicial=0, intervalo=120, max_frames=6
        )

        if resultado:
            for r in resultado:
                r["streamer"] = streamer
            resultados_finais.extend(resultado)

    return resultados_finais

# ---------------- STREAMLIT UI ----------------
st.title("🎥 VODs via API Twitch")
streamers = st.text_area("Streamers (um por linha)").splitlines()
data_inicio = st.date_input("Data início", value=datetime.today() - timedelta(days=7))
data_fim = st.date_input("Data fim", value=datetime.today())

if st.button("🔍 Buscar VODs"):
    if not streamers:
        st.warning("Adicione pelo menos um streamer.")
    else:
        with st.spinner("Buscando VODs..."):
            resultados = []
            for s in streamers:
                resultados.extend(buscar_vods_por_streamer_e_periodo(s, data_inicio, data_fim))

        if resultados:
            st.success(f"{len(resultados)} VODs encontrados.")
            df = pd.DataFrame(resultados)
            st.dataframe(df)
            salvar_deteccao("vods", resultados)
        else:
            st.warning("Nenhuma VOD encontrada.")
