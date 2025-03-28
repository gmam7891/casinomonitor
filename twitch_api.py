import streamlit as st
import requests

CLIENT_ID = st.secrets["CLIENT_ID"]
ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]

HEADERS_TWITCH = {
    'Client-ID': CLIENT_ID,
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}
BASE_URL = "https://api.twitch.tv/helix"

def get_user_data(login):
    res = requests.get(f"{BASE_URL}/users?login={login}", headers=HEADERS_TWITCH)
    return res.json().get("data", [])[0] if res.ok else None

def get_stream_data(user_id):
    res = requests.get(f"{BASE_URL}/streams?user_id={user_id}", headers=HEADERS_TWITCH)
    return res.json().get("data", [])[0] if res.ok else None

def get_vods(user_id, limit=20):
    res = requests.get(f"{BASE_URL}/videos?user_id={user_id}&type=archive&first={limit}", headers=HEADERS_TWITCH)
    return res.json().get("data", []) if res.ok else []

def get_game_name(game_id):
    res = requests.get(f"{BASE_URL}/games?id={game_id}", headers=HEADERS_TWITCH)
    data = res.json().get("data", [])
    return data[0]['name'] if data else "Desconhecida"
