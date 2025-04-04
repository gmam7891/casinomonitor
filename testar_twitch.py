import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Certifique-se de que seu .env está no mesmo diretório

CLIENT_ID = os.getenv("tyi97jgeg2j9r6pvakldbmdd2aie3m")
ACCESS_TOKEN = os.getenv("2806bhs736qdnj2fdm2nfc1u6e7oyt")

HEADERS_TWITCH = {
    "Client-ID": CLIENT_ID,
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

BASE_URL = "https://api.twitch.tv/helix/streams?first=1"

response = requests.get(BASE_URL, headers=HEADERS_TWITCH)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
