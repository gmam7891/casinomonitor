import os
import requests
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

client_id = os.getenv("TWITCH_CLIENT_ID")
client_secret = os.getenv("TWITCH_CLIENT_SECRET")

if not client_id or not client_secret:
    print("❌ TWITCH_CLIENT_ID ou TWITCH_CLIENT_SECRET não estão definidos no .env")
    exit()

# URL de autenticação da Twitch
auth_url = "https://id.twitch.tv/oauth2/token"

# Requisição POST para gerar o access_token
params = {
    "client_id": client_id,
    "client_secret": client_secret,
    "grant_type": "client_credentials"
}

response = requests.post(auth_url, params=params)

if response.status_code == 200:
    data = response.json()
    access_token = data.get("access_token")
    print("✅ Access Token gerado com sucesso:")
    print(access_token)

    # Salva no .env (opcional)
    with open(".env", "a") as f:
        f.write(f"\nTWITCH_ACCESS_TOKEN={access_token}\n")
    print("✅ Token adicionado ao .env com sucesso.")
else:
    print("❌ Erro ao gerar token:")
    print(response.text)
