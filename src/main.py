import cv2
import pandas as pd
from detect import carregar_tabela_provedores, detectar_jogo

# Carregar tabela de provedores/jogos
df_provedores = carregar_tabela_provedores()

# Simulação: abrir uma imagem de teste (substitua pelo seu frame real)
frame = cv2.imread("exemplo_frame.jpg")

if frame is None:
    print("🚫 Imagem não encontrada! Verifique o caminho.")
else:
    provedor, jogo = detectar_jogo(frame, df_provedores)

    if provedor:
        print(f"✅ Provedor detectado: {provedor} | Jogo: {jogo}")
    else:
        print("🚫 Nenhum jogo/provedor detectado neste frame.")
