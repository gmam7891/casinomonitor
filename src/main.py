from db import criar_tabelas, popular_provedores, salvar_deteccao_db, top_jogos
from detect import carregar_tabela_provedores, detectar_jogo
import cv2
import datetime

# Etapa 1: criar tabelas
criar_tabelas()

# Etapa 2: popular provedores uma vez (roda só na primeira vez)
popular_provedores()

# Etapa 3: simulação de detecção
df_provedores = carregar_tabela_provedores()
frame = cv2.imread("exemplo_frame.jpg")

if frame is not None:
    provedor, jogo = detectar_jogo(frame, df_provedores)
    if provedor:
        salvar_deteccao_db(
            data=datetime.date.today().isoformat(),
            streamer="Streamer_X",
            provedor=provedor,
            jogo=jogo,
            viewers=1234,
            screenshot="frame001.jpg"
        )

# Etapa 4: consulta Top Jogos
print("\n🎯 Top Jogos:")
for r in top_jogos():
    print(r)
