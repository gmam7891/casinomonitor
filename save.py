import pandas as pd
import os

def salvar_deteccao(data, streamer, provedor, jogo, screenshot):
    """
    Salva uma linha de detecção em um CSV local.
    """
    registro = [{
        "Data": data,
        "Streamer": streamer,
        "Provedor": provedor,
        "Jogo": jogo,
        "Screenshot": screenshot
    }]

    df_novo = pd.DataFrame(registro)

    caminho = "output/detecoes.csv"
    os.makedirs("output", exist_ok=True)

    if os.path.exists(caminho):
        df_existente = pd.read_csv(caminho)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(caminho, index=False)
    print(f"[INFO] Detecção salva em {caminho}")
