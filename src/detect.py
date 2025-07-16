import os
import json
import pandas as pd
import pytesseract
import cv2


def carregar_tabela_provedores():
    """
    Carrega a tabela mestre de provedores e jogos do arquivo JSON.
    Fallback: tenta CSV se JSON não existir.
    """
    caminho_json = os.path.join("data", "provedores_jogos.json")
    caminho_csv = os.path.join("data", "provedores_jogos.csv")

    try:
        with open(caminho_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            print(f"[INFO] JSON carregado com {len(df)} registros.")
            return df
    except Exception as e:
        print(f"[WARNING] Erro ao carregar JSON: {e}")
        if os.path.exists(caminho_csv):
            df = pd.read_csv(caminho_csv)
            print(f"[INFO] CSV backup carregado com {len(df)} registros.")
            return df
        else:
            print("[ERROR] Nenhuma tabela de provedores encontrada!")
            return pd.DataFrame(columns=["Provedor", "Jogo", "Aliases"])


def detectar_jogo(frame, df_provedores):
    """
    Recebe um frame (numpy array) e o DataFrame de provedores.
    Retorna Provedor e Jogo detectados, ou None se não encontrar.
    """
    # Converter para escala de cinza para melhorar OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rodar OCR
    texto_extraido = pytesseract.image_to_string(gray).lower()

    print(f"[OCR] Texto extraído: {texto_extraido}")

    # Comparar com aliases
    for _, row in df_provedores.iterrows():
        for alias in row['Aliases']:
            if alias.lower() in texto_extraido:
                return row['Provedor'], row['Jogo']

    return None, None
