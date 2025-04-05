# storage.py

import pandas as pd
import os
from datetime import datetime

DADOS_DIR = "dados"
os.makedirs(DADOS_DIR, exist_ok=True)

def salvar_deteccao(tipo, dados):
    """Salva dados de detecção (lives, vods, varredura, etc.)"""
    nome_arquivo = f"{DADOS_DIR}/{tipo}.csv"
    df_novo = pd.DataFrame(dados)
    df_novo["data_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(nome_arquivo):
        df_existente = pd.read_csv(nome_arquivo)
        df = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df = df_novo

    df.to_csv(nome_arquivo, index=False)

def carregar_historico(tipo):
    """Carrega o histórico salvo para o tipo de dado"""
    nome_arquivo = f"{DADOS_DIR}/{tipo}.csv"
    if os.path.exists(nome_arquivo):
        return pd.read_csv(nome_arquivo)
    return pd.DataFrame()

def limpar_historico(tipo):
    """Remove os dados salvos do tipo especificado"""
    nome_arquivo = f"{DADOS_DIR}/{tipo}.csv"
    if os.path.exists(nome_arquivo):
        os.remove(nome_arquivo)

def limpar_todos_historicos():
    """Remove todos os arquivos de histórico salvos"""
    for arquivo in os.listdir(DADOS_DIR):
        if arquivo.endswith(".csv"):
            os.remove(os.path.join(DADOS_DIR, arquivo))
