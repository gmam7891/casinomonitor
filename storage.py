
import os
import pandas as pd
from datetime import datetime

DADOS_DIR = "dados"
os.makedirs(DADOS_DIR, exist_ok=True)

def salvar_deteccao(tipo, dados):
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
    """Carrega um CSV salvo anteriormente no diretório 'dados'."""
    nome_arquivo = f"{DADOS_DIR}/{tipo}.csv"
    if os.path.exists(nome_arquivo):
        return pd.read_csv(nome_arquivo)
    else:
        return pd.DataFrame()

def limpar_historico(tipo):
    """Remove um CSV específico do diretório 'dados'."""
    nome_arquivo = f"{DADOS_DIR}/{tipo}.csv"
    if os.path.exists(nome_arquivo):
        os.remove(nome_arquivo)
