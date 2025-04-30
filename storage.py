
import os
import pandas as pd
from datetime import datetime

DADOS_DIR = "dados"
os.makedirs(DADOS_DIR, exist_ok=True)

def salvar_deteccao(tipo, dados):
    try:
        output_dir = "resultados"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"resultados_{tipo}.csv")

        df_novo = pd.DataFrame(dados)
        df_novo["hora_inferencia_brasilia"] = (datetime.utcnow() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')

        if os.path.exists(output_path):
            df_existente = pd.read_csv(output_path)

            # Verificação de duplicatas: mesmo streamer + vod_url (ou título e data, se não tiver URL)
            colunas_chave = ['streamer', 'url'] if 'url' in df_novo.columns else ['streamer', 'data']

            df_total = pd.concat([df_existente, df_novo], ignore_index=True)
            df_total.drop_duplicates(subset=colunas_chave, keep='first', inplace=True)
        else:
            df_total = df_novo

        df_total.to_csv(output_path, index=False)
        print(f"✅ Resultados atualizados em: {output_path}")

    except Exception as e:
        print(f"❌ Erro ao salvar detecção: {e}")


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

def salvar_deteccao(tipo, resultados):
    try:
        # 1. Corrigir horário para Brasília
        df = pd.DataFrame(resultados)

        # 2. Adicionar horário de Brasília, se ainda não existir
        if 'hora_inferencia_brasilia' not in df.columns:
            df['hora_inferencia_brasilia'] = (datetime.utcnow() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')

        # 3. Atualizar resultados corrigidos
        resultados_corrigidos = df.to_dict(orient='records')

        # 4. Salvar CSV
        output_dir = "resultados"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"resultados_{tipo}.csv")
        df.to_csv(output_path, index=False)

        print(f"✅ Resultados salvos no arquivo: {output_path}")

    except Exception as e:
        print(f"❌ Erro ao salvar detecção: {e}")
