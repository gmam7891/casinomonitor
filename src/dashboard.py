import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import os

DB_PATH = os.path.join("output", "monitor.db")

# Função para carregar as detecções do banco
@st.cache_data
def carregar_deteccoes():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM Deteccoes"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Carregar dados
df = carregar_deteccoes()

st.title("🎰 CasinoMonitor Dashboard — Multi-Provedor")

# Verifica se há dados
if df.empty:
    st.warning("Nenhuma detecção encontrada. Execute o pipeline de varredura primeiro.")
    st.stop()

# Filtros
st.sidebar.header("🔍 Filtros")

provedores = ["Todos"] + sorted(df['provedor'].dropna().unique().tolist())
provedor_selecionado = st.sidebar.selectbox("Provedor", provedores)

jogos = ["Todos"] + sorted(df['jogo'].dropna().unique().tolist())
jogo_selecionado = st.sidebar.selectbox("Jogo", jogos)

streamers = ["Todos"] + sorted(df['streamer'].dropna().unique().tolist())
streamer_selecionado = st.sidebar.selectbox("Streamer", streamers)

# Filtragem
df_filtrado = df.copy()

if provedor_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['provedor'] == provedor_selecionado]

if jogo_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['jogo'] == jogo_selecionado]

if streamer_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['streamer'] == streamer_selecionado]

# KPIs
col1, col2 = st.columns(2)

col1.metric("📅 Total Detecções", len(df_filtrado))
col2.metric("👀 Total Streamers", df_filtrado['streamer'].nunique())

# Tabela Detalhada
st.subheader("📑 Detecções Filtradas")
st.dataframe(df_filtrado)

# Gráfico: Top Jogos
st.subheader("🏆 Top Jogos Detectados")

top_jogos = (
    df_filtrado.groupby(['provedor', 'jogo'])
    .size()
    .reset_index(name='Total')
    .sort_values(by='Total', ascending=False)
    .head(10)
)

fig_jogos = px.bar(
    top_jogos,
    x='Total',
    y='jogo',
    color='provedor',
    orientation='h',
    title="Top Jogos por Detecções",
    height=400
)
st.plotly_chart(fig_jogos, use_container_width=True)

# Gráfico: Share of Voice por Provedor
st.subheader("📊 Share of Voice por Provedor")

sov = (
    df_filtrado.groupby('provedor')
    .size()
    .reset_index(name='Total')
    .sort_values(by='Total', ascending=False)
)

fig_sov = px.pie(
    sov,
    names='provedor',
    values='Total',
    title="Distribuição de Detecções por Provedor"
)
st.plotly_chart(fig_sov, use_container_width=True)

# Rodapé
st.caption("CasinoMonitor © 2025 | Dados em tempo real do seu monitor multi-provedor 🎰")
