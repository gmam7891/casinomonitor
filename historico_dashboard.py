import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def exibir_dashboard_historico():
    st.title("📆 Histórico por Semana")

    arquivos = sorted(glob.glob("dados_semanais/semana_*.csv"))
    if not arquivos:
        st.warning("⚠️ Nenhum arquivo de semana encontrado na pasta 'dados_semanais/'.")
        return

    semana_escolhida = st.sidebar.selectbox("📅 Escolha a semana", arquivos)
    df = pd.read_csv(semana_escolhida)

    st.subheader(f"📊 Detecções da semana: {os.path.basename(semana_escolhida)}")
    st.dataframe(df)

    if "jogo_detectado" in df.columns:
        st.subheader("🎮 Distribuição de jogos na semana")
        st.bar_chart(df["jogo_detectado"].value_counts())

    if "streamer" in df.columns:
        st.subheader("🧑‍💻 Distribuição por streamer")
        st.bar_chart(df["streamer"].value_counts())
