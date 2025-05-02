import streamlit as st
import matplotlib.pyplot as plt

def exibir_dashboard_cluster(perfil_df, resumo_clusters_df):
    st.title("📊 Clusterização de Streamers")

    cluster_id = st.sidebar.selectbox("Selecione o cluster", sorted(perfil_df["cluster"].unique()))

    st.subheader("🧠 Visualização dos clusters")
    fig, ax = plt.subplots()
    for c in sorted(perfil_df["cluster"].unique()):
        grupo = perfil_df[perfil_df["cluster"] == c]
        ax.scatter(grupo["pca_x"], grupo["pca_y"], label=f"Cluster {c}")
        for _, row in grupo.iterrows():
            ax.text(row["pca_x"] + 0.02, row["pca_y"], row["streamer"], fontsize=8)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Distribuição dos Streamers por Cluster")
    ax.legend()
    st.pyplot(fig)

    st.subheader(f"📋 Streamers do Cluster {cluster_id}")
    st.dataframe(perfil_df[perfil_df["cluster"] == cluster_id][[
        "streamer", "%PP", "total_frames", "lives_detectadas", "media_segundos_por_live"
    ]].reset_index(drop=True))

    st.subheader("📌 Resumo dos clusters")
    st.dataframe(resumo_clusters_df)
