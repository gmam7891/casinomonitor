import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def carregar_dados_simulados():
    streamers = ["tecnosh", "dona", "yuuri22", "smzinho", "minerva", "piuzinho", "jukes", "ookina"]
    data = []
    for i, streamer in enumerate(streamers):
        total = 100 + i * 10
        pp = int(total * (0.7 if i < 3 else 0.2))
        lives = 5 + (i % 3)
        segundos = lives * (300 + i * 20)
        data.append({
            "streamer": streamer,
            "total_frames": total,
            "frames_PP": pp,
            "%PP": pp / total,
            "lives_detectadas": lives,
            "media_segundos_por_live": segundos / lives
        })
    return pd.DataFrame(data)

def clusterizar_streamers(perfil_df, n_clusters=3):
    features = ["total_frames", "frames_PP", "%PP", "lives_detectadas", "media_segundos_por_live"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(perfil_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    perfil_df["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    perfil_df["pca_x"] = X_pca[:, 0]
    perfil_df["pca_y"] = X_pca[:, 1]

    resumo_clusters = (
        perfil_df.groupby("cluster")
        .agg(
            streamers=("streamer", list),
            total_streamers=("streamer", "count"),
            media_percent_PP=("%PP", "mean"),
            media_total_frames=("total_frames", "mean"),
            media_lives=("lives_detectadas", "mean"),
            media_duracao_por_live=("media_segundos_por_live", "mean")
        )
        .reset_index()
    )

    return perfil_df, resumo_clusters
