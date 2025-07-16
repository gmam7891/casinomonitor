
import sqlite3
import os
import json

DB_PATH = os.path.join("output", "monitor.db")

def criar_tabelas():
    os.makedirs("output", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Deteccoes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        streamer TEXT,
        provedor TEXT,
        jogo TEXT,
        viewers INTEGER,
        screenshot TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Provedores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        jogo TEXT,
        aliases TEXT
    );
    """)

    conn.commit()
    conn.close()
    print("[INFO] Tabelas criadas ou já existem.")


def salvar_deteccao_db(data, streamer, provedor, jogo, viewers, screenshot):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO Deteccoes (data, streamer, provedor, jogo, viewers, screenshot)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (data, streamer, provedor, jogo, viewers, screenshot))

    conn.commit()
    conn.close()
    print(f"[INFO] Detecção inserida: {provedor} - {jogo}")


def popular_provedores():
    caminho_json = os.path.join("data", "provedores_jogos.json")
    if not os.path.exists(caminho_json):
        print("[ERROR] JSON de provedores não encontrado.")
        return

    with open(caminho_json, "r", encoding="utf-8") as f:
        provedores = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for p in provedores:
        cursor.execute("""
        INSERT INTO Provedores (nome, jogo, aliases)
        VALUES (?, ?, ?);
        """, (p["Provedor"], p["Jogo"], ','.join(p["Aliases"])))

    conn.commit()
    conn.close()
    print(f"[INFO] Provedores carregados no banco.")


def top_jogos(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT provedor, jogo, COUNT(*) as total
    FROM Deteccoes
    GROUP BY provedor, jogo
    ORDER BY total DESC
    LIMIT ?;
    """, (limit,))

    resultados = cursor.fetchall()
    conn.close()
    return resultados
