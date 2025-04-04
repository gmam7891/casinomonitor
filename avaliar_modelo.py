
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- CONFIGURAÇÕES ---
MODEL_PATH = "modelo_pragmatic.keras"  # Caminho do modelo
IMAGE_DIR = "imagens"  # Pasta com as imagens para predição
IMAGE_SIZE = (224, 224)  # Tamanho esperado pelo modelo

# 🔤 Mapeamento opcional dos índices para nomes de jogos
# (ajuste conforme suas classes reais)
CLASSES = [
    "Blackjack Clássico",
    "Sugar Rush 1000",
    "Dragon Hero",
    "Sweet Bonanza 1000",
    "Gates of Olympus 1000",
    "Volcano Goddess",
    "Big Bass Bonanza",
    "Fat Panda"
]

def preparar_imagem(img_path, target_size):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def main():
    print("🔁 Carregando modelo...")
    model = load_model(MODEL_PATH)

    print(f"🔍 Lendo imagens da pasta: {IMAGE_DIR}")
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(IMAGE_DIR, filename)
            img_array = preparar_imagem(img_path, IMAGE_SIZE)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)

            predicted_label = CLASSES[predicted_index] if predicted_index < len(CLASSES) else f"Classe {predicted_index}"

            print(f"🖼️ {filename} → 🎰 {predicted_label} ({confidence:.2%} confiança)")

if __name__ == "__main__":
    main()
