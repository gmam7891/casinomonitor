import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from collections import Counter

# Caminhos
DATASET_DIR = "dataset"
MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_pragmatic.keras")

# Hiperparâmetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# Garante diretórios
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError("A pasta 'dataset/' não foi encontrada.")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Geração de dados
print("🔄 Preparando dados...")
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Modelo base
print("🔧 Montando arquitetura com MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # congelar base

model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Class weights balanceado
counter = Counter(train_gen.classes)
total = float(sum(counter.values()))
class_weight = {cls: total / count for cls, count in counter.items()}

# Treinamento
print("🚀 Iniciando treinamento...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight
)

# Salvar modelo
model.save(MODEL_PATH)
print(f"✅ Modelo salvo em: {MODEL_PATH}")
