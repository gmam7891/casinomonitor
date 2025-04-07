import os
import traceback
import matplotlib
matplotlib.use("Agg")  # <- ESSA LINHA AQUI é essencial para ambientes headless
import matplotlib.pyplot as plt
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
import numpy as np

def criar_modelo_mobilenet_binario():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def treinar_modelo(st, base_path="dataset", model_path="modelo/modelo_pragmatic.keras", epochs=10):
    try:
        st.markdown("### 🔄 Iniciando treinamento do modelo...")

        subdirs = os.listdir(base_path)
        if not subdirs or len(subdirs) < 2:
            st.error("❌ O diretório 'dataset/' deve conter pelo menos 2 subpastas com classes diferentes.")
            return False

        st.info(f"📁 Classes detectadas: `{', '.join(subdirs)}`")

        datagen = ImageDataGenerator(
            validation_split=0.2,
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        img_size = (224, 224)
        batch_size = 32

        train_gen = datagen.flow_from_directory(
            base_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )

        val_gen = datagen.flow_from_directory(
            base_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )

        class_counts = Counter(train_gen.classes)
        st.write("📊 Distribuição das classes no treino:", dict(class_counts))

        model = criar_modelo_mobilenet_binario()

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        total = sum(class_counts.values())
        class_weight = {
            0: total / (2.0 * class_counts[0]),
            1: total / (2.0 * class_counts[1])
        }

        st.write("⚖️ Pesos de classe aplicados:", class_weight)

        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True)
        ]

        st.markdown("### ⏳ Treinando modelo...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

        st.success("✅ Modelo treinado e salvo com sucesso!")

        # Curvas de aprendizado
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(history.history['loss'], label='Treino')
        axs[0].plot(history.history['val_loss'], label='Validação')
        axs[0].set_title('Loss por Época')
        axs[0].set_xlabel('Época')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(history.history['accuracy'], label='Treino')
        axs[1].plot(history.history['val_accuracy'], label='Validação')
        axs[1].set_title('Acurácia por Época')
        axs[1].set_xlabel('Época')
        axs[1].set_ylabel('Acurácia')
        axs[1].legend()

        st.session_state["curva_fig"] = fig

        # Avaliação com relatório de classificação
        val_preds = model.predict(val_gen)
        pred_labels = (val_preds > 0.5).astype(int).flatten()
        true_labels = val_gen.classes

        if len(np.unique(true_labels)) < 2:
            st.warning("⚠️ Apenas uma classe presente na validação. O relatório será limitado.")

        labels_ordenadas = sorted(train_gen.class_indices.values())
        report = classification_report(
            true_labels,
            pred_labels,
            labels=labels_ordenadas,
            target_names=subdirs,
            zero_division=0
        )

               st.markdown("### 📋 Relatório de Classificação")
        st.code(report)

        return True, model

    except Exception as e:
        st.error("❌ Erro durante o treinamento:")
        st.code(traceback.format_exc())
        return False, None
