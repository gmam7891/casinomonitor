# ml_training.py

import os
import traceback
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

def treinar_modelo(st, base_path="dataset", model_path="modelo/modelo_pragmatic.keras", epochs=5):
    try:
        st.markdown("### 🔄 Iniciando treinamento do modelo...")

        # 🚨 Verificação da estrutura do dataset
        subdirs = os.listdir(base_path)
        if not subdirs or len(subdirs) < 2:
            st.error("❌ O diretório 'dataset/' deve conter pelo menos 2 subpastas com classes diferentes.")
            return False

        st.info(f"📁 Classes detectadas: `{', '.join(subdirs)}`")

        # 🔁 Geração de dados
        datagen = ImageDataGenerator(
            validation_split=0.2,
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
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

        # 📊 Distribuição das classes
        class_counts = Counter(train_gen.classes)
        st.write("📊 Distribuição das classes no treino:", dict(class_counts))

        # 🧠 Construção do modelo
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # ⚖️ Pesos de classe
        total = sum(class_counts.values())
        class_weight = {
            0: total / (2.0 * class_counts[0]),
            1: total / (2.0 * class_counts[1])
        }
        st.write("⚖️ Pesos de classe aplicados:", class_weight)

        # ⏳ Treinamento
        st.markdown("### ⏳ Treinando modelo...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weight,
            verbose=1
        )

        # 💾 Salvando modelo
        model.save(model_path)
        st.success("✅ Modelo treinado e salvo com sucesso!")

        # 📈 Curvas de aprendizado
        st.markdown("### 📊 Curvas de Aprendizado")
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

        st.pyplot(fig)

        return True

    except Exception as e:
        st.error("❌ Erro durante o treinamento:")
        st.code(traceback.format_exc())
        return False
