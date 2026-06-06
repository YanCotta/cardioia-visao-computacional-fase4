# -*- coding: utf-8 -*-
"""
Smoke test do ambiente local (Apple Silicon / Metal).

Verifica, ANTES de rodar o notebook completo, se:
  1) o Keras 2 legado está ativo (necessário para ImageDataGenerator);
  2) a GPU Metal é detectada pelo TensorFlow;
  3) um treino mínimo roda sem crash e sem NaN (op-suporte do Metal OK);
  4) ImageDataGenerator importa e funciona.

Uso:
    conda run -n cardioia python scripts/smoke_test_metal.py
"""
import os
# IMPORTANTE: definir ANTES de importar o tensorflow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

print("=" * 55)
print("TensorFlow:", tf.__version__)
try:
    import keras
    print("Keras (frontend):", keras.__version__)
except Exception as e:
    print("Keras import:", e)

# 1) GPU Metal detectada?
gpus = tf.config.list_physical_devices("GPU")
print("GPUs detectadas:", gpus)
usando_gpu = len(gpus) > 0
print("→ Acelerador:", "GPU Metal ✅" if usando_gpu else "CPU (Metal NÃO detectada) ⚠️")

# 2) Treino mínimo (2 passos) com dados aleatórios
print("\n[teste] Treinando um modelo mínimo por 2 passos...")
from tensorflow.keras import layers, models

x = np.random.rand(64, 32, 32, 3).astype("float32")
y = np.random.randint(0, 2, size=(64, 1)).astype("float32")

m = models.Sequential([
    layers.Input((32, 32, 3)),
    layers.Conv2D(8, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid"),
])
m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist = m.fit(x, y, epochs=2, batch_size=16, verbose=2)

loss_final = hist.history["loss"][-1]
loss_ok = np.isfinite(loss_final)
print(f"→ Loss final: {loss_final:.4f}  ({'OK ✅' if loss_ok else 'NaN/Inf ❌'})")

# 3) ImageDataGenerator funciona?
print("\n[teste] ImageDataGenerator...")
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=12)
    batch = next(gen.flow(x, y, batch_size=8))
    idg_ok = batch[0].shape == (8, 32, 32, 3)
    print("→ ImageDataGenerator:", "OK ✅" if idg_ok else "shape inesperado ❌")
except Exception as e:
    idg_ok = False
    print("→ ImageDataGenerator FALHOU ❌:", e)

# Veredito
print("\n" + "=" * 55)
if loss_ok and idg_ok:
    if usando_gpu:
        print("✅ TUDO OK — pode rodar o notebook localmente na GPU Metal.")
    else:
        print("⚠️ Funciona, mas SEM GPU (CPU). Treino será mais lento.")
        print("   Verifique a instalação do tensorflow-metal.")
else:
    print("❌ Problema detectado — NÃO rode o dataset inteiro ainda.")
    print("   Reporte a saída acima para diagnóstico.")
print("=" * 55)
