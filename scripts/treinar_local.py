# -*- coding: utf-8 -*-
"""
Treino headless local (Apple Silicon / Metal) — ESPELHA a lógica do notebook
notebooks/CardioIA_Fase4.ipynb para produzir métricas REAIS e modelos salvos.

NÃO faz parte da entrega (helper local). Saídas:
  - modelos/cnn_do_zero.keras, modelos/resnet50_finetuned.keras
  - docs/resultados_run.json  (métricas dos 2 modelos + fairness + stats do dataset)

Uso:
    conda run -n cardioia python scripts/treinar_local.py
Variáveis de ambiente opcionais (para encurtar o treino):
    EPOCHS_CNN (default 25), EPOCHS_RN1 (15), EPOCHS_RN2 (12)
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # ANTES de importar o TF (mantém ImageDataGenerator)

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_CNN = int(os.environ.get("EPOCHS_CNN", 25))
EPOCHS_RN1 = int(os.environ.get("EPOCHS_RN1", 15))
EPOCHS_RN2 = int(os.environ.get("EPOCHS_RN2", 12))

DATA_ROOT = Path("./cardio_data")
MODELOS_DIR = Path("modelos"); MODELOS_DIR.mkdir(exist_ok=True)
CLASSES = ["normal", "cardiomegalia"]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POSITIVOS = ("cardio", "true", "yes", "sick", "abnormal", "disease", "positive", "1")

print("=" * 60)
print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
print(f"Epochs -> CNN:{EPOCHS_CNN} RN1:{EPOCHS_RN1} RN2:{EPOCHS_RN2}")
print("=" * 60)


def mapear_classe(nome: str) -> str:
    return "cardiomegalia" if any(t in nome for t in POSITIVOS) else "normal"


# ---------- Parte 1: dataset + split ----------
registros = [{"caminho": str(p), "classe_bruta": p.parent.name.lower().strip()}
             for p in DATA_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
df = pd.DataFrame(registros)
df["classe"] = df["classe_bruta"].apply(mapear_classe)
print("Total imagens:", len(df))
print(df["classe"].value_counts().to_dict())

df_treino, df_temp = train_test_split(df, test_size=0.30, stratify=df["classe"], random_state=SEED)
df_val, df_teste = train_test_split(df_temp, test_size=0.50, stratify=df_temp["classe"], random_state=SEED)
print(f"Split -> treino {len(df_treino)} | val {len(df_val)} | teste {len(df_teste)}")

y_treino = df_treino["classe"].map({"normal": 0, "cardiomegalia": 1}).values
pesos = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_treino)
class_weight = {0: float(pesos[0]), 1: float(pesos[1])}
print("class_weight =", class_weight)


def construir_geradores(modo):
    if modo == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        comum = dict(preprocessing_function=preprocess_input)
    else:
        comum = dict(rescale=1.0 / 255)
    aug = ImageDataGenerator(rotation_range=12, width_shift_range=0.08, height_shift_range=0.08,
                             zoom_range=0.10, brightness_range=(0.9, 1.1), horizontal_flip=True,
                             fill_mode="nearest", **comum)
    ev = ImageDataGenerator(**comum)
    tr = aug.flow_from_dataframe(df_treino, x_col="caminho", y_col="classe", target_size=IMG_SIZE,
                                 batch_size=BATCH, class_mode="binary", color_mode="rgb",
                                 classes=CLASSES, shuffle=True, seed=SEED)
    va = ev.flow_from_dataframe(df_val, x_col="caminho", y_col="classe", target_size=IMG_SIZE,
                                batch_size=BATCH, class_mode="binary", color_mode="rgb",
                                classes=CLASSES, shuffle=False)
    te = ev.flow_from_dataframe(df_teste, x_col="caminho", y_col="classe", target_size=IMG_SIZE,
                                batch_size=BATCH, class_mode="binary", color_mode="rgb",
                                classes=CLASSES, shuffle=False)
    return tr, va, te


def callbacks_para(caminho):
    return [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            ModelCheckpoint(caminho, monitor="val_loss", save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)]


def avaliar(modelo, gen):
    gen.reset()
    y_true = gen.classes
    y_prob = modelo.predict(gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "precisao": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "_y_true": y_true, "_y_prob": y_prob, "_y_pred": y_pred,
    }


# ---------- CNN do zero ----------
print("\n>>> Treinando CNN do zero...")
t0 = time.time()
treino_cnn, val_cnn, teste_cnn = construir_geradores("rescale")
cnn = models.Sequential(name="CNN_do_zero")
cnn.add(layers.Input(shape=(224, 224, 3)))
for f in (32, 64, 128):
    cnn.add(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.GlobalAveragePooling2D())
cnn.add(layers.Dense(128, activation="relu"))
cnn.add(layers.Dropout(0.5))
cnn.add(layers.Dense(1, activation="sigmoid"))
cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
cnn.fit(treino_cnn, validation_data=val_cnn, epochs=EPOCHS_CNN, class_weight=class_weight,
        callbacks=callbacks_para(str(MODELOS_DIR / "cnn_do_zero.keras")), verbose=2)
metricas_cnn = avaliar(cnn, teste_cnn)
t_cnn = time.time() - t0
print("CNN do zero:", {k: round(v, 4) for k, v in metricas_cnn.items() if not k.startswith("_")})


# ---------- ResNet50 Transfer Learning ----------
print("\n>>> Treinando ResNet50 (Transfer Learning)...")
t0 = time.time()
treino_rn, val_rn, teste_rn = construir_geradores("resnet")
from tensorflow.keras.applications import ResNet50
inputs = layers.Input(shape=(224, 224, 3))
base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(1, activation="sigmoid")(x)
resnet = models.Model(inputs, out, name="ResNet50_TransferLearning")
resnet.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
print("Fase 1 (base congelada)...")
resnet.fit(treino_rn, validation_data=val_rn, epochs=EPOCHS_RN1, class_weight=class_weight,
           callbacks=callbacks_para(str(MODELOS_DIR / "resnet50_fase1.keras")), verbose=2)

print("Fase 2 (fine-tuning últimos 30 layers)...")
base.trainable = True
for c in base.layers[:-30]:
    c.trainable = False
resnet.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-5), loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
resnet.fit(treino_rn, validation_data=val_rn, epochs=EPOCHS_RN2, class_weight=class_weight,
           callbacks=callbacks_para(str(MODELOS_DIR / "resnet50_finetuned.keras")), verbose=2)
metricas_rn = avaliar(resnet, teste_rn)
t_rn = time.time() - t0
print("ResNet50:", {k: round(v, 4) for k, v in metricas_rn.items() if not k.startswith("_")})


# ---------- Fairness (melhor modelo = ResNet50) ----------
cm = confusion_matrix(metricas_rn["_y_true"], metricas_rn["_y_pred"])
tn, fp, fn, tp = [int(v) for v in cm.ravel()]
tpr_cardio = tp / (tp + fn) if (tp + fn) else 0.0
fnr_cardio = fn / (tp + fn) if (tp + fn) else 0.0
tnr_normal = tn / (tn + fp) if (tn + fp) else 0.0
fpr_normal = fp / (tn + fp) if (tn + fp) else 0.0
fairness = {
    "matriz_confusao": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    "recall_cardiomegalia_TPR": round(tpr_cardio, 4),
    "fnr_cardiomegalia": round(fnr_cardio, 4),
    "recall_normal_TNR": round(tnr_normal, 4),
    "fpr_normal": round(fpr_normal, 4),
    "gap_recall_classes": round(abs(tpr_cardio - tnr_normal), 4),
}
print("\nFairness ResNet50:", fairness)


# ---------- Persistência dos resultados ----------
def limpa(m):
    return {k: round(v, 4) for k, v in m.items() if not k.startswith("_")}

resultado = {
    "tensorflow": tf.__version__,
    "seed": SEED,
    "dataset": {
        "total": len(df),
        "por_classe": df["classe"].value_counts().to_dict(),
        "split": {"treino": len(df_treino), "val": len(df_val), "teste": len(df_teste)},
        "prevalencia_cardiomegalia_pct": round(100 * (df["classe"] == "cardiomegalia").mean(), 1),
    },
    "epochs": {"cnn": EPOCHS_CNN, "resnet_fase1": EPOCHS_RN1, "resnet_fase2": EPOCHS_RN2},
    "tempo_treino_s": {"cnn": round(t_cnn, 1), "resnet": round(t_rn, 1)},
    "class_weight": class_weight,
    "metricas": {"cnn_do_zero": limpa(metricas_cnn), "resnet50": limpa(metricas_rn)},
    "fairness_resnet50": fairness,
    "melhor_por_auc": "resnet50" if metricas_rn["auc"] >= metricas_cnn["auc"] else "cnn_do_zero",
}
Path("docs").mkdir(exist_ok=True)
with open("docs/resultados_run.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)
print("\n✅ Resultados salvos em docs/resultados_run.json")
print(json.dumps(resultado, ensure_ascii=False, indent=2))
