# -*- coding: utf-8 -*-
"""
Backend Flask — CardioIA Fase 4 (Ir Além 2).

Serve o modelo ResNet50 treinado (Transfer Learning) para o app React Native:
recebe uma radiografia de tórax, devolve a classe (Cardiomegalia × Normal), a
confiança e o mapa de calor Grad-CAM sobreposto (em base64).

Rotas:
    GET  /health   -> status do serviço e se o modelo foi carregado
    POST /predict  -> multipart/form-data com campo 'image' (arquivo)
                      resposta JSON: { label, prob_cardiomegalia, confianca, gradcam_png_b64 }

Uso:
    cd mobile/backend
    pip install -r requirements.txt
    # aponte para o modelo gerado pelo notebook/treino:
    export MODELO_PATH=../../modelos/resnet50_finetuned.keras
    python app.py        # sobe em http://0.0.0.0:5050

⚠️ Protótipo educacional — não é dispositivo médico.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # modelo salvo com Keras 2 legado

import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)
GRADCAM_LAYER = "conv5_block3_out"
MODELO_PATH = os.environ.get("MODELO_PATH", "../../modelos/resnet50_finetuned.keras")

app = Flask(__name__)
CORS(app)  # libera chamadas do app mobile (origem diferente)

# --- Carrega o modelo uma única vez ---
modelo = None
erro_carregamento = None
try:
    modelo = tf.keras.models.load_model(MODELO_PATH)
    print(f"✅ Modelo carregado: {MODELO_PATH}")
except Exception as e:  # noqa: BLE001
    erro_carregamento = str(e)
    print(f"⚠️ Falha ao carregar modelo ({MODELO_PATH}): {e}")


def preparar(img_pil):
    """Replica o pré-processamento do treino (ResNet50)."""
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr_pre = preprocess_input(np.expand_dims(arr.copy(), axis=0))
    return arr, arr_pre


def gradcam(img_array, nome_camada=GRADCAM_LAYER):
    grad_model = tf.keras.models.Model(
        modelo.inputs, [modelo.get_layer(nome_camada).output, modelo.output]
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array)
        classe = pred[:, 0]
    grads = tape.gradient(classe, conv_out)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.reduce_sum(conv_out[0] * pesos, axis=-1)
    hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
    return hm.numpy()


def sobrepor(img_rgb_uint8, hm):
    hm = cv2.resize(hm, IMG_SIZE)
    hm = np.uint8(255 * hm)
    cor = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cor = cv2.cvtColor(cor, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb_uint8.astype("uint8"), 0.6, cor, 0.4, 0)


def para_b64(img_uint8):
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if modelo is not None else "modelo_indisponivel",
        "modelo_carregado": modelo is not None,
        "modelo_path": MODELO_PATH,
        "erro": erro_carregamento,
    })


@app.post("/predict")
def predict():
    if modelo is None:
        return jsonify({"erro": f"Modelo não carregado: {erro_carregamento}"}), 503
    if "image" not in request.files:
        return jsonify({"erro": "Envie um arquivo no campo 'image'."}), 400

    try:
        img_pil = Image.open(request.files["image"].stream)
    except Exception as e:  # noqa: BLE001
        return jsonify({"erro": f"Imagem inválida: {e}"}), 400

    arr, arr_pre = preparar(img_pil)
    prob = float(modelo.predict(arr_pre, verbose=0).ravel()[0])  # P(cardiomegalia)
    is_cardio = prob >= 0.5
    confianca = prob if is_cardio else 1 - prob

    try:
        overlay = sobrepor(arr, gradcam(arr_pre))
        gradcam_b64 = para_b64(overlay)
    except Exception as e:  # noqa: BLE001
        print("Grad-CAM indisponível:", e)
        gradcam_b64 = None

    return jsonify({
        "label": "Cardiomegalia detectada" if is_cardio else "Cardiomegalia NÃO detectada",
        "classe": "cardiomegalia" if is_cardio else "normal",
        "prob_cardiomegalia": round(prob, 4),
        "confianca": round(confianca, 4),
        "gradcam_png_b64": gradcam_b64,
        "aviso": "Protótipo educacional — não substitui avaliação médica.",
    })


if __name__ == "__main__":
    porta = int(os.environ.get("PORT", 5050))  # 5050: porta 5000 no macOS costuma estar ocupada (AirPlay)
    app.run(host="0.0.0.0", port=porta, debug=False)
