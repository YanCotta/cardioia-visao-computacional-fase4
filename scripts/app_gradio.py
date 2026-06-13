# -*- coding: utf-8 -*-
"""
Protótipo Gradio standalone — CardioIA Fase 4.

Carrega o modelo ResNet50 treinado (salvo em .keras) e abre a interface web
(upload → predição + confiança + Grad-CAM). É a MESMA lógica da célula Gradio do
notebook, empacotada para rodar fora do Colab (inclusive via Docker).

Uso:
    export MODELO_PATH=modelos/resnet50_finetuned.keras
    python scripts/app_gradio.py      # abre em http://0.0.0.0:7860

⚠️ Protótipo educacional — não é dispositivo médico.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # modelo salvo com Keras 2 legado

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)
GRADCAM_LAYER = "conv5_block3_out"
MODELO_PATH = os.environ.get("MODELO_PATH", "modelos/resnet50_finetuned.keras")
PORT = int(os.environ.get("PORT", 7860))

print(f"Carregando modelo: {MODELO_PATH}")
modelo = tf.keras.models.load_model(MODELO_PATH, compile=False)
print("✅ Modelo carregado.")


def preparar(img_pil):
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
    cor = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb_uint8.astype("uint8"), 0.6, cor, 0.4, 0)


def classificar(img_pil):
    arr, arr_pre = preparar(img_pil)
    prob = float(modelo.predict(arr_pre, verbose=0).ravel()[0])
    if prob >= 0.5:
        rotulo, conf = "🔴 Cardiomegalia detectada", prob
    else:
        rotulo, conf = "🟢 Cardiomegalia NÃO detectada", 1 - prob
    try:
        overlay = sobrepor(arr, gradcam(arr_pre))
    except Exception as e:  # noqa: BLE001
        print("Grad-CAM indisponível:", e)
        overlay = arr.astype("uint8")
    texto = (f"**{rotulo}**\n\nConfiança: **{conf*100:.1f}%**\n\n"
             f"_Protótipo educacional — não substitui avaliação médica._")
    return {"Cardiomegalia": prob, "Normal": 1 - prob}, overlay, texto


demo = gr.Interface(
    fn=classificar,
    inputs=gr.Image(type="pil", label="Radiografia de tórax (upload)"),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilidades"),
        gr.Image(label="Grad-CAM (regiões de atenção do modelo)"),
        gr.Markdown(label="Resultado"),
    ],
    title="🫀 CardioIA — Detecção de Cardiomegalia (protótipo)",
    description=("Faça upload de uma radiografia de tórax. O modelo ResNet50 estima a "
                 "probabilidade de cardiomegalia e destaca as regiões analisadas (Grad-CAM). "
                 "⚠️ Uso educacional — não é diagnóstico médico."),
    flagging_mode="never",
)

if __name__ == "__main__":
    # GRADIO_SHARE=1 força link público (gradio.live). Em máquinas normais o launch
    # local funciona direto; em ambientes onde o self-check de localhost falha
    # (sandboxes/proxies) caímos automaticamente no link compartilhável.
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    try:
        demo.launch(server_name="0.0.0.0", server_port=PORT, share=share)
    except ValueError as e:
        print("Launch local indisponível, criando link público (share=True):", e)
        demo.launch(server_name="0.0.0.0", server_port=PORT, share=True)
