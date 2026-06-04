# -*- coding: utf-8 -*-
"""
Gerador do notebook CardioIA - Fase 4 (Visão Computacional).

Constrói o arquivo notebooks/CardioIA_Fase4.ipynb usando apenas a stdlib
(módulo json), garantindo um .ipynb válido sem dependências externas.

Uso:
    python3 scripts/gerar_notebook.py
"""
import json
import os

CELLS = []


def md(text: str):
    """Adiciona uma célula de markdown."""
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": _split(text),
    })


def code(text: str):
    """Adiciona uma célula de código."""
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(text),
    })


def _split(text: str):
    """Converte string em lista de linhas terminadas em \n (formato nbformat)."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


# =====================================================================
# CÉLULA 1 — Título, objetivos, dataset e justificativa
# =====================================================================
md(r"""
# 🫀 CardioIA — Fase 4: Assistente Cardiológico Virtual com Visão Computacional

### Classificação de **Cardiomegalia** em radiografias de tórax

---

## 1. Objetivos da Fase 4

Este notebook implementa um **protótipo completo de Visão Computacional** que transforma
imagens médicas simuladas (raios-X de tórax) em informação interpretável para **apoio à
decisão clínica**. Os objetivos são:

1. **Pré-processar** e organizar um dataset público de imagens médicas (treino / validação / teste).
2. **Treinar e avaliar** duas abordagens de CNN:
   - uma **CNN simples treinada do zero**;
   - **Transfer Learning** com a **ResNet50** pré-treinada (ImageNet) + *fine-tuning* leve.
3. **Avaliar** com métricas clínicas relevantes (acurácia, matriz de confusão, precisão,
   recall, F1-score e AUC).
4. **Apresentar** os resultados em um **protótipo interativo (Gradio)** com upload de imagem,
   predição, nível de confiança e **mapa de calor Grad-CAM**.

> **Importante (escopo acadêmico):** este é um protótipo educacional. **Não** é um dispositivo
> médico e **não** deve ser usado para diagnóstico real. A discussão de limitações e ética está
> na seção final.

---

## 2. Dataset escolhido e justificativa

**Dataset:** [*Cardiomegaly Disease Prediction Using CNN*](https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn)
(Kaggle — derivado do **NIH Chest X-ray**).

**Por que este dataset:**

| Critério | Justificativa |
|---|---|
| **Relevância temática** | Cardiomegalia (aumento da área cardíaca) é uma condição **cardiológica** detectável em raio-X — alinhada diretamente ao nome e propósito do **CardioIA**. |
| **Tarefa bem definida** | **Classificação binária** (Cardiomegalia × Normal), ideal para demonstrar o pipeline CNN do zero vs. Transfer Learning. |
| **Tamanho gerenciável** | Volume adequado para treinar no **Google Colab** (GPU gratuita) dentro do tempo de um protótipo. |
| **Origem reconhecida** | Derivado do **NIH Chest X-ray**, base amplamente usada em pesquisa médica. |

**Stack:** Google Colab + TensorFlow/Keras + scikit-learn + Gradio.
""")

# =====================================================================
# CÉLULA 2 — Estrutura do notebook
# =====================================================================
md(r"""
## 3. Estrutura do notebook

| Bloco | Conteúdo |
|---|---|
| **Setup** | Imports, verificação de GPU, sementes de reprodutibilidade |
| **Parte 1 — Pré-processamento** | Download (Kaggle API), exploração, redimensionamento 224×224, normalização, *data augmentation* médico-seguro, split treino/validação/teste |
| **Parte 2 — CNN do zero** | Arquitetura simples, treinamento, avaliação |
| **Parte 2 — Transfer Learning** | ResNet50 + *fine-tuning* + avaliação comparativa |
| **Análise** | Curvas de treino, matriz de confusão, *classification report*, AUC |
| **Protótipo** | Interface Gradio (upload → predição + confiança + Grad-CAM) |
| **Relatório** | Documento técnico final em Markdown |
""")

# =====================================================================
# CÉLULA 3 — Setup do ambiente
# =====================================================================
md(r"""
---
# ⚙️ Setup do ambiente

Importação de bibliotecas, verificação de GPU e definição de **sementes** para
garantir **reprodutibilidade** dos experimentos.

> **Compatibilidade Keras (importante):** versões recentes do Colab usam **Keras 3**, onde
> o clássico `ImageDataGenerator` foi **removido**. Para manter o pipeline "como ensinado em
> aula" funcionando de forma estável, forçamos o **Keras 2 legado** (`tf_keras`) na célula
> abaixo, que **deve ser a primeira a rodar**.
""")

code(r"""
# --- Compatibilidade: força Keras 2 legado (mantém ImageDataGenerator disponível) ---
# DEVE rodar ANTES de importar o tensorflow. Em TF < 2.16 a variável é ignorada (inócua).
!pip install -q tf_keras
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
print("✅ Keras 2 legado ativado (TF_USE_LEGACY_KERAS=1).")
""")

code(r"""
# --- Bibliotecas principais ---
import os
import random
import shutil
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

warnings.filterwarnings("ignore")

# --- Reprodutibilidade ---
# Fixamos as sementes de todas as fontes de aleatoriedade para que os
# resultados sejam reproduzíveis entre execuções.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)

# --- Verificação de GPU ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU disponível: {gpus[0].name}")
    print("   (Runtime > Alterar tipo de ambiente de execução > GPU)")
else:
    print("⚠️ Nenhuma GPU detectada. O treinamento será LENTO na CPU.")
    print("   Recomendado: Runtime > Alterar tipo de ambiente de execução > T4 GPU")
""")

# =====================================================================
# CÉLULA 4 — Kaggle setup (markdown)
# =====================================================================
md(r"""
---
# 📦 Parte 1 — Pré-processamento e Organização das Imagens

## 1.1 Download do dataset via Kaggle API

Para baixar o dataset diretamente no Colab, precisamos do seu token da API do Kaggle
(arquivo **`kaggle.json`**).

**Como obter o `kaggle.json`:**
1. Acesse [kaggle.com](https://www.kaggle.com) → clique na sua foto → **Settings**.
2. Seção **API** → **Create New Token** → baixa o arquivo `kaggle.json`.
3. **Rode a célula abaixo** e faça o **upload** do arquivo quando solicitado.

> 🔒 O `kaggle.json` contém uma credencial pessoal. Ele é gravado apenas no ambiente
> temporário do Colab (`~/.kaggle/`) e **não** deve ser commitado no GitHub.
""")

# =====================================================================
# CÉLULA 5 — Kaggle upload + download
# =====================================================================
code(r"""
# --- Configuração da credencial do Kaggle ---
# No Colab, faz upload interativo do kaggle.json. Fora do Colab, espera que o
# arquivo já esteja em ~/.kaggle/kaggle.json.
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_DIR.mkdir(exist_ok=True)

try:
    from google.colab import files  # type: ignore
    if not (KAGGLE_DIR / "kaggle.json").exists():
        print("👉 Faça o upload do seu arquivo kaggle.json:")
        uploaded = files.upload()
        for name in uploaded:
            if name.endswith(".json"):
                shutil.move(name, KAGGLE_DIR / "kaggle.json")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Fora do Colab — usando ~/.kaggle/kaggle.json existente.")

# Permissão exigida pela API do Kaggle (apenas leitura do dono)
os.chmod(KAGGLE_DIR / "kaggle.json", 0o600)
print("✅ Credencial do Kaggle configurada.")
""")

code(r"""
# --- Instala a CLI do Kaggle e baixa o dataset ---
!pip install -q kaggle

DATASET_SLUG = "rahimanshu/cardiomegaly-disease-prediction-using-cnn"
DATA_ROOT = Path("/content/cardio_data") if IN_COLAB else Path("./cardio_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Baixa (-d) e descompacta (--unzip) o dataset
!kaggle datasets download -d {DATASET_SLUG} -p {DATA_ROOT} --unzip

print("\n✅ Download concluído. Conteúdo de", DATA_ROOT, ":")
for p in sorted(DATA_ROOT.rglob("*"))[:40]:
    rel = p.relative_to(DATA_ROOT)
    print("   ", "📁" if p.is_dir() else "  ", rel)
""")

# =====================================================================
# CÉLULA 6 — Detecção automática da estrutura
# =====================================================================
md(r"""
## 1.2 Exploração e detecção automática da estrutura

Datasets do Kaggle variam na organização (podem vir pré-divididos em `train/test` ou
em pasta única, com nomes de classe diferentes). Para tornar o pipeline **robusto**,
a célula abaixo:

1. percorre **recursivamente** todas as imagens;
2. infere o **rótulo** a partir do nome da pasta-pai;
3. consolida tudo em um **DataFrame** (`caminho`, `classe_bruta`).

Em seguida, mapeamos os nomes brutos para **duas classes**: `cardiomegalia` (positiva)
e `normal` (negativa).
""")

code(r"""
# --- Varre todas as imagens e infere o rótulo pela pasta-pai ---
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

registros = []
for p in DATA_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in EXTS:
        registros.append({"caminho": str(p), "classe_bruta": p.parent.name.lower().strip()})

df = pd.DataFrame(registros)
print(f"Total de imagens encontradas: {len(df)}")
print("\nDistribuição por pasta-pai (classe bruta):")
print(df["classe_bruta"].value_counts())
""")

code(r"""
# --- Mapeia nomes brutos -> 2 classes (cardiomegalia / normal) ---
# Heurística: pastas cujo nome sugere doença viram a classe POSITIVA.
POSITIVOS = ("cardio", "true", "yes", "sick", "abnormal", "disease", "positive", "1")

def mapear_classe(nome: str) -> str:
    return "cardiomegalia" if any(t in nome for t in POSITIVOS) else "normal"

df["classe"] = df["classe_bruta"].apply(mapear_classe)

print("Mapeamento aplicado (classe_bruta -> classe):")
print(df.groupby(["classe_bruta", "classe"]).size())
print("\nDistribuição final das classes:")
print(df["classe"].value_counts())

# ⚠️ CONFIRA o mapeamento acima. Se algum nome de pasta foi classificado errado,
# ajuste a tupla POSITIVOS ou edite df["classe"] manualmente antes de continuar.
""")

# =====================================================================
# CÉLULA 7 — Visualização de amostras + distribuição
# =====================================================================
code(r"""
# --- Gráfico de distribuição das classes ---
plt.figure(figsize=(5, 4))
contagem = df["classe"].value_counts()
plt.bar(contagem.index, contagem.values, color=["#d62728", "#2ca02c"])
plt.title("Distribuição das classes")
plt.ylabel("Nº de imagens")
for i, v in enumerate(contagem.values):
    plt.text(i, v, str(v), ha="center", va="bottom")
plt.tight_layout()
plt.show()

razao = contagem.max() / contagem.min()
print(f"Razão de desbalanceamento (maior/menor): {razao:.2f}x")
if razao > 1.5:
    print("→ Dataset desbalanceado: usaremos class_weight no treinamento.")
else:
    print("→ Dataset razoavelmente balanceado.")
""")

code(r"""
# --- Visualiza amostras de cada classe ---
from PIL import Image

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for linha, classe in enumerate(["cardiomegalia", "normal"]):
    amostras = df[df["classe"] == classe]["caminho"].sample(4, random_state=SEED).tolist()
    for col, cam in enumerate(amostras):
        img = Image.open(cam).convert("RGB")
        axes[linha, col].imshow(img)
        axes[linha, col].set_title(f"{classe}\n{img.size}", fontsize=9)
        axes[linha, col].axis("off")
plt.suptitle("Amostras do dataset (note a variação de tamanho/contraste)", y=1.02)
plt.tight_layout()
plt.show()
""")

# =====================================================================
# CÉLULA 8 — Split treino/val/teste (markdown)
# =====================================================================
md(r"""
## 1.3 Divisão em treino / validação / teste

Fazemos uma divisão **estratificada** (mantém a proporção de classes em cada conjunto):

- **Treino:** 70% — ajuste dos pesos do modelo.
- **Validação:** 15% — *EarlyStopping*, escolha do melhor *checkpoint* e ajuste de hiperparâmetros.
- **Teste:** 15% — avaliação **final e imparcial** (o modelo nunca vê esses dados no treino).

A estratificação é importante em dados médicos para que nenhum conjunto fique sem
exemplos suficientes da classe minoritária.
""")

code(r"""
from sklearn.model_selection import train_test_split

# 70% treino / 15% validação / 15% teste — estratificado por classe
df_treino, df_temp = train_test_split(
    df, test_size=0.30, stratify=df["classe"], random_state=SEED
)
df_val, df_teste = train_test_split(
    df_temp, test_size=0.50, stratify=df_temp["classe"], random_state=SEED
)

print(f"Treino:    {len(df_treino):5d} imagens")
print(f"Validação: {len(df_val):5d} imagens")
print(f"Teste:     {len(df_teste):5d} imagens")
print("\nProporção de classes por conjunto:")
for nome, d in [("treino", df_treino), ("val", df_val), ("teste", df_teste)]:
    props = (d["classe"].value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"  {nome:7s}: {props}")
""")

# =====================================================================
# CÉLULA 9 — Pré-processamento / geradores (markdown)
# =====================================================================
md(r"""
## 1.4 Pipeline de pré-processamento e *data augmentation*

**Decisões técnicas e justificativas:**

| Etapa | Escolha | Por quê |
|---|---|---|
| **Redimensionamento** | **224×224** px | Tamanho de entrada padrão da ResNet50/VGG16 (treinadas em ImageNet 224×224). Mantém compatibilidade com Transfer Learning e reduz custo computacional. |
| **Canais** | **RGB (3 canais)** | A ResNet50 espera 3 canais; convertemos as radiografias (tons de cinza) para RGB replicando o canal. |
| **Normalização** | `rescale=1/255` (CNN do zero) / `preprocess_input` (ResNet50) | Cada arquitetura exige sua normalização específica — usar a errada degrada muito o Transfer Learning. |
| ***Data augmentation*** | rotação ≤12°, flip horizontal, zoom ≤10%, brilho 0.9–1.1, deslocamento ≤8% | Aumenta a robustez **sem distorcer a anatomia**. Evitamos flip vertical e rotações grandes (um coração de cabeça para baixo não é um raio-X plausível). |

> **Médico-seguro:** as transformações simulam variações reais de posicionamento/equipamento,
> mas preservam a orientação anatômica e as proporções do tórax.
""")

code(r"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)   # entrada padrão ResNet50/VGG16
BATCH = 32

def construir_geradores(preprocess_mode: str):
    '''Cria geradores treino/val/teste.

    preprocess_mode='rescale'  -> normalização 1/255 (CNN do zero)
    preprocess_mode='resnet'   -> resnet50.preprocess_input (Transfer Learning)

    Augmentation é aplicado APENAS no treino. Val e teste recebem só a
    normalização (nunca augmentation, para avaliação imparcial).
    '''
    if preprocess_mode == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        comum = dict(preprocessing_function=preprocess_input)
    else:
        comum = dict(rescale=1.0 / 255)

    aug_treino = ImageDataGenerator(
        rotation_range=12,        # pequenas rotações (posicionamento)
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        brightness_range=(0.9, 1.1),
        horizontal_flip=True,     # raio-X espelhado ainda é plausível
        fill_mode="nearest",
        **comum,
    )
    gen_eval = ImageDataGenerator(**comum)  # sem augmentation

    treino = aug_treino.flow_from_dataframe(
        df_treino, x_col="caminho", y_col="classe",
        target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="binary", color_mode="rgb",
        classes=["normal", "cardiomegalia"],  # 0=normal, 1=cardiomegalia (fixo)
        shuffle=True, seed=SEED,
    )
    val = gen_eval.flow_from_dataframe(
        df_val, x_col="caminho", y_col="classe",
        target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="binary", color_mode="rgb",
        classes=["normal", "cardiomegalia"], shuffle=False,
    )
    teste = gen_eval.flow_from_dataframe(
        df_teste, x_col="caminho", y_col="classe",
        target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="binary", color_mode="rgb",
        classes=["normal", "cardiomegalia"], shuffle=False,
    )
    return treino, val, teste

# Geradores para a CNN do zero (normalização 1/255)
treino_cnn, val_cnn, teste_cnn = construir_geradores("rescale")
print("\nÍndices de classe:", treino_cnn.class_indices)
""")

code(r"""
# --- Visualiza o efeito do data augmentation em uma imagem de treino ---
batch_imgs, batch_lbls = next(treino_cnn)
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for i in range(5):
    axes[i].imshow(batch_imgs[i])
    rotulo = "cardiomegalia" if batch_lbls[i] == 1 else "normal"
    axes[i].set_title(rotulo, fontsize=10)
    axes[i].axis("off")
plt.suptitle("Exemplos de treino após augmentation (médico-seguro)", y=1.05)
plt.tight_layout()
plt.show()
treino_cnn.reset()  # reseta o gerador após o next()
""")

code(r"""
# --- class_weight: compensa desbalanceamento penalizando mais a classe rara ---
from sklearn.utils.class_weight import compute_class_weight

y_treino = df_treino["classe"].map({"normal": 0, "cardiomegalia": 1}).values
pesos = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_treino)
class_weight = {0: float(pesos[0]), 1: float(pesos[1])}
print("class_weight =", class_weight)
print("(>1 para a classe minoritária — o modelo 'presta mais atenção' nela)")
""")

# =====================================================================
# CÉLULA 10 — CNN do zero (markdown)
# =====================================================================
md(r"""
---
# 🧠 Parte 2 — Classificação com CNN

## 2.1 Modelo 1 — CNN simples treinada do zero

Arquitetura **deliberadamente simples** (baseline), para servir de comparação ao
Transfer Learning. Componentes e justificativas:

- **3 blocos convolucionais** (`Conv2D + BatchNormalization + MaxPooling`) com filtros
  crescentes (32 → 64 → 128): capturam padrões de complexidade crescente (bordas →
  texturas → estruturas).
- **`BatchNormalization`**: estabiliza e acelera o treino.
- **`GlobalAveragePooling2D`**: reduz parâmetros vs. `Flatten` (menos *overfitting*).
- **`Dropout(0.5)`**: regularização contra *overfitting*.
- **Saída `Dense(1, sigmoid)`**: probabilidade de cardiomegalia (classificação binária).
""")

code(r"""
def criar_cnn_simples(input_shape=(224, 224, 3)):
    '''CNN básica treinada do zero (baseline).'''
    modelo = models.Sequential(name="CNN_do_zero")
    modelo.add(layers.Input(shape=input_shape))

    for filtros in (32, 64, 128):
        modelo.add(layers.Conv2D(filtros, (3, 3), padding="same", activation="relu"))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.MaxPooling2D((2, 2)))

    modelo.add(layers.GlobalAveragePooling2D())
    modelo.add(layers.Dense(128, activation="relu"))
    modelo.add(layers.Dropout(0.5))
    modelo.add(layers.Dense(1, activation="sigmoid"))  # binária

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return modelo

cnn = criar_cnn_simples()
cnn.summary()
""")

# =====================================================================
# CÉLULA 11 — Callbacks + treino CNN
# =====================================================================
md(r"""
### Callbacks: *EarlyStopping*, *ModelCheckpoint* e *ReduceLROnPlateau*

- **EarlyStopping**: para o treino quando a `val_loss` não melhora (evita *overfitting*).
- **ModelCheckpoint**: salva automaticamente o **melhor** modelo (menor `val_loss`).
- **ReduceLROnPlateau**: reduz a taxa de aprendizado quando o treino estagna.
""")

code(r"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

MODELOS_DIR = Path("modelos"); MODELOS_DIR.mkdir(exist_ok=True)

def callbacks_para(caminho_modelo):
    return [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(caminho_modelo, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

# Em CPU o treino é lento; reduza EPOCHS se necessário.
EPOCHS = 25

hist_cnn = cnn.fit(
    treino_cnn,
    validation_data=val_cnn,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks_para(str(MODELOS_DIR / "cnn_do_zero.keras")),
    verbose=1,
)
""")

# =====================================================================
# CÉLULA 12 — Funções de avaliação (markdown + util)
# =====================================================================
md(r"""
## 2.3 Funções utilitárias de avaliação

Definimos funções reutilizáveis para avaliar **qualquer** modelo de forma consistente:
curvas de treino, matriz de confusão, *classification report* e curva ROC/AUC.
""")

code(r"""
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score, accuracy_score)

CLASSES = ["normal", "cardiomegalia"]

def plotar_curvas(hist, titulo):
    '''Curvas de loss e accuracy (treino vs. validação).'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(hist.history["loss"], label="treino")
    ax1.plot(hist.history["val_loss"], label="validação")
    ax1.set_title(f"{titulo} — Loss"); ax1.set_xlabel("época"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(hist.history["accuracy"], label="treino")
    ax2.plot(hist.history["val_accuracy"], label="validação")
    ax2.set_title(f"{titulo} — Acurácia"); ax2.set_xlabel("época"); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout(); plt.show()

def avaliar_modelo(modelo, gerador_teste, titulo):
    '''Avaliação completa no conjunto de TESTE. Retorna dict de métricas.'''
    gerador_teste.reset()
    y_true = gerador_teste.classes
    y_prob = modelo.predict(gerador_teste, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n===== {titulo} — Conjunto de TESTE =====")
    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {prec:.4f}")
    print(f"Recall   : {rec:.4f}  (sensibilidade — crítico em saúde)")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    # Matriz de confusão + curva ROC lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax1, cmap="Blues", colorbar=False)
    ax1.set_title(f"{titulo} — Matriz de confusão")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="gray")
    ax2.set_xlabel("Falso Positivo"); ax2.set_ylabel("Verdadeiro Positivo")
    ax2.set_title(f"{titulo} — Curva ROC"); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    return {"modelo": titulo, "acuracia": acc, "precisao": prec,
            "recall": rec, "f1": f1, "auc": auc}
""")

code(r"""
# --- Avaliação da CNN do zero ---
plotar_curvas(hist_cnn, "CNN do zero")
metricas_cnn = avaliar_modelo(cnn, teste_cnn, "CNN do zero")
""")

# =====================================================================
# CÉLULA 13 — Transfer Learning (markdown)
# =====================================================================
md(r"""
## 2.4 Modelo 2 — Transfer Learning com ResNet50

**Por que ResNet50 (e não VGG16)?**

- **Conexões residuais** (*skip connections*) permitem redes profundas sem o problema do
  gradiente que desaparece → melhor extração de características.
- **Menos parâmetros** que a VGG16 (~25M vs. ~138M) → treino mais rápido e menor risco de
  *overfitting* em datasets médicos pequenos.
- Pré-treinada no **ImageNet**: já aprendeu bordas, texturas e formas reaproveitáveis
  para radiografias.

**Estratégia em 2 fases:**

1. **Extração de características** — congelamos a base ResNet50 e treinamos só a "cabeça"
   (camadas densas novas). Rápido e estável.
2. ***Fine-tuning* leve** — descongelamos os últimos blocos convolucionais e treinamos com
   taxa de aprendizado **baixa** (1e-5) para adaptar características de alto nível ao
   domínio médico, sem destruir o conhecimento pré-treinado.

> **Atenção:** a ResNet50 exige `preprocess_input` específico (não `1/255`). Por isso
> recriamos os geradores com `preprocess_mode='resnet'`.
""")

code(r"""
# --- Geradores com pré-processamento da ResNet50 ---
treino_rn, val_rn, teste_rn = construir_geradores("resnet")
""")

code(r"""
from tensorflow.keras.applications import ResNet50

def criar_resnet50(input_shape=(224, 224, 3)):
    '''ResNet50 (ImageNet, sem topo) + cabeça de classificação binária.

    Usamos a **API Funcional** com `input_tensor` (em vez de aninhar a base em um
    Sequential). Assim TODAS as camadas convolucionais ficam no MESMO grafo da saída,
    o que torna o **Grad-CAM** confiável (evita o erro 'Graph disconnected').
    '''
    inputs = layers.Input(shape=input_shape)
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False  # Fase 1: base congelada

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    modelo = models.Model(inputs, out, name="ResNet50_TransferLearning")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return modelo, base

resnet, base_resnet = criar_resnet50()
print(f"Camadas na base ResNet50: {len(base_resnet.layers)}")
resnet.summary()
""")

code(r"""
# --- FASE 1: extração de características (base congelada) ---
hist_rn1 = resnet.fit(
    treino_rn,
    validation_data=val_rn,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks_para(str(MODELOS_DIR / "resnet50_fase1.keras")),
    verbose=1,
)
""")

code(r"""
# --- FASE 2: fine-tuning leve (descongela os últimos blocos) ---
# Descongelamos as ~30 últimas camadas (blocos conv5) e treinamos com LR baixo.
base_resnet.trainable = True
for camada in base_resnet.layers[:-30]:
    camada.trainable = False

# Recompila com taxa de aprendizado MUITO menor (essencial no fine-tuning)
resnet.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

treinaveis = sum(1 for c in base_resnet.layers if c.trainable)
print(f"Camadas treináveis na base após descongelar: {treinaveis}")

hist_rn2 = resnet.fit(
    treino_rn,
    validation_data=val_rn,
    epochs=12,
    class_weight=class_weight,
    callbacks=callbacks_para(str(MODELOS_DIR / "resnet50_finetuned.keras")),
    verbose=1,
)
""")

code(r"""
# --- Une o histórico das duas fases para plotar curvas contínuas ---
class HistUnido:
    def __init__(self, *hs):
        self.history = {}
        for h in hs:
            for k, v in h.history.items():
                self.history.setdefault(k, []).extend(v)

hist_rn = HistUnido(hist_rn1, hist_rn2)
plotar_curvas(hist_rn, "ResNet50 (Transfer Learning + fine-tuning)")
metricas_rn = avaliar_modelo(resnet, teste_rn, "ResNet50")
""")

# =====================================================================
# CÉLULA 14 — Comparação (markdown + tabela)
# =====================================================================
md(r"""
## 2.5 Comparação dos modelos

Comparamos **CNN do zero** × **ResNet50 (Transfer Learning)** nas métricas de teste.
Em triagem médica, **Recall (sensibilidade)** e **AUC** costumam ser os mais importantes:
um **falso negativo** (deixar de detectar cardiomegalia) é clinicamente mais grave que um
falso positivo.
""")

code(r"""
# --- Tabela e gráfico comparativos ---
comp = pd.DataFrame([metricas_cnn, metricas_rn]).set_index("modelo")
comp_pct = (comp * 100).round(2)
print(comp_pct.to_string())

ax = comp_pct.plot(kind="bar", figsize=(11, 5), rot=0)
ax.set_title("Comparação de métricas (%) — Teste")
ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.legend(loc="lower right")
ax.grid(axis="y", alpha=.3)
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.1f", fontsize=7)
plt.tight_layout(); plt.show()

melhor_nome = comp["auc"].idxmax()
print(f"\n🏆 Melhor modelo por AUC: {melhor_nome}")
""")

# =====================================================================
# CÉLULA 15 — Grad-CAM (markdown + code)
# =====================================================================
md(r"""
## 2.6 Interpretabilidade — Grad-CAM

O **Grad-CAM** gera um **mapa de calor** destacando as regiões da imagem que mais
influenciaram a decisão do modelo. Em saúde, isso é essencial para **explicabilidade**:
o clínico pode verificar se o modelo "olhou" para a **região cardíaca** (e não para
artefatos irrelevantes da imagem).
""")

code(r"""
def gerar_gradcam(modelo, img_array, nome_camada="conv5_block3_out"):
    '''Calcula o heatmap Grad-CAM para a ResNet50 (modelo Funcional).

    img_array: batch (1, 224, 224, 3) já pré-processado p/ ResNet.
    Retorna heatmap normalizado [0,1] no tamanho do mapa de ativação.

    Como o modelo é Funcional, a camada convolucional alvo pertence ao mesmo
    grafo da saída — `modelo.get_layer(...)` resolve sem desconexão.
    '''
    grad_model = tf.keras.models.Model(
        modelo.inputs,
        [modelo.get_layer(nome_camada).output, modelo.output],
    )
    with tf.GradientTape() as tape:
        conv_out, predicao = grad_model(img_array)
        classe = predicao[:, 0]
    grads = tape.gradient(classe, conv_out)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pesos, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

print("✅ Função Grad-CAM pronta (usada também no protótipo Gradio).")
""")

# =====================================================================
# CÉLULA 16 — Protótipo Gradio (markdown)
# =====================================================================
md(r"""
---
# 🎛️ Protótipo de apresentação — Gradio

Interface interativa para **demonstração clínica simulada**:

1. O usuário faz **upload** de uma radiografia de tórax.
2. A imagem passa pelo **mesmo pré-processamento** do treino (224×224 + `preprocess_input`).
3. O modelo **ResNet50** (melhor desempenho) retorna:
   - **predição**: *Cardiomegalia detectada* / *Não detectada*;
   - **nível de confiança** (probabilidade);
   - **mapa de calor Grad-CAM** sobreposto, mostrando as regiões de atenção.

> ⚠️ **Aviso exibido na interface:** protótipo educacional, não substitui avaliação médica.
""")

code(r"""
!pip install -q gradio
""")

code(r"""
import gradio as gr
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

MODELO_DEMO = resnet  # melhor modelo (ResNet50)

def preparar_imagem(img_pil):
    '''Replica EXATAMENTE o pré-processamento do treino (ResNet50).'''
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr_pre = preprocess_input(np.expand_dims(arr.copy(), axis=0))
    return img, arr, arr_pre

def sobrepor_gradcam(img_rgb_uint8, heatmap):
    '''Sobrepõe o heatmap Grad-CAM na imagem original.'''
    hm = cv2.resize(heatmap, IMG_SIZE)
    hm = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb_uint8.astype("uint8"), 0.6, hm_color, 0.4, 0)

def classificar(img_pil):
    img, arr, arr_pre = preparar_imagem(img_pil)
    prob = float(MODELO_DEMO.predict(arr_pre, verbose=0).ravel()[0])

    if prob >= 0.5:
        rotulo = "🔴 Cardiomegalia detectada"
        confianca = prob
    else:
        rotulo = "🟢 Cardiomegalia NÃO detectada"
        confianca = 1 - prob

    rotulos = {"Cardiomegalia": prob, "Normal": 1 - prob}

    # Grad-CAM sobreposto
    try:
        heatmap = gerar_gradcam(MODELO_DEMO, arr_pre)
        overlay = sobrepor_gradcam(np.array(img), heatmap)
    except Exception as e:
        print("Grad-CAM indisponível:", e)
        overlay = np.array(img)

    texto = (f"**{rotulo}**\n\n"
             f"Confiança: **{confianca*100:.1f}%**\n\n"
             f"_Protótipo educacional — não substitui avaliação médica._")
    return rotulos, overlay, texto

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
    allow_flagging="never",
)

demo.launch(share=True, debug=False)
""")

# =====================================================================
# CÉLULA 17 — Salvar modelos (markdown + code)
# =====================================================================
md(r"""
## Persistência dos modelos

Os melhores modelos já são salvos automaticamente pelo `ModelCheckpoint`. A célula abaixo
confirma os arquivos salvos e (opcionalmente) os copia para o Google Drive.
""")

code(r"""
print("Modelos salvos em:", MODELOS_DIR.resolve())
for f in sorted(MODELOS_DIR.glob("*.keras")):
    print("  ", f.name, f"({f.stat().st_size/1e6:.1f} MB)")

# Para salvar no Google Drive (opcional), descomente:
# from google.colab import drive
# drive.mount('/content/drive')
# shutil.copytree(MODELOS_DIR, '/content/drive/MyDrive/CardioIA_modelos', dirs_exist_ok=True)
""")

# =====================================================================
# CÉLULA 18 — Relatório técnico (markdown)
# =====================================================================
md(r"""
---
# 📄 Relatório Técnico — CardioIA Fase 4

> Documento de 1–2 páginas resumindo escolhas, pipeline, resultados e limitações.
> *Os números de desempenho devem ser preenchidos com os resultados da sua execução
> (ver tabela comparativa da seção 2.5).*

## 1. Introdução e objetivo
O **CardioIA** é um Assistente Cardiológico Virtual. Nesta Fase 4 aplicamos **Visão
Computacional** para detectar **cardiomegalia** em radiografias de tórax, demonstrando um
pipeline completo de **pré-processamento + classificação com CNN** como apoio à decisão
clínica.

## 2. Dataset escolhido e justificativa
Utilizamos o dataset **Cardiomegaly Disease Prediction Using CNN** (Kaggle, derivado do
**NIH Chest X-ray**). A escolha se justifica pela **relevância cardiológica direta**, pela
**tarefa binária bem definida** (Cardiomegalia × Normal) e pelo **tamanho gerenciável** no
Colab.

## 3. Pipeline de pré-processamento
- **Detecção automática** da estrutura de pastas e consolidação em DataFrame.
- **Redimensionamento** para **224×224** (compatível com ResNet50/VGG16).
- **Conversão para RGB** (3 canais exigidos pela ResNet50).
- **Normalização** específica por modelo (`1/255` para CNN do zero; `preprocess_input`
  para ResNet50).
- ***Data augmentation* médico-seguro** (rotação ≤12°, flip horizontal, zoom ≤10°,
  brilho 0.9–1.1) aplicado **apenas no treino**.
- **Divisão estratificada** 70% / 15% / 15% (treino / validação / teste).
- **`class_weight`** para compensar eventual desbalanceamento.

## 4. Metodologia
Comparamos duas abordagens:
- **CNN do zero** (3 blocos conv + BatchNorm + GAP + Dropout) — *baseline*.
- **Transfer Learning ResNet50** (ImageNet) em 2 fases: extração de características (base
  congelada) + *fine-tuning* leve (últimos blocos, LR=1e-5).

Treinamento com **Adam**, **binary_crossentropy** e *callbacks* **EarlyStopping /
ModelCheckpoint / ReduceLROnPlateau**.

## 5. Resultados e análise
| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| CNN do zero | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| ResNet50    | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |

*Análise esperada:* o Transfer Learning (ResNet50) tende a superar a CNN do zero,
especialmente em **AUC** e **Recall**, por reaproveitar características visuais já
aprendidas no ImageNet — vantagem relevante em datasets médicos de tamanho limitado.
A **matriz de confusão** e a **curva ROC** detalham o equilíbrio entre falsos positivos e
falsos negativos.

## 6. Limitações e considerações éticas
- **Dataset simulado/limitado:** não representa toda a diversidade populacional, de
  equipamentos e de posicionamentos clínicos.
- **Possíveis vieses:** desbalanceamento de classes e rótulos derivados automaticamente
  (NIH) podem conter ruído; o modelo pode aprender atalhos espúrios.
- **Não é dispositivo médico:** o protótipo **não** deve ser usado para diagnóstico real;
  serve para fins educacionais e de pesquisa.
- **Responsabilidade e privacidade:** dados de saúde exigem consentimento, anonimização e
  conformidade (LGPD). Decisões clínicas devem permanecer com o profissional de saúde
  (*human-in-the-loop*).
- **Explicabilidade:** o **Grad-CAM** mitiga parcialmente o problema de "caixa-preta",
  permitindo auditar onde o modelo focou.

## 7. Conclusão e próximos passos
O protótipo demonstra, de ponta a ponta, como a Visão Computacional pode apoiar a triagem
cardiológica. **Próximos passos para o CardioIA:**
- Validar com datasets maiores e **multi-institucionais**;
- Avaliar **fairness** por subgrupos (Ir Além 1);
- Integrar a um **app mobile** (React Native, Ir Além 2);
- Calibrar probabilidades e definir limiares por custo clínico;
- Submeter a validação clínica supervisionada antes de qualquer uso real.

---
### Referências
- NIH Chest X-ray Dataset — https://www.kaggle.com/datasets/nih-chest-xrays/data
- Dataset utilizado — https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn
- He et al. (2016), *Deep Residual Learning for Image Recognition* (ResNet).
- Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks*.
""")


# =====================================================================
# Monta e grava o notebook
# =====================================================================
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

saida = os.path.join(os.path.dirname(__file__), "..", "notebooks", "CardioIA_Fase4.ipynb")
saida = os.path.abspath(saida)
os.makedirs(os.path.dirname(saida), exist_ok=True)
with open(saida, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook gerado: {saida}")
print(f"Total de células: {len(CELLS)}  "
      f"(markdown: {sum(1 for c in CELLS if c['cell_type']=='markdown')}, "
      f"code: {sum(1 for c in CELLS if c['cell_type']=='code')})")
