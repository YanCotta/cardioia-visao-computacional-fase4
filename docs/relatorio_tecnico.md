# Relatório Técnico — CardioIA Fase 4
## Assistente Cardiológico Virtual com Visão Computacional

**Tarefa:** Classificação binária de **Cardiomegalia** em radiografias de tórax
**Stack:** TensorFlow/Keras · scikit-learn · Gradio · Flask · React Native · Docker

> Os valores de desempenho na Seção 5 são de uma **execução real** do pipeline
> (treino local em GPU **Apple M5 Pro / Metal**, TensorFlow 2.16.2, `seed=42`). O mesmo
> notebook roda no **Google Colab (T4)** sem alterações.

---

### 1. Introdução e objetivo
O **CardioIA** é um Assistente Cardiológico Virtual. Nesta Fase 4 aplicamos **Visão
Computacional** para detectar **cardiomegalia** (aumento da área cardíaca) em radiografias
de tórax, demonstrando um pipeline completo de **pré-processamento + classificação com CNN**
como ferramenta de **apoio à decisão clínica**.

### 2. Dataset escolhido e justificativa
Utilizamos o dataset **Cardiomegaly Disease Prediction Using CNN** (Kaggle, derivado do
**NIH Chest X-ray**). Justificativa:
- **Relevância cardiológica direta** — alinhada ao propósito do CardioIA;
- **Tarefa binária bem definida** (Cardiomegalia × Normal), ideal para comparar CNN do zero
  vs. Transfer Learning;
- **Tamanho gerenciável** dentro do tempo de um protótipo.

São **5.552 imagens**, perfeitamente **balanceadas** (2.776 por classe). Reconsolidamos as
pastas originais (`true`/`false`) e redividimos de forma estratificada em **70/15/15**:
**3.886** treino · **833** validação · **833** teste.

### 3. Pipeline de pré-processamento
1. **Detecção automática** da estrutura de pastas (robusta a `train/test` pré-dividido ou
   pasta única) e consolidação em um DataFrame `(caminho, classe)`.
2. **Redimensionamento** para **224×224 px** — entrada padrão da ResNet50/VGG16.
3. **Conversão para RGB** (3 canais exigidos pela ResNet50).
4. **Normalização específica por modelo:** `rescale=1/255` para a CNN do zero;
   `resnet50.preprocess_input` para o Transfer Learning (usar a normalização errada degrada
   muito o desempenho).
5. ***Data augmentation* médico-seguro** — rotação ≤12°, *flip* horizontal, zoom ≤10%,
   brilho 0.9–1.1, deslocamentos ≤8% — aplicado **apenas no treino**; preserva a anatomia
   (sem *flip* vertical nem rotações grandes).
6. **Divisão estratificada** 70% / 15% / 15% (treino / validação / teste).
7. **`class_weight`** balanceado para compensar eventual desbalanceamento de classes.

### 4. Metodologia
Duas abordagens comparadas:
- **CNN do zero (baseline):** 3 blocos `Conv2D + BatchNormalization + MaxPooling`
  (32→64→128 filtros) + `GlobalAveragePooling2D` + `Dense(128)` + `Dropout(0.5)` +
  `Dense(1, sigmoid)`.
- **Transfer Learning ResNet50 (ImageNet):** em duas fases — (a) extração de características
  com a base congelada; (b) *fine-tuning* leve dos últimos ~30 layers com taxa de
  aprendizado baixa (1e-5). Implementada com **API Funcional** (`input_tensor`) para tornar
  o **Grad-CAM** confiável.

Treinamento com **Adam legado** (`tf.keras.optimizers.legacy.Adam` — recomendado em Apple
Silicon/Metal, onde o `Adam` novo trava o grafo; e compatível com o Colab), perda
**binary_crossentropy** e *callbacks* **EarlyStopping**, **ModelCheckpoint** (salva o melhor
modelo) e **ReduceLROnPlateau**.

**Ambiente de execução (run de referência):** treino **local** em **Apple M5 Pro** (48 GB
de memória unificada, GPU **Metal** via `tensorflow-metal 1.2.0`), **TensorFlow 2.16.2** +
**tf-keras 2.16** (Keras 2 legado), Python 3.11, `seed=42`. Tempos: CNN do zero ≈ **4,0 min**;
ResNet50 (2 fases) ≈ **8,6 min**. O pipeline também roda no **Colab (T4)** e, via **Docker**
(CPU), em qualquer máquina.

### 5. Resultados e análise
Resultados no conjunto de **teste** (833 imagens):

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| CNN do zero | 0.611 | 0.666 | 0.448 | 0.536 | 0.639 |
| **ResNet50 (Transfer Learning)** | **0.733** | **0.708** | **0.796** | **0.749** | **0.806** |

**Análise:** o **Transfer Learning (ResNet50) superou a CNN do zero em todas as métricas**.
O maior ganho está no **Recall (0.796 vs. 0.448)** e na **AUC (0.806 vs. 0.639)** — coerente
com a teoria: a ResNet reaproveita características visuais do ImageNet, vantagem decisiva num
dataset médico de porte moderado. O salto de **Recall (sensibilidade)** é o mais importante
clinicamente, pois um **falso negativo** (cardiomegalia não detectada) é mais grave que um
falso positivo. Matriz de confusão da ResNet no teste: **TP=332, FN=85, TN=279, FP=137** —
alta sensibilidade para a doença, ao custo de mais falsos positivos nos normais (detalhado
na análise de *fairness*, `ir_alem1_fairness.md`). A CNN do zero permanece como *baseline*
fraco, o que evidencia o valor do Transfer Learning.

### 6. Limitações e considerações éticas
- **Dataset simulado/limitado:** não cobre toda a diversidade de população, equipamentos e
  posicionamentos clínicos.
- **Vieses possíveis:** embora as **classes estejam balanceadas** (50/50), os rótulos
  derivados automaticamente (NIH) podem conter ruído e o modelo pode aprender atalhos
  espúrios (ex.: marcadores na imagem em vez da anatomia cardíaca). A **ausência de
  metadados demográficos** impede auditar *fairness* por subgrupo (ver Ir Além 1,
  `ir_alem1_fairness.md`).
- **Não é dispositivo médico:** uso estritamente educacional/pesquisa — **não** serve para
  diagnóstico real.
- **Privacidade e responsabilidade:** dados de saúde exigem consentimento, anonimização e
  conformidade (LGPD); a decisão clínica permanece com o profissional (*human-in-the-loop*).
- **Explicabilidade:** o **Grad-CAM** mitiga parcialmente a "caixa-preta", permitindo
  auditar onde o modelo focou.

### 7. Entregáveis realizados (incl. Ir Além)
Além das Partes 1 e 2 (pré-processamento, CNN do zero, Transfer Learning, métricas e protótipo
Gradio), **ambos os "Ir Além" foram implementados**:
- **Ir Além 1 — Ética e *fairness*:** relatório `ir_alem1_fairness.md` + seção de experimentos
  no notebook (FNR/FPR por classe, *equal-opportunity gap* e mitigações).
- **Ir Além 2 — App mobile:** app **React Native (Expo)** + **backend Flask** em `mobile/`,
  **validado end-to-end** com o modelo treinado (predição + Grad-CAM). Ver `mobile/README.md`.
- **Portabilidade:** pipeline empacotado em **Docker** (backend, demo Gradio e treino) — roda
  em qualquer máquina.

### 8. Próximos passos
- Validar com datasets maiores e **multi-institucionais**;
- **Calibrar probabilidades** e definir **limiares por custo clínico** (reduzir falsos negativos);
- Coletar **metadados demográficos** para auditar fairness por subgrupo;
- Submeter a **validação clínica supervisionada** antes de qualquer uso real.

---
### Referências
- NIH Chest X-ray Dataset — https://www.kaggle.com/datasets/nih-chest-xrays/data
- Dataset utilizado — https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn
- He et al. (2016), *Deep Residual Learning for Image Recognition* (ResNet).
- Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks*.
