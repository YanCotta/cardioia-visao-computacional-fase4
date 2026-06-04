# Relatório Técnico — CardioIA Fase 4
## Assistente Cardiológico Virtual com Visão Computacional

**Tarefa:** Classificação binária de **Cardiomegalia** em radiografias de tórax
**Stack:** Google Colab · TensorFlow/Keras · scikit-learn · Gradio

> ⚠️ Os valores de desempenho na Seção 5 devem ser preenchidos com os resultados da
> execução do notebook no Colab (ver tabela comparativa da Seção 2.5 do notebook).

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
- **Tamanho gerenciável** para a GPU gratuita do Colab dentro do tempo de um protótipo.

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

Treinamento com **Adam**, perda **binary_crossentropy** e *callbacks* **EarlyStopping**,
**ModelCheckpoint** (salva o melhor modelo) e **ReduceLROnPlateau**.

### 5. Resultados e análise
| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| CNN do zero | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
| ResNet50    | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |

**Análise esperada:** o Transfer Learning (ResNet50) tende a superar a CNN do zero,
especialmente em **AUC** e **Recall (sensibilidade)** — métrica crítica em saúde, pois um
**falso negativo** (cardiomegalia não detectada) é clinicamente mais grave que um falso
positivo. A **matriz de confusão** e a **curva ROC** detalham esse equilíbrio.

### 6. Limitações e considerações éticas
- **Dataset simulado/limitado:** não cobre toda a diversidade de população, equipamentos e
  posicionamentos clínicos.
- **Vieses possíveis:** desbalanceamento de classes e rótulos derivados automaticamente
  (NIH) podem conter ruído; o modelo pode aprender atalhos espúrios (ex.: marcadores na
  imagem em vez da anatomia cardíaca).
- **Não é dispositivo médico:** uso estritamente educacional/pesquisa — **não** serve para
  diagnóstico real.
- **Privacidade e responsabilidade:** dados de saúde exigem consentimento, anonimização e
  conformidade (LGPD); a decisão clínica permanece com o profissional (*human-in-the-loop*).
- **Explicabilidade:** o **Grad-CAM** mitiga parcialmente a "caixa-preta", permitindo
  auditar onde o modelo focou.

### 7. Conclusão e próximos passos
O protótipo demonstra, de ponta a ponta, como a Visão Computacional pode apoiar a triagem
cardiológica. **Próximos passos:**
- Validar com datasets maiores e **multi-institucionais**;
- Avaliar **fairness** por subgrupos (Ir Além 1);
- Integrar a um **app mobile** React Native (Ir Além 2);
- Calibrar probabilidades e definir limiares por custo clínico;
- Submeter a validação clínica supervisionada antes de qualquer uso real.

---
### Referências
- NIH Chest X-ray Dataset — https://www.kaggle.com/datasets/nih-chest-xrays/data
- Dataset utilizado — https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn
- He et al. (2016), *Deep Residual Learning for Image Recognition* (ResNet).
- Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks*.
