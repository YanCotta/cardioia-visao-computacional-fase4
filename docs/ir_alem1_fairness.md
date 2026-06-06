# Ir Além 1 — Ética e Governança em Visão Computacional (Fairness)
## CardioIA Fase 4 — análise de vieses e *fairness*

**Modelo analisado:** ResNet50 (Transfer Learning), conjunto de **teste** (833 imagens).
**Run de referência:** treino local em **Apple M5 Pro / GPU Metal**, TensorFlow 2.16.2, `seed=42`.

> Os experimentos desta análise estão também no **notebook** (seção *"Ir Além 1 — Ética,
> vieses e fairness"*), reprodutíveis a cada execução.

---

### 1. Objetivo
Analisar **limitações e vieses** do dataset escolhido e medir o desempenho do modelo **por
grupo**, propondo práticas de mitigação. Em saúde, um modelo "preciso na média" pode ser
**injusto** com um subgrupo — e, em triagem, o erro mais grave é o **falso negativo**
(deixar de detectar a doença).

### 2. Limitações do dataset (representatividade)
- **Balanceamento de classes:** o dataset é **balanceado** (2.776 cardiomegalia / 2.776
  normal — 50/50). Isso **elimina o viés de prevalência** no treino (o `class_weight`
  resultou em `{0: 1.0, 1: 1.0}`), mas **não** garante representatividade clínica real, onde
  a cardiomegalia costuma ser minoritária.
- **Ausência de metadados demográficos:** o dataset **não traz sexo, idade, etnia ou
  equipamento por imagem**. Logo, **não é possível auditar fairness por subgrupo
  demográfico** — esta é, por si só, uma limitação de representatividade a ser reportada.
- **Rótulos derivados automaticamente (NIH):** podem conter ruído; o modelo pode aprender
  **atalhos espúrios** (ex.: marcadores/anotações na imagem em vez da anatomia cardíaca).
- **Origem única:** sem validação **multi-institucional**, o desempenho pode não se
  transferir para outros hospitais/aparelhos.

### 3. Métricas de *fairness* por classe
Como não há atributos demográficos, tratamos as **duas classes como os "grupos"**
mensuráveis e comparamos suas taxas de erro. Matriz de confusão da ResNet50 (teste):

|  | Previsto: normal | Previsto: cardiomegalia |
|---|---|---|
| **Real: normal** (416) | TN = 279 | FP = 137 |
| **Real: cardiomegalia** (417) | FN = 85 | TP = 332 |

| Grupo (classe) | Recall / TPR | Taxa de erro | Tipo de erro |
|---|---|---|---|
| **normal** | 0.671 (especificidade) | **0.329** | Falso Positivo (FPR) |
| **cardiomegalia** | 0.796 (sensibilidade) | **0.204** | Falso Negativo (FNR) |

- **Gap de oportunidade igual** (|TPR_cardio − TNR_normal|) = **0.1255**.

### 4. Interpretação
- O modelo é **mais sensível à doença** (TPR cardiomegalia = 0.796) do que **específico para
  os normais** (TNR = 0.671). Em **triagem**, essa assimetria é até desejável — prioriza
  **não perder** casos de cardiomegalia (FNR menor, 0.204).
- O custo é um **FPR de 0.329**: cerca de **1 em cada 3 exames normais** recebe um alarme
  falso, o que geraria exames de confirmação desnecessários.
- O **gap de 0.126 (> 0.10)** indica desempenho **não totalmente equilibrado** entre os
  grupos — há espaço para calibração.

### 5. Práticas de mitigação
**Aplicadas neste projeto:**
- **Divisão estratificada** 70/15/15 (preserva a proporção de classes em todos os conjuntos).
- **`class_weight` balanceado** no treino (neutro aqui por o dataset já ser 50/50, mas
  essencial caso a base fosse desbalanceada).
- ***Data augmentation* médico-seguro** (sem distorcer a anatomia) para reduzir *overfitting*.
- **Grad-CAM** para inspecionar se o modelo "olha" para a região cardíaca (e não para
  artefatos) — auditoria de atalhos espúrios.

**Recomendadas (próximos passos):**
- **Ajuste de limiar de decisão** (< 0.5 para a classe positiva) para reduzir ainda mais o
  FNR em cenário de triagem, aceitando trade-off no FPR — escolhido por **custo clínico**.
- **Coletar metadados demográficos** para medir fairness por sexo/idade/etnia/equipamento.
- **Validação externa multi-institucional** para checar representatividade.
- **Calibração de probabilidades** (ex.: Platt/temperature scaling) antes de qualquer uso.

### 6. Conclusão
O dataset é **balanceado em classes**, o que evita viés de prevalência, mas **carece de
metadados demográficos**, impedindo auditoria de fairness por subgrupo — limitação que deve
ser explicitada a qualquer stakeholder. O modelo prioriza **sensibilidade** (bom para
triagem), com um **gap de 0.126** entre grupos que justifica **calibração de limiar** e
**validação externa** antes de qualquer aplicação real. Reforça-se o princípio
**human-in-the-loop**: a decisão clínica permanece com o profissional de saúde.

---
### Referências
- Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks*.
- Hardt et al. (2016), *Equality of Opportunity in Supervised Learning* (equal-opportunity gap).
- NIH Chest X-ray Dataset — https://www.kaggle.com/datasets/nih-chest-xrays/data
