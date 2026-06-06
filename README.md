# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/">
  <img src="assets/logo-fiap.png" 
       alt="FIAP - Faculdade de Informática e Administração Paulista" 
       width="40%">
</a>
</p>

<br>

# 🫀 CardioIA - Fase 4: Assistente Cardiológico Virtual com Visão Computacional

## Nome do grupo

### Grupo São Paulo e Interior

## Integrantes

- <a href="https://www.linkedin.com/in/jonastadeufernandes">Jonas Tadeu V. Fernandes - RM563027</a>
- <a href="https://www.linkedin.com/">Levi Passos Silveira Marques - RM56557</a>
- <a href="https://www.linkedin.com/in/raphaelsilva-phael">Raphael da Silva - RM561452</a>
- <a href="https://www.linkedin.com/in/raphael-dinelli-8a01b278">Raphael Dinelli Neto - RM562892</a>
- <a href="https://www.linkedin.com/in/yan-cotta">Yan Pimental Cotta - RM562836</a>

## Professores

### Tutor

- <a href="https://www.linkedin.com/in/caique-nonato/">Caique Nonato da Silva Bezerra</a>

### Coordenador

- <a href="https://www.linkedin.com/in/andregodoichiovato">André Godoi</a>

## 🎯 A Proposta: Visão Computacional no apoio à decisão clínica

Após estruturar o monitoramento contínuo nas fases anteriores, o **CardioIA** avança para a
**análise de imagens médicas** com Visão Computacional. Esta Fase 4 entrega um protótipo que
transforma **radiografias de tórax** em informação interpretável, classificando a presença de
**Cardiomegalia** (aumento da área cardíaca) — uma condição cardiológica detectável em raio-X.

O projeto demonstra, de ponta a ponta, um **pipeline de Visão Computacional aplicado à saúde**:
pré-processamento de imagens, treino e avaliação de **CNNs**, **explicabilidade** (Grad-CAM) e
um **protótipo interativo**, sempre com atenção a **limitações, vieses e ética** no uso de IA
em dados médicos.

> ⚠️ Protótipo **educacional**. Não é dispositivo médico e **não** deve ser usado para
> diagnóstico real.

## ⚙️ Arquitetura e Tecnologias Utilizadas

A solução foi arquitetada para ser reproduzível no **Google Colab** (GPU gratuita) e também
localmente em **Apple Silicon (Metal)**, cumprindo os requisitos da disciplina:

* **Linguagem Principal:** Python
* **Deep Learning / Visão Computacional:** TensorFlow / Keras (CNN do zero + Transfer Learning ResNet50)
* **Métricas e avaliação:** scikit-learn (acurácia, matriz de confusão, precisão, recall, F1, AUC)
* **Explicabilidade:** Grad-CAM (mapa de calor das regiões de atenção do modelo)
* **Protótipo de interface:** Gradio (upload → predição + confiança + Grad-CAM)
* **Dados:** Kaggle API (dataset *Cardiomegaly Disease Prediction* — derivado do NIH Chest X-ray)
* **Ir Além — Mobile:** App React Native (Expo) + backend Flask servindo o modelo treinado

### Como a solução atende aos critérios da Fase 4:

| Critério (10 pts) | Onde está |
|---|---|
| **Pipeline de pré-processamento (3)** | `notebooks/CardioIA_Fase4.ipynb` — Parte 1: redimensionamento 224×224, RGB, normalização por modelo, *data augmentation* médico-seguro, split estratificado 70/15/15, `class_weight` |
| **CNN do zero — treino + avaliação (2)** | Notebook — Parte 2.1: 3 blocos Conv+BatchNorm+Pooling + métricas |
| **Transfer Learning funcional (2)** | Notebook — Parte 2.4: ResNet50 (ImageNet) em 2 fases (extração + *fine-tuning*) |
| **Protótipo de apresentação (2)** | Notebook — Gradio (upload → predição + Grad-CAM) |
| **Documentação clara (1)** | `docs/relatorio_tecnico.md` + este README |
| **Trabalho em equipe (1 extra)** | Grupo de 5 integrantes (acima) |
| **Ir Além 1 — Ética e *fairness*** | `docs/ir_alem1_fairness.md` + seção de fairness no notebook |
| **Ir Além 2 — App mobile** | `mobile/` (React Native + Flask) |

## 📊 Resultados (execução real)

Pipeline **executado de fato** (não apenas escrito): treino **local** na GPU **Apple M5 Pro
(Metal)**, TensorFlow 2.16.2, `seed=42`, dataset com **5.552 imagens balanceadas** (split
70/15/15). Métricas no conjunto de **teste** (833 imagens):

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| CNN do zero | 0.611 | 0.666 | 0.448 | 0.536 | 0.639 |
| **ResNet50 (Transfer Learning)** | **0.733** | **0.708** | **0.796** | **0.749** | **0.806** |

> O **Transfer Learning superou a CNN do zero em todas as métricas**, com destaque no
> **Recall (0.80 vs. 0.45)** e na **AUC (0.81 vs. 0.64)** — o ganho de sensibilidade é o mais
> relevante clinicamente. Detalhes de matriz de confusão e *fairness* em
> [`docs/ir_alem1_fairness.md`](docs/ir_alem1_fairness.md).

## 📦 Entregáveis — o que foi feito e como

Todos os itens abaixo estão **implementados e executados de verdade** (modelos treinados e
versionados no repositório). **Falta apenas gravar o vídeo** do Ir Além 2.

| # | Entregável | Status | Onde |
|---|---|---|---|
| 1 | Pré-processamento (Parte 1) | ✅ feito | notebook · Parte 1 |
| 2 | CNN do zero (Parte 2) | ✅ feito | notebook · Parte 2.1 |
| 3 | Transfer Learning (Parte 2) | ✅ feito | notebook · Parte 2.4 |
| 4 | Protótipo de apresentação (Parte 2) | ✅ feito | Gradio (notebook + `scripts/app_gradio.py`) |
| 5 | Documentação | ✅ feito | este README + `docs/` |
| 6 | Trabalho em equipe (extra) | ✅ feito | 5 integrantes |
| 7 | Ir Além 1 — Ética/*fairness* | ✅ feito | `docs/ir_alem1_fairness.md` + notebook |
| 8 | Ir Além 2 — App mobile | ✅ feito (código) · ⏳ **vídeo pendente** | `mobile/` |

### 🅰️ Parte 1 — Pré-processamento e organização das imagens
**O que:** preparar o dataset público de raio-X de tórax para treino/validação/teste.
**Como** (`notebooks/CardioIA_Fase4.ipynb`, Parte 1):
- Download via **Kaggle API** e **detecção automática** da estrutura de pastas (`true`/`false`),
  consolidando tudo em um DataFrame `(caminho, classe)`.
- **Redimensionamento** para **224×224** e **conversão para RGB** (3 canais exigidos pela ResNet50).
- **Normalização específica por modelo:** `1/255` (CNN do zero) e `resnet50.preprocess_input` (TL).
- ***Data augmentation* médico-seguro** (rotação ≤12°, flip horizontal, zoom ≤10%, brilho
  0.9–1.1, deslocamentos ≤8%) aplicado **apenas no treino**, preservando a anatomia.
- **Divisão estratificada 70/15/15** + **`class_weight`** balanceado.

### 🅱️ Parte 2 — Classificação de imagens médicas com CNN
**O que:** treinar, avaliar e comparar duas abordagens, e apresentar em um protótipo.
**Como** (notebook, Parte 2):
- **CNN do zero** (baseline): 3 blocos `Conv2D + BatchNorm + MaxPooling` (32→64→128) +
  `GlobalAveragePooling2D` + `Dense(128)` + `Dropout(0.5)` + saída sigmoide.
- **Transfer Learning ResNet50** (ImageNet) em **2 fases**: extração de características (base
  congelada) + *fine-tuning* leve (últimos ~30 layers, LR 1e-5), com **API Funcional** para o
  **Grad-CAM** ser confiável.
- **Avaliação:** acurácia, precisão, recall, F1, **AUC**, **matriz de confusão** e **curva ROC**.
- **Protótipo de apresentação:** interface **Gradio** (upload → predição + confiança +
  **Grad-CAM**) — no notebook e também como app standalone (`scripts/app_gradio.py`, dockerizado).
- **Resultado:** a ResNet50 superou a CNN do zero em todas as métricas (tabela acima).

### ➕ Ir Além 1 — Ética, governança e *fairness*
**O que:** identificar vieses/limitações do dataset e medir o desempenho **por grupo**.
**Como** (`docs/ir_alem1_fairness.md` + seção *"Ir Além 1"* no notebook):
- Limitações: dataset **balanceado em classes**, porém **sem metadados demográficos** (impede
  auditar fairness por sexo/idade).
- Métricas de fairness por classe: **FNR/FPR**, recall por grupo e **gap de oportunidade igual**
  (0.126) — com matriz de confusão da ResNet50 no teste.
- Mitigações **aplicadas** (`class_weight`, split estratificado, Grad-CAM) e **recomendadas**
  (ajuste de limiar por custo clínico, coleta de metadados, validação multi-institucional).

### ➕ Ir Além 2 — App mobile (React Native + Flask)
**O que:** levar a classificação da CNN para um protótipo mobile.
**Como** (`mobile/` — ver [`mobile/README.md`](mobile/README.md)):
- **App React Native (Expo)** (`mobile/app`): tela de **upload** → envia ao backend → exibe a
  **classe**, a **confiança** e o **Grad-CAM**.
- **Backend Flask** (`mobile/backend`): serve o **modelo ResNet50 treinado** (`POST /predict`,
  `GET /health`), devolvendo predição + Grad-CAM em base64.
- **Validado end-to-end** com o modelo real (cardiomegalia → 0.81 · normal → 0.26 · Grad-CAM OK).
- ⏳ **Único item pendente da entrega:** gravar o **vídeo de até 3 min** demonstrando o app.

## 📁 Estrutura de pastas

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:

- <b>notebooks</b>: Notebook principal (`CardioIA_Fase4.ipynb`) — pipeline ponta a ponta
  (pré-processamento, CNN do zero, Transfer Learning, métricas, fairness e protótipo Gradio).

- <b>scripts</b>: Código-fonte auxiliar — `gerar_notebook.py` (constrói o `.ipynb`, só stdlib),
  `treinar_local.py` (treino headless que gera os modelos), `app_gradio.py` (demo Gradio
  standalone), `setup_local_m5.sh` (cria o ambiente conda + Metal) e `smoke_test_metal.py`
  (valida a GPU).

- <b>docs</b>: Documentação textual — `relatorio_tecnico.md` (relatório técnico),
  `ir_alem1_fairness.md` (relatório de ética e *fairness*) e `resultados_run.json` (métricas
  da execução real).

- <b>modelos</b>: Modelos treinados **versionados** — `cnn_do_zero.keras` e
  `resnet50_finetuned.keras` (permitem rodar a demo/app **sem re-treinar**).

- <b>mobile</b>: Protótipo mobile (Ir Além 2) — app React Native (Expo) em `mobile/app` e
  backend Flask em `mobile/backend`.

- <b>Dockerfile · docker-compose.yml · .dockerignore</b>: Empacotamento Docker (backend,
  demo Gradio e treino) para rodar em qualquer máquina.

- <b>requirements.txt</b>: Dependências do notebook/pipeline.

- <b>README.md</b>: Arquivo que serve como guia e explicação geral sobre o projeto (o mesmo que
  você está lendo agora).

## 🔧 Como executar o código

### Opção A — Google Colab (recomendado)

1. Abra `notebooks/CardioIA_Fase4.ipynb` no [Google Colab](https://colab.research.google.com/).
2. **Runtime → Alterar tipo de ambiente de execução → T4 GPU**.
3. Tenha em mãos seu token do Kaggle (`kaggle.json` ou `access_token`).
4. **Rode as células em ordem.** A primeira força o **Keras 2 legado** (compatibilidade com
   `ImageDataGenerator`); a célula do Kaggle pedirá o upload da credencial.
5. Ao final, a célula do **Gradio** abre uma interface com link público para demonstração.

> O notebook é gerado por `scripts/gerar_notebook.py`. Para editar o conteúdo, altere o script
> e rode `python3 scripts/gerar_notebook.py` (não requer dependências externas).

### Opção B — Local (Apple Silicon / GPU Metal)

Foi como **treinamos de fato** este projeto (Apple **M5 Pro**, 48 GB, GPU **Metal**):

```bash
# cria o ambiente conda com TensorFlow + Metal
bash scripts/setup_local_m5.sh
conda run -n cardioia python scripts/smoke_test_metal.py     # valida a GPU Metal
conda run -n cardioia python scripts/treinar_local.py        # treina e gera modelos/ + docs/resultados_run.json
```

> Em Apple Silicon usamos `tf.keras.optimizers.legacy.Adam` (o `Adam` novo trava o grafo no
> `tensorflow-metal`). O mesmo código roda no Colab sem alterações.

### Opção C — Docker (roda em qualquer máquina) 🐳

Imagens **CPU portáteis** (sem dependência de GPU). Requer um modelo treinado em
`./modelos/resnet50_finetuned.keras` (gerado pela Opção A/B, ou pelo serviço `train`).

```bash
# Sobe a API Flask (porta 5050) + a demo web Gradio (porta 7860)
docker compose up backend gradio

# (Opcional) (re)treinar dentro do container — requer o dataset em ./cardio_data
docker compose --profile training run --rm train
```

- **Backend:** http://localhost:5050/health · **Demo Gradio:** http://localhost:7860

### Ir Além 2 — App mobile

Veja as instruções em [`mobile/README.md`](mobile/README.md) (subir o backend Flask e rodar o
app Expo).

## 📎 Links e Observações

- <b>Listagem de Links</b>:
  - <a href="https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn">Dataset utilizado (Kaggle)</a>
  - <a href="https://www.kaggle.com/datasets/nih-chest-xrays/data">NIH Chest X-ray Dataset (origem)</a>
  - Vídeo de demonstração (Ir Além 2): _a inserir_

- <b>Explicação de decisões técnicas</b>:
  - As **credenciais do Kaggle** (`kaggle.json` / `access_token`) **não** são versionadas no
    repositório (estão no `.gitignore`); cada integrante usa o próprio token.
  - O **dataset** (`cardio_data/`) fica fora do Git por ser grande e **regenerável** via
    Kaggle. Já os **modelos finais** (`modelos/*.keras`) **são versionados** para que a demo e o
    app rodem **sem re-treinar** (o modelo intermediário de fase 1 é ignorado por ser redundante).
  - Fixamos o **Keras 2 legado** (`tf-keras`) para manter o `ImageDataGenerator` funcionando no
    Keras 3 do Colab, e usamos `tf.keras.optimizers.legacy.Adam` (compatível com o
    `tensorflow-metal` do Apple Silicon e com o Colab).
  - Pin de `gradio>=4,<5` e `huggingface_hub<1.0` para o protótipo não quebrar com versões novas.

## 🗃 Histórico de lançamentos

* 0.1.0 - 06/06/2026

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/SabrinaOtoni/TEMPLATE-FIAP-GRAD-ON-IA">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">FIAP</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
