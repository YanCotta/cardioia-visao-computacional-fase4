# 🫀 CardioIA — Fase 4: Visão Computacional

Protótipo do **CardioIA Fase 4** — classificação de **cardiomegalia** em radiografias de
tórax com Visão Computacional. Pipeline completo de pré-processamento, **CNN do zero +
Transfer Learning (ResNet50)** e protótipo interativo com **Gradio** (upload → predição +
confiança + Grad-CAM).

## 📂 Estrutura do repositório

```text
.
├── notebooks/
│   └── CardioIA_Fase4.ipynb      # Notebook principal (Colab) — ponta a ponta
├── scripts/
│   └── gerar_notebook.py         # Gera o notebook a partir do código-fonte (stdlib)
├── docs/
│   └── relatorio_tecnico.md      # Relatório técnico (1–2 páginas)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Como executar (Google Colab)

1. Abra `notebooks/CardioIA_Fase4.ipynb` no [Google Colab](https://colab.research.google.com/).
2. **Runtime → Alterar tipo de ambiente de execução → T4 GPU**.
3. Tenha em mãos seu **`kaggle.json`** (Kaggle → Settings → API → *Create New Token*).
4. **Rode as células em ordem.** A primeira célula de código força o **Keras 2 legado**
   (compatibilidade com `ImageDataGenerator`); a célula de Kaggle pedirá o upload do
   `kaggle.json`.
5. Ao final, a célula do **Gradio** abre uma interface com link público para demonstração.

> O notebook é gerado por `scripts/gerar_notebook.py`. Para editar o conteúdo, altere o
> script e rode `python3 scripts/gerar_notebook.py` (não requer dependências externas).

## ✅ Entregáveis (Core)

| Parte | Entregável | Status |
|---|---|---|
| **1** | Pipeline de pré-processamento + split treino/val/teste | ✅ no notebook |
| **2** | CNN do zero (treino + avaliação) | ✅ no notebook |
| **2** | Transfer Learning ResNet50 + fine-tuning | ✅ no notebook |
| **2** | Métricas (acurácia, matriz de confusão, precisão, recall, F1, AUC) | ✅ no notebook |
| **2** | Protótipo de apresentação (Gradio + Grad-CAM) | ✅ no notebook |
| **Doc** | Relatório técnico (1–2 páginas) | ✅ `docs/relatorio_tecnico.md` + célula final |

## ⚠️ Aviso

Protótipo **educacional**. Não é dispositivo médico e **não** deve ser usado para
diagnóstico real. Ver "Limitações e considerações éticas" no relatório.

## 📊 Dataset

[Cardiomegaly Disease Prediction Using CNN](https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn)
(Kaggle, derivado do NIH Chest X-ray).
