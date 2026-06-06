#!/usr/bin/env bash
# =====================================================================
# Setup do ambiente local para rodar o notebook no Apple Silicon (M5 Pro).
# Cria um ambiente conda dedicado (Python 3.11) com TensorFlow + Metal (GPU).
#
# Uso:
#   bash scripts/setup_local_m5.sh
#
# Por que Python 3.11: é a faixa mais compatível com tensorflow-metal em 2026
# (3.9 é antigo demais; 3.13 ainda costuma não ter wheels do metal).
# Combo de versões alinhado (TF 2.16 <-> metal 1.2.0 <-> tf-keras 2.16).
# =====================================================================
set -euo pipefail

ENV_NAME="cardioia"
PY_VER="3.11"

echo "==> Criando ambiente conda '$ENV_NAME' (Python $PY_VER) via conda-forge..."
# Usamos conda-forge com --override-channels para evitar os canais 'defaults'
# da Anaconda (que exigem aceite de Termos de Serviço) e por ser o canal
# recomendado p/ Apple Silicon.
conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PY_VER" pip

echo "==> Atualizando pip..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "==> Instalando TensorFlow + Metal (GPU) e dependências..."
# Combo conhecido-compatível. Se houver erro de versão, troque por:
#   tensorflow tensorflow-metal   (sem pin, deixa o pip resolver) e rode o smoke test.
conda run -n "$ENV_NAME" python -m pip install \
    "tensorflow==2.16.2" \
    "tensorflow-metal==1.2.0" \
    "tf-keras==2.16.0" \
    "scikit-learn>=1.3" "pandas>=2.0" "numpy<2.0" \
    "matplotlib>=3.7" "pillow>=10.0" "opencv-python>=4.8" \
    "gradio>=4.0" "kaggle>=1.6" "jupyter" "ipykernel"

echo "==> Registrando kernel Jupyter 'cardioia'..."
conda run -n "$ENV_NAME" python -m ipykernel install --user \
    --name "$ENV_NAME" --display-name "Python (CardioIA M5)"

echo
echo "✅ Ambiente '$ENV_NAME' pronto."
echo "   Próximo passo: rode o smoke test ->"
echo "   conda run -n $ENV_NAME python scripts/smoke_test_metal.py"
