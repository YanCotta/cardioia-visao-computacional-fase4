# CardioIA Fase 4 — imagem do pipeline (demo Gradio + treino) — CPU, portátil.
# Roda em qualquer máquina com Docker (sem GPU dentro do container).
FROM python:3.11-slim

# Dependências de sistema do OpenCV (libGL + glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODELO_PATH=/app/modelos/resnet50_finetuned.keras \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/

EXPOSE 7860
# Demo Gradio por padrão; o serviço de treino sobrescreve o command (ver docker-compose.yml)
CMD ["python", "scripts/app_gradio.py"]
