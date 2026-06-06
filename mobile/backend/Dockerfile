# Backend Flask — CardioIA Fase 4 (Ir Além 2)
# Imagem CPU portátil (roda em qualquer máquina com Docker; sem GPU dentro do container).
FROM python:3.11-slim

# Dependências de sistema do OpenCV (libGL + glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODELO_PATH=/models/resnet50_finetuned.keras \
    PORT=5050

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5050
# O modelo é montado em /models (volume) — ver docker-compose.yml
CMD ["python", "app.py"]
