# 📱 CardioIA Mobile — Ir Além 2 (React Native + Flask)

Protótipo **mobile** que leva a classificação de **cardiomegalia** da CNN para um app
**React Native (Expo)**: o usuário seleciona uma radiografia de tórax, o app envia para o
**backend Flask** (que serve o modelo ResNet50 treinado), e recebe de volta a **classe**,
a **confiança** e o **mapa de calor Grad-CAM**.

> **Status:** backend **validado end-to-end** com o modelo real (cardiomegalia → 0.81 ·
> normal → 0.26 · Grad-CAM gerado). App e backend prontos — **falta apenas gravar o vídeo**
> de até 3 min exigido pelo Ir Além 2.

```text
mobile/
├── backend/            # API Flask que serve o modelo .keras
│   ├── app.py
│   └── requirements.txt
└── app/                # App React Native (Expo)
    ├── App.js
    ├── config.js       # ← ajuste a BACKEND_URL aqui
    ├── app.json
    └── package.json
```

## Pré-requisito

Ter o modelo treinado salvo (gerado pelo notebook ou pelo treino local):
`modelos/resnet50_finetuned.keras` na raiz do repositório.

## 1) Subir o backend Flask

### Opção Docker (recomendada — roda em qualquer máquina) 🐳

A partir da **raiz do repositório** (usa o `docker-compose.yml`):

```bash
docker compose up backend          # API em http://localhost:5050
```

### Opção local (Python)

```bash
cd mobile/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# aponte para o modelo treinado (caminho relativo à pasta backend)
export MODELO_PATH=../../modelos/resnet50_finetuned.keras
export PORT=5050         # macOS: a porta 5000 costuma estar ocupada pelo AirPlay Receiver
python app.py            # http://0.0.0.0:5050
```

Teste rápido (validado neste projeto com o modelo real):

```bash
curl http://localhost:5050/health
curl -F "image=@/caminho/para/uma_radiografia.png" http://localhost:5050/predict
```

## 2) Rodar o app React Native (Expo)

```bash
cd mobile/app
npm install
# edite config.js -> BACKEND_URL com o IP da máquina do backend
npx expo start          # abra no Expo Go (celular) ou em um emulador
```

> **BACKEND_URL** (em `app/config.js`):
> - Emulador Android: `http://10.0.2.2:5050`
> - iOS Simulator: `http://localhost:5050`
> - Celular físico (Expo Go): `http://SEU_IP_LOCAL:5050` (mesma rede Wi-Fi;
>   descubra com `ipconfig getifaddr en0`).

## Fluxo da interface

1. **Selecionar radiografia** → escolhe a imagem da galeria.
2. **Classificar** → envia ao backend.
3. Exibe **classe** (Cardiomegalia × Normal), **confiança (%)**, **P(cardiomegalia)** e o
   **Grad-CAM** sobreposto, com aviso de uso educacional.

## 🎥 Gravação do vídeo (Ir Além 2)

Roteiro sugerido (≤ 3 min): (1) subir o backend (`docker compose up backend`), (2) abrir o app
no Expo Go, (3) fazer upload de um raio-X **normal** e mostrar o resultado, (4) repetir com um
raio-X com **cardiomegalia**, destacando confiança e Grad-CAM, (5) citar que é protótipo
educacional. Publicar como **não listado** no YouTube e colar o link no `README.md` da raiz.

## ⚠️ Aviso

Protótipo **educacional** — não é dispositivo médico e não substitui avaliação clínica.
