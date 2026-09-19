# BioVision - environnement de développement partagé pour toute l'équipe
FROM python:3.11-slim

# Dépendances système requises par OpenCV et le traitement vidéo
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python d'abord (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Le code sera monté en volume via docker-compose plutôt que copié,
# pour permettre l'édition à chaud pendant le développement
COPY . .

CMD ["bash"]
