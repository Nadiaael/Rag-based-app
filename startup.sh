#!/bin/bash

# Install system dependencies
apt-get update && \
apt-get install -y \
    build-essential \
    python3-dev \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libgl1 \
    poppler-utils \
    libleptonica-dev \
    libmupdf-dev \
    libjpeg-dev \
    zlib1g-dev

# Configure Python environment
export PIP_DEFAULT_TIMEOUT=300
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

# Start Gunicorn with explicit Python path
/opt/python/3.10.8/bin/python -m gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app
