#!/bin/bash
# Install system dependencies
apt-get update && \
apt-get install -y \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    poppler-utils \
    libleptonica-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev

# Install Python packages
pip install --upgrade pip && \
pip install -r requirements.txt && \

# Start Gunicorn
gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app
