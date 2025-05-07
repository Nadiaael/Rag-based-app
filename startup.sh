#!/bin/bash

# Make sure the script doesn't fail silently
set -e
echo "Starting deployment process..."

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
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
echo "Setting up Python environment..."
export PIP_DEFAULT_TIMEOUT=300
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

# Create temp uploads directory
echo "Creating temp uploads directory..."
mkdir -p /home/site/wwwroot/temp_uploads
chmod 777 /home/site/wwwroot/temp_uploads

# Check if Gunicorn is installed
echo "Checking Gunicorn installation..."
if ! python -m pip list | grep -q gunicorn; then
    echo "Gunicorn not found, installing..."
    python -m pip install gunicorn
fi

# Debug: List files in directory
echo "Contents of current directory:"
ls -la

# Start Gunicorn
echo "Starting Gunicorn server..."
python -m gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app
