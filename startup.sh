#!/bin/bash

# Install Tesseract OCR (required for pytesseract)
if ! command -v tesseract &> /dev/null; then
    echo "Installing Tesseract OCR..."
    apt-get update
    apt-get install -y tesseract-ocr
fi

# Run the application
echo "Starting application..."
gunicorn --bind=0.0.0.0:8000 --timeout 120 app:app