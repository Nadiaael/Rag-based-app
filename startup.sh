#!/bin/bash
set -e
echo "Starting deployment process..."

# Set up logging
LOG_FILE=/home/site/wwwroot/deployment.log
touch $LOG_FILE
echo "$(date): Starting deployment" >> $LOG_FILE

# Configure Python environment
echo "Setting up Python environment..." >> $LOG_FILE
which python >> $LOG_FILE 2>&1
python --version >> $LOG_FILE 2>&1
python -m pip install --upgrade pip >> $LOG_FILE 2>&1
python -m pip install --no-cache-dir -r requirements.txt >> $LOG_FILE 2>&1

# Create temp uploads directory
echo "Creating temp uploads directory..." >> $LOG_FILE
mkdir -p /home/site/wwwroot/temp_uploads
chmod 777 /home/site/wwwroot/temp_uploads

# Start Gunicorn
echo "Starting Gunicorn server..." >> $LOG_FILE
cd /home/site/wwwroot
export PORT="${PORT:-8000}"
gunicorn --bind=0.0.0.0:$PORT --timeout 600 app:app
