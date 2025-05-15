#!/bin/bash
set -e
echo "Starting deployment process..."

# Set up logging
LOG_FILE=/home/site/wwwroot/deployment.log
touch $LOG_FILE
echo "$(date): Starting deployment" >> $LOG_FILE

# Check for compressed application
if [ -f /home/site/wwwroot/output.tar.gz ]; then
    echo "Found compressed application, extracting..." >> $LOG_FILE
    
    # Create app directory if it doesn't exist
    mkdir -p /home/site/wwwroot/app
    
    # Extract the archive to the app directory
    tar -xzf /home/site/wwwroot/output.tar.gz -C /home/site/wwwroot/app
    
    # Change to the app directory
    cd /home/site/wwwroot/app
    echo "Extracted to: $(pwd)" >> $LOG_FILE
    echo "Files in app directory:" >> $LOG_FILE
    ls -la >> $LOG_FILE 2>&1
else
    echo "No compressed application found, using current directory" >> $LOG_FILE
    cd /home/site/wwwroot
fi

# Find main Python file
if [ -f app.py ]; then
    APP_MODULE="app:app"
    echo "Found app.py, using $APP_MODULE" >> $LOG_FILE
elif [ -f main.py ]; then
    APP_MODULE="main:app"
    echo "Found main.py, using $APP_MODULE" >> $LOG_FILE
elif [ -f application.py ]; then
    APP_MODULE="application:app"
    echo "Found application.py, using $APP_MODULE" >> $LOG_FILE
else
    echo "ERROR: Could not find a Python application entry point!" >> $LOG_FILE
    echo "Files in current directory:" >> $LOG_FILE
    ls -la >> $LOG_FILE
    exit 1
fi

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..." >> $LOG_FILE
    pip install --upgrade pip >> $LOG_FILE 2>&1
    pip install -r requirements.txt >> $LOG_FILE 2>&1
    pip install gunicorn >> $LOG_FILE 2>&1
else
    echo "No requirements.txt found, skipping dependency installation" >> $LOG_FILE
fi

# Create temp uploads directory
mkdir -p temp_uploads
chmod 777 temp_uploads

# Start Gunicorn
export PORT="${PORT:-8000}"
echo "Starting Gunicorn on port $PORT with module $APP_MODULE" >> $LOG_FILE
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --log-level debug $APP_MODULE
