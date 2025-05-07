cat > startup.sh << EOL
#!/bin/bash
# Make sure the script doesn't fail silently
set -e
echo "Starting deployment process..."

# Create a log file for debugging
LOG_FILE=/home/site/wwwroot/deployment.log
touch \$LOG_FILE
echo "$(date): Starting deployment" >> \$LOG_FILE

# Install system dependencies
echo "Installing system dependencies..." >> \$LOG_FILE
apt-get update >> \$LOG_FILE 2>&1 || echo "apt-get update failed, continuing..." >> \$LOG_FILE
apt-get install -y --no-install-recommends \
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
    zlib1g-dev >> \$LOG_FILE 2>&1 || echo "apt-get install failed, continuing..." >> \$LOG_FILE

# Configure Python environment
echo "Setting up Python environment..." >> \$LOG_FILE
export PIP_DEFAULT_TIMEOUT=300
which python >> \$LOG_FILE 2>&1
python --version >> \$LOG_FILE 2>&1
python -m pip install --upgrade pip >> \$LOG_FILE 2>&1
python -m pip install --no-cache-dir -r requirements.txt >> \$LOG_FILE 2>&1

# Create temp uploads directory
echo "Creating temp uploads directory..." >> \$LOG_FILE
mkdir -p /home/site/wwwroot/temp_uploads
chmod 777 /home/site/wwwroot/temp_uploads

# Check if Gunicorn is installed
echo "Checking Gunicorn installation..." >> \$LOG_FILE
if ! python -m pip list | grep -q gunicorn; then
    echo "Gunicorn not found, installing..." >> \$LOG_FILE
    python -m pip install gunicorn >> \$LOG_FILE 2>&1
fi

# Debug: List files in directory
echo "Contents of current directory:" >> \$LOG_FILE
ls -la >> \$LOG_FILE 2>&1

# Start Gunicorn
echo "Starting Gunicorn server..." >> \$LOG_FILE
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app
EOL

# 4. Make sure to include a web.config file for Azure
cat > web.config << EOL
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
    </handlers>
    <httpPlatform processPath="bash" arguments="startup.sh" requestTimeout="00:04:00" startupTimeLimit="120" startupRetryCount="3" stdoutLogEnabled="true" stdoutLogFile="log\stdout.log">
      <environmentVariables>
        <environmentVariable name="PORT" value="%HTTP_PLATFORM_PORT%" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
EOL

# 5. Ensure app.py can be found
echo "Checking app.py existence..." 
if [ ! -f app.py ]; then
    echo "WARNING: app.py not found in current directory. Make sure it exists!"
fi

echo "Setup completed. Now you should commit these changes and push to GitHub."
