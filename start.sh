#!/bin/bash
# Startup script for cloud deployment with Google Drive model download

echo "==> Starting Credit Default Prediction API..."

# Check if models directory exists and has files
if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models 2>/dev/null)" ]; then
    echo "==> Models not found. Downloading from Google Drive..."
    
    # Install gdown if not already installed
    pip install --quiet gdown
    
    # Download models folder from Google Drive
    if [ ! -z "$GDRIVE_FOLDER_ID" ]; then
        echo "==> Downloading models folder (ID: $GDRIVE_FOLDER_ID)..."
        gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O /app/models --quiet
        
        if [ $? -eq 0 ]; then
            echo "==> Models downloaded successfully!"
            ls -lh /app/models/
        else
            echo "==> ERROR: Failed to download models"
        fi
    else
        echo "==> WARNING: GDRIVE_FOLDER_ID not set"
    fi
else
    echo "==> Models already present"
fi

echo "==> Starting uvicorn..."
uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}