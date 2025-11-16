FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for ML packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_deployment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_deployment.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/
COPY config/ ./config/

# Expose port (Render uses PORT env variable)
EXPOSE 10000

# Run the application
CMD uvicorn app.api:app --host 0.0.0.0 --port $PORT
