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
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_deployment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_deployment.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY start.sh ./start.sh

# Make start script executable
RUN chmod +x ./start.sh

# Expose port (Render uses PORT env variable)
EXPOSE 10000

# Use start script that downloads models at runtime
CMD ["./start.sh"]
