# Use slim python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install gcloud CLI and ensure gsutil is available
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin
RUN gcloud config set pass_credentials_to_gsutil true

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/deployment/app.py .
COPY src/model/utils ./model/utils

# Create model directory
RUN mkdir -p /app/models/current

# Copy model loading script
COPY src/deployment/load_model.py .

# Environment variables
ENV MODEL_DIR=/app/models
ENV MODEL_ARTIFACTS_BUCKET=${_ARTIFACT_REGISTRY}
ENV MODEL_REGISTRY_PATH=${_MODEL_PATH}
ENV PORT=8080
ENV MAX_RETRIES=3
ENV RETRY_DELAY=10

# Start application with model loading and retry on failure
CMD ["sh", "-c", "for i in $(seq 1 3); do python load_model.py && exec python app.py && break || sleep 10; done"]
