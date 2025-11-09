# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir pip --upgrade

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies in a single layer
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create models directory
RUN mkdir -p models

# Copy only necessary files
COPY api_service.py .
COPY models/best_*.joblib ./models/

# Ensure correct permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose a default port (Render will provide $PORT at runtime)
EXPOSE 8000

# Command to run the FastAPI application. Use the PORT env var when available (Render provides $PORT).
# Use a shell form so environment variable expansion works.
CMD ["sh", "-c", "uvicorn api_service:app --host 0.0.0.0 --port ${PORT:-8000}"]