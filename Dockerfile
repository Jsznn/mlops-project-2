FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration
COPY config/ config/

# Copy source code
COPY src/ src/

# Copy models (if they exist locally, otherwise they might be mounted or pulled)
COPY models/ models/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]