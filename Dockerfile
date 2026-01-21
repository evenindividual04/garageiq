# Dockerfile for GarageIQ
# Multi-stage build for smaller image size

FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ src/
COPY ui/ ui/

# Environment variables
ENV AMI_ENV=production
ENV AMI_USE_OLLAMA=true
ENV AMI_USE_NLLB=false
ENV AMI_HOST=0.0.0.0
ENV AMI_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.automotive_intent.app:app", "--host", "0.0.0.0", "--port", "8000"]
