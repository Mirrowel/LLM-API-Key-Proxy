# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set PATH for user-installed packages in builder stage
ENV PATH=/root/.local/bin:$PATH

# Copy requirements first for better caching
COPY requirements.txt .

# Copy the local rotator_library for editable install
COPY src/rotator_library ./src/rotator_library

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/

# Create directories for logs and oauth credentials
RUN mkdir -p logs oauth_creds

# Expose the default port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Default command - runs proxy with the correct PYTHONPATH
CMD ["python", "src/proxy_app/main.py", "--port", "8000"]
