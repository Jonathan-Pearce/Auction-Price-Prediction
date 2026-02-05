# =============================================================================
# Auction Price Prediction - Dockerfile
# =============================================================================
# Simplified container for web scraping and Hugging Face uploads
# Usage: docker build -t auction-scraper:latest .
# =============================================================================

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for development
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    vim \
    nano \
    htop \
    libxcb1 \
    libxcb-shm0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy project metadata
COPY pyproject.toml README.md ./

# Install data collection, ML, API, and dev dependencies
RUN pip install -e ".[data,ml,api,dev]"

# Copy source code
COPY src/ ./src/

# Create data directories
RUN mkdir -p data/raw data/interim data/processed data/external

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
