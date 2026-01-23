# =============================================================================
# Auction Price Prediction - Dockerfile
# =============================================================================
# Multi-stage build for development and production
# Usage:
#   Development: docker build --target dev -t auction-predictor:dev .
#   Production:  docker build --target prod -t auction-predictor:latest .
# =============================================================================

# -----------------------------------------------------------------------------
# Base Stage - Common dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Development Stage
# -----------------------------------------------------------------------------
FROM base as dev

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY README.md ./

# Install all dependencies including dev tools
RUN pip install -e ".[all]"

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY tests/ ./tests/

# Create data directories
RUN mkdir -p data/raw data/interim data/processed data/external models notebooks

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# Default command for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------------------------------------------------------
# Production Stage
# -----------------------------------------------------------------------------
FROM base as prod

# Copy only necessary files for production
COPY pyproject.toml ./
COPY README.md ./

# Install production dependencies only
RUN pip install -e ".[data,ml,api]"

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Create data directories
RUN mkdir -p data models

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with multiple workers
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# -----------------------------------------------------------------------------
# GPU Stage (for training with CUDA)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy and install
COPY pyproject.toml README.md ./
RUN pip install -e ".[all]"

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY src/ ./src/
COPY api/ ./api/

RUN mkdir -p data/raw data/interim data/processed data/external models
RUN chown -R appuser:appuser /app

USER appuser

# Default to training mode
CMD ["python", "-m", "src.modeling.train", "--all"]
