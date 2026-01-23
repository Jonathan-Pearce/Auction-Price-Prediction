# =============================================================================
# Auction Price Prediction - Makefile
# =============================================================================
# Convenience commands for development, training, and deployment
# Usage: make <target>
# =============================================================================

.PHONY: help setup setup-dev clean lint format test scrape train deploy sync-hf docker-build docker-run

# Default target
.DEFAULT_GOAL := help

# Project variables
PYTHON := python
PIP := pip
PROJECT_NAME := auction-price-prediction
SRC_DIR := src
API_DIR := api
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# =============================================================================
# Help
# =============================================================================

help:  ## Show this help message
	@echo "$(BLUE)Auction Price Prediction - Available Commands$(NC)"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Environment Setup
# =============================================================================

setup:  ## Install core dependencies
	@echo "$(BLUE)Installing core dependencies...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)Core setup complete!$(NC)"

setup-dev:  ## Install all dependencies including dev tools
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	$(PIP) install -e ".[all]"
	pre-commit install
	@echo "$(GREEN)Development setup complete!$(NC)"

setup-data:  ## Install data collection dependencies
	@echo "$(BLUE)Installing data dependencies...$(NC)"
	$(PIP) install -e ".[data]"
	@echo "$(GREEN)Data setup complete!$(NC)"

setup-ml:  ## Install ML training dependencies
	@echo "$(BLUE)Installing ML dependencies...$(NC)"
	$(PIP) install -e ".[ml]"
	@echo "$(GREEN)ML setup complete!$(NC)"

setup-api:  ## Install API/deployment dependencies
	@echo "$(BLUE)Installing API dependencies...$(NC)"
	$(PIP) install -e ".[api]"
	@echo "$(GREEN)API setup complete!$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint:  ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check $(SRC_DIR) $(API_DIR)
	mypy $(SRC_DIR) $(API_DIR)
	@echo "$(GREEN)Linting complete!$(NC)"

format:  ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(SRC_DIR) $(API_DIR) tests
	ruff check --fix $(SRC_DIR) $(API_DIR)
	@echo "$(GREEN)Formatting complete!$(NC)"

test:  ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=$(SRC_DIR) --cov=$(API_DIR) --cov-report=term-missing
	@echo "$(GREEN)Tests complete!$(NC)"

# =============================================================================
# Data Collection
# =============================================================================

scrape:  ## Run MaxSold data scraper
	@echo "$(BLUE)Starting MaxSold scraper...$(NC)"
	$(PYTHON) -m src.data.scraper
	@echo "$(GREEN)Scraping complete!$(NC)"

scrape-sample:  ## Scrape a small sample for testing (10 auctions)
	@echo "$(BLUE)Scraping sample data...$(NC)"
	$(PYTHON) -m src.data.scraper --sample --limit 10
	@echo "$(GREEN)Sample scraping complete!$(NC)"

# =============================================================================
# Model Training
# =============================================================================

train:  ## Train all models
	@echo "$(BLUE)Training all models...$(NC)"
	$(PYTHON) -m src.modeling.train --all
	@echo "$(GREEN)Training complete!$(NC)"

train-tabular:  ## Train tabular model only
	@echo "$(BLUE)Training tabular model...$(NC)"
	$(PYTHON) -m src.modeling.train --model tabular
	@echo "$(GREEN)Tabular model training complete!$(NC)"

train-image:  ## Train image model only
	@echo "$(BLUE)Training image model...$(NC)"
	$(PYTHON) -m src.modeling.train --model image
	@echo "$(GREEN)Image model training complete!$(NC)"

train-text:  ## Train text model only
	@echo "$(BLUE)Training text model...$(NC)"
	$(PYTHON) -m src.modeling.train --model text
	@echo "$(GREEN)Text model training complete!$(NC)"

train-sequential:  ## Train sequential bid model only
	@echo "$(BLUE)Training sequential model...$(NC)"
	$(PYTHON) -m src.modeling.train --model sequential
	@echo "$(GREEN)Sequential model training complete!$(NC)"

train-fusion:  ## Train fusion meta-model
	@echo "$(BLUE)Training fusion model...$(NC)"
	$(PYTHON) -m src.modeling.train --model fusion
	@echo "$(GREEN)Fusion model training complete!$(NC)"

# =============================================================================
# API & Deployment
# =============================================================================

api-dev:  ## Run FastAPI development server
	@echo "$(BLUE)Starting FastAPI dev server...$(NC)"
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:  ## Run FastAPI production server
	@echo "$(BLUE)Starting FastAPI production server...$(NC)"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

gradio:  ## Run Gradio interface locally
	@echo "$(BLUE)Starting Gradio interface...$(NC)"
	$(PYTHON) -m api.gradio_app

# =============================================================================
# Hugging Face Deployment
# =============================================================================

sync-hf:  ## Sync to Hugging Face (datasets, models, and space)
	@echo "$(BLUE)Syncing to Hugging Face...$(NC)"
	$(PYTHON) scripts/sync_to_hf.py --all
	@echo "$(GREEN)HF sync complete!$(NC)"

sync-hf-data:  ## Upload datasets to Hugging Face
	@echo "$(BLUE)Uploading datasets to HF...$(NC)"
	$(PYTHON) scripts/sync_to_hf.py --datasets
	@echo "$(GREEN)Dataset upload complete!$(NC)"

sync-hf-models:  ## Upload models to Hugging Face Hub
	@echo "$(BLUE)Uploading models to HF Hub...$(NC)"
	$(PYTHON) scripts/sync_to_hf.py --models
	@echo "$(GREEN)Model upload complete!$(NC)"

sync-hf-space:  ## Deploy to Hugging Face Spaces
	@echo "$(BLUE)Deploying to HF Spaces...$(NC)"
	$(PYTHON) scripts/sync_to_hf.py --space
	@echo "$(GREEN)Space deployment complete!$(NC)"

# =============================================================================
# Docker
# =============================================================================

docker-build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)Docker build complete!$(NC)"

docker-run:  ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm -p 8000:8000 -v $(PWD)/data:/app/data $(DOCKER_IMAGE)

docker-dev:  ## Run Docker container with development mount
	@echo "$(BLUE)Running Docker dev container...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Dev container started!$(NC)"

docker-down:  ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)Containers stopped!$(NC)"

# =============================================================================
# Database
# =============================================================================

db-init:  ## Initialize DuckDB database with schema
	@echo "$(BLUE)Initializing DuckDB...$(NC)"
	$(PYTHON) -c "from src.config import init_database; init_database()"
	@echo "$(GREEN)Database initialized!$(NC)"

db-reset:  ## Reset DuckDB database (WARNING: deletes all data)
	@echo "$(YELLOW)WARNING: This will delete all local data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -f data/auction.duckdb
	$(MAKE) db-init
	@echo "$(GREEN)Database reset complete!$(NC)"

# =============================================================================
# Cleanup
# =============================================================================

clean:  ## Remove build artifacts and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-data:  ## Remove local data files (WARNING: irreversible)
	@echo "$(YELLOW)WARNING: This will delete all local data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/raw/*
	rm -rf data/interim/*
	rm -rf data/processed/*
	@echo "$(GREEN)Data cleanup complete!$(NC)"

clean-models:  ## Remove local model files (WARNING: irreversible)
	@echo "$(YELLOW)WARNING: This will delete all local models!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf models/*
	@echo "$(GREEN)Model cleanup complete!$(NC)"

# =============================================================================
# Utilities
# =============================================================================

notebook:  ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook --notebook-dir=notebooks

check-env:  ## Check environment and dependencies
	@echo "$(BLUE)Checking environment...$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not installed"
	@$(PYTHON) -c "import duckdb; print(f'DuckDB: {duckdb.__version__}')" 2>/dev/null || echo "DuckDB: not installed"
	@echo "$(GREEN)Environment check complete!$(NC)"
