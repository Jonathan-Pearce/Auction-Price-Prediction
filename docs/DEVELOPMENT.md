# Development Workflow

> **Guide for developing the auction price prediction system**

## Overview

This document covers the development environment setup, workflow patterns, and best practices for contributing to the project.

---

## Environment Setup

### Option 1: GitHub Codespaces (Recommended)

1. **Open in Codespaces**
   - Go to GitHub repository
   - Click "Code" > "Codespaces" > "Create codespace on main"
   - Wait for container to build (~2-3 minutes)

2. **Verify Setup**
   ```bash
   python --version  # Should be 3.11+
   make check        # Run linting and tests
   ```

### Option 2: Local Docker

1. **Prerequisites**
   - Docker Desktop installed
   - Git installed

2. **Clone and Start**
   ```bash
   git clone https://github.com/Jonathan-Pearce/Auction-Price-Prediction-.git
   cd Auction-Price-Prediction-
   
   # Build and start container
   docker-compose up -d
   
   # Attach to container
   docker-compose exec app bash
   ```

3. **VS Code Integration**
   - Install "Dev Containers" extension
   - Open folder in container: Cmd/Ctrl+Shift+P > "Reopen in Container"

### Option 3: Local Python (No Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

---

## Project Structure

```
Auction-Price-Prediction-/
├── .devcontainer/          # Codespaces configuration
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   ├── instructions/       # Copilot agent instructions
│   └── copilot-instructions.md
├── api/                    # FastAPI + Gradio app
├── data/                   # Local data (gitignored)
│   ├── raw/               # Original scraped data
│   ├── interim/           # Intermediate processing
│   ├── processed/         # Final features
│   └── external/          # Third-party data
├── docs/                   # Documentation
├── models/                 # Model checkpoints (gitignored)
├── notebooks/              # Jupyter notebooks
├── references/             # Data dictionaries, schemas
├── reports/figures/        # Generated visualizations
├── scripts/                # Utility scripts
├── src/                    # Main source code
│   ├── data/              # Data collection
│   ├── modeling/          # ML models
│   ├── config.py          # Configuration
│   ├── dataset.py         # Data loading
│   └── features.py        # Feature engineering
├── tests/                  # Test files
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## Development Workflow

### Git Branching Strategy

```
main (protected)
├── feat/scraper-improvements
├── feat/image-model
├── fix/rate-limiting-bug
├── data/update-historical
└── docs/api-documentation
```

**Branch Naming:**
- `feat/` - New features
- `fix/` - Bug fixes
- `data/` - Data collection/processing
- `model/` - Model experiments
- `docs/` - Documentation updates

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feat/my-feature

# 2. Make changes, commit frequently
git add .
git commit -m "feat: add new feature"

# 3. Push and create PR
git push -u origin feat/my-feature
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(scraper): add retry logic for failed requests
fix(model): correct learning rate scheduler
docs(api): update endpoint documentation
```

---

## Using Makefile Commands

```bash
# Setup & Installation
make setup          # Install all dependencies
make install-dev    # Install dev dependencies only

# Development
make format         # Format code with Black
make lint           # Run Ruff linter
make test           # Run pytest
make check          # Format + lint + test

# Data
make scrape         # Run data scraper
make features       # Generate features

# Training
make train          # Train all models
make train-tabular  # Train only tabular model

# API
make api-dev        # Run API locally (hot reload)
make api-prod       # Run API in production mode

# Deployment
make sync-hf        # Push to Hugging Face
make deploy         # Full deployment pipeline

# Utilities
make clean          # Remove generated files
make notebook       # Start Jupyter Lab
```

---

## Working with Copilot Agents

### Agent Overview

Four specialized agents assist with different tasks:

| Agent | Activates On | Expertise |
|-------|-------------|-----------|
| Data Engineering | `src/data/**` | Scraping, validation |
| ML Training | `src/modeling/**` | Model architecture |
| Deployment | `api/**` | FastAPI, Gradio, HF |
| Project Management | `docs/**` | Documentation, planning |

### How It Works

**Automatic Activation:**
- Open a file in the relevant directory
- Agent context is automatically loaded
- Copilot suggestions reflect domain expertise

**Manual Invocation:**
- Type `/` in Copilot Chat to see available agents
- Reference specific agent: `@data-engineer how should I handle rate limiting?`

### Example Workflows

**Data Collection Task:**
```
1. Open src/data/scraper.py
2. Data Engineering agent activates
3. Ask: "How should I implement pagination for the auction list?"
4. Agent provides MaxSold-specific guidance
```

**Model Development Task:**
```
1. Open src/modeling/train.py
2. ML Training agent activates
3. Ask: "What's a good architecture for the image model?"
4. Agent suggests appropriate CNN configurations
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_scraper.py

# With coverage
pytest --cov=src --cov-report=html

# Only fast tests
pytest -m "not slow"
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_scraper.py       # Data collection tests
├── test_features.py      # Feature engineering tests
├── test_models.py        # Model tests
└── test_api.py           # API endpoint tests
```

### Writing Tests

```python
# tests/test_features.py
import pytest
from src.features import compute_tabular_features

class TestTabularFeatures:
    def test_word_count(self):
        result = compute_tabular_features(title="Test Item Title")
        assert result["title_word_count"] == 3
    
    def test_missing_description(self):
        result = compute_tabular_features(title="Test", description=None)
        assert result["description_length"] == 0
    
    @pytest.mark.slow
    def test_large_dataset(self, large_dataset):
        # Slow test marked for optional execution
        ...
```

---

## Notebooks Workflow

### Notebook Naming Convention

```
notebooks/
├── 01-data-exploration.ipynb
├── 02-feature-analysis.ipynb
├── 03-tabular-model-experiments.ipynb
├── 04-image-model-training.ipynb
└── 05-model-comparison.ipynb
```

### Best Practices

1. **Clear outputs before committing**
   ```bash
   make clean-notebooks
   ```

2. **Use functions from `src/`**
   ```python
   # In notebook
   from src.features import compute_features
   from src.modeling.train import train_model
   ```

3. **Document findings in markdown cells**

4. **Move production code to `src/`**

---

## Hugging Face Sync

### Initial Setup

```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/jonathan-pearce/auction-predictor

# Login to HF
huggingface-cli login
```

### Manual Sync

```bash
# Push to HF Space
git push hf main

# Or use make command
make sync-hf
```

### Automatic Sync (GitHub Actions)

On push to `main`, GitHub Actions automatically:
1. Runs tests
2. Builds Docker image
3. Pushes to HF Space

---

## Data Management

### Local Data (Not in Git)

```
data/
├── raw/                 # Original scraped JSON/Parquet
├── interim/             # Cleaned, intermediate data
├── processed/           # Final features for models
└── external/            # External datasets
```

### Syncing with HF Datasets

```bash
# Download dataset
make download-data

# Upload new data
make upload-data
```

```python
# In code
from datasets import load_dataset

# Load from HF
dataset = load_dataset("jonathan-pearce/maxsold-auctions")

# Save locally for development
dataset.save_to_disk("data/raw/maxsold")
```

---

## Debugging

### FastAPI Debugging

```bash
# Run with auto-reload
uvicorn api.main:app --reload --port 8000

# With debug logging
LOG_LEVEL=debug uvicorn api.main:app --reload
```

### Model Debugging

```python
# In notebook or script
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Set debug mode
torch.autograd.set_detect_anomaly(True)
```

### Profiling

```python
# Memory profiling
from memory_profiler import profile

@profile
def process_data():
    ...

# Time profiling
import cProfile
cProfile.run('train_model()', 'output.prof')
```

---

## Common Tasks

### Adding a New Feature

1. Add feature computation in `src/features.py`
2. Update DuckDB schema in `references/schema.sql`
3. Add tests in `tests/test_features.py`
4. Update documentation if needed

### Adding a New Model

1. Create model class in `src/modeling/`
2. Add training script
3. Update fusion model to include new predictions
4. Add model to HF Hub upload pipeline

### Updating API

1. Add/modify endpoint in `api/main.py`
2. Update Gradio interface if needed
3. Add API tests
4. Update API documentation

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Docker build fails | Check Dockerfile syntax, rebuild with `--no-cache` |
| Import errors | Run `pip install -e .` to install package |
| DuckDB lock error | Close other DuckDB connections |
| HF push rejected | Check HF_TOKEN, verify repo permissions |

### Getting Help

1. Check existing documentation
2. Search GitHub Issues
3. Ask Copilot agent for guidance
4. Create new Issue with details

---

*Last updated: January 2026*
