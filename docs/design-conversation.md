# Design Conversation Log - Auction Price Prediction Project

**Date**: January 23, 2026  
**Participants**: Jonathan Pearce (Project Owner), GitHub Copilot (Design Assistant)

## Project Overview

Full-stack data science project to predict winning prices for ongoing auctions on MaxSold.com. The system will:
- Scrape ~10,000 completed auctions (~1,000,000 items) with full bidding history
- Train multiple ML models (tabular, image, text, sequential bid data)
- Deploy a meta-model (fusion model) combining predictions
- Provide a web interface for users to input auction URLs and receive price predictions

## Key Design Decisions

### 1. Technology Stack (ML & Web Framework)

**Decision**: 
- **Deep Learning**: PyTorch
- **Statistical Models**: scikit-learn
- **Backend**: FastAPI
- **Frontend**: Gradio on Hugging Face Spaces (MVP)
- **Database**: DuckDB (analytical queries)

**Discussion**:
- Initially considered GitHub Pages + separate FastAPI backend
- **Chosen approach**: All-in on Hugging Face ecosystem for lean, integrated solution
  - HF Datasets for data storage (unlimited free for public datasets)
  - HF Spaces for deployment (FastAPI + Gradio in one space)
  - HF Hub for model versioning
- DuckDB selected over SQLite for better analytical query performance with ~1M items
- Can migrate to custom frontend (GitHub Pages) later if needed

**Rationale**: Single ecosystem, ML-native tools, free tier sufficient for personal project, great for portfolio visibility

---

### 2. Hosting & Deployment Architecture

**Decision**:
```
Development: VSCode → GitHub (main repo)
Deployment: GitHub → Hugging Face (Datasets + Models + Spaces)
```

**Workflow**:
- Primary development in GitHub Codespaces and local Docker containers
- GitHub repository as source of truth
- Deploy to Hugging Face via:
  - Dual git remotes, OR
  - GitHub Actions auto-sync, OR
  - HF Space linked directly to GitHub repo

**Data Hosting**: Hugging Face Datasets (instead of Kaggle) for better ML integration

---

### 3. Data Source & API Structure

**Decision**: MaxSold.com auction data via public API

**Three API Endpoints**:
1. **Auction Items List**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={id}&limit=2500`
2. **Item + Bidding History**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={id}&itemid={item_id}`
3. **Enriched Item Info**: `https://api.maxsold.com/listings/am/{item_id}/enriched`

**Special Auction Mechanisms**:
- **Zero-bid items**: Some items receive no bids (winning price = $0)
- **Soft closing**: Bids in last 2 minutes extend auction by 2 minutes
- Need to handle these in model training and feature engineering

---

### 4. Project Structure & Organization

**Decision**: Follow **cookiecutter-data-science** structure with production focus

**Rationale**: 
- Production-ready pipeline (not just research/notebooks)
- Industry-standard organization
- Clear separation: raw data → processed data → models → deployment
- Reference: https://cookiecutter-data-science.drivendata.org/

**Key Principles**:
- Data is immutable (raw data never edited in place)
- Data not in source control (hosted on HF Datasets)
- Notebooks for exploration, source files for production
- Clear DAG workflow

---

### 5. Multi-Agent GitHub Copilot System

**Decision**: Implement 4 specialized Copilot agents with automatic context switching

**Four Agent Personas**:
1. **Data Engineering Agent** - Scraping, API clients, data validation
2. **ML Training Agent** - Model architecture, training pipelines, evaluation
3. **Deployment Agent** - FastAPI, Gradio UI, HF Spaces configuration
4. **Project Management Agent** - Documentation, planning, coordination

**Implementation**:
```
.github/
├── copilot-instructions.md              # Base context (always active)
└── instructions/
    ├── data-engineering.instructions.md  # applyTo: "src/data/**"
    ├── ml-training.instructions.md       # applyTo: "src/models/**"
    ├── deployment.instructions.md        # applyTo: "api/**"
    └── project-management.instructions.md # applyTo: "docs/**"
```

**Orchestration**: 
- **Automatic**: Copilot detects file context and activates appropriate agent
- **Manual**: Use @-mentions in chat for explicit agent invocation

**Benefits**: Consistent domain-specific guidance, reduced context switching, version-controlled expertise

---

### 6. Development Environment

**Decision**: Docker containers for both cloud and local development

**Setup**:
- **Cloud**: GitHub Codespaces with devcontainer configuration
- **Local**: Docker + docker-compose for consistent environment
- Both environments use same Dockerfile

**Why Docker**: Environment reproducibility, dependency isolation, GPU support configuration

---

## Model Architecture Overview

### Four Specialized Models:
1. **Tabular Model**: XGBoost/Random Forest for structured auction/item features
2. **Image Model**: CNN for product images
3. **Text Model**: Transformer for item descriptions
4. **Sequential Model**: LSTM for bidding history time series

### Meta-Model (Fusion):
- Combines predictions from 4 specialized models
- Ensemble approach for final winning price prediction

---

## Data Storage Strategy

**Training Data**:
- Raw scraped data → Hugging Face Datasets (Parquet format)
- DuckDB for feature store and serving layer
- Processed features cached for model training

**Models**:
- Trained weights → Hugging Face Hub
- Model configs and metadata in Git

**Application Data**:
- DuckDB embedded in HF Space for inference-time features
- Direct query to HF Datasets when needed

---

## Open Questions & Future Decisions

### Captured during design:

1. **Frontend Evolution**: Start with Gradio (MVP), potentially migrate to custom GitHub Pages frontend later

2. **Experiment Tracking**: HF Hub for model versioning, consider adding Weights & Biases or MLflow for detailed experiment tracking

3. **Rate Limiting**: Need to determine acceptable MaxSold API request rates and implement backoff strategies

4. **DuckDB Organization**: Single database vs. separate files per model type? Partition strategy by auction date?

---

## Project Constraints

- **Team Size**: Solo project (one human developer + AI agents)
- **Budget**: $0 hosting costs (leveraging free tiers)
- **Data Size**: ~10k auctions, ~1M items, full bid histories
- **Hosting**: All data external to Git (size constraints)

---

## Development Philosophy

- **Production-first**: Focus on deployable pipeline over exploratory notebooks
- **Version control everything**: Code, configs, docs (but not data/models)
- **Lean tech stack**: Minimize services and dependencies
- **Portfolio-ready**: Public repos showcasing full ML engineering skills
- **Incremental development**: Start with MVP (Gradio), iterate to custom UI if needed

---

## Next Steps

1. Implement folder structure following cookiecutter pattern
2. Set up multi-agent Copilot instructions
3. Create comprehensive technical documentation
4. Configure development environments (Docker, Codespaces)
5. Document MaxSold API specifications
6. Build project automation (Makefile, GitHub Actions)

---

## References

- Cookiecutter Data Science: https://cookiecutter-data-science.drivendata.org/
- MaxSold API Endpoints: (documented in `references/maxsold-api-reference.md`)
- Hugging Face Documentation: Hub, Datasets, Spaces
- GitHub Copilot Instructions: VS Code documentation

---

*This document captures the design conversation and key decisions made during initial project planning. It serves as a reference for understanding the "why" behind architectural choices.*
