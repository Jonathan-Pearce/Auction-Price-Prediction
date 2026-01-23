## Plan: Auction Price Prediction Project Specifications

Build a production-ready full-stack ML system following cookiecutter-data-science structure to predict MaxSold auction winning prices. The project will include MaxSold API scraping (~10k auctions, ~1M items with bidding history), multi-model training (tabular, image, text, sequential) with PyTorch/scikit-learn, meta-model fusion, and FastAPI web interface deployed via Hugging Face. Development uses GitHub/Codespaces with Docker, orchestrated by 4 specialized GitHub Copilot agents.

### Steps

1. **Create cookiecutter-inspired folder structure** with root-level [Makefile](Makefile), [pyproject.toml](pyproject.toml), [.env.example](.env.example); `data/` subdirs (raw/, interim/, processed/, external/); `src/` package (config.py, dataset.py, features.py, modeling/); `models/` for checkpoints; `notebooks/` for exploration; `docs/` for documentation; `references/` for data dictionaries; `reports/figures/` for visualizations; `api/` for FastAPI backend; `.devcontainer/` for Codespaces; `docker/` for containers; `.github/workflows/` for CI/CD and HF deployment.

2. **Design multi-agent GitHub Copilot system** with [.github/copilot-instructions.md](.github/copilot-instructions.md) containing base project context (MaxSold APIs, soft-close mechanisms, no-bid handling, DuckDB schema, HF deployment targets), plus [.github/instructions/](.github/instructions/) with 4 specialized agents: data-engineering.instructions.md (`applyTo: "src/data/**"`), ml-training.instructions.md (`applyTo: "src/models/**"`), deployment.instructions.md (`applyTo: "api/**"`), and project-management.instructions.md (`applyTo: "docs/**"`).

3. **Write comprehensive technical documentation** including [docs/DESIGN.md](docs/DESIGN.md) (system architecture, data flow from 3 MaxSold APIs → preprocessing → 4 models + fusion → FastAPI → Gradio UI, handling soft-close and zero-bid items), [docs/DATA.md](docs/DATA.md) (API endpoints schemas, scraping strategy with rate limiting, Parquet/DuckDB storage, HF Datasets upload), [docs/MODELS.md](docs/MODELS.md) (4 specialized models architecture: tabular XGBoost/Random Forest, CNN for images, transformer for text, LSTM for bid sequences, plus fusion meta-model), and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) (HF Spaces setup, GitHub Actions sync, Gradio interface).

4. **Generate development environment configurations** with [Dockerfile](Dockerfile) (Python 3.11, PyTorch, FastAPI, DuckDB, GPU support), [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) for Codespaces with extensions (Python, Jupyter, Docker), [docker-compose.yml](docker-compose.yml) for local development, [pyproject.toml](pyproject.toml) with dependencies grouped (data collection, ML training, API serving, deployment), and [.gitignore](.gitignore) excluding `data/`, `models/`, `.env`, and DuckDB files.

5. **Create MaxSold-specific data collection specifications** in [src/data/](src/data/) with `maxsold_client.py` (API wrapper for 3 endpoints: auction items list, item+bidding history, enriched item info), `scraper.py` (orchestrates ~10k auction collection with retry logic and rate limiting), `schemas.py` (Pydantic models for auction/item/bid data), and [references/maxsold-api-reference.md](references/maxsold-api-reference.md) documenting endpoint structures, soft-close logic, and response examples.

6. **Build project management and automation files** including updated [README.md](README.md) with project overview, setup instructions for Codespaces/Docker, MaxSold context, and agent usage guide; [Makefile](Makefile) with targets (`make setup`, `make scrape`, `make train`, `make deploy`, `make sync-hf`); [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development workflow (agent orchestration, GitHub↔HF sync, testing strategy); [.github/workflows/deploy-to-hf.yml](.github/workflows/deploy-to-hf.yml) for automated HF Spaces deployment; and DuckDB schema DDL in [references/schema.sql](references/schema.sql).

### Further Considerations

1. **Gradio vs custom frontend?** Recommend starting with Gradio on HF Spaces (single Python app, faster MVP) - takes URL input, displays prediction with confidence intervals. Can migrate to GitHub Pages + separate FastAPI later if custom UI needed.

2. **Model experiment tracking?** HF Hub handles versioning, but consider adding Weights & Biases or MLflow integration for experiment tracking during model development. Set up in ml-training agent instructions?

3. **Rate limiting for MaxSold API?** Need to determine acceptable request rate to avoid getting blocked. Should we implement exponential backoff, respect robots.txt, or add configurable delays in scraper? Document in data engineering agent instructions.

4. **DuckDB data organization?** Use single database file or separate files per model (tabular features, embeddings, bid sequences)? Consider partitioning by auction date for efficient queries. Define schema strategy in [docs/DATA.md](docs/DATA.md).
