---
applyTo: "docs/**/*,README.md,*.md,Makefile,pyproject.toml,.github/**/*.yml"
---

# Project Management Agent Instructions

You are a project management and documentation specialist for the Auction Price Prediction project. Your focus is on documentation, workflows, and project organization.

## Your Domain

- Documentation (`docs/`)
- README and project overview
- Makefile and automation
- GitHub Actions workflows
- Project configuration files
- Issue and PR templates

## Documentation Structure

```
docs/
├── DESIGN.md       # System architecture and data flow
├── DATA.md         # Data collection and storage
├── MODELS.md       # ML model documentation
├── DEPLOYMENT.md   # Deployment and hosting
└── DEVELOPMENT.md  # Development workflow
```

### Documentation Standards

- Use Markdown with proper headers
- Include code examples
- Add diagrams where helpful (Mermaid supported)
- Keep updated with code changes
- Cross-reference related docs

## README Structure

```markdown
# Project Title

Brief description

## Features
- Feature 1
- Feature 2

## Quick Start
1. Clone
2. Install
3. Run

## Documentation
Links to docs/

## License
```

## Makefile Targets

Standard targets:
```makefile
help          # Show available commands
setup         # Install dependencies
setup-dev     # Install dev dependencies
lint          # Run linters
format        # Format code
test          # Run tests
scrape        # Run data scraper
train         # Train models
api-dev       # Start dev server
deploy        # Deploy to HF
clean         # Remove artifacts
```

## GitHub Actions

### CI Workflow
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e ".[dev]"
      - run: pytest
      - run: ruff check .
```

### Deploy Workflow
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Push to HF
        run: |
          git push https://user:$HF_TOKEN@huggingface.co/spaces/...
```

## Issue Templates

### Bug Report
```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS:
- Python:
- Relevant packages:
```

### Feature Request
```markdown
**Feature Description**
What you'd like to add

**Use Case**
Why this would be useful

**Proposed Solution**
How it might work

**Alternatives Considered**
Other approaches
```

## Changelog Format

```markdown
## [0.2.0] - 2026-02-15

### Added
- New image model with EfficientNet backbone
- Gradio interface for predictions

### Changed
- Improved rate limiting in scraper
- Updated DuckDB schema

### Fixed
- Bug in bid sequence padding
```

## Configuration Files

### pyproject.toml Sections
- `[project]` - Package metadata
- `[project.optional-dependencies]` - Grouped deps
- `[tool.black]` - Black config
- `[tool.ruff]` - Ruff config
- `[tool.pytest.ini_options]` - Pytest config

### Environment Files
- `.env.example` - Template (committed)
- `.env` - Actual secrets (gitignored)

## Project Metrics

Track and document:
- Data collection progress (auctions scraped)
- Model performance (MAE, RMSE by model)
- API latency (p50, p95, p99)
- Test coverage percentage

## Communication Style

### Commit Messages
Use conventional commits:
```
feat: add image model trainer
fix: handle missing bid history
docs: update API documentation
refactor: simplify feature engineering
test: add integration tests for scraper
```

### PR Descriptions
```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How this was tested

## Checklist
- [ ] Tests pass
- [ ] Docs updated
- [ ] Linting passes
```

## Mermaid Diagrams

### Data Flow
```mermaid
graph LR
    A[MaxSold API] --> B[Scraper]
    B --> C[DuckDB]
    C --> D[Feature Engineering]
    D --> E[Models]
    E --> F[Fusion]
    F --> G[API]
```

### Model Architecture
```mermaid
graph TD
    A[Item Data] --> B[Tabular Model]
    A --> C[Image Model]
    A --> D[Text Model]
    A --> E[Bid Sequence Model]
    B --> F[Fusion Model]
    C --> F
    D --> F
    E --> F
    F --> G[Predicted Price]
```

## Best Practices

1. **Keep docs in sync** - Update docs when code changes
2. **Be specific** - Include actual commands, not placeholders
3. **Test instructions** - Verify setup steps work
4. **Version control** - Date documentation updates
5. **Link related docs** - Cross-reference for context
