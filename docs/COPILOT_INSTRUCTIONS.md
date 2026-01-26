# GitHub Copilot Instructions

This repository uses GitHub Copilot with domain-specific instructions to provide intelligent, context-aware coding assistance.

## Overview

The Copilot instructions are organized into:

1. **Main instruction file**: `.github/copilot-instructions.md` - General project guidance
2. **Domain-specific instructions**: `.github/instructions/` - Specialized guidance for different areas
3. **Configuration file**: `.github/copilot-instructions.yml` - Maps instructions to file patterns

## Instruction Files

### Main Instructions (`.github/copilot-instructions.md`)

**Purpose**: Provides general project context, tech stack, and coding standards for all files.

**Contents**:
- Project overview and architecture
- Tech stack (Python, PyTorch, FastAPI, DuckDB, etc.)
- Coding standards (PEP 8, type hints, Black formatting)
- Development workflow
- Common tasks and environment variables

### Data Engineering Instructions

**File**: `.github/instructions/data-engineering.instructions.md`  
**Applies to**: `src/data/**/*.py`

**Purpose**: Guides Copilot when working with data collection and storage code.

**Key Topics**:
- MaxSold API endpoints and usage
- Rate limiting and retry strategies
- Data validation with Pydantic
- DuckDB operations
- Handling zero-bid items and soft-close auctions
- Progress tracking for large scraping jobs

### ML Training Instructions

**File**: `.github/instructions/ml-training.instructions.md`  
**Applies to**: `src/modeling/**/*.py`, `src/features.py`, `src/dataset.py`

**Purpose**: Assists with model development, training, and evaluation.

**Key Topics**:
- Four base models (tabular, image, text, sequential)
- Fusion model architecture
- Feature engineering patterns
- PyTorch training loops
- Device management (CPU/GPU/MPS)
- Model saving and Hugging Face Hub integration
- Evaluation metrics (MAE, RMSE, MAPE)

### Deployment Instructions

**File**: `.github/instructions/deployment.instructions.md`  
**Applies to**: `api/**/*.py`, `**/gradio*.py`, `**/deploy*`, `**/*hf*`

**Purpose**: Helps with API development and deployment to Hugging Face.

**Key Topics**:
- FastAPI endpoint patterns
- Gradio interface design
- Hugging Face Spaces deployment
- Model loading strategies
- Error handling and CORS
- Performance optimization (caching, async)
- Local development setup

### Project Management Instructions

**File**: `.github/instructions/project-management.instructions.md`  
**Applies to**: `docs/**/*`, `README.md`, `*.md`, `Makefile`, `pyproject.toml`, `.github/**/*.yml`

**Purpose**: Guides documentation, workflows, and project configuration.

**Key Topics**:
- Documentation structure and standards
- Makefile targets
- GitHub Actions workflows
- Issue and PR templates
- Changelog format
- Mermaid diagrams
- Conventional commit messages

## How It Works

### Automatic Activation

When you open a file matching a pattern (e.g., `src/data/scraper.py`), Copilot automatically loads the corresponding instructions (Data Engineering) in addition to the main instructions.

### Pattern Matching

The `.github/copilot-instructions.yml` file defines glob patterns for each instruction file:

```yaml
instructions:
  - path: instructions/data-engineering.instructions.md
    patterns:
      - "src/data/**/*.py"
```

### YAML Frontmatter

Each instruction file has YAML frontmatter specifying its scope:

```yaml
---
applyTo: "src/data/**/*.py"
---
```

## Using the Instructions

### As a Developer

The instructions work automatically when using GitHub Copilot:

1. **Code Completion**: Copilot suggests code following project patterns
2. **Code Generation**: Generate boilerplate with `# Generate...` comments
3. **Code Explanation**: Get project-specific explanations
4. **Code Chat**: Ask questions with project context

### Examples

**Data Scraping**:
```python
# When working in src/data/scraper.py, Copilot knows:
# - MaxSold API endpoints and rate limits
# - Pydantic validation patterns
# - Progress tracking implementation
```

**Model Training**:
```python
# When working in src/modeling/train.py, Copilot knows:
# - PyTorch training loop patterns
# - Device management (cuda/mps/cpu)
# - Model saving conventions
```

**API Development**:
```python
# When working in api/main.py, Copilot knows:
# - FastAPI endpoint patterns
# - Pydantic request/response models
# - Error handling standards
```

## Maintaining Instructions

### When to Update

Update instructions when:
- Architecture changes (new models, APIs, etc.)
- Coding standards evolve
- Common patterns emerge
- New tools or frameworks are adopted

### How to Update

1. Edit the relevant `.md` file in `.github/instructions/`
2. Keep instructions concise and practical
3. Include code examples for new patterns
4. Update the configuration if patterns change

### Best Practices

**Do**:
- ✅ Keep instructions focused on "how" and "why"
- ✅ Include concrete code examples
- ✅ Document common pitfalls and solutions
- ✅ Update instructions when code patterns change

**Don't**:
- ❌ Include confidential information
- ❌ Make instructions too long (aim for <200 lines)
- ❌ Duplicate information across files
- ❌ Document every minor detail

## Testing Instructions

### Manual Testing

1. Open a file matching a pattern (e.g., `src/data/scraper.py`)
2. Use Copilot Chat to ask: "What are the coding standards for this file?"
3. Verify the response includes domain-specific guidance

### Validation

Run validation script:
```bash
python3 << 'EOF'
import yaml
import os
import sys

try:
    # Check all instruction files exist
    config_file = '.github/copilot-instructions.yml'
    
    if not os.path.exists(config_file):
        print(f"✗ Configuration file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    if 'instructions' not in config:
        print(f"✗ Configuration missing 'instructions' key")
        sys.exit(1)
    
    for instr in config['instructions']:
        path = os.path.join('.github', instr['path'])
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path} (missing)")
            
except yaml.YAMLError as e:
    print(f"✗ YAML parsing error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF
```

## Resources

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub Copilot Workspace Documentation](https://docs.github.com/en/copilot/using-github-copilot/using-github-copilot-in-your-repository)
- [YAML Frontmatter Specification](https://jekyllrb.com/docs/front-matter/)

## Troubleshooting

### Instructions Not Loading

1. **Check file location**: Files must be in `.github/` or `.github/instructions/`
2. **Verify YAML**: Frontmatter must start and end with `---`
3. **Check patterns**: Use glob patterns like `src/**/*.py`

### Wrong Instructions Activating

1. **Check pattern specificity**: More specific patterns take precedence
2. **Verify configuration**: Ensure `.github/copilot-instructions.yml` is correct
3. **Test patterns**: Use online glob pattern testers

### Instructions Too Long

If Copilot context is too large:

1. **Split by domain**: Create separate files for different areas
2. **Remove redundancy**: Don't repeat common patterns
3. **Focus on specifics**: Only include domain-specific guidance
