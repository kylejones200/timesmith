# Contributing to TimeSmith

Thank you for your interest in contributing to TimeSmith! This document provides guidelines and instructions for contributing.

## Ways to Contribute

- Report bugs and suggest features via GitHub Issues
- Improve documentation - fix typos, add examples, clarify explanations
- Submit bug fixes - help resolve existing issues
- Add new features - implement new time series algorithms or utilities
- Write tests - improve test coverage
- Share examples - contribute tutorial notebooks or example workflows

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/timesmith.git
cd timesmith
```

### 2. Set Up Development Environment

```bash
# Install uv if not already installed: https://github.com/astral-sh/uv
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (uv handles virtual environment automatically)
uv sync --group dev

# For full development with docs and examples:
uv sync --group dev --extra docs --extra examples
```

### 3. Create a Branch

```bash
# Create a new branch for your feature or fix
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

We use Black for code formatting and ruff for linting:

```bash
# Format code
uv run black timesmith tests

# Check linting
uv run ruff check timesmith tests
```

Key conventions:

- Line length: 88 characters (Black default)
- Use type hints where possible
- Follow PEP 8 naming conventions
- Write docstrings for all public functions/classes (Google style)

### Writing Tests

All new features and bug fixes should include tests:

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=timesmith --cov-report=html
```

Test guidelines:

- Place tests in tests/ directory
- Name test files test_*.py
- Name test functions test_*
- Use descriptive test names
- Mock external dependencies
- Aim for >80% code coverage

### Documentation

Update documentation for any user-facing changes:

```bash
# Build documentation locally (using Sphinx)
cd docs
make html
# Or on Windows
make.bat html

# View documentation
open _build/html/index.html
```

Documentation is built with Sphinx and hosted on Read the Docs.

### Commit Messages

Write clear, descriptive commit messages:

```
feat: add spatial cross-validation with buffer zones

- Implement BlockCV class with configurable buffer sizes
- Add tests for edge cases
- Update documentation with examples

Closes #123
```

Commit message format:

- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Test additions/changes
- refactor: Code refactoring
- perf: Performance improvements
- chore: Maintenance tasks

## Testing Checklist

Before submitting a PR, ensure:

- All tests pass: `uv run pytest tests/`
- Code is formatted: `uv run black timesmith tests`
- Linting passes: `uv run ruff check timesmith tests`
- Type checking (optional): `uv run mypy timesmith`
- Documentation builds: `cd docs && make html`
- New features have tests
- New features have documentation

## Submitting a Pull Request

1. Push your changes
2. Create a Pull Request on GitHub
3. Use a clear, descriptive title
4. Reference related issues (e.g., "Fixes #123")
5. Describe what changed and why
6. Respond to reviewer feedback
7. Once approved, a maintainer will merge your PR

## Project Structure

```
timesmith/
├── timesmith/              # Main package
│   ├── __init__.py         # Package initialization and exports
│   ├── __main__.py         # CLI entry point
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── serialization.py    # Model save/load utilities
│   ├── logging_config.py   # Logging configuration
│   ├── core/               # Core base classes and utilities
│   │   ├── base.py         # BaseObject, BaseEstimator, etc.
│   │   ├── validate.py     # Input validation
│   │   ├── data_validation.py  # Data quality checks
│   │   └── ...
│   ├── forecasters/        # Forecasting models
│   ├── compose/            # Pipeline and composition utilities
│   ├── eval/               # Evaluation and backtesting
│   ├── tasks/              # Task definitions
│   ├── typing/             # Type definitions and validators
│   ├── network/            # Network analysis
│   ├── utils/               # Utility functions
│   └── examples/           # Example implementations
├── tests/                  # Test suite
│   ├── test_core_base.py
│   ├── test_forecasters.py
│   ├── test_pipelines.py
│   ├── test_serialization.py
│   └── ...
├── docs/                   # Sphinx documentation
│   └── source/
├── examples/               # Example scripts and notebooks
└── .github/                # GitHub workflows and configs
```

## Questions?

- Open an issue for questions
- Email: <kyletjones@gmail.com>
- Check existing issues and PRs first

Thank you for contributing!
