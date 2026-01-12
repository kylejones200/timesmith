# Development Setup

This guide helps you set up a local development environment to run CI checks before pushing.

## Quick Start

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e ".[dev]"
   ```

3. **Run CI checks locally:**
   ```bash
   ./run_ci_checks.sh
   ```

## Manual CI Checks

You can also run individual checks:

```bash
# Activate venv first
source venv/bin/activate

# Run tests
pytest -v --tb=short --maxfail=5 --durations=10 --cov=timesmith --cov-report=term-missing

# Run linting (critical issues only)
ruff check timesmith/ tests/ --select E,F

# Check formatting
ruff format --check timesmith/ tests/

# Type checking (non-blocking)
mypy timesmith/ --show-error-codes
```

## Pre-Push Workflow

Before pushing to GitHub, run:

```bash
./run_ci_checks.sh
```

This runs the same checks as GitHub Actions CI, so you can catch issues locally before pushing.

## Virtual Environment

The `venv/` directory is already in `.gitignore` and won't be committed.

To deactivate the virtual environment:
```bash
deactivate
```

