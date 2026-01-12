#!/bin/bash
# Run CI checks locally before pushing

set -e  # Exit on error

echo "=========================================="
echo "Running TimeSmith CI Checks Locally"
echo "=========================================="
echo ""

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "Activated virtual environment"
    else
        echo "Error: venv not found. Run: python3 -m venv venv && source venv/bin/activate && pip install -e '.[dev]'"
        exit 1
    fi
fi

echo ""
echo "1. Running tests with coverage..."
echo "-----------------------------------"
pytest -v --tb=short --maxfail=5 --durations=10 --cov=timesmith --cov-report=term-missing --cov-report=xml

echo ""
echo "2. Linting with ruff (critical issues only)..."
echo "-----------------------------------"
ruff check timesmith/ tests/ --select E,F

echo ""
echo "3. Format check with ruff..."
echo "-----------------------------------"
ruff format --check timesmith/ tests/ || echo "WARNING: Formatting issues found"

echo ""
echo "4. Type checking with mypy (non-blocking)..."
echo "-----------------------------------"
mypy timesmith/ --show-error-codes || echo "INFO: Type check issues found (non-blocking)"

echo ""
echo "=========================================="
echo "All CI checks completed!"
echo "=========================================="

