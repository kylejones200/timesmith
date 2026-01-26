# Migration to uv

This project has been migrated from pip/requirements.txt to [uv](https://github.com/astral-sh/uv) for dependency management.

## Quick Reference: Old vs New Commands

### Installation

| Old (pip) | New (uv) |
|-----------|----------|
| `pip install timesmith` | `pip install timesmith` (still works for end users) |
| `pip install -e ".[dev]"` | `uv sync --group dev` |
| `pip install -e ".[dev,docs,all]"` | `uv sync --group dev --extra docs --extra examples` |

### Running Commands

| Old (pip) | New (uv) |
|-----------|----------|
| `pytest tests/` | `uv run pytest tests/` |
| `black timesmith tests` | `uv run black timesmith tests` |
| `ruff check timesmith tests` | `uv run ruff check timesmith tests` |
| `flake8 timesmith tests` | `uv run ruff check timesmith tests` |
| `mypy timesmith` | `uv run mypy timesmith` |
| `python -m build` | `uv run python -m build` |
| `twine check dist/*` | `uv run twine check dist/*` |

### Development Workflow

| Old (pip) | New (uv) |
|-----------|----------|
| `python -m venv venv`<br>`source venv/bin/activate`<br>`pip install -e ".[dev]"` | `uv sync --group dev` (uv manages venv automatically) |
| `pip install <package>` | `uv add <package>` (for runtime deps)<br>`uv add --group dev <package>` (for dev deps) |
| `pip freeze > requirements.txt` | Not needed - `uv.lock` is auto-generated |

## Key Changes

1. **Single source of truth**: `pyproject.toml` is now the single source of truth for all dependencies
2. **Lock file**: `uv.lock` is committed to the repository for reproducible builds
3. **Dev dependencies**: Moved from `[project.optional-dependencies].dev` to `[dependency-groups].dev`
4. **CI/CD**: GitHub Actions workflows now use `uv sync --frozen` and `uv run` for all commands
5. **Virtual environment**: uv automatically manages the virtual environment - no need to activate manually

## Benefits

- **Faster**: uv is 10-100x faster than pip
- **Reproducible**: Lock file ensures consistent installs across environments
- **Simpler**: No need to manually manage virtual environments
- **Modern**: Uses PEP 621 standard for project metadata

## Installation

If you don't have uv installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## For Contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for updated development setup instructions.

## Legacy Files

- `docs/requirements.txt` - Kept for ReadTheDocs compatibility, but `pyproject.toml` is the source of truth

