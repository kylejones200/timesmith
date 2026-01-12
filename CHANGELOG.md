# Changelog

All notable changes to TimeSmith will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-01-XX

### Changed
- Reduced core dependencies from 4 to 2 packages (pandas, numpy only)
- Made scipy optional (moved to `timesmith[scipy]` optional dependency)
- Made networkx optional (moved to `timesmith[network]` optional dependency)
- Removed duplicate statsmodels entry from optional dependencies
- Updated CI to ignore E501 line length errors (handled by formatter)
- Added pre-commit hooks for automatic linting

### Fixed
- Fixed all critical linting errors (F401, F841)
- Fixed unused imports and variables
- Improved error messages for missing optional dependencies

### Added
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- Fallback implementations for scipy functions (median filter)
- Better import error messages for optional dependencies

## [0.1.1] - 2024-XX-XX

### Added
- Initial release of TimeSmith
- Core architecture with four-layer design (Typing, Core, Compose, Tasks & Eval)
- Base classes: BaseObject, BaseEstimator, BaseTransformer, BaseForecaster, BaseDetector, BaseFeaturizer
- Pipeline and adapter composition utilities
- ForecastTask and DetectTask for task semantics
- Backtesting and evaluation tools
- Multiple forecasters (ARIMA, Exponential Smoothing, Moving Averages, etc.)
- Network analysis capabilities
- Data loaders for FRED and Yahoo Finance
- Comprehensive time series utilities

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

[Unreleased]: https://github.com/kylejones200/timesmith/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/kylejones200/timesmith/releases/tag/v0.1.1
