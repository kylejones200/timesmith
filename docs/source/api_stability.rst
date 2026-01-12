API Stability
=============

TimeSmith follows semantic versioning and provides API stability guarantees.

Versioning Policy
-----------------

TimeSmith uses `Semantic Versioning <https://semver.org/>`_ (SemVer):

- **MAJOR** version (X.0.0): Breaking changes
- **MINOR** version (0.X.0): New features, backward compatible
- **PATCH** version (0.0.X): Bug fixes, backward compatible

Current Version
---------------

TimeSmith is currently at version **0.1.1**, which means:

- The API is still evolving
- Minor version updates may include new features
- Major version updates (1.0.0+) will maintain backward compatibility for at least one major version

Stable APIs (Post-1.0)
-----------------------

Once TimeSmith reaches 1.0.0, the following will be considered stable:

- Core base classes (BaseObject, BaseEstimator, BaseForecaster, etc.)
- Public API functions in ``timesmith/__init__.py``
- Exception hierarchy
- Serialization API (save_model, load_model)
- Task definitions (ForecastTask, DetectTask)
- Evaluation functions (backtest_forecaster, summarize_backtest)

Experimental APIs
-----------------

The following are considered experimental and may change:

- Network analysis functions (may be reorganized)
- Some utility functions (may be moved or renamed)
- Internal implementation details

Deprecation Policy
-----------------

When APIs are deprecated:

1. **Deprecation notice**: Warnings will be issued for at least one minor version
2. **Documentation**: Deprecated APIs will be marked in documentation
3. **Removal**: Deprecated APIs will be removed in the next major version

Example:

.. code-block:: python

   import warnings
   warnings.filterwarnings('default', category=DeprecationWarning)

   # Deprecated function will show warning
   old_function()  # DeprecationWarning: old_function is deprecated

Migration Guide
---------------

When upgrading between versions:

- **Patch versions (0.1.X)**: No changes needed
- **Minor versions (0.X.0)**: Check release notes for new features
- **Major versions (X.0.0)**: Check migration guide in CHANGELOG.md

Checking Your Version
---------------------

.. code-block:: python

   import timesmith
   print(timesmith.__version__)

Reporting Issues
---------------

If you encounter breaking changes in a non-major version update, please report it:

- GitHub Issues: https://github.com/kylejones200/timesmith/issues
- Email: kyletjones@gmail.com

