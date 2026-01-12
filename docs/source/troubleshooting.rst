Troubleshooting
===============

Common issues and solutions when using TimeSmith.

Installation Issues
-------------------

Problem: Import errors for optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see import errors for forecasters like ``ProphetForecaster`` or ``LSTMForecaster``, you need to install the optional dependencies:

.. code-block:: bash

   # For Prophet
   pip install timesmith[forecasters]

   # For LSTM
   pip install timesmith[deep_learning]

   # For all optional dependencies
   pip install timesmith[all]

Problem: Python version compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TimeSmith requires Python 3.12 or higher. If you're using an older version:

.. code-block:: bash

   # Check your Python version
   python --version

   # Upgrade to Python 3.12+
   # Use pyenv, conda, or download from python.org

Model Fitting Issues
--------------------

Problem: "NotFittedError" when calling predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This error occurs when you try to predict without fitting the model first.

**Solution:**

.. code-block:: python

   from timesmith import SimpleMovingAverageForecaster
   from timesmith.exceptions import NotFittedError

   forecaster = SimpleMovingAverageForecaster(window=5)

   # Always fit before predicting
   forecaster.fit(y)
   forecast = forecaster.predict(fh=10)

Problem: "Need at least N data points"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some forecasters require a minimum amount of data to fit properly.

**Solution:**

.. code-block:: python

   # Check data length
   print(f"Data length: {len(y)}")

   # Use a forecaster with lower requirements
   from timesmith import NaiveForecaster
   forecaster = NaiveForecaster()  # Works with minimal data

   # Or ensure you have enough data
   if len(y) < required_length:
       raise ValueError(f"Need at least {required_length} data points")

Data Validation Issues
----------------------

Problem: "ValidationError" with data types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TimeSmith expects specific data formats (SeriesLike, PanelLike, TableLike).

**Solution:**

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Convert numpy array to Series
   y_array = np.array([1, 2, 3, 4, 5])
   y = pd.Series(y_array, index=pd.date_range("2020-01-01", periods=5))

   # Ensure datetime index
   from timesmith.utils import ensure_datetime_index
   y = ensure_datetime_index(y)

Problem: Data contains NaN or infinite values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some operations don't handle missing or infinite values well.

**Solution:**

.. code-block:: python

   # Remove NaN values
   y_clean = y.dropna()

   # Fill missing values
   y_filled = y.fillna(method='ffill')  # Forward fill
   y_filled = y.fillna(y.mean())        # Mean imputation

   # Check for infinite values
   import numpy as np
   if np.any(np.isinf(y)):
       y = y.replace([np.inf, -np.inf], np.nan).dropna()

Serialization Issues
--------------------

Problem: "Failed to save model"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model serialization can fail if the model contains non-serializable objects.

**Solution:**

.. code-block:: python

   from timesmith import save_model, load_model

   # Try different formats
   save_model(forecaster, "model.pkl", format="pickle")
   # Or
   save_model(forecaster, "model.joblib", format="joblib")

   # Check if model is fitted
   if not forecaster.is_fitted:
       raise ValueError("Model must be fitted before saving")

Problem: "Failed to load model"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loading can fail if the model file is corrupted or from a different version.

**Solution:**

.. code-block:: python

   # Check file exists
   from pathlib import Path
   if not Path("model.pkl").exists():
       raise FileNotFoundError("Model file not found")

   # Try loading with explicit format
   loaded = load_model("model.pkl", format="pickle")

   # Check metadata if available
   from timesmith.serialization import get_model_metadata
   try:
       metadata = get_model_metadata("model.pkl")
       print(f"Model class: {metadata['class_name']}")
       print(f"TimeSmith version: {metadata.get('timesmith_version', 'unknown')}")
   except:
       pass

Pipeline Issues
---------------

Problem: Pipeline steps not executing in order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure transformers are added in the correct order.

**Solution:**

.. code-block:: python

   from timesmith import make_forecaster_pipeline
   from timesmith.examples import LogTransformer, NaiveForecaster

   # Order matters: transformers first, then forecaster
   transformer = LogTransformer(offset=1.0)
   forecaster = NaiveForecaster()
   pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

   # Fit and predict
   pipeline.fit(y)
   forecast = pipeline.predict(fh=10)

Performance Issues
-----------------

Problem: Slow performance with large datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TimeSmith operations can be slow with very large time series.

**Solution:**

.. code-block:: python

   # Use joblib for parallel processing (if available)
   pip install timesmith[performance]

   # Resample to lower frequency if appropriate
   from timesmith.utils import resample_ts
   y_resampled = resample_ts(y, freq="W")  # Weekly instead of daily

   # Use simpler forecasters for large datasets
   from timesmith import SimpleMovingAverageForecaster
   forecaster = SimpleMovingAverageForecaster(window=7)

Logging and Debugging
---------------------

Problem: Too much or too little logging output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adjust logging level via environment variables or code.

**Solution:**

.. code-block:: python

   import os
   os.environ["TIMESMITH_LOG_LEVEL"] = "DEBUG"  # or INFO, WARNING, ERROR

   # Or programmatically
   from timesmith.logging_config import configure_logging
   configure_logging(level="INFO", format_string="simple")

Problem: Understanding error messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TimeSmith exceptions include context for debugging.

**Solution:**

.. code-block:: python

   from timesmith import NotFittedError, DataError

   try:
       forecaster.predict(fh=5)
   except NotFittedError as e:
       print(f"Error message: {e.message}")
       print(f"Context: {e.context}")
       # Context includes: estimator name, operation, etc.

Getting Help
------------

If you're still experiencing issues:

1. **Check the documentation**: https://timesmith.readthedocs.io/
2. **Search existing issues**: https://github.com/kylejones200/timesmith/issues
3. **Create a new issue**: Include:
   - Python version
   - TimeSmith version
   - Minimal reproducible example
   - Full error traceback
4. **Email**: kyletjones@gmail.com

Common Error Messages
---------------------

- ``NotFittedError``: Call ``fit()`` before ``predict()``
- ``ValidationError``: Check data format and types
- ``DataError``: Check data quality (NaN, length, etc.)
- ``ConfigurationError``: Check parameter values
- ``UnsupportedOperationError``: Operation not available for this estimator

