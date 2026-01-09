Getting Started
================

Quick Example
-------------

Here's a simple example of using TimeSmith:

.. code-block:: python

   import pandas as pd
   import timesmith as ts

   # Create a time series
   dates = pd.date_range('2020-01-01', periods=100, freq='D')
   values = [100 + i * 0.1 + np.random.randn() for i in range(100)]
   y = pd.Series(values, index=dates)

   # Create a forecast task
   task = ts.ForecastTask(y=y, fh=10, frequency='D')

   # Create and fit a forecaster
   forecaster = ts.SimpleMovingAverageForecaster(window=7)
   forecaster.fit(y)

   # Make predictions
   forecast = forecaster.predict(task.fh)
   print(forecast.predicted)

Architecture
------------

TimeSmith uses a four-layer architecture:

1. **Typing Layer**: Scientific types with runtime validation
2. **Core Layer**: Base classes for transformers, forecasters, detectors, and featurizers
3. **Compose Layer**: Pipelines, adapters, and feature unions
4. **Tasks & Eval Layer**: Task definitions and evaluation tools

For more examples, see the `example notebooks <https://github.com/kylejones200/timesmith/tree/main/examples/notebooks>`_.

