Getting Started
===============

This guide walks through one complete TimeSmith workflow. It uses a single time series and a simple model. The goal is clarity, not performance.

Creating Data
-------------

We start with a pandas Series that represents monthly demand.

.. code-block:: python

   import numpy as np
   import pandas as pd

   # Set random seed for reproducibility
   np.random.seed(42)

   # Create monthly time series data
   idx = pd.date_range("2020-01-01", periods=60, freq="M")
   y = pd.Series(100 + np.cumsum(np.random.normal(0, 2, size=60)), index=idx)

TimeSmith treats this as a SeriesLike object. Validation happens once at the boundary.

.. code-block:: python

   from timesmith.typing.validators import assert_series_like

   assert_series_like(y)  # Validates the data format

Defining a Task
---------------

We now define a forecasting task. The task holds meaning. The model holds parameters. Evaluation stays separate.

.. code-block:: python

   from timesmith.tasks import ForecastTask

   task = ForecastTask(
       y=y,
       fh=12,  # Forecast 12 months ahead
       frequency="M"  # Monthly frequency
   )

Choosing a Forecaster
---------------------

We choose a forecaster. Any forecaster that follows the TimeSmith interface will work here.

.. code-block:: python

   from timesmith.forecasters import SimpleMovingAverageForecaster

   forecaster = SimpleMovingAverageForecaster(window=3)

   # Fit the forecaster
   forecaster.fit(y)

   # Make predictions
   forecast = forecaster.predict(fh=12)
   print(f"Forecast: {forecast.y_pred}")

Running a Backtest
------------------

We evaluate the model with a backtest using time series cross-validation.

.. code-block:: python

   from timesmith.eval import backtest_forecaster, summarize_backtest

   # Run backtest
   result = backtest_forecaster(
       forecaster=forecaster,
       task=task
   )

   # Summarize results
   summary = summarize_backtest(result)
   print(f"Mean MAE: {summary['aggregate_metrics']['mean_mae']:.4f}")
   print(f"Mean RMSE: {summary['aggregate_metrics']['mean_rmse']:.4f}")

The result is a table. Each row represents one cutoff. Each column has clear meaning.

.. code-block:: python

   print(result.results.head())

Using Pipelines
---------------

TimeSmith supports composing transformers and forecasters into pipelines:

.. code-block:: python

   from timesmith import make_forecaster_pipeline
   from timesmith.examples import LogTransformer, NaiveForecaster

   # Create a pipeline with transformation and forecasting
   transformer = LogTransformer(offset=1.0)
   forecaster = NaiveForecaster()
   pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

   # Fit and predict
   pipeline.fit(y)
   forecast = pipeline.predict(fh=12)

Summary
-------

This is the full TimeSmith loop. Data enters once. Tasks define intent. Models do work. Evaluation reports results. Nothing hides state. Nothing mixes concerns.

For more examples, see the `example notebooks <https://github.com/kylejones200/timesmith/tree/main/examples/notebooks>`_.
