Getting Started
===============

This guide walks through one complete Timesmith workflow. It uses a single time series and a simple model. The goal is clarity, not performance.

We start with a pandas Series that represents monthly demand.

.. code-block:: python

   import numpy as np
   import pandas as pd

   idx = pd.date_range("2020-01-01", periods=60, freq="M")
   y = pd.Series(100 + np.cumsum(np.random.normal(0, 2, size=60)), index=idx)

Timesmith treats this as a SeriesLike object. Validation happens once at the boundary.

.. code-block:: python

   from timesmith.typing.validators import assert_series_like

   assert_series_like(y)

We now define a forecasting task. The task holds meaning. The model holds parameters. Evaluation stays separate.

.. code-block:: python

   from timesmith.tasks import ForecastTask

   task = ForecastTask(
       y=y,
       fh=12
   )

We choose a forecaster. Any forecaster that follows the Timesmith interface will work here.

.. code-block:: python

   from timesmith.forecasters import SimpleMovingAverageForecaster

   model = SimpleMovingAverageForecaster(window=3)

We evaluate the model with a backtest.

.. code-block:: python

   from timesmith.eval import backtest_forecaster

   results = backtest_forecaster(
       forecaster=model,
       task=task
   )

The result is a table. Each row represents one cutoff. Each column has clear meaning.

.. code-block:: python

   print(results.head())

This is the full Timesmith loop. Data enters once. Tasks define intent. Models do work. Evaluation reports results. Nothing hides state. Nothing mixes concerns.
