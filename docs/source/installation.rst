Installation
============

TimeSmith can be installed via pip:

.. code-block:: bash

   pip install timesmith

Optional Dependencies
---------------------

TimeSmith has several optional dependencies for extended functionality:

.. code-block:: bash

   # For forecasting methods
   pip install timesmith[forecasters]

   # For network analysis
   pip install timesmith[network]

   # For change point detection
   pip install timesmith[changepoint]

   # For Bayesian forecasting
   pip install timesmith[bayesian]

   # For all optional dependencies
   pip install timesmith[all]

Development Installation
------------------------

For development:

.. code-block:: bash

   git clone https://github.com/kylejones200/timesmith.git
   cd timesmith
   pip install -e ".[dev]"

