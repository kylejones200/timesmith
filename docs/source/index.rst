TimeSmith
=========

Timesmith builds structure from time.

Most time series libraries optimize for models. Timesmith optimizes for work. It treats time series analysis as a system made of data representations, methods, tasks, and workflows. Each part stays separate. Each part stays testable. The result stays readable.

Timesmith fits practitioners who work with real series. These series arrive late, break alignment, change frequency, and grow sideways into panels. Timesmith accepts this reality. It provides a stable core that downstream libraries build on.

Timesmith does not aim to replace existing forecasting libraries. It provides the connective tissue they lack. You can fit a model, evaluate it, compose it into pipelines, and reuse the same semantics across forecasting, anomaly detection, and domain packages.

If you already work with pandas, NumPy, and Matplotlib, Timesmith will feel familiar. If you care about correctness and reuse, it will feel necessary.

Start with the getting started guide. It shows the full path from raw series to evaluated results.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   architecture
   design_decisions
   api_reference
   troubleshooting
   api_stability

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
