Architecture
============

Timesmith follows a four layer architecture.

The first layer holds objects. Objects represent data and meaning. They include time series views, panels, and result containers. Objects never depend on algorithms or workflows.

The second layer holds primitives. Primitives implement methods. A primitive may fit a model, transform a series, or compute a score. Primitives never load data. Primitives never plot.

The third layer holds tasks. Tasks bind intent. A task defines what problem is being solved. Forecasting, detection, and simulation live here. Tasks validate inputs and orchestrate primitives.

The fourth layer holds workflows. Workflows handle reality. They load files. They save results. They call plotting libraries. They connect Timesmith to the outside world.

Each layer imports only from layers below it. This rule prevents accidental coupling. It also allows downstream libraries to reuse the same mental model.

Timesmith typing lives outside this stack. SeriesLike and PanelLike form a shared language across the smith ecosystem. Anomsmith, ResSmith, GeoSmith, and Plotsmith all rely on the same definitions.

This structure trades convenience for clarity. It pays off as soon as a project grows beyond one notebook.

