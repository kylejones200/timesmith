Design Decisions
=================

Timesmith makes strong choices.

Timesmith does not hide state. Fitted models expose parameters. Results expose tables. Nothing mutates silently.

Timesmith does not mix plotting with computation. Visualization belongs to Plotsmith. This keeps Timesmith dependency light and predictable.

Timesmith does not invent new data containers when pandas already works. It wraps meaning, not arrays.

Timesmith does not optimize for one model family. It optimizes for composition. A simple model that fits cleanly beats a complex model that breaks pipelines.

Timesmith does not chase every edge case. It enforces clear contracts. If inputs violate those contracts, Timesmith fails early with clear errors.

These decisions make the library smaller. They also make it usable across many domains.

