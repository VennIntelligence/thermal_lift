"""Fig 19 — Detector pitch vs spatial resolution vs SR output grid (ACL-023).

The standard resolution-distinction diagram (docs/plotting_standards.md §
Resolution-Distinction Diagram Standard): one physical-distance axis showing
the 20 µm/px detector sampling pitch, the 20 µm calibrated spatial-resolution
footprint, and the 10 µm/sample 2x SR output grid — three quantities that
must never be conflated. Rendered via the repo's own
thermal_core.ep03.plot_sampling_resolution_diagram.

Run: uv run python docs/publication_figures/scripts/fig19_sampling_resolution.py
"""

from thermal_core.ep03 import (
    build_sampling_resolution_table,
    plot_sampling_resolution_diagram,
)

from pubfig_style import save_fig, setup_academic_style

setup_academic_style()

table = build_sampling_resolution_table(
    detector_pitch_um=20.0,
    spatial_resolution_um=20.0,
    target_grid_um=10.0,
)
fig = plot_sampling_resolution_diagram(
    table,
    detector_pitch_um=20.0,
    spatial_resolution_um=20.0,
    target_grid_um=10.0,
    annotation_fontsize=9.0,
)

paths = save_fig(fig, "fig19_sampling_resolution")
print("\n".join(str(p) for p in paths))
