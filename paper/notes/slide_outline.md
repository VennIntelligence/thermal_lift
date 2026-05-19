# Beamer Report Outline

Narrative:

Filename and stage prior were useful for planning, but data-driven alignment
became the evidence boundary. Classic SR supports 2x contour-level visibility,
while MAP-TV and IBP remain risk-controlled candidates rather than metrology
proof.

## Full Deck Plan

1. Title: Thermal Lift 2x contour-level LWIR micro-scan SR.
2. Industrial inspection goal.
3. Final claim boundary: 2x contour-level, not 5 um metrology.
4. Evidence ladder.

### Data Reality and Session Gate

5. Raw data contract: 263 TXT/BMP, 480x640 temperature matrices.
6. Filename order is not acquisition order.
7. Three thermal sessions; main session only.
8. Main-session raster coverage.
9. SR input rule: 255-frame main session.

### Stage/Filename Prior Was Not Alignment Truth

10. Coordinate-to-shift prior: useful geometry, not truth.
11. Raster path explains Y-only calibration failure.
12. Small-step NCC: local diagnostic, not global calibration.
13. AVI direction check supports theta but does not replace config.
14. Historical filename/coordinate NCC story fails.
15. EP05 reassessment: visible motion exists but is data-dependent.

### Physics Boundaries and Localization Gates

16. Detector pitch vs spatial resolution vs output grid.
17. MTF/SNR risk map: possible 2x window, high 4x risk.
18. Noise floor and local contour observability.
19. ESF/CRB is an anchor gate, not SR proof.
20. EP04 global anchor benchmark.
21. Anchor coverage: alignment input, holdout validation, SR target.
22. Inner contour failure means not truth, not not useful.

### Data-Driven Alignment Turning Point

23. Alignment methods compared on same holdout edges.
24. Quantitative reversal: contour refined beats stage prior.
25. 2x phase capacity is full; 3x/4x remain risk diagnostics.
26. Data-driven correction magnitude is not cosmetic.
27. Overlay sanity: visual aid, not SR metric.

### Classic 2x SR POC

28. EP06 method stack: LR, bicubic, SAA, IBP, MAP-TV.
29. Full-view 2x SR comparison.
30. ROI evidence: chip-center contour visibility.
31. Raw-temperature control track.
32. Quantitative summary: SAA stable, IBP sharper-riskier, MAP-TV conservative.
33. MAP-TV forward-model risk: lambda selected by stability, not sharpness.
34. IBP forward-model risk: artifact can rise despite synthetic pass.
35. Alignment sweep: default contour refined remains main alignment.
36. EP06 conclusion: 2x contour-level POC supported, no 4x/metrology claim.

### INR / Deep Prior Exploratory Track

37. EP08 gate: PyTorch forward/highpass equivalence passed.
38. SIREN vs WIRE: WIRE sharper but artifact risk higher.
39. Deep Decoder / Stage 2 status: useful control, not final winner.

### Final Takeaways

40. What was overturned and what survived.
41. Recommended customer-facing statement.

## Compression Notes

For a 30--35 minute report, merge slides 10--12, 18--19, 32--34, and 37--39.
For a defense-style report, keep the full 40-slide outline and place EP08 in an
appendix-style section.

## Core Figures

- `paper/figures/ep01_session_detection.png`
- `paper/figures/ep01_raster_trajectory.png`
- `paper/figures/ep02_data_driven_alignment.png`
- `paper/figures/ep03_mtf_snr_recoverability.png`
- `paper/figures/ep04_gate_recommendations.png`
- `paper/figures/ep05_alignment_methods.png`
- `paper/figures/ep05_alignment_tuning_heatmap.png`
- `paper/figures/ep06_main_comparison.png`
- `paper/figures/ep06_raw_control.png`
- `paper/figures/ep06_center_raw_temperature.png`
- `paper/figures/ep06_sweep_metrics.png`
- `paper/figures/ep06_sweep_lambda.png`

