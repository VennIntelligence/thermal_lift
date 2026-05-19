# AAAI Paper Outline

Working title:

**Evidence-Bounded 2x Thermal Micro-Scan Super-Resolution for Chip Contour Inspection**

Alternative:

**Data-Driven Micro-Scan Alignment for Contour-Level Thermal Super-Resolution**

## Abstract Draft

Long-wave infrared chip inspection often provides raw temperature matrices but
no paired high-resolution thermal ground truth, making conventional
super-resolution claims difficult to validate. We study a bounded 2x
contour-level thermal SR setting using a 255-frame step-and-shoot micro-scan
session, where the goal is improved visibility of chip-internal contours rather
than 5 um metrology or absolute-temperature recovery. Our pipeline treats stage
coordinates only as priors, then estimates data-driven alignment through
high-pass NCC initialization and contour-refined quality gates. On the main
session, data-driven contour refinement reduces held-out contour Chamfer from
0.381 px without alignment and 0.240 px with stage prior to 0.134 px, while
preserving all four 2x phase bins. We compare SAA, IBP, and MAP-TV under the
same alignment and evaluation protocol, using highpass structure maps,
raw-temperature controls, split-half consistency, artifact audits, and contour
proxies. SAA provides the most stable multi-frame baseline, IBP tests
forward-model sharpening but raises artifact risk, and MAP-TV selects strong
regularization and acts as a conservative diagnostic rather than the sharpest
method. The result is an evidence-bounded 2x thermal SR POC for chip contour
inspection, not a claim of 4x recovery or calibrated 5 um spatial resolution.

## Structure

1. Introduction
   - Claim: industrial LWIR chip SR needs an evidence-bounded formulation
     because there is no HR thermal truth or optical registration.
   - Contributions: main-session 2x POC, data-driven alignment, SAA/IBP/MAP-TV
     ablation, raw-control and artifact gates.

2. Problem Setting and Claim Boundary
   - Separate detector pitch, spatial resolution, and output-grid sampling.
   - Target contour visibility, not metrology.
   - Use `ep01_sr_data_basis_summary.csv` and EP03 physical constants.

3. Main-Session Data and Micro-Scan Alignment
   - Only the 255-frame main session is valid SR input.
   - Stage and filename shifts are priors, not truth.
   - Data-driven NCC + contour refinement gives better held-out alignment and
     full 2x phase coverage.

4. Quality-Gated Contour Anchors
   - EP04 localization is an alignment anchor and holdout gate.
   - Inner contour failures remain SR targets, not truth.

5. 2x Reconstruction Methods
   - SAA, IBP, and MAP-TV share frame set, 2x grid, and alignment convention.
   - Highpass track is a structure-map reconstruction.
   - Raw-temperature track is a control.

6. Experiments and Evaluation Protocol
   - No single proxy metric proves SR.
   - Evidence combines visual contour clarity, raw-control agreement,
     split-half, artifact score, and EP04 Chamfer proxy.
   - Synthetic validation is only a smoke test.

7. Results and Ablations
   - Show `comparison_fullview.png`, `comparison_center_raw_temperature.png`,
     `sweep_metric_bars.png`, and `sweep_map_tv_lambda_selection.png`.
   - Default contour refined remains preferred.
   - Tuned refined is sensitivity evidence.
   - NCC init is a phase-prior control with higher artifact risk.

8. Limitations and Conclusion
   - Current result is an evidence-bounded 2x contour-level POC.
   - Future work: EP07 phantom benchmarking, EP08 INR methods, cross-ROI
     stability, edge-transfer validation.

## Claims to Avoid

- Do not claim 5 um spatial resolution or 5 um temperature metrology.
- Do not claim 4x SR or true physical-resolution doubling.
- Do not treat stage command, filename affine, AVI direction, or EP04 anchors
  as ground truth.
- Do not prove SR from gradient, Chamfer, split-half, residual, or artifact
  proxy alone.
- Do not call MAP-TV best or sharpest.
- Do not call IBP a clear winner.
- Do not interpret highpass red/blue response as absolute temperature.
- Do not mix cross-session frames or use BMP/AVI as numeric SR input.

