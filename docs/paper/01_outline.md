# Paper Outline

## Working titles

1. **Trust but Verify: Evidence-Bounded Burst Super-Resolution for Thermal Chip Inspection without Ground Truth** (preferred)
2. When Can You Trust Super-Resolution without Ground Truth? A Calibrated Thermal Microscan Study
3. Evidence-Bounded 2x Thermal Micro-Scan Super-Resolution for Chip Contour Inspection (legacy, EP01–EP06 era)

## Abstract skeleton

- Industrial LWIR chip inspection yields raw temperature matrices but no paired HR thermal
  ground truth; conventional SR claims are unverifiable in this regime.
- We study a bounded 2x contour-level burst-SR setting on a fully calibrated microscan system
  (measured stage-to-pixel rotation, PSF range, noise floor, raster acquisition structure).
- Three findings structure the paper:
  (i) **Information existence**: phase-stratified split-half FRC bounds the coherent information
  (1/7 cutoff 17.0 µm); fine-grid MAP-TV deconvolution converts it into limited but real contour
  gains (zigzag median FWHM 114→100 µm).
  (ii) **Information delivery**: with 1x-grid statistical inputs, learned models are invariant to
  loss design w.r.t. the finest structures — the sub-pixel phase information collapses before the
  network sees it; injecting it via 2x drizzle input channels restores it. [pending V9A]
  (iii) **Information fidelity**: trained-on-synthetic networks drift along the *null space* of the
  observation operator — forward-consistency losses sit at their floor while real-data artifact
  proxies degrade monotonically; the drift is invisible to observation anchoring and must be
  handled by evidence injection and checkpoint-selection protocols.
- Together these yield a GT-free evaluation protocol and an honest verdict: in this no-GT regime
  there is **no certifiable single winner**. Classical anchored TGV / MAP-TV is the most
  observation-faithful method on the verifiable proxies but imprints a mild total-variation
  staircase on the finest contours; evidence-injected learning is sharpest by inspection but
  carries the most unverifiable high-frequency content and drifts on observation fidelity. We
  report verifiable fidelity proxies, a structural grain proxy with dual-domain visual panels,
  and an explicitly task-level contour-legibility preference — none of which licenses
  metrology-grade claims.

## Section plan

1. **Introduction** (`02_introduction.md`)
   - Hook: no-GT industrial SR; hallucination risk is the central obstacle, not sharpness.
   - Contributions C1–C4 (below).
2. **Related work** (`03_related_work.md`)
   - Classical MFSR; deep burst SR; thermal SR; sim-to-real degradation modeling;
     GT-free evaluation (FRC, split-half); range/null-space decompositions & data consistency.
3. **Problem setting and calibrated forward model** (`04_problem_forward_model.md`)
   - System, data sessions, calibration chain (θ, PSF, noise), claim boundary
     (pitch ≠ resolution ≠ output grid), alignment gates.
4. **Method** (`05_method.md`)
   - Matched synthetic platform (TCForge); contour-oriented loss; anchor variants;
     evidence-injecting hybrid drizzle input; classical arms.
5. **GT-free evaluation protocol** (`06_evaluation_protocol.md`)
   - FRC bands + controls; coupled proxies (artifact / raw-control corr) and why they
     anti-correlate; null-space drift diagnosis; checkpoint-selection rule; visual gates.
6. **Experiments** (`07_experiments.md`)
   - Main comparison (classical vs learned); drift trajectories; ablation matrix
     (input × anchor); frame budget; robustness; selection protocol in action.
7. **Limitations and conclusion** (`08_limitations_conclusion.md`)

## Contributions

- **C1 — Calibrated no-GT benchmark setting.** A fully measured LWIR microscan burst-SR problem
  (rotation 47.6°±0.1, PSF σ∈[0.2,0.5] LR px with arbitration, noise floor 0.0724 °C, raster
  acquisition structure, 248-frame clean main session) plus a physics-matched synthetic
  generation platform. The forward model is measured, not assumed.
- **C2 — GT-free evaluation protocol.** Phase-stratified split-half FRC with positive/negative
  controls; observation-anchored proxy pair with a coupling analysis showing they read the same
  stylization axis in opposite signs; a mechanical checkpoint-selection rule on the proxy Pareto
  front gated by visual panels.
- **C3 — Null-space drift: finding and remedy.** Synthetic-prior-trained networks drift in the
  null space of the (PSF ∘ downsample) observation operator: forward-consistency loss flat at
  ~1e-2 floor while real-data artifact score climbs 0.37→0.65; band-limited and full-band anchors
  do not suppress it [V9D pending]. Remedy: inject sub-pixel evidence at the *input* (2x drizzle
  channels) [V9A pending] and select checkpoints on the proxy Pareto front.
- **C4 — Honest classical-vs-learned verdict: complementary failure modes, no GT-certifiable
  winner.** Anisotropic coverage-weighted TGV resolves raster-anisotropy stripe artifacts
  (artifact 3.87→0.695, raw-control corr 0.916) and is the most observation-faithful method on the
  fidelity proxies, but it imprints a mild total-variation staircase ("beading") on the finest
  diagonal contours — an artifact absent from the continuous multi-frame observation and reflected
  in TGV being the highest-`lattice` faithful reference. Evidence-injected learned reconstructions
  are sharpest by inspection and resolve the central fine comb most crisply, but carry the most
  high-frequency content (plausibly-real detail mixed with unverifiable grain) and drift measurably
  on observation fidelity. The verdict is three-part and GT-free — verifiable fidelity proxies
  (favor classical/drizzle), a structural grain proxy plus dual-domain panels, and an explicitly
  task-level contour-legibility preference — and certifies no single winner; the visual preference
  is not fidelity evidence and nothing here supports metrology-grade claims.

## Figures (target ~7 + 2 tables; assets in `09_figures_tables_assets.md`)

- F1 system + calibration chain + raster/microscan schematic (+ pitch/resolution/grid distinction)
- F2 information existence: FRC curves with controls + band table
- F3 null-space drift: per-arm trajectories (artifact, corr vs step) + forward-loss floor inset
- F4 proxy Pareto + checkpoint selection (TGV reference point)
- F5 main visual comparison: bicubic / drizzle / TGV / MAP-TV / UNet best (temperature + highpass),
  with a high-zoom central-comb row making the TGV staircase vs learned grain failure modes visible
- F6 input-mode ablation visuals: center thin lines, edge staircase (v8.1a vs v9a vs v9c)
- F7 frame-budget and robustness curves
- T1 main quantitative table (single-harness numbers)
- T2 ablation matrix (input × anchor)

## Claims to avoid (red-team checklist; inherited + extended)

- No 5 µm spatial-resolution or temperature-metrology claims; 5 µm output sample ≠ 5 µm resolution;
  2x grid Nyquist period is 10 µm.
- No 4x/5x physical recovery claims; fine grids are contour oversampling.
- Stage commands / filename coordinates / EP04 anchors are priors and gates, **never** alignment
  ground truth.
- FRC high-frequency rebound (10–12 µm band) is flagged as coverage/lattice + drift risk — do not
  cite it as aperture-zero or resolution evidence. Cutoff statement stays at 17.0 µm.
- Proxy metrics (split-half, artifact score, raw-control corr, Tenengrad, Chamfer) never prove SR
  alone; they gate and select.
- TGV (and likely MAP-TV, same TV family) imprints a mild staircase/beading on the finest diagonal
  contours; never present classical reconstruction as an unconditional / "trustworthy-deliverable"
  winner without this caveat. There is no GT-certifiable single winner.
- Learned outputs are sharper by inspection but carry the most high-frequency content (highest
  `lattice`), unverifiable without GT; never call them "cleaner" or more faithful — visual
  preference is not fidelity evidence.
- `sharp_p95` is blind to contour continuity and is inflated by both beading and grain; never cite
  sharpness alone — pair it with `lattice` and the dual-domain visual gate.
- Any "contour legibility" gain from inspection is an explicitly task-level preference, listed
  separately from verifiable proxies, never resolution or fidelity evidence.
- MAP-TV zigzag gain is "limited contour enhancement" (mixed per-profile), not a strong positive.
- Do not compare proxy values across input modes (1x-stat vs hybrid drizzle inputs).
- Do not mix cross-session frames; do not use rendered AVI/BMP as numeric SR input.
- v8.1b (PixelShuffle) and EP12 4x are reported as negative results, not silently dropped.
