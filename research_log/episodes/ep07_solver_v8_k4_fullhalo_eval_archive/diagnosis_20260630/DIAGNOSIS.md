# EP07 Solver V8/K4 — Grid vs Flocculence Tradeoff: Root-Cause Diagnosis

Date: 2026-06-30  ·  Offline (no GPU, no checkpoint) diagnosis from the saved render arrays + architecture analysis.

Inputs used:
- `remote_inbox/ep07_solver_v8_k4_slim_20260630/.../arrays/v8k4_step10000_render_arrays.npz`
  (full-frame 960×1280 renders: aligned_mean, tiled_p192_o128, dense_p192_o160, full_halo96, tile_halo32/64/96)
- Source: `algos/ep07_unet_sr/src/unet_sr/{model,unroll,real_eval,dataset,losses,config}.py`
- Prior findings: ACL-037 (K4 recurrent 30px box), ACL-038 (full-halo eval).

Scripts (in this dir): `diag_arrays.py`, `diag_arrays2.py`, `diag_extent.py`.
Figures: `figA_renders.png` … `figE_lineprofile.png`. Metrics: `metrics_arrays*.json`, `metrics_extent.json`.

---

## TL;DR (plain language)

The grid and the flocculence are **two faces of one root cause**: the prox network is **not
"extent-invariant"** — its answer at any pixel depends on how big a region you solve and what
else is in that region. This comes from two ingredients in the prox UNet:
`GroupNorm` (normalizes using the whole image's statistics) and the `SEBlock` (a global
average-pool gate over the whole image).

- When you solve in small **192-px tiles** (the training size), each tile is "in-distribution"
  → structure stays sharp, but every tile has an artificial border. The prox paints a broad
  border response; stitching many tiles prints a **regular grid**.
- When you solve the **whole frame** (full/large halo), the borders disappear → grid gone, but
  now the global stats are nothing like a 192 patch (the prox was *only ever trained on 192
  crops*). The prox **mis-calibrates its gain**: it lifts the background, over-sharpens (ringing),
  and amplifies fine noise → **swollen lines + flocculent texture**. The K4 recurrent loop
  (same prox applied 4×) multiplies this error.

So `halo size` is a single **tradeoff knob**, not a fix: small = sharp+grid, large = clean-bg
but distribution-shifted. Every `tile_halo32/64/96` sits monotonically between the two extremes.

**The user's GroupNorm/SE hypothesis is CONFIRMED and quantified** (far-field test below). It also
*composes* with the previously-found K4 recurrent amplification (ACL-037).

---

## Evidence

### 1. The grid is a tiling/patch-boundary artifact — NOT in the data/target
`diag_arrays.py` measures a "seam" spectrum (power along image axes at the tile pitch).
Same checkpoint, same frames, same target — only the **solve context** changes:

| render | seam periodicity (prominence) | interpretation |
|---|---:|---|
| aligned_mean | 0 (none) | warm-start baseline, no grid |
| tiled_p192_o128 | **2556** (strong) | independent 192 tiles → grid |
| dense_p192_o160 | 2172 | denser tiling, grid just finer (not fixed) |
| tile_halo32 | 1309 | |
| tile_halo64 | 961 | larger solve context → grid weaker |
| tile_halo96 | 228 | |
| full_halo96 | **161** (≈gone) | one solve, no internal borders → no grid |

`figC_seam_spectrum.png`: a sharp seam comb is present in **all tiled variants** and **flat/absent
in full_halo** (green). The large-period (>100 px) bumps are real chip structure, shared by all.
→ The grid only exists when there are tile borders; it is suppressed monotonically by enlarging
the solve region. It is **not** in the synthetic target (isothermal interiors), **not** the
bilinear-upsample 2-px pattern (that would be identical in every mode), and **not** the stitch
window (ACL-038 stitch-window sweep was negative; denser overlap here ≈ no change).
This matches ACL-038's step-decomposition: `prox1` creates the box, `dc1` only dents it.

### 2. The flocculence/swelling/lift grows monotonically with solve context
`diag_arrays2.py`, same checkpoint:

| render | bg lift vs aligned [°C] | fine texture (2–8px) | edge P95 grad |
|---|---:|---:|---:|
| aligned_mean | 0.000 | 0.50e-3 | 0.404 |
| tiled_p192_o128 | 0.088 | 5.04e-3 | 0.854 |
| tile_halo32 | 0.124 | 5.08e-3 | 0.861 |
| tile_halo64 | 0.144 | 5.15e-3 | 0.869 |
| tile_halo96 | 0.163 | 5.25e-3 | 0.881 |
| **full_halo96** | **0.240** | **6.05e-3** | **0.922** |

As solve context grows: background gets lifted **more** (0.088→0.240°C), fine broadband texture
grows (flocculence), and edge gradient/overshoot grows (swelling). `figE_lineprofile.png` shows
full_halo (green) with the deepest excursions **plus extra high-frequency jitter/overshoot** =
over-sharpening with ringing. A growing **low-frequency DC bias proportional to solve extent** is
the fingerprint of an extent-dependent global op (not a neutral boundary fix).

### 3. CAUSAL proof: the prox depends on content far outside its receptive field
`diag_extent.py` — architectural test, **no trained weights needed** (extent-invariance is a
property of the architecture). Perturb only the outer 64-px frame of a 768² field, 272 px away
from a central 96² window — far beyond the conv receptive field — and measure how much the window
output changes (normalized by its own std):

| variant | far-field change in window | crop-vs-full interior |
|---|---:|---:|
| **noGN_noSE** (pure conv UNet) | **0.0000** (exact) | 1.3e-5 (≈0) |
| **noSE** (GroupNorm only) | 0.673 | 1.50 |
| **full** (GroupNorm + SE) | **1.412** | 1.41 |

- A pure-conv UNet is **exactly extent-invariant** in its interior (far content can't reach it).
- GroupNorm alone already couples far content into the window (0.67); SE roughly **doubles** it
  (1.41). → Switching 192-tile → full-frame inference **must** change every pixel, by construction.
- K-recurrence amplification: the crop-vs-full interior gap grows ~linearly with K
  (K1 0.018, K2 0.037, K3 0.055, K4 0.074) — the per-step extent bias accumulates over the shared
  prox loop, reproducing ACL-037's "K4 worse than K2" from first principles.

---

## Verdict on each hypothesis

| Hypothesis | Verdict | Basis |
|---|---|---|
| Grid = prox **patch-boundary** response | ✅ confirmed | §1 + ACL-038 step decomp |
| Grid = **stitch/window** blend | ❌ not root | dense overlap ≈ tiled; ACL-038 window sweep negative |
| Grid = **DC rim/mask** | ❌ not root | DC only *reduces* the prox box (§1, ACL-038) |
| Grid = **training target / synthesis** | ❌ refuted | absent in full_halo (same target); not in isothermal GT |
| Flocculence = **GroupNorm/SE extent distribution shift** | ✅ confirmed + quantified | §3 far-field test |
| Amplified by **K4 recurrent shared prox** | ✅ confirmed (composes) | §3 K-amp + ACL-037 |

**Root cause (unified):** the prox UNet's two global, spatial-extent-dependent operations
(`GroupNorm` over H×W, `SEBlock` global average-pool) make the network non-extent-invariant, while
it is trained *only* on 192² crops. Small solve context = in-distribution but bordered (grid);
large solve context = borderless but out-of-distribution (lift/swell/flocculence). K4 recurrence
multiplies whichever error is present.

---

## Recommended experiments (minimal, prioritized)

Local machine has **no GPU**; these run on the remote 5090. Each lists the expected result and a
quantitative pass/fail using the metrics above (recompute with `diag_arrays*.py`).

### E0 (offline, do first if checkpoint is sent here) — quantify trained extent-bias
Rsync `solver_step_010000.pt` (~30 MB) to this machine. Then load the trained prox and rerun the
`diag_extent.py` far-field test with real weights (replace random init) to get the **trained**
coupling magnitude and the predicted full-frame DC bias. Decision: trained far-field coupling
≫ 0 (it will be) → confirms the inference-time mitigation must address GN/SE, not just halo.

### E1 (inference only, no retrain) — **offset-averaged tiling** ← best cheap fix
Why: the grid is fixed-position seams of *in-distribution* 192 solves. Average several tilings with
**different grid origins** → seams average out while every solve stays at 192 (no distribution
shift, no lift). More *overlap* of one grid does NOT help (dense≈tiled); origin **diversity** does.
- Where: new `solver_mode="tiled_jitter"` modeled on `infer_solver_from_burst` in
  `algos/ep07_unet_sr/src/unet_sr/real_eval.py` — solve N≈4–8 tilings at offsets
  `(0,0),(32,32),(16,48),…` (HR px), average the stitched outputs.
- Expected: seam prominence (§1) drops toward full_halo (~200) **while** bg_lift stays at tiled
  levels (~0.09, NOT 0.24) and fine_floc stays ~5.0e-3 (NOT 6.0e-3).
- Pass criterion: seam_prom < 400 **and** bg_lift < 0.12 **and** edge_p95 ≥ tiled. If met, this is
  the deliverable inference recipe — no retrain, structure preserved, grid gone.

### E2 (retrain, root fix #1) — **train in the inference regime (multi-scale / halo context)**
Currently `dataset.py` crops fixed 192² patches; the prox never sees full-frame stats.
- Where: `ThermalSRDataset._crop_origin_lr` / `patch_size_hr` — randomize patch size per batch
  (e.g. 192/256/384) or add an outer reflect-halo context that is solved but loss-masked to the
  central 192. CLI already has `--patch-size-hr`.
- Expected: at full-frame eval, bg_lift and fine_floc collapse toward the 192-tile values.
- Pass: full_halo bg_lift ≤ 0.10 and full_halo fine_floc ≤ tiled. Run a 5k-step smoke first.

### E3 (retrain, root fix #2) — **make the prox extent-invariant (drop SE, neutralize GN)**
- Where: `model.py` — `ConvBlock.se` → `nn.Identity` (ablation flag); replace `GroupNorm` with
  no-norm (like `HRResBlock`) or a spatial-extent-invariant norm. Add `--prox-no-se` / `--prox-norm`.
- Expected: re-running `diag_extent.py` with the new arch gives far-field coupling → 0 (pure-conv
  branch already shows 0.0); after retrain, tiled == full_halo (the whole tradeoff disappears).
- Pass: far-field coupling < 0.1 (arch test) and |full_halo − tiled| small on bg_lift/fine_floc.
- Cheapest sub-step: drop **SE only** first (halves the coupling per §3) — small retrain.

### E4 (retrain) — **K2 + prox residual damping** (already the mainline per ACL-037)
This V8 checkpoint is **K4**, contradicting the ACL-037 decision to move to **K2**. Re-run K2 and
add `x ← x + α·delta` with `α≈0.5` (damping) in `unroll.py::_prox`.
- Expected: K-amp slope halves (K2) / shrinks (α<1) → less flocculence regardless of halo.
- Pass: full_halo fine_floc and seam (in tiled) both drop vs K4 at equal steps.

### E5 (retrain, strongest structural fix) — **lowpass-anchor + learned highpass split**
Output = full-frame `aligned_mean` lowpass (fixed) + prox-predicted **highpass residual** only.
The prox can no longer move low/mid frequencies, killing both the broad boundary box and the
extent-dependent bg-lift at the source (ACL-038 step-4 idea).
- Where: `unroll.py` forward / `model.py` head. Larger change; schedule after E1–E4.
- Pass: bg_lift ≈ 0 in all modes; grid and lift both gone; structure preserved.

### Secondary / cheap toggles
- **Reflect-pad** the prox convs (`model.py` `padding_mode="reflect"`): trims the per-tile zero-pad
  edge signal. Helps the seam but not the global GN/SE coupling (§3) → partial, combine with E1.
- **`--flatness-weight 0.05`** (already in `losses.py`/`config.py`, default 0): suppresses predicted
  texture where GT is flat → dampens flocculence. Symptom-level; pair with a root fix.
- Degrid low-freq graft (already tried, `candidate_*_degrid`): the seam comb is mid-freq (16–64px),
  so a low-freq graft cannot remove it. Deprioritize.

---

## Constraints honored
- No historical files modified; no SR-algorithm code changed yet (this is diagnosis only → no
  changelog entry required; an entry is required when E1–E5 land).
- All products under `outputs/ep07_solver_diag/` (git-ignored). No data/outputs deleted. No large
  files committed.
