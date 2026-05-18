# Search Leads and Sources

This file records initial leads found before creating the research-agent prompt.
It is not a complete literature review. A research agent should verify and expand
these leads.

## Local Sources

| Source | Why It Matters |
|---|---|
| `reports/ep03_theoretical_limits/theoretical_limits_report.md` | MTF/SNR/ESF/CRB boundary; 2x supported, 4x high-risk. |
| `reports/ep04_global_validation/validation_report.md` | Alignment anchors, quality gates, holdout roles. |
| `reports/ep05_sr_reassessment/displacement_reassessment.md` | 2x phase capacity and alignment baseline. |
| `reports/ep06_sr_poc/sr_poc_report.md` | Classic 2x SR results: SAA, IBP, MAP-TV. |
| `docs/dataset_description.md` | Dataset, scanning pattern, sessions, constants. |
| `algos/ep06_sr_poc/` | Current implemented algorithms and synthetic smoke tests. |

## Foundational / Cross-Domain Methods

### Drizzle

- URL: https://arxiv.org/abs/astro-ph/9808087
- Notes: Linear reconstruction from undersampled, dithered data. Preserves
  photometry and resolution, supports weighting and distortion handling.
- Relevance: Strong candidate to replace or augment SAA with coverage/weight maps.

### Thermal Multi-Image 4x SR

- URL: https://link.springer.com/article/10.1007/s12518-019-00253-y
- Notes: Thermal multi-image SR paper using many slightly shifted thermal images
  and upsampling factor 4.
- Relevance: Directly related to thermal MFSR, but must compare assumptions,
  scene type, sensor MTF, validation evidence, and whether 4x is physical or
  output-grid.

### Deep Burst Super-Resolution

- URL: https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.html
- Notes: RAW noisy burst to denoised SR RGB using flow-aligned deep embeddings and
  attention fusion.
- Relevance: Strong mobile-burst method family; transfer to LWIR requires caution.

### NTIRE 2021 Burst SR Challenge

- URL: https://arxiv.org/abs/2106.03839
- Notes: 4x RAW burst SR challenge with synthetic and real-world mobile tracks.
- Relevance: Source of high-performing burst SR ideas, but mostly RGB/mobile.

## Thermal/Infrared SR and Guided SR Leads

### DBFE Infrared SR

- URL: https://www.nature.com/articles/s41598-024-60238-9
- Notes: Infrared image SR framework with structure/textural encoding.
- Relevance: Good example of deep infrared SR, likely single-image and benchmark
  driven; check scale factors and hallucination risk.

### Guided Thermal SR 8x/16x

- URL: https://www.sciencedirect.com/science/article/pii/S0925231223013206
- Notes: Guided thermal SR using visible guidance; reports high upscale factors.
- Relevance: Important for explaining why papers can report 8x/16x. Not directly
  comparable to pure thermal multi-frame recovery.

### Thermal-Physics Guided Infrared SR / ThesIS

- URL: https://ojs.aaai.org/index.php/AAAI/article/view/38381
- Notes: 2026 infrared SR with thermal-physics guidance and dynamic
  high-frequency amplification.
- Relevance: Recent top-tier lead; needs careful review for dataset, code, and
  physical fidelity claims.

### Real-IISR

- URL: https://arxiv.org/abs/2603.04745
- URL: https://github.com/JZD151/Real-IISR
- Notes: 2026 real-world infrared SR benchmark/framework lead.
- Relevance: Strong candidate for modern real-world degradation modeling.

### SwinFuSR

- URL: https://arxiv.org/abs/2404.14533
- Notes: RGB-guided thermal image SR.
- Relevance: Guided method lead; useful but high optical-injection risk.

## Synthetic Thermal Dataset / Generator Leads

### DIRSIG

- URL: https://dirsig.org/docs/reveal.js/overview.html
- URL: https://dirsig.cis.rit.edu/docs/new/thermal.html
- Notes: Physics-driven EO/IR synthetic image generation model from RIT. Supports
  visible through thermal infrared and user-defined sensors/scenes.
- Relevance: Best current candidate for the user's remembered "generate whatever
  infrared image you want" tool if the prior dataset was physics-rendered.
- Caveat: Access/licensing may be restrictive; research agent must verify.

### ThRend

- URL: https://github.com/jpaguerre/ThRend
- Notes: Ray-tracing-based LWIR thermography renderer. Can output apparent
  temperature PNGs and a `temps` matrix.
- Relevance: Very relevant because it can output numeric temperature-like data.
- Caveat: Developed around urban scene thermography, not chip microscale.

### ThermalSynth

- URL: https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Madan_ThermalSynth_A_Novel_Approach_for_Generating_Synthetic_Thermal_Human_Scenarios_WACVW_2023_paper.html
- URL: https://github.com/NeeluMadan/Thermal-Synth
- Notes: Unity-based synthetic thermal human scenarios using thermal shader and
  real thermal backgrounds.
- Relevance: Good example of code-generated thermal data; likely less suitable
  for chip SR unless only used as a generator pattern.

### generatedTIR_tracking / TIRGen

- URL: https://github.com/zhanglichao/generatedTIR_tracking
- Notes: Synthetic data generation for end-to-end TIR tracking.
- Relevance: Possible match to remembered code-generated TIR dataset; likely
  image-translation/tracking oriented.

### DeepIR

- URL: https://github.com/vishwa91/DeepIR
- Notes: Deep InfraRed image processing framework; uses multiple images with
  small camera motion and models scene-dependent flux plus non-uniformity.
- Relevance: Very relevant self-supervised/optimization-style thermal lead.

### MIRAGE

- URL: https://github.com/donkeymouse/MIRAGE
- Notes: Large aligned RGB-TIR dataset for scalable multispectral translation.
- Relevance: Useful for guided/generative methods, not necessarily physical HR
  truth for micro-scanning.

### ULB17-VT / VGTSR / PBVS GTISR

- Notes: Guided thermal SR datasets used by RGB-guided thermal SR papers and
  challenges.
- Relevance: Useful for optical-guided thermal SR benchmarking; likely not
  micro-scan true-HR thermal.

## Questions the Research Agent Must Resolve

1. What was the likely remembered self-generating infrared dataset/tool?
2. Which tools can output radiometric/temperature matrices rather than 8-bit
   pseudo-thermal images?
3. Which methods actually use multi-frame thermal measurements rather than
   single-image learned priors?
4. Which 4x/8x/16x claims are physically comparable to our LWIR 20 um spatial
   resolution dataset?
5. Which modern self-supervised methods can be validated by held-out LR
   prediction rather than HR truth?
6. Which open-source implementations can realistically be integrated into this
   repo without a long rewrite?
