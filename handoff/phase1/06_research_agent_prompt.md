# Research Agent Prompt

Copy this prompt into a research agent. The expected output is a research report,
not code.

---

You are a research agent helping a software/physics team decide the next stage
for a LWIR micro-scanning super-resolution project.

## Project Context

We have a real industrial chip-inspection LWIR dataset:

- 255 usable TXT temperature frames in the main session.
- Each frame is a 480 x 640 Celsius temperature matrix.
- Detector sampling pitch is 10 um/pixel.
- Calibrated spatial resolution is 20 um.
- Waveband is LWIR 8-14 um.
- Noise floor is 0.0724 C.
- Stage coordinate set is `{0,2,4,6,8,10,12,14,16,18,20,24,28,32,36,40}` um
  on both axes.
- Stage-to-pixel rotation is theta=47.6 deg.
- Stage command is only a prior, not alignment truth.
- Main goal is contour/shape visibility for chip internal structure, not
  5 um absolute-temperature metrology.

Current project results:

- EP03 found that 2x contour-level SR is physically reasonable.
- EP03 found that 4x is high risk because MTF at 4x Nyquist is tiny under
  realistic PSF assumptions.
- EP04 produced alignment anchors and quality gates, but not final SR truth.
- EP05 showed 2x phase bins are fully covered and data-driven alignment improves
  held-out contour Chamfer.
- EP06 implemented classic 2x SR: SAA, SAA-weighted, IBP, MAP-TV.
- EP06 result: SAA denoises; IBP sharpens modestly; MAP-TV is sharpest but has
  highest artifact score.
- EP06 raw-temperature control track reproduces major structures after output
  highpass, reducing highpass-only hallucination risk.

We also have clearer optical images. They are useful for human hallucination
audit and geometry priors, but they are not currently registered thermal ground
truth and should not automatically be treated as thermal truth.

## Research Goals

Find what the broader literature and open-source ecosystem suggest for the next
episodes:

1. Strong validation protocol using real data + optical human audit.
2. Synthetic benchmark with true HR thermal ground truth and generated subpixel
   micro-shift LR bursts.
3. Candidate algorithms beyond classic SAA/IBP/MAP-TV.
4. Explain why some infrared papers report 4x/8x/16x and whether their claims are
   comparable to our LWIR industrial micro-scanning setting.
5. Identify the likely synthetic dataset/tool the user remembers: a code-driven
   thermal/infrared generator where one can generate arbitrary scenes/images.

## Required Search Scope

Search 2019-2026, but include older foundational methods where necessary.

Search these domains:

### Classic and physical multi-frame SR

- Multi-frame super-resolution for thermal/infrared imagery.
- Iterative back-projection, MAP, robust SR, total variation, TGV, Huber prior.
- Drizzle / variable-pixel linear reconstruction from astronomy.
- Lucky imaging and quality-gated frame/patch fusion.
- Blind/semi-blind PSF estimation and micro-scanning SR.

### Astronomy and remote sensing cross-over

- Drizzle, Hubble/deep field, undersampled dithered reconstruction.
- Remote sensing multi-frame SR and sensor MTF-aware SR.
- Any methods explicitly using coverage maps, uncertainty maps, or bad-pixel
  rejection.

### Mobile burst photography and video SR

- Deep Burst Super-Resolution.
- NTIRE burst SR challenge methods.
- BasicVSR, EDVR, recurrent/transformer video SR, implicit neural burst SR.
- Which ideas transfer to monochrome LWIR micro-scanning and which do not.

### Infrared-specific deep SR

- Single-image infrared SR.
- Burst/video/multi-frame infrared SR.
- Guided thermal SR using RGB/visible images.
- Thermal-physics-guided SR.
- Real-world infrared SR benchmarks.
- 4x/8x/16x papers: inspect whether the factor is true physics recovery,
  synthetic downsampling, guided generation, or display-grid upsampling.

### Self-supervised / zero-shot methods

- Deep Image Prior.
- DeepIR or thermal-specific self-supervised reconstruction.
- Zero-shot SR.
- Noise2Noise/Noise2Void/Self2Self style thermal denoising/SR.
- Internal learning and test-time optimization.
- Plug-and-play priors and RED.

### Industrial and micro-scanning setting

- Infrared micro-scanning for electronics inspection, PCB/chip thermal imaging,
  thermography SR, industrial part inspection.
- Algorithms that report actual spatial-resolution gains versus visual
  enhancement.
- Any papers using calibrated targets or optical references to audit SR outputs.

### Synthetic datasets and generators

Find code/datasets/tools for generating thermal/infrared imagery, especially:

- Physics-driven tools.
- Ray-tracing thermal renderers.
- Synthetic data generators that can produce arbitrary scenes.
- Tools that output temperature/radiance matrices, not just 8-bit PNGs.
- Datasets with HR thermal truth or paired LR/HR thermal.
- Datasets useful for validating 2x/4x/8x SR under known degradation.

Specifically investigate these leads:

- DIRSIG: Digital Imaging and Remote Sensing Image Generation.
- ThRend: ray-tracing-based LWIR thermography renderer.
- ThermalSynth.
- generatedTIR_tracking / TIRGen.
- DeepIR.
- Real-IISR.
- InfraredSR / ThesIS.
- ULB17-VT, VGTSR, PBVS GTISR / guided thermal SR datasets.
- MIRAGE RGB-TIR dataset.
- Any Blender/Unity/Unreal/AirSim/Isaac Sim thermal generation pipeline.

Try to identify the likely dataset/tool the user remembers as "we downloaded an
infrared dataset where images are generated by code; you can generate whatever
you want." Give confidence levels and why.

## Required Output

Produce a report with:

1. Executive recommendation for EP07 and EP08.
2. Table of candidate methods:
   - method name;
   - domain;
   - scale factors reported;
   - whether it is classical, self-supervised, supervised, or guided;
   - required inputs;
   - GitHub link if available;
   - paper link;
   - how comparable it is to our data;
   - hallucination risk;
   - implementation effort;
   - recommended priority.
3. Table of synthetic datasets/generators:
   - name;
   - type: physics renderer, style transfer, paired dataset, benchmark;
   - output format;
   - supports LWIR/radiance/temperature or only pseudo-thermal;
   - can generate arbitrary scenes;
   - license/access;
   - GitHub/data link;
   - suitability for chip-like micro-scan SR benchmark.
4. Explanation of why literature can report 4x/8x/16x:
   - distinguish synthetic downsampling, guided visible transfer, perceptual SR,
     and true multi-frame physical recovery.
5. Recommended validation design:
   - real data validation;
   - optical hallucination audit;
   - synthetic truth benchmark;
   - metrics for false edges and contour consistency.
6. Short list of 5-8 methods we should implement first.
7. Bibliography with URLs and dates.

## Ground Rules

- Prefer primary sources: papers, official project pages, official GitHub repos.
- Do not treat paper scale factor as physical resolution improvement unless the
  paper demonstrates it with real sensor/MTF/ground truth evidence.
- Clearly flag when a method is likely to hallucinate optical or learned priors.
- For each recommendation, explain what evidence would falsify it on our project.
- Be explicit about whether a dataset is LWIR thermal, NIR, RGB-to-thermal
  translation, or pseudo-thermal.

## Project-Specific Decision Bias

We value methods in this order:

1. Strong validation and low hallucination risk.
2. Compatibility with 255 real micro-shift frames.
3. Ability to use known shifts/quality gates/forward model.
4. Ability to produce uncertainty/coverage maps.
5. Only then visual sharpness.

The expected final deliverable should help us decide how to design EP07 and EP08.
