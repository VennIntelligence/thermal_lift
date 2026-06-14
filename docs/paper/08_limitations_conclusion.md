# 7. Limitations and Conclusion (draft)

## Limitations

- **Single instrument, single scene family.** All real-data evidence comes from one chip on one
  calibrated system (248-frame clean main session). The protocol and the null-space diagnosis are
  formulated generically, but their external validity is argued, not demonstrated; we position
  the paper accordingly (calibrated case study + reusable protocol, not a general benchmark).
- **No external high-resolution anchor.** Independent higher-resolution thermal validation
  (e.g., macro optics of the same scene) was not available; "trustworthy" therefore means
  consistency under internal gates (FRC controls, observation anchoring, structure metrics),
  not agreement with withheld truth.
- **FRC controls partially misbehave.** The positive/negative controls around the 10–12 µm
  rebound do not behave ideally (Sec. 5.1); we mitigated by claiming only the 17 µm cutoff and
  flagging the rebound as artifact risk. A cleaner phase-stratified control design is future work.
- **Proxy ceiling.** Artifact/corr proxies and split-half consistency gate against hallucination
  but cannot certify recovered detail; the adoption rule is conservative by design and may reject
  genuinely useful reconstructions.
- **PSF uncertainty.** The credible σ range [0.2, 0.5] LR px is arbitrated, not pinned; all
  PSF-dependent results either scan it or report σ. The ESF-vs-forward discrepancy (thermal edge
  width) is itself a finding but caps deconvolution aggressiveness.
- **Thinnest structures.** Lines at 1–2 px (10–20 µm) sit at the resolution limit; the deliverable
  for them is contour visibility and stability, not restoration — by physics, not by method choice.

## Conclusion

We studied burst super-resolution where it is hardest to defend: an industrial LWIR microscan
system with no high-resolution truth at any stage. On a fully measured forward model we showed
(i) the data carries real but bounded coherent information beyond the single-frame grid
(17 µm FRC cutoff); (ii) conventional 1x statistical featurization hides the burst's sub-pixel
phase information before learning begins, and injecting it as fine-grid drizzle evidence exposes a
different fine-structure trade-off without certifying fidelity by itself; (iii) synthetic-prior-trained networks drift along
the observation operator's null space — a failure mode invisible to forward-consistency losses of
any band or legal 1x-anchor variant, measurable with two curves, and manageable by evidence
injection, residual-over-observation parameterization, and Pareto-gated checkpoint selection rather
than stronger penalties. Classical anchored reconstruction
(anisotropic coverage-weighted TGV; fine-grid MAP-TV) sets the acceptance gate and is the most
observation-faithful arm, though it imprints a mild total-variation staircase on the finest
contours; learned arms are sharper by inspection but carry the most unverifiable high-frequency
content. In this no-GT regime we therefore certify no single winner: we report verifiable fidelity
proxies, a structural grain proxy with dual-domain visual gates, and an explicitly task-level
contour-legibility preference, and deliver only what survives the controls.

The bounded claim — contour-level 2x enhancement under explicit gates — is, we argue, the honest
and transferable formulation for no-GT industrial SR: measure the operator, bound the
information, deliver what survives the controls, and report what does not.

> TODO(§7): add one-sentence future work (multi-scene acquisition protocol exists as a design,
> external thermal anchor highest-value) and compress limitations to final page budget.
