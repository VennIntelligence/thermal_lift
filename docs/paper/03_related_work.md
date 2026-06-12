# 2. Related Work (draft, citations as [REF] placeholders)

## Classical multi-frame super-resolution

Registration-and-fusion MFSR dates to frequency-domain formulations [REF: Tsai & Huang 1984] and
spatial-domain iterative back-projection [REF: Irani & Peleg 1991]; robust and fast variants
[REF: Farsiu et al. 2004] established the shift-PSF-decimation observation model we adopt.
Drizzle [REF: Fruchter & Hook 2002] performs flux-conserving sub-pixel scattering onto a finer
grid and is our minimal-assumption fusion baseline. Regularized inversion with total variation
and total generalized variation [REF: Rudin et al. 1992; Bredies et al. 2010] remains the
strongest classical family in our setting; we contribute an anisotropic, coverage-weighted TGV
variant addressing raster-scan anisotropy (Sec. 4). MAP formulations with explicit PSF and
aperture integration [REF: Elad & Feuer 1997; Hardie et al. 1997] underpin our deconvolution
anchor. Our use of these methods differs in role: they are *anchors and acceptance gates* for
learned methods, not competitors to be dismissed.

## Deep burst super-resolution

Deep burst SR for RGB and RAW imagery includes DBSR [REF: Bhat et al. 2021], NTIRE burst-SR
challenges [REF], BIPNet [REF: Dudhane et al. 2022], and Burstormer [REF: Dudhane et al. 2023];
satellite multi-frame work includes HighRes-net and RAMS on PROBA-V [REF: Deudon et al. 2019;
Salvetti et al. 2020], a rare *measured* multi-frame benchmark albeit with imperfect HR truth.
These methods assume either paired HR supervision or proxy-paired data on the deployment domain.
Our setting removes that assumption entirely; we show the resulting failure mode (null-space
drift) and what survives it. Architecturally our reconstruction network is deliberately plain
(a UNet with a contour-oriented loss): architecture is not the subject of study, the
information path is.

## Thermal-image super-resolution

Thermal SISR/SR has been driven by PBVS thermal SR challenges [REF: Rivadeneira et al. 2020–2023]
and GAN-based thermal enhancement [REF: TherISuRNet etc.]. Most operate on rendered 8-bit imagery
with synthetic degradations and full-reference evaluation against held-out HR thermal images from
higher-end cameras. We differ in three ways: raw temperature matrices (not rendered video, which
we show is unusable as numeric SR input), a microscan burst with measured priors, and no HR truth
at any stage.

## Synthetic-to-real training and degradation modeling

Real-world SR work synthesizes degradations to bridge domain gaps [REF: KernelGAN, Real-ESRGAN,
BSRGAN]. Our synthetic platform is physics-matched rather than adversarially matched: scene
geometry rendered with sub-pixel coverage anti-aliasing, measured PSF range, detector box
integration, measured noise floor, and the actual raster shift distribution. The remaining
distribution gap is precisely what our drift analysis quantifies on real data — synthetic
matching reduces but does not eliminate prior-driven stylization.

## Evaluation without ground truth

Split-half consistency and Fourier Ring/Shell Correlation are standard in cryo-EM and
super-resolution microscopy [REF: van Heel & Schatz 2005; Nieuwenhuizen et al. 2013; Banterle
et al. 2013], yet rarely ported to learned SR. We adapt phase-stratified split-half FRC with
explicit positive/negative/drift controls, and report the controls honestly (including a
flagged high-frequency rebound attributed to coverage/lattice artifacts rather than optics).
No-reference IQA metrics [REF: NIQE etc.] are insufficient here because sharper-looking is the
failure mode, not the goal.

## Data consistency and null-space decompositions in inverse problems

Decomposing reconstructions into range and null-space components of the forward operator is
established in learned inverse problems [REF: Schwab, Antholzer & Haltmeier 2019 — null-space
networks; Chen et al. range-null decomposition; deep decoder / DIP as prior-only baselines], and
data-consistency layers are standard in MRI reconstruction [REF: Schlemper et al. 2018]. Our
contribution is empirical and diagnostic: on a measured industrial system we show that the
*training-time drift on real data lives almost entirely in the null space* — observation-side
losses sit at their floor while real-data artifact proxies degrade — and that the practical
remedies are input-side evidence injection and selection protocols rather than stronger
consistency penalties. We package this as a reusable diagnosis (forward-loss floor vs proxy
trajectory) for no-GT deployments.

> TODO(related): fill refs.bib; verify the null-space networks citation list; add FRC-for-SR
> precedents if any exist (check microscopy SR literature); decide whether EP12 4x negative
> result warrants citing 4x thermal SR claims literature for contrast.
