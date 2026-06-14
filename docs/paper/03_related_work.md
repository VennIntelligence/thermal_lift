# 2. Related Work (draft, citations as natbib \citep/\citet keys)

## Classical multi-frame super-resolution

Registration-and-fusion MFSR dates to frequency-domain formulations \citep{tsai1984multiframe} and
spatial-domain iterative back-projection \citep{irani1991improving}; robust and fast variants
\citep{farsiu2004fast} established the shift-PSF-decimation observation model we adopt.
Drizzle \citep{fruchter2002drizzle} performs flux-conserving sub-pixel scattering onto a finer
grid and is our minimal-assumption fusion baseline. Regularized inversion with total variation
and total generalized variation \citep{rudin1992nonlinear,bredies2010tgv} remains the
strongest classical family in our setting; we contribute an anisotropic, coverage-weighted TGV
variant addressing raster-scan anisotropy (Sec. 4). MAP formulations with explicit PSF and
aperture integration \citep{elad1997restoration,hardie1997joint} underpin our deconvolution
anchor. Our use of these methods differs in role: they are *anchors and acceptance gates* for
learned methods, not competitors to be dismissed.

## Deep burst super-resolution

Deep burst SR for RGB and RAW imagery includes DBSR \citep{bhat2021deep}, NTIRE burst-SR
challenges \citep{bhat2021ntire}, BIPNet \citep{dudhane2022burst}, and Burstormer \citep{dudhane2023burstormer};
satellite multi-frame work includes HighRes-net and RAMS on PROBA-V \citep{deudon2020highres,salvetti2020rams},
a rare *measured* multi-frame benchmark albeit with imperfect HR truth.
These methods assume either paired HR supervision or proxy-paired data on the deployment domain.
Our setting removes that assumption entirely; we show the resulting failure mode (null-space
drift) and what survives it. Architecturally our reconstruction network is deliberately plain
(a UNet with a contour-oriented loss): architecture is not the subject of study, the
information path is.

## Thermal-image super-resolution

Thermal SISR/SR has been driven by PBVS thermal SR challenges \citep{rivadeneira2020thermal,rivadeneira2023thermal}
and dedicated CNN/GAN-based thermal SR networks \citep{chudasama2020therisurnet,rivadeneira2020novel}. Most operate on rendered 8-bit imagery
with synthetic degradations and full-reference evaluation against held-out HR thermal images from
higher-end cameras. We differ in three ways: raw temperature matrices (not rendered video, which
we show is unusable as numeric SR input), a microscan burst with measured priors, and no HR truth
at any stage.

## Synthetic-to-real training and degradation modeling

Real-world SR work synthesizes degradations to bridge domain gaps \citep{bellkligler2019blind,wang2021real,zhang2021designing}. Our synthetic platform is physics-matched rather than adversarially matched: scene
geometry rendered with sub-pixel coverage anti-aliasing, measured PSF range, detector box
integration, measured noise floor, and the actual raster shift distribution. The remaining
distribution gap is precisely what our drift analysis quantifies on real data — synthetic
matching reduces but does not eliminate prior-driven stylization.

## Evaluation without ground truth

Split-half consistency and Fourier Ring/Shell Correlation are standard in cryo-EM and
super-resolution microscopy \citep{vanheel2005fourier,nieuwenhuizen2013measuring,banterle2013fourier}, yet rarely ported to learned SR. We adapt phase-stratified split-half FRC with
explicit positive/negative/drift controls, and report the controls honestly (including a
flagged high-frequency rebound attributed to coverage/lattice artifacts rather than optics).
No-reference IQA metrics \citep{mittal2013making} are insufficient here because sharper-looking is the
failure mode, not the goal.

## Data consistency and null-space decompositions in inverse problems

Decomposing reconstructions into range and null-space components of the forward operator is
established in learned inverse problems \citep{schwab2019deep,chen2021equivariant,ulyanov2018deep,heckel2019deep}, and
data-consistency layers are standard in MRI reconstruction \citep{schlemper2018deep}. Our
contribution is empirical and diagnostic: on a measured industrial system we show that the
*training-time drift on real data lives almost entirely in the null space* — observation-side
losses sit at their floor while real-data artifact proxies degrade — and that the practical
remedies are input-side evidence injection and selection protocols rather than stronger
consistency penalties. We package this as a reusable diagnosis (forward-loss floor vs proxy
trajectory) for no-GT deployments.

> TODO(related): fill refs.bib; verify the null-space networks citation list; add FRC-for-SR
> precedents if any exist (check microscopy SR literature); decide whether EP12 4x negative
> result warrants citing 4x thermal SR claims literature for contrast.
