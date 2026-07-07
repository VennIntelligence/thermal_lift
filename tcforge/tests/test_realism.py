"""Tests for the realism augmentations (defects / isothermal temp / field noise)."""
import json
from pathlib import Path

import numpy as np

from tcforge import _noise_stats, physics, realism

_GOLDEN = Path(__file__).parent / "data" / "defects_golden_v1.npz"
_GOLDEN_FIELD = Path(__file__).parent / "data" / "field_noise_golden_v1.npz"


def _field_noise_burst_fixture() -> np.ndarray:
    """Exact burst construction mirrored from scratchpad/make_noise_goldens.py (golden pin)."""
    m, h, w = 8, 64, 80
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 20.0 + 0.01 * yy + 0.005 * xx
    return np.broadcast_to(base, (m, h, w)).astype(np.float32).copy()


def _solid_block_cov() -> np.ndarray:
    cov = np.zeros((220, 260), np.float32)
    cov[30:190, 30:230] = 1.0
    return cov


def test_apply_defects_only_removes_material():
    rng = np.random.default_rng(0)
    cov = np.zeros((220, 260), np.float32)
    cov[30:190, 30:230] = 1.0                      # one solid block
    defected, meta = realism.apply_defects(
        cov, rng, severity_range=(1.0, 1.0), max_holes=4, max_notches=4, max_cracks=3)
    assert defected.shape == cov.shape
    assert 0.0 <= float(defected.min()) and float(defected.max()) <= 1.0
    assert float(defected.sum()) < float(cov.sum())          # defects subtract material
    assert set(meta) >= {"holes", "notches", "cracks", "severity"}
    assert meta["holes"] + meta["notches"] + meta["cracks"] >= 1


def test_apply_defects_default_path_matches_golden():
    """Hard contract (ACL-063): with the new hole knobs at their defaults, apply_defects is
    BIT-IDENTICAL to the pre-change implementation — output array, meta dict AND the RNG
    stream (sentinel draw after the call). Golden fixture generated from HEAD e518703
    (pre-change realism.py) by scratchpad/make_defects_golden.py; construction mirrored here."""
    golden = np.load(_GOLDEN)

    # Case 1: fully default kwargs (seed chosen so holes/notches/cracks all fire).
    rng = np.random.default_rng(20260712)
    defected, meta = realism.apply_defects(_solid_block_cov(), rng)
    sentinel = rng.random()
    ref = golden["default_defected"]
    assert defected.dtype == ref.dtype
    assert defected.tobytes() == ref.tobytes()
    assert json.dumps(meta, sort_keys=True) == str(golden["default_meta_json"])
    assert sentinel == float(golden["default_sentinel"])

    # Case 2: explicit legacy kwargs (all pre-existing parameters, none of the new ones).
    rng = np.random.default_rng(42)
    defected, meta = realism.apply_defects(
        _solid_block_cov(), rng,
        severity_range=(1.0, 1.0),
        hole_radius_px=(4, 13), notch_radius_px=(3, 10),
        crack_len_px=(60, 260), crack_width_px=(2, 4),
        max_holes=8, max_notches=6, max_cracks=4,
    )
    sentinel = rng.random()
    ref = golden["legacy_defected"]
    assert defected.dtype == ref.dtype
    assert defected.tobytes() == ref.tobytes()
    assert json.dumps(meta, sort_keys=True) == str(golden["legacy_meta_json"])
    assert sentinel == float(golden["legacy_sentinel"])


def test_hole_depth_range_bounds_the_coverage_valley():
    rng = np.random.default_rng(11)
    cov = _solid_block_cov()
    d_lo, d_hi = 0.4, 0.7
    defected, meta = realism.apply_defects(
        cov, rng, severity_range=(1.0, 1.0), max_holes=10, min_holes=6,
        max_notches=0, max_cracks=0, hole_depth_range=(d_lo, d_hi))
    assert meta["holes"] >= 6
    removal = (cov - defected)[cov == 1.0]
    # Hard edges: every touched pixel carries the full per-hole depth in [d_lo, d_hi];
    # overlaps take the max depth, still <= d_hi. Valley floor within [1-d_hi, 1-d_lo].
    touched = removal[removal > 1e-6]
    assert touched.size > 0
    assert float(touched.min()) >= d_lo - 1e-5 and float(touched.max()) <= d_hi + 1e-5
    valley = float(defected[cov == 1.0].min())
    assert 1.0 - d_hi - 1e-5 <= valley <= 1.0 - d_lo + 1e-5
    # Meta: per-hole depth list matches count, deepest drawn depth explains the valley.
    assert len(meta["hole_depths"]) == meta["holes"]
    assert all(d_lo <= d <= d_hi for d in meta["hole_depths"])
    assert abs((1.0 - valley) - max(meta["hole_depths"])) < 1e-3
    assert len(meta["hole_radii"]) == meta["holes"]


def test_hole_edge_softness_bandwidth_and_hard_edge_control():
    # Deterministic circle (irregularity=0 => boundary == radius exactly): linear ramp of
    # total width `softness` centred on the boundary, 0.5 exactly ON the boundary.
    shape, r, s = (101, 101), 20.0, 4.0
    rng = np.random.default_rng(5)
    soft = realism.irregular_blob(shape, 50, 50, r, rng, irregularity=0.0, edge_softness_px=s)
    rng = np.random.default_rng(5)
    hard = realism.irregular_blob(shape, 50, 50, r, rng, irregularity=0.0)
    assert hard.dtype == bool and soft.dtype == np.float32
    row = soft[50, :]
    inter = (row > 0.0) & (row < 1.0)
    assert int(inter.sum()) == 6                     # |x-50| in {19,20,21} per side: band ~= s px
    assert row[50 - 20] == 0.5 and row[50 + 20] == 0.5   # exactly 0.5 ON the nominal boundary
    assert row[50 - 18] == 1.0 and row[50 + 22] == 0.0   # full depth / zero outside the band
    assert np.array_equal(soft >= 0.5, hard)         # softness does not move the 0.5-level set
    # Hard-edge control at apply_defects level: no intermediate removal values.
    rng = np.random.default_rng(13)
    cov = _solid_block_cov()
    defected_hard, meta_h = realism.apply_defects(
        cov, rng, severity_range=(1.0, 1.0), max_holes=10, min_holes=6,
        max_notches=0, max_cracks=0)
    assert meta_h["holes"] >= 6
    rem_h = cov - defected_hard
    assert np.all((rem_h < 1e-6) | (rem_h > 1.0 - 1e-6))
    # Soft path produces intermediate values.
    rng = np.random.default_rng(13)
    defected_soft, meta_s = realism.apply_defects(
        cov, rng, severity_range=(1.0, 1.0), max_holes=10, min_holes=6,
        max_notches=0, max_cracks=0, hole_edge_softness_px=2.0)
    rem_s = cov - defected_soft
    assert np.any((rem_s > 1e-6) & (rem_s < 1.0 - 1e-6))
    assert meta_s["hole_edge_softness_px"] == 2.0


def test_min_holes_floor_is_respected_across_seeds():
    cov = _solid_block_cov()
    for seed in range(10):
        rng = np.random.default_rng(seed)
        _, meta = realism.apply_defects(
            cov, rng, severity_range=(1.0, 1.0), max_holes=10, min_holes=4,
            max_notches=0, max_cracks=0)
        assert 4 <= meta["holes"] <= 10
        assert meta["min_holes"] == 4 and meta["min_holes_effective"] == 4
    # min_holes > ceiling: clamped to ceil (severity keeps sole control of the ceiling).
    rng = np.random.default_rng(3)
    _, meta = realism.apply_defects(
        cov, rng, severity_range=(1.0, 1.0), max_holes=6, min_holes=50,
        max_notches=0, max_cracks=0)
    assert meta["holes"] == 6
    assert meta["min_holes"] == 50 and meta["min_holes_effective"] == 6


def test_small_soft_holes_render_nonempty_and_not_full_depth():
    # r = 1-2 px dots with 1 px soft edges (A1-dots pilot regime): every scene must show
    # actual removal, with intermediate (partial-depth) pixels — not the all-or-nothing
    # hard-edge rendering.
    cov = _solid_block_cov()
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        defected, meta = realism.apply_defects(
            cov, rng, severity_range=(1.0, 1.0), hole_radius_px=(1, 2),
            max_holes=30, min_holes=12, max_notches=0, max_cracks=0,
            hole_edge_softness_px=1.0)
        assert meta["holes"] >= 12
        removal = cov - defected
        affected = removal > 1e-6
        assert int(affected.sum()) >= meta["holes"]          # dots do not vanish
        assert float(removal.max()) > 0.5                    # clearly visible dots
        intermediate = affected & (removal < 1.0 - 1e-6)
        assert int(intermediate.sum()) > 0                   # soft edges: partial depth exists
        assert int(intermediate.sum()) < int(affected.sum()) or float(removal.max()) < 1.0


def test_isothermal_is_uniform_within_a_connected_structure():
    rng = np.random.default_rng(1)
    mask = np.zeros((128, 128), np.float32)
    mask[30:90, 30:90] = 1.0                       # single connected block
    f = realism.render_isothermal_field(
        mask, rng, t_bg_c=21.0, delta_t_c=2.0, level_min=0.95, edge_sigma=0.0, low_freq_amplitude_c=0.0)
    assert float(f.min()) >= 21.0 - 1e-4           # never below background
    core = f[45:75, 45:75]                          # interior of the block
    assert float(core.std()) < 0.05                # ~isothermal within one structure


def test_field_noise_fpn_fixed_across_burst_grain_per_frame():
    rng = np.random.default_rng(2)
    burst = np.full((8, 64, 80), 20.0, np.float32)
    noisy = realism.field_noise_burst(burst, rng, vignette_c=0.2, stripe_c=0.05, grain_c=0.10)
    assert noisy.shape == burst.shape
    resid = noisy - burst
    fixed_est = resid.mean(axis=0)                  # vignette + stripe (fixed across frames)
    per_frame_std = resid.std(axis=0).mean()        # ~grain (fixed pattern cancels in std-over-frames)
    assert float(fixed_est.std()) > 0.05            # a real fixed pattern is present
    assert 0.04 < float(per_frame_std) < 0.20       # grain is per-frame (~grain_c)


def test_field_noise_accepts_single_frame():
    rng = np.random.default_rng(3)
    out = realism.field_noise_burst(np.full((48, 60), 19.0, np.float32), rng)
    assert out.shape == (48, 60)


def test_field_noise_default_path_matches_golden():
    """Hard contract (v7 noise upgrade): with the new noise knobs at their defaults (all OFF),
    field_noise_burst is BIT-IDENTICAL to the pre-change implementation — output array AND the
    RNG stream (sentinel draw after the call). Golden pinned from HEAD by
    scratchpad/make_noise_goldens.py; both the fully-default and explicit-legacy-kwargs cases."""
    golden = np.load(_GOLDEN_FIELD)

    # Case 1: fully default kwargs.
    rng = np.random.default_rng(20260708)
    out = realism.field_noise_burst(_field_noise_burst_fixture(), rng)
    sentinel = float(rng.random())
    ref = golden["default_out"]
    assert out.dtype == ref.dtype
    assert out.tobytes() == ref.tobytes()
    assert sentinel == float(golden["default_sentinel"])

    # Case 2: explicit legacy kwargs (all pre-existing parameters spelled out, no v7 knobs).
    rng = np.random.default_rng(77)
    out = realism.field_noise_burst(
        _field_noise_burst_fixture(), rng,
        vignette_c=0.13, stripe_c=0.028, stripe_col_sigma=(2.5, 5.0), grain_c=0.10)
    sentinel = float(rng.random())
    ref = golden["legacy_out"]
    assert out.dtype == ref.dtype
    assert out.tobytes() == ref.tobytes()
    assert sentinel == float(golden["legacy_sentinel"])


# ---------------------------------------------------------------------------------------------
# §5.2 per-item statistical acceptance of the four v7 noise upgrades (all measured with the
# §5.0 same-source estimators in tcforge._noise_stats, i.e. identically to the real-noise audit).
# ---------------------------------------------------------------------------------------------

def _flat_burst(m, h, w) -> np.ndarray:
    return np.zeros((m, h, w), np.float32)


def test_row_stripe_amplitude_and_corr_length():
    """(a) row-stripe FPN. The generator injects a UNIT-STD row profile scaled by row_stripe_c, so
    the RAW row-profile std == row_stripe_c essentially exactly (§1.2 'unit-std x coef'); the
    correlation length ~ 2*sigma_row. No column leakage.

    NOTE (deviation from the plan's literal wording): the §5.0 stripe_profiles estimator BG-subtracts
    a sigma=25 px Gaussian (matched to the real audit), which high-pass-attenuates the *measured*
    row amplitude to ~0.66-0.8x the injected value. That attenuation is identical for real and
    synthetic data, so the absolute °C calibration is closed by the §5.4 pilot loop; here we assert
    the generator's unit invariant on the RAW profile and only that the estimator sees the stripe."""
    amp = 0.049
    rng = np.random.default_rng(101)
    field = realism.field_noise_burst(
        _flat_burst(1, 200, 200), rng, vignette_c=0.0, stripe_c=0.0, grain_c=0.0,
        row_stripe_c=amp, stripe_row_sigma=(5.5, 5.5))[0]
    raw_row_std = float(field.mean(axis=1).std())          # RAW row profile (no BG subtraction)
    assert abs(raw_row_std - amp) <= 0.15 * amp
    sp = _noise_stats.stripe_profiles(field)               # §5.0 same-source estimator
    assert sp["row_amp"] > 5.0 * max(sp["col_amp"], 1e-9)  # row stripe present & dominant
    assert sp["col_amp"] < 0.1 * amp                        # no column leakage
    lens = []
    for s in range(40):
        f = realism.field_noise_burst(
            _flat_burst(1, 200, 200), np.random.default_rng(3000 + s),
            vignette_c=0.0, stripe_c=0.0, grain_c=0.0,
            row_stripe_c=amp, stripe_row_sigma=(5.5, 5.5))[0]
        L = _noise_stats.autocorr_1e_length(f.mean(axis=1), np.ones(f.shape[0], bool))
        if L is not None:
            lens.append(L)
    assert 8 <= int(np.median(lens)) <= 14                  # ~ 2*sigma_row = 11


def test_col_stripe_corr_length_formula():
    """Pin the column-stripe 2*sigma correlation-length mapping: sigma_col=3.5 => 1/e in [5,9] px."""
    lens = []
    for s in range(40):
        f = realism.field_noise_burst(
            _flat_burst(1, 200, 200), np.random.default_rng(3000 + s),
            vignette_c=0.0, stripe_c=0.065, grain_c=0.0, stripe_col_sigma=(3.5, 3.5))[0]
        L = _noise_stats.stripe_profiles(f)["col_1e_length_px"]
        if L is not None:
            lens.append(L)
    assert 5 <= int(np.median(lens)) <= 9


def test_powerlaw_field_slope():
    """(b) physics.powerlaw_field: radial PSD slope ~ target, high log-log R^2, unit std."""
    target = 1.77
    alphas, r2s, stds = [], [], []
    for s in range(8):
        f = physics.powerlaw_field((160, 160), target, np.random.default_rng(s))
        stds.append(float(f.std()))
        sl = _noise_stats.radial_psd_slope(f.astype(np.float64), 160)
        alphas.append(sl["alpha"])
        r2s.append(sl["r2_loglog_fit"])
    assert abs(float(np.mean(alphas)) - target) <= 0.15
    assert min(r2s) > 0.85
    assert abs(float(np.mean(stds)) - 1.0) < 0.05


def test_field_noise_composite_psd_slope():
    """(b) with the full v7 amplitude set on, the composite mean-image PSD slope stays in a
    detector-like band alpha in [1.5, 2.0] (1/f field + white pixel-FPN floor combined)."""
    rng = np.random.default_rng(7)
    o = realism.field_noise_burst(
        _flat_burst(64, 180, 180), rng,
        vignette_c=0.13, stripe_c=0.065, stripe_col_sigma=(2.5, 4.5), grain_c=0.10,
        row_stripe_c=0.049, stripe_row_sigma=(5.5, 5.5), lowfreq_c=0.07, lowfreq_alpha=(1.77, 1.77),
        pixel_fpn_c=0.07, grain_ar1_rho=0.63)
    crop = o.mean(axis=0)[10:170, 10:170].astype(np.float64)
    alpha = _noise_stats.radial_psd_slope(crop, 160)["alpha"]
    assert 1.5 <= alpha <= 2.0


def test_grain_ar1_lag1_and_fusion_gain():
    """(c) AR(1) grain: lag-1 autocorr == rho; per-frame marginal std == grain_c; multi-frame mean
    std == grain_c*sqrt((1+rho)/((1-rho)*M)) — the realistic (slowed) multi-frame fusion gain."""
    rho, gc, M = 0.63, 0.10, 248
    o = realism.field_noise_burst(
        _flat_burst(M, 64, 64), np.random.default_rng(3),
        vignette_c=0.0, stripe_c=0.0, grain_c=gc, grain_ar1_rho=rho)
    assert abs(_noise_stats.lag1_autocorr_median(o) - rho) <= 0.05
    per_frame_std = float(o.std(axis=(1, 2)).mean())
    assert abs(per_frame_std - gc) <= 0.10 * gc
    mean_std = float(o.mean(axis=0).std())
    pred = gc * np.sqrt((1.0 + rho) / ((1.0 - rho) * M))
    assert abs(mean_std - pred) <= 0.20 * pred


def test_grain_ar1_rho_zero_bit_identical():
    """(c) rho=0 leaves the grain (and thus the whole output + RNG stream) bit-identical to not
    passing the knob — the AR(1) recursion introduces no extra draw."""
    b = np.full((16, 48, 60), 20.0, np.float32)
    r1 = np.random.default_rng(9)
    o1 = realism.field_noise_burst(b, r1, grain_c=0.1)
    s1 = float(r1.random())
    r2 = np.random.default_rng(9)
    o2 = realism.field_noise_burst(b, r2, grain_c=0.1, grain_ar1_rho=0.0)
    s2 = float(r2.random())
    assert o1.tobytes() == o2.tobytes()
    assert s1 == s2


def test_pixel_fpn_static_across_burst():
    """(d) static per-pixel FPN: zero temporal std, spatial std == pixel_fpn_c, and it survives
    multi-frame averaging (mean-image std unchanged)."""
    pc, M = 0.07, 32
    o = realism.field_noise_burst(
        _flat_burst(M, 120, 120), np.random.default_rng(2),
        vignette_c=0.0, stripe_c=0.0, grain_c=0.0, pixel_fpn_c=pc)
    assert float(o.std(axis=0).mean()) < 1e-6                   # identical across frames
    assert abs(float(o.mean(axis=0).std()) - pc) <= 0.10 * pc   # spatial amplitude == pc
    assert abs(float(o[0].std()) - pc) <= 0.10 * pc             # single frame carries the full FPN


def test_burst_semantics_partition():
    """Variance decomposition with everything on: resid.mean(axis=0) recovers the fixed structure
    (vignette + col/row stripes + 1/f field + pixel FPN) well above the averaged-grain floor, and
    resid.std(axis=0) recovers the per-frame grain marginal std (== grain_c at rho=0)."""
    gc, M = 0.10, 64
    base = np.full((M, 96, 96), 20.0, np.float32)
    o = realism.field_noise_burst(
        base, np.random.default_rng(11),
        vignette_c=0.13, stripe_c=0.065, stripe_col_sigma=(2.5, 4.5), grain_c=gc,
        row_stripe_c=0.049, stripe_row_sigma=(5.5, 5.5), lowfreq_c=0.07, lowfreq_alpha=(1.77, 1.77),
        pixel_fpn_c=0.07, grain_ar1_rho=0.0)
    resid = o - base
    assert float(resid.mean(axis=0).std()) > 0.10           # fixed structure present
    per_frame_std = float(resid.std(axis=0).mean())
    assert abs(per_frame_std - gc) <= 0.12 * gc             # per-frame grain == grain_c
