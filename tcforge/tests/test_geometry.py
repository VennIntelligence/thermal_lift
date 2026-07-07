from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.geometry as geometry

_GEOM_GOLDEN = Path(__file__).parent / "data" / "geometry_golden_v1.npz"

_V6_WEIGHTS = {  # pool_2x_v6_cpu.json verbatim
    "pga_grid": 0.28, "die_bga": 0.24, "multi_die": 0.18,
    "trace_bus": 0.14, "heat_spreader": 0.10, "generic": 0.06,
}

_GOLDEN_GEOM_KW = dict(
    rotation_deg_center=0.0, rotation_jitter_deg=0.0,
    canvas_shape=(240, 320), pixel_size_um=20.0, scale=2,
    antialias=True, ssaa_factor=2, inscribe_disc=True,
)


def test_geometry_defaults_remain_2x_and_exposes_4x_constants() -> None:
    assert geometry.DEFAULT_CANVAS_SHAPE == (960, 1280)
    assert geometry.DEFAULT_SCALE == 2
    assert geometry.CANVAS_SHAPE_4X == (1920, 2560)
    assert geometry.SCALE_4X == 4


def test_rectangle_mask_uses_um_units_and_uint8_binary_values() -> None:
    mask = geometry.make_rectangle(
        cx_um=100.0,
        cy_um=80.0,
        w_um=40.0,
        h_um=20.0,
        canvas_shape=(64, 80),
        pixel_size_um=10.0,
        scale=2,
    )
    assert mask.shape == (64, 80)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() > 0


def test_composite_and_rotate_preserve_binary_mask_contract() -> None:
    a = geometry.make_rectangle(100, 100, 40, 40, canvas_shape=(64, 64), pixel_size_um=10, scale=2)
    b = geometry.make_cross(150, 120, 20, 60, canvas_shape=(64, 64), pixel_size_um=10, scale=2)
    combined = geometry.composite(a, b, canvas_shape=(64, 64))
    rotated = geometry.rotate_mask(combined, 47.6)

    assert combined.dtype == np.uint8
    assert rotated.dtype == np.uint8
    assert set(np.unique(rotated).tolist()) <= {0, 1}
    assert rotated.shape == combined.shape
    assert rotated.sum() > 0


def test_curved_pad_primitives_are_binary_and_seedless_deterministic() -> None:
    circle = geometry.make_circle_pad(120, 100, 40, canvas_shape=(64, 80), pixel_size_um=10, scale=2)
    ellipse = geometry.make_ellipse_pad(
        120,
        100,
        60,
        30,
        angle_deg=35,
        canvas_shape=(64, 80),
        pixel_size_um=10,
        scale=2,
    )
    vias = geometry.make_via_array(
        2,
        3,
        35,
        16,
        120,
        100,
        stagger=True,
        canvas_shape=(64, 80),
        pixel_size_um=10,
        scale=2,
    )

    for mask in (circle, ellipse, vias):
        assert mask.dtype == np.uint8
        assert mask.shape == (64, 80)
        assert set(np.unique(mask).tolist()) <= {0, 1}
        assert mask.sum() > 0


def test_build_scene_mask_is_seed_reproducible_and_difficulty_sensitive() -> None:
    easy_a = geometry.build_scene_mask(
        "easy", 123, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2, antialias=False,
    )
    easy_b = geometry.build_scene_mask(
        "easy", 123, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2, antialias=False,
    )
    hard = geometry.build_scene_mask(
        "hard", 124, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2, antialias=False,
    )

    assert easy_a.shape == (96, 128)
    assert easy_a.dtype == np.uint8
    assert np.array_equal(easy_a, easy_b)
    assert not np.array_equal(easy_a, hard)


def test_build_scene_mask_defaults_to_soft_antialiased_coverage() -> None:
    mask_a, meta_a = geometry.build_scene_mask_with_metadata(
        "hard",
        123,
        canvas_shape=(96, 128),
        pixel_size_um=10.0,
        scale=2,
    )
    mask_b, _meta_b = geometry.build_scene_mask_with_metadata(
        "hard",
        123,
        canvas_shape=(96, 128),
        pixel_size_um=10.0,
        scale=2,
    )

    partial = (mask_a > 0.0) & (mask_a < 1.0)
    assert mask_a.shape == (96, 128)
    assert mask_a.dtype == np.float32
    assert 0.0 <= float(mask_a.min()) <= float(mask_a.max()) <= 1.0
    assert int(partial.sum()) > 0
    assert np.array_equal(mask_a, mask_b)
    assert meta_a["antialias"] is True
    assert meta_a["ssaa_factor"] == 4
    assert meta_a["mask_semantics"] == "coverage"


def test_downsample_coverage_keeps_diagonal_edge_monotonic_and_partial() -> None:
    super_mask = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[:32, :32]
    super_mask[xx >= yy] = 1.0

    coverage = geometry._downsample_coverage(super_mask, 4)
    profile = coverage[4]

    assert coverage.shape == (8, 8)
    assert coverage.dtype == np.float32
    assert np.any((coverage > 0.0) & (coverage < 1.0))
    assert np.all(np.diff(profile) >= -1e-6)


def test_build_scene_mask_with_metadata_accepts_explicit_4x_canvas() -> None:
    mask_a, meta_a = geometry.build_scene_mask_with_metadata(
        "medium",
        42,
        canvas_shape=(128, 160),
        pixel_size_um=10.0,
        scale=4,
        antialias=False,
    )
    mask_b, _meta_b = geometry.build_scene_mask_with_metadata(
        "medium",
        42,
        canvas_shape=(128, 160),
        pixel_size_um=10.0,
        scale=4,
        antialias=False,
    )

    assert mask_a.shape == (128, 160)
    assert mask_a.dtype == np.uint8
    assert set(np.unique(mask_a).tolist()) <= {0, 1}
    assert np.array_equal(mask_a, mask_b)
    assert meta_a["difficulty"] == "medium"
    primitive_types = {str(item["type"]) for item in meta_a["primitives"]}  # type: ignore[index]
    assert primitive_types & {"circle_pad", "ellipse_pad", "via_array"}


def test_multi_temp_mask_and_edge_diffusion_keep_shape_and_ranges() -> None:
    labels, meta = geometry.build_multi_temp_mask_with_metadata(
        "hard",
        9,
        n_temp_levels=4,
        canvas_shape=(96, 128),
        pixel_size_um=10.0,
        scale=4,
    )
    soft = geometry.apply_edge_diffusion(labels > 0, sigma_um=5.0, pixel_size_um=10.0, scale=4)

    assert labels.shape == (96, 128)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels).tolist()) <= {0, 1, 2, 3}
    assert int(meta["n_temp_levels"]) == 4
    assert soft.shape == labels.shape
    assert soft.dtype == np.float32
    assert 0.0 <= float(soft.min()) <= float(soft.max()) <= 1.0


def test_inscribe_disc_zeros_corners() -> None:
    mask, meta = geometry.build_scene_mask_with_metadata(
        "easy",
        seed=4242,
        rotation_deg_center=0.0,
        rotation_jitter_deg=0.0,
        canvas_shape=(240, 320),
        pixel_size_um=20.0,
        scale=2,
        antialias=True,
        ssaa_factor=4,
        inscribe_disc=True,
    )
    assert meta["inscribe_disc"] is True
    # All four corners lie outside the inscribed disc → must be zero.
    assert float(mask[0, 0]) == 0.0
    assert float(mask[0, -1]) == 0.0
    assert float(mask[-1, 0]) == 0.0
    assert float(mask[-1, -1]) == 0.0


def test_inscribe_disc_conserves_inscribed_mass_across_rotations() -> None:
    canvas = (240, 320)
    base, _ = geometry.build_scene_mask_with_metadata(
        "medium", seed=909, rotation_deg_center=0.0, rotation_jitter_deg=0.0,
        canvas_shape=canvas, pixel_size_um=20.0, scale=2,
        antialias=True, ssaa_factor=4, inscribe_disc=True,
    )
    h, w = base.shape
    yy, xx = np.mgrid[:h, :w]
    r = min(h, w) / 2.0
    disc = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= r ** 2
    base_mass = float(base[disc].sum())
    for ang in (0.0, 37.0, 90.0, 213.0, 359.0):
        rot, _ = geometry.build_scene_mask_with_metadata(
            "medium", seed=909, rotation_deg_center=float(ang), rotation_jitter_deg=0.0,
            canvas_shape=canvas, pixel_size_um=20.0, scale=2,
            antialias=True, ssaa_factor=4, inscribe_disc=True,
        )
        drift = abs(float(rot[disc].sum()) - base_mass) / (base_mass + 1e-9)
        assert drift < 0.05


def test_scene_mask_legacy_and_v6_paths_match_golden() -> None:
    """Hard byte-identity contract (v7 integration §1.2): adding the scene_composer /
    composer_params keywords must leave the legacy (no-motif) and v6-motif code paths
    BIT-IDENTICAL — mask bytes + full metadata JSON. Golden generated from pre-change
    HEAD by scratchpad/make_v7_goldens.py."""
    golden = np.load(_GEOM_GOLDEN, allow_pickle=False)

    legacy_mask, legacy_meta = geometry.build_scene_mask_with_metadata(
        "medium", 101, motif_weights=None, **_GOLDEN_GEOM_KW)
    assert str(legacy_mask.dtype) == str(golden["legacy_dtype"])
    assert legacy_mask.tobytes() == golden["legacy_mask"].tobytes()
    assert json.dumps(legacy_meta, sort_keys=True) == str(golden["legacy_meta_json"])

    for seed, tag in ((202303, "v6a"), (404505, "v6b")):
        mask, meta = geometry.build_scene_mask_with_metadata(
            "medium", seed, motif_weights=_V6_WEIGHTS, **_GOLDEN_GEOM_KW)
        assert str(mask.dtype) == str(golden[f"{tag}_dtype"])
        assert mask.tobytes() == golden[f"{tag}_mask"].tobytes()
        assert json.dumps(meta, sort_keys=True) == str(golden[f"{tag}_meta_json"])


def _v7_scene(seed, tier, *, rot=0.0, canvas=(480, 640), ssaa=4):
    return geometry.build_scene_mask_with_metadata(
        "medium", seed, rotation_deg_center=rot, rotation_jitter_deg=0.0,
        canvas_shape=canvas, pixel_size_um=20.0, scale=2,
        antialias=True, ssaa_factor=ssaa, inscribe_disc=True,
        scene_composer="panel_cluster_v7", composer_params={"force_tier": tier})


def test_panel_cluster_v7_contract() -> None:
    """Composer contract (v7 integration §6 test 13): seed reproducibility, occupancy
    tier bands (G3), trace width >= 28um floor, no >=0.9mm plain panels, XL roles in
    {lined,textured}, every void crossed by >= 1 trace, cluster inside the inscribe disc."""
    FLOOR = 28.0

    # seed reproducibility (mask bytes + metadata JSON)
    a_cov, a_meta = _v7_scene(4242, "high", rot=91.0)
    b_cov, b_meta = _v7_scene(4242, "high", rot=91.0)
    assert a_cov.tobytes() == b_cov.tobytes()
    assert json.dumps(a_meta, sort_keys=True) == json.dumps(b_meta, sort_keys=True)

    seeds = range(6000, 6016)
    occ = {"mid": [], "high": [], "xl": []}
    for seed in seeds:
        for tier in ("mid", "high", "xl"):
            cov, meta = _v7_scene(seed, tier, rot=float((seed * 7) % 360))
            occ[tier].append(float((cov > 0.5).mean()))
            assert meta["scene_tier"] == tier
            assert meta["scene_family"] == "panel_cluster_v7"
            # trace narrow-dim >= FLOOR
            for t in meta["traces"]:
                assert min(t["h_um"], t["w_um"]) >= FLOOR - 0.5
            # XL: every panel role in {lined, textured}
            if tier == "xl":
                assert all(p["role"] in ("lined", "textured") for p in meta["panels"])
            # cluster inside inscribe disc: 0.7x circumradius acceptance (v6 die rule)
            H_um = cov.shape[0] * 10.0
            W_um = cov.shape[1] * 10.0
            disc_r = min(W_um, H_um) / 2.0
            for p in meta["panels"]:
                reach = (np.hypot(p["cy_um"] - H_um / 2, p["cx_um"] - W_um / 2)
                         + 0.7 * np.hypot(p["h_um"], p["w_um"]) / 2)
                assert reach <= disc_r * 0.975

    med = {k: float(np.median(v)) for k, v in occ.items()}
    assert 0.05 <= med["mid"] <= 0.16, med
    assert med["high"] >= 0.16, med
    assert min(min(v) for v in occ.values()) >= 0.02, med
    # XL capability tier (occ>=0.40) is a full-frame property (the small canvas above
    # underestimates absolute occupancy); pooled max over 8 full-res XL scenes >= 0.40.
    xl_full = []
    for seed in range(2020001, 2020009):
        cov, _ = geometry.build_scene_mask_with_metadata(
            "medium", seed, rotation_deg_center=float((seed * 7) % 360),
            rotation_jitter_deg=0.0, canvas_shape=(960, 1280), pixel_size_um=20.0,
            scale=2, antialias=True, ssaa_factor=4, inscribe_disc=True,
            scene_composer="panel_cluster_v7", composer_params={"force_tier": "xl"})
        xl_full.append(float((cov > 0.5).mean()))
    assert max(xl_full) >= 0.40, xl_full

    # no >=0.9mm plain panel: every large panel carries carved dark structure
    # (checked unrotated so bboxes align with the coverage grid)
    for seed in (6100, 6101, 6102, 6103):
        for tier in ("mid", "high", "xl"):
            cov, meta = _v7_scene(seed, tier, rot=0.0)
            H, W = cov.shape
            for p in meta["panels"]:
                if min(p["h_um"], p["w_um"]) >= 900.0:
                    y0 = max(int((p["cy_um"] - p["h_um"] / 2) / 10.0), 0)
                    y1 = min(int((p["cy_um"] + p["h_um"] / 2) / 10.0), H)
                    x0 = max(int((p["cx_um"] - p["w_um"] / 2) / 10.0), 0)
                    x1 = min(int((p["cx_um"] + p["w_um"] / 2) / 10.0), W)
                    sub = cov[y0:y1, x0:x1]
                    if sub.size:
                        assert float((sub > 0.5).mean()) < 0.995, (seed, tier, p["role"])

    # every void carries >= 1 crossing trace (owner r3 verdict, no empty voids):
    # direct invariant on the private carver — a windowed panel void always records
    # at least one void_span trace with a >= FLOOR narrow dimension.
    import tcforge.composer_v7 as C
    rng = np.random.default_rng(0)
    comp = C._PanelClusterComposer(rng, {}, (1920, 2560), 12800.0, 9600.0, 20.0)
    before = len(comp.traces)
    comp._void_with_traces(4800.0, 6400.0, 3000.0, 4000.0, 60.0, island=True, w_floor=FLOOR)
    spans = [t for t in comp.traces[before:] if t["kind"] == "void_span"]
    assert len(spans) >= 1
    assert all(min(t["h_um"], t["w_um"]) >= FLOOR - 0.5 for t in spans)


def test_scene_composer_default_none_is_inert() -> None:
    """Explicitly passing scene_composer=None (and composer_params=None) is byte-for-byte
    identical to not passing them — the new keywords default to legacy behaviour with zero
    extra RNG draws."""
    base_mask, base_meta = geometry.build_scene_mask_with_metadata(
        "medium", 202303, motif_weights=_V6_WEIGHTS, **_GOLDEN_GEOM_KW)
    explicit_mask, explicit_meta = geometry.build_scene_mask_with_metadata(
        "medium", 202303, motif_weights=_V6_WEIGHTS,
        scene_composer=None, composer_params=None, **_GOLDEN_GEOM_KW)
    assert base_mask.tobytes() == explicit_mask.tobytes()
    assert json.dumps(base_meta, sort_keys=True) == json.dumps(explicit_meta, sort_keys=True)


def test_inscribe_disc_default_false_unchanged() -> None:
    kwargs = dict(
        difficulty="hard", seed=2024, rotation_deg_center=33.0, rotation_jitter_deg=0.0,
        canvas_shape=(240, 320), pixel_size_um=20.0, scale=2,
        antialias=True, ssaa_factor=4,
    )
    default_mask, default_meta = geometry.build_scene_mask_with_metadata(**kwargs)
    explicit_false, _ = geometry.build_scene_mask_with_metadata(**kwargs, inscribe_disc=False)
    assert np.array_equal(default_mask, explicit_false)
    assert default_meta["inscribe_disc"] is False
