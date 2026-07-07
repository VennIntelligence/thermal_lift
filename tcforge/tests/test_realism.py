"""Tests for the realism augmentations (defects / isothermal temp / field noise)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tcforge import _noise_stats, physics, realism

_GOLDEN = Path(__file__).parent / "data" / "defects_golden_v1.npz"
_ISO_GOLDEN = Path(__file__).parent / "data" / "isothermal_golden_v1.npz"
_GOLDEN_FIELD = Path(__file__).parent / "data" / "field_noise_golden_v1.npz"


def _iso_golden_mask() -> np.ndarray:
    mask = np.zeros((160, 200), np.float32)
    mask[20:70, 30:90] = 1.0
    mask[90:140, 110:170] = 1.0
    mask[95:110, 40:60] = 1.0
    return mask


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


def test_isothermal_default_path_matches_golden():
    """Byte-identity contract (v7 integration §3.6): the new zones/zone_rotation/
    zone_level_jitter keywords must leave the default (zones=None) path of
    render_isothermal_field BIT-IDENTICAL — output array + trailing RNG sentinel.
    Golden generated from pre-change HEAD by scratchpad/make_v7_goldens.py."""
    golden = np.load(_ISO_GOLDEN, allow_pickle=False)
    rng = np.random.default_rng(777)
    field = realism.render_isothermal_field(
        _iso_golden_mask(), rng, t_bg_c=21.0, delta_t_c=2.6, level_min=0.82,
        edge_sigma=1.4, low_freq_amplitude_c=0.1, low_freq_sigma_px=96.0)
    sentinel = float(rng.random())
    ref = golden["field"]
    assert str(field.dtype) == str(golden["dtype"])
    assert field.shape == tuple(int(v) for v in golden["shape"])
    assert field.tobytes() == ref.tobytes()
    assert sentinel == float(golden["sentinel"])


def test_isothermal_zones_none_matches_no_zones_kwarg():
    """Explicitly passing zones=None equals omitting it (inert new keyword)."""
    rng_a = np.random.default_rng(31)
    a = realism.render_isothermal_field(_iso_golden_mask(), rng_a, level_min=0.7)
    sa = rng_a.random()
    rng_b = np.random.default_rng(31)
    b = realism.render_isothermal_field(
        _iso_golden_mask(), rng_b, level_min=0.7,
        zones=None, zone_rotation_deg=17.0, zone_level_jitter=0.05)
    sb = rng_b.random()
    assert a.tobytes() == b.tobytes()
    assert sa == sb


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


# ── v7 integration: RNG-invariance + new-capability tests (§6 tests 4-11) ──────

def test_record_instances_consumes_no_rng():
    """§6 test 4: record_instances/label_map must add ZERO RNG draws — output array,
    the trailing sentinel AND (bar the new instances/next_id keys) the meta dict are
    identical on/off."""
    cov = _solid_block_cov()
    rng_a = np.random.default_rng(2024)
    da, ma = realism.apply_defects(cov, rng_a, severity_range=(1.0, 1.0),
                                   max_holes=8, max_notches=4, max_cracks=3)
    sa = rng_a.random()
    rng_b = np.random.default_rng(2024)
    lm = np.zeros(cov.shape, np.int16)
    db, mb = realism.apply_defects(cov, rng_b, severity_range=(1.0, 1.0),
                                   max_holes=8, max_notches=4, max_cracks=3,
                                   record_instances=True, label_map=lm, next_id=1)
    sb = rng_b.random()
    assert da.tobytes() == db.tobytes()
    assert sa == sb
    # meta identical apart from the record-only keys
    stripped = {k: v for k, v in mb.items() if k not in ("instances", "next_id")}
    assert json.dumps(ma, sort_keys=True) == json.dumps(stripped, sort_keys=True)
    assert mb["holes"] + mb["notches"] + mb["cracks"] == len(mb["instances"])


def test_hole_margin_default_matches_golden():
    """§6 test 5: explicit hole_margin_px=8 == the legacy _disk(8) default (byte-identical)."""
    cov = _solid_block_cov()
    rng_a = np.random.default_rng(20260712)
    a, ma = realism.apply_defects(cov, rng_a)
    rng_b = np.random.default_rng(20260712)
    b, mb = realism.apply_defects(cov, rng_b, hole_margin_px=8)
    assert a.tobytes() == b.tobytes()
    assert json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)


def _thin_line_cov():
    cov = np.zeros((120, 400), np.float32)
    cov[58:61, 20:380] = 1.0            # a 3px-wide horizontal line: erosion by _disk(8) => empty
    return cov


def test_hole_margin_adaptive_fills_thin_line_scenes():
    """§6 test 6: on a 3px line the default _disk(8) interior is empty (silent punch-through);
    adaptive shrinks the margin so min_holes is met and records the effective margin, while the
    non-adaptive path records holes_shortfall."""
    cov = _thin_line_cov()
    # default (non-adaptive): interior empty => 0 holes, shortfall recorded
    rng = np.random.default_rng(7)
    d0, m0 = realism.apply_defects(cov, rng, severity_range=(1.0, 1.0), min_holes=10,
                                   max_holes=20, max_notches=0, max_cracks=0,
                                   hole_radius_px=(1, 2), hole_edge_softness_px=1.0)
    assert m0["holes"] == 0
    assert m0["holes_shortfall"] >= 1
    assert "hole_margin_effective_px" not in m0
    # adaptive: interior filled via the ladder => holes placed, effective margin recorded
    rng = np.random.default_rng(7)
    d1, m1 = realism.apply_defects(cov, rng, severity_range=(1.0, 1.0), min_holes=10,
                                   max_holes=20, max_notches=0, max_cracks=0,
                                   hole_radius_px=(1, 2), hole_edge_softness_px=1.0,
                                   hole_margin_adaptive=True)
    assert m1["holes"] >= 10
    assert m1["hole_margin_effective_px"] < 8
    assert float((cov - d1).sum()) > 0.0


def test_hole_edge_fraction_places_boundary_band_holes():
    """§6 test 7: hole_edge_fraction>0 places a share of holes in the boundary band, tagged
    context='edge'; with fraction=0 no hole is context='edge'."""
    cov = _solid_block_cov()
    rng = np.random.default_rng(9)
    _, m = realism.apply_defects(cov, rng, severity_range=(1.0, 1.0), min_holes=40,
                                 max_holes=60, max_notches=0, max_cracks=0,
                                 hole_radius_px=(1, 3), hole_edge_softness_px=1.0,
                                 hole_edge_fraction=0.5, record_instances=True,
                                 label_map=np.zeros(cov.shape, np.int16))
    holes = [i for i in m["instances"] if i["type"] == "hole"]
    edge = [i for i in holes if i["context"] == "edge"]
    assert len(edge) >= 1                      # boundary-band holes exist
    assert 0.15 <= len(edge) / max(len(holes), 1) <= 0.85
    # edge holes sit near the struct boundary (within the erosion margin)
    struct = cov > 0.5
    interior = ndimage.binary_erosion(struct, realism._disk(8))
    for i in edge:
        y, x = i["center_yx_hr"]
        assert struct[y, x] and not interior[y, x]
    # fraction=0 => no edge holes
    rng = np.random.default_rng(9)
    _, m0 = realism.apply_defects(cov, rng, severity_range=(1.0, 1.0), min_holes=40,
                                  max_holes=60, max_notches=0, max_cracks=0,
                                  hole_radius_px=(1, 3), hole_edge_softness_px=1.0,
                                  record_instances=True, label_map=np.zeros(cov.shape, np.int16))
    assert all(i["context"] != "edge" for i in m0["instances"] if i["type"] == "hole")


def test_carve_trace_breaks_rotation_mapping():
    """§6 test 8: calibrate the ndimage.rotate forward-point map, then verify a gap carved from
    the pre-rotation trace table lands ON the rotated trace (main + secondary compound angle)."""
    # (a) forward map matches an actual ndimage.rotate of a delta image, all signs
    img = np.zeros((121, 121), np.float32)
    img[40, 48] = 1.0
    cy0, cx0 = (121 - 1) / 2, (121 - 1) / 2
    for ang in (15.0, -15.0, 47.0, -47.0, 90.0, -90.0, 200.0, -123.0):
        rot = ndimage.rotate(img, ang, reshape=False, order=1, mode="constant")
        yy, xx = np.unravel_index(int(np.argmax(rot)), rot.shape)
        fy, fx = realism._ndimage_rotate_point_forward(40, 48, cy0, cx0, ang)
        assert max(abs(yy - fy), abs(xx - fx)) < 1.2, ang

    hr = 10.0
    H, W = 200, 260
    # a horizontal trace centred at (1000,1300) um => (100,130) HR px, 40um tall x 1600um long
    cy_um, cx_um, h_um, w_um = 1000.0, 1300.0, 40.0, 1600.0
    pre = np.zeros((H, W), np.float32)
    y0 = int((cy_um - h_um / 2) / hr); y1 = int((cy_um + h_um / 2) / hr)
    x0 = int((cx_um - w_um / 2) / hr); x1 = int((cx_um + w_um / 2) / hr)
    pre[y0:y1, x0:x1] = 1.0
    center = ((H - 1) / 2.0, (W - 1) / 2.0)
    for ang in (0.0, 33.0, -50.0):
        rot_cov = realism._ndimage_rotate_point_forward  # noqa (keeps import used)
        rotated = np.clip(ndimage.rotate(pre, ang, reshape=False, order=1, mode="constant"), 0, 1)
        trace = {"cy_um": cy_um, "cx_um": cx_um, "h_um": h_um, "w_um": w_um,
                 "angle_deg": 0.0, "kind": "bridge"}
        rng = np.random.default_rng(3)
        carved, inst, nid = realism.carve_trace_breaks(
            rotated, [trace], rng, scene_rotation_deg=ang, canvas_center_yx=center,
            hr_pitch_um=hr, break_p=1.0, count_range=(1, 1), gap_px=(12.0, 14.0),
            record_instances=True, label_map=np.zeros((H, W), np.int16))
        removed = (rotated > 0.5) & (carved <= 0.5)
        assert int(removed.sum()) >= 8, (ang, int(removed.sum()))     # a real cut in the trace
        assert len(inst) == 1 and inst[0]["type"] == "broken_trace"

    # secondary compound angle: trace pre-rotated by angle_deg then scene-rotated
    ang_part, ang_scene = 40.0, 25.0
    # build the actual oriented trace in the pre-scene frame by rasterising an oriented rect
    d_part = realism._ndimage_rotate_vec_forward(0.0, 1.0, ang_part)
    reg = realism._oriented_rect_mask((H, W), cy_um / hr, cx_um / hr, d_part,
                                      (w_um / hr) / 2.0, (h_um / hr) / 2.0)
    pre2 = reg.astype(np.float32)
    rotated2 = np.clip(ndimage.rotate(pre2, ang_scene, reshape=False, order=1, mode="constant"), 0, 1)
    trace2 = {"cy_um": cy_um, "cx_um": cx_um, "h_um": h_um, "w_um": w_um,
              "angle_deg": ang_part, "kind": "bus"}
    rng = np.random.default_rng(5)
    carved2, inst2, _ = realism.carve_trace_breaks(
        rotated2, [trace2], rng, scene_rotation_deg=ang_scene, canvas_center_yx=center,
        hr_pitch_um=hr, break_p=1.0, count_range=(1, 1), gap_px=(12.0, 14.0))
    removed2 = (rotated2 > 0.5) & (carved2 <= 0.5)
    assert int(removed2.sum()) >= 8, int(removed2.sum())


def test_apply_thermal_defects_signs_and_bounds():
    """§6 test 9: hot spots raise T; dark blobs lower T but never below T_bg; instances tagged."""
    cov = np.zeros((200, 240), np.float32)
    cov[40:160, 50:190] = 1.0
    tbg, dt = 20.0, 3.0
    field = realism.render_isothermal_field(cov, np.random.default_rng(1), t_bg_c=tbg,
                                             delta_t_c=dt, level_min=0.6, edge_sigma=0.6)
    lm = np.zeros(cov.shape, np.int16)
    out, inst, nid = realism.apply_thermal_defects(
        field, cov, np.random.default_rng(4), t_bg_c=tbg, delta_t_c=dt,
        hot_spot_count=(4, 6), dark_blob_p=1.0, dark_blob_count=(2, 2),
        record_instances=True, label_map=lm, next_id=1)
    assert float(out.min()) >= tbg - 1e-3                    # dark blobs never below background
    assert float(out.max()) > float(field.max()) - 1e-3     # hot spots add heat
    kinds = [i["type"] for i in inst]
    assert kinds.count("dark_blob") == 2
    assert 4 <= kinds.count("hot_spot") <= 6
    assert nid == 1 + len(inst)
    assert set(np.unique(lm)) - {0}                          # labels painted


def test_defect_instance_schema_and_label_map():
    """§6 test 10: id density, label subset of instances, centres in-support, counts consistent."""
    cov = _solid_block_cov()
    lm = np.zeros(cov.shape, np.int16)
    rng = np.random.default_rng(123)
    d, m = realism.apply_defects(cov, rng, severity_range=(1.0, 1.0), min_holes=8, max_holes=16,
                                 max_notches=4, max_cracks=3, hole_radius_px=(2, 5),
                                 hole_depth_range=(0.3, 1.0), hole_edge_softness_px=1.0,
                                 record_instances=True, label_map=lm, next_id=1)
    inst = m["instances"]
    ids = [i["id"] for i in inst]
    assert ids == list(range(1, len(inst) + 1))              # dense, 1-based
    assert m["next_id"] == len(inst) + 1
    assert m["holes"] + m["notches"] + m["cracks"] == len(inst)
    label_ids = set(int(v) for v in np.unique(lm)) - {0}
    assert label_ids.issubset(set(ids))                      # every label is a real instance
    for i in inst:
        y, x = i["center_yx_hr"]
        assert 0 <= y < cov.shape[0] and 0 <= x < cov.shape[1]
        assert set(i.keys()) == {"id", "type", "stage", "center_yx_hr", "radius_px",
                                 "depth_or_amplitude", "edge_softness_px", "length_px",
                                 "width_px", "gap_px", "context", "trace_index", "area_px"}


def test_min_notches_floor_and_default_inert():
    """min_notches enforces a per-scene edge-notch floor (G8); default 0 is byte-identical."""
    cov = _solid_block_cov()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        _, m = realism.apply_defects(cov, rng, severity_range=(1.0, 1.0), max_holes=0,
                                     max_notches=6, max_cracks=0, min_notches=2)
        assert m["notches"] >= 2
    # default (min_notches=0) unchanged vs omitting
    a, ma = realism.apply_defects(cov, np.random.default_rng(3), severity_range=(1.0, 1.0))
    b, mb = realism.apply_defects(cov, np.random.default_rng(3), severity_range=(1.0, 1.0),
                                  min_notches=0)
    assert a.tobytes() == b.tobytes() and json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)


def test_stratified_anchor_widens_spread_and_default_inert():
    """stratified_anchor pins one big component dark + one bright (G7); default off is inert."""
    cov = np.zeros((200, 200), np.float32)
    cov[10:60, 10:60] = 1.0
    cov[10:60, 130:180] = 1.0
    cov[130:180, 10:60] = 1.0
    cov[130:180, 130:180] = 1.0     # four big components
    # anchored: spread spans nearly the full [level_min, 1] range
    f = realism.render_isothermal_field(cov, np.random.default_rng(1), t_bg_c=0.0, delta_t_c=1.0,
                                        level_min=0.6, edge_sigma=0.0, stratified_anchor=True)
    lbl, n = ndimage.label(cov >= 0.5)
    means = ndimage.mean(f, lbl, index=np.arange(1, n + 1))
    assert float(np.max(means) - np.min(means)) >= 0.30
    # default off == omitting (byte-identical + sentinel)
    ra = np.random.default_rng(2)
    a = realism.render_isothermal_field(cov, ra, level_min=0.6)
    sa = ra.random()
    rb = np.random.default_rng(2)
    b = realism.render_isothermal_field(cov, rb, level_min=0.6, stratified_anchor=False)
    sb = rb.random()
    assert a.tobytes() == b.tobytes() and sa == sb


def test_isothermal_zones_group_levels():
    """§6 test 11: components inside a zone share the zone base level (spread within a zone <<
    the global spread); zones=None path is unaffected (pinned separately by the golden test)."""
    # three separated blocks; two of them fall inside a single zone => near-equal levels
    cov = np.zeros((200, 200), np.float32)
    cov[20:50, 20:50] = 1.0        # block A (in zone)
    cov[20:50, 120:150] = 1.0      # block B (in zone)
    cov[150:180, 100:130] = 1.0    # block C (outside zone)
    hr = 10.0
    zones = [{"cy_um": 350.0, "cx_um": 850.0, "h_um": 400.0, "w_um": 1500.0, "kind": "pads"}]
    rng = np.random.default_rng(2)
    field = realism.render_isothermal_field(cov, rng, t_bg_c=0.0, delta_t_c=1.0, level_min=0.6,
                                            edge_sigma=0.0, zones=zones, zone_rotation_deg=0.0,
                                            zone_level_jitter=0.0, hr_pitch_um=hr)
    lvl_a = float(field[30:40, 30:40].mean())
    lvl_b = float(field[30:40, 130:140].mean())
    assert abs(lvl_a - lvl_b) < 0.02        # jitter=0 => identical zone base for A and B
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
