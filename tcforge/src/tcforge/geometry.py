"""Binary chip-structure mask generators for ThermalChipPhantom."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULT_CANVAS_SHAPE = (960, 1280)
DEFAULT_PIXEL_SIZE_UM = 10.0
DEFAULT_SCALE = 2
CANVAS_SHAPE_4X = (1920, 2560)
SCALE_4X = 4
DEFAULT_ANTIALIAS = True
DEFAULT_SSAA_FACTOR = 4


def _canvas(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = _as_shape(shape, "canvas_shape")
    return np.zeros((rows, cols), dtype=np.uint8)


def _as_shape(shape: tuple[int, int], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must be a two-value (rows, cols) tuple")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name} entries must be positive")
    return rows, cols


def _hr_pitch_um(pixel_size_um: float, scale: int) -> float:
    pixel_size = float(pixel_size_um)
    scale = int(scale)
    if pixel_size <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return pixel_size / scale


def _length_px(value_um: float, pixel_size_um: float, scale: int, name: str) -> int:
    value = float(value_um)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return max(1, int(round(value / _hr_pitch_um(pixel_size_um, scale))))


def _coord_px(value_um: float, pixel_size_um: float, scale: int) -> int:
    return int(round(float(value_um) / _hr_pitch_um(pixel_size_um, scale)))


def _paste_rect(mask: np.ndarray, cx_px: int, cy_px: int, w_px: int, h_px: int) -> None:
    rows, cols = mask.shape
    x0 = cx_px - w_px // 2
    y0 = cy_px - h_px // 2
    x1 = x0 + w_px
    y1 = y0 + h_px
    xs0, xs1 = max(0, x0), min(cols, x1)
    ys0, ys1 = max(0, y0), min(rows, y1)
    if xs0 < xs1 and ys0 < ys1:
        mask[ys0:ys1, xs0:xs1] = 1


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8, copy=False)


def _downsample_coverage(mask: np.ndarray, factor: int) -> np.ndarray:
    """Block-average an SSAA binary/float mask back to coverage in [0, 1]."""

    f = int(factor)
    if f <= 0:
        raise ValueError("factor must be > 0")
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    rows, cols = arr.shape
    if rows % f != 0 or cols % f != 0:
        raise ValueError("mask shape must be divisible by factor")
    out = arr.reshape(rows // f, f, cols // f, f).mean(axis=(1, 3))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _rotate_coverage_mask(
    mask: np.ndarray,
    angle_deg: float,
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Rotate a float coverage mask without re-binarizing it."""

    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    # Fast path: skip rotation for near-zero angles
    if abs(float(angle_deg)) < 0.01:
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    rotated = ndimage.rotate(
        arr,
        float(angle_deg),
        reshape=False,
        order=int(order),
        mode=mode,
        cval=float(cval),
        prefilter=False,
    )
    return np.clip(rotated, 0.0, 1.0).astype(np.float32, copy=False)


def make_rectangle(
    cx_um: float,
    cy_um: float,
    w_um: float,
    h_um: float,
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a filled rectangle mask on the HR canvas."""

    mask = _canvas(canvas_shape)
    _paste_rect(
        mask,
        _coord_px(cx_um, pixel_size_um, scale),
        _coord_px(cy_um, pixel_size_um, scale),
        _length_px(w_um, pixel_size_um, scale, "w_um"),
        _length_px(h_um, pixel_size_um, scale, "h_um"),
    )
    return mask


def make_ellipse_pad(
    cx_um: float,
    cy_um: float,
    w_um: float,
    h_um: float,
    *,
    angle_deg: float = 0.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a filled elliptical pad/via mask on the HR canvas."""

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    pitch = _hr_pitch_um(pixel_size_um, scale)
    rx = float(w_um) / (2.0 * pitch)
    ry = float(h_um) / (2.0 * pitch)
    if rx <= 0 or ry <= 0:
        raise ValueError("w_um and h_um must be > 0")
    cx_px = float(cx_um) / pitch
    cy_px = float(cy_um) / pitch
    radius = max(rx, ry) + 2.0
    x0 = max(0, int(np.floor(cx_px - radius)))
    x1 = min(cols, int(np.ceil(cx_px + radius + 1.0)))
    y0 = max(0, int(np.floor(cy_px - radius)))
    y1 = min(rows, int(np.ceil(cy_px + radius + 1.0)))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    if x0 >= x1 or y0 >= y1:
        return mask
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    x = xx - cx_px
    y = yy - cy_px
    theta = np.radians(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    mask[y0:y1, x0:x1] = (((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0).astype(np.uint8)
    return mask


def make_circle_pad(
    cx_um: float,
    cy_um: float,
    diameter_um: float,
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a filled circular pad/via mask on the HR canvas."""

    return make_ellipse_pad(
        cx_um,
        cy_um,
        diameter_um,
        diameter_um,
        angle_deg=0.0,
        canvas_shape=canvas_shape,
        pixel_size_um=pixel_size_um,
        scale=scale,
    )


def make_via_array(
    n_rows: int,
    n_cols: int,
    spacing_um: float,
    pad_diameter_um: float,
    cx_um: float,
    cy_um: float,
    *,
    stagger: bool = False,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a regular circular via/pad array centered at ``cx_um, cy_um``."""

    rows = int(n_rows)
    cols = int(n_cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("n_rows and n_cols must be > 0")
    if float(spacing_um) <= 0:
        raise ValueError("spacing_um must be > 0")
    masks: list[np.ndarray] = []
    y_offsets = (np.arange(rows, dtype=float) - (rows - 1) / 2.0) * float(spacing_um)
    x_offsets = (np.arange(cols, dtype=float) - (cols - 1) / 2.0) * float(spacing_um)
    for row_idx, y_off in enumerate(y_offsets):
        row_shift = 0.5 * float(spacing_um) if stagger and row_idx % 2 else 0.0
        for x_off in x_offsets:
            masks.append(
                make_circle_pad(
                    cx_um + x_off + row_shift,
                    cy_um + y_off,
                    pad_diameter_um,
                    canvas_shape=canvas_shape,
                    pixel_size_um=pixel_size_um,
                    scale=scale,
                )
            )
    return composite(*masks, canvas_shape=canvas_shape)


def make_frame(
    cx_um: float,
    cy_um: float,
    outer_w_um: float,
    outer_h_um: float,
    inner_w_um: float,
    inner_h_um: float,
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a rectangular frame mask as outer rectangle minus inner cutout."""

    if inner_w_um >= outer_w_um or inner_h_um >= outer_h_um:
        raise ValueError("inner_w_um/inner_h_um must be smaller than the outer dimensions")
    outer = make_rectangle(
        cx_um,
        cy_um,
        outer_w_um,
        outer_h_um,
        canvas_shape=canvas_shape,
        pixel_size_um=pixel_size_um,
        scale=scale,
    )
    inner = make_rectangle(
        cx_um,
        cy_um,
        inner_w_um,
        inner_h_um,
        canvas_shape=canvas_shape,
        pixel_size_um=pixel_size_um,
        scale=scale,
    )
    outer[inner == 1] = 0
    return outer


def make_pin_array(
    n_pins: int,
    spacing_um: float,
    pin_w_um: float,
    pin_l_um: float,
    cx_um: float,
    cy_um: float,
    direction: str = "horizontal",
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a parallel pin array; direction names the long pin axis."""

    n = int(n_pins)
    if n <= 0:
        raise ValueError("n_pins must be > 0")
    if float(spacing_um) <= 0:
        raise ValueError("spacing_um must be > 0")
    offsets = (np.arange(n, dtype=float) - (n - 1) / 2.0) * float(spacing_um)
    masks: list[np.ndarray] = []
    if direction == "horizontal":
        for off in offsets:
            masks.append(
                make_rectangle(
                    cx_um,
                    cy_um + off,
                    pin_l_um,
                    pin_w_um,
                    canvas_shape=canvas_shape,
                    pixel_size_um=pixel_size_um,
                    scale=scale,
                )
            )
    elif direction == "vertical":
        for off in offsets:
            masks.append(
                make_rectangle(
                    cx_um + off,
                    cy_um,
                    pin_w_um,
                    pin_l_um,
                    canvas_shape=canvas_shape,
                    pixel_size_um=pixel_size_um,
                    scale=scale,
                )
            )
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    return composite(*masks, canvas_shape=canvas_shape)


def make_cross(
    cx_um: float,
    cy_um: float,
    arm_w_um: float,
    arm_l_um: float,
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a plus-shaped cross mask."""

    return composite(
        make_rectangle(
            cx_um,
            cy_um,
            arm_l_um,
            arm_w_um,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_rectangle(
            cx_um,
            cy_um,
            arm_w_um,
            arm_l_um,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        canvas_shape=canvas_shape,
    )


def make_trenches(
    cx_um: float,
    cy_um: float,
    width_um: float,
    n_trenches: int,
    spacing_um: float,
    direction: str = "horizontal",
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate full-canvas parallel trench/bar masks."""

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    pitch = _hr_pitch_um(pixel_size_um, scale)
    canvas_w_um = cols * pitch
    canvas_h_um = rows * pitch
    n = int(n_trenches)
    if n <= 0:
        raise ValueError("n_trenches must be > 0")
    if float(spacing_um) <= 0:
        raise ValueError("spacing_um must be > 0")
    offsets = (np.arange(n, dtype=float) - (n - 1) / 2.0) * float(spacing_um)
    if direction == "horizontal":
        masks = [
            make_rectangle(
                cx_um,
                cy_um + off,
                canvas_w_um,
                width_um,
                canvas_shape=canvas_shape,
                pixel_size_um=pixel_size_um,
                scale=scale,
            )
            for off in offsets
        ]
    elif direction == "vertical":
        masks = [
            make_rectangle(
                cx_um + off,
                cy_um,
                width_um,
                canvas_h_um,
                canvas_shape=canvas_shape,
                pixel_size_um=pixel_size_um,
                scale=scale,
            )
            for off in offsets
        ]
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    return composite(*masks, canvas_shape=canvas_shape)


def make_l_shape(
    cx_um: float,
    cy_um: float,
    w1_um: float,
    h1_um: float,
    w2_um: float,
    h2_um: float,
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate an L mask from a vertical leg and a bottom horizontal leg."""

    total_w = max(float(w1_um), float(w2_um))
    total_h = max(float(h1_um), float(h2_um))
    vertical_cx = float(cx_um) - total_w / 2.0 + float(w1_um) / 2.0
    vertical_cy = float(cy_um)
    horizontal_cx = float(cx_um)
    horizontal_cy = float(cy_um) + total_h / 2.0 - float(h2_um) / 2.0
    return composite(
        make_rectangle(
            vertical_cx,
            vertical_cy,
            w1_um,
            h1_um,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_rectangle(
            horizontal_cx,
            horizontal_cy,
            w2_um,
            h2_um,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        canvas_shape=canvas_shape,
    )


def composite(
    *masks: np.ndarray,
    canvas_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Merge masks with OR logic, padding/cropping to ``canvas_shape`` if set."""

    if not masks and canvas_shape is None:
        raise ValueError("composite requires at least one mask or canvas_shape")
    if canvas_shape is None:
        rows = max(int(mask.shape[0]) for mask in masks)
        cols = max(int(mask.shape[1]) for mask in masks)
    else:
        rows, cols = _as_shape(canvas_shape, "canvas_shape")
    out = np.zeros((rows, cols), dtype=np.uint8)
    for mask in masks:
        arr = _binary(mask)
        rr = min(rows, arr.shape[0])
        cc = min(cols, arr.shape[1])
        out[:rr, :cc] |= arr[:rr, :cc]
    return out


def rotate_mask(
    mask: np.ndarray,
    angle_deg: float,
    *,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Rotate a mask in-place canvas coordinates and re-binarize at 0.5."""

    arr = _binary(mask)
    rotated = ndimage.rotate(arr, float(angle_deg), reshape=False, order=int(order), mode=mode, cval=float(cval))
    return (rotated >= 0.5).astype(np.uint8, copy=False)


def apply_edge_diffusion(
    mask: np.ndarray,
    *,
    sigma_um: float = 5.0,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
    mode: str = "nearest",
) -> np.ndarray:
    """Blur structure edges in physical units to mimic thermal spreading.

    Binary inputs return a soft mask in [0, 1]. Integer label maps keep their
    numeric range and become a continuous temperature-prior surface.
    """

    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    sigma = float(sigma_um)
    if sigma < 0:
        raise ValueError("sigma_um must be >= 0")
    if sigma == 0:
        return arr.copy()
    sigma_px = sigma / _hr_pitch_um(pixel_size_um, scale)
    out = ndimage.gaussian_filter(arr, sigma=sigma_px, mode=mode)
    if np.isin(mask, [0, 1]).all():
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def build_scene_mask(
    difficulty: str,
    seed: int,
    *,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
    antialias: bool = DEFAULT_ANTIALIAS,
    ssaa_factor: int = DEFAULT_SSAA_FACTOR,
    inscribe_disc: bool = False,
) -> np.ndarray:
    """Build a deterministic full-scene mask for a requested difficulty level."""

    mask, _metadata = build_scene_mask_with_metadata(
        difficulty,
        seed,
        rotation_deg_center=rotation_deg_center,
        rotation_jitter_deg=rotation_jitter_deg,
        canvas_shape=canvas_shape,
        pixel_size_um=pixel_size_um,
        scale=scale,
        antialias=antialias,
        ssaa_factor=ssaa_factor,
        inscribe_disc=inscribe_disc,
    )
    return mask


def build_multi_temp_mask(
    difficulty: str,
    seed: int,
    *,
    n_temp_levels: int = 4,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Build an integer label map with multiple structure temperature levels."""

    labels, _metadata = build_multi_temp_mask_with_metadata(
        difficulty,
        seed,
        n_temp_levels=n_temp_levels,
        rotation_deg_center=rotation_deg_center,
        rotation_jitter_deg=rotation_jitter_deg,
        canvas_shape=canvas_shape,
        pixel_size_um=pixel_size_um,
        scale=scale,
    )
    return labels


def build_multi_temp_mask_with_metadata(
    difficulty: str,
    seed: int,
    *,
    n_temp_levels: int = 4,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a deterministic multi-label IC mask for multi-temperature scenes.

    Label 0 is background. Labels 1..``n_temp_levels - 1`` partition the same
    structural support using a smooth random field, so neighboring regions tend
    to share a thermal layer instead of salt-and-pepper labels.
    """

    levels = int(n_temp_levels)
    if levels < 2 or levels > 255:
        raise ValueError("n_temp_levels must be in [2, 255]")
    base, metadata = build_scene_mask_with_metadata(
        difficulty,
        seed,
        rotation_deg_center=rotation_deg_center,
        rotation_jitter_deg=rotation_jitter_deg,
        canvas_shape=canvas_shape,
        pixel_size_um=pixel_size_um,
        scale=scale,
        antialias=False,
    )
    active = base >= 0.5
    labels = np.zeros(base.shape, dtype=np.uint8)
    if np.any(active):
        rng = np.random.default_rng(seed + 7919)
        field = rng.normal(size=base.shape).astype(np.float32)
        sigma_px = max(2.0, min(base.shape) / 28.0)
        field = ndimage.gaussian_filter(field, sigma=sigma_px, mode="nearest")
        values = field[active]
        if levels == 2:
            labels[active] = 1
        else:
            cuts = np.quantile(values, np.linspace(0.0, 1.0, levels)[1:-1])
            labels[active] = np.digitize(values, cuts).astype(np.uint8) + 1
    counts = {str(label): int(np.count_nonzero(labels == label)) for label in np.unique(labels)}
    out_meta = dict(metadata)
    out_meta.update(
        {
            "n_temp_levels": levels,
            "label_counts": counts,
            "implementation": "tcforge.geometry.build_multi_temp_mask",
        }
    )
    return labels, out_meta


def build_scene_mask_with_metadata(
    difficulty: str,
    seed: int,
    *,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
    antialias: bool = DEFAULT_ANTIALIAS,
    ssaa_factor: int = DEFAULT_SSAA_FACTOR,
    inscribe_disc: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a dense IC-layout-style chip mask with blocks and routing.

    Generates scenes resembling real chip die photos:
      - Large solid rectangular blocks filling most of the canvas
      - Thin L-shaped routing traces with varying widths
      - Fine detail features (small rectangles, crosses)
      - Round/elliptical pads and via arrays
      - Peripheral pin arrays
    """

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    aa_factor = int(ssaa_factor) if bool(antialias) else 1
    if aa_factor <= 0:
        raise ValueError("ssaa_factor must be > 0")
    draw_shape = (rows * aa_factor, cols * aa_factor)
    draw_scale = int(scale) * aa_factor
    pitch = _hr_pitch_um(pixel_size_um, scale)
    canvas_w_um = cols * pitch
    canvas_h_um = rows * pitch
    cx = canvas_w_um / 2.0
    cy = canvas_h_um / 2.0
    rng = np.random.default_rng(seed)

    levels = {
        "easy": (80.0, 60.0, 45.0),
        "medium": (55.0, 42.0, 30.0),
        "hard": (35.0, 28.0, 22.0),
        "stress": (22.0, 16.0, 12.0),
    }
    if difficulty not in levels:
        raise ValueError("difficulty must be one of: easy, medium, hard, stress")
    major, minor, line = levels[difficulty]

    common: dict[str, object] = dict(
        canvas_shape=draw_shape, pixel_size_um=pixel_size_um, scale=draw_scale,
    )
    mask = np.zeros(draw_shape, dtype=np.uint8)
    subtract_mask = np.zeros(draw_shape, dtype=np.uint8)
    primitives: list[dict[str, object]] = []

    def _add_part(part: np.ndarray) -> None:
        nonlocal mask
        mask |= _binary(part)

    def _add_subtract_part(part: np.ndarray) -> None:
        nonlocal subtract_mask
        subtract_mask |= _binary(part)

    def _rand_pos(margin: float = 0.08) -> tuple[float, float]:
        return (
            float(rng.uniform(margin * canvas_w_um, (1 - margin) * canvas_w_um)),
            float(rng.uniform(margin * canvas_h_um, (1 - margin) * canvas_h_um)),
        )

    def _mirror(x: float, y: float) -> tuple[float, float]:
        """Mirror a position across the canvas center."""
        return (2.0 * cx - x, 2.0 * cy - y)

    def _add_rect(bx: float, by: float, w: float, h: float, layer: int) -> None:
        _add_part(make_rectangle(bx, by, w, h, **common))
        primitives.append({
            "type": "rectangle", "layer": layer,
            "cx_um": bx, "cy_um": by, "w_um": w, "h_um": h,
        })

    # ── Layer 0: Large solid blocks (3-6) ───────────────────────────────
    # Big filled rectangles; reduced size to leave room for medium features.
    # ~40% are mirrored for partial symmetry.
    n_blocks = int(rng.integers(3, 7))
    for _ in range(n_blocks):
        bx, by = _rand_pos(0.05)
        w = canvas_w_um * float(rng.uniform(0.10, 0.30))
        h = canvas_h_um * float(rng.uniform(0.08, 0.25))
        _add_rect(bx, by, w, h, 0)
        if rng.random() < 0.40:
            mx, my = _mirror(bx, by)
            _add_rect(mx, my, w, h, 0)

    # ── Layer 0.5: Medium-width pillars and bars (4-8) ───────────────────
    # Vertical pillars and horizontal bars that connect between blocks.
    # These create the "middle scale" structure between big blocks and fine traces.
    n_pillars = int(rng.integers(4, 9))
    for _ in range(n_pillars):
        px, py = _rand_pos(0.06)
        # Medium width: between fine traces and big blocks
        pillar_w = line * float(rng.uniform(2.0, 6.0))
        pillar_l = canvas_h_um * float(rng.uniform(0.15, 0.50))  # long
        if rng.random() < 0.5:
            # Vertical pillar
            _add_rect(px, py, pillar_w, pillar_l, 0)
        else:
            # Horizontal bar
            _add_rect(px, py, pillar_l, pillar_w, 0)
        # Mirror ~40% for symmetry
        if rng.random() < 0.40:
            mx, my = _mirror(px, py)
            if rng.random() < 0.5:
                _add_rect(mx, my, pillar_w, pillar_l, 0)
            else:
                _add_rect(mx, my, pillar_l, pillar_w, 0)

    # ── Layer 1: L-shaped routing traces (8-16) ─────────────────────────
    # Thin right-angle traces; ~50% get a mirrored twin for symmetry.
    def _add_l_trace(x1: float, y1: float, x2: float, y2: float,
                     trace_w: float) -> None:
        if rng.random() < 0.5:
            corner_x, corner_y = x2, y1
        else:
            corner_x, corner_y = x1, y2
        seg_h_cx = (x1 + corner_x) / 2.0
        seg_h_w = abs(corner_x - x1) + trace_w
        if seg_h_w > trace_w * 1.5:
            _add_rect(seg_h_cx, corner_y, seg_h_w, trace_w, 1)
        seg_v_cy = (corner_y + y2) / 2.0
        seg_v_h = abs(y2 - corner_y) + trace_w
        if seg_v_h > trace_w * 1.5:
            _add_rect(corner_x, seg_v_cy, trace_w, seg_v_h, 1)

    n_traces = int(rng.integers(10, 21))
    for _ in range(n_traces):
        if rng.random() < 0.25:
            trace_w = line * float(rng.uniform(1.5, 3.5))   # thick
        else:
            trace_w = line * float(rng.uniform(0.2, 1.0))    # thin
        x1, y1 = _rand_pos(0.03)
        x2, y2 = _rand_pos(0.03)
        _add_l_trace(x1, y1, x2, y2, trace_w)
        # Mirror ~50% of traces for symmetry
        if rng.random() < 0.50:
            mx1, my1 = _mirror(x1, y1)
            mx2, my2 = _mirror(x2, y2)
            _add_l_trace(mx1, my1, mx2, my2, trace_w)

    # ── Layer 2: Medium features — frames and sub-blocks (2-5) ───────────
    n_medium = int(rng.integers(2, 6))
    for _ in range(n_medium):
        fx, fy = _rand_pos(0.10)
        factor = float(rng.uniform(0.5, 1.2))

        if rng.random() < 0.5:
            # Frame (hollow rectangle) — creates the "cutout" look
            outer_w = float(rng.uniform(2.0, 5.0)) * major * factor
            outer_h = float(rng.uniform(1.5, 4.0)) * major * factor
            border = line * float(rng.uniform(1.0, 3.0)) * factor
            inner_w = max(outer_w - 2.0 * border, outer_w * 0.3)
            inner_h = max(outer_h - 2.0 * border, outer_h * 0.3)
            _add_part(make_frame(fx, fy, outer_w, outer_h, inner_w, inner_h, **common))
            primitives.append({
                "type": "frame", "layer": 2,
                "cx_um": fx, "cy_um": fy,
                "outer_w_um": outer_w, "outer_h_um": outer_h,
                "inner_w_um": inner_w, "inner_h_um": inner_h,
            })
        else:
            # L-shape structure — right-angle bend
            w1 = line * factor * float(rng.uniform(1.0, 3.0))
            h1 = float(rng.uniform(2.0, 5.0)) * major * factor
            w2 = float(rng.uniform(2.0, 5.0)) * major * factor
            h2 = line * factor * float(rng.uniform(1.0, 3.0))
            _add_part(make_l_shape(fx, fy, w1, h1, w2, h2, **common))
            primitives.append({
                "type": "l_shape", "layer": 2,
                "cx_um": fx, "cy_um": fy,
                "w1_um": w1, "h1_um": h1, "w2_um": w2, "h2_um": h2,
            })

    # ── Layer 3: Fine detail (6-12) — small rects, crosses, pins ─────────
    n_detail = int(rng.integers(6, 13))
    for _ in range(n_detail):
        fx, fy = _rand_pos(0.08)
        factor = float(rng.uniform(0.3, 0.8))
        choice = float(rng.random())

        if choice < 0.40:
            # Small solid rectangle
            w = float(rng.uniform(0.8, 2.5)) * major * factor
            h = float(rng.uniform(0.8, 2.5)) * major * factor
            _add_part(make_rectangle(fx, fy, w, h, **common))
            primitives.append({
                "type": "rectangle", "layer": 3,
                "cx_um": fx, "cy_um": fy, "w_um": w, "h_um": h,
            })
        elif choice < 0.65:
            # Cross marker
            arm_w = line * factor
            arm_l = float(rng.uniform(1.5, 3.0)) * major * factor
            _add_part(make_cross(fx, fy, arm_w, arm_l, **common))
            primitives.append({
                "type": "cross", "layer": 3,
                "cx_um": fx, "cy_um": fy,
                "arm_w_um": arm_w, "arm_l_um": arm_l,
            })
        else:
            # Small pin array
            n_pins = int(rng.choice([3, 4, 5, 6]))
            direction = str(rng.choice(["horizontal", "vertical"]))
            spacing = minor * factor
            pin_w = line * factor
            pin_l = float(rng.uniform(1.0, 2.5)) * major * factor
            _add_part(make_pin_array(
                n_pins, spacing, pin_w, pin_l, fx, fy, direction, **common,
            ))
            primitives.append({
                "type": "pin_array", "layer": 3,
                "cx_um": fx, "cy_um": fy,
                "n_pins": n_pins, "direction": direction,
                "spacing_um": spacing, "pin_w_um": pin_w, "pin_l_um": pin_l,
            })

    # ── Layer 3.5: Curved pads/vias (2-6) ───────────────────────────────
    # Round or elliptical structures reduce overfitting to Manhattan-only
    # geometry and better cover via pads, BGA-like contacts, and rounded
    # thermal islands seen in real packages.
    n_curved = int(rng.integers(2, 7))
    for _ in range(n_curved):
        px, py = _rand_pos(0.08)
        factor = float(rng.uniform(0.5, 1.5))
        choice = float(rng.random())

        if choice < 0.35:
            diameter = line * factor * float(rng.uniform(1.4, 4.0))
            _add_part(make_circle_pad(px, py, diameter, **common))
            primitives.append({
                "type": "circle_pad", "layer": 3,
                "cx_um": px, "cy_um": py, "diameter_um": diameter,
            })
        elif choice < 0.70:
            w = line * factor * float(rng.uniform(2.0, 6.0))
            h = line * factor * float(rng.uniform(1.2, 3.5))
            angle = float(rng.uniform(0.0, 180.0))
            _add_part(make_ellipse_pad(px, py, w, h, angle_deg=angle, **common))
            primitives.append({
                "type": "ellipse_pad", "layer": 3,
                "cx_um": px, "cy_um": py,
                "w_um": w, "h_um": h, "angle_deg": angle,
            })
        else:
            n_rows = int(rng.choice([2, 3, 4]))
            n_cols = int(rng.choice([2, 3, 4, 5]))
            diameter = line * factor * float(rng.uniform(0.9, 2.0))
            spacing = max(diameter * float(rng.uniform(1.7, 2.4)), minor * factor * 0.65)
            stagger = bool(rng.random() < 0.35)
            _add_part(
                make_via_array(
                    n_rows,
                    n_cols,
                    spacing,
                    diameter,
                    px,
                    py,
                    stagger=stagger,
                    **common,
                )
            )
            primitives.append({
                "type": "via_array", "layer": 3,
                "cx_um": px, "cy_um": py,
                "n_rows": n_rows, "n_cols": n_cols,
                "spacing_um": spacing, "diameter_um": diameter,
                "stagger": stagger,
            })

    # ── Layer 4: Edge pin arrays — always symmetric opposing pairs ────────
    # Pick 1-2 axis pairs; each pair places pins on both opposing edges.
    axis_pairs = [("top", "bottom"), ("left", "right")]
    rng.shuffle(axis_pairs)
    n_pairs = int(rng.integers(1, 3))  # 1 or 2 pairs
    for pair in axis_pairs[:n_pairs]:
        factor = float(rng.uniform(0.6, 1.3))
        n_pins = int(rng.choice([6, 8, 10, 12, 16]))
        spacing = minor * factor
        pin_w = line * factor
        pin_l = float(rng.uniform(2.5, 5.0)) * major * factor
        for edge in pair:
            if edge in ("top", "bottom"):
                py = (0.03 * canvas_h_um) if edge == "top" else (0.97 * canvas_h_um)
                _add_part(make_pin_array(
                    n_pins, spacing, pin_w, pin_l, cx, py, "horizontal", **common,
                ))
            else:
                px = (0.03 * canvas_w_um) if edge == "left" else (0.97 * canvas_w_um)
                _add_part(make_pin_array(
                    n_pins, spacing, pin_w, pin_l, px, cy, "vertical", **common,
                ))
            primitives.append({
                "type": "pin_array", "layer": 4, "edge": edge,
                "n_pins": n_pins, "direction": "horizontal" if edge in ("top", "bottom") else "vertical",
            })

    # ── Layer 5: Subtractive channels — carve concave indentations ────────
    # Thin rectangular channels subtracted from the mask to create concave
    # polygon outlines on blocks. Mix of straight and L-shaped channels.
    n_channels = int(rng.integers(4, 10))
    for _ in range(n_channels):
        ch_w = line * float(rng.uniform(0.3, 1.8))
        ch_type = float(rng.random())

        if ch_type < 0.45:
            # Straight horizontal channel
            ch_y = float(rng.uniform(0.10 * canvas_h_um, 0.90 * canvas_h_um))
            ch_x = canvas_w_um / 2.0
            ch_len = canvas_w_um * float(rng.uniform(0.30, 0.85))
            _add_subtract_part(make_rectangle(ch_x, ch_y, ch_len, ch_w, **common))
            primitives.append({"type": "channel_h", "layer": 5,
                               "cx_um": ch_x, "cy_um": ch_y, "len_um": ch_len, "w_um": ch_w})
            # Mirror for symmetry ~50%
            if rng.random() < 0.50:
                my = 2.0 * cy - ch_y
                _add_subtract_part(make_rectangle(ch_x, my, ch_len, ch_w, **common))
        elif ch_type < 0.90:
            # Straight vertical channel
            ch_x = float(rng.uniform(0.10 * canvas_w_um, 0.90 * canvas_w_um))
            ch_y = canvas_h_um / 2.0
            ch_len = canvas_h_um * float(rng.uniform(0.30, 0.85))
            _add_subtract_part(make_rectangle(ch_x, ch_y, ch_w, ch_len, **common))
            primitives.append({"type": "channel_v", "layer": 5,
                               "cx_um": ch_x, "cy_um": ch_y, "len_um": ch_len, "w_um": ch_w})
            if rng.random() < 0.50:
                mx = 2.0 * cx - ch_x
                _add_subtract_part(make_rectangle(mx, ch_y, ch_w, ch_len, **common))
        else:
            # L-shaped channel (two perpendicular cuts)
            x1, y1 = _rand_pos(0.08)
            x2, y2 = _rand_pos(0.08)
            corner_x, corner_y = x2, y1
            seg_h_cx = (x1 + corner_x) / 2.0
            seg_h_w = abs(corner_x - x1) + ch_w
            if seg_h_w > ch_w * 1.5:
                _add_subtract_part(make_rectangle(seg_h_cx, corner_y, seg_h_w, ch_w, **common))
            seg_v_cy = (corner_y + y2) / 2.0
            seg_v_h = abs(y2 - corner_y) + ch_w
            if seg_v_h > ch_w * 1.5:
                _add_subtract_part(make_rectangle(corner_x, seg_v_cy, ch_w, seg_v_h, **common))
            primitives.append({"type": "channel_l", "layer": 5,
                               "x1_um": x1, "y1_um": y1, "x2_um": x2, "y2_um": y2, "w_um": ch_w})

    # ── Assemble and rotate ──────────────────────────────────────────────
    # Subtract channels to create concave indentations
    mask = mask & ~subtract_mask
    # Optionally inscribe a centered disc so corner content cannot be clipped
    # by reshape=False rotation across arbitrary 0–360° angles (self-check T4).
    if bool(inscribe_disc):
        draw_rows, draw_cols = draw_shape
        yy_d, xx_d = np.mgrid[:draw_rows, :draw_cols]
        radius = min(draw_rows, draw_cols) / 2.0
        disc = (
            (yy_d - draw_rows / 2.0) ** 2 + (xx_d - draw_cols / 2.0) ** 2
        ) <= radius ** 2
        mask = mask & disc.astype(np.uint8)
    angle = float(rotation_deg_center) + float(
        rng.uniform(-rotation_jitter_deg, rotation_jitter_deg)
    )
    metadata: dict[str, object] = {
        "units": "um",
        "rotation_deg": angle,
        "rotation_deg_center": float(rotation_deg_center),
        "rotation_jitter_deg": float(rotation_jitter_deg),
        "difficulty": difficulty,
        "n_primitives": len(primitives),
        "primitives": primitives,
        "antialias": bool(antialias),
        "ssaa_factor": aa_factor,
        "inscribe_disc": bool(inscribe_disc),
        "mask_semantics": "coverage" if bool(antialias) else "binary",
        "implementation": "tcforge.geometry.build_scene_mask_ic_layout",
    }
    if bool(antialias):
        rotated = _rotate_coverage_mask(mask, angle, order=1)
        return _downsample_coverage(rotated, aa_factor), metadata
    return rotate_mask(mask, angle, order=0), metadata
