"""Binary chip-structure mask generators for ThermalChipPhantom."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULT_CANVAS_SHAPE = (960, 1280)
# LR detector pitch (um/pixel); the HR canvas pitch is this divided by `scale`.
# 20.0 per the corrected project truth (the historical 10.0 was a 2x BMP scale-bar
# misread, ACL-023/AGENTS.md). Pool configs pass pixel_size_um explicitly; this
# default only guards bare calls.
DEFAULT_PIXEL_SIZE_UM = 20.0
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


def make_pad_grid(
    n_rows: int,
    n_cols: int,
    pitch_x_um: float,
    pitch_y_um: float,
    pad_w_um: float,
    pad_h_um: float,
    cx_um: float,
    cy_um: float,
    *,
    shape: str = "square",
    stagger: bool = False,
    present: np.ndarray | None = None,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a regular rectangular lattice of pads (BGA / PGA / via array).

    Pads are square (``shape="square"``) or round/elliptical (``shape="round"``).
    Independent ``pitch_x_um`` / ``pitch_y_um`` allow an anisotropic pitch;
    ``stagger`` offsets alternate rows by half a column pitch (hex-like packing).
    ``present`` is an optional ``(n_rows, n_cols)`` boolean mask selecting which
    sites are populated (missing balls / keep-out corners). Rasterised by
    vectorised stamping so large arrays stay cheap.
    """

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    nr, nc = int(n_rows), int(n_cols)
    if nr <= 0 or nc <= 0:
        raise ValueError("n_rows and n_cols must be > 0")
    if float(pitch_x_um) <= 0 or float(pitch_y_um) <= 0:
        raise ValueError("pitch_x_um and pitch_y_um must be > 0")
    if str(shape) not in ("square", "round"):
        raise ValueError("shape must be 'square' or 'round'")
    out = np.zeros((rows, cols), dtype=np.uint8)
    pitch = _hr_pitch_um(pixel_size_um, scale)
    pw = max(1, int(round(float(pad_w_um) / pitch)))
    ph = max(1, int(round(float(pad_h_um) / pitch)))
    # One stamp of pixel offsets from the pad centre, reused for every site.
    dyy, dxx = np.mgrid[0:ph, 0:pw].astype(np.float64)
    ccy = (ph - 1) / 2.0
    ccx = (pw - 1) / 2.0
    if str(shape) == "round":
        rx = max(pw / 2.0, 0.5)
        ry = max(ph / 2.0, 0.5)
        inside = (((dxx - ccx) / rx) ** 2 + ((dyy - ccy) / ry) ** 2) <= 1.0
    else:
        inside = np.ones((ph, pw), dtype=bool)
    off_dy = np.rint(dyy[inside] - ccy).astype(np.int64)
    off_dx = np.rint(dxx[inside] - ccx).astype(np.int64)
    if present is not None:
        present = np.asarray(present, dtype=bool)
        if present.shape != (nr, nc):
            raise ValueError("present must have shape (n_rows, n_cols)")
    y_off = (np.arange(nr, dtype=np.float64) - (nr - 1) / 2.0) * float(pitch_y_um)
    x_off = (np.arange(nc, dtype=np.float64) - (nc - 1) / 2.0) * float(pitch_x_um)
    cyc: list[int] = []
    cxc: list[int] = []
    for r in range(nr):
        row_shift = 0.5 * float(pitch_x_um) if (stagger and r % 2) else 0.0
        py = int(round((float(cy_um) + y_off[r]) / pitch))
        for c in range(nc):
            if present is not None and not bool(present[r, c]):
                continue
            cyc.append(py)
            cxc.append(int(round((float(cx_um) + x_off[c] + row_shift) / pitch)))
    if not cyc:
        return out
    cyc_arr = np.asarray(cyc, dtype=np.int64)
    cxc_arr = np.asarray(cxc, dtype=np.int64)
    for dy, dx in zip(off_dy.tolist(), off_dx.tolist()):
        ys = cyc_arr + dy
        xs = cxc_arr + dx
        m = (ys >= 0) & (ys < rows) & (xs >= 0) & (xs < cols)
        if np.any(m):
            out[ys[m], xs[m]] = 1
    return out


def make_trace_bus(
    cx_um: float,
    cy_um: float,
    n_traces: int,
    trace_w_um: float,
    pitch_um: float,
    trace_len_um: float,
    *,
    direction: str = "horizontal",
    spine_w_um: float = 0.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a bundle of parallel conductor traces (a routing bus / comb).

    ``n_traces`` bars of width ``trace_w_um`` and length ``trace_len_um`` spaced
    by ``pitch_um`` perpendicular to their long axis; ``direction`` names that
    long axis. When ``spine_w_um > 0`` a perpendicular spine ties one end of every
    trace (bond-finger / lead-frame comb); otherwise the traces are an
    unconnected parallel bus.
    """

    n = int(n_traces)
    if n <= 0:
        raise ValueError("n_traces must be > 0")
    if float(pitch_um) <= 0:
        raise ValueError("pitch_um must be > 0")
    common = dict(canvas_shape=canvas_shape, pixel_size_um=pixel_size_um, scale=scale)
    offsets = (np.arange(n, dtype=float) - (n - 1) / 2.0) * float(pitch_um)
    span_um = (n - 1) * float(pitch_um) + float(trace_w_um)
    masks: list[np.ndarray] = []
    if direction == "horizontal":
        for off in offsets:
            masks.append(make_rectangle(cx_um, cy_um + off, trace_len_um, trace_w_um, **common))
        if float(spine_w_um) > 0:
            spine_x = float(cx_um) + float(trace_len_um) / 2.0 - float(spine_w_um) / 2.0
            masks.append(make_rectangle(spine_x, cy_um, spine_w_um, span_um, **common))
    elif direction == "vertical":
        for off in offsets:
            masks.append(make_rectangle(cx_um + off, cy_um, trace_w_um, trace_len_um, **common))
        if float(spine_w_um) > 0:
            spine_y = float(cy_um) + float(trace_len_um) / 2.0 - float(spine_w_um) / 2.0
            masks.append(make_rectangle(cx_um, spine_y, span_um, spine_w_um, **common))
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
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


def make_rounded_rect(
    cx_um: float,
    cy_um: float,
    w_um: float,
    h_um: float,
    corner_radius_um: float,
    *,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a filled rectangle with rounded (chamfered) corners.

    ``corner_radius_um`` is clamped to at most half the shorter side, so a large
    radius degrades gracefully to a stadium/disc-like body. Rasterised via the
    standard rounded-box signed-distance test on the local bounding box.
    """

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    pitch = _hr_pitch_um(pixel_size_um, scale)
    hw = float(w_um) / (2.0 * pitch)
    hh = float(h_um) / (2.0 * pitch)
    if hw <= 0 or hh <= 0:
        raise ValueError("w_um and h_um must be > 0")
    r = float(corner_radius_um) / pitch
    r = max(0.0, min(r, min(hw, hh)))
    cx_px = float(cx_um) / pitch
    cy_px = float(cy_um) / pitch
    mask = np.zeros((rows, cols), dtype=np.uint8)
    x0 = max(0, int(np.floor(cx_px - hw - 1.0)))
    x1 = min(cols, int(np.ceil(cx_px + hw + 2.0)))
    y0 = max(0, int(np.floor(cy_px - hh - 1.0)))
    y1 = min(rows, int(np.ceil(cy_px + hh + 2.0)))
    if x0 >= x1 or y0 >= y1:
        return mask
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    qx = np.abs(xx - cx_px) - (hw - r)
    qy = np.abs(yy - cy_px) - (hh - r)
    outside = (
        np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        + np.minimum(np.maximum(qx, qy), 0.0)
        - r
    )
    mask[y0:y1, x0:x1] = (outside <= 0.0).astype(np.uint8)
    return mask


def _fill_polygon(rows: int, cols: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Rasterise a filled polygon (even-odd / PNPoly) over its bounding box.

    ``xs`` are column (x) and ``ys`` are row (y) vertex coordinates, in pixels.
    """

    mask = np.zeros((int(rows), int(cols)), dtype=np.uint8)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3:
        return mask
    x0 = max(0, int(np.floor(xs.min())))
    x1 = min(int(cols), int(np.ceil(xs.max())) + 1)
    y0 = max(0, int(np.floor(ys.min())))
    y1 = min(int(rows), int(np.ceil(ys.max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return mask
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    inside = np.zeros(yy.shape, dtype=bool)
    n = xs.size
    j = n - 1
    for i in range(n):
        yi, yj = float(ys[i]), float(ys[j])
        xi, xj = float(xs[i]), float(xs[j])
        cond = (yi > yy) != (yj > yy)
        slope = (xj - xi) / (yj - yi) if (yj - yi) != 0.0 else 0.0
        xint = xi + (yy - yi) * slope
        inside ^= cond & (xx < xint)
        j = i
    mask[y0:y1, x0:x1] = inside.astype(np.uint8)
    return mask


def make_regular_polygon(
    cx_um: float,
    cy_um: float,
    radius_um: float,
    n_sides: int,
    *,
    rotation_deg: float = 0.0,
    vertex_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Generate a filled regular n-gon (octagon/hexagon package footprints, ...).

    ``radius_um`` is the circumradius. ``vertex_jitter`` (with ``rng``) perturbs
    each vertex radius/angle to yield an irregular-boundary part instead of a
    clean polygon.
    """

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    n = int(n_sides)
    if n < 3:
        raise ValueError("n_sides must be >= 3")
    pitch = _hr_pitch_um(pixel_size_um, scale)
    radius_px = float(radius_um) / pitch
    if radius_px <= 0:
        raise ValueError("radius_um must be > 0")
    cx_px = float(cx_um) / pitch
    cy_px = float(cy_um) / pitch
    angles = np.radians(float(rotation_deg)) + 2.0 * np.pi * np.arange(n) / n
    radii = np.full(n, radius_px, dtype=np.float64)
    jit = float(vertex_jitter)
    if jit > 0.0 and rng is not None:
        radii = radius_px * (1.0 + rng.uniform(-jit, jit, size=n))
        angles = angles + rng.uniform(-jit, jit, size=n) * (np.pi / n)
    xs = cx_px + radii * np.cos(angles)
    ys = cy_px + radii * np.sin(angles)
    return _fill_polygon(rows, cols, xs, ys)


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


# ── CPU / electronic-part geometry family ────────────────────────────────────
# Motif builders that dominate the v6 "cpu" pool: die/package outlines, fine
# regular pad grids (PGA/BGA/via lattices), routing buses and heat-spreader
# fins — the "细密规则栅格/阵列" motif flagged in synthetic_data_realism.md. All
# grid pitches stay > detector pitch (ACL-023 band honesty: recoverable, not
# sub-pitch fantasy). Randomisation is scoped WITHIN this parts family.
CPU_SCENE_FAMILIES: tuple[str, ...] = (
    "pga_grid",
    "die_bga",
    "multi_die",
    "trace_bus",
    "heat_spreader",
    "generic",
)


def _pick_weighted(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Sample a key from a name->weight mapping (non-negative, positive sum)."""

    names = [str(k) for k in weights]
    vals = np.asarray([max(0.0, float(weights[k])) for k in names], dtype=np.float64)
    total = float(vals.sum())
    if not names or total <= 0:
        raise ValueError("motif_weights must be non-empty with a positive sum")
    return names[int(rng.choice(len(names), p=vals / total))]


def _present_mask(
    rng: np.random.Generator, nr: int, nc: int, drop_p: float, keepout: bool
) -> np.ndarray | None:
    """Boolean (nr, nc) populate-mask: random dropped sites + optional keep-out corner."""

    if drop_p <= 0 and not keepout:
        return None
    present = np.ones((nr, nc), dtype=bool)
    if drop_p > 0:
        present &= rng.random((nr, nc)) >= float(drop_p)
    if keepout and nr >= 3 and nc >= 3:
        kr = int(rng.integers(1, max(2, nr // 3) + 1))
        kc = int(rng.integers(1, max(2, nc // 3) + 1))
        corner = int(rng.integers(0, 4))
        if corner == 0:
            present[:kr, :kc] = False
        elif corner == 1:
            present[:kr, nc - kc:] = False
        elif corner == 2:
            present[nr - kr:, :kc] = False
        else:
            present[nr - kr:, nc - kc:] = False
    if int(present.sum()) < 2:
        present[:] = True
    return present


def _compose_cpu_scene(
    rng: np.random.Generator,
    *,
    motif_weights: dict[str, float],
    common: dict[str, object],
    draw_shape: tuple[int, int],
    pixel_size_um: float,
    canvas_w_um: float,
    canvas_h_um: float,
    cx: float,
    cy: float,
    detector_pitch_um: float,
) -> tuple[np.ndarray, list[dict[str, object]], str, float]:
    """Compose one CPU/part-styled scene as distinct hot structures on cool background.

    Difficulty tiers are GONE from this branch (redesign): a single continuous
    ``density`` scalar controls the AMOUNT/intricacy of internal structure while a
    hard ~28um (1.4-pitch) feature-size floor holds the finest feature safely inside
    the 20-40um recoverable band on EVERY scene. Part footprint is held constant-large
    independent of density. Picks one primary package archetype from ``motif_weights``
    and adds shared part clutter (passives, vias, edge I/O). Returns the (uint8
    draw-canvas) mask, the primitive metadata list, the chosen scene family and the
    sampled ``density`` (stored in scene metadata).
    """

    mask = np.zeros(draw_shape, dtype=np.uint8)
    subtract = np.zeros(draw_shape, dtype=np.uint8)
    primitives: list[dict[str, object]] = []

    def _add(part: np.ndarray) -> None:
        nonlocal mask
        mask |= _binary(part)

    def _sub(part: np.ndarray) -> None:
        nonlocal subtract
        subtract |= _binary(part)

    det = float(detector_pitch_um)
    # ── Feature-size honesty floor (ACL-023/030) ─────────────────────────────────
    # Detector pitch = 20um; recoverable band ~20-40um. The finest structural feature
    # (line/trace/channel/gap/pad/fin/moat width) must stay >= ~1.4 pitch = 28um on
    # EVERY scene — comfortably inside the band, robust through PSF+noise, so nothing
    # is sub-pitch hallucination bait (old 珠串/FM-1 beading). Pad/array PITCH floor
    # stays >= 1.6 pitch = 32um (band-honest as a periodic lattice).
    FLOOR = det * 1.4              # 28um: universal finest-feature floor
    pitch_floor = det * 1.6       # 32um: pad/array pitch floor
    pad_floor = FLOOR             # 28um pad-element floor (was det*0.9 = 18um, sub-pitch)
    # Fixed feature-size units (NO difficulty scaling): a base line width == the floor,
    # plus medium/major length units for passives / edge I/O. Every site re-floors at FLOOR.
    line = FLOOR                  # 28um base trace/line width (was difficulty 12-45um)
    minor = det * 3.0             # 60um small-feature length unit
    major = det * 5.0             # 100um medium-feature length unit
    # ── Continuous density (replaces the discrete difficulty tiers) ──────────────
    # density in [0,1] controls AMOUNT/intricacy (element/trace/sub-block/channel/cross/
    # die counts, pad-grid extent & per-axis element cap, symmetry probability, bus &
    # clutter counts), NOT feature fineness — the finest feature is FLOOR on every scene.
    density = float(rng.uniform(0.0, 1.0))

    def _lerp(a: float, b: float) -> float:
        return float(a + density * (b - a))

    def _dcount(lo: int, hi: int) -> int:
        """Integer count in [lo, hi] biased toward hi by density (with +/-1 jitter)."""
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return lo
        center = _lerp(float(lo), float(hi)) + float(rng.uniform(-0.7, 0.7))
        return int(np.clip(round(center), lo, hi))

    sym_p = _lerp(0.28, 0.72)     # symmetry (mirrored-twin) probability grows with density
    # Pad/array pitch band: coarse at density 0, tight (but always >= ~34um, still >
    # pitch_floor) at density 1. A grid's FOOTPRINT is held constant-large (set by the
    # family extent, below); density controls the grid FINENESS (pitch), and the element
    # count = extent/pitch follows => denser grids have MORE pads over the SAME large area,
    # never finer than the floor. ncap is only a high safety cap (never the footprint knob).
    p_lo = max(_lerp(90.0, pitch_floor * 1.06), pitch_floor)
    p_hi = max(_lerp(150.0, pitch_floor * 1.90), p_lo + 8.0)
    ncap = 220

    def _rpos(mw: float = 0.12, mh: float = 0.12) -> tuple[float, float]:
        return (
            float(rng.uniform(mw * canvas_w_um, (1.0 - mw) * canvas_w_um)),
            float(rng.uniform(mh * canvas_h_um, (1.0 - mh) * canvas_h_um)),
        )

    def _carve_routing(ox: float, oy: float, w: float, h: float) -> int:
        """Carve the die's dark internal routing: thin COOL channels subtracted from
        a filled die (the real anchor's interlocking metal pattern). Combs enter from
        alternating edges and span only part of the die, so the hot metal weaves
        between them and the die stays (near-)connected => still one isothermal body
        (temperature is per-component, NOT split by channel/line width). Returns the
        channel count."""
        cw = max(line * float(rng.uniform(0.9, 1.4)), FLOOR)       # cool routing channel >= 28um
        horiz = rng.random() < 0.5
        span = h if horiz else w                                   # spacing axis
        rt_pitch = max(pitch_floor * float(rng.uniform(1.3, 2.2)), cw * 2.5)
        n = int(np.clip(round(span * 0.8 / rt_pitch), 0, 24))
        if n <= 0:
            return 0
        offs = (np.arange(n, dtype=float) - (n - 1) / 2.0) * rt_pitch
        for i, off in enumerate(offs):
            frac = float(rng.uniform(0.55, 0.85))
            side = -1.0 if (i % 2 == 0) else 1.0                   # alternate => interlock
            if horiz:
                seg_w = w * frac
                _sub(make_rectangle(ox + side * (w - seg_w) / 2.0, oy + off,
                                    seg_w, cw, **common))
            else:
                seg_h = h * frac
                _sub(make_rectangle(ox + off, oy + side * (h - seg_h) / 2.0,
                                    cw, seg_h, **common))
        return int(n)

    def _add_grid(gx: float, gy: float, rw: float, rh: float, *,
                  keepout: bool = False, tag: str = "pad_grid",
                  on_die: bool = False, clip: np.ndarray | None = None) -> None:
        if on_die:
            # Pads sitting ON a filled die are separated by a carved COOL moat ring. Derive
            # the pitch from pad + a >=FLOOR moat so BOTH the hot pad AND the cool ring stay
            # >= 28um (no sub-pitch separation ring): a coarse, honestly-resolved bump field.
            # Higher density => smaller pads (toward the floor) => finer pitch => MORE bumps.
            pw = max(_lerp(det * 3.4, det * 2.2) * float(rng.uniform(0.85, 1.15)), pad_floor)
            ph = max(pw * float(rng.uniform(0.9, 1.12)), pad_floor)
            px = pw + 2.0 * FLOOR
            py = ph + 2.0 * FLOOR
        else:
            # Free-standing lattice on cool background: fixed floored pitch band; pad element
            # >= 28um. Denser scenes lean toward the tighter (but always >= ~34um) pitch.
            px = float(rng.uniform(p_lo, p_hi))
            py = max(px * float(rng.uniform(0.9, 1.12)), pitch_floor)
            fr = float(rng.uniform(0.44, 0.62))
            pw = max(px * fr, pad_floor)
            ph = max(py * fr, pad_floor)
        nc_g = int(np.clip(round(rw / px), 2, ncap))
        nr_g = int(np.clip(round(rh / py), 2, ncap))
        shape = "round" if rng.random() < 0.5 else "square"
        stagger = bool(shape == "round" and rng.random() < 0.35)
        drop_p = float(rng.uniform(0.0, 0.10)) if rng.random() < 0.5 else 0.0
        present = _present_mask(rng, nr_g, nc_g, drop_p, keepout)
        pads = make_pad_grid(nr_g, nc_g, px, py, pw, ph, gx, gy, shape=shape,
                             stagger=stagger, present=present, **common)
        if clip is not None:  # conform the grid to a non-rectangular die body
            pads = (pads.astype(bool) & clip.astype(bool)).astype(np.uint8)
            if not pads.any():
                return
        _add(pads)
        if on_die:
            # Carve the cool moat ring so each pad is a SEPARATE connected component (own
            # isothermal level => small inter-pad / pad-vs-die ΔT), honouring the per-
            # component rule. moat == FLOOR (28um) and the pitch was built as pad + 2*moat,
            # so the ring is a fully-resolved >= 28um cool gap (no sub-pitch separation).
            moat = FLOOR
            big = make_pad_grid(nr_g, nc_g, px, py, pw + 2.0 * moat, ph + 2.0 * moat,
                                gx, gy, shape=shape, stagger=stagger,
                                present=present, **common)
            _sub((big.astype(bool) & ~pads.astype(bool)).astype(np.uint8))
        primitives.append({
            "type": tag, "cx_um": gx, "cy_um": gy, "n_rows": nr_g, "n_cols": nc_g,
            "pitch_x_um": px, "pitch_y_um": py, "pad_w_um": pw, "pad_h_um": ph,
            "pad_shape": shape, "stagger": stagger, "drop_prob": round(drop_p, 3),
            "keepout": bool(keepout), "on_die": bool(on_die),
        })

    def _draw_body(bx: float, by: float, w: float, h: float, *,
                   tag: str, allow_nonrect: bool = True) -> tuple[np.ndarray, str]:
        """Draw a FILLED hot part/die body (Fix B: mix rectangles with round,
        rounded-rect, octagon/hexagon, ellipse and irregular-boundary outlines).

        Returns ``(body_mask, shape_name)``; the body is added to the scene mask
        and a primitive recorded. All bodies stay filled-hot; internal structure
        is layered on afterwards by ``_enrich_die``."""
        span = min(w, h)
        shape = "rect"
        if allow_nonrect and span > pitch_floor * 2.5 and rng.random() < 0.42:
            s = float(rng.random())
            if s < 0.22:
                cr = span * float(rng.uniform(0.15, 0.35))
                body = make_rounded_rect(bx, by, w, h, cr, **common)
                shape = "rounded_rect"
            elif s < 0.42:
                body = make_regular_polygon(bx, by, 0.5 * span * float(rng.uniform(0.98, 1.08)),
                                            8, rotation_deg=float(rng.uniform(0.0, 45.0)), **common)
                shape = "octagon"
            elif s < 0.58:
                body = make_regular_polygon(bx, by, 0.5 * span * float(rng.uniform(1.0, 1.12)),
                                            6, rotation_deg=float(rng.uniform(0.0, 60.0)), **common)
                shape = "hexagon"
            elif s < 0.74:
                body = make_circle_pad(bx, by, span * float(rng.uniform(0.9, 1.02)), **common)
                shape = "disc"
            elif s < 0.88:
                body = make_ellipse_pad(bx, by, w * float(rng.uniform(0.9, 1.02)),
                                        h * float(rng.uniform(0.9, 1.02)),
                                        angle_deg=float(rng.uniform(0.0, 180.0)), **common)
                shape = "ellipse"
            else:
                body = make_regular_polygon(bx, by, 0.5 * span * float(rng.uniform(1.0, 1.15)),
                                            int(rng.integers(6, 11)),
                                            rotation_deg=float(rng.uniform(0.0, 60.0)),
                                            vertex_jitter=float(rng.uniform(0.14, 0.30)),
                                            rng=rng, **common)
                shape = "irregular"
        else:
            body = make_rectangle(bx, by, w, h, **common)
        _add(body)
        primitives.append({
            "type": tag, "cx_um": bx, "cy_um": by,
            "outer_w_um": w, "outer_h_um": h, "body_shape": shape, "filled": True,
        })
        return body, shape

    def _enrich_die(body: np.ndarray, ox: float, oy: float, w: float, h: float,
                    *, heavy: bool) -> None:
        """Layer legacy-vocabulary internal structure onto a FILLED die (Fix A).

        The mix (per the isothermal-per-component rule):
          - SUBTRACTIVE cool channels / frame cavities / L-cuts carve the interior
            (the die body stays ~one isothermal connected component);
          - MOATED sub-blocks / pillars-bars / crosses / framed panels / a fine pad
            grid become SEPARATE components (own isothermal level, small inter-
            component ΔT);
          - ATTACHED thick/thin L-traces share the die temperature (routing fingers
            that may extend off the die edge).
        ~40-50% of features get a twin mirrored across the die centre."""
        span = min(w, h)
        if span < pitch_floor * 3.0:
            return
        body_bool = body.astype(bool)
        moat = FLOOR                                    # >= 28um cool separation ring

        def _dmir(x: float, y: float) -> tuple[float, float]:
            return (2.0 * ox - x, 2.0 * oy - y)

        def _place(feat: np.ndarray, grown: np.ndarray, meta: dict[str, object]) -> None:
            f = feat.astype(bool) & body_bool          # keep the island on the die
            if not f.any():
                return
            _add(f.astype(np.uint8))
            _sub((grown.astype(bool) & ~f).astype(np.uint8))   # thin moat => own component
            primitives.append(meta)

        # ── separate (moated) hot components: own isothermal level ──────────
        def _emit_block(px: float, py: float, bw: float, bh: float) -> None:
            _place(make_rectangle(px, py, bw, bh, **common),
                   make_rectangle(px, py, bw + 2.0 * moat, bh + 2.0 * moat, **common),
                   {"type": "die_subblock", "cx_um": px, "cy_um": py, "w_um": bw, "h_um": bh})

        def _sub_blocks() -> None:
            for _ in range(_dcount(1, 4)):
                if rng.random() < 0.5:                 # compact solid sub-block
                    bw = w * float(rng.uniform(0.10, 0.28))
                    bh = h * float(rng.uniform(0.10, 0.28))
                else:                                  # medium pillar / bar
                    if rng.random() < 0.5:
                        bw = max(line * float(rng.uniform(2.0, 5.0)), FLOOR)
                        bh = h * float(rng.uniform(0.30, 0.60))
                    else:
                        bw = w * float(rng.uniform(0.30, 0.60))
                        bh = max(line * float(rng.uniform(2.0, 5.0)), FLOOR)
                px = ox + float(rng.uniform(-0.26, 0.26)) * w
                py = oy + float(rng.uniform(-0.26, 0.26)) * h
                _emit_block(px, py, bw, bh)
                if rng.random() < sym_p:
                    _emit_block(*_dmir(px, py), bw, bh)

        def _emit_cross(px: float, py: float, aw: float, al: float) -> None:
            _place(make_cross(px, py, aw, al, **common),
                   make_cross(px, py, aw + 2.0 * moat, al + 2.0 * moat, **common),
                   {"type": "die_cross", "cx_um": px, "cy_um": py, "arm_w_um": aw, "arm_l_um": al})

        def _crosses() -> None:
            for _ in range(_dcount(1, 4)):
                aw = max(line * float(rng.uniform(0.9, 1.5)), FLOOR)
                al = span * float(rng.uniform(0.10, 0.24))
                px = ox + float(rng.uniform(-0.28, 0.28)) * w
                py = oy + float(rng.uniform(-0.28, 0.28)) * h
                _emit_cross(px, py, aw, al)
                if rng.random() < sym_p:
                    _emit_cross(*_dmir(px, py), aw, al)

        def _pad_texture() -> None:                    # fine pad grid = ONE texture option
            gw = w * float(rng.uniform(0.45, 0.85))
            gh = h * float(rng.uniform(0.45, 0.85))
            gx = ox + float(rng.uniform(-0.08, 0.08)) * w
            gy = oy + float(rng.uniform(-0.08, 0.08)) * h
            _add_grid(gx, gy, gw, gh, tag="die_pad_grid", on_die=True, clip=body)

        # ── framed panel: carve a ring => isolated inner panel (有空心有实心) ──
        def _emit_framed(px: float, py: float, ow: float, oh: float) -> None:
            border = max(line * float(rng.uniform(0.9, 1.8)), FLOOR)
            iw = max(ow - 2.0 * border, ow * 0.4)
            ih = max(oh - 2.0 * border, oh * 0.4)
            ring = make_frame(px, py, ow, oh, iw, ih, **common)
            _sub((ring.astype(bool) & body_bool).astype(np.uint8))
            primitives.append({"type": "die_framed_panel", "cx_um": px, "cy_um": py,
                               "outer_w_um": ow, "outer_h_um": oh, "border_um": border})

        def _framed_panels() -> None:
            ow = w * float(rng.uniform(0.30, 0.55))
            oh = h * float(rng.uniform(0.30, 0.55))
            px = ox + float(rng.uniform(-0.14, 0.14)) * w
            py = oy + float(rng.uniform(-0.14, 0.14)) * h
            _emit_framed(px, py, ow, oh)
            if rng.random() < sym_p:
                _emit_framed(*_dmir(px, py), ow, oh)

        # ── subtractive cool interior: channels, cavities, L-cuts ───────────
        def _frame_cavity() -> None:                   # small hollow well(s): occasional accent
            # SEVERAL SMALL wells (each <= ~18% of the die, spread out), never one big punched
            # hole: the die's "空心" character should read as fine interlock + thin frames, not
            # large black cavities (was 1-2 wells at 12-30% => the round3 punched-hole look).
            for _ in range(_dcount(2, 4)):
                iw = max(w * float(rng.uniform(0.07, 0.18)), FLOOR)
                ih = max(h * float(rng.uniform(0.07, 0.18)), FLOOR)
                px = ox + float(rng.uniform(-0.26, 0.26)) * w
                py = oy + float(rng.uniform(-0.26, 0.26)) * h
                _sub(make_rectangle(px, py, iw, ih, **common))
                primitives.append({"type": "die_cavity", "cx_um": px, "cy_um": py,
                                   "w_um": iw, "h_um": ih})
                if rng.random() < sym_p:
                    _sub(make_rectangle(*_dmir(px, py), iw, ih, **common))

        def _hv_channels() -> None:
            for _ in range(_dcount(2, 5)):
                cw = max(line * float(rng.uniform(0.9, 1.5)), FLOOR)
                if rng.random() < 0.5:                 # horizontal cool channel
                    cyc = oy + float(rng.uniform(-0.40, 0.40)) * h
                    clen = w * float(rng.uniform(0.45, 0.92))
                    cxc = ox + float(rng.uniform(-0.08, 0.08)) * w
                    _sub(make_rectangle(cxc, cyc, clen, cw, **common))
                    primitives.append({"type": "die_channel_h", "cx_um": cxc, "cy_um": cyc,
                                       "len_um": clen, "w_um": cw})
                    if rng.random() < 0.5:
                        _sub(make_rectangle(cxc, 2.0 * oy - cyc, clen, cw, **common))
                else:                                  # vertical cool channel
                    cxc = ox + float(rng.uniform(-0.40, 0.40)) * w
                    clen = h * float(rng.uniform(0.45, 0.92))
                    cyc = oy + float(rng.uniform(-0.08, 0.08)) * h
                    _sub(make_rectangle(cxc, cyc, cw, clen, **common))
                    primitives.append({"type": "die_channel_v", "cx_um": cxc, "cy_um": cyc,
                                       "len_um": clen, "w_um": cw})
                    if rng.random() < 0.5:
                        _sub(make_rectangle(2.0 * ox - cxc, cyc, cw, clen, **common))

        def _l_cut(x1: float, y1: float, x2: float, y2: float, cw: float) -> None:
            corner_x, corner_y = x2, y1
            seg_h_w = abs(corner_x - x1) + cw
            if seg_h_w > cw * 1.5:
                _sub(make_rectangle((x1 + corner_x) / 2.0, corner_y, seg_h_w, cw, **common))
            seg_v_h = abs(y2 - corner_y) + cw
            if seg_v_h > cw * 1.5:
                _sub(make_rectangle(corner_x, (corner_y + y2) / 2.0, cw, seg_v_h, **common))

        def _l_channels() -> None:
            for _ in range(_dcount(1, 3)):
                cw = max(line * float(rng.uniform(0.9, 1.5)), FLOOR)
                x1 = ox + float(rng.uniform(-0.35, 0.35)) * w
                y1 = oy + float(rng.uniform(-0.35, 0.35)) * h
                x2 = ox + float(rng.uniform(-0.35, 0.35)) * w
                y2 = oy + float(rng.uniform(-0.35, 0.35)) * h
                _l_cut(x1, y1, x2, y2, cw)
                primitives.append({"type": "die_channel_l", "x1_um": x1, "y1_um": y1,
                                   "x2_um": x2, "y2_um": y2, "w_um": cw})
                if rng.random() < sym_p:
                    _l_cut(*_dmir(x1, y1), *_dmir(x2, y2), cw)

        # ── attached routing traces (share die temperature) ─────────────────
        def _l_add(x1: float, y1: float, x2: float, y2: float, tw: float) -> None:
            corner_x, corner_y = x2, y1
            seg_h_w = abs(corner_x - x1) + tw
            if seg_h_w > tw * 1.5:
                _add(make_rectangle((x1 + corner_x) / 2.0, corner_y, seg_h_w, tw, **common))
            seg_v_h = abs(y2 - corner_y) + tw
            if seg_v_h > tw * 1.5:
                _add(make_rectangle(corner_x, (corner_y + y2) / 2.0, tw, seg_v_h, **common))

        def _l_traces() -> None:
            for _ in range(_dcount(2, 5)):
                tw = (line * float(rng.uniform(1.5, 3.2)) if rng.random() < 0.30
                      else max(line * float(rng.uniform(0.9, 1.4)), FLOOR))
                x1 = ox + float(rng.uniform(-0.28, 0.28)) * w   # anchored on the die
                y1 = oy + float(rng.uniform(-0.28, 0.28)) * h
                x2 = ox + float(rng.uniform(-0.60, 0.60)) * w   # may stick out as a finger
                y2 = oy + float(rng.uniform(-0.60, 0.60)) * h
                _l_add(x1, y1, x2, y2, tw)
                primitives.append({"type": "die_l_trace", "x1_um": x1, "y1_um": y1,
                                   "x2_um": x2, "y2_um": y2, "w_um": tw})
                if rng.random() < sym_p:
                    _l_add(*_dmir(x1, y1), *_dmir(x2, y2), tw)

        # Fine interlocking routing (thin cool channels + the _carve_routing comb) is the die's
        # DOMINANT internal texture, matching the real anchor. Big hollow wells are demoted to a
        # rare accent (below); hollow FRAMES (make_frame rings) carry most of the "空心" character.
        sub_menu = [lambda: _carve_routing(ox, oy, w, h), _hv_channels, _l_channels]
        comp_menu = [_sub_blocks, _crosses, _pad_texture, _framed_panels]

        ks = _dcount(1, len(sub_menu)) + (1 if heavy else 0)   # fine-routing carves scale with density
        for i in rng.choice(len(sub_menu), size=min(ks, len(sub_menu)), replace=False):
            sub_menu[int(i)]()
        kc = _dcount(1, len(comp_menu)) if heavy else _dcount(0, 2)
        for i in rng.choice(len(comp_menu), size=min(kc, len(comp_menu)), replace=False):
            comp_menu[int(i)]()
        if rng.random() < _lerp(0.3, 0.7):   # hollow frame ring => thin-framed "空心", not holes
            _framed_panels()
        if rng.random() < _lerp(0.08, 0.22):  # big hollow wells: occasional accent, denser => more
            _frame_cavity()
        if heavy and rng.random() < _lerp(0.5, 0.85):
            _l_traces()

    def _add_outline(ox: float, oy: float, w: float, h: float) -> np.ndarray:
        # Filled, near-isothermal die/package BODY (round-2 fix) — now with a
        # diversified outline (Fix B) and a legacy-style internal mix (Fix A) of
        # hollow+solid blocks, thick/thin L-traces, crosses and carved channels.
        body, _shape = _draw_body(ox, oy, w, h, tag="die_body")
        _enrich_die(body, ox, oy, w, h, heavy=True)
        return body

    def _add_border_ring(ox: float, oy: float, w: float, h: float) -> None:
        # Thin package/base-plate edge ring AROUND an already-populated interior
        # (heat-spreader fins). NOT the empty-die bug: the interior is filled by the
        # fin field, so this stays an honest package boundary, not a hollow frame.
        border = line * float(rng.uniform(1.5, 3.0))
        iw = max(w - 2.0 * border, w * 0.5)
        ih = max(h - 2.0 * border, h * 0.5)
        _add(make_frame(ox, oy, w, h, iw, ih, **common))
        primitives.append({
            "type": "spreader_ring", "cx_um": ox, "cy_um": oy,
            "outer_w_um": w, "outer_h_um": h, "border_um": border,
        })

    def _add_die(dx: float, dy: float, w: float, h: float) -> np.ndarray:
        # Filled hot die (diversified outline) with a lighter internal mix of
        # carved routing / cavities and a moated sub-block or two.
        body, _shape = _draw_body(dx, dy, w, h, tag="die")
        _enrich_die(body, dx, dy, w, h, heavy=False)
        return body

    def _add_passive(qx: float, qy: float) -> None:
        # Two-terminal passive: both terminals AND the cool gap between them stay >= 28um.
        pl = max(minor * float(rng.uniform(0.6, 1.1)), FLOOR)          # terminal length
        pw = max(pl * float(rng.uniform(0.55, 0.9)), FLOOR)           # terminal width
        cool = max(pw * float(rng.uniform(0.5, 1.4)), FLOOR)         # cool inter-terminal gap
        gap = pw + cool                                              # centre-to-centre spacing
        bar = max(pw * 0.5, FLOOR)                                   # connecting bar width
        if rng.random() < 0.5:
            _add(make_rectangle(qx - gap / 2.0, qy, pw, pl, **common))
            _add(make_rectangle(qx + gap / 2.0, qy, pw, pl, **common))
            if rng.random() < 0.5:
                _add(make_rectangle(qx, qy, gap, bar, **common))
        else:
            _add(make_rectangle(qx, qy - gap / 2.0, pl, pw, **common))
            _add(make_rectangle(qx, qy + gap / 2.0, pl, pw, **common))
            if rng.random() < 0.5:
                _add(make_rectangle(qx, qy, bar, gap, **common))
        primitives.append({"type": "passive", "cx_um": qx, "cy_um": qy})

    def _add_bus(bx: float, by: float, extent: float, direction: str) -> None:
        n = _dcount(4, 12)                                            # trace count grows with density
        tw = max(line * float(rng.uniform(0.9, 1.6)), FLOOR)         # trace width >= 28um
        bpitch = max(tw * float(rng.uniform(2.2, 3.6)), pitch_floor)  # pitch >= 32um => gap >= 28um
        tlen = extent * float(rng.uniform(0.5, 0.95))
        spine = max(line * float(rng.uniform(1.2, 2.6)), FLOOR) if rng.random() < 0.6 else 0.0
        _add(make_trace_bus(bx, by, n, tw, bpitch, tlen, direction=direction,
                            spine_w_um=spine, **common))
        primitives.append({
            "type": "trace_bus", "cx_um": bx, "cy_um": by, "n_traces": n,
            "trace_w_um": tw, "pitch_um": bpitch, "len_um": tlen,
            "direction": direction, "spine_w_um": spine,
        })

    def _add_fins(fx: float, fy: float, w: float, h: float) -> None:
        vertical = rng.random() < 0.5
        fw = max(line * float(rng.uniform(1.0, 2.2)), FLOOR)         # fin width >= 28um
        fpitch = max(fw * float(rng.uniform(2.0, 3.2)), pitch_floor)  # pitch >= 32um => gap >= 28um
        if vertical:
            n = int(np.clip(round(w / fpitch), 3, 48))
            fl = h * float(rng.uniform(0.6, 0.95))
            _add(make_pin_array(n, fpitch, fw, fl, fx, fy, "vertical", **common))
        else:
            n = int(np.clip(round(h / fpitch), 3, 48))
            fl = w * float(rng.uniform(0.6, 0.95))
            _add(make_pin_array(n, fpitch, fw, fl, fx, fy, "horizontal", **common))
        primitives.append({
            "type": "fins", "cx_um": fx, "cy_um": fy, "n": n, "fin_w_um": fw,
            "pitch_um": fpitch, "direction": "vertical" if vertical else "horizontal",
        })

    family = _pick_weighted(rng, motif_weights)

    # Part FOOTPRINT is held constant-large (die/body ~50-72% of canvas) INDEPENDENT of
    # density; density only changes what's INSIDE / how many. No tiny sparse "easy" parts.
    if family == "pga_grid":
        on_die = rng.random() < 0.6
        body = None
        if on_die:
            ow = canvas_w_um * float(rng.uniform(0.60, 0.72))
            oh = canvas_h_um * float(rng.uniform(0.58, 0.72))
            body = _add_outline(cx, cy, ow, oh)  # filled die body under the pin field
            rw, rh = ow * 0.82, oh * 0.82
        else:
            rw = canvas_w_um * float(rng.uniform(0.58, 0.72))
            rh = canvas_h_um * float(rng.uniform(0.56, 0.72))
        _add_grid(
            cx + canvas_w_um * float(rng.uniform(-0.04, 0.04)),
            cy + canvas_h_um * float(rng.uniform(-0.04, 0.04)),
            rw, rh, keepout=(rng.random() < 0.55), tag="pga_grid", on_die=on_die,
            clip=body,
        )
        for _ in range(_dcount(0, 2)):  # secondary arrays scale with density
            gx, gy = _rpos(0.15, 0.15)
            _add_grid(gx, gy, canvas_w_um * 0.18, canvas_h_um * 0.18, tag="bga_grid")

    elif family == "die_bga":
        ow = canvas_w_um * float(rng.uniform(0.52, 0.70))
        oh = canvas_h_um * float(rng.uniform(0.50, 0.68))
        ox = cx + canvas_w_um * float(rng.uniform(-0.10, 0.10))
        oy = cy + canvas_h_um * float(rng.uniform(-0.10, 0.10))
        body = _add_outline(ox, oy, ow, oh)  # filled hot die with internal routing
        _add_grid(ox, oy, ow * 0.80, oh * 0.80, keepout=(rng.random() < 0.30),
                  tag="bga_grid", on_die=True, clip=body)  # BGA pads on top, moated => separate
        for _ in range(_dcount(0, 3)):  # I/O pad rows outside the package
            gx, gy = _rpos(0.10, 0.10)
            _add_grid(gx, gy, canvas_w_um * 0.14, canvas_h_um * 0.06, tag="io_pads")

    elif family == "multi_die":
        for _ in range(_dcount(2, 5)):  # consistent count, moderate non-tiny dies
            dx, dy = _rpos(0.16, 0.16)
            dw = canvas_w_um * float(rng.uniform(0.16, 0.30))
            dh = canvas_h_um * float(rng.uniform(0.14, 0.28))
            if rng.random() < 0.45:
                _add_die(dx, dy, dw, dh)
            else:
                body = _add_outline(dx, dy, dw, dh)  # filled die body
                if rng.random() < 0.7:
                    _add_grid(dx, dy, dw * 0.78, dh * 0.78, tag="bga_grid",
                              on_die=True, clip=body)
            if rng.random() < sym_p:
                side = float(rng.choice([-1.0, 1.0]))
                _add_bus(dx + dw * 0.6 * side, dy, min(dw, dh),
                         str(rng.choice(["horizontal", "vertical"])))

    elif family == "trace_bus":
        dx, dy = _rpos(0.20, 0.20)
        dw = canvas_w_um * float(rng.uniform(0.20, 0.34))
        dh = canvas_h_um * float(rng.uniform(0.18, 0.30))
        (_add_die if rng.random() < 0.5 else _add_outline)(dx, dy, dw, dh)
        for _ in range(_dcount(1, 4)):  # bus count scales with density
            bx, by = _rpos(0.15, 0.15)
            _add_bus(bx, by, canvas_w_um * float(rng.uniform(0.20, 0.50)),
                     str(rng.choice(["horizontal", "vertical"])))
        if rng.random() < _lerp(0.4, 0.8):
            gx, gy = _rpos(0.15, 0.15)
            _add_grid(gx, gy, canvas_w_um * 0.22, canvas_h_um * 0.08, tag="io_pads")

    elif family == "heat_spreader":
        ow = canvas_w_um * float(rng.uniform(0.52, 0.70))
        oh = canvas_h_um * float(rng.uniform(0.50, 0.68))
        ox = cx + canvas_w_um * float(rng.uniform(-0.08, 0.08))
        oy = cy + canvas_h_um * float(rng.uniform(-0.08, 0.08))
        if rng.random() < 0.6:
            _add_border_ring(ox, oy, ow, oh)  # base-plate edge ring around the fins
        _add_fins(ox, oy, ow * 0.85, oh * 0.85)
        if rng.random() < 0.5:  # a die block to one side of the lid
            sx, sy = _rpos(0.12, 0.12)
            _add_die(sx, sy, canvas_w_um * float(rng.uniform(0.12, 0.22)),
                     canvas_h_um * float(rng.uniform(0.10, 0.20)))

    else:  # "generic" — minority diversity: plain blocks + L traces + passives
        for _ in range(_dcount(2, 4)):
            bx, by = _rpos(0.08, 0.08)
            _add_die(bx, by, canvas_w_um * float(rng.uniform(0.14, 0.30)),
                     canvas_h_um * float(rng.uniform(0.12, 0.26)))
        for _ in range(_dcount(3, 8)):
            x1, y1 = _rpos(0.05, 0.05)
            x2, y2 = _rpos(0.05, 0.05)
            tw = max(line * float(rng.uniform(0.9, 1.6)), FLOOR)
            corner_x, corner_y = x2, y1
            seg_h_w = abs(corner_x - x1) + tw
            if seg_h_w > tw * 1.5:
                _add(make_rectangle((x1 + corner_x) / 2.0, corner_y, seg_h_w, tw, **common))
            seg_v_h = abs(y2 - corner_y) + tw
            if seg_v_h > tw * 1.5:
                _add(make_rectangle(corner_x, (corner_y + y2) / 2.0, tw, seg_v_h, **common))
            primitives.append({"type": "trace_l", "x1_um": x1, "y1_um": y1,
                               "x2_um": x2, "y2_um": y2, "w_um": tw})

    # ── Shared part clutter (all families) — passives, vias, edge I/O ────────
    # Counts scale with density; every element stays >= 28um / pitch >= 32um.
    for _ in range(_dcount(0, 3)):
        qx, qy = _rpos(0.06, 0.06)
        _add_passive(qx, qy)
    for _ in range(_dcount(0, 4)):
        vx, vy = _rpos(0.05, 0.05)
        d = max(line * float(rng.uniform(1.2, 3.0)), FLOOR)
        _add(make_circle_pad(vx, vy, d, **common))
        primitives.append({"type": "via", "cx_um": vx, "cy_um": vy, "diameter_um": d})
    if rng.random() < _lerp(0.3, 0.7):
        edge = int(rng.integers(0, 4))
        n = int(rng.choice([8, 10, 12, 16, 20]))
        sp = max(float(rng.uniform(p_lo, p_hi)) * 0.9, pitch_floor)  # pin pitch >= 32um
        pw = max(line * float(rng.uniform(0.9, 1.6)), FLOOR)         # pin width >= 28um
        pl = max(minor * float(rng.uniform(1.0, 2.2)), FLOOR)
        if edge == 0:
            _add(make_pin_array(n, sp, pw, pl, cx, 0.05 * canvas_h_um, "horizontal", **common))
        elif edge == 1:
            _add(make_pin_array(n, sp, pw, pl, cx, 0.95 * canvas_h_um, "horizontal", **common))
        elif edge == 2:
            _add(make_pin_array(n, sp, pw, pl, 0.05 * canvas_w_um, cy, "vertical", **common))
        else:
            _add(make_pin_array(n, sp, pw, pl, 0.95 * canvas_w_um, cy, "vertical", **common))
        primitives.append({"type": "edge_io", "edge": int(edge), "n_pins": n})

    mask = mask & ~subtract
    return mask, primitives, family, density


def _finalize_scene_mask(
    mask: np.ndarray,
    primitives: list[dict[str, object]],
    scene_family: str,
    *,
    rng: np.random.Generator,
    rotation_deg_center: float,
    rotation_jitter_deg: float,
    antialias: bool,
    aa_factor: int,
    inscribe_disc: bool,
    draw_shape: tuple[int, int],
    difficulty: str,
    density: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Shared tail: optional disc inscribe, rotate + AA-downsample, build metadata."""

    if bool(inscribe_disc):
        draw_rows, draw_cols = draw_shape
        yy_d, xx_d = np.mgrid[:draw_rows, :draw_cols]
        radius = min(draw_rows, draw_cols) / 2.0
        disc = ((yy_d - draw_rows / 2.0) ** 2 + (xx_d - draw_cols / 2.0) ** 2) <= radius ** 2
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
        "density": None if density is None else float(density),
        "n_primitives": len(primitives),
        "primitives": primitives,
        "antialias": bool(antialias),
        "ssaa_factor": aa_factor,
        "inscribe_disc": bool(inscribe_disc),
        "mask_semantics": "coverage" if bool(antialias) else "binary",
        "scene_family": scene_family,
        "implementation": "tcforge.geometry.build_scene_mask_cpu_family",
    }
    if bool(antialias):
        rotated = _rotate_coverage_mask(mask, angle, order=1)
        return _downsample_coverage(rotated, aa_factor), metadata
    return rotate_mask(mask, angle, order=0), metadata


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
    motif_weights: dict[str, float] | None = None,
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

    # CPU / electronic-part geometry family (v6): when motif weights are supplied
    # the scene is composed from part motifs (die outlines, fine pad grids, buses,
    # fins) instead of the legacy generic IC layout. Absent => legacy path below,
    # byte-identical to prior behaviour.
    if motif_weights is not None:
        cpu_mask, cpu_primitives, scene_family, density = _compose_cpu_scene(
            rng,
            motif_weights=motif_weights,
            common=common,
            draw_shape=draw_shape,
            pixel_size_um=float(pixel_size_um),
            canvas_w_um=canvas_w_um,
            canvas_h_um=canvas_h_um,
            cx=cx,
            cy=cy,
            detector_pitch_um=float(pixel_size_um),
        )
        return _finalize_scene_mask(
            cpu_mask,
            cpu_primitives,
            scene_family,
            rng=rng,
            rotation_deg_center=rotation_deg_center,
            rotation_jitter_deg=rotation_jitter_deg,
            antialias=bool(antialias),
            aa_factor=aa_factor,
            inscribe_disc=bool(inscribe_disc),
            draw_shape=draw_shape,
            difficulty=difficulty,
            density=density,
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
