"""Binary chip-structure mask generators for ThermalChipPhantom."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage

DEFAULT_CANVAS_SHAPE = (960, 1280)
DEFAULT_PIXEL_SIZE_UM = 10.0
DEFAULT_SCALE = 2


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


def build_scene_mask(
    difficulty: str,
    seed: int,
    *,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
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
    )
    return mask


def build_scene_mask_with_metadata(
    difficulty: str,
    seed: int,
    *,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = DEFAULT_CANVAS_SHAPE,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    scale: int = DEFAULT_SCALE,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a deterministic full-scene mask and return generation metadata."""

    rows, cols = _as_shape(canvas_shape, "canvas_shape")
    pitch = _hr_pitch_um(pixel_size_um, scale)
    cx = cols * pitch / 2.0
    cy = rows * pitch / 2.0
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
    jitter = lambda span: rng.uniform(-span, span)
    primitive_names = ["frame", "pin_array_horizontal", "pin_array_vertical", "cross", "l_shape", "trenches"]
    parts: Iterable[np.ndarray] = (
        make_frame(
            cx,
            cy,
            8 * major,
            5 * major,
            5 * major,
            2.8 * major,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_pin_array(
            8,
            minor,
            line,
            2.5 * major,
            cx,
            cy - 2.2 * major,
            "horizontal",
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_pin_array(
            6,
            minor,
            line,
            2.0 * major,
            cx - 3.4 * major,
            cy,
            "vertical",
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_cross(
            cx + 2.2 * major + jitter(minor),
            cy + jitter(minor),
            line,
            2.2 * major,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_l_shape(
            cx + 1.5 * major,
            cy + 1.35 * major,
            line,
            2.1 * major,
            2.4 * major,
            line,
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
        make_trenches(
            cx,
            cy + 2.1 * major,
            line,
            5,
            minor,
            "vertical",
            canvas_shape=canvas_shape,
            pixel_size_um=pixel_size_um,
            scale=scale,
        ),
    )
    mask = composite(*parts, canvas_shape=canvas_shape)
    angle = float(rotation_deg_center) + float(rng.uniform(-rotation_jitter_deg, rotation_jitter_deg))
    metadata = {
        "units": "um",
        "rotation_deg": angle,
        "rotation_deg_center": float(rotation_deg_center),
        "rotation_jitter_deg": float(rotation_jitter_deg),
        "difficulty": difficulty,
        "primitives": primitive_names,
        "implementation": "tcforge.geometry.build_scene_mask",
    }
    return rotate_mask(mask, angle, order=0), metadata
