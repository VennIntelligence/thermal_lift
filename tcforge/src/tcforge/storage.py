"""Compact scene storage for TCForge training data."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

COMPACT_SCENE_FILES: tuple[str, ...] = (
    "hr_mask_4x.png",
    "hr_edge_4x.png",
    "obs_features_1x.npz",
    "shifts.npy",
    "metadata.json",
)


def save_scene_compact(
    scene_dir: str | Path,
    *,
    hr_mask: np.ndarray,
    hr_edge: np.ndarray,
    obs_features: np.ndarray,
    shifts: np.ndarray,
    metadata: dict[str, Any],
    classical_sr: np.ndarray | None = None,
    lr_burst: np.ndarray | None = None,
) -> Path:
    """Save one compact scene without HR temperature arrays.

    If *classical_sr* is provided it is saved as ``classical_sr_{scale}x.npy``
    alongside the standard compact files. If *lr_burst* is provided it is
    saved as optional ``lr_burst.npy`` for deferred EP12 feature building.
    ``hr_mask`` may be binary or a soft coverage mask in ``[0, 1]``; it is
    quantized to 8-bit grayscale PNG and restored as float32 coverage.
    """

    root = Path(scene_dir)
    root.mkdir(parents=True, exist_ok=True)
    mask = _as_coverage(hr_mask, "hr_mask")
    edge = _as_binary(hr_edge, "hr_edge")
    features = _as_features(obs_features)
    shift_arr = _as_shifts(shifts)
    meta = dict(metadata)

    _write_png_gray8(root / "hr_mask_4x.png", _coverage_to_uint8(mask))
    _write_png_gray8(root / "hr_edge_4x.png", edge * np.uint8(255))
    np.savez_compressed(root / "obs_features_1x.npz", obs_features=features.astype(np.float16, copy=False))
    np.save(root / "shifts.npy", shift_arr.astype(np.float32, copy=False))
    if classical_sr is not None:
        csr = np.asarray(classical_sr, dtype=np.float32)
        if csr.ndim != 2:
            raise ValueError("classical_sr must be 2D")
        scale = int(meta.get("scale", 4))
        np.save(root / f"classical_sr_{scale}x.npy", csr)
    if lr_burst is not None:
        burst = np.asarray(lr_burst, dtype=np.float16)
        if burst.ndim != 3:
            raise ValueError("lr_burst must have shape (N, H_lr, W_lr)")
        if not np.isfinite(burst).all():
            raise ValueError("lr_burst contains NaN or Inf")
        np.save(root / "lr_burst.npy", burst)
    (root / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def load_scene_compact(scene_dir: str | Path) -> dict[str, Any]:
    """Load one compact scene directory.

    If a ``classical_sr_{scale}x.npy`` file exists it is included in the
    returned dict under ``"classical_sr"``; otherwise that key is absent.
    Optional EP12 feature-builder artifacts are loaded when present without
    changing the required compact scene contract.
    """

    root = Path(scene_dir)
    _validate_compact_files(root)
    with np.load(root / "obs_features_1x.npz") as data:
        if "obs_features" in data:
            obs_features = data["obs_features"]
        else:
            obs_features = data[data.files[0]]
    with (root / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    result: dict[str, Any] = {
        "scene_dir": root,
        "hr_mask": (_read_png_gray8(root / "hr_mask_4x.png").astype(np.float32) / 255.0).astype(np.float32),
        "hr_edge": (_read_png_gray8(root / "hr_edge_4x.png") > 0).astype(np.uint8),
        "obs_features": np.asarray(obs_features),
        "shifts": np.load(root / "shifts.npy").astype(np.float32, copy=False),
        "metadata": metadata,
    }
    scale = int(metadata.get("scale", 4))
    csr_path = root / f"classical_sr_{scale}x.npy"
    if csr_path.exists():
        result["classical_sr"] = np.load(csr_path).astype(np.float32, copy=False)
    lr_burst_path = root / "lr_burst.npy"
    if lr_burst_path.exists():
        result["lr_burst"] = np.load(lr_burst_path, mmap_mode="r")
    drizzle_variants_path = root / f"drizzle_variants_{scale}x.npy"
    if drizzle_variants_path.exists():
        result["drizzle_variants"] = np.load(drizzle_variants_path, mmap_mode="r")
    optional_features = {
        "obs_features_4x": "obs_features_4x.npz",
        "obs_features_2x_up4x": "obs_features_2x_up4x.npz",
        "obs_features_1x_up4x": "obs_features_1x_up4x.npz",
    }
    for key, name in optional_features.items():
        path = root / name
        if path.exists():
            result[key] = _load_feature_npz(path)
    return result


def _load_feature_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "obs_features" in data:
            features = data["obs_features"]
        elif "features" in data:
            features = data["features"]
        else:
            features = data[data.files[0]]
    return np.asarray(features)


def _validate_compact_files(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"scene_dir not found: {root}")
    missing = [name for name in COMPACT_SCENE_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} missing required compact scene files: {missing}")


def _as_binary(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if not np.isin(arr, [0, 1, False, True]).all():
        raise ValueError(f"{name} must be binary")
    return (arr > 0).astype(np.uint8, copy=False)


def _as_coverage(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise ValueError(f"{name} must be binary or soft coverage in [0, 1]")
    return arr.astype(np.float32, copy=False)


def _coverage_to_uint8(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    return np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _as_features(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("obs_features must have shape (C, H, W)")
    if not np.isfinite(arr).all():
        raise ValueError("obs_features contains NaN or Inf")
    return arr


def _as_shifts(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("shifts must have shape (N, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("shifts contain NaN or Inf")
    return arr


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png_gray8(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("PNG image must be 2D")
    rows, cols = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(rows))
    ihdr = struct.pack(">IIBBBBB", cols, rows, 8, 0, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", ihdr)
    payload += _png_chunk(b"IDAT", zlib.compress(raw))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _read_png_gray8(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    pos = 8
    width = height = None
    compressed_parts: list[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if bit_depth != 8 or color_type != 0 or compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("only non-interlaced 8-bit grayscale PNG is supported")
        elif kind == b"IDAT":
            compressed_parts.append(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG missing IHDR")
    raw = zlib.decompress(b"".join(compressed_parts))
    stride = width + 1
    if len(raw) != stride * height:
        raise ValueError("PNG payload size mismatch")
    out = np.empty((height, width), dtype=np.uint8)
    prev = np.zeros(width, dtype=np.uint8)
    for row in range(height):
        start = row * stride
        filter_type = raw[start]
        scan = np.frombuffer(raw[start + 1 : start + stride], dtype=np.uint8).copy()
        if filter_type == 0:
            recon = scan
        elif filter_type == 1:
            recon = _unfilter_sub(scan)
        elif filter_type == 2:
            recon = (scan.astype(np.uint16) + prev.astype(np.uint16)).astype(np.uint8)
        elif filter_type == 3:
            recon = _unfilter_average(scan, prev)
        elif filter_type == 4:
            recon = _unfilter_paeth(scan, prev)
        else:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
        out[row] = recon
        prev = recon
    return out


def _unfilter_sub(scan: np.ndarray) -> np.ndarray:
    out = scan.copy()
    for idx in range(1, out.size):
        out[idx] = (int(out[idx]) + int(out[idx - 1])) & 0xFF
    return out


def _unfilter_average(scan: np.ndarray, prev: np.ndarray) -> np.ndarray:
    out = scan.copy()
    for idx in range(out.size):
        left = int(out[idx - 1]) if idx > 0 else 0
        above = int(prev[idx])
        out[idx] = (int(out[idx]) + (left + above) // 2) & 0xFF
    return out


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_paeth(scan: np.ndarray, prev: np.ndarray) -> np.ndarray:
    out = scan.copy()
    for idx in range(out.size):
        left = int(out[idx - 1]) if idx > 0 else 0
        above = int(prev[idx])
        upper_left = int(prev[idx - 1]) if idx > 0 else 0
        out[idx] = (int(out[idx]) + _paeth_predictor(left, above, upper_left)) & 0xFF
    return out
