"""thermal_core.io — 数据读取、解析、审计工具

将 Notebook 中反复出现的数据加载/解析/统计逻辑封装于此，
使 Notebook 片段只需一行调用 + 结果展示。
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 文件名解析
# ──────────────────────────────────────────────

def parse_filename(fname: str) -> tuple[int, int, int] | None:
    """从 X_Y_R.ext 格式解析坐标，返回 (X, Y, R) 或 None。"""
    parts = Path(fname).stem.split("_")
    if len(parts) == 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return None


# ──────────────────────────────────────────────
# 逐帧读取 + 审计
# ──────────────────────────────────────────────

def load_frame(filepath: Path) -> np.ndarray:
    """读取单帧 TXT 温度矩阵，返回 ndarray。

    使用 pd.read_csv 来正确处理行尾多余 tab。
    """
    data = pd.read_csv(filepath, delimiter='\t', header=None)
    return data.dropna(axis=1, how='all').values


def audit_all_frames(data_dir: Path) -> pd.DataFrame:
    """扫描 data_dir 下全部 TXT，返回逐帧审计 DataFrame。

    列: file, X, Y, R, rows, cols, has_nan, has_inf,
        T_min, T_max, T_mean, T_std, mtime, filename_order, acquisition_order
    """
    txt_files = sorted(data_dir.glob("*.txt"))
    results = []
    errors = []

    for f in txt_files:
        coord = parse_filename(f.name)
        if coord is None:
            errors.append({"file": f.name, "error": "无法解析文件名"})
            continue
        x, y, r = coord
        try:
            data = load_frame(f)
            rows, cols = data.shape
            results.append({
                "file": f.name, "X": x, "Y": y, "R": r,
                "mtime": float(f.stat().st_mtime),
                "rows": rows, "cols": cols,
                "has_nan": bool(np.any(np.isnan(data))),
                "has_inf": bool(np.any(np.isinf(data))),
                "T_min": float(np.nanmin(data)),
                "T_max": float(np.nanmax(data)),
                "T_mean": float(np.nanmean(data)),
                "T_std": float(np.nanstd(data)),
            })
        except Exception as e:
            errors.append({"file": f.name, "error": str(e)})

    df = pd.DataFrame(results)
    if errors:
        print(f"⚠️ 读取失败 {len(errors)} 帧:")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")
    if df.empty:
        return df

    df = df.sort_values("file").reset_index(drop=True)
    df["filename_order"] = np.arange(len(df), dtype=int)
    df = df.merge(
        df[["file", "mtime"]]
        .sort_values(["mtime", "file"])
        .reset_index(drop=True)
        .assign(acquisition_order=lambda x: np.arange(len(x), dtype=int))
        [["file", "acquisition_order"]],
        on="file",
        how="left",
    )
    return df


# ──────────────────────────────────────────────
# BMP-TXT 配对
# ──────────────────────────────────────────────

def check_bmp_txt_pairing(data_dir: Path) -> dict:
    """检查 BMP/TXT 配对，返回统计字典。"""
    txt_stems = {f.stem for f in data_dir.glob("*.txt")}
    bmp_stems = {f.stem for f in data_dir.glob("*.bmp")}
    return {
        "n_txt": len(txt_stems),
        "n_bmp": len(bmp_stems),
        "n_paired": len(txt_stems & bmp_stems),
        "only_txt": sorted(txt_stems - bmp_stems),
        "only_bmp": sorted(bmp_stems - txt_stems),
    }


# ──────────────────────────────────────────────
# 坐标覆盖率
# ──────────────────────────────────────────────

def build_coord_repeat_map(df: pd.DataFrame) -> dict[tuple, set]:
    """从审计 DataFrame 构建 (X,Y) → {R 值集合} 映射。"""
    coords = defaultdict(set)
    for _, row in df.iterrows():
        coords[(int(row["X"]), int(row["Y"]))].add(int(row["R"]))
    return dict(coords)


def compute_coverage_grid(
    coord_map: dict[tuple, set],
    valid_coords: list[int],
) -> np.ndarray:
    """返回 (len(valid_coords), len(valid_coords)) 的 repeat-count 矩阵。"""
    vals = sorted(valid_coords)
    grid = np.zeros((len(vals), len(vals)), dtype=int)
    for i, y in enumerate(vals):
        for j, x in enumerate(vals):
            grid[i, j] = len(coord_map.get((x, y), set()))
    return grid


# ──────────────────────────────────────────────
# Session 检测
# ──────────────────────────────────────────────

def detect_sessions(
    df_sorted: pd.DataFrame,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按均温跳变检测 session 边界。

    Parameters
    ----------
    df_sorted : DataFrame
        已按目标时序排序、含 T_mean 列的审计 DataFrame。
    threshold : float, optional
        跳变阈值 (°C)。默认 max(median_jump * 10, 0.5)。

    Returns
    -------
    session_ids : ndarray
        每帧的 session ID。
    break_indices : ndarray
        断点索引。
    threshold : float
        实际使用的阈值。
    """
    t_means = df_sorted["T_mean"].values
    jumps = np.abs(np.diff(t_means))
    median_jump = float(np.median(jumps))
    if threshold is None:
        threshold = max(median_jump * 10, 0.5)

    break_indices = np.where(jumps > threshold)[0]
    session_ids = np.zeros(len(t_means), dtype=int)
    for i, brk in enumerate(break_indices):
        session_ids[brk + 1:] = i + 1

    return session_ids, break_indices, threshold
