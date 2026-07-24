#!/usr/bin/env python3
"""
Per-scene content-density audit for the v6 synthetic training pool
(data/synthetic/pool_2x_v6_5k). READ-ONLY: never writes into the pool dir.

For each scene computes:
  - occupancy: fraction of HR pixels with coverage > 0.5, from hr_mask_4x.png
    (mask_semantics == "coverage", quantized uint8 0-255 -> divide by 255).
  - mean_coverage: mean of the continuous coverage array (sanity companion
    to the thresholded occupancy).
  - connected-component stats (8-connectivity) on the binary (coverage>0.5)
    mask: n_components, median/max/mean component area (in HR pixels).
  - geometry_metadata fields from metadata.json: scene_family, n_primitives,
    density (self-reported generator density knob), defect counts/severity,
    rotation_deg, difficulty.

Runs in chunks (--start/--count) with worker-pool parallelism so a single
process can be killed/resumed without losing prior work. Results are
appended to a CSV; a summary.json is written separately once by
--finalize after all chunks are done (aggregates the CSV, does not
recompute per-scene work).

Usage:
  # one scene smoke test
  uv run python scripts/audit_v6_density.py --start 0 --count 1 --workers 1

  # chunked full run
  uv run python scripts/audit_v6_density.py --start 0 --count 5000 --workers 32

  # after all chunks landed in the CSV
  uv run python scripts/audit_v6_density.py --finalize

输入: data/synthetic/pool_2x_v6_5k（POOL_DIR 常量；每场景 hr_mask_4x.png + metadata.json）
输出: output/v6_density_audit/per_scene.csv（逐场景追加）、summary.json（--finalize 时生成）
"""
import argparse
import csv
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np

POOL_DIR = "data/synthetic/pool_2x_v6_5k"
OUT_DIR = "output/v6_density_audit"
CSV_PATH = os.path.join(OUT_DIR, "per_scene.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")

FIELDS = [
    "scene_index", "scene_id", "difficulty", "scene_family", "n_primitives",
    "geom_density", "defects_holes", "defects_notches", "defects_cracks",
    "defects_severity", "rotation_deg", "hr_h", "hr_w", "occupancy",
    "mean_coverage", "n_components", "median_component_area",
    "mean_component_area", "max_component_area", "status",
]


def process_scene(scene_index):
    """Returns a dict matching FIELDS, or a dict with status='error' + reason."""
    scene_id = f"scene_{scene_index:04d}"
    scene_dir = os.path.join(POOL_DIR, scene_id)
    row = {k: "" for k in FIELDS}
    row["scene_index"] = scene_index
    row["scene_id"] = scene_id
    try:
        from PIL import Image
        import scipy.ndimage as ndi

        meta_path = os.path.join(scene_dir, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        gm = meta.get("geometry_metadata", {})
        defects = gm.get("defects", {})

        row["difficulty"] = meta.get("difficulty", "")
        row["scene_family"] = gm.get("scene_family", "")
        row["n_primitives"] = gm.get("n_primitives", "")
        row["geom_density"] = gm.get("density", "")
        row["defects_holes"] = defects.get("holes", "")
        row["defects_notches"] = defects.get("notches", "")
        row["defects_cracks"] = defects.get("cracks", "")
        row["defects_severity"] = defects.get("severity", "")
        row["rotation_deg"] = meta.get("rotation_deg", "")

        mask_semantics = meta.get("storage", {}).get("hr_mask_semantics", "")
        if mask_semantics != "coverage":
            row["status"] = f"error:unexpected_mask_semantics={mask_semantics}"
            return row

        mask_path = os.path.join(scene_dir, "hr_mask_4x.png")
        arr = np.array(Image.open(mask_path))
        row["hr_h"], row["hr_w"] = arr.shape[0], arr.shape[1]
        cov = arr.astype(np.float32) / 255.0
        binm = cov > 0.5

        row["occupancy"] = float(binm.mean())
        row["mean_coverage"] = float(cov.mean())

        structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
        labeled, n_comp = ndi.label(binm, structure=structure)
        row["n_components"] = int(n_comp)
        if n_comp > 0:
            areas = ndi.sum(binm, labeled, index=np.arange(1, n_comp + 1))
            row["median_component_area"] = float(np.median(areas))
            row["mean_component_area"] = float(np.mean(areas))
            row["max_component_area"] = float(np.max(areas))
        else:
            row["median_component_area"] = 0.0
            row["mean_component_area"] = 0.0
            row["max_component_area"] = 0.0

        row["status"] = "ok"
    except Exception as e:
        row["status"] = f"error:{type(e).__name__}:{e}"
    return row


def run_chunk(start, count, workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    indices = list(range(start, start + count))
    write_header = not os.path.exists(CSV_PATH)

    t0 = time.time()
    results = []
    if workers <= 1:
        for i in indices:
            results.append(process_scene(i))
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap(process_scene, indices, chunksize=8):
                results.append(r)

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for r in results:
            w.writerow(r)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    dt = time.time() - t0
    print(f"[chunk start={start} count={count}] ok={n_ok} err={n_err} "
          f"elapsed={dt:.1f}s ({dt/max(1,count):.3f}s/scene)")
    if n_err:
        for r in results:
            if r["status"] != "ok":
                print("  ERROR", r["scene_id"], r["status"])


def finalize():
    import pandas as pd
    if not os.path.exists(CSV_PATH):
        print("No CSV found at", CSV_PATH)
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(subset="scene_index", keep="last")
    n_total = len(df)
    ok = df[df["status"] == "ok"]
    n_ok = len(ok)

    def deciles(s):
        return {f"p{p}": float(np.percentile(s, p)) for p in range(0, 101, 10)}

    hist_edges = np.arange(0, 1.0001, 0.05)
    hist_counts, _ = np.histogram(ok["occupancy"], bins=hist_edges)
    hist = {f"{hist_edges[i]:.2f}-{hist_edges[i+1]:.2f}": int(hist_counts[i])
            for i in range(len(hist_counts))}

    by_family = ok.groupby("scene_family")["occupancy"].agg(
        ["count", "median", "mean"]).to_dict(orient="index")

    summary = {
        "n_total_requested": n_total,
        "n_ok": n_ok,
        "n_error": n_total - n_ok,
        "sampling": "full population (no subsampling)" if n_total >= 5000
                    else f"PARTIAL: only {n_total}/5000 scenes processed",
        "occupancy_deciles": deciles(ok["occupancy"]),
        "occupancy_histogram_bin0.05": hist,
        "n_components_deciles": deciles(ok["n_components"]),
        "median_component_area_deciles": deciles(ok["median_component_area"]),
        "max_component_area_deciles": deciles(ok["max_component_area"]),
        "occupancy_by_scene_family": by_family,
        "geom_density_vs_occupancy_corr_pearson": float(
            np.corrcoef(ok["geom_density"], ok["occupancy"])[0, 1]),
        "n_primitives_vs_occupancy_corr_pearson": float(
            np.corrcoef(ok["n_primitives"], ok["occupancy"])[0, 1]),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    if args.finalize:
        finalize()
    else:
        if args.count <= 0:
            print("Need --count > 0 (or --finalize)")
            sys.exit(1)
        run_chunk(args.start, args.count, args.workers)
