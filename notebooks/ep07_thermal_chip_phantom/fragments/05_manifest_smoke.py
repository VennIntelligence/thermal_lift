# %% [markdown]
# ## 4. Manifest、产物目录与 Smoke 验收
#
# 数据集生成器后台做了大量“看不见”的检查。这里把这些检查显式展开：哪些文件必须存在、每个文件给谁用、shape/dtype/finite 是否匹配、highpass 是否能用独立 scipy reference 复算。

# %%
metadata = {
    "schema_version": "0.1-demo",
    "dataset": "ThermalChipPhantom",
    "engine": "TCForge",
    "scene_id": DEMO_CONFIG.scene_id,
    "generator": "notebook_adaptive_demo",
    "tcforge_import_status": TCFORGE_VERSION,
    "scene_generation_mode": SCENE_GENERATION_MODE,
    "seed": DEMO_CONFIG.seed,
    "scale": DEMO_CONFIG.scale,
    "lr_shape": list(DEMO_CONFIG.lr_shape),
    "hr_shape": [DEMO_CONFIG.lr_shape[0] * DEMO_CONFIG.scale, DEMO_CONFIG.lr_shape[1] * DEMO_CONFIG.scale],
    "pixel_size_um": DEMO_CONFIG.pixel_size_um,
    "spatial_resolution_um": 20.0,
    "geometry": {
        "difficulty": "easy",
        "truth_files": ["hr_mask_2x.npy", "hr_temperature_2x.npy", "hr_edge_map_2x.npy"],
    },
    "physics": {
        "T_bg_c": DEMO_CONFIG.base_temp_c,
        "delta_T_c": DEMO_CONFIG.delta_temp_c,
        "noise_sigma_c": DEMO_CONFIG.noise_sigma_c,
        "psf_sigma_lr_px": DEMO_CONFIG.psf_sigma_lr_px,
        "forward_mode": FORWARD_MODE,
        "highpass_sigma_lr_px": DEMO_CONFIG.highpass_sigma_lr_px,
        "highpass_mode": "nearest",
        "drift_model": "none",
    },
    "shifts": {
        "source": SHIFT_SOURCE,
        "convention": "LR-to-reference alignment shift",
        "columns": ["dx_px", "dy_px"],
    },
    "provenance": {
        "notebook": relative(NOTEBOOK_DIR),
        "output_dir": relative(DEMO_DIR),
    },
}
(DEMO_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

artifact_catalog = pd.DataFrame(
    [
        {
            "file": "hr_mask_2x.npy",
            "producer": "geometry",
            "consumer": "SR metric / contour metric",
            "meaning": "sharp binary structure truth",
            "git": "ignored output",
        },
        {
            "file": "hr_temperature_2x.npy",
            "producer": "physics",
            "consumer": "temperature reconstruction metric",
            "meaning": "HR temperature truth",
            "git": "ignored output",
        },
        {
            "file": "hr_edge_map_2x.npy",
            "producer": "physics.edge_map",
            "consumer": "boundary/contour diagnostics",
            "meaning": "morphological contour proxy",
            "git": "ignored output",
        },
        {
            "file": "lr_burst_raw.npy",
            "producer": "forward + noise/drift",
            "consumer": "SR algorithm input",
            "meaning": "ordinary LR temperature burst",
            "git": "ignored output",
        },
        {
            "file": "lr_burst_highpass.npy",
            "producer": "EP06-compatible highpass",
            "consumer": "structure-oriented SR input / diagnostic",
            "meaning": "local structure response",
            "git": "ignored output",
        },
        {
            "file": "shifts.npy",
            "producer": "shift profile",
            "consumer": "synthetic forward/reconstruction control",
            "meaning": "LR-to-reference dx/dy in LR pixels",
            "git": "ignored output",
        },
        {
            "file": "metadata.json",
            "producer": "manifest layer",
            "consumer": "smoke/evaluate/reproducibility",
            "meaning": "physical contract and provenance",
            "git": "ignored output",
        },
        {
            "file": "manifest.csv",
            "producer": "manifest layer",
            "consumer": "batch scripts",
            "meaning": "flat scene/file index",
            "git": "ignored output",
        },
    ]
)
display(artifact_catalog)

# %% [markdown]
# > **数据说明**: 这张表列出 demo 数据包中的核心产物、生成模块、消费者和语义。它回答“这些文件到底有什么用”。
# >
# > **怎么看**: SR 算法通常消费 `lr_burst_raw.npy` 或 `lr_burst_highpass.npy`，评价时再对比 `hr_temperature_2x.npy`、`hr_mask_2x.npy` 或 `hr_edge_map_2x.npy`。
# >
# > **异常是否正常**: 所有这些都是 `output/` 下的可重建产物，不应提交 Git；应提交的是 generator 源码、notebook fragments、配置和报告。
# >
# > **核心发现**: EP07 的产物不是单张图片，而是一套带 metadata 和用途约束的数据包。

# %%
required = {
    "hr_temperature_2x.npy": tuple(metadata["hr_shape"]),
    "hr_mask_2x.npy": tuple(metadata["hr_shape"]),
    "hr_edge_map_2x.npy": tuple(metadata["hr_shape"]),
    "lr_burst_raw.npy": (DEMO_CONFIG.n_frames, *DEMO_CONFIG.lr_shape),
    "lr_burst_highpass.npy": (DEMO_CONFIG.n_frames, *DEMO_CONFIG.lr_shape),
    "shifts.npy": (DEMO_CONFIG.n_frames, 2),
}

manifest_rows = []
for filename, expected_shape in required.items():
    arr = np.load(DEMO_DIR / filename)
    manifest_rows.append(
        {
            "scene_id": DEMO_CONFIG.scene_id,
            "scene_dir": relative(DEMO_DIR),
            "file": filename,
            "expected_shape": "x".join(map(str, expected_shape)),
            "shape": "x".join(map(str, arr.shape)),
            "dtype": str(arr.dtype),
        }
    )
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(DEMO_DIR / "manifest.csv", index=False)
display(manifest_df)

# %% [markdown]
# > **数据说明**: Manifest 表是文件索引和最小 shape/dtype 摘要。正式 CLI 的 manifest 还会包含 difficulty、seed、forward mode、shift profile 和 metadata hash。
# >
# > **怎么看**: `expected_shape` 与 `shape` 必须一致；如果某个数组 shape 漂移，后续算法会在读取阶段或评估阶段出现不可解释错误。
# >
# > **异常是否正常**: Notebook demo manifest 比正式 manifest 更小，这是刻意限制；正式 P0 数据集应由 CLI 生成完整 manifest。
# >
# > **核心发现**: Demo 的每个核心数组都已落盘并可由 manifest 独立发现，而不是依赖 notebook 内存变量。

# %%
def _check_row(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "pass": bool(passed), "detail": detail}


smoke_rows = []
for filename, expected_shape in required.items():
    path = DEMO_DIR / filename
    exists = path.exists()
    if exists:
        arr = np.load(path)
        shape_ok = tuple(arr.shape) == expected_shape
        finite = bool(np.isfinite(arr).all())
        smoke_rows.append(_check_row(f"{filename}: exists", exists, relative(path)))
        smoke_rows.append(_check_row(f"{filename}: shape", shape_ok, f"{arr.shape} expected {expected_shape}"))
        smoke_rows.append(_check_row(f"{filename}: finite", finite, "all finite" if finite else "contains NaN/Inf"))
    else:
        smoke_rows.append(_check_row(f"{filename}: exists", False, relative(path)))

mask = np.load(DEMO_DIR / "hr_mask_2x.npy")
hp = np.load(DEMO_DIR / "lr_burst_highpass.npy")
raw = np.load(DEMO_DIR / "lr_burst_raw.npy")
shifts_loaded = np.load(DEMO_DIR / "shifts.npy")
hp_indices = np.unique(np.linspace(0, raw.shape[0] - 1, min(8, raw.shape[0]), dtype=int))
hp_reference = _ep06_like_highpass(raw[hp_indices], sigma_bg=DEMO_CONFIG.highpass_sigma_lr_px, mode="nearest")
hp_diff = hp[hp_indices] - hp_reference
constant_hp = _ep06_like_highpass(np.full(DEMO_CONFIG.lr_shape, 21.0, dtype=np.float32), sigma_bg=DEMO_CONFIG.highpass_sigma_lr_px)

smoke_rows.extend(
    [
        _check_row("mask binary uint8", mask.dtype == np.uint8 and set(np.unique(mask).tolist()) <= {0, 1}, f"dtype={mask.dtype}, values={sorted(np.unique(mask).tolist())}"),
        _check_row("highpass dtype float32", hp.dtype == np.float32, f"dtype={hp.dtype}"),
        _check_row("shift convention recorded", metadata["shifts"]["convention"] == "LR-to-reference alignment shift", metadata["shifts"]["convention"]),
        _check_row("shift values finite", bool(np.isfinite(shifts_loaded).all()), f"max_norm={np.linalg.norm(shifts_loaded, axis=1).max():.4f} LR px"),
        _check_row("highpass reference allclose", bool(np.allclose(hp[hp_indices], hp_reference, rtol=1e-5, atol=1e-5)), f"frames={len(hp_indices)}, max_abs_diff={np.max(np.abs(hp_diff)):.3e} C"),
        _check_row("constant-frame highpass near zero", float(np.max(np.abs(constant_hp))) < 1e-5, f"max_abs={np.max(np.abs(constant_hp)):.3e} C"),
        _check_row("metadata forward mode recorded", "exact_ep06_point" in metadata["physics"]["forward_mode"], metadata["physics"]["forward_mode"]),
    ]
)
smoke_df = pd.DataFrame(smoke_rows)
smoke_df.to_csv(DEMO_DIR / "smoke_summary.csv", index=False)
display(smoke_df)
print(f"Smoke pass: {bool(smoke_df['pass'].all())}")
print(f"Manifest: {relative(DEMO_DIR / 'manifest.csv')}")
print(f"Metadata: {relative(DEMO_DIR / 'metadata.json')}")
print(f"Smoke summary: {relative(DEMO_DIR / 'smoke_summary.csv')}")

# %% [markdown]
# > **数据说明**: Smoke 表把文件存在、shape、finite、mask 二值性、shift 约定、高通独立复算和常数帧高通行为逐项列出。
# >
# > **怎么看**: `pass=True` 是 P0 硬门槛；`highpass reference allclose` 说明 notebook 没有只拿 generator 自己和自己比较，而是用独立 scipy 路径复算采样帧。
# >
# > **异常是否正常**: Highpass 正负值不是异常；NaN/Inf、shape 不一致、mask 非 0/1、metadata 丢失 forward/shift 约定才是异常。
# >
# > **核心发现**: EP07 后台的关键检查已经在 notebook 中显式展示，读者可以看到生成产物不是“看起来像”数据，而是通过了结构性 smoke gate。
