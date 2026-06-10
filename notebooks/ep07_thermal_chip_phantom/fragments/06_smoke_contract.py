# %% [markdown]
# ## 5. 产物契约与 Smoke 验收
#
# 正式 P0 数据集由 CLI 生成；Notebook demo 复用同一文件契约，但缩小尺寸。下表列出每个落盘文件的消费者和物理含义。

# %%
import json
import numpy as np

from thermal_core.ep07_cache import ep06_like_highpass

if cache.demo_skipped:
    display(Markdown("Smoke 单元需要完整 demo 缓存。"))
else:
    artifact_catalog = pd.DataFrame(
        [
            {"file": "hr_mask_2x.npy", "producer": "geometry", "consumer": "contour / mask metric", "meaning": "binary structure truth"},
            {"file": "hr_temperature_2x.npy", "producer": "physics", "consumer": "temperature metric", "meaning": "clean HR temperature truth"},
            {"file": "hr_edge_map_2x.npy", "producer": "physics.edge_map", "consumer": "boundary diagnostic", "meaning": "morphological contour proxy"},
            {"file": "lr_burst_raw.npy", "producer": "forward+noise", "consumer": "SR input", "meaning": "ordinary LR temperature burst"},
            {"file": "lr_burst_highpass.npy", "producer": "highpass", "consumer": "structure SR input", "meaning": "signed structure response"},
            {"file": "shifts.npy", "producer": "shifts", "consumer": "synthetic alignment control", "meaning": "LR-to-reference (dx, dy)"},
            {"file": "metadata.json", "producer": "manifest", "consumer": "reproducibility", "meaning": "physical contract + provenance"},
        ]
    )
    display(artifact_catalog)

# %% [markdown]
# > **数据说明**: 所有 `.npy` 产物在 Git 中 ignore，只跟踪生成器源码和配置。
# >
# > **核心发现**: SR 算法应消费 `lr_burst_raw` 或 `lr_burst_highpass`；`hr_temperature_2x` / `hr_mask_2x` 仅作 synthetic ground truth。

# %%
if not cache.demo_skipped:
    cfg = cache.demo_config
    required = {
        "hr_temperature_2x.npy": (cfg.lr_shape[0] * cfg.scale, cfg.lr_shape[1] * cfg.scale),
        "hr_mask_2x.npy": (cfg.lr_shape[0] * cfg.scale, cfg.lr_shape[1] * cfg.scale),
        "hr_edge_map_2x.npy": (cfg.lr_shape[0] * cfg.scale, cfg.lr_shape[1] * cfg.scale),
        "lr_burst_raw.npy": (cfg.n_frames, *cfg.lr_shape),
        "lr_burst_highpass.npy": (cfg.n_frames, *cfg.lr_shape),
        "shifts.npy": (cfg.n_frames, 2),
    }

    smoke_rows = []
    for filename, expected_shape in required.items():
        arr = np.load(DEMO_DIR / filename)
        smoke_rows.append({
            "check": f"{filename}: shape",
            "pass": tuple(arr.shape) == expected_shape,
            "detail": f"{arr.shape} expected {expected_shape}",
        })
        smoke_rows.append({
            "check": f"{filename}: finite",
            "pass": bool(np.isfinite(arr).all()),
            "detail": "all finite" if np.isfinite(arr).all() else "contains NaN/Inf",
        })

    mask = np.load(DEMO_DIR / "hr_mask_2x.npy")
    raw = np.load(DEMO_DIR / "lr_burst_raw.npy")
    hp = np.load(DEMO_DIR / "lr_burst_highpass.npy")
    hp_idx = np.unique(np.linspace(0, raw.shape[0] - 1, min(8, raw.shape[0]), dtype=int))
    hp_ref = ep06_like_highpass(raw[hp_idx], sigma_bg=cfg.highpass_sigma_lr_px, mode="nearest")
    smoke_rows.extend([
        {
            "check": "mask binary uint8",
            "pass": mask.dtype == np.uint8 and set(np.unique(mask).tolist()) <= {0, 1},
            "detail": f"dtype={mask.dtype}",
        },
        {
            "check": "highpass independent allclose",
            "pass": bool(np.allclose(hp[hp_idx], hp_ref, rtol=1e-5, atol=1e-5)),
            "detail": f"max_abs_diff={np.max(np.abs(hp[hp_idx] - hp_ref)):.3e} C",
        },
        {
            "check": "physics_checks all pass",
            "pass": bool(cache.physics_checks["pass"].all()),
            "detail": f"{int(cache.physics_checks['pass'].sum())}/{len(cache.physics_checks)}",
        },
    ])
    smoke_df = pd.DataFrame(smoke_rows)
    display(compact_table(smoke_df, ["check", "pass", "detail"]))
    print(f"Smoke pass: {bool(smoke_df['pass'].all())}")
    print(f"Metadata: {rel(DEMO_DIR / 'metadata.json')}")

# %% [markdown]
# > **图表说明**: Smoke 表检查 shape/finite、mask 二值性、highpass 独立复算一致性和物理检查汇总。
# >
# > **怎么看**: `highpass independent allclose` 用 SciPy 重算，不依赖生成器自证；任一 fail 应阻止 benchmark 交付。
# >
# > **核心发现**: demo 数据包满足 P0 契约的子集验收，正式全幅数据仍需 CLI `smoke_test_thermal_chip_phantom.py` 门控。
