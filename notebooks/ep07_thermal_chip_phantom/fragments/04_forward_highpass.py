# %% [markdown]
# ## 3. LR Burst 生成：位移、Forward Model 与 Highpass
#
# HR truth 不能直接喂给 SR 算法。TCForge 会根据 shift profile 生成多帧 LR observation，并同步保存 raw temperature burst 与 EP06-compatible highpass burst。
#
# | 环节 | 数学/工程含义 | 落盘产物 | 关键边界 |
# |---|---|---|---|
# | Shift profile | 每帧相对 reference 的 `(dx, dy)`，单位 LR pixel | `shifts.npy` | synthetic 控制量，不是真实 stage/alignment 真值 |
# | Forward model | HR 温度场经 PSF 和采样变成 LR observation | `lr_burst_raw.npy` | `exact_ep06_point` 与 `physical_block_average` 必须分开报告 |
# | Highpass | 每帧减去空间 Gaussian 背景 | `lr_burst_highpass.npy` | 输出是结构响应，红/蓝不是绝对温度 |
# | Boundary crop | 仅用于图像展示，去掉 outside-FOV 边界带 | 不落盘 | 不改变 smoke/evaluate 的原始数组 |

# %%
forward_contract = pd.DataFrame(
    [
        {
            "forward_mode": "exact_ep06_point",
            "sampling": "EP06 point-sampling reference operator",
            "why_keep_it": "Locks the sign convention used by EP06 algorithms and tests.",
            "metric_rule": "Report separately from block-average results.",
        },
        {
            "forward_mode": "physical_block_average",
            "sampling": "Shifted detector block average after PSF",
            "why_keep_it": "Closer to a detector-area measurement assumption.",
            "metric_rule": "Report separately; do not mix with exact_ep06_point.",
        },
    ]
)
display(forward_contract)

# %% [markdown]
# > **数据说明**: 这张表说明 TCForge 为什么同时保留两个 forward mode。它不是性能结果表，而是数据契约表。
# >
# > **怎么看**: `exact_ep06_point` 用于锁定 EP06 符号约定和回归测试；`physical_block_average` 是另一种物理假设。二者可能生成不同 LR 图像。
# >
# > **异常是否正常**: 两种 mode 不完全一致是正常的；如果把它们混在同一评估表中，算法排名会失去解释性。
# >
# > **核心发现**: EP07 已把 forward model 作为显式维度管理，这是后续 synthetic benchmark 可复现的关键。

# %%
def _phase_grid_shifts(n_frames: int) -> np.ndarray:
    phases = np.array([(dx, dy) for dy in (0.0, 0.25, 0.5, 0.75) for dx in (0.0, 0.25, 0.5, 0.75)], dtype=np.float32)
    if n_frames <= len(phases):
        return phases[:n_frames]
    reps = int(np.ceil(n_frames / len(phases)))
    return np.tile(phases, (reps, 1))[:n_frames]


def _fallback_forward_block_average(hr_temperature: np.ndarray, shifts_lr_dxdy: np.ndarray, config: DemoConfig) -> np.ndarray:
    frames = []
    for shift_lr_x, shift_lr_y in shifts_lr_dxdy:
        shifted = ndi_shift(
            hr_temperature,
            shift=(-shift_lr_y * config.scale, -shift_lr_x * config.scale),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        lr = shifted.reshape(config.lr_shape[0], config.scale, config.lr_shape[1], config.scale).mean(axis=(1, 3))
        frames.append(lr.astype(np.float32))
    return np.stack(frames, axis=0).astype(np.float32)


def _make_shifts(config: DemoConfig) -> tuple[np.ndarray, str]:
    try:
        from tcforge.shifts import load_shift_profile

        shifts, info = load_shift_profile("ideal_phase_grid", n_frames=config.n_frames, scale=config.scale, phase_steps=4, seed=config.seed)
        return np.asarray(shifts, dtype=np.float32), f"tcforge.shifts.load_shift_profile({info['profile']})"
    except Exception as exc:
        print(f"TCForge shift API fallback: {exc.__class__.__name__}: {exc}")
        return _phase_grid_shifts(config.n_frames), "notebook_4x4_phase_grid"


def _make_lr_burst(
    hr_temperature: np.ndarray,
    shifts_lr_dxdy: np.ndarray,
    config: DemoConfig,
    *,
    forward_mode: str = "exact_ep06_point",
) -> tuple[np.ndarray, str]:
    try:
        from tcforge.forward import generate_lr_burst

        raw = generate_lr_burst(
            hr_temperature,
            shifts_lr_dxdy,
            forward_mode=forward_mode,
            psf_sigma_lr_px=config.psf_sigma_lr_px,
            scale=config.scale,
        )
        return raw.astype(np.float32, copy=False), f"tcforge.forward.generate_lr_burst({forward_mode})"
    except Exception as exc:
        print(f"TCForge forward API fallback ({forward_mode}): {exc.__class__.__name__}: {exc}")
        return _fallback_forward_block_average(hr_temperature, shifts_lr_dxdy, config), "notebook_fallback_physical_block_average"


def _ep06_like_highpass(frames: np.ndarray, sigma_bg: float = 5.0, mode: str = "nearest") -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return arr - gaussian_filter(arr, sigma=sigma_bg, mode=mode)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")
    bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
    return (arr - bg).astype(np.float32, copy=False)


def _make_highpass(frames: np.ndarray, config: DemoConfig) -> tuple[np.ndarray, str]:
    try:
        from tcforge.highpass import highpass_preprocess

        return (
            highpass_preprocess(frames, sigma_bg=config.highpass_sigma_lr_px, mode="nearest"),
            "tcforge.highpass.highpass_preprocess",
        )
    except Exception as exc:
        print(f"TCForge highpass API fallback: {exc.__class__.__name__}: {exc}")
        return _ep06_like_highpass(frames, sigma_bg=config.highpass_sigma_lr_px), "notebook_ep06_like_highpass"


def _crop_lr_image(image: np.ndarray, margin: int) -> np.ndarray:
    margin = int(margin)
    if margin <= 0:
        return image
    if min(image.shape) <= 2 * margin:
        return image
    return image[margin:-margin, margin:-margin]


def _crop_lr_stack(frames: np.ndarray, margin: int) -> np.ndarray:
    margin = int(margin)
    if margin <= 0:
        return frames
    if min(frames.shape[-2:]) <= 2 * margin:
        return frames
    return frames[:, margin:-margin, margin:-margin]


shifts, SHIFT_SOURCE = _make_shifts(DEMO_CONFIG)
lr_burst_raw, FORWARD_MODE = _make_lr_burst(
    scene["hr_temperature_2x"],
    shifts,
    DEMO_CONFIG,
    forward_mode="exact_ep06_point",
)
lr_burst_highpass, HIGHPASS_SOURCE = _make_highpass(lr_burst_raw, DEMO_CONFIG)
block_preview, BLOCK_FORWARD_MODE = _make_lr_burst(
    scene["hr_temperature_2x"],
    shifts[:1],
    DEMO_CONFIG,
    forward_mode="physical_block_average",
)
EDGE_VIS_MARGIN_LR_PX = int(np.ceil(2.5 * DEMO_CONFIG.highpass_sigma_lr_px))

np.save(DEMO_DIR / "shifts.npy", shifts)
np.save(DEMO_DIR / "lr_burst_raw.npy", lr_burst_raw)
np.save(DEMO_DIR / "lr_burst_highpass.npy", lr_burst_highpass)

forward_stats = pd.DataFrame(
    [
        array_contract_row("shifts", shifts, role="LR-to-reference alignment shifts [LR px]"),
        array_contract_row("lr_burst_raw", lr_burst_raw, role="ordinary LR temperature observations [C]"),
        array_contract_row("lr_burst_highpass", lr_burst_highpass, role="EP06-compatible structure response [C]"),
    ]
)
display(compact_numeric_table(forward_stats, ["array", "role", "shape", "dtype", "min", "max", "finite"]))
print(f"Shift source: {SHIFT_SOURCE}")
print(f"Forward mode: {FORWARD_MODE}")
print(f"Block-average preview mode: {BLOCK_FORWARD_MODE}")
print(f"Highpass source: {HIGHPASS_SOURCE}")
print(f"Visualization crop margin: {EDGE_VIS_MARGIN_LR_PX} LR px")

# %% [markdown]
# > **数据说明**: 这张表汇总 shifts、LR raw burst 和 LR highpass burst 的 shape/dtype/range。这里的 LR burst 是后续算法真正看到的 synthetic observation。
# >
# > **怎么看**: `shifts` 的列顺序是 `(dx, dy)`，单位 LR pixel。`lr_burst_raw` 是普通温度序列；`lr_burst_highpass` 是扣除低频背景后的结构响应。
# >
# > **异常是否正常**: Highpass 可以有正负值，白色通常表示接近零变化，红/蓝表示相对局部背景的正/负响应。由于 point-forward 的 outside-FOV 边界约定，全图边缘可能有强响应，展示时会裁掉边界带。
# >
# > **核心发现**: EP07 已把位移、forward 和 highpass 三个后续算法最容易误用的约定写入了可检查数组和 metadata。

# %%
fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4))
axes = axes.ravel()

frame_idx = 5
raw_frame_vis = _crop_lr_image(lr_burst_raw[frame_idx], EDGE_VIS_MARGIN_LR_PX)
highpass_frame_vis = _crop_lr_image(lr_burst_highpass[frame_idx], EDGE_VIS_MARGIN_LR_PX)
highpass_abs_vis = np.mean(np.abs(_crop_lr_stack(lr_burst_highpass, EDGE_VIS_MARGIN_LR_PX)), axis=0)
point_preview = _crop_lr_image(lr_burst_raw[0], EDGE_VIS_MARGIN_LR_PX)
block_preview_vis = _crop_lr_image(block_preview[0], EDGE_VIS_MARGIN_LR_PX)
forward_diff = point_preview - block_preview_vis

show_image(
    axes[0],
    raw_frame_vis,
    title="LR raw frame (interior)",
    cmap=COLORMAPS["temperature"],
    colorbar_label="Temperature [$^\\circ$C]",
    robust=True,
)
show_image(
    axes[1],
    highpass_frame_vis,
    title="LR highpass frame (interior)",
    cmap=COLORMAPS["residual_diff"],
    colorbar_label="Highpass [$^\\circ$C]",
    robust=True,
    symmetric=True,
)
show_image(
    axes[2],
    highpass_abs_vis,
    title="Mean abs highpass (interior)",
    cmap="magma",
    colorbar_label="Abs response [$^\\circ$C]",
    robust=True,
)

axes[3].scatter(shifts[:, 0], shifts[:, 1], c=np.arange(len(shifts)), cmap=COLORMAPS["coverage"], s=36, edgecolors="black", linewidths=0.25)
axes[3].set_title("Sub-pixel phase coverage")
axes[3].set_xlabel("dx [LR px]")
axes[3].set_ylabel("dy [LR px]")
axes[3].invert_yaxis()
axes[3].set_aspect("equal", adjustable="box")
axes[3].grid(True, alpha=0.25, linewidth=0.5)

show_image(
    axes[4],
    forward_diff,
    title="Point minus block preview",
    cmap=COLORMAPS["residual_diff"],
    colorbar_label="Difference [$^\\circ$C]",
    robust=True,
    symmetric=True,
)

axes[5].axis("off")
axes[5].text(0.0, 0.9, "Forward/highpass settings", fontsize=10, fontweight="bold", transform=axes[5].transAxes)
axes[5].text(
    0.0,
    0.72,
    f"forward: exact_ep06_point\n"
    f"PSF sigma: {DEMO_CONFIG.psf_sigma_lr_px:.2f} LR px\n"
    f"highpass sigma: {DEMO_CONFIG.highpass_sigma_lr_px:.1f} LR px\n"
    f"frames: {DEMO_CONFIG.n_frames}\n"
    f"shift max norm: {np.linalg.norm(shifts, axis=1).max():.3f} LR px",
    fontsize=8,
    va="top",
    transform=axes[5].transAxes,
)

for label, ax in zip(["a", "b", "c", "d", "e"], axes[:5], strict=True):
    add_panel_label(ax, label)
save_fig(fig, "demo_forward_highpass.png")

# %% [markdown]
# > **图表说明**: 这张图展示同一 synthetic scene 的 LR raw frame、LR highpass frame、burst 平均结构响应、亚像素 phase coverage，以及 `exact_ep06_point` 与 `physical_block_average` 的单帧差异预览。
# >
# > **怎么看**: Raw 图用于判断普通温度观测是否合理；highpass 图用于看边缘和内部结构响应；phase coverage 越分散，说明 demo 有多个亚像素采样相位；forward difference 说明 forward mode 是真实会改变数据的契约，不是命名装饰。
# >
# > **异常是否正常**: Highpass 图中白色接近零，红/蓝为相对局部背景的正/负响应。边界裁剪只是展示层操作，用来避免 outside-FOV 边界伪影主导色标；落盘数组仍是完整 LR frame。
# >
# > **核心发现**: EP07 不只生成图像，还显式固定了位移相位、forward mode 和 highpass 预处理，这些是 SR 回归可解释性的前提。
