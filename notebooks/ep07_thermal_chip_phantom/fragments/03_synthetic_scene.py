# %% [markdown]
# ## 2. HR 场景生成：从结构真值到温度真值
#
# 这一节把后台生成逻辑显式展示出来。TCForge 先在统一 HR canvas 上生成二值结构 mask，再用温度背景、结构温升和低频热背景得到 HR temperature truth，最后派生 edge map 作为 contour proxy。
#
# Demo 使用 `lr_shape=(64, 96)`、`scale=2`、`n_frames=16`，只用于报告层 smoke，不生成全幅 255 帧数据。正式 P0 数据集应走 CLI/API 路径：
#
# ```bash
# uv run python scripts/generate_thermal_chip_phantom.py \
#   --config configs/synthetic/phantom_smoke.json \
#   --shift-config configs/synthetic/shift_profiles.json \
#   --output-root data/synthetic/thermal_chip_phantom
# ```

# %%
generation_recipe = pd.DataFrame(
    [
        {
            "step": "1. Geometry",
            "operation": "Compose frame, pins, crosses, trenches, then rotate and re-binarize",
            "output": "hr_mask_2x.npy",
            "why_it_exists": "Defines the exact chip contour/shape truth for regression.",
        },
        {
            "step": "2. Physics",
            "operation": "Map mask to temperature with background, delta-T and low-frequency thermal variation",
            "output": "hr_temperature_2x.npy",
            "why_it_exists": "Defines the HR temperature target before LR observation simulation.",
        },
        {
            "step": "3. Contour proxy",
            "operation": "Morphological dilation/erosion around the binary structure",
            "output": "hr_edge_map_2x.npy",
            "why_it_exists": "Supports contour-level diagnostics without claiming optical ground truth.",
        },
        {
            "step": "4. Multi-scale sanity",
            "operation": "Generate the same scene family on a 4x canvas for display sanity only",
            "output": "hr_mask_4x.npy / hr_temperature_4x.npy",
            "why_it_exists": "Shows that the generator is scale-aware; EP07 P0 remains 2x.",
        },
    ]
)
display(generation_recipe)

# %% [markdown]
# > **数据说明**: 这张表把 HR 场景生成拆成结构、物理、轮廓 proxy 和多尺度 sanity 四步，直接对应后面落盘的 `.npy` 产物。
# >
# > **怎么看**: `hr_mask_2x.npy` 是最锐利的几何 truth；`hr_temperature_2x.npy` 是温度 truth；`hr_edge_map_2x.npy` 是边界 proxy。三者用途不同，不能混成一类图。
# >
# > **异常是否正常**: 如果 HR temperature 中存在低频亮暗变化，这是合成热背景；它不代表结构边界错位。真正的光学 PSF 退化在下一节 forward model 中作用。
# >
# > **核心发现**: EP07 生成的是有明确中间层的合成数据包，而不是只把一张 mask 直接缩小成 LR 图片。

# %%
def _fallback_make_scene(config: DemoConfig) -> tuple[dict[str, np.ndarray], str]:
    rng = np.random.default_rng(config.seed)
    hr_shape = (config.lr_shape[0] * config.scale, config.lr_shape[1] * config.scale)
    yy, xx = np.mgrid[: hr_shape[0], : hr_shape[1]]

    mask = np.zeros(hr_shape, dtype=np.uint8)
    mask[24:104, 32:160] = 1
    mask[48:80, 58:138] = 0
    for col in range(36, 156, 16):
        mask[18:28, col : col + 6] = 1
        mask[100:112, col : col + 6] = 1
    for row in range(34, 98, 14):
        mask[row : row + 5, 24:36] = 1
        mask[row : row + 5, 156:168] = 1

    channel = np.exp(-((yy - 64) ** 2) / (2 * 5.0**2)) * ((xx > 50) & (xx < 144))
    hotspot = np.exp(-(((yy - 44) ** 2) + ((xx - 124) ** 2)) / (2 * 10.0**2))
    lowfreq = 0.08 * np.sin(xx / 45.0) + 0.05 * np.cos(yy / 38.0)
    hr_temperature = (
        config.base_temp_c
        + config.delta_temp_c * mask.astype(np.float32)
        + 0.35 * channel
        + 0.25 * hotspot
        + lowfreq
    ).astype(np.float32)
    hr_temperature += rng.normal(0.0, config.noise_sigma_c, hr_shape).astype(np.float32)

    edge_y, edge_x = np.gradient(mask.astype(np.float32))
    edge_map = np.hypot(edge_y, edge_x).astype(np.float32)

    hr_shape_4x = (config.lr_shape[0] * 4, config.lr_shape[1] * 4)
    yy_4x, xx_4x = np.mgrid[: hr_shape_4x[0], : hr_shape_4x[1]]
    mask_4x = np.zeros(hr_shape_4x, dtype=np.uint8)
    mask_4x[48:208, 64:320] = 1
    mask_4x[96:160, 116:276] = 0
    channel_4x = np.exp(-((yy_4x - 128) ** 2) / (2 * 10.0**2)) * ((xx_4x > 100) & (xx_4x < 288))
    hotspot_4x = np.exp(-(((yy_4x - 88) ** 2) + ((xx_4x - 248) ** 2)) / (2 * 20.0**2))
    lowfreq_4x = 0.08 * np.sin(xx_4x / 90.0) + 0.05 * np.cos(yy_4x / 76.0)
    hr_temperature_4x = (
        config.base_temp_c
        + config.delta_temp_c * mask_4x.astype(np.float32)
        + 0.35 * channel_4x
        + 0.25 * hotspot_4x
        + lowfreq_4x
    ).astype(np.float32)
    hr_temperature_4x += rng.normal(0.0, config.noise_sigma_c, hr_shape_4x).astype(np.float32)

    return {
        "hr_mask_2x": mask,
        "hr_temperature_2x": hr_temperature.astype(np.float32),
        "hr_edge_map_2x": edge_map,
        "hr_mask_4x": mask_4x,
        "hr_temperature_4x": hr_temperature_4x,
    }, "notebook_fallback_geometry_physics"


def _make_scene(config: DemoConfig) -> tuple[dict[str, np.ndarray], str]:
    try:
        from tcforge.geometry import build_scene_mask
        from tcforge.physics import add_noise, edge_map, render_temperature_field

        hr_shape = (config.lr_shape[0] * config.scale, config.lr_shape[1] * config.scale)
        mask = build_scene_mask(
            "easy",
            config.seed,
            canvas_shape=hr_shape,
            pixel_size_um=config.pixel_size_um,
            scale=config.scale,
            rotation_jitter_deg=0.0,
        )
        hr_temperature = render_temperature_field(
            mask,
            t_bg_c=config.base_temp_c,
            delta_t_c=config.delta_temp_c,
            low_freq_amplitude_c=0.08,
            low_freq_sigma_px=max(8.0, min(hr_shape) / 8.0),
            seed=config.seed,
        )
        hr_temperature = add_noise(hr_temperature, noise_sigma_c=config.noise_sigma_c, seed=config.seed + 1)

        hr_shape_4x = (config.lr_shape[0] * 4, config.lr_shape[1] * 4)
        mask_4x = build_scene_mask(
            "easy",
            config.seed,
            canvas_shape=hr_shape_4x,
            pixel_size_um=config.pixel_size_um,
            scale=4,
            rotation_jitter_deg=0.0,
        )
        hr_temperature_4x = render_temperature_field(
            mask_4x,
            t_bg_c=config.base_temp_c,
            delta_t_c=config.delta_temp_c,
            low_freq_amplitude_c=0.08,
            low_freq_sigma_px=max(8.0, min(hr_shape_4x) / 8.0),
            seed=config.seed,
        )
        hr_temperature_4x = add_noise(hr_temperature_4x, noise_sigma_c=config.noise_sigma_c, seed=config.seed + 1)

        return {
            "hr_mask_2x": np.asarray(mask, dtype=np.uint8),
            "hr_temperature_2x": np.asarray(hr_temperature, dtype=np.float32),
            "hr_edge_map_2x": np.asarray(edge_map(mask), dtype=np.float32),
            "hr_mask_4x": np.asarray(mask_4x, dtype=np.uint8),
            "hr_temperature_4x": np.asarray(hr_temperature_4x, dtype=np.float32),
        }, "tcforge.geometry+tcforge.physics"
    except Exception as exc:
        print(f"TCForge scene API fallback: {exc.__class__.__name__}: {exc}")
        return _fallback_make_scene(config)


scene, SCENE_GENERATION_MODE = _make_scene(DEMO_CONFIG)
for name, arr in scene.items():
    np.save(DEMO_DIR / f"{name}.npy", arr)

scene_stats = pd.DataFrame(
    [
        array_contract_row("hr_mask_2x", scene["hr_mask_2x"], role="2x binary structure truth"),
        array_contract_row("hr_temperature_2x", scene["hr_temperature_2x"], role="2x HR temperature truth [C]"),
        array_contract_row("hr_edge_map_2x", scene["hr_edge_map_2x"], role="2x contour proxy"),
        array_contract_row("hr_mask_4x", scene["hr_mask_4x"], role="4x display sanity mask"),
        array_contract_row("hr_temperature_4x", scene["hr_temperature_4x"], role="4x display sanity temperature [C]"),
    ]
)
display(compact_numeric_table(scene_stats, ["array", "role", "shape", "dtype", "min", "max", "finite"]))
print(f"Scene generation mode: {SCENE_GENERATION_MODE}")

# %% [markdown]
# > **数据说明**: 这张表汇总 HR 级别的数组产物，包括正式 2x truth 和仅用于展示 sanity 的 4x truth。
# >
# > **怎么看**: `shape` 必须符合 `lr_shape * scale`；mask 必须是 `uint8`；温度图必须是 `float32` 且所有数值 finite。
# >
# > **异常是否正常**: 4x 产物只是证明生成器对尺度参数敏感，不表示 EP07 要交付 4x SR benchmark。正式 P0 仍以 2x contour-level 回归为主。
# >
# > **核心发现**: Demo 场景已经在磁盘上形成结构真值、温度真值和轮廓 proxy，后续章节会用这些文件生成 LR burst 与 smoke/evaluate 指标。

# %%
fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.2))
axes = axes.ravel()
show_image(
    axes[0],
    scene["hr_mask_2x"],
    title="HR mask 2x",
    cmap="gray",
    colorbar_label="Mask",
    vmin=0,
    vmax=1,
)
show_image(
    axes[1],
    scene["hr_temperature_2x"],
    title="HR temperature 2x",
    cmap=COLORMAPS["temperature"],
    colorbar_label="Temperature [$^\\circ$C]",
    robust=True,
)
show_image(
    axes[2],
    scene["hr_edge_map_2x"],
    title="HR edge proxy 2x",
    cmap=COLORMAPS["coverage"],
    colorbar_label="Edge",
    vmin=0,
    vmax=max(1.0, float(np.max(scene["hr_edge_map_2x"]))),
)
show_image(
    axes[3],
    scene["hr_mask_4x"],
    title="HR mask 4x sanity",
    cmap="gray",
    colorbar_label="Mask",
    vmin=0,
    vmax=1,
)
show_image(
    axes[4],
    scene["hr_temperature_4x"],
    title="HR temperature 4x sanity",
    cmap=COLORMAPS["temperature"],
    colorbar_label="Temperature [$^\\circ$C]",
    robust=True,
)
axes[5].axis("off")
axes[5].text(
    0.0,
    0.86,
    "Generation contract",
    fontsize=10,
    fontweight="bold",
    transform=axes[5].transAxes,
)
axes[5].text(
    0.0,
    0.68,
    f"scene_id: {DEMO_CONFIG.scene_id}\n"
    f"seed: {DEMO_CONFIG.seed}\n"
    f"LR shape: {DEMO_CONFIG.lr_shape[0]} x {DEMO_CONFIG.lr_shape[1]}\n"
    f"2x HR shape: {scene['hr_temperature_2x'].shape[0]} x {scene['hr_temperature_2x'].shape[1]}\n"
    f"pixel pitch: {DEMO_CONFIG.pixel_size_um:.1f} um/LR px",
    fontsize=8,
    va="top",
    transform=axes[5].transAxes,
)
for label, ax in zip(["a", "b", "c", "d", "e"], axes[:5], strict=True):
    add_panel_label(ax, label)
save_fig(fig, "demo_hr_scene.png")

# %% [markdown]
# > **图表说明**: 这张图用 CVPR 风格展示 HR 结构真值、温度真值、轮廓 proxy，以及 4x sanity 视图。右下角不是数据图，而是生成参数摘要，避免读者需要回看代码。
# >
# > **怎么看**: mask 面板看几何边界；temperature 面板看温度分布和低频背景；edge proxy 面板看 contour 位置。色标单位只有温度图是摄氏度。
# >
# > **异常是否正常**: 温度图与 mask 不必完全视觉一致，因为温度场包含低频背景；edge proxy 是形态学边界，不是光学边缘检测结果。
# >
# > **核心发现**: EP07 demo 中的“真值”已经分层：结构、温度、边界 proxy 各司其职，后续指标应按对应任务消费。
