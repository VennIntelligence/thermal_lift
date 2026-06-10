# %% [markdown]
# ## 6. 结论与交接
#
# EP07 验证的是 **合成数据引擎成熟度**，不是真实芯片 SR 成效。

# %%
if cache.demo_skipped:
    display(Markdown("结论单元需要完整缓存。请运行 `uv run python scripts/build_ep07_cache.py --force`。"))
else:
    handoff = pd.DataFrame(
        [
            {"area": "Thermal field", "evidence": "demo_thermal_field_decomposition.png + physics_checks", "rule": "keep structure / low-freq / full temperature separable"},
            {"area": "PSF", "evidence": "demo_psf_blur_check.png + demo_metadata.json", "rule": "default follows EP09 provisional Route A sigma; legacy_upper=0.5 only for stress"},
            {"area": "Noise", "evidence": "demo_noise_check.png + demo_noise_real_vs_synthetic.png", "rule": "inject on LR burst only; model + RMS anchor must be recorded"},
            {"area": "SNR budget", "evidence": "demo_snr_budget.png + snr_budget.csv", "rule": "risk bands are necessary conditions, not SR proof"},
            {"area": "LR burst", "evidence": "demo_forward_highpass.png + demo_dataset_overview.png", "rule": "highpass is structure response, not absolute temperature"},
            {"area": "Smoke", "evidence": "physics_checks + independent highpass allclose", "rule": "all checks pass before SR regression"},
        ]
    )
    display(handoff)

# %% [markdown]
# > **核心发现**: 最新 TCForge 引擎已形成「配置 → HR 真值 → forward/PSF → LR 加噪 → highpass → 验收」闭环；EP07 demo 用 9 张图和 5 张表把此前缺失的热场/PSF/噪声/SNR 检查补齐。
# >
# > **不作声明**: 不把 synthetic shifts 当真实 alignment 真值；不把 demo 指标外推为真实主 session SR 结论。

# %%
figures = [
    "demo_hr_scene.png",
    "demo_thermal_field_decomposition.png",
    "demo_psf_blur_check.png",
    "demo_noise_check.png",
    "demo_noise_real_vs_synthetic.png",
    "demo_snr_budget.png",
    "demo_forward_highpass.png",
    "demo_dataset_overview.png",
    "demo_profiles_generation_vs_observation.png",
]
if not cache.demo_skipped:
    display(Markdown(
        "\n".join([
            "**EP07 缓存产物**",
            "",
            f"- 缓存目录: `{rel(OUTPUT_DIR)}`",
            f"- Demo 数据包: `{rel(DEMO_DIR)}`",
            f"- 重建命令: `{REBUILD_CMD}`",
            "",
            "**图片**",
            *[f"  - `{rel(OUTPUT_DIR / name)}`" for name in figures],
            "",
            "**表格**",
            f"  - `{rel(OUTPUT_DIR / 'physics_checks.csv')}`",
            f"  - `{rel(OUTPUT_DIR / 'snr_budget.csv')}`",
            f"  - `{rel(OUTPUT_DIR / 'noise_model_checks.csv')}`",
            f"  - `{rel(OUTPUT_DIR / 'scene_stats.csv')}`",
            f"  - `{rel(OUTPUT_DIR / 'forward_stats.csv')}`",
            f"  - `{rel(OUTPUT_DIR / 'demo_metrics.csv')}`",
        ])
    ))
