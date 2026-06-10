# %% [markdown]
# ## 1. 生成管线与配置契约
#
# TCForge 把合成芯片场景拆成可审计的物理退化链。配置层定义边界，生成器产出 HR 真值与 LR burst，验收层写 manifest / smoke / evaluate 摘要。
#
# ```mermaid
# flowchart LR
#     C["phantom_smoke.json<br/>PSF, noise, delta_T, highpass"]
#     G1["geometry → hr_mask"]
#     G2["physics → hr_temperature"]
#     G3["forward + PSF → lr_burst_clean"]
#     G4["add_noise → lr_burst_raw"]
#     G5["highpass → lr_burst_highpass"]
#     V["manifest / smoke / evaluate"]
#     C --> G1 --> G2 --> G3 --> G4 --> G5 --> V
# ```
#
# **关键边界**:
# - 探测器噪声在 **forward 之后** 注入 LR burst，不在 HR 温度场上加噪。
# - `exact_ep06_point` 与 `physical_block_average` 是两种独立 forward 假设，指标必须分开报告。
# - synthetic `shifts.npy` 是控制先验，不是真实 stage alignment 真值。

# %%
import json

if cache.demo_skipped:
    display(Markdown("缓存未生成。请先运行 `uv run python scripts/build_ep07_cache.py --force`。"))
else:
    phantom = json.loads(PHANTOM_CONFIG.read_text(encoding="utf-8"))
    psf_cal = json.loads((PROJECT_ROOT / "configs" / "psf_calibration.json").read_text(encoding="utf-8"))
    demo = asdict(cache.demo_config)
    contract = pd.DataFrame(
        [
            {
                "parameter": "psf_profile",
                "phantom_smoke": phantom.get("psf_profile"),
                "ep07_demo": demo["psf_profile"],
                "ep09_sigma_lr_px": f"{psf_cal['psf_sigma_lr_px']:.4f}",
                "unit": "-",
                "role": "PSF profile selector; default follows EP09 Route A provisional, not final optical truth",
            },
            {
                "parameter": "noise_sigma_c",
                "phantom_smoke": phantom["noise_sigma_c"],
                "ep07_demo": demo["noise_sigma_c"],
                "ep09_sigma_lr_px": "",
                "unit": "C",
                "role": "detector noise RMS anchor from smooth adjacent-coordinate MAE",
            },
            {
                "parameter": "noise_model",
                "phantom_smoke": phantom.get("noise_model", "iid_gaussian"),
                "ep07_demo": demo["noise_model"],
                "ep09_sigma_lr_px": "",
                "unit": "-",
                "role": "LR post-forward detector residual texture model",
            },
            {
                "parameter": "psf_sigma_lr_px",
                "phantom_smoke": phantom["psf_sigma_lr_px"],
                "ep07_demo": demo["psf_sigma_lr_px"],
                "ep09_sigma_lr_px": f"{psf_cal['psf_sigma_lr_px']:.4f}",
                "unit": "LR px",
                "role": "Gaussian PSF in forward model",
            },
            {
                "parameter": "delta_T (easy)",
                "phantom_smoke": phantom["delta_T_c_by_difficulty"]["easy"],
                "ep07_demo": demo["delta_temp_c"],
                "ep09_sigma_lr_px": "",
                "unit": "C",
                "role": "structure-to-background temperature rise",
            },
            {
                "parameter": "low_freq_amplitude_c",
                "phantom_smoke": phantom["low_freq_amplitude_c"],
                "ep07_demo": demo["low_freq_amplitude_c"],
                "ep09_sigma_lr_px": "",
                "unit": "C",
                "role": "smooth thermal background variation on HR grid",
            },
            {
                "parameter": "highpass_sigma_lr_px",
                "phantom_smoke": phantom["highpass_sigma_lr_px"],
                "ep07_demo": demo["highpass_sigma_lr_px"],
                "ep09_sigma_lr_px": "",
                "unit": "LR px",
                "role": "EP06-compatible spatial background removal",
            },
            {
                "parameter": "forward_mode",
                "phantom_smoke": phantom["forward_mode"],
                "ep07_demo": cache.forward_mode,
                "ep09_sigma_lr_px": "",
                "unit": "-",
                "role": "primary LR observation operator",
            },
        ]
    )
    display(contract)

# %% [markdown]
# > **图表说明**: 上表对比正式 P0 配置 `phantom_smoke.json` 与 EP07 demo 缓存使用的物理参数。
# >
# > **数据分布**: demo 保持 `lr_shape=(64,96)`、`n_frames=16` 的轻量 smoke；噪声 RMS、PSF profile、ΔT 和低频热场幅度与 P0 配置一致。
# >
# > **核心发现**: PSF 函数仍是 EP06/EP09 使用的 Gaussian + `exact_ep06_point` forward，变化是默认 σ 从旧 0.5 LR px 切到 EP09 Route A provisional σ≈0.226 LR px；EP09 三路线尚未通过一致性门控，因此这里不是最终光学 ground truth。
