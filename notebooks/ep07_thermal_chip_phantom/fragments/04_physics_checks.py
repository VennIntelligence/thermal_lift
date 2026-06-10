# %% [markdown]
# ## 3. 热场 / PSF / 噪声 / SNR 检查
#
# 这一节专门验收物理退化链的四项核心假设：
#
# | 组件 | 检查内容 | 通过意味着什么 |
# |---|---|---|
# | 热场 | mask 温升 + 低频背景分解 | HR 真值可解释、可分解 |
# | PSF | Gaussian kernel + 边缘梯度软化 | forward 前模糊与 EP09 provisional σ≈0.226 LR px 一致 |
# | 噪声 | LR residual 纹理 / RMS / 真实对照 | 噪声在 forward 后注入且幅度同量级 |
# | SNR | difficulty 预算 vs 噪声底 | 各难度档位的可观测性风险可量化 |

# %% [markdown]
# ### 3.1 热场分解

# %%
if cache.demo_skipped:
    display(Markdown("物理检查图不可用。"))
else:
    show_fig("demo_thermal_field_decomposition.png")

# %% [markdown]
# > **图表说明**: 四宫格分别显示结构 mask、仅结构温升、仅低频背景、完整 HR 温度；右下角 inset 为同一行的剖面对比。
# >
# > **怎么看**: 红/蓝背景图以零为中心，白区表示接近零变化；完整温度 = 结构分量 + 低频分量。
# >
# > **核心发现**: 热场边缘比二值 mask 更平滑，符合「结构发热 + 空间低频漂移」的物理模型，而非锐利几何阶跃。

# %% [markdown]
# ### 3.2 PSF 模糊

# %%
if not cache.demo_skipped:
    show_fig("demo_psf_blur_check.png")

# %% [markdown]
# > **图表说明**: PSF kernel（HR 网格）、PSF 前/后 HR 温度、以及无噪声 LR 第 0 帧；inset 对比边缘剖面软化。
# >
# > **怎么看**: PSF 后边缘梯度应低于 PSF 前；σ_HR = σ_LR × scale（demo 默认约 0.226 × 2 = 0.451 HR px）。旧 0.5 LR px 现在只作为 `legacy_upper` 压力档。
# >
# > **核心发现**: 轮廓恢复难度来自 PSF 衰减，而非 mask 本身不够锐利；当前 σ 跟随 EP09 Route A forward residual，但 EP09 门控未通过，不能写成最终光学真值。

# %% [markdown]
# ### 3.3 探测器噪声

# %%
if not cache.demo_skipped:
    show_fig("demo_noise_check.png")

# %% [markdown]
# > **图表说明**: 同一 LR 帧的 clean forward、加噪后观测、噪声残差图和残差直方图。
# >
# > **怎么看**: `iid_gaussian` 是白噪声 baseline；默认 `mixed` 会加入低频 FPN 和列偏置，但总 RMS 仍归一到 0.0724°C。残差图红/蓝不代表升温/降温，只是相对 clean 帧的扰动。
# >
# > **核心发现**: 噪声只加在 LR burst 上，与当前训练池生成器 `generate_training_pool.py` 行为一致；大块斑块若来自 `low_freq_amplitude_c`，属于热场低频背景，不是探测器噪声。

# %% [markdown]
# ### 3.4 真实 vs 合成残差纹理

# %%
if not cache.demo_skipped:
    show_fig("demo_noise_real_vs_synthetic.png")
    display(compact_table(cache.noise_model_checks, [
        "noise_model", "std_c", "target_sigma_c", "lag1_column_corr", "column_bias_std_c", "row_bias_std_c"
    ]))

# %% [markdown]
# > **图表说明**: 左上是真实主 session 单帧减去 σ=5 LR px 空间 Gaussian 背景后的 residual crop；右上是合成 noisy-clean residual；下方对比列均值和归一化径向功率谱。
# >
# > **怎么看**: 真实 residual 含热漂移、alignment/model mismatch 和真实探测器纹理，不是纯噪声 ground truth；合成侧只要求空间纹理统计和幅度同量级。表中 std 越接近 0.0724°C 越符合 RMS 锚定，lag-1/列均值 proxy 用来区分白噪声与相关纹理。
# >
# > **核心发现**: `mixed` 比纯 `iid_gaussian` 更能表达 FPN/列条纹这类弱空间相关，但它仍是轻量统计模型，不复现真实帧的像素级残差。

# %% [markdown]
# ### 3.5 SNR 预算

# %%
if not cache.demo_skipped:
    show_fig("demo_snr_budget.png")
    display(compact_table(cache.snr_budget, [
        "difficulty", "delta_t_c", "input_snr", "effective_snr_2x", "risk_band", "passes_3x_noise", "passes_5x_noise"
    ]))

# %% [markdown]
# > **图表说明**: 左图对比各 difficulty 的 input SNR 与 PSF 衰减后的 effective SNR（2x）；右图按 risk band 着色。
# >
# > **怎么看**: effective SNR = ΔT × MTF(0.5) / noise；≥3 为 borderline，≥5 为 observable。这是必要风险指标，不是 SR 成功证明。
# >
# > **核心发现**: easy 档在 2x 下仍属 borderline/observable 区间；stress 档更接近 noise-dominated，应用作算法压力测试而非默认交付档。

# %% [markdown]
# ### 3.6 数值门控汇总

# %%
if not cache.demo_skipped:
    checks = cache.physics_checks.copy()
    display(compact_table(checks, ["component", "check", "value", "unit", "expected", "pass"]))
    n_pass = int(checks["pass"].sum())
    print(f"Physics checks: {n_pass}/{len(checks)} passed")

# %% [markdown]
# > **数据说明**: `physics_checks.csv` 汇总热场、PSF、噪声、SNR 的自动验收项。
# >
# > **异常是否正常**: 小尺寸 demo 的 measured noise std 允许略有偏差（阈值 ±0.03°C）；若多项 fail，应优先检查 TCForge 版本和缓存是否过期。
# >
# > **核心发现**: 全部 pass 是 demo 数据包可用于 SR regression 的前置硬门槛。
