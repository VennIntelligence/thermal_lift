# %% [markdown]
# ## Step 4 — ESF/CRB 定位精度只定义 Anchor 置信度
#
# 1D ESF 的 Cramér-Rao Bound 可以说明“某个局部边缘在给定噪声、温差、PSF 和相位覆盖下，边缘位置能否作为稳定 anchor”。它不能证明整体形状已经被重建，也不能替代 EP05 的 contour/shape evidence。
#
# 可以把 CRB 理解成“在模型完全正确、噪声假设成立时，任何无偏估计器也很难优于这个位置误差”。因此它适合做乐观的理论下界，而不是实测性能保证。真实数据里还会有热漂移、配准误差、非理想 PSF、局部纹理混叠等因素。

# %%
crb_table = build_crb_localization_table(
    NOISE_SIGMA,
    contrasts_c=(0.3, 0.7, 1.0, 2.0),
    sigma_values_px=(0.5, 1.0),
    n_frames=16,
    phase_coverage_px=1.0,
)
crb_table.to_csv(OUTPUT_DIR / "crb_esf_localization_bounds.csv", index=False)

crb_display = crb_table.pivot_table(
    index=["delta_t_c", "sigma_psf_px"],
    columns="model",
    values="crb_px",
).reset_index()
display(crb_display)

nominal_single = float(
    crb_table[
        crb_table["delta_t_c"].eq(0.7)
        & crb_table["sigma_psf_px"].eq(0.5)
        & crb_table["model"].eq("single_frame")
    ]["crb_px"].iloc[0]
)
nominal_multi = float(
    crb_table[
        crb_table["delta_t_c"].eq(0.7)
        & crb_table["sigma_psf_px"].eq(0.5)
        & crb_table["model"].eq("16_frame_known_shift")
    ]["crb_px"].iloc[0]
)
wide_edge_multi = float(
    crb_table[
        crb_table["delta_t_c"].eq(2.0)
        & crb_table["sigma_psf_px"].eq(1.0)
        & crb_table["model"].eq("16_frame_known_shift")
    ]["crb_px"].iloc[0]
)
print(f"Nominal single-frame CRB (DeltaT=0.7 C, sigma=0.5 px): {nominal_single:.4f} px")
print(f"Nominal 16-frame CRB (known 1 px phase coverage): {nominal_multi:.4f} px")
print(f"Wide but strong edge 16-frame CRB (DeltaT=2.0 C, sigma=1.0 px): {wide_edge_multi:.4f} px")

# %% [markdown]
# > **数据说明**: 表格扫描局部温差 `DeltaT=0.3/0.7/1.0/2.0 C` 和等效边缘宽度 `sigma=0.5/1.0 px`，比较单帧与 16 帧已知相位覆盖下的 ESF 位置 CRB。
# > **怎么读**: 行里的 `delta_t_c` 越大，表示边缘两侧温差越强；`sigma_psf_px` 越大，表示边缘越宽或越模糊；CRB 数值越小，表示理论上边缘位置越容易估准。打印的三个关键数值只是代表性场景，用于建立量级直觉。
# > **正常/异常理解**: 正常模式是温差越高，CRB 越低；边缘越宽，CRB 越高；多帧已知相位覆盖能降低 anchor 位置不确定度。若某个组合给出很低 CRB，也只说明该局部模型下有希望稳定定位，不说明整幅图都能达到这个误差。
# > **核心发现**: ESF/CRB 支持把高 SNR 边缘作为 alignment anchor 和质量门控基准；它不支持把一个局部 edge localization 数值写成完整芯片内部形状已经重建。CRB 是 EP05 的筛选依据，不是 EP05 的验收结果。

# %%
fig = plot_crb_esf_localization(
    crb_table,
    noise_sigma_c=NOISE_SIGMA,
    seed=RNG_SEED,
)
save_fig(fig, "crb_esf_localization_anchor.png")

# %% [markdown]
# > **图表说明**: 左图模拟带噪声 ESF 的局部边缘位置观测；右图展示不同温差、PSF 宽度和帧数条件下的 CRB。左图帮助建立“边缘被 PSF 拉宽并被噪声扰动”的直观图像，右图给出理论误差量级。
# > **怎么读**: 左图中曲线越陡、噪声越小，边缘位置越容易确定；右图中柱/点越低，理论定位越精确。比较单帧和多帧时，应理解为“在相位覆盖已知且对齐模型成立时”的上限条件。
# > **正常/异常理解**: 高温差、窄边缘和有效多帧相位覆盖会给出更低位置下界；低温差或宽边缘会降低 anchor 置信度。若真实 EP05 对齐结果比 CRB 差很多，这通常提示存在模型外误差，而不是 CRB 本身失效。
# > **核心发现**: CRB 的正确用法是给 EP04/EP05 提供局部 anchor confidence 与 quality gate，而不是用来替代 SR 输出的 contour/shape 一致性验证。理论下界必须被真实数据 POC 校验。
