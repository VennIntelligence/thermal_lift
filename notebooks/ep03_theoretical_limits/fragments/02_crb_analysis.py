# %% [markdown]
# ## Step 1 — Pixel Pitch、空间分辨率与 2x Grid
#
# `10 um/pixel` 是 TXT 温度矩阵的 detector sampling pitch；`20 um` 是当前系统的空间分辨率；`5 um` 目标意味着输出网格至少要到 2x sampling grid。这里先把三个量画在同一个物理坐标轴上，避免把采样 pitch、空间分辨率和 SR 显示倍率混为一谈。
#
# 对不熟悉红外成像的读者，关键是不要把“采样更密”和“光学真的分辨得更细”混在一起。2x SR 输出网格可以把结果显示在 5 um 间隔上，但这只是重建网格；它必须通过多帧相位、对齐质量和结构一致性证明有新增 contour 信息，不能直接改写系统空间分辨率。

# %%
sampling_resolution = build_sampling_resolution_table(
    detector_pitch_um=DETECTOR_PITCH_UM,
    spatial_resolution_um=SPATIAL_RESOLUTION_UM,
    target_grid_um=TARGET_GRID_UM,
)
sampling_resolution.to_csv(OUTPUT_DIR / "sampling_resolution_distinction.csv", index=False)

display(sampling_resolution)

fig = plot_sampling_resolution_diagram(
    sampling_resolution,
    detector_pitch_um=DETECTOR_PITCH_UM,
    spatial_resolution_um=SPATIAL_RESOLUTION_UM,
    target_grid_um=TARGET_GRID_UM,
)
save_fig(fig, "sampling_resolution_distinction.png")

# %% [markdown]
# > **图表说明**: 表格和示意图把 detector sample、当前 spatial resolution、2x grid 和 4x grid 放在同一条物理长度轴上。表格适合查具体数值，示意图适合看这些尺度之间的相对距离。
# > **怎么读**: 先看 detector sample 的刻度间隔，它表示原始温度矩阵相邻像素中心间隔；再看 spatial resolution 的跨度，它表示光学系统已经把小于该尺度的结构混合到一起；最后看 2x/4x grid，它们只是候选输出网格密度。
# > **正常/异常理解**: 正常结论是 20 um resolution 比 10 um pitch 更宽，因此单帧图像已经过采样了部分光学模糊。若只看到更密的输出网格、却没有对齐和结构证据，那是显示/插值，不是 SR 证据。
# > **核心发现**: EP03 支持把 2x 作为 contour-level SR POC 的默认输出网格；它不支持把插值后的 5 um grid 直接写成 5 um 计量级空间分辨率。真实增益必须在 EP05 用主 session 数据验证。

# %%
measurement_script = PROJECT_ROOT / "scripts" / "measure_pixel_size.py"
subprocess.run(
    [sys.executable, str(measurement_script)],
    cwd=PROJECT_ROOT,
    check=True,
    capture_output=True,
    text=True,
)

with open(OUTPUT_DIR / "pixel_size_measurement.json", encoding="utf-8") as f:
    pixel_measurement = json.load(f)

axis_result = pixel_measurement["axis_method"]
contour_result = pixel_measurement["contour_cross_check"]
resolution_result = pixel_measurement["resolution_distinction"]

pixel_pitch_summary = pd.DataFrame(
    [
        {
            "measurement": "BMP mm-axis ticks",
            "value_um_per_pixel": float(axis_result["pixel_size_mean_um"]),
            "evidence": "Rendered data crop is 640 x 480; 1 mm tick spacing is 100 rendered px",
        },
        {
            "measurement": "TXT/BMP contour cross-check",
            "value_um_per_pixel": float(contour_result["pixel_size_mean_um"]),
            "evidence": f"Outer-mask IoU={contour_result['mask_iou']:.4f}",
        },
        {
            "measurement": "Current spatial resolution",
            "value_um_per_pixel": float(resolution_result["current_spatial_resolution_um"]),
            "evidence": "Calibrated resolving scale, not detector pitch",
        },
    ]
)
pixel_pitch_summary.to_csv(OUTPUT_DIR / "pixel_pitch_measurement_summary.csv", index=False)

display(pixel_pitch_summary)
display(NotebookImage(filename=str(OUTPUT_DIR / "pixel_size_measurement.png")))

# %% [markdown]
# > **图表说明**: 表格汇总两种 pitch 校验和一条 spatial-resolution 参考；`pixel_size_measurement.png` 左侧标出 BMP 导出图中的 640×480 数据绘图区和 mm 刻度，右侧用测得 pitch 重新标注 TXT 温度矩阵，并叠加外轮廓。
# > **怎么读**: 表格中前两行应互相接近，因为它们都在验证 detector pitch；第三行数值更大，含义是系统能分辨结构的尺度。图中如果 TXT 外轮廓和 BMP 轮廓能较好重合，说明 pitch 换算与数据尺寸是一致的。
# > **正常/异常理解**: 正常情况是坐标轴方法和 TXT/BMP 外轮廓交叉验证都落在约 10 um/pixel。如果两者明显偏离，应优先怀疑导出图裁剪、轴刻度识别或坐标换算，而不是立刻修改 SR 算法。
# > **核心发现**: `stage_calibration.json` 中的 `pixel_size_um=10.0` 是 detector-pixel 单位换算；后续 SR 设计应把 20 um 当作系统传递函数边界，而不是把它写进像素 pitch。这一节只校准尺度语言，不证明 SR 已经有效。
