# %% [markdown]
# ## Center 4x Temperature Comparison (v8_aa UNet vs TGV)
#
# 这一节读取 `output/ep11_dl_benchmark/v8_aa_zoom4x/` 中**同一 HR 网格、同一中心 ROI、同一 4x 显示放大**下的并排温度图。UNet 与 TGV 都通过脚本重新推理/重建，不使用 PNG 二次裁切，也不使用 bicubic 2x 作为算法对照。

# %%
ZOOM4X_OUTPUT_DIR = OUTPUT_DIR / "v8_aa_zoom4x"
REBUILD_ZOOM4X_COMMAND = (
    "cd algos/ep11_dl_benchmark && uv sync && "
    "uv run python scripts/run_unet_vs_drizzle_2x.py "
    "--checkpoint ../ep07_unet_sr/outputs/ep07_v8_aa/checkpoint_step_040000.pt "
    "--baseline-name 'TGV best 2x' "
    "--output-dir ../../output/ep11_dl_benchmark/v8_aa_zoom4x "
    "--zoom 4.0 --center-fraction 0.3333333 --overlap 128 "
    "--patch-size-hr 256 --device cuda:0 --allow-cuda0 "
    "--reconstruct-tgv --workers 4"
)


def show_zoom4x_fig(name: str) -> None:
    path = ZOOM4X_OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP11 4x figure: {path}\nRun: {REBUILD_ZOOM4X_COMMAND}")
    display(NotebookImage(filename=str(path), retina=True))


print(f"EP11 4x output: {ZOOM4X_OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"Rebuild command: {REBUILD_ZOOM4X_COMMAND}")

# %%
show_zoom4x_fig("unet_vs_tgv_2x_center_zoom4x_temperature.png")

# %% [markdown]
# > **图表说明**: 左右两 panel 都是完整 HR 温度场（960×1280）上取几何中心 `1/3 × 1/3` ROI，再统一做 `4x` 显示放大；色阶取两幅放大图的共同 1–99 percentile。左图为 EP07 `v8_aa` checkpoint@step40000 的 UNet 2x 温度输出；右图为 EP10 MAP-TGV best 2x 温度图（由 highpass 重建结果加回参考低频背景）。
# >
# > **怎么看**: 重点比较中心交汇区、八向走线和焊盘内轮廓是否更连续。温度域适合看整体热场是否自然，以及内部结构边缘是否比 TGV 更清楚；若 UNet 仅边缘更亮而内部仍糊，需要谨慎解释。
# >
# > **异常是否正常**: 两 panel 尺寸和裁切位置一致，可以直接并排看结构；这不是 4x SR 重建，只是显示放大。TGV 在 CPU 上重建，UNet 在空闲 GPU 上推理，耗时差异不代表算法优劣。
# >
# > **核心发现**: 这是面向中心结构的公平温度域对照，用来判断 v8_aa@40000 相对经典 MAP-TGV 是否在 contour-level 上带来可读的结构增益。

# %%
show_zoom4x_fig("unet_vs_tgv_2x_center_zoom4x_highpass.png")

# %% [markdown]
# > **图表说明**: 与上图相同 ROI/放大规则，但两 panel 都转到 highpass 域（`sigma=5.0` 背景扣除），并使用对称 99th-percentile 色阶。
# >
# > **怎么看**: 红/蓝是相对于局部背景的正/负结构响应，白底通常接近零变化。这里更适合看细线、直角边和重复纹理是否更稳定；highpass 会放大噪声和振铃，不能单独作为 SR 成功证据。
# >
# > **异常是否正常**: highpass 图边缘会比温度图更“锐”，这是域变换预期行为，不代表真实温度热点。
# >
# > **核心发现**: highpass 并排图是 EP11 的传统主证据；若 UNet 在 highpass 与 temperature 两域都呈现更清楚的中心结构，才更值得继续投入 v8_aa 方向。
