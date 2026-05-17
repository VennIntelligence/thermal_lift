# %% [markdown]
# ## 2.4 坐标覆盖率可视化
#
# 16×16 坐标网格的 repeat 计数热力图。颜色越深 = repeat 越多。

# %%
grid = compute_coverage_grid(coord_map, list(VALID_COORDS))
plot_coverage_heatmap(grid, list(VALID_COORDS),
                      save_path="coordinate_coverage_map.png", save_fn=save_fig)

# %% [markdown]
# > **图表说明**: 16×16 坐标网格热力图，横轴为 X 坐标 (µm)，纵轴为 Y 坐标 (µm)，
# > 颜色编码每个 (X,Y) 位置的重复测量次数 (0/1/2/3)。
# >
# > **数据分布**: 绝大多数坐标 (249/256) 仅有 1 次测量 (R=0)。
# > 4 个坐标拥有完整 3-repeat: (0,0)(2,0)(6,0)(8,0)，均位于 Y=0 行。
# > 3 个坐标完全缺失 (值=0): (14,6)(16,6)(16,16)。
# >
# > **核心发现**: 坐标覆盖率 253/256 = 98.8%，数据集近乎完整。
# > 3-repeat 坐标仅 4 个（非旧项目声称的 6 个），可用于 EP02 重复定位精度评估。
