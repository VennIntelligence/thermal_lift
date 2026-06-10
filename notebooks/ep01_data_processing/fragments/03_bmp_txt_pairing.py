# %%
print(
    f"TXT: {pairing['n_txt']}  |  BMP: {pairing['n_bmp']}  |  配对: {pairing['n_paired']}"
)
print(f"孤立 TXT: {len(pairing['only_txt'])}  |  孤立 BMP: {len(pairing['only_bmp'])}")
display(pairing_detail)

# %%
display(rename_special)

# %% [markdown]
# ### 🔄 文件配对与重命名审计
#
# 1. **TXT 与 BMP 配对**：经审计，263 帧温度矩阵（TXT）与渲染图像（BMP）实现 $1:1$ 精准配对，未发生帧丢失。
# 2. **重命名冲突消歧**：针对原始命名的多解歧义（例如 `2400` 可能代表 $X=24,Y=0$ 或 $X=2,Y=40$），已通过前导零约束及物理规则完成消歧，并映射为标准 $X\_Y\_R$ 命名格式。
# 3. **时序追踪**：在缺少原始命名映射表时，标准坐标命名仅能作为空间先验，真实采集顺序仍须严格依赖 `acquisition_order`（文件修改时间 `mtime`）。
