# %% [markdown]
# ## B7. 六项合成 Total 损失（demo 数值）
#
# 下表展示了我们模拟的 demo 场景下各个 Loss 分量的原始数值、配方权重、加权贡献以及所占的百分比。

# %%
import pandas as pd
from IPython.display import display, Markdown

bd = manifest["loss_breakdown"]
recipe = manifest["recipe"]
rows = [
    ("lf (低频 L1)", bd["lf"], recipe["lf_weight"]),
    ("hf (高通 L1)", bd["hf"], recipe["hf_weight"]),
    ("edge (Sobel 细+粗)", bd["edge"], recipe["edge_weight"]),
    ("forward (物理前向一致性)", bd["forward"], recipe["forward_weight"]),
    ("nll (异方差 NLL)", bd["nll"], recipe["nll_weight"]),
    ("hf_detail (反 Coverage 细节)", bd["hf_detail"], recipe["hf_detail_weight"]),
]
df = pd.DataFrame(rows, columns=["Loss子项", "原始值", "权重"])
df["加权贡献"] = df["原始值"] * df["权重"]
df["占比 %"] = 100 * df["加权贡献"] / bd["total"]
display(df.style.format({"原始值": "{:.6f}", "权重": "{:.3g}", "加权贡献": "{:.6f}", "占比 %": "{:.1f}"}))
display(Markdown(f"**demo total = {bd['total']:.6f}**（TCForge 中心 4x patch）"))

# %%
save_fig("16_total_loss_recipe_4x.png")

# %% [markdown]
# ## 总结：EP12 4x 算法核心调参指导
#
# | 损失组件 | 作用与物理机制 | 默认权重与调参建议 |
# |---|---|---|
# | **lf (Low Frequency)** | 锚定 DC 绝对温度，防止温度偏离与浮动 | **1.0** (保持开启，保护绝对读数) |
# | **hf (High Frequency)** | Coverage 加权引导重建主要 4x 结构轮廓 | **0.3** (调大可使核心边缘更陡峭) |
# | **edge (Sobel)** | 细/粗多尺度约束，增强轮廓连接性与连续性 | **0.1** (0.05~0.15 稳定梯度，防止边缘发毛) |
# | **forward (Consistency)** | PSF 卷积并 Pooling 回 1x 数据保真，最核心物理约束 | **0.2** (可增至 0.2~0.5 保证符合实测输入) |
# | **nll (Heteroscedastic)** | 不确定性对数似然，自适应吸收对齐误差与高噪区 | **0.05** (必须与 `log_var` 一起开启，平抑边缘震荡) |
# | **hf_detail (Weak Edges)** | 反 Coverage 加权，弱纹理与稀疏覆盖区保真 | **0.3** (对信噪比低/边缘无覆盖的细节尤为关键) |
#
# ### 核心结论
# - **8通道输入融合**为 4x 网络提供了强大的亚像素位移先验，Drizzle 的存在使得网络能够从 scatter-add 中直接汲取高频对齐特征，而不需要从零虚构。
# - **不确定度预测 log_var 与物理前向一致性 forward** 的引入是 4x 算法成功的双引擎。前者避免了过度拟合低信噪比噪声，后者确保了物理成像一致性。
