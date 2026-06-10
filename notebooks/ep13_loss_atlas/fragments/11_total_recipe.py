# %% [markdown]
# ## B6. 四项合成 total（demo 数值）

# %%
import pandas as pd

bd = manifest["loss_breakdown"]
recipe = manifest["recipe"]
rows = [
    ("mse", bd["mse"], recipe["mse_weight"]),
    ("highpass", bd["highpass"], recipe["highpass_weight"]),
    ("edge", bd["edge"], recipe["edge_weight"]),
    ("ssim", bd["ssim"], recipe["ssim_weight"]),
]
df = pd.DataFrame(rows, columns=["loss项", "原始值", "权重"])
df["加权贡献"] = df["原始值"] * df["权重"]
df["占比 %"] = 100 * df["加权贡献"] / bd["total"]
display(df.style.format({"原始值": "{:.6f}", "权重": "{:.3g}", "加权贡献": "{:.6f}", "占比 %": "{:.1f}"}))
display(Markdown(f"**demo total = {bd['total']:.6f}**（TCForge 中心 patch）"))

# %%
save_fig("16_total_loss_recipe.png")


# %% [markdown]
# > **图表说明**: 表格 + 柱状图展示 highpass 对 total 的主导程度。
# >
# > **核心发现**: total 下降优先反映 highpass 是否变好，不是温度场是否平滑。
