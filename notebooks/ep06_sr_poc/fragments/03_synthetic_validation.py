# %% [markdown]
# ## 2. Synthetic Validation
#
# 合成验证只检查代码链路和 forward/backward 位移约定是否自洽。真实结论仍以后续主 session 的直接视觉对比、split-half 和控制轨为准。

# %%
import json

synthetic_rows = []
for name in [
    "saa_synthetic_validation.json",
    "ibp_synthetic_validation.json",
    "map_tv_synthetic_validation.json",
]:
    path = OUTPUT_DIR / name
    if not path.exists():
        continue
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    synthetic_rows.append({"file": name, **payload})

synthetic_table = pd.DataFrame(synthetic_rows)
display(synthetic_table)

# %% [markdown]
# > **数据说明**: 这张 synthetic table 汇总 SAA、IBP、MAP-TV 三条算法脚本在合成小图上的 smoke test 结果。合成图有已知的生成过程，所以它主要用来检查代码链路、forward/backward shift 约定、迭代是否稳定，而不是用来替代真实红外数据结论。
# >
# > **怎么看**: `psnr` 或类似误差指标越好，通常说明算法能在这个简化场景里把合成观测还原得更接近已知答案；`selected_lambda`、`iterations`、`finite` 等字段用于确认脚本有没有正常跑完，以及 MAP-TV 的正则强度选择是否有记录。
# >
# > **正常/异常**: 第一张表里如果看到某些单元格是 `NaN`，优先理解为不同 JSON 文件的字段名不完全一致：例如某个算法没有 `selected_lambda`，pandas 合并成表时就会把该列填成 `NaN`。这不等于真实重建图里有 NaN，也不等于算法输出坏掉；真正的数值有效性要看 `finite` 字段和后续真实数据评估。
# >
# > **核心发现**: 合成验证只能说明 EP06 脚本没有明显的方向约定错误或数值爆炸。真实 POC 是否成立，仍以后面的主 session highpass 图、raw-temperature 中心检查、split-half 和伪影审计为准。
