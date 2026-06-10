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
# 此合成验证表（Synthetic Table）汇总了 Shift-and-Add（SAA）、Iterative Back-Projection（IBP）以及 Maximum A Posteriori with Total Variation（MAP-TV）三种重建算法在合成图像序列上的冒烟测试（Smoke Test）指标。合成场景包含已知的退化过程（前向成像模型 $y_k = D H W_k x + n_k$），主要用于验证代码逻辑链路、前向与逆向位移约定的自洽性以及迭代优化过程的数值稳定性，并非用来替代真实热像数据的重建结论。
# 表中如峰值信噪比（PSNR）等图像保真度指标的提升，用以衡量算法在理想退化条件下的数学收敛性。`selected_lambda`、`iterations` 及 `finite` 等字段记录了算法的运行参数和数值完整性。由于不同算法的前向模型和正则化机制有所差异，部分特有字段合并后在 pandas DataFrame 中显示为 `NaN` 属于正常现象，并不代表重建图发生数值异常。合成验证的通过，确立了 EP06 核心算法管道无方向性偏差或数值爆炸的底线；算法在真实物理场景下的超分辨率有效性，仍需在后续章节中通过主扫描会话的高通滤波图像、原始温度图像、split-half 交叉验证以及伪影程度进行严谨判定。
