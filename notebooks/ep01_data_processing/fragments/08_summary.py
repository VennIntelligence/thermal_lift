# %% [markdown]
# ## 4. SR 数据基础汇总

# %%
from thermal_core.ep01 import (
    translate_ep01_frame_audit_contract_zh,
    translate_ep01_summary_table_zh,
)

display(cache.summary_table)

# %%
display(translate_ep01_summary_table_zh(cache.summary_table))


# %%
display(cache.boundary_jump_table.round(3))

# %% [markdown]
# > **数据解读**：边界跳变表量化了各 Session 之间的均温落差，最大跳变幅度达 3.55°C（约 49 倍噪声底），从定量角度证实了跨 Session 混合数据的危害性。


# %%
display(cache.frame_audit_contract)

# %%
display(translate_ep01_frame_audit_contract_zh(cache.frame_audit_contract))


# %%
print(f"审计数据: {OUTPUT_DIR / 'frame_audit.csv'} ({len(df)} frames)")
print(f"SR 汇总: {OUTPUT_DIR / 'sr_data_basis_summary.csv'}")
print(f"审计报告: {REPORT_DIR / 'audit_report.md'}")
print("图表缓存:", ", ".join(EP01_FIGURE_ARTIFACTS))

# %% [markdown]
# ### 📜 EP01 交付与下游算法数据契约
#
# 本阶段完成了对芯片红外热像微扫描全部原始数据的健康度与物理状态审计，建立并交付了下游算法必须遵守的**数据契约**（保存在 [frame_audit.csv](file:///home/ujs/mycode/thermal_lift/output/ep01_data_processing/frame_audit.csv) 中）：
#
# 1. **物理一致性**：下游算法（如 EP02 位移标定、EP04 亚像素对齐和 EP10/11 超分辨率重建）必须统一以 `is_main_session == True` 过滤输入数据，仅保留热平衡状态下的 255 帧黄金帧。
# 2. **时序唯一指引**：舍弃文件名排序，在时序相关的运算中必须使用物理采集序号 `acquisition_order`。
# 3. **空间盲区声明**：下游重建算法在填充网格时，须对空间缺失坐标 $(14, 6)$、$(16, 6)$、$(16, 16)$ 预留插值退化及 Mask 标记，防范虚假轮廓生成。
