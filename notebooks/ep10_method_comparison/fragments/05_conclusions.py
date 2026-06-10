# %% [markdown]
# ## Conclusions

# %%
if not summary_table.empty:
    best_lines = []
    for _, row in summary_table.iterrows():
        best_lines.append(
            f"- **{row['method']}**: `{row['variant']}`; "
            f"split-half NRMSE `{row['split_half_nrmse']:.6f}`, "
            f"artifact `{row['artifact_score']:.6f}`, "
            f"holdout MSE `{row['holdout_mse']:.6g}`, "
            f"raw-control corr `{row['raw_control_corr']:.6f}`"
        )
    display(Markdown("**Current EP10 best candidates**  \n" + "  \n".join(best_lines)))
else:
    display(Markdown("**Current EP10 best candidates**: missing until at least one sweep CSV is generated."))

# %% [markdown]
# > **数据说明**: 结论汇总只使用已经存在的 EP10 Drizzle、MAP-TV 和 TGV 产物；Notebook 没有重新计算 SR，也没有改动各算法输出。
# >
# > **怎么看**: 当前 best 候选是进入后续人工裁决和报告写作的候选池。优先级不是“某一列最低”，而是 split-half 稳定、artifact 风险、holdout residual、raw-control 一致性和中心 ROI 轮廓观感的合取。
# >
# > **异常是否正常**: Drizzle、MAP-TV、TGV 的参数和优化目标不同，所以 best 行不是严格同分布实验；这里的横向比较用于 POC 决策，而不是统计显著性声明。任何缺失产物都应在算法输出目录补生成，而不是由本 Notebook 启动长实验。
# >
# > **核心发现**: EP10 notebook 已转为三算法可视化对比入口。下一步应在正式报告中明确选用候选的理由，并把 highpass 视觉证据与 raw-control/coverage 控制证据一起呈现。
