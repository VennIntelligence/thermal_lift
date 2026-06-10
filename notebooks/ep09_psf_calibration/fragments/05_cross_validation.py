# %% [markdown]
# ## Cross-Validation 与 4x 门控
#
# 最终门控同时检查路线一致性、Route A 残差曲线清晰度、置信区间宽度和 ESF 有效 segment 数。4x 判决必须服从这些门控，而不是只看最乐观的单一路线。

# %%
if not route_table.empty:
    display(route_table.round(4))

gate_rows = []
if summary:
    gate_rows = [
        {"gate": "Route consistency <= 0.05 px", "pass": summary["routes_consistent_within_0p05_px"]},
        {"gate": "Forward residual clear minimum", "pass": summary["forward_curve_single_minimum_gate_pass"]},
        {"gate": "95% CI width <= 0.10 px", "pass": summary["ci_width_gate_pass"]},
        {"gate": "ESF valid segments >= 10", "pass": summary["esf_valid_segment_gate_pass"]},
        {"gate": "Overall EP09 gate", "pass": summary["overall_gate_pass"]},
    ]
    display(pd.DataFrame(gate_rows))
    display(
        Markdown(
            f"**Final sigma (Route A primary, provisional)**: {summary['final_sigma_lr_px']:.4f} LR px  \n"
            f"**95% CI**: [{summary['final_ci95_lr_px'][0]:.4f}, {summary['final_ci95_lr_px'][1]:.4f}] LR px  \n"
            f"**Route spread**: {summary['route_spread_lr_px']:.4f} LR px  \n"
            f"**4x verdict**: `{summary['four_x_verdict']}`"
        )
    )

# %% [markdown]
# > **数据说明**: Route table 来自 `route_sigma_summary.csv`，gate table 来自 `calibration_summary.json`。全局配置 `configs/psf_calibration.json` 已写入 Route A 的 provisional effective sigma，并明确标记为 `provisional_needs_review`。
# >
# > **怎么看**: `Route consistency` 是 P0 门控，必须通过才可称为精确标定。`Forward residual clear minimum` 不只看是否有数学最小值，还要求 residual depth 足够明显。
# >
# > **异常是否正常**: 本次 CI width 约 0.032 px，小于 0.10 px；Route A forward curve 也清除单路线门控。但 route spread 仍约 1.010 px，表示 bootstrap 只量化了 Route A 内部不确定性，没有覆盖模型系统误差。
# >
# > **核心发现**: EP09 没有清除 4x 启动门控。当前最稳妥结论是：2x 工作继续；4x/PnP 不应作为物理可行主线启动，除非后续先解释 ESF 与 forward effective sigma 的冲突。
