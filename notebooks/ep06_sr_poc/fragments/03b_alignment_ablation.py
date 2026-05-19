# %% [markdown]
# ## 2.1 Alignment Strategy Ablation
#
# Alignment ablation 用来回答一个很具体的问题：EP06 的 contour-level 增益是否依赖某一种位移策略。这里不把 stage command 当成真值，而是比较不同 alignment strategy 在同一 SR/evaluation 管线里的表现，并把 ablation 作为质量门控。

# %%
ablation_script = PROJECT_ROOT / "scripts" / "run_ep06_alignment_ablation.py"
ablation_figures = discover_outputs(ABLATION_OUTPUT_PATTERNS["figures"], ABLATION_OUTPUT_DIR)
ablation_tables = discover_outputs(ABLATION_OUTPUT_PATTERNS["tables"], ABLATION_OUTPUT_DIR)

print(f"Alignment ablation script: {relative(ablation_script)}")
print(f"Script present: {ablation_script.exists()}")
print(f"Ablation output dir: {relative(ABLATION_OUTPUT_DIR)}")
print(f"Discovered ablation figures: {len(ablation_figures)}")
print(f"Discovered ablation tables: {len(ablation_tables)}")

if not ablation_figures and not ablation_tables:
    print("No alignment ablation artifacts found in output/ep06_alignment_ablation.")
    print("Generate them with:")
    print("  uv run python scripts/run_ep06_alignment_ablation.py")
    if not ablation_script.exists():
        print("Current checkout does not contain that script yet; sync or add it before running the command.")

for table_path in ablation_tables:
    print(f"\nTable: {relative(table_path)}")
    table = pd.read_csv(table_path)
    display(table.head(20).round(4))

for figure_path in ablation_figures:
    print(f"\nFigure: {relative(figure_path)}")
    display(NotebookImage(filename=str(figure_path)))

# %% [markdown]
# > **图表说明**: 本节自动读取 `output/ep06_alignment_ablation/` 中的 CSV 和 PNG。预期产物由 `uv run python scripts/run_ep06_alignment_ablation.py` 生成，用于比较 default contour refined、NCC init、tuned contour refined 和 filename affine 在同一 SAA 评估口径下的差异；如果当前 checkout 还没有脚本或产物，上一个 cell 会给出复现命令并继续执行。
# >
# > **怎么看**: 表格里若出现 `split_half_nrmse`、`artifact_score`、`contour_chamfer_lr_px`、`mean_gradient`、`p95_gradient` 等列，方向性与后文一致：split-half NRMSE / artifact / Chamfer 通常越小越稳，gradient 只表示边缘响应更强，不能单独判胜。图像里则重点看同一 ROI 的边缘位置是否稳定、轮廓是否连续、背景是否出现条纹或局部过冲。
# >
# > **正常/异常**: 产物缺失不是 notebook 错误，只说明 ablation 尚未运行或脚本尚未同步。若 ablation 表中出现 `NaN`，先确认是否是某个策略没有对应字段，而不是直接判定算法输出有 NaN；真实数值健康仍以 `finite` 字段和输出数组检查为准。
# >
# > **核心发现**: Alignment ablation 是 EP06 结论边界的一部分。当前正式 ablation 显示 default contour refined 在 split-half 上最稳，tuned refined 在 artifact proxy 上最低但 split-half 略差，NCC init 保留更完整的高倍率 phase prior，filename affine 是强 control 而不是最终 truth。
