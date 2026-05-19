# %% [markdown]
# ## 6. Alignment Tuning Study / 对齐调参研究
#
# 本节补充 EP05 alignment tuning 的实验过程、调参曲线、候选比较和结论边界。它读取 `output/ep05_alignment_tuning` 或 `output/ep05_alignment_tuning_study` 中已经生成的 CSV/PNG；如果这些产物不存在，Notebook 会显示需要运行的脚本命令，而不是抛错中断。
#
# 调参过程分三步：
#
# 1. **96-frame screening**: 在主 session 的前 96 帧上扫 ROI size、edge percentile、Chamfer refinement radius/step，先筛掉明显不稳定的组合。
# 2. **255-frame finalist run**: 对筛出的候选跑完整 255 帧 contour alignment，检查 held-out Chamfer median/P90、gradient correlation 和 shift span。
# 3. **capacity re-score**: 用同一套 255 帧对候选重新跑 alignment method comparison 和 2x phase-bin coverage，确认 tuning 选择没有破坏 EP06 所需的 phase diversity。
#
# 这节只选择 EP06 的 alignment gate 参数，不证明 SR 已经成功。NCC init 仍保留为连续位移 prior；contour refinement 是局部锚定和质量门控；stage/filename 仍是 prior/control，不是 ground truth。

# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, display

from thermal_core.ep05 import (
    alignment_tuning_capacity_method_table,
    alignment_tuning_conclusion_table,
    alignment_tuning_full_candidate_table,
    alignment_tuning_limit_table,
    alignment_tuning_status_table,
    load_alignment_tuning_outputs,
)

TUNING_DIR_CANDIDATES = [
    PROJECT_ROOT / "output" / "ep05_alignment_tuning",
    PROJECT_ROOT / "output" / "ep05_alignment_tuning_study",
]
TUNING_RUN_HINT = """
**Alignment tuning 产物未完整生成。** 可按下面模式先生成候选，再重新构建 Notebook：

```bash
# 推荐：一键生成 quick/full 调参 study、CSV/JSON 和 PNG
uv run python scripts/run_ep05_alignment_tuning_study.py \
  --mode quick --limit-frames 96 --n-jobs 8

# 单个候选的 96-frame screening 示例
uv run python scripts/run_ep05_contour_alignment_validation.py \\
  --output-dir output/ep05_alignment_tuning/r360_e93_rad100_s0125 \\
  --roi-size 360 --edge-percentile 93 \\
  --refine-radius-px 1.0 --refine-step-px 0.125 \\
  --limit-frames 96 --n-jobs 8 --skip-figures

# 单个 finalist 的 255-frame full run + capacity re-score 示例
uv run python scripts/run_ep05_contour_alignment_validation.py \\
  --output-dir output/ep05_alignment_tuning/full_r360_e93_rad100_s0125 \\
  --roi-size 360 --edge-percentile 93 \\
  --refine-radius-px 1.0 --refine-step-px 0.125 \\
  --n-jobs 8 --skip-figures

uv run python scripts/run_ep05_alignment_sr_capacity_check.py \\
  --alignment-csv output/ep05_alignment_tuning/full_r360_e93_rad100_s0125/contour_alignment_results.csv \\
  --output-dir output/ep05_alignment_tuning/full_r360_e93_rad100_s0125_capacity_eval93 \\
  --roi-size 360 --edge-percentile 93
```

`limit96_tuning_summary.csv` 和 `full_candidate_eval93_summary.csv` 是这些候选运行结果的汇总表；若暂时没有汇总 CSV，本节仍会展示已存在的 candidate capacity PNG/CSV。
"""
TUNING_STUDY_DIR = PROJECT_ROOT / "output" / "ep05_alignment_tuning_study"


def _relative_text(value):
    if value is None or value == "":
        return ""
    text = str(value)
    for base in [PROJECT_ROOT, *TUNING_DIR_CANDIDATES]:
        try:
            text = text.replace(str(base), str(base.relative_to(PROJECT_ROOT)))
        except ValueError:
            text = text.replace(str(base), str(base))
    return text


def _display_missing_notice():
    display(Markdown(TUNING_RUN_HINT))


def _round_existing(df, decimals):
    if df.empty:
        return df
    cols = {key: value for key, value in decimals.items() if key in df.columns}
    return df.round(cols)


tuning_outputs = load_alignment_tuning_outputs(TUNING_DIR_CANDIDATES)
tuning_status = alignment_tuning_status_table(tuning_outputs)
tuning_status["path"] = tuning_status["path"].map(_relative_text)
display(tuning_status)
if tuning_status["status"].astype(str).str.contains("missing").any():
    _display_missing_notice()

# %% [markdown]
# > **数据说明**: 上表检查 alignment tuning 所需的可选产物是否存在，包括 96-frame tuning summary、full-candidate summary，以及每个 finalist 的 capacity re-score 目录。
# > **怎么看**: `available` 或 `N found` 表示后续 cell 会读取并展示这些 CSV/PNG；`missing` 表示该部分会用运行命令说明替代，不会让 Notebook 崩溃。
# > **正常/异常理解**: tuning appendix 是后补实验，所以允许部分产物暂缺。缺失 summary CSV 时，不能据此否定 alignment；它只说明当前机器还没有重建调参汇总。
# > **核心发现**: 当前 Notebook 的 tuning 证据边界由本表决定：存在的 CSV/PNG 会进入候选比较，不存在的产物只记录可复现运行路径。

# %% [markdown]
# ### 6.0 Reproducible Tuning Script Outputs

# %%
study_summary_path = TUNING_STUDY_DIR / "tuning_summary.csv"
study_candidate_path = TUNING_STUDY_DIR / "candidate_comparison_summary.csv"
study_phase_path = TUNING_STUDY_DIR / "candidate_phase_coverage.csv"
if study_summary_path.exists():
    study_summary = pd.read_csv(study_summary_path)
    display(
        study_summary[
            [
                "candidate",
                "edge_percentile",
                "refine_radius_px",
                "refine_step_px",
                "eval_refined_holdout_median_px",
                "eval_refined_holdout_p90_px",
                "eval_refined_gradient_corr_median",
                "is_default_candidate",
            ]
        ]
        .head(10)
        .round(4)
    )
else:
    print(f"Missing table: {TUNING_STUDY_DIR.relative_to(PROJECT_ROOT)}/tuning_summary.csv")
    print("Run: uv run python scripts/run_ep05_alignment_tuning_study.py --mode quick --limit-frames 96 --n-jobs 8")

if study_candidate_path.exists():
    study_candidate = pd.read_csv(study_candidate_path)
    display(study_candidate.round(4))

if study_phase_path.exists():
    study_phase = pd.read_csv(study_phase_path)
    display(
        study_phase[
            ["method_label", "scale", "occupied_bins", "bad_bins", "total_bins", "entropy_fraction", "min_count", "max_count"]
        ].round(4)
    )

for name in ["tuning_heatmap_heldout_chamfer.png", "candidate_alignment_comparison.png"]:
    path = TUNING_STUDY_DIR / name
    if path.exists():
        display(Image(filename=str(path), width=1000))

# %% [markdown]
# > **数据说明**: 这一组输出来自 `scripts/run_ep05_alignment_tuning_study.py`，它把参数扫描、统一 eval-edge 评分、candidate comparison 和 phase coverage 固化为一个可复现实验；如果上方显示缺失提示，说明当前机器还没有运行该脚本。
# > **怎么看**: `tuning_summary.csv` 看候选参数排序，`candidate_comparison_summary.csv` 看 tuned/default/NCC/filename 四类策略的 Chamfer 与 gradient correlation，`candidate_phase_coverage.csv` 看 2x/3x/4x phase-bin 是否完整。PNG 中 heatmap 越低越好，comparison 图则同时看 Chamfer、correlation 和 phase coverage。
# > **正常/异常理解**: quick 模式只覆盖较少帧，适合展示实验过程和调参趋势；full 模式或既有 `output/ep05_alignment_tuning/full_*` 目录才是最终 handoff 证据。若 tuned refined 在 3x/4x 出现 phase collapse，这是预期风险，不支持高倍率声明。
# > **核心发现**: 可复现实验脚本把本轮手动调参固化下来：tuned refined 可以降低 held-out Chamfer，但 NCC init 仍是连续相位 prior，default refined/tuned refined 需要进入 EP06 ablation 后再决定谁做主 gate。

# %% [markdown]
# ### 6.1 96-Frame Screening Sweep

# %%
limit_table = alignment_tuning_limit_table(tuning_outputs["limit_summary"])
if limit_table.empty:
    _display_missing_notice()
else:
    display(
        _round_existing(
            limit_table,
            {
                "edge": 1,
                "radius": 3,
                "step": 3,
                "init_med": 4,
                "refined_med": 4,
                "refined_p90": 4,
                "gain_vs_init_pct": 2,
                "gain_vs_noalign_pct": 2,
                "corr_gain_med": 4,
                "shift_norm_med": 4,
                "worse_than_init_frac": 4,
            },
        )
    )

# %% [markdown]
# > **数据说明**: 表格来自 `limit96_tuning_summary.csv`，每一行是一个 ROI/edge/refinement 参数组合在 96 帧 screening 子集上的结果；如果上方显示运行提示而不是表格，说明该 CSV 暂缺。`refined_med` 和 `refined_p90` 是 refinement 后 held-out Chamfer，越小越好；`gain_vs_init_pct` 表示相对 NCC init 的中位 Chamfer 降幅；`worse_than_init_frac` 是 refinement 反而变差的帧比例。
# > **怎么看**: 先按 `rank` 看候选排序，再检查 P90 和 worse fraction。一个可用候选不只要 median 低，还要尾部稳定、变差比例低，并且 shift norm 不出现明显异常。
# > **正常/异常理解**: 96-frame sweep 是快速筛选，不是最终结论。若两个候选 median 非常接近，应优先保留 full-run 复评，而不是只用 screening 排名决定 EP06 参数。
# > **核心发现**: screening 的作用是把候选缩小到少数稳定组合；真正推荐参数必须再经过 255 帧 full run 和 2x phase/capacity re-score。

# %%
if limit_table.empty:
    _display_missing_notice()
else:
    plot_df = limit_table.sort_values("rank").reset_index(drop=True)
    x = np.arange(len(plot_df))
    labels = plot_df["name"].astype(str).str.replace("_", "\n", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), constrained_layout=True)
    axes[0].plot(x, plot_df["init_med"], marker="o", linewidth=1.2, label="NCC init")
    axes[0].plot(x, plot_df["refined_med"], marker="o", linewidth=1.2, label="Contour refined")
    axes[0].set_title("Median Chamfer")
    axes[0].set_ylabel("Held-out Chamfer [px]")
    axes[0].legend(frameon=False)

    axes[1].plot(x, plot_df["refined_p90"], marker="o", color="#B55D60", linewidth=1.2)
    axes[1].set_title("Tail Risk")
    axes[1].set_ylabel("Refined P90 Chamfer [px]")

    axes[2].bar(x, plot_df["worse_than_init_frac"] * 100.0, color="#5A8F7B", width=0.72)
    axes[2].set_title("Regression Fraction")
    axes[2].set_ylabel("Frames worse than NCC init [%]")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    display(fig)
    plt.close(fig)

# %% [markdown]
# > **图表说明**: 左图比较 NCC init 与 contour refined 的 median Chamfer，中图显示 refined P90，右图显示 refinement 比 NCC init 更差的帧比例；如果上方显示运行提示而不是图，说明 tuning summary CSV 暂缺。横轴是按 screening 排名排序的候选参数。
# > **怎么看**: 左图越低越好，说明典型帧轮廓更贴近；中图越低越好，说明尾部帧不容易失配；右图越接近 0 越好，说明 refinement 不会频繁伤害 NCC 初值。
# > **正常/异常理解**: median 下降但 P90 或 worse fraction 升高时，需要谨慎，因为这类参数可能只改善典型帧、牺牲难帧。Chamfer 是轮廓 proxy，不是 SR 图像质量本身。
# > **核心发现**: tuning 曲线用于选择稳定的 contour-refinement gate；它不能替代 EP06 的 split-half、forward consistency 和视觉轮廓增益验证。

# %% [markdown]
# ### 6.2 Full-Frame Finalist Comparison

# %%
full_table = alignment_tuning_full_candidate_table(tuning_outputs["full_summary"])
if full_table.empty:
    _display_missing_notice()
else:
    display(
        _round_existing(
            full_table,
            {
                "edge": 1,
                "radius": 3,
                "step": 3,
                "eval93_refined_med": 4,
                "eval93_refined_p90": 4,
                "eval93_refined_corr_med": 4,
                "eval93_ncc_med": 4,
                "eval93_ncc_corr_med": 4,
                "eval93_filename_med": 4,
                "eval93_filename_corr_med": 4,
                "refined_gain_vs_ncc_pct": 2,
                "refined_gain_vs_filename_pct": 2,
                "phase2_entropy": 3,
            },
        )
    )

# %% [markdown]
# > **数据说明**: 表格来自 `full_candidate_eval93_summary.csv`，只保留 screening 后的少数 finalist，并在完整 255 帧主 session 上重新评分；如果上方显示运行提示而不是表格，说明 finalist 汇总暂缺。`eval93_*` 表示用 edge percentile 93 的统一口径复评；`phase2_min/max/entropy` 检查 2x phase-bin 覆盖是否仍然健康。
# > **怎么看**: 推荐候选应同时具备低 refined median/P90、相对 NCC 和 filename 的 Chamfer gain、以及没有 2x phase 空洞。gradient correlation 不要求 contour refined 最大，因为 refinement 可能用几何 Chamfer 换取局部梯度相关性的轻微下降。
# > **正常/异常理解**: 如果 refined Chamfer 低于 NCC/filename，但 `eval93_refined_corr_med` 明显下降，应回到 PNG 和 worst-frame 表检查是否过度吸附。若 phase2 min count 接近 0，则不能把该候选作为 2x SR 主线。
# > **核心发现**: full-run comparison 才是 EP06 参数选择的主证据；96-frame sweep 只负责减少候选搜索空间。

# %%
if full_table.empty:
    _display_missing_notice()
else:
    plot_df = full_table.sort_values("rank").reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.24
    labels = plot_df["name"].astype(str).str.replace("full_", "", regex=False).str.replace("_", "\n", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), constrained_layout=True)
    axes[0].bar(x - width, plot_df["eval93_filename_med"], width=width, label="Filename affine", color="#6C7A89")
    axes[0].bar(x, plot_df["eval93_ncc_med"], width=width, label="NCC init", color="#4C72B0")
    axes[0].bar(x + width, plot_df["eval93_refined_med"], width=width, label="Contour refined", color="#55A868")
    axes[0].set_title("Full-Run Median Chamfer")
    axes[0].set_ylabel("Held-out Chamfer [px]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(x, plot_df["eval93_refined_p90"], marker="o", label="Refined P90", color="#B55D60")
    axes[1].plot(x, plot_df["phase2_entropy"], marker="s", label="2x phase entropy", color="#8172B2")
    axes[1].set_title("Tail and Phase Check")
    axes[1].set_ylabel("P90 Chamfer [px] / entropy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", alpha=0.25)
    display(fig)
    plt.close(fig)

# %% [markdown]
# > **图表说明**: 左图在同一批 255 帧上比较 filename affine、NCC init 和 contour refined 的 median Chamfer；右图同时展示 refined P90 与 2x phase entropy；如果上方显示运行提示而不是图，说明 finalist 汇总暂缺。
# > **怎么看**: 左图越低越好，用于判断 refinement 是否在几何轮廓上优于 prior/control；右图中 P90 越低越好，phase entropy 越接近 1 越均匀。二者要一起读，避免选出 Chamfer 好但 phase 覆盖差的参数。
# > **正常/异常理解**: contour refined 的 Chamfer 低于 NCC init 是对局部几何锚定有利的信号；NCC init 的 gradient correlation 更高仍然正常，因为 NCC 直接优化局部图像相关性。两者不是互相替代，而是 prior + gate 的关系。
# > **核心发现**: full-run 曲线把推荐参数限定为“alignment gate 选择”，不是 SR 成功声明；EP06 仍需要把该参数放进重建流程中做 split-half 和结构一致性验证。

# %% [markdown]
# ### 6.3 Candidate Capacity Appendix

# %%
capacity_method_table = alignment_tuning_capacity_method_table(tuning_outputs)
if capacity_method_table.empty:
    _display_missing_notice()
else:
    best_candidate = full_table.iloc[0]["name"] if not full_table.empty else capacity_method_table.iloc[0]["candidate"]
    best_capacity_table = capacity_method_table[capacity_method_table["candidate"].eq(best_candidate)].copy()
    if best_capacity_table.empty:
        best_capacity_table = capacity_method_table.copy()
    display(
        _round_existing(
            best_capacity_table,
            {
                "holdout_chamfer_median_px": 4,
                "holdout_chamfer_p90_px": 4,
                "gradient_corr_median": 4,
                "gradient_corr_p10": 4,
                "shift_norm_median_px": 4,
                "shift_norm_p90_px": 4,
            },
        )
    )

# %% [markdown]
# > **数据说明**: 表格读取 finalist capacity 目录里的 `alignment_method_summary.csv`；如果上方显示运行提示而不是表格，说明 candidate capacity CSV 暂缺。默认显示 full-run 表中排名第一候选的 method comparison；如果 best candidate 的目录不存在，则显示当前可用的候选 method summaries。
# > **怎么看**: 这张表把 tuning 推荐参数重新放回 EP05 的五个 alignment method 对照体系中。Chamfer median/P90 越低越好，gradient correlation median/P10 越高越好，shift norm 用于检查是否出现异常大修正。
# > **正常/异常理解**: contour refined 在 Chamfer 上最好、NCC init 在 gradient correlation 上最好是合理模式。若 refined Chamfer 降低但 P10 correlation 崩坏，说明参数可能过拟合局部 edge，需要在 EP06 gate 掉。
# > **核心发现**: tuning 选择必须通过 method-comparison 对照重新解释；不能只看候选内部的 refined Chamfer。

# %%
artifacts = tuning_outputs["capacity_artifacts"]
best_candidate = full_table.iloc[0]["name"] if not full_table.empty else None
if best_candidate is None and not artifacts.empty:
    best_candidate = artifacts.iloc[0]["candidate"]

if best_candidate is None or artifacts.empty:
    _display_missing_notice()
else:
    row = artifacts[artifacts["candidate"].eq(best_candidate)]
    if row.empty:
        row = artifacts.iloc[[0]]
    row = row.iloc[0]
    shown = False
    for key, width in [("comparison_png", 1000), ("phase_png", 1000), ("overlay_png", 1000)]:
        path = row[key]
        if path.exists():
            display(Image(filename=str(path), width=width))
            shown = True
    if not shown:
        _display_missing_notice()

# %% [markdown]
# > **图表说明**: 上方 PNG 来自 finalist 的 capacity re-score 目录，通常包含 `alignment_method_comparison.png`、`phase_bin_coverage_2x.png` 和 `alignment_overlay_evidence.png`；如果上方显示运行提示而不是图片，说明这些 PNG 暂缺。第一张看 method-level Chamfer/correlation，第二张看 2x phase-bin 覆盖，第三张看 edge-density visual sanity check。
# > **怎么看**: method comparison 用于确认 tuning 参数在 EP05 对照体系中仍然有效；phase-bin 图用于确认 2x 没有空相位格；overlay 只用于人工检查轮廓堆叠是否集中，不能当作 SR metric。
# > **正常/异常理解**: 如果 PNG 缺失，说明 capacity re-score 没有完整运行或 `--output-dir` 指向了不同目录；此时应使用本节开头给出的脚本命令重建。即使 PNG 看起来更清楚，也不能跳过数值表和 EP06 split-half 验证。
# > **核心发现**: candidate PNG 是 tuning 结果的可视化 appendix，帮助发现明显异常；最终结论仍以 held-out Chamfer、gradient correlation、phase coverage 和 EP06 重建稳定性共同决定。

# %% [markdown]
# ### 6.4 Tuning Conclusion Boundary

# %%
conclusion_table = alignment_tuning_conclusion_table(
    tuning_outputs["limit_summary"],
    tuning_outputs["full_summary"],
)
if conclusion_table.empty:
    _display_missing_notice()
else:
    display(conclusion_table)

# %% [markdown]
# > **数据说明**: 结论表把 screening winner、full-run 推荐参数、相对 NCC 的收益和 2x phase health 汇总成可交接给 EP06 的边界声明；如果上方显示运行提示而不是表格，说明 tuning 汇总 CSV 暂缺。
# > **怎么看**: `answer` 是当前 tuning study 支持的选择，`evidence` 是对应 CSV 中的直接数值，`boundary` 明确说明这项证据不能外推到哪里。
# > **正常/异常理解**: tuning 能选择 alignment gate，但不能证明 2x SR 已经成功；Chamfer 下降说明轮廓 proxy 更稳定，phase 健康说明采样容量可用，二者都还不是真实光学分辨率或温度计量证据。
# > **核心发现**: 当前 tuning study 支持 EP06 使用 full-run 最优 contour-refinement 参数作为局部质量门控，同时保留 NCC init 作为连续 phase prior、filename/stage 作为对照。最终验收必须留到 EP06 的 2x contour-level SR 重建、split-half 一致性和结构可解释性评估。
