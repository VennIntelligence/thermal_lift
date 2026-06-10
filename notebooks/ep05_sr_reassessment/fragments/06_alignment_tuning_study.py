# %% [markdown]
# ## 6. Alignment Tuning Study / 对齐调参研究
#
# 本节补充 EP05 alignment tuning 的实验过程、调参曲线、候选比较和结论边界。它读取 `output/ep05_alignment_tuning_study` 中已经生成的 CSV/PNG；如果这些产物不存在，Notebook 会显示需要运行的脚本命令，而不是抛错中断。
#
# 调参过程分三步：
#
# 1. **96-frame screening**: 在主 session 的前 96 帧上扫 ROI size、edge percentile、Chamfer refinement radius/step，先筛掉明显不稳定的组合。
# 2. **clean-input finalist run**: 对筛出的候选跑完整 clean main input，检查 held-out Chamfer median/P90、gradient correlation 和 shift span。
# 3. **capacity re-score**: 用同一套 clean main input 对候选重新跑 alignment method comparison 和 2x phase-bin coverage，确认 tuning 选择没有破坏 EP06 所需的 phase diversity。
#
# 这节只选择 EP06 的 alignment gate 参数，不证明 SR 已经成功。NCC init 仍保留为连续位移 prior；contour refinement 是局部锚定和质量门控；stage/filename 仍是 prior/control，不是 ground truth。

# %%
from IPython.display import Markdown, display

from thermal_core.ep05 import (
    alignment_tuning_capacity_method_table,
    alignment_tuning_conclusion_table,
    alignment_tuning_full_candidate_table,
    alignment_tuning_limit_table,
    alignment_tuning_status_table,
)

TUNING_DIR_CANDIDATES = list(cache.tuning_dir_candidates)
TUNING_RUN_HINT = """
**Alignment tuning 产物未完整生成。** 可按下面模式先生成候选，再重新构建 Notebook：

```bash
# 推荐：一键生成 quick/full 调参 study、CSV/JSON 和 PNG
uv run python scripts/run_ep05_alignment_tuning_study.py \
  --mode quick --limit-frames 96 --n-jobs 8

# 单个候选的 96-frame screening 示例
uv run python scripts/run_ep05_contour_alignment_validation.py \\
  --output-dir output/ep05_alignment_tuning_study/manual_r360_e93_rad100_s0125 \\
  --roi-size 360 --edge-percentile 93 \\
  --refine-radius-px 1.0 --refine-step-px 0.125 \\
  --limit-frames 96 --n-jobs 8 --skip-figures

# 单个 finalist 的 clean-input full run + capacity re-score 示例
uv run python scripts/run_ep05_contour_alignment_validation.py \\
  --output-dir output/ep05_alignment_tuning_study/manual_full_r360_e93_rad100_s0125 \\
  --roi-size 360 --edge-percentile 93 \\
  --refine-radius-px 1.0 --refine-step-px 0.125 \\
  --n-jobs 8 --skip-figures

uv run python scripts/run_ep05_alignment_sr_capacity_check.py \\
  --alignment-csv output/ep05_alignment_tuning_study/manual_full_r360_e93_rad100_s0125/contour_alignment_results.csv \\
  --output-dir output/ep05_alignment_tuning_study/manual_full_r360_e93_rad100_s0125_capacity_eval93 \\
  --roi-size 360 --edge-percentile 93
```

`limit96_tuning_summary.csv` 和 `full_candidate_eval93_summary.csv` 是这些候选运行结果的汇总表；若暂时没有汇总 CSV，本节仍会展示已存在的 candidate capacity PNG/CSV。
"""


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


tuning_status = alignment_tuning_status_table(tuning_outputs)
tuning_status["path"] = tuning_status["path"].map(_relative_text)
display(tuning_status)
if tuning_status["status"].astype(str).str.contains("missing").any():
    _display_missing_notice()

# %% [markdown]
# 本表用于检查配准调参（Alignment Tuning）所需的各项输出文件与中间产物是否已在本地目录中生成。这包括 96 帧快速筛选汇总表（96-frame sweep summary）、完整候选评估文件以及各 finalist 参数组的重打分目录（capacity re-score directories）。
# 当表中的状态项标记为 `available` 时，后续的分析与可视化单元格将自动读取并展示对应的定量结果。若部分指标显示为 `missing`，则会展示对应的重新计算与复现命令，这允许在本地环境未完整跑完所有候选参数时，仍能查看已有的实验结论，防止脚本运行中断。
# 配准参数的调优是保障超分辨率算法能够获取稳定相位和几何投影的必要工作。由于调优计算具有一定的开销，允许部分辅助分析产物暂缺，但这并不影响当前已标定推荐参数在 EP06 实验中的有效性。

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

path = TUNING_STUDY_DIR / "tuning_heatmap_heldout_chamfer.png"
if path.exists():
    show_fig(path, width=1000)

# %% [markdown]
# Figure 9: Tuning heatmap from reproducible script outputs. Held-out Chamfer distance is mapped across alignment tuning parameters.

# %%
path = TUNING_STUDY_DIR / "candidate_alignment_comparison.png"
if path.exists():
    show_fig(path, width=1000)

# %% [markdown]
# Figure 10: Candidate alignment comparison from reproducible script outputs. Candidate strategies are compared by contour alignment metrics.

# %% [markdown]
# 上述结果来自于一键化配准调优研究脚本（`run_ep05_alignment_tuning_study.py`）。该流程将多维参数扫描、边缘百分位数评估、配准策略对比以及相位覆盖分析整合为统一的自动化实验。
# 其中，`tuning_summary.csv` 记录了不同 ROI 尺寸与边缘比例下 held-out 轮廓 Chamfer 距离的变化；`candidate_comparison_summary.csv` 对比了不同配准算法分支在轮廓逼近和梯度相关性（Gradient Correlation）上的综合表现；`candidate_phase_coverage.csv` 则统计了各算法在 $2\text{x}$、$3\text{x}$ 及 $4\text{x}$ 重建网格上的相位直方图分布。
# 定量数据与可视化热图展示了 Chamfer 距离在不同参数组合下的演化趋势。若在 $3\text{x}$ 或 $4\text{x}$ 亚像素尺度上出现相位直方图空洞（Phase Collapse），则表明当前配准精细化参数在高倍率下无法提供有效的信息冗余，这也印证了将超分辨率边界限制在 2x 的合理性。精细化对齐（Contour Refined）可以明显改善 held-out 轮廓的贴合度，但其必须基于连续的 NCC 初始化先验，且其与默认精细化参数（default refined）在重建质量上的最终优劣仍需在 EP06 烧蚀实验（ablation study）中进一步检验。

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
# 该表格呈现了在 96 帧快速筛选子集（Screening Sweep）上，关于 ROI 尺寸、边缘百分位数（Edge Percentile）及精细化搜索步长等多维参数网格搜索的初步排序。评估核心指标为配准后的中位 held-out Chamfer 距离（`refined_med`）及其第 90 百分位数（`refined_p90`），以及相对 NCC 初始化的轮廓贴合提升比例（`gain_vs_init_pct`）与配准退化率（`worse_than_init_frac`）。
# 优质的参数组合应当在中位 Chamfer 距离与长尾误差指标（P90）之间取得平衡，同时保证退化帧比例处于极低水平，避免算法在复杂噪声或弱边缘帧上发生过度吸附或漂移。由于该阶段仅在 96 帧子集上运行，其结果旨在快速过滤掉不稳定的超参数空间，并不作为最终的推荐依据，各优胜候选的性能释放在后续的 clean main input 完整序列运行及相位容量重打分中得到进一步确认。

# %%
if (TUNING_STUDY_DIR / "tuning_heatmap_heldout_chamfer.png").exists():
    show_fig(TUNING_STUDY_DIR / "tuning_heatmap_heldout_chamfer.png", width=1000)

# %% [markdown]
# Figure 11: Tuning heatmap for the screening sweep. Held-out Chamfer distance highlights stable parameter regions.

# %%
if not (TUNING_STUDY_DIR / "tuning_heatmap_heldout_chamfer.png").exists():
    if limit_table.empty:
        _display_missing_notice()
    else:
        print("Missing cached figure: output/ep05_alignment_tuning_study/tuning_heatmap_heldout_chamfer.png")
        print("Run: uv run python scripts/run_ep05_alignment_tuning_study.py --mode quick --limit-frames 96 --n-jobs 8")

# %% [markdown]
# 此处的二维参数热图直观展现了在不同边缘百分位数与搜索步长下，held-out 轮廓 Chamfer 距离的分布特征。热图中的深色低值区域代表在几何轮廓指标上更为贴合的参数配置。
# 在分析热图分布时，不仅要识别中位数最低的超参数节点，还需结合误差分布表，防范因过拟合局部强边缘而导致难帧（p90 异常）精度骤降的风险。需要强调的是，Chamfer 距离仅作为几何轮廓贴合度的一个代理度量（proxy metric），其改善并不直接等同于重建后温度场的高频清晰度。因此，通过该调参曲线选定的轮廓对齐参数，其对于超分辨率质量的贡献仍必须在后续 EP06 中通过前向一致性（Forward Consistency）与 split-half 鲁棒性进行联合标定。

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
# 本表汇总了入围候选参数组（Finalists）在完整 clean main input 上的重打分评估结果。为了保证评估的公平性，统一使用边缘百分位数 $93\%$ 对所有精细化配准后的结果进行复核（`eval93` 系列指标），并对 $2\text{x}$ 空间下的相位直方图均匀度（以相位熵 `phase2_entropy` 及最小计数为代表）进行统计。
# 合理的优选参数应当实现在完整序列上的中位及 P90 Chamfer 距离相比于初始对齐（NCC / Filename Affine）均有稳健提升，同时保证 $2\text{x}$ 重建网格内的相位样本分布均匀、无空白相位格。梯度相关性中位数（Gradient Correlation）通常允许在精细化配准后发生极其微弱的偏离，这是因为轮廓精细化优先约束了几何边界的一致性，在局部强梯度点上与图像互相关可能存在微小的物理差异。若发生 Chamfer 距离减小但梯度相关性崩溃的现象，则表明算法受到了局部伪影的过度吸引，需通过限制精细化搜索半径来予以纠正。

# %%
if (TUNING_STUDY_DIR / "candidate_alignment_comparison.png").exists():
    show_fig(TUNING_STUDY_DIR / "candidate_alignment_comparison.png", width=1000)

# %% [markdown]
# Figure 12: Full-frame finalist comparison. Candidate alignment branches are compared on held-out Chamfer and phase entropy.

# %%
if not (TUNING_STUDY_DIR / "candidate_alignment_comparison.png").exists():
    if full_table.empty:
        _display_missing_notice()
    else:
        print("Missing cached figure: output/ep05_alignment_tuning_study/candidate_alignment_comparison.png")
        print("Run: uv run python scripts/run_ep05_alignment_tuning_study.py --mode quick --limit-frames 96 --n-jobs 8")

# %% [markdown]
# 该对比图表并列展示了不同候选参数下文件名仿射（Filename Affine）、初始互相关（NCC Init）与轮廓精细化配准（Contour Refined）的 held-out Chamfer 距离分布，并叠加了相位熵值的变化曲线。
# 轮廓精细化在几何贴合度上超越初始互相关对齐，是确保微扫描序列能够在芯片内部结构边缘处实现高保真度亚像素叠合的关键证据。NCC 初始化提供全局位移先验，而轮廓精细化则在子像素级别提供局部几何锚定与质量滤波，两者共同协作以抑制由于器件热演化和辐射漂移导致的定位失效。

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
# 本表格提取自优选候选参数目录下的对齐方法对比摘要（`alignment_method_summary.csv`），将经过细致调优后的精细化算法重新置于包含无配准、命令先验、文件名仿射、初始互相关在内的完整对照组体系中进行评估。
# 在各项指标中，中位及 P90 轮廓距离越低，且低分位梯度相关性（如 P10 分位数）越高，代表该配置在典型帧及劣质帧上均具备高度的鲁棒性。微扫描配准的亚像素修正量中位数（`shift_norm`）需控制在合理物理区间内（如 $< 1\text{ px}$），任何超出此尺度的异常大修正通常对应着对噪声或热瞬态伪影的错误配准，应作为劣质对齐帧在重建的质量门控中予以剔除。

# %%
artifacts = tuning_outputs["capacity_artifacts"]
best_candidate = full_table.iloc[0]["name"] if not full_table.empty else None
if best_candidate is None and not artifacts.empty:
    best_candidate = artifacts.iloc[0]["candidate"]

if best_candidate is None or artifacts.empty:
    _display_missing_notice()
    row = None
else:
    row = artifacts[artifacts["candidate"].eq(best_candidate)]
    if row.empty:
        row = artifacts.iloc[[0]]
    row = row.iloc[0]
    capacity_paths = [row[key] for key in ["comparison_png", "phase_png", "overlay_png"]]
    if not any(path.exists() for path in capacity_paths):
        _display_missing_notice()

if row is not None:
    path = row["comparison_png"]
    if path.exists():
        show_fig(path, width=1000)

# %% [markdown]
# Figure 13: Candidate capacity method comparison. The selected candidate is compared with baseline alignment branches.

# %%
if row is not None:
    path = row["phase_png"]
    if path.exists():
        show_fig(path, width=1000)

# %% [markdown]
# Figure 14: Candidate phase capacity. Sub-pixel phase occupancy is checked for the selected tuning candidate.

# %%
if row is not None:
    path = row["overlay_png"]
    if path.exists():
        show_fig(path, width=1000)

# %% [markdown]
# Figure 15: Candidate overlay appendix. Edge overlay evidence is displayed for the selected tuning candidate.

# %% [markdown]
# 这组诊断图表作为参数评估的可视化附录，直观表现了优选配置下的各项关键特征，涵盖各方法的 held-out Chamfer 累计分布函数、2x 空间网格内的亚像素相位直方图覆盖，以及轮廓累加图的目视对比。
# 直观的轮廓图仅用于人眼合理性校验（Sanity Check），并非量化分辨率提升的科学标准。配准的有效性最终建立在统计指标（Chamfer CDF 右侧长尾的收敛程度）以及相位图的无盲区分布之上，以保障进入超分辨率算法的信息均匀性与结构鲁棒性。

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
# 最终汇总的调参结论表将快速筛选的胜出配置、完整序列的性能收益比以及 $2\text{x}$ 相位健康状态整理为清晰的参数交接依据。每一项决策（Answer）均附有具体的量化数值证据（Evidence），并定义了严格的物理解释边界（Boundary），用以指导 EP06 的超分辨率重建工作。
# 此项调优研究确立了后续超分辨率算法的对齐门限与物理先验，但这并不构成超分辨率重建成功的最终物理证明。任何 Chamfer 距离的减小与相位熵的提升仅表明多帧序列在几何与空间采样上具备了亚像素层面的可行性，而真正的图像分辨率增益及温度场轮廓恢复能力，仍需通过 EP06 的 2x 亚像素成像前向物理模型、空间频率谱分析及 split-half 重建一致性予以联合验证。
