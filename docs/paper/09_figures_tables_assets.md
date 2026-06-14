# 9. Figures & Tables — 资产映射与生产状态

> 论文图表 → 仓库资产路径的权威映射。每个条目：目标内容 / 现成资产 / 状态 / 生产路径。
> 状态图例：✅ 可直接用 · 🔧 需重绘（数据在）· ⬜ 缺失（依赖未落地）· 🔄 已派发执行中。
> 终稿统一落盘位置：`output/paper_figures/`（PNG + PDF 双格式，300 dpi，CVPR 风格）。
> 数字红线：**TB-scale 与 EP11-harness scale 绝不混在同一图/表**（见 `07_experiments.md` 头注）。

## 主文图（目标 5 图 + 2 表，8 页版式）

### F0 — Teaser（§1 页首，单栏）

- **内容**: F5 的中心 zigzag ROI 三列裁剪：bicubic / TGV / 最优学习臂（温度域）
- **状态**: 🔧 F5 终稿已生成，仍需从 `fig05_main_visual` 裁出 teaser 版三列 crop
- **现成参考**: `output/paper_figures/fig05_main_visual.png`；旧样稿在
  `output/ep11_dl_benchmark/checkpoint_selection/v9b_step11000/`

### F1 — 系统 + 标定链 + 采样/分辨率区分（§3，双栏全宽）

- **内容**: (a) raster 微扫描几何 + θ=47.6° 旋转 + command→px 向量；(b) 观测算子链
  y_k = D·B·H·S·x + n 流程图（每个算子标测量来源）；(c) pitch 10 µm ≠ resolution 20 µm ≠
  SR grid 5 µm 同轴示意
- **状态**: ✅ 已生成（Task A）→ `scripts/paper_figures/fig01_system_calibration.py`
- **终稿资产**: `output/paper_figures/fig01_system_calibration.png`、
  `output/paper_figures/fig01_system_calibration.pdf`；
  source map: `output/paper_figures/fig01_system_calibration.json`
- **组件资产**: `output/ep03_theoretical_limits/sampling_resolution_distinction.png`（(c) 原型，
  `thermal_core.ep03.plot_sampling_resolution_diagram`）；
  `output/ep02_displacement_calibration/ep02_raster_acquisition_path.png`（(a) 参考，
  `thermal_core.ep02.plot_raster_acquisition_path`）
- **权威数字**: `configs/stage_calibration.json`；noise 0.0724 °C；σ∈[0.2,0.5]
  （`output/ep15_info_limit/m3_sigma/sigma_summary.json`）

### F2 — 信息存在性：FRC 曲线 + 控制组（§5，单栏）

- **内容**: phase-stratified split-half FRC 主曲线（1/7 cutoff 17.0 µm，含 half-bit 判据
  与逐 seed cutoff 刻度）+ 正/负/漂移控制组；10–12 µm 反弹风险带已直接画入图中
- **状态**: ✅ 已生成 CVPR 风格终稿（06-12）→ `scripts/paper_figures/fig02_frc.py`
- **终稿资产**: `output/paper_figures/fig02_frc.png`、`output/paper_figures/fig02_frc.pdf`；
  supp 档案版 `figS01_frc_archive.{png,pdf}` 同脚本产出
- **源数据**: `output/ep15_info_limit/m2_frc/{frc_curve,frc_controls,frc_band_table,frc_repeats}.csv` + `frc_summary.json`
- **图注红线**: 10–12 µm 反弹必须标注为 coverage/lattice + drift 风险，不作分辨率证据

### F3 — null-space drift 轨迹 + forward-loss inset（§6.2，双栏全宽；全文核心机制图）

- **内容**: 1x 输入五臂（v6/v8.1a/v8.1b/v9b/v9d）的 artifact & corr vs step 双 panel；
  inset：`loss/forward_model` 贴底曲线（v6/v9b/v9d，log y）；canonical ○ / 60K 端点 ×
- **状态**: ✅ 当前稿已生成（Task B）→ `algos/ep07_unet_sr/scripts/plot_drift_trajectories_paper.py`；
  v9d/v9a/v9c 训练已完成，当前稿可按需 `--refresh`
- **终稿资产**: `output/paper_figures/fig03_nullspace_drift.png`、
  `output/paper_figures/fig03_nullspace_drift.pdf`；
  companion: `output/paper_figures/fig03s_v9a_trajectory.png/.pdf`
- **资产**: 轨迹主体 `output/ep11_dl_benchmark/checkpoint_selection/checkpoint_metrics.csv`
  （v6/v8.1a/v8.1b/v9b）+ `fig_trajectories.png`（旧版无 inset）；forward loss 在各 run
  `tb_logs/` 的 tag `loss/forward_model`
- **红线**: v9a（hybrid 输入）不进主图（proxy 跨输入模式不可横比），出 supp companion 图

### F4 — proxy Pareto + checkpoint 选择（§6.6，单栏）

- **内容**: 四臂 Pareto 散点 + TGV 参考点 (0.695, 0.916)（TB-scale）+ canonical 标注
- **状态**: ✅ `output/ep11_dl_benchmark/checkpoint_selection/fig_pareto.png` +
  `checkpoint_candidates.csv`；V9A/V9C/V9D/V10 终稿数字以统一 harness 表为准，F4 可按最终文案可选 refresh
- **视觉 gate 配套**: `panel_v6/v8.1a/v8.1b/v9b.png`（panel 类材料放 supp）

### F5 — 主视觉对比（§6.1，双栏全宽，含 F6 消融行）

- **内容**: 行 1 温度域 / 行 2 highpass 域 × 列 {drizzle, TGV, V9A late 60K,
  V10 λ=1.2@15K}；中心 zigzag ROI 用来显示 drizzle softness、TGV staircase、V9A late
  over-thickening、V10 sharp/grain trade-off。
- **状态**: ✅ 统一 harness 已生成 F5（Task D）→
  `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`
- **终稿资产**: `output/paper_figures/fig05_main_visual.png`、
  `output/paper_figures/fig05_main_visual.pdf`
- **源数据**: `output/ep11_unified_harness/run_manifest.json` +
  `output/ep11_unified_harness/all_arm_metrics.csv`
- **partial 参考**: `output/ep10_method_comparison/temperature_comparison.png`（五经典列、无 UNet）、
  `output/ep15_info_limit/m4_deconv_anchor/four_arm_comparison.png` + `four_arm_highpass.png`
  （四臂、无 TGV）、`output/ep11_dl_benchmark/unet_vs_tgv_2x_center_zoom3x_highpass.png`
- **展示红线**: F5 是 task-level visual gate，不是 fidelity/resolution evidence；图注必须写明。

### ~~F6~~ — 并入 F5 第三行（独立成图方案备份在 supp D）

### F7 — frame-budget + 鲁棒性（建议降级 supp；主文 §6.4–6.5 留结论句）

- **内容**: 左 panel：指标 vs N ∈ {31,62,124,248}；右 panel：指标 vs shift 噪声
  σ ∈ {0,0.05,0.1,0.2} px；对齐源消融（command vs refined）以小表呈现
- **状态**: ✅ 经典臂（drizzle/TGV，CPU）已完成（Task C，EP16）；
  MAP-TV/UNet 臂 ⬜ 等 GPU 空闲
- **产出位置**: `output/ep16_budget_robustness/` + `output/paper_figures/fig07_budget_robustness.*`
- **实测摘要**: `frame_budget.csv` 17 行 all success；`shift_robustness.csv` 20 行 all success；
  `alignment_source.csv` 4 行 all success；`run_manifest.json` 37 unique runs all success。
  Drizzle N-budget raw-control corr 0.747±0.032@31 → 0.771@248；contour-refined vs command
  prior improves drizzle corr 0.662→0.771 and TGV corr 0.642→0.741.

### T1 — 主定量表（§6.1）

- **列**: split-half NRMSE / artifact / raw-control corr / FRC@{16,14,12} µm /
  zigzag median FWHM & dip / runtime；**行**: bicubic / drizzle / MAP-TV / TGV / UNet-best / V9A-best
- **状态**: ✅ 统一 harness 完成（Task D）
- **终稿数据**: `output/ep11_unified_harness/t1_metrics.csv`；
  全臂扩展：`output/ep11_unified_harness/all_arm_metrics.csv`；
  scale audit：`output/ep11_unified_harness/tb_vs_harness_scale_check.csv`
- **partial 参考**: `output/ep10_tgv_sr/sweep_results.csv`、
  `output/ep11_dl_benchmark/checkpoint_selection/v9b_step11000/comparison_summary.csv`、
  `output/ep15_info_limit/m4_deconv_anchor/zigzag_profile_metrics.csv`

### T2 — 消融矩阵 input × anchor（§6.3）

- **格**: {1x stats, hybrid drizzle} × {none, band-limited, full-band, legal}；
  已填 1x×none(v8.1a)、1x×band(v9b)、1x×full(V9D)、hybrid×none(V9A)、
  hybrid×legal(V9C)，另列 V10 residual-over-observation 作为参数化输出对照
- **状态**: ✅ 统一 harness 完成（Task D）
- **数据源**: `output/ep11_unified_harness/t2_metrics.csv` + 各 arm `config.json` 字段

## Supplementary 图表（节选，详见 `10_writing_handover.md` §C/D）

| 编号 | 内容 | 资产/来源 | 状态 |
|---|---|---|---|
| S-F1 | FRC 档案：逐 seed cutoff + 控制组全曲线（含各自 cutoff）+ band 表 + 零覆盖统计 | `output/paper_figures/figS01_frc_archive.{png,pdf}`（`fig02_frc.py` 产出） | ✅ CVPR 风格（06-12） |
| S-F2 | PSF 三路证据链（Route A/C 残差扫描 / Route B 表观 ESF 分布 + M3 边缘族 / M3 仲裁与采纳区间） | `output/paper_figures/figS02_psf_evidence.{png,pdf}`（`scripts/paper_figures/figS02_psf_evidence.py`；源 `output/ep09_psf_calibration/` + `m3_sigma/` CSV） | ✅ CVPR 风格（06-12） |
| S-F3 | v9a hybrid 轨迹 companion 图 | `output/paper_figures/fig03s_v9a_trajectory.*` | ✅ 当前稿，训练后 refresh |
| S-F4 | 四臂 checkpoint 视觉 gate panel | `checkpoint_selection/panel_*.png` | ✅ |
| S-F5 | 各 arm step 序列视觉演化（漂移可视化） | `algos/ep07_unet_sr/outputs/*/eval_real/` | ✅ 选图即可 |
| S-F6 | 负结果档案图（PixelShuffle 条纹 / 4x 失败 / AVI 排除审计） | EP11/EP12/EP01 输出 | 🔧 选图整理 |
| S-F7 | 对齐管线与 gate：五步链 Chamfer + 2x 相位 bin 占用 + gate 空间分布 | `output/paper_figures/figS07{a,b,c}_*.png`（收编自 `ep05_alignment_sr_capacity/` 与 `ep04_global_validation/`，`collect_promoted_supp.py`） | ✅ 选编收编（06-12） |
| S-F8 | E3 对齐源消融 + F7 全曲线 | `output/ep16_budget_robustness/` | ✅ 经典臂完成 |
| S-F9 | 零训练融合 baseline Pareto 叠加（V9A 轨迹 + 4 条融合曲线 + 支配象限 + TGV/drizzle 参考点） | `output/paper_figures/figS09_fusion_pareto.{png,pdf}`（`scripts/paper_figures/figS09_fusion_pareto.py`；源 `output/ep07_v9_review/*.csv`） | ✅ CVPR 风格（06-12）；V10 落地后叠 V10 工作点 |
| S-F10 | V9A fine-window 演化条带（TGV/v8.1a 参照 + 5K–60K 序列，per-panel 归一化 + 保真/锐度标注） | `output/paper_figures/figS10_v9a_strip.{png,pdf}`（`scripts/paper_figures/figS10_v9a_strip.py`；源 cache npy + `output/ep10_tgv_sr/best_hr_temperature.npy`） | ✅ CVPR 风格（06-12）；诊断原稿仍在 `output/ep07_v9_review/` |
| S-F11 | 数据审计链：文件名序 vs 采集序（13 假 session 教训）+ raster 采集序网格 | `output/paper_figures/figS11{a,b}_*.png`（收编自 `ep01_data_processing/`） | ✅ 选编收编（06-12） |
| S-F12 | AVI θ 验证 forest plot（16 AVI × 两特征域 + 合并 CI + X/Y 系统差） | `output/paper_figures/figS12_theta_forest.png`（收编自 `ep02_displacement_calibration/`） | ✅ 选编收编（06-12） |
| S-F13 | 主 session 累计位移轨迹（raster 几何 + 慢漂移可视化） | `output/paper_figures/figS13_cumulative_trajectory.png`（收编自 `ep05_sr_reassessment/`） | ✅ 选编收编（06-12） |
| S-F14 | MAP-TV 锚结构证据：zigzag 三剖面对照 + 四臂 highpass 全景 | `output/paper_figures/figS14{a,b}_*.png`（收编自 `ep15_info_limit/m4_deconv_anchor/`） | ✅ 选编收编（06-12） |
| S-F15 | MTF 频响 + 有效 SNR 可恢复性热图（2x 可行/4x 出界边界） | `output/paper_figures/figS15{a,b}_*.png`（收编自 `ep03_theoretical_limits/`） | ✅ 选编收编（06-12） |
| S-T1 | T1 扩展版（全 selected arms × 全列） | `output/ep11_unified_harness/all_arm_metrics.csv` | ✅ |
| S-T2 | TGV/MAP-TV 参数网格全表 | `output/ep10_tgv_sr/sweep_results.csv` 等 | ✅ |
| S-T3 | TCForge 合成参数全表 / 训练 config 对照表 | 各 run `config.json`（supp C.1/C.3 已成表） | ✅ 已汇总进 supp 草稿 |
| S-T4 | 融合 baseline λ 扫描全表 + fine-window 四指标口径表 | `output/ep07_v9_review/{fusion_baseline_metrics,v9a_pareto_metrics}.csv`（supp D.0/D.7 已成表） | ✅ |

## 策展 Notebook（图表展示与解读层）

- `notebooks/paper_main_figures/`（F1–F7 + 待定稿占位表）与
  `notebooks/paper_supp_figures/`（S-F1/2/3/4/7/8/9/10/11/12/13/14/15 + 占位表）——
  fragments 入 Git，构建命令 `uv run python scripts/build_notebook.py notebooks/paper_{main,supp}_figures --execute`。
  每张图附教程式解读（是什么/怎么看/异常是否正常/能得出什么）与重建命令。
- **Episode 图收编机制**: `scripts/paper_figures/collect_promoted_supp.py` 把 EP01–EP15
  管线中已达学术标准且支撑 supp 叙事的图按稳定 figSxx 编号拷贝进
  `output/paper_figures/`（含 provenance manifest），LaTeX 单目录引用、episode 管线保持唯一生产源。

## 生产排程依赖

```
已完成（CPU/GPU）: F1(Task A) · F2+S-F1(06-12 重绘) · F3 当前稿(Task B) ·
              F5/T1/T2(Task D) · F7 经典臂(Task C) · S-F2/S-F9/S-F10(06-12 重绘) ·
              两个策展 notebook
待做（CPU）: F0 teaser crop · S-F6 负结果组图 · S-F7 对齐 gate 图（素材齐，选图组版） ·
              F3/F3s 按最终 V9C/V10 文案可选 refresh
可选 GPU 后续: F7 learned/GPU 臂补线（非主文硬门槛）
```
