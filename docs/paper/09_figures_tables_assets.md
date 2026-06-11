# 9. Figures & Tables — 资产映射与生产状态

> 论文图表 → 仓库资产路径的权威映射。每个条目：目标内容 / 现成资产 / 状态 / 生产路径。
> 状态图例：✅ 可直接用 · 🔧 需重绘（数据在）· ⬜ 缺失（依赖未落地）· 🔄 已派发执行中。
> 终稿统一落盘位置：`output/paper_figures/`（PNG + PDF 双格式，300 dpi，CVPR 风格）。
> 数字红线：**TB-scale 与 EP11-harness scale 绝不混在同一图/表**（见 `07_experiments.md` 头注）。

## 主文图（目标 5 图 + 2 表，8 页版式）

### F0 — Teaser（§1 页首，单栏）

- **内容**: F5 的中心 zigzag ROI 三列裁剪：bicubic / TGV / 最优学习臂（温度域）
- **状态**: ⬜ 依赖 F5 终稿（统一 harness + V9A/V9C checkpoint）
- **现成参考**: `output/ep11_dl_benchmark/checkpoint_selection/v9b_step11000/` 下对比图可作排版样稿

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

- **内容**: phase-stratified split-half FRC 主曲线（1/7 cutoff 17.0 µm）+ 正/负/漂移控制组；
  band 表并入 T1 或作图内小表
- **状态**: ✅ 资产齐全，仅需排版统一
- **资产**: `output/ep15_info_limit/m2_frc/frc_curve.png` + `frc_controls.png`；
  源数据 `frc_curve.csv`、`frc_controls.csv`、`frc_band_table.csv`、`frc_repeats.csv`
- **图注红线**: 10–12 µm 反弹必须标注为 coverage/lattice + drift 风险，不作分辨率证据

### F3 — null-space drift 轨迹 + forward-loss inset（§6.2，双栏全宽；全文核心机制图）

- **内容**: 1x 输入五臂（v6/v8.1a/v8.1b/v9b/v9d）的 artifact & corr vs step 双 panel；
  inset：`loss/forward_model` 贴底曲线（v6/v9b/v9d，log y）；canonical ○ / 60K 端点 ×
- **状态**: ✅ 当前稿已生成（Task B）→ `algos/ep07_unet_sr/scripts/plot_drift_trajectories_paper.py`；
  v9d/v9a 仍在训练，训练完成后 `--refresh` 终稿
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
  `checkpoint_candidates.csv`；V9A/V9C/V9D 落地后同脚本重出
- **视觉 gate 配套**: `panel_v6/v8.1a/v8.1b/v9b.png`（panel 类材料放 supp）

### F5 — 主视觉对比（§6.1，双栏全宽，含 F6 消融行）

- **内容**: 行 1 温度域 / 行 2 highpass 域 × 列 {bicubic, drizzle, MAP-TV, TGV, UNet-best,
  V9A-best}；行 3（消融行，原 F6）：thin-line + edge-staircase 裁剪 {v8.1a, v9a, v9c}
- **状态**: ⬜ 依赖：① V9A/V9C/V9D canonical checkpoint；② 统一口径 harness 重跑；
  ③ ROI 坐标冻结（中心 zigzag + 一处块边界）
- **partial 资产**: `output/ep10_method_comparison/temperature_comparison.png`（五经典列、无 UNet）、
  `output/ep15_info_limit/m4_deconv_anchor/four_arm_comparison.png` + `four_arm_highpass.png`
  （四臂、无 TGV）、`output/ep11_dl_benchmark/unet_vs_tgv_2x_center_zoom3x_highpass.png`
- **阻塞**: 客户许可（芯片热像脱敏展示）——形态可能需限于中心 ROI

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
- **状态**: ⬜ 硬前提：单一口径 harness（`run_unet_vs_drizzle_2x.py` 谱系）重跑全部臂；
  依赖 V9A/V9C/V9D checkpoint 选定
- **partial 数据**: `output/ep10_tgv_sr/sweep_results.csv`、
  `output/ep11_dl_benchmark/checkpoint_selection/v9b_step11000/comparison_summary.csv`、
  `output/ep15_info_limit/m4_deconv_anchor/zigzag_profile_metrics.csv`

### T2 — 消融矩阵 input × anchor（§6.3）

- **格**: {1x stats, hybrid drizzle} × {none, band-limited, full-band, legal}；
  已填 1x×none(v8.1a)、1x×band(v9b)；🔄 1x×full(V9D)、hybrid×none(V9A) 训练中；
  ⬜ hybrid×legal(V9C) 待跑
- **数据源**: 各 arm selected-checkpoint 过统一 harness 后的指标 + `config.json` 字段

## Supplementary 图表（节选，详见 `10_writing_handover.md` §C/D）

| 编号 | 内容 | 资产/来源 | 状态 |
|---|---|---|---|
| S-F1 | FRC band×seed 全表 + bicubic/shuffle/drift 控制组完整曲线 | `output/ep15_info_limit/m2_frc/` | ✅ |
| S-F2 | PSF 三路证据链（forward 残差曲线 / ESF 分布 / M3 仲裁图） | `output/ep09_psf_calibration/` + `m3_sigma/` | ✅ |
| S-F3 | v9a hybrid 轨迹 companion 图 | `output/paper_figures/fig03s_v9a_trajectory.*` | ✅ 当前稿，训练后 refresh |
| S-F4 | 四臂 checkpoint 视觉 gate panel | `checkpoint_selection/panel_*.png` | ✅ |
| S-F5 | 各 arm step 序列视觉演化（漂移可视化） | `algos/ep07_unet_sr/outputs/*/eval_real/` | ✅ 选图即可 |
| S-F6 | 负结果档案图（PixelShuffle 条纹 / 4x 失败 / AVI 排除审计） | EP11/EP12/EP01 输出 | 🔧 选图整理 |
| S-F7 | 对齐管线与 gate（Chamfer 0.381→0.240→0.134；EP04 角色表） | `output/ep05_*` / `output/ep04_*` | 🔧 |
| S-F8 | E3 对齐源消融 + F7 全曲线 | `output/ep16_budget_robustness/` | ✅ 经典臂完成 |
| S-T1 | T1 扩展版（全 checkpoint × 全列） | 统一 harness 输出 | ⬜ |
| S-T2 | TGV/MAP-TV 参数网格全表 | `output/ep10_tgv_sr/sweep_results.csv` 等 | ✅ |
| S-T3 | TCForge 合成参数全表 / 训练 config 对照表 | 各 run `config.json` | 🔧 汇总即可 |

## 生产排程依赖

```
现在（CPU，已完成）: F1(Task A) · F3 当前稿(Task B) · F7 经典臂(Task C)
现在（CPU，待做）: F2/F4 排版微调（主线）
V9A 落地（≈06-12 晨）: V9A checkpoint 选择 → F4 更新
V9D/V9C 落地（GPU1 串行）: F3 --refresh 终稿 → T2 填格
GPU 空闲窗口: 统一 harness 重跑全臂 → T1 → F5/F0 → F7 GPU 臂补线
客户许可确认: F5/F0 终稿形态（全幅 or 中心 ROI 脱敏）
```
