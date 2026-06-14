# 补充材料 E —— 复现说明

## E.1 代码与环境

### E.1.1 仓库结构

随附代码包按以下结构组织：

| 目录 | 内容 | 论文对应 |
|---|---|---|
| `core/` | 共享层：数据审计、位移模型、理论界计算、锚点 gate、绘图规范 | §3 全部标定数字 |
| `configs/` | 物理参数单一来源：stage 标定、PSF 标定、噪声底、对齐结果 | §3.3 标定链 |
| `tcforge/` | 物理匹配合成平台（独立包） | §4.2 |
| `algos/ep06_sr_poc/` | forward 算子 + 指标库 | §3.4 / §5.2 |
| `algos/ep07_unet_sr/` | 学习臂训练/推理/评估代码 + 诊断管线 | §4.3 / §6 |
| `algos/ep10_*/` | drizzle / MAP-TV / TGV 经典臂 | §4.1 |
| `algos/ep15_info_limit/` | FRC、$\sigma$ 仲裁、去卷积锚 | §5.1 / §3.3 |
| `algos/ep16_budget_robustness/` | 帧预算与鲁棒性矩阵 | §6.4–6.5 |
| `scripts/paper_figures/` | 论文图生产脚本 | F1–F7 |
| `reports/`、`research_log/` | 正式报告与全程实验日志 | 可追溯性 |

原始数据（`data/`）和实验产物（`output/`）不入版本控制，前者需手动放置，后者由脚本重建。

### E.1.2 环境配置

本项目使用 UV 进行依赖管理。根目录 venv（`uv sync` + `uv pip install -e core/`）用于全局脚本、notebook 和数据审计。每个算法目录（`algos/xxx/`）是独立项目，拥有自己的 venv 和 lockfile，通过 `uv sync` 创建。共享代码以 `pip install -e ../../core` 方式安装。TGV 经典臂因包含 C/CUDA 编译依赖而使用 conda 环境（`environment.yml`）。各算法环境之间零耦合，可独立运行和删除。

### E.1.3 新机器一键部署

```bash
git clone <repo> thermal_lift && cd thermal_lift
# 放置 data/data_raw/（不随包分发，见 E.3）
uv sync && uv pip install -e core/
uv run python scripts/build_ep01_cache.py       # 生成 frame_audit.csv
uv run python scripts/build_all_notebooks.py --execute  # 全部产物
```

---

## E.2 图表重建命令

下表列出每张论文图表的生成脚本、所需环境和输入依赖。环境标注中 "root" 表示仓库根 venv，"ep07" 等表示对应算法目录内的 venv。

### E.2.1 主文图表

| 图表 | 脚本 | 环境 | 耗时 |
|---|---|---|---|
| F1 系统标定 | `scripts/paper_figures/fig01_system_calibration.py` | root | 秒级 |
| F2 FRC 曲线 | `scripts/paper_figures/fig02_frc.py` | root | 秒级 |
| F3 零空间漂移 | `algos/ep07_unet_sr/scripts/plot_drift_trajectories_paper.py` | ep07 | 秒级 |
| F4 Pareto 选点 | `algos/ep07_unet_sr/scripts/plot_checkpoint_selection.py` | ep07 | 分钟级 |
| F5 主视觉对比 | `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py --device cuda:0 --workers 4 --output-dir ../../output/ep11_unified_harness` | ep11 | ~3 min（cache warm） |
| F7 预算鲁棒性 | `algos/ep16_budget_robustness/scripts/run_ep16_classical.py --summarize-only` | ep16 | 秒级（汇总）/ ~3.2 h（全量） |

T1/T2 与 F5 同源生成命令：

```bash
cd algos/ep11_dl_benchmark
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_unified_harness_t1_t2.py \
  --device cuda:0 \
  --workers 4 \
  --output-dir ../../output/ep11_unified_harness
```

输出：`output/ep11_unified_harness/{all_arm_metrics.csv,t1_metrics.csv,t2_metrics.csv,run_manifest.json}`，
以及 `output/paper_figures/fig05_main_visual.{png,pdf}`。

### E.2.2 V9 Review 诊断管线

以下脚本在 ep07 环境下依序执行，输出统一到 `output/ep07_v9_review/`：

```bash
uv run python scripts/v9_review/extract_tb_metrics.py          # TB 标量提取
uv run python scripts/v9_review/run_pareto_sweep.py --device cuda  # checkpoint 推理（~30 s/ckpt）
uv run python scripts/v9_review/render_comparison_panels.py     # 对照面板
uv run python scripts/v9_review/run_fusion_baseline.py          # 融合扫描
```

### E.2.3 补充材料图表

| 资产 | 重建脚本 | 环境 |
|---|---|---|
| S-F1 FRC 档案图 | `scripts/paper_figures/fig02_frc.py` | root |
| S-F2 PSF 证据链 | `scripts/paper_figures/figS02_psf_evidence.py` | root |
| S-F9 融合 Pareto | `scripts/paper_figures/figS09_fusion_pareto.py` | root |
| S-F10 V9A 演化条带 | `scripts/paper_figures/figS10_v9a_strip.py` | root |
| D.2 MAP-TV 锚 | `algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py` | ep15（~76 min GPU） |

### E.2.4 训练复现

各臂的完整训练 CLI 存档于 `algos/ep07_unet_sr/scripts/` 下的 `run_training.md`、`run_v9.md`、`run_v10.md`。单臂 60K 步约需 10–20 小时（取决于 batch size 和 GPU 型号）。

随机性控制：合成数据与 split 全部使用显式 seed（FRC: 42/123/456；EP16 子集: 101/202/303、扰动: 401–403；训练池与 drizzle 变体按 $[\text{seed}, \text{epoch}, \text{scene}]$ 确定性采样）。长任务输出 run manifest 记录全部配置与结果状态。

---

## E.3 数据与伦理声明

**样品性质。** 论文热像来自实验室采集的工业芯片 LWIR 温度矩阵（主 session 248 帧 clean set），全幅 FOV（$640 \times 480$ LR / $1280 \times 960$ HR）可直接用于配图，无需 ROI 脱敏或型号匿名化。

**数据分发。** 原始 TXT 温度矩阵不随论文公开（实验室数据管理惯例）。读者可在 TCForge 合成数据上完整复现训练与评估协议。

**可公开内容。** TCForge 合成平台全部源码、所有标定常数（$\theta = 47.6°$、pitch 10 µm、空间分辨率 20 µm、PSF $\sigma$ 区间 [0.2, 0.5]、噪声底 0.0724 °C）、评估脚本和 frame\_audit.csv schema 均随代码包提供。

**诚实声明。** 真实数据结论的第三方复现依赖三层证据：协议（本文）、实验日志（`research_log/` 随仓库提供）和中间产物 CSV（`reports/` 随仓库提供）。
