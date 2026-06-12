# Supplementary E — 复现包说明（中文草稿）

> **角色**: technical appendix 的复现块——代码结构、环境矩阵、每个图表的重建命令。
> **语言**: 中文草稿（2026-06-12 决策），迁 LaTeX 时翻译为英文。
> **对应清单**: `10_writing_handover.md` §3.E。

---

## E.1 代码与环境

### E.1.1 仓库结构（投稿随附代码包的导览）

| 目录 | 内容 | 论文对应 |
|---|---|---|
| `core/src/thermal_core/` | 共享层：数据审计（ep01）、位移模型（displacement）、理论界（ep03）、锚点 gate（ep04）、绘图规范（plotting） | §3 全部标定数字的实现 |
| `configs/` | 物理参数单一来源：`stage_calibration.json`、`psf_calibration.json`、`noise_floor.json`、`alignment/` | §3.3 标定链 |
| `tcforge/` | 物理匹配合成平台（独立包） | §4.2 |
| `algos/ep06_sr_poc/` | forward 算子 + 指标库（`src/common/{forward_model,metrics}.py`） | §3.4 / §5.2 |
| `algos/ep07_unet_sr/` | 学习臂全部训练/推理/评估代码 + `scripts/v9_review/` 诊断管线 | §4.3 / §6 |
| `algos/ep10_*` | drizzle / MAP-TV / TGV 经典臂 | §4.1 |
| `algos/ep15_info_limit/` | M1–M4：FRC、σ 仲裁、去卷积锚 | §5.1 / §3.3 |
| `algos/ep16_budget_robustness/` | E1/E2/E3 预算与鲁棒性矩阵 | §6.4–6.5 |
| `scripts/paper_figures/` + 各 algo `scripts/` | 论文图生产脚本 | F1–F7 |
| `reports/`、`research_log/` | 正式报告与全程实验日志（含 ACL 变更日志） | 可追溯性 |
| `data/`、`output/` | **不入 Git**：原始数据手动放置；产物由脚本重建 | — |

### E.1.2 环境矩阵

| 环境 | 创建 | 用途 |
|---|---|---|
| 根 venv | `uv sync` + `uv pip install -e core/` | 全局脚本、notebook、数据审计 |
| `algos/ep07_unet_sr/.venv` | `cd algos/ep07_unet_sr && uv sync` | UNet 训练/推理/评估（PyTorch + CUDA） |
| `algos/ep10_tgv_sr` conda env | `environment.yml`（含 C/CUDA 编译组件） | TGV 重建（EP16 经由 `--tgv-env` 调用） |
| 其余 algo venv | 各自 `uv sync` | 互相零耦合，可独立删除 |

原则：每个 `algos/xxx/` 是独立项目（独立 venv + lockfile）；共享代码经 `pip install -e ../../core` 注入；**绝不在根 venv 跑算法实验**。

### E.1.3 一键重建链（新机器）

```bash
git clone <repo> thermal_lift && cd thermal_lift
# 放置 data/data_raw/（不随包分发，见 E.3）
uv sync && uv pip install -e core/
uv run python scripts/build_ep01_cache.py        # frame_audit.csv（B.1）
uv run python scripts/build_all_notebooks.py --execute   # 全部 EP notebook + output/ 产物
```

随机性控制：合成与 split 全部显式 seed（FRC 42/123/456；EP16 子集 101/202/303、扰动 401–403；训练 pool 与 drizzle 变体按 [seed, epoch, scene] 确定性）；长任务输出 run manifest（如 `run_manifest.json` 37 runs all success）。

**E.1 待回填**：无——可成稿（蓝本 `AGENTS.md`；终稿删内部路径约定，保留结构与命令）。

---

## E.2 图表 → 生成脚本映射（逐图命令清单）

> 约定：环境列「root」= 仓库根 `uv run`；「ep07」= `algos/ep07_unet_sr` 内 `uv run`；「ep15/ep16」同理。耗时为实测/量级估计。

### E.2.1 主文图表

| 图表 | 脚本 + 命令 | 环境 | 输入依赖 | 耗时 |
|---|---|---|---|---|
| F1 系统+标定链 | `uv run python scripts/paper_figures/fig01_system_calibration.py`（无参） | root | `configs/*.json`、`m3_sigma/sigma_summary.json` | 秒级 |
| F2 FRC 曲线+控制组 | ⬜ **脚本待写** `scripts/paper_figures/fig02_frc.py`——从 `m2_frc/{frc_curve,frc_controls,frc_band_table}.csv` 重绘 CVPR 风格（数据齐，纯排版） | root | `output/ep15_info_limit/m2_frc/` | 秒级 |
| F3 null-space drift | `uv run python scripts/plot_drift_trajectories_paper.py [--refresh]` | ep07 | `checkpoint_selection/{checkpoint_metrics,forward_loss_curves}.csv` | 秒级；⬜ V9D/V9C 落地后 `--refresh` 终稿 |
| F4 Pareto+选点 | `uv run python scripts/plot_checkpoint_selection.py --input-csv ... --output-dir ...` | ep07 | `checkpoint_metrics.csv`（先跑 `extract_checkpoint_metrics.py --arms ...`） | 分钟级；⬜ V9 系列臂扩展后重出 |
| F5/F0 主视觉对比 | ⬜ 等统一 harness + canonical 选点 + 客户许可（形态：全幅 or 中心 ROI 脱敏） | ep07 | T1 同源 | — |
| F7 预算+鲁棒性 | `uv run python scripts/run_ep16_classical.py --summarize-only`（全量重跑 `--arms both --run-tgv`） | ep16 | `output/ep16_budget_robustness/*.csv` | 汇总秒级 / 全量 ~3.2 h |
| T1 主表 | ⬜ 统一 harness（`run_unet_vs_drizzle_2x.py` 谱系）单次重跑全部臂；等 V9A/V9C/V10 选点 | ep11 | GPU 窗口 | 小时级 |
| T2 消融矩阵 | 同 T1 harness + `outputs/*/config.json` 字段（supp C.3 表直读） | ep11 | 同上 | 同上 |

### E.2.2 V9 review / 融合 baseline（supp D.0/D.3/D.7 全部产物）

执行顺序（ep07 环境，输出统一 `output/ep07_v9_review/`）：

```bash
uv run python scripts/v9_review/extract_tb_metrics.py            # TB 标量 → ep07_eval_real_metrics.csv
uv run python scripts/v9_review/run_pareto_sweep.py --device cuda  # checkpoint 推理+缓存 → v9a_pareto_metrics.csv + 散点/strip 图（~30 s/ckpt）
uv run python scripts/v9_review/render_comparison_panels.py      # 对照面板
uv run python scripts/v9_review/run_fusion_baseline.py           # 融合扫描 → fusion_*.{csv,md,png}
```

checkpoint 列表参数化（`--checkpoint LABEL=RUN:STEP:MODE`）→ ⬜ V9C/V10 落地后同命令扫新臂。

### E.2.3 supp 图表

| supp 资产 | 重建 | 环境 |
|---|---|---|
| S-F1 FRC 全档案 | `uv run python scripts/run_m2_frc.py`（~分钟级，CPU） | ep15 |
| S-F2 PSF 三路证据链 | EP09 notebook（`build_all_notebooks.py` 内）+ `run_m3_sigma_arbitration.py`；⬜ 组合图排版脚本待写 | root/ep15 |
| S-F4 视觉 gate panels | `plot_checkpoint_selection.py`（同 F4） | ep07 |
| S-F8 / D.5 三个矩阵 | `run_ep16_classical.py`（同 F7） | ep16 |
| D.2 M4 锚全套 | `uv run python scripts/run_m4_deconv_anchor.py --device cuda`（~76 min GPU） | ep15 |
| D.4.2 EP12 gate | `uv run python scripts/run_ep07x2up_vs_ep12_4x.py` | ep12_benchmark |
| S-T3 合成参数表 | supp C.1 表格（来源 `configs/synthetic/*.json`，无需运行） | — |

### E.2.4 训练复现（学习臂）

各臂完整 CLI 存档于 `algos/ep07_unet_sr/scripts/{run_training.md, run_v9.md, run_v10.md}`；单臂 60K 约 10–20 h（bs 与卡型相关）。⬜ V10 三臂命令以 `run_v10.md` 为准（λ 取值待 smoke 标定，`tmp/codex_next_move_prompt.md` G1.5）。

**E.2 待回填**：⬜ F2 排版脚本；⬜ S-F2 组合脚本；⬜ T1/T2 harness 命令与最终耗时；⬜ V10 命令冻结。

---

## E.3 数据与许可声明 ⬜（等客户许可结论，本周内问）

**框架（许可结论落地后按勾选成稿）**：

1. **原始数据**：客户芯片 LWIR 温度矩阵不公开（商业敏感）；论文展示形态二选一——⬜ 全幅脱敏（匿名型号）/ ⬜ 仅中心 ROI 裁剪。
2. **可公开部分**：TCForge 合成平台 + 全部标定常数（θ/σ/噪声底为测量值，不含客户几何）→ 评估协议与训练管线可在合成数据上端到端复现；真实数据侧提供 `frame_audit.csv` schema 与全部脚本。
3. **诚实声明**：无法随包发布真实帧 → 真实数据结论的复现依赖「协议 + 日志 + 中间产物 CSV」三层证据（reports/ 与 research_log/ 随包提供）。

**E.3 待回填**：⬜ 客户许可结果（`00_status_and_plan.md` 待办第一条）；⬜ 选定展示形态后回填 F5/F0 生产路径（E.2.1）。
