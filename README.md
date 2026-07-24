# Thermal Lift — 红外热像微扫描超分辨率（2x Contour-Level SR POC）

面向工业芯片检测，在 20 µm 采样 pitch / 20 µm 空间分辨率的 LWIR 温度矩阵上，用主会话 248 帧微扫描序列验证 **2x 轮廓级超分辨率**（输出 10 µm/sample 网格）：让芯片内部结构/形状的轮廓更清楚、更稳定，而非追求计量级温度读数。

技术路线：物理前向算子（PSF 模糊 + 2x 块降采样）+ 物理约束展开求解器（unrolled solver：神经 prox 先验与数据一致性 DC 步交替）+ TCForge 程序化合成训练池；以经典方法 drizzle / TGV / MAP-TV 为独立参照与"要打败的对象"。2x 是当前数据的合理倍率——实测亚像素相位占用仅 11/25 bin，>2x 存在相位饥饿（EP15-M1，GALLERY fig34）。

项目已进入收尾交接阶段（2026-07）。全部 80 条算法变更记录在 `research_log/algorithm_changelog.md`（ACL-001–080）。

---

## 关键成果摘要

以下每条结论均可在 `docs/publication_figures/GALLERY.md`（头条成果表）与 `research_log/algorithm_changelog.md` 中溯源：

1. **权威可恢复频带 25.45 ± 0.73 µm** — 修复约 0.3 px 的逐帧对齐误差（头号信息瓶颈）后，可恢复空间周期从 34.07 µm 改善到 25.45 ± 0.73 µm（ACL-048；GALLERY fig21）。
2. **评测方法论翻案** — 神经输出网格自带 +0.5 px 角点约定，曾系统性压低所有神经×经典对比约 0.4 FRC 点；校正后 v11×TGV cross-FRC@30µm 从 0.04 升至 0.83，"神经带内破坏"旧结论被推翻（ACL-049；GALLERY fig07 / fig92）。
3. **神经冠军 depb9v6** — 真实域 cross-FRC@30µm 0.6611（最强经典基线 TGV 为 0.7017 / cutoff 23.03 µm），但在 13/13 个 OOD 池上完胜 TGV(oracle)（Δ+0.171），且低频干净（GALLERY fig02 / fig03 / fig06 / fig99）。
4. **点保真攻坚：孤立点抹除率 43% → 0.00%** — 工业检测最不可妥协的轴：小暗缺陷点抹除率从 v7 训练池病理的 43%，经 v8（4.35%）、v9（1.55%）逐代修复至 v9-3k 的 0.00%（GALLERY fig01 / fig62；ACL-066~074）。
5. **点保真 ↔ OOD 权衡定律** — 点保真冠军 v9-3k 在域外 0/13 全输，两轴呈单调权衡；最终判决选择均衡冠军 depb9v6，而非任一单轴极端（ACL-079；GALLERY fig99）。

---

## 数据与物理口径速查

新接手者最容易搞错的几个口径（完整 Ground Truth 表见 `AGENTS.md`）：

| 口径 | 值 | 备注 |
|---|---|---|
| 原始主扫描 session | 255 帧（session=2） | 仅作物理温度段诊断，不直接作 SR 输入 |
| SR 默认输入 | **248 帧 clean set** | 剔除 `R != 0` 重复/补采帧后的 `is_sr_usable` 集合 |
| 采样 pitch / 空间分辨率 / SR 网格 | 20 µm/px / 20 µm / 10 µm/sample | 三个概念不可互换（GALLERY fig19） |
| 台架→像素旋转角 θ | 47.6° | AVI 连续扫描独立验证覆盖；stage command 只能作 prior，不是对齐真值 |
| 探测器噪声底 | 0.0724 °C | |
| session 边界 | 必须按 `acquisition_order`（mtime）判定 | 跨 session 帧绝不混用（温度跳变中位 3.55 °C） |

---

## 目录地图

| 路径 | 说明 | Git |
|---|---|---|
| `AGENTS.md` | 项目持久记忆：规范、物理常数、硬教训、部署流程 | ✅ |
| `core/` | 共享库 `thermal_core`（24 模块：IO、坐标模型、缓存、指标、绘图） | ✅ |
| `algos/` | 12 个完全隔离的算法子项目，各自独立 venv（索引见下节） | ✅ 代码（`.venv`/`outputs` ❌） |
| `tcforge/` | 程序化合成训练池生成引擎（独立 UV 包，见 [`tcforge/README.md`](tcforge/README.md)） | ✅ |
| `notebooks/` | 17 个分析 notebook，`fragments/` 构建制（`.ipynb` 是构建产物） | ✅ fragments（`.ipynb` ❌） |
| `scripts/` | 约 61 个构建/工具脚本（`build_notebook.py`、`build_all_notebooks.py`、`build_all_caches.py` 等） | ✅ |
| `configs/` | 全局物理常数 + alignment + synthetic 训练池配置（v8/v9 等） | ✅ |
| `research_log/` | `algorithm_changelog.md`（ACL-001–080）+ `episodes/` 各 Episode 进度与决策 | ✅ |
| `paper/` | `reports/` 正式分析报告 + aaai/zh_conf/slides 论文骨架（见 `paper/README.md`） | ✅ |
| `docs/` | `dataset_description.md`、`plotting_standards.md`、`publication_figures/` 成果图库 | ✅ |
| `data/` | 原始与预处理数据 | ❌（手动拷贝） |
| `output/` | 实验数据产物（CSV、图表） | ❌（notebook/脚本重建） |
| `remote_inbox/` | 机器间产物投递目录，仅允许 rsync/scp 同步 | ❌（硬规则，严禁入 git） |

---

## 推荐阅读顺序

新接手者按此顺序建立全貌：

1. **`AGENTS.md`** — 项目规范、已确认物理常数（Ground Truth 表）、硬教训清单、目录约定与部署流程。必读第一站。
2. **`research_log/README.md`** — Episode 路线图：各 Episode 做了什么、当前主线在哪。
3. **`docs/publication_figures/GALLERY.md`** — 72 图叙事图册，最完整的成果展示：从数据审计、评价方法论修复、solver 架构演化，到点保真攻坚与冠军判决。开头"项目一页纸"可 10 分钟速览全局。
4. **`research_log/algorithm_changelog.md`** — 需要深挖某个具体判决时，按 ACL 编号查完整证据链。
5. **各 `algos/*/README.md`** — 复现或续做某条算法路线时读对应子项目文档（索引见下节）。

---

## 算法子项目索引（algos/）

每个 `algos/xxx/` 都是独立 UV/conda 项目（自带 `pyproject.toml` 与 venv），互相零耦合；共享代码通过 `pip install -e ../../core` 引入。绝不在根目录 venv 中运行算法实验。

| 子项目 | 角色 | 一句话说明 | 文档 |
|---|---|---|---|
| `ep07_unet_sr` | **主线** | UNet 基线 → 物理约束展开 solver（神经 prox + DC），冠军 depb9v6 的训练与评测主战场；配合 `tcforge/` 合成池与 `configs/synthetic` v8/v9 | [README](algos/ep07_unet_sr/README.md) |
| `ep12_4x_sr` | 4x 路线 | drizzle-informed 混合式 4x 温度恢复模型 | [README](algos/ep12_4x_sr/README.md) |
| `ep06_sr_poc` | 经典基线 | 早期经典 SR POC（SAA/IBP/MAP-TV，`src/common/` 被 EP09/EP10 复用） | [README](algos/ep06_sr_poc/README.md) |
| `ep10_drizzle` | 经典基线 | STScI drizzle 多帧网格化叠加，cross-FRC 评测的独立参照锚 | [README](algos/ep10_drizzle/README.md) |
| `ep10_tgv_sr` | 经典基线 | MAP-TGV（CCPi 正则器）重建，最强经典基线（cutoff 23.03 µm） | [README](algos/ep10_tgv_sr/README.md) |
| `ep10_map_tv_sweep` | 经典基线 | MAP-TV 正则强度 × PSF σ 联合参数扫描 | [README](algos/ep10_map_tv_sweep/README.md) |
| `ep11_dl_benchmark` | 评测资产 | 真实 248 帧上的 2x 视觉基准（UNet vs TGV，同一 highpass 域） | [README](algos/ep11_dl_benchmark/README.md) |
| `ep12_4x_benchmark` | 评测资产 | 4x 视觉基准（drizzle-informed UNet vs bare drizzle） | [README](algos/ep12_4x_benchmark/README.md) |
| `ep15_info_limit` | 理论审计 | 第一性原理信息上限检查（相位占用分析 → "2x 是合理倍率"判决） | [README](algos/ep15_info_limit/README.md) |
| `ep16_budget_robustness` | 理论审计 | 帧预算与稳健性经典对照（N=31/62/124/248 相位分层子集） | [README](algos/ep16_budget_robustness/README.md) |
| `ep08_inr_sr` | 实验分支 | 隐式神经表示（SIREN/WIRE/DeepDecoder/DIP）SR 路线，后被 solver 主线取代 | [README](algos/ep08_inr_sr/README.md) |
| `ep09_psf_calibration` | 标定 | PSF σ 标定（点校准 FAIL → 收敛为 σ∈[0.1, 0.4] px 鲁棒带策略） | [README](algos/ep09_psf_calibration/README.md) |

---

## 快速开始（新机器部署）

完整步骤见 `AGENTS.md` 的「🚀 新机器部署」节，概要如下：

```bash
# 1. 克隆仓库，安装 uv
git clone <repo-url> thermal_lift && cd thermal_lift

# 2. 放置数据（data/ 不入 git，需手动拷贝）
#    data/data_raw/{infrared_avi, optical_fig, name_rules.txt}

# 3. 根环境依赖 + core 共享库
uv sync
uv pip install -e core/

# 4. 一键构建并执行全部 notebook（生成 .ipynb 与 output/ 产物）
uv run python scripts/build_all_notebooks.py --execute
```

- notebook 全部由 `fragments/` 构建，**绝不手动编辑 `.ipynb`**；单个构建用 `scripts/build_notebook.py notebooks/epXX_name --execute`。
- 各 `algos/xxx/` 子项目需进入对应目录单独 `uv sync`（个别使用 conda，见其 README）。
- 缓存类产物可用 `scripts/build_all_caches.py` 重建。

## 新样本接入

未来接入新的芯片样本/新采集数据时，按 [`docs/new_data_intake.md`](docs/new_data_intake.md) 的流程执行（命名解码、采集顺序审计、session 切分、clean set 筛选）。

