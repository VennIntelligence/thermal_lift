# scripts/ 目录索引

本目录共 72 个脚本（主目录 62 个 `.py` + 1 个 `.sh`，`paper_figures/` 子目录 9 个），按用途分为 7 类。

**日常入口**（新接手者最常用）：

- 构建链：`build_all_caches.py` → `build_all_notebooks.py --execute`（Git 迁移后一键重建全部缓存与 notebook）
- 合成池生产：`generate_training_pool.py`（tcforge 训练池主入口）+ `check_pool_integrity.py`（生成后校验）
- 真实数据评测：`reconstruct_halves.py`（split-half 重建，供 FRC 评测）

其余大部分是**历史诊断脚本**：它们对应 `research_log/algorithm_changelog.md`（ACL-XXX）和各 Episode 记录中的具体结论，保留以维持这些记录的可复现性，**不建议删除**。`paper_figures/` 子目录是旧论文图脚本，已被 `docs/publication_figures/` 取代，仅作历史留档。

除特别说明外，脚本均在根 UV 环境运行：`uv run python scripts/xxx.py`。

## 1. 构建链（日常入口，17 个）

Notebook 与缓存的标准重建流程。Notebook 展示层只读缓存，重计算全部在 cache 脚本中完成；数据或 core 逻辑变更后先重建缓存，再重建 notebook。

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `build_notebook.py` | 将 `notebooks/epXX/fragments/` 下 `.py` 片段按 manifest 拼接为 `.ipynb`，`--execute` 执行并嵌入输出 | `notebooks/epXX/*.ipynb` |
| `build_all_notebooks.py` | 扫描 notebooks/ 下所有含 fragments/ 的目录，批量调用 build_notebook.py | 全部 `.ipynb` |
| `sync_notebook_to_fragments.py` | build_notebook 的逆操作：把直接编辑过的 `.ipynb` cell 源码同步回 fragments/ | `fragments/*.py` |
| `build_all_caches.py` | 按依赖顺序（ep01 → ep10）依次调用各 build_epXX_cache.py | `output/epXX/` 缓存 |
| `build_ep01_cache.py` | 从原始 TXT/BMP 构建 EP01 帧审计缓存 | `output/ep01_data_processing/` |
| `build_ep02_cache.py` | 从原始帧与 EP01 审计构建 EP02 位移标定缓存（θ、pitch） | `output/ep02_displacement_calibration/` |
| `build_ep03_cache.py` | 从配置、EP01 审计和参考帧构建 EP03 缓存（pitch / 分辨率 / 噪声） | `output/ep03_*/` |
| `build_ep04_cache.py` | 构建 EP04 段验证缓存（支持 `--n-jobs` 并行） | `output/ep04_*/` |
| `build_ep05_cache.py` | 校验 EP05 四组输出目录（displacement/capacity/contour/overlay）并写缓存 manifest | `output/ep05_*/` manifest |
| `build_ep06_cache.py` | 校验 EP06 SR 输出并构建 4x ROI 缓存图 | `output/ep06_*/` |
| `build_ep07_cache.py` | 构建 EP07 demo 图与表缓存 | `output/ep07/` |
| `build_ep08_cache.py` | 构建 EP08 Stage 3 notebook 图缓存 | `output/ep08_*/` |
| `build_ep09_cache.py` | 校验 EP09 PSF 标定产物并写缓存 manifest | `output/ep09_psf_calibration/` |
| `build_ep10_cache.py` | 构建 EP10 三算法对比 notebook 图缓存 | `output/ep10_*/` |
| `build_ep13_cache.py` | 构建 EP13 Loss Atlas 缓存（调用 build_ep13_tcforge_demo + loss 图） | `output/ep13_loss_atlas/` |
| `build_ep14_cache.py` | 构建 EP14 4x Loss Atlas 缓存（调用 build_ep14_tcforge_demo） | `output/ep14_4x_loss_atlas/` |
| `build_slides.sh` | 两轮 pdflatex 编译 Beamer slides | `paper/slides/main.pdf` |

## 2. 数据接入与标定（6 个）

原始数据接入（重命名）与物理常数标定 / 实测（pitch、噪声、位移表、真实坐标、光学参考）。这些结论已固化到 `configs/` 与 AGENTS.md 的 Ground Truth 表。

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `rename_data.py` | 将原始连写命名的 TXT/BMP 重命名为 `X_Y_R.ext`（默认 dry-run，`--execute` 执行） | 重命名文件 + `reports/` CSV 映射 |
| `measure_pixel_size.py` | 从 BMP 毫米轴刻度测量探测器采样 pitch，并用 Otsu 外轮廓在 BMP/TXT 间交叉验证 | pitch 测量 JSON / 图 |
| `measure_optical_reference.py` | 从 7 张带标尺的光学显微图测量芯片真实结构几何（指宽/间距/pad），参数化 v7 `quad_meander` motif | 结构几何测量值 |
| `audit_real_noise.py` | 从 248 帧真实 burst 的平坦区实测探测器噪声（条纹 FPN、频谱斜率、时域噪声、坏点率），供合成噪声族取参 | `output/real_noise_audit/` |
| `recompute_ep02_displacement_tables.py` | 从原始帧重算 EP02 TXT 位移诊断表（X 步进尺度拟合仅作可见性诊断，非对齐真值） | `output/ep02_displacement_calibration/` |
| `solve_true_coordinates.py` | 用漂移分解后的轮廓位移反推 repeat 帧的真实台面坐标 | 坐标反推 CSV/JSON |

## 3. AVI 方向诊断（EP02 辅助验证，4 个）

用 AVI 连续扫描视频做位移**方向**的独立验证。AVI 是渲染视频，不是 SR 输入、不是对齐真值（硬教训 #3/#13），这批脚本仅提供方向性旁证。

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `avi_theta_estimation.py` | 从 AVI X/Y-scan 运动方向估计旋转角 θ（辅助验证，不替换 `configs/stage_calibration.json`） | `output/ep02_displacement_calibration/` θ 估计图表 |
| `avi_txt_xline_match_check.py` | 校验 x-scan AVI 与 TXT 固定 Y 行的对应假设（vs 固定 X 列） | 同上目录匹配检查图 |
| `avi_txt_yline_match_check.py` | 校验 y-scan AVI 与 TXT 坐标线的对应关系，区分命名/时序问题与热演化偏差 | 同上目录匹配检查图 |
| `avi_y_direction_check.py` | AVI 运动配准总检：逐轴方向、速度稳定性、重复帧率、异常段 | `avi_direction_summary.csv` 等 |

## 4. EP04–EP06 对齐与容量验证（13 个）

EP04 localization 锚定 / 质量门控、EP05 数据驱动对齐验证与 SR 相位容量评估、EP06 对齐策略消融。结论支撑「contour_refined 对齐」这条默认位移链。

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `run_ep04a_validation.py` | EP04-A 数据驱动多帧 ESF 全局验证，产出 EP06 gate 建议 | `output/ep04_global_validation/` |
| `run_ep05_contour_alignment_validation.py` | 无台面方向假设的数据驱动轮廓对齐验证（highpass NCC 初始化 + Chamfer 精修 + held-out 边缘验证） | 对齐 CSV（默认位移链源头） |
| `run_ep05_alignment_sr_capacity_check.py` | 对比对齐方法，并评估主 session 是否有足够亚像素相位多样性支撑 2x SR | `output/ep05_alignment_sr_capacity/` |
| `run_ep05_alignment_tuning_study.py` | EP05 对齐调参研究：扫 edge percentile / Chamfer 半径 / 精修步长，统一 held-out 打分 | 调参 CSV/JSON + 图 |
| `run_ep05_displacement_reassessment.py` | 用全局帧间配准重评 TXT 微扫描位移，显式分离多种位移定义 | 位移重评 CSV + 图 |
| `run_ep05_edge_line_overlay.py` | 白底边缘线叠加图（TXT/BMP），表观运动视觉体检 | 叠加 PNG |
| `run_ep05_overlay_4x4_check.py` | 4x4 TXT/BMP 叠加网格（列 = 对齐方法，行 = 均值叠加 / 边缘持久性） | 叠加网格 PNG |
| `run_ep05_overlay_alignment_check.py` | 同批帧在不同对齐假设下叠加，看哪种假设保持轮廓锐利 | 叠加诊断 PNG + 指标 |
| `run_ep06_alignment_ablation.py` | EP06 对齐策略消融：SR 固定为 SAA 只换位移来源，报 split-half 稳定性 / 梯度伪影 proxy / 相位覆盖 | `output/ep06_*/` 消融结果 |
| `summarize_ep06_alignment_sweep.py` | 汇总 EP06 对齐扫描的 CSV/JSON 摘要（刻意不加载 `.npy` 重建，可在长跑中途执行） | 汇总表 + 图 |
| `diagnose_affine_outliers.py` | 诊断 filename-affine 拟合的离群帧与稳健重拟合影响 | 离群统计 |
| `diagnose_stage_outliers.py` | 诊断 stage prior（old_stage_model）离群帧（纯标准库实现） | 离群统计 |
| `y14_outlier_analysis.py` | EP02 Y=14 µm 坐标离群分析：是否噪声离群、移除后投影比是否改善 | 终端统计输出 |

## 5. 合成池生成与审计（tcforge 配套，20 个）

TCForge 合成训练池的生成、校验、审计与规划期原型 / 预览。`generate_training_pool.py` 是生产入口；audit / verify / check 系列对应各代池（v3–v9）的质量门控，与 ACL 记录一一对应。

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `generate_training_pool.py` | 生成 TCForge 合成训练池（场景几何 → 温度场 → 前向退化 → LR burst → 特征），**生产主入口** | `data/synthetic/pool_*` |
| `check_pool_integrity.py` | 训练池逐场景**硬校验**（目录/manifest/形状/NaN/PNG 可读性），退出码报告结果 | 终端报告（exit code） |
| `audit_generated_pool.py` | 抽样审计池的自洽与诚实性：GT↔burst roundtrip、band honesty（ACL-023）、形状范围 | 终端报告（exit code） |
| `verify_pool_sharpness.py` | 校验池 GT 锐度（out_of_band ratio，ACL-030 v5 检查）+ 基本完整性 | 终端报告 + 可选 PNG |
| `audit_v6_density.py` | v6 池逐场景内容密度审计（occupancy、连通域统计），只读、可分块续跑 | 密度 CSV + summary.json |
| `audit_defect_detectability.py` | 零训练缺陷可检测性审计：算每个 hole 的输入侧 CNR（analytic+empirical），验证小暗点擦除假设（ACL-063/065-067） | CNR 表 + 报告 |
| `audit_v7_demo_gates.py` | v7 composer demo r4 的 8 项预注册量化 gate（G1–G8，阈值先于运行锁定） | gate 报告 |
| `audit_v7_tcforge_gates.py` | 用真实 tcforge 生产管线重跑 v7 G1–G8 gate（阈值与原型完全相同） | gate 报告 |
| `forward_roundtrip_selfcheck.py` | tcforge 前向模型 5 项自检：位移约定 / 可逆性 / 混叠 / 旋转守恒 / FRC 频带截止 | `output/forward_selfcheck/` |
| `precompute_drizzle_variants.py` | 池侧预生成 K 个增广 drizzle 特征变体，修复训练时现场计算慢 / OOM（ACL-018） | 池内 `drizzle_variants_*.npy` |
| `evaluate_thermal_chip_phantom.py` | 写 ThermalChipPhantom 合成数据集的轻量评估摘要 | `output/thermal_chip_phantom/` |
| `preview_v6_cpu.py` | v6 CPU/part 池并排预览图（motif 多样性、可恢复性、与真实帧锚点对比、中心缩放） | 预览 PNG |
| `preview_v7_planning_compare.py` | v7 规划期内容轴对比 sheets（v5 legacy / v6 current / v6 reweighted 几何 mask） | 对比 sheets PNG |
| `v7_composer_demo.py` | v7 chip composer 原型 r4（owner 视觉评审用，后收编为 tcforge motif family） | demo sheets PNG |
| `v7_content_demo.py` | v7 内容族 demo（v6→v7 重生成前的 owner eyeball gate；被否决原型留档在文件内） | demo sheets PNG |
| `generate_v7_demo_minipool.py` | 用 v7 composer demo r4 机械生成 50 场景 mini-pool + eyeball sheets（一次性评审产物） | `outputs/v7_demo_minipool/` |
| `visualize_scene_samples.py` | 快速生成 6 个随机场景样例（HR mask + 模糊 LR 预览） | `output/ep07/scene_samples/` |
| `make_tcforge_pipeline_figure.py` | 生成 TCForge 四阶段合成管线示意图（paper §4.3 素材） | 管线示意 PNG |
| `build_ep13_tcforge_demo.py` | 生成 EP13 Loss Atlas 用 TCForge 训练 demo bundle（被 `build_ep13_cache.py` 调用） | demo bundle |
| `build_ep14_tcforge_demo.py` | 生成 EP14 4x Loss Atlas 用训练 demo bundle（被 `build_ep14_cache.py` 调用） | demo bundle |

## 6. Split-half 重建与评测（3 个）

对 248 帧真实数据做相位分层 A/B split-half 重建，供 FRC 等稳定性评测消费；外加一个 drizzle 伪影诊断。依赖 `algos/ep07_unet_sr`、`algos/ep10_*`、`algos/ep15_info_limit` 的代码与 checkpoint。

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `reconstruct_halves.py` | 相位分层 split-half（A/B）后用 v11 solver / TGV / MAP-TV 各重建两半（`--seed/--methods` 支持多 split 复验） | `output/stage0c_frc_recons/*_{a,b}.npy` |
| `reconstruct_c_d_halves.py` | reconstruct_halves 的特化版：对 C_nodr / D_dr01 两个 v13 checkpoint 重建 A/B 半（路径硬编码） | 同上目录 `C_nodr_*` / `D_dr01_*` |
| `diagnose_drizzle_waffle.py` | FFT 诊断 phase-bin drizzle 的背景网格伪影（ACL-032），对比 aligned_mean 与 phase-bin 两种 warm-start | 终端报告 + 可选 PNG |

## 7. 旧论文图（历史，`paper_figures/` 9 个）

早期论文初稿的图脚本，输出到 `output/paper_figures/`。**已被 `docs/publication_figures/` 的新图系统取代**，保留仅为历史草稿的可复现性；新图请一律在 `docs/publication_figures/scripts/` 下维护。（原 `fig00_teaser_placeholder.py` 为无引用的空白占位图，2026-07-24 收尾时已删除。）

| 脚本 | 一句话用途 | 主要输出 |
|------|-----------|---------|
| `collect_promoted_supp.py` | 把各 episode 已产的补充图拷贝为稳定 `figSxx_*` 命名 + provenance manifest | `output/paper_figures/figSxx_*` |
| `fig01_observation_pipeline.py` | 观测管线示意图（用 tcforge 实渲染缩略图） | fig01 PNG/PDF/SVG |
| `fig01_system_calibration.py` | F1 系统几何 / 标定链 / 网格尺度图 | fig01 PNG/PDF |
| `fig02_frc.py` | F2 + S-F1 相位分层 split-half FRC 曲线（读 EP15 M2 产物） | fig02 PNG/PDF |
| `fig05_combined_visual.py` | 合并版 F5 视觉图：中心 comb ROI + held-out ROI2，温度 / highpass 双域 4 行 | fig05 PNG/PDF |
| `fig05b_roi2_holdout.py` | F5b held-out 右上 ROI2 图 + 结构 proxy CSV（固定几何窗口，非挑选） | fig05b PNG/PDF + CSV |
| `figS02_psf_evidence.py` | S-F2 三路 PSF 证据链 + M3 仲裁图（读 EP09/EP15 产物） | figS02 PNG/PDF |
| `figS09_fusion_pareto.py` | S-F9 零训练融合基线 Pareto（fidelity vs sharpness proxy） | figS09 PNG/PDF |
| `figS10_v9a_strip.py` | S-F10 Hybrid 细线窗口 checkpoint 演化条带（温度域 3×3） | figS10 PNG/PDF |

---

各脚本的详细参数、输入契约与运行示例见其头部 docstring；算法变更背景见 `research_log/algorithm_changelog.md`（ACL-XXX 编号在上表中已标注对应关系）。
