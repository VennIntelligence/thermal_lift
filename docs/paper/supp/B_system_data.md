# Supplementary B — 系统与数据细节（中文草稿）

> **角色**: technical appendix 的系统/数据块，支撑主文 §3.1–3.3；审稿人核查「benchmark 可信度」的主要落点。
> **语言**: 中文草稿（2026-06-12 决策），迁 LaTeX 时翻译为英文。
> **对应清单**: `10_writing_handover.md` §3.B（B.1/B.2 ★，B.3）。

---

## B.1 数据审计：263 → 255 → 248（对应主文 §3.2）

### B.1.1 审计链总览

| 指标 | 数值 | 出处 |
|---|---:|---|
| 原始 TXT/BMP 对 | 263 / 263（矩阵 480×640） | `reports/ep01_data_processing/audit_report.md` L11–13 |
| 物理主 session（session=2） | 255 帧 | 同上 L14, L79 |
| Clean SR 默认输入 | **248 帧** | 同上 L15, L59 |
| Clean 集坐标覆盖 | 248/256（16×16 raster） | 同上 L18 |
| Clean 集均温跨度 | 0.544 °C | 同上 L17 |
| 噪声底 | 0.0724 °C | 同上 L88–93；`configs/noise_floor.json` |

### B.1.2 三个温度段（按 acquisition_order）

| session | 帧数 | acquisition_order | 均温 mean/median (°C) | 出处 |
|---:|---:|---|---|---|
| 0 | 1 | 0 | 21.757 | `audit_report.md` L77 |
| 1 | 7 | 1–7 | 19.862 / 19.852 | L78 |
| 2 | 255 | 8–262 | 23.292 / 23.263 | L79 |

**边界跳变**：S0→S1 −1.662 °C（22.9× 噪声）、S1→S2 +4.157 °C（57.4× 噪声）；audit 汇总口径为**中位 2.91 °C / 最大 4.16 °C（约 40×/57× 噪声底）**（L16, L85–93, L122）。

> **🔴 口径警告（写作必须处理）**：`AGENTS.md` 与主文草稿 `04_problem_forward_model.md` §3.2 目前写「median 3.55 °C (49×)」，与权威 audit 报告的 **2.91/4.16 °C（40×/57×）** 不一致（3.55≈两口径均值，疑为早期版本遗留）。supp 以 `audit_report.md` 为准；⬜ 主文 §3.2 与 `AGENTS.md` 对应行待统一修正后，此警告框删除。

结论不变：跨 session 跳变达噪声底数十倍，**跨 session 帧绝不混用**。

### B.1.3 R≠0 剔除与 is_sr_usable 定义

263 帧角色分配（`audit_report.md` L47–59）：R=0 共 253 帧、R=1 共 4 帧、R=2 共 6 帧；主 session 内剔除 R≠0 共 7 帧（255−248），另 5 帧属其他 session、3 帧为 prewarm 重复。

```
is_sr_usable = (session == 2) AND (R == 0) AND (480×640 且无 NaN/Inf)
```

实现：`core/src/thermal_core/ep01.py` L525–583（`is_main_session` 为剔除后等价别名）。重复帧保留为 repeat_diagnostic 角色（重复测量差异 1–2 °C std，不跨 repeat 混合——`AGENTS.md` 硬教训 7）。缺失坐标（非 R 问题）：(14,6), (16,6), (16,16)（L61–69）。

### B.1.4 命名解码与歧义特判

原始名为 X+Y+R 连写数字；X,Y ∈ {0,2,…,20,24,28,32,36,40}，R ∈ {0,1,2}。解码按前导零约束枚举合法分割，多解时优先较小 X（`scripts/rename_data.py` L134–215）。特判例：

| 原始名 | 解码 (X,Y,R) | 性质 |
|---|---|---|
| `2400` | (24, 0, 0) | 硬编码特判（非 (2,40,0)） |
| `0200` | (0, 20, 0) | 前导零约束 |
| `2000` | (20, 0, 0) | 显式特判 |
| `2，400` | (2, 40, 0) | 中文逗号标注 |

### B.1.5 「13 个假 session」教训与 frame_audit 生成

文件名是坐标标识不是时序：按文件名字母序检测 session 会得到 **13 个假温度段**（重复帧/早期帧与 raster 网格交错制造伪断点）；按 mtime 生成的 `acquisition_order` 检测得到 3 个真实温度段（`audit_report.md` L73–79）。

`frame_audit.csv` 管线（`core/src/thermal_core/io.py` L42–176 + `ep01.py`）：读 TXT 记录 mtime 与温度统计 → `acquisition_order` = mtime 升序（同 mtime 按文件名）→ 相邻均温跳变分段（阈值 max(10×中位跳变, 0.5 °C)）→ 追加 `is_sr_usable` 列。重建命令：`uv run python scripts/build_ep01_cache.py`。

### B.1.6 AVI/BMP 排除依据（数值输入侧）

| 项 | 数值 | 出处 |
|---|---|---|
| BMP | 263 张，渲染图，仅视觉参考 | `audit_report.md` L23–33 |
| AVI 总数 | 16（8 X-scan + 8 Y-scan），覆盖仅 0–14 µm 精细段 | `reports/ep02_displacement_calibration/`（addenda） |
| 格式 | 8-bit 渲染视频，无温度矩阵 | 同上 |
| 重复帧率 | ~67%（X 66.9% / Y 67.2%） | `avi_registration_addendum.md` L20–21 |
| 去重后运动段 | 中位 202（X）/198（Y）对/视频 | 同上 L21–22 |

角色边界：AVI 仅用于位移**方向**的独立 consistency check（supp A.5.2）；绝不作 SR 数值输入（`AGENTS.md` 硬教训 3）。

**B.1 资产依赖**：`reports/ep01_data_processing/audit_report.md`（权威）；`core/src/thermal_core/{io.py, ep01.py}`；`scripts/rename_data.py`；`data/processed/frame_audit.csv`（重建：`scripts/build_ep01_cache.py`）。
**B.1 待回填**：⬜ 主文/`AGENTS.md` 跨 session 跳变数字统一（见 B.1.2 警告框）。

---

## B.2 对齐管线与质量 gate（对应主文 §3.3「Alignment and gates」）

### B.2.1 两级对齐管线

**第一级 highpass NCC 初始化**（`algos/ep05_*` 谱系，脚本 `scripts/run_ep05_contour_alignment_validation.py`）：

| 参数 | 值 |
|---|---|
| 预处理 | highpass = frame − Gaussian(σ=6.0 px) |
| 参考帧 | `6_16_0.txt`；ROI 中心 360×360 |
| 整数搜索 | ±18 px |
| 亚像素 | 相关峰 3×3 邻域二次抛物线拟合（无 FFT 上采样） |

**第二级 contour refinement**：

| 步骤 | 方法 |
|---|---|
| 边缘提取 | highpass → Sobel 梯度幅值 ≥ 93rd percentile |
| 目标 | 最小化移动帧边缘点到参考边缘 EDT 的均值（Chamfer） |
| 搜索 | NCC 初值 ±1.0 px 网格、步长 0.25 px |
| 验证 | 边缘点奇偶分为 fit/holdout，报告 **held-out** Chamfer |
| 规模 | 每帧最多 8000 边缘点（确定性子采样） |

### B.2.2 Held-out Chamfer 全链（五步）

| 对齐源 | Chamfer median / P90 (px) | 出处 |
|---|---|---|
| 无对齐 | 0.3976 / 0.7080 | `reports/ep05_sr_reassessment/displacement_reassessment.md` L72 |
| stage command prior | 0.2462 / 0.4422 | L73 |
| filename affine prior | 0.1701 / 0.2043 | L74 |
| highpass NCC 初始化 | 0.1558 / 0.1752 | L75 |
| **contour refined** | **0.1332 / 0.1610** | L76 |

> **口径注意**：主文常用三步缩写「0.381 → 0.240 → 0.134」（无对齐 → stage prior → refined）；若提及 NCC 中间步，应写 ~0.156 px。逐帧修正量级：相对 stage prior 中位 0.39–0.43 px（L13）——位移先验与数据驱动对齐之间的真实差距。

**位移场落盘**：`configs/alignment/contour_alignment_results.csv`（248 帧，含 `refined_align_dx/dy_px`、`refined_holdout_chamfer_px`、`ncc_peak`；`paths.json` 为路径索引）。全部 SR 输入默认走 contour_refined（`unet_sr/real_eval.py` 同口径）。

### B.2.3 EP04 锚点体系（localization quality gate）

**构成**：参考帧上外轮廓 Otsu+形态学提取、stride 8 px 分段 → **84 段**；外 mask 内正/反 Otsu、min_area 100 px、stride 6 px → **390 段**；13 条完整 R=0 X-scanline（208 帧 ⊂ 248 clean）逐线评估（`reports/ep04_global_validation/validation_report.md` L15–33；实现 `core/src/thermal_core/ep03.py` L450–658、`ep04.py`）。

**门控阈值**（`ep04.py` L39–48）：SNR≥8、ΔT≥0.5 °C、NCC peak≥0.85、相位覆盖≥0.15 px、拟合 σ∈[0.8,1.3] px、split-half ≤0.04（A 级）/0.06（B 级）、PSF 敏感性 ≤0.03。

**A 级定位精度**：

| 指标 | 外轮廓 | 内轮廓 |
|---|---:|---:|
| A 级段数 | 28 | 139 |
| split-half 中位 | 0.0277 px | 0.0273 px |
| split-half P90 | 0.0622 px | 0.0847 px |
| CRB ratio 中位 | 1.89× | 2.03× |
| NCC peak 中位 | 0.9825 | 0.9863 |

（`validation_report.md` L34–43；CRB 对照 supp A.1.6——实测约为理论下界 2 倍，合理。）

### B.2.4 三角色划分（gate 的使用方式）

划分逻辑（`ep04.py` L2206–2234）：robust = pass 且 pass_rate≥0.70 且 split-half≤0.06 且 CRB≤5× 且相位覆盖≥0.15 px；robust 中 segment_id % 4 ≠ 0 → **alignment_input**，% 4 == 0 → **holdout_validation**；其余 → **sr_target_not_truth**。

| contour | 角色 | 段数 | split-half 中位 | pass rate 中位 |
|---|---|---:|---:|---:|
| outer | alignment_input | 23 | 0.0193 px | 1.000 |
| outer | holdout_validation | 23 | 0.0250 px | 0.692 |
| outer | sr_target_not_truth | 38 | 0.0607 px | 0.000 |
| inner | alignment_input | 42 | 0.0189 px | 1.000 |
| inner | holdout_validation | 45 | 0.0213 px | 0.692 |
| inner | sr_target_not_truth | **303** | 0.0375 px | 0.000 |

通过率：外轮廓 54.8% / 内轮廓 22.3%；内轮廓主要失败原因 `sigma_out_of_range` 145 段、`fit_error` 71、`split_half_high` 45、`low_phase_coverage` 18（`validation_report.md` L35–37, L68–71, L111–120）。

**论文立场（必写）**：未过 gate 的 303 个内轮廓段恰恰是**重建要改善可见性的目标**——它们作 target-not-truth，绝不作对齐真值；这是「EP04 是 anchor/gate 而非交付目标」的具体化（`AGENTS.md` 关键定义）。

### B.2.5 相位 bin 保持性（2x 的关键前提）

| 对齐源 | 2x 占用 bin | 帧数/格 | entropy |
|---|---|---|---:|
| 无对齐 | 1/4（坏 3） | 0–248 | 0.000 |
| stage prior | 4/4 | 60–64 | 1.000 |
| NCC init | 4/4 | 59–66 | 0.999 |
| contour refined | 4/4 | 59–67 | 0.999 |

（`displacement_reassessment.md` L70–102）数据驱动修正**没有破坏** stage 设计的相位均匀性 → 2x 重建的相位多样性前提成立。**风险注记**：contour refined 在 3x/4x 网格出现 phase collapse（3x 仅 4/5、4x 仅 4/12 占用）→ 不能作为高倍率可行性证据（与 A.1.5 4x 出界结论同向）。

**B.2 资产依赖**：`reports/ep04_global_validation/validation_report.md`、`reports/ep05_sr_reassessment/displacement_reassessment.md`、`reports/ep02_displacement_calibration/calibration_report.md`（EP02 对照表）；`configs/alignment/{contour_alignment_results.csv, paths.json}`；`core/src/thermal_core/{ep03.py, ep04.py, displacement.py}`；S-F7（对齐链与 gate 图，🔧 选图整理，源 `output/ep04_*`、`output/ep05_*`）。
**B.2 待回填**：无——可成稿。⬜ 主文 §3.3 是否保留 EP04 表格由压页结果定（handover §2 建议细节全下沉到本节）。

---

## B.3 微扫描运动学与热漂移（对应主文 §3.2 的 anisotropy 句）

### B.3.1 raster 采集间隔的各向异性

| 诊断 | 数值 | 出处 |
|---|---:|---|
| 行内 X 相邻坐标 acquisition gap 中位 | **1 帧** | `reports/ep02_displacement_calibration/calibration_report.md` L35–36 |
| 行间 Y 相邻坐标 gap 中位 | **~16 帧** | 同上 |
| 行内 X 相邻对（clean） | 232 对 | L16, L103 |
| 行间 Y 跳变次数 | 15 | L17 |
| X-scan AVI↔TXT 正确轴 gap | 中位 1 | L88 |
| Y-scan AVI↔TXT 正确轴 gap | 中位 16 | L89 |

**对方法的影响**（主文 §4.1 TGV 的设计动机）：X 邻居时间连续、Y 邻居隔整行 → 数据约束各向异性 + bilinear scatter 权重集中在固定 HR 行 → 各向同性正则会产生水平条纹；TGV 的椭圆对偶投影（Y 比 1.5）与 coverage 加权由此而来（supp C.4.2）。

### B.3.2 Y-only 标定失效（stage ≠ truth 的实证）

| 指标 | 数值 | 期望 | 出处 |
|---|---:|---:|---|
| Y 4 µm / 2 µm 可见位移比 | **0.64** | 2.0 | `calibration_report.md` L40, L64 |
| X 4 µm / 2 µm 比（对照） | 2.05 | 2.0 | L39 |
| Y 2 µm projection/prior | 1.601 | ~1 | L64 |
| Y 4 µm projection/prior | 0.511 | ~1 | L64 |
| 行间过渡 NCC RMS | ~1.96 px | — | L54–56 |

机理：固定 X 的 Y 邻帧隔 ~16 帧采集，热场演化污染 NCC，位移单调性被破坏（X 对照组正常）→ **Y-only 坐标相邻帧不可做定量位移标定**（`AGENTS.md` 硬教训 12），同时构成「stage command 只能作 prior」的直接证据（与 B.2.2 的 0.39–0.43 px 修正量级互证）。

### B.3.3 热场演化时间尺度

| 尺度 | 数值 | 出处 |
|---|---:|---|
| session=2 相邻帧均温跳变中位 | 0.008 °C | `reports/ep01_data_processing/`（dataset_description）|
| session=2 首尾均温漂移 | −0.60 °C | 同上 |
| clean 248 帧均温跨度 | 0.544 °C | `audit_report.md` L17 |
| 跨 session 边界跳变 | 中位 2.91 / 最大 4.16 °C | `audit_report.md` L16（B.1.2 警告框同源） |
| 主 session 累计位移轨迹 | 2.49 × 7.11 px 包络，路径长 61.18 px | `displacement_reassessment.md` L34–35 |

判读：帧际漂移（0.008 °C ≪ 噪声底）支持行内相邻帧可比；session 内慢漂移（−0.60 °C，8.3× 噪声底）则要求任何 split 设计保持时间混合（A.4.4 漂移控制正是该效应的度量）；跨 session 跳变（40–57×）直接禁止混用。

### B.3.4 位移覆盖实测（clean main session）

| 口径 | median (px) | 出处 |
|---|---:|---|
| X 相邻 2 µm 实测位移 | 0.0992 | `displacement_reassessment.md` L30 |
| X 相邻 4 µm | 0.1937 | L31 |
| R=0 X scanline 40 µm 端到端 | 2.0768 | L32 |
| R=0 Y column 40 µm 端到端 | 4.4161 | L33 |

（设计值对照：2 µm 命令 = 0.2 px 幅值；行内实测略低，端到端与 4.0 px 设计同量级——细步进的逐步可见位移系统性小于命令值，是「prior 而非 truth」的又一证据。）

**B.3 资产依赖**：`reports/ep02_displacement_calibration/calibration_report.md`、`reports/ep05_sr_reassessment/displacement_reassessment.md`、`reports/ep01_data_processing/audit_report.md`；`AGENTS.md` 硬教训 7/11/12。
**B.3 待回填**：无——可成稿。
