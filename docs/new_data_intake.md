# 新样本数据接入指南（New Data Intake Guide）

> 适用场景：教授/客户提供一批**新的红外微扫描数据**（TXT 温度矩阵 + BMP 参考图），需要从零接入本项目并跑通 2x contour-level SR pipeline。前置阅读：根目录 `AGENTS.md`。
> 本文所有命令均已对照脚本源码（argparse）核实；未核实的细节一律标注「见脚本 `--help`」。

---

## 1. 数据放置与命名

### 1.1 目录放置

原始文件放入 `data/data_raw/infrared_avi/`（TXT + BMP + 可选 AVI），光学参考图放 `data/data_raw/optical_fig/`。`data/` 整体不入 git，靠手动拷贝。**关键**：拷贝必须保留文件 mtime（`rsync -a` / `cp -p`），否则采集顺序（acquisition_order）被破坏，session 检测失效——这是整条 pipeline 的时间轴基础。

### 1.2 命名规则与重命名脚本

原始文件名是 X+Y+R 连写数字（如 `2100` = X=2, Y=10, R=0），目标格式为 `X_Y_R.ext`。使用：

```bash
uv run python scripts/rename_data.py                    # dry-run（默认，只打印计划）
uv run python scripts/rename_data.py --report           # 额外输出 CSV 映射表到 reports/rename_mapping.csv
uv run python scripts/rename_data.py --execute --report # 实际执行重命名 + 输出 CSV
uv run python scripts/rename_data.py --data-dir <路径>  # 数据不在默认目录时指定
```

脚本消歧逻辑（已核实源码 `scripts/rename_data.py`）：

- 合法坐标集合从 `configs/coordinate_set.json` 读取（X/Y ∈ {0,2,...,20,24,28,32,36,40}，R ∈ {0,1,2}）；**新样本若坐标集不同，先改这个 config**，否则解码会大量报「无合法分割」。
- 最右 1 位 = R；剩余部分枚举 X|Y 分割，排除前导零（`0200` → X=0, Y=20, R=0）。
- 硬编码特判 `2400` → X=24, Y=0, R=0（不是 X=2, Y=40）；中文逗号标注 `2，400` → X=2, Y=40, R=0。
- 多个合法候选时取 X 最小的解释，并在终端打印消歧记录，需人工复核。

执行前先看 dry-run 报告的「ERRORS / AMBIGUITIES」段；目标文件名有冲突时脚本会拒绝执行。

### 1.3 ⚠ 流程断点：rename 映射表路径

`rename_data.py --report` 把映射表写到 `reports/rename_mapping.csv`，但 EP01 cache 构建（`thermal_core/ep01_cache.py`）从 `output/ep01_data_processing/rename_mapping.csv` 读取它做溯源审计。**需手工拷贝一次**：

```bash
mkdir -p output/ep01_data_processing
cp reports/rename_mapping.csv output/ep01_data_processing/
```

不拷贝也能跑（frame_audit 会标注 provenance 缺失），但会丢失原始文件名溯源。

---

## 2. 数据审计（EP01）

### 2.1 构建 frame audit

```bash
uv sync && uv pip install -e core/      # 根环境准备（仅首次）
uv run python scripts/build_ep01_cache.py          # 增量构建
uv run python scripts/build_ep01_cache.py --force  # 数据变更后强制重建
```

产物写入 `output/ep01_data_processing/`（核心是 `frame_audit.csv`），审计报告写入 `paper/reports/ep01_data_processing/audit_report.md`。

### 2.2 审计逻辑要点（已核实 `thermal_core/io.py` + `ep01.py`）

- **acquisition_order 按 mtime 生成**（mtime 相同再按文件名），这是唯一合法时间轴。**绝不按文件名字母序推断采集顺序**——旧数据上字母序会制造 13 个假 session。
- **session 检测**：按 acquisition_order 排序后，相邻帧均温跳变超过阈值（默认 `max(10×中位跳变, 0.5°C)`）即断开温度段；帧数最多的段为主 session。
- **clean SR 输入集**（`is_sr_usable=True`）：主 session ∩ R=0 ∩ 矩阵有效（480×640、无 NaN/Inf）。旧数据上是 255 帧主段 → 248 帧 clean set。下游一律用 `frame_audit.csv` 的 `is_sr_usable` 和 `acquisition_order` 列选帧，不要自行写筛选逻辑。

### 2.3 ⚠ 新探测器注意

`ep01.py` 的 `add_sr_input_selection_columns` 中矩阵有效性检查**硬编码 480×640**。新样本若探测器尺寸不同，该处（及 `configs/stage_calibration.json` 的 `detector_rows/cols`）需要同步修改，否则所有帧被判为 `invalid_matrix`。

### 2.4 人工检查清单

逐一查看 `session_summary.csv`（温度段数量/帧数是否符合采集记录）、`boundary_jump_table.csv`（跨段跳变是否远超噪声底，旧数据中位 3.55°C ≈ 49×）、`missing_coordinate_table.csv` 与 `raster_row_summary.csv`（坐标缺失和 raster 行序是否符合预期；`coordinate_set.json` 的 `known_missing_r0` 是旧数据专属，新样本要清空或更新）。

---

## 3. 必须重新标定的物理常数（新样机 / 新样本）

| 常数 | 配置文件 | 标定入口 | 回写方式 |
|---|---|---|---|
| 旋转角 θ | `configs/stage_calibration.json` (`theta_deg`) | EP02 流程（见下） | **手工改 config** |
| detector pitch | `configs/stage_calibration.json` (`pixel_size_um`) | `scripts/measure_pixel_size.py` | **手工改 config** |
| 噪声底 | `configs/noise_floor.json` | `scripts/audit_real_noise.py`（见下） | **手工改 config** |
| PSF σ | `configs/psf_calibration.json` | `algos/ep09_psf_calibration/` | `summarize_calibration.py` **自动回写** |

### 3.1 旋转角 θ —— 第一天必须标定！

旧项目前 6 个 Episode 因位移模型用错 θ 而全部报废。当前 θ=47.6° 是**旧样机的标定值，对新样机无效**。本仓库没有一键 θ 标定脚本（流程断点，见 §7），可用的诊断工具：

```bash
uv run python scripts/recompute_ep02_displacement_tables.py  # 无 CLI 参数；TXT 位移诊断表（依赖 EP01 cache）
uv run python scripts/avi_theta_estimation.py                # 无 CLI 参数；AVI 运动方向辅助验证（若新数据有 AVI）
```

注意：AVI gradient 估计只是方向性辅助验证（旧数据 θ≈47.14°，95% CI 覆盖 47.6°，但 X/Y 存在约 3° 系统差），**不能直接当标定源写入 config**。新样机应结合 stage 命令方向与图像位移响应重新拟合 θ（可参考 `thermal_core/displacement.py` 的 `fit_rotation_angle` / `bootstrap_theta_ci`），确认后手工更新 `stage_calibration.json`。

### 3.2 detector pitch

```bash
uv run python scripts/measure_pixel_size.py \
  --bmp data/data_raw/infrared_avi/<某帧>.bmp \
  --txt data/data_raw/infrared_avi/<同名>.txt \
  --output-dir output/ep03_theoretical_limits
```

从 BMP 坐标轴刻度测 mm/pixel，并用 Otsu 外轮廓交叉验证；结果写 `pixel_size_measurement.json/png`。注意区分 **采样 pitch**（此脚本测的）与 **空间分辨率**（需光学标定）；旧项目曾因 BMP 标尺 2× 误读把 pitch 记成 10 µm。确认后手工更新 `stage_calibration.json` 的 `pixel_size_um`。

### 3.3 噪声底

当前 0.0724°C 来自旧项目 Ep004 的 smooth adjacent-coordinate MAE，**没有本仓库脚本自动重算并回写**（流程断点，见 §7）。可用 `scripts/audit_real_noise.py`（平坦区噪声族审计：FPN、时间噪声、坏点；产物在 `output/real_noise_audit/`，参数见脚本 `--help`）获得新数据噪声画像，再按同方法学估计 noise floor 后手工更新 `configs/noise_floor.json`。

### 3.4 PSF σ

EP09 使用根 UV 环境（无独立 venv）：

```bash
uv run python algos/ep09_psf_calibration/scripts/run_forward_residual.py   # Route A（主估计）
uv run python algos/ep09_psf_calibration/scripts/run_esf_fitting.py        # Route B（交叉验证）
uv run python algos/ep09_psf_calibration/scripts/run_joint_estimation.py   # Route C（交叉验证）
uv run python algos/ep09_psf_calibration/scripts/summarize_calibration.py  # 汇总 + 自动回写 configs/psf_calibration.json
```

σ 单位是 LR 探测器像素（除非字段名带 `hr_px_at_2x`）。EP09 依赖 EP01/EP04/EP06 产物链，需先完成前置 cache。

---

## 4. 可复用 vs 必须重跑

| 组件 | 可复用性 |
|---|---|
| `core/`（IO、审计、坐标模型、绘图） | ✅ 直接复用；仅 §2.3 的探测器尺寸硬编码和 config 需按新样机调整 |
| `scripts/`、`notebooks/*/fragments/` | ✅ 直接复用（notebook 用 `build_all_notebooks.py --execute` 重建） |
| `tcforge/` 合成引擎 | ✅ 代码复用；但合成参数（噪声族、PSF σ、位移先验）必须按新标定值重新配置后**重新生成训练池** |
| `configs/*.json` | ⚠ 结构复用、数值作废：θ、pitch、噪声底、PSF σ、坐标集、`known_missing_r0` 都要按 §3 重标 |
| `output/` 全部 cache | ❌ 必须重跑（`build_all_caches.py`） |
| 训练好的 champion 权重 | ❌ checkpoints（`*.pt`）不入 git。同机可从 `algos/ep07_unet_sr/outputs/` 或备份拷贝；新机器/新物理常数下应**重新生成训练池并重训**，因为合成 forward model 绑定旧标定值 |

---

## 5. 跑通 pipeline 的最小路径

前提：§1 重命名完成、§3 物理常数已重标。

```bash
# ① 根环境 + 全部 EP cache（EP01→EP10 依赖序；也可 --only ep01,ep02 分步跑）
uv sync && uv pip install -e core/
uv run python scripts/build_all_caches.py            # 支持 --only/--force/--skip-missing

# ② 经典基线：EP10 drizzle 2x（最快拿到 SR 结果的路径）
cd algos/ep10_drizzle && uv sync
uv run python scripts/run_drizzle.py                 # 默认 --scale 2 --alignment-method contour_refined
#   常用参数：--psf-sigma（改为新标定值）--workers N --limit N（调试限帧）；全量见 --help
#   产物：output/ep10_drizzle/

# ③ 学习型主线：EP07 UNet solver（需要 GPU；依赖 tcforge 训练池）
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json --num-scenes 5 \
  --output-dir /tmp/smoke_pool --workers 1          # 先 smoke；正式池调大 num-scenes/workers
cd algos/ep07_unet_sr && uv sync
uv run python -m unet_sr.train --training-pool-dir <池目录> \
  --output-dir outputs/run1 --scale 2 --total-steps 40000 --device cuda --amp
#   完整推荐参数见 algos/ep07_unet_sr/scripts/run_training.md；推理用 unet_sr.inference API（见其 README）
```

验收以 contour-level 可视化增益 + alignment 质量门控（EP04）为准，不以单一锐度指标为准。SR 算法任何改动都要在 `research_log/algorithm_changelog.md` 记 ACL 条目。

---

## 6. 硬教训 checklist（新数据必读）

接入新样本时逐条自查（完整版见 `AGENTS.md`「硬教训」节）：

- [ ] **旋转角 θ 第一天标定**——旧项目前 6 个 EP 因 θ 错误全部报废；θ 不确认，一切位移先验都是错的。
- [ ] **session 检测必须用 acquisition_order（mtime）**，绝不按文件名字母序；拷贝数据必须保留 mtime。
- [ ] **跨 session 帧绝不混合重建**——旧数据跨段温度跳变中位 3.55°C ≈ 49× 噪声底。
- [ ] **R≠0 重复帧剔除出 SR 输入**——重复测量差异 1–2°C std，只作诊断用。
- [ ] **stage command 只能作 prior/初始化/正则**，不能当对齐真值；实际 alignment 必须由数据约束（EP04 anchor/gate）。
- [ ] **Y-only 坐标相邻帧不做定量位移标定**——raster 路径使其 acquisition gap ≈ 16 帧，热场演化污染 NCC。
- [ ] **AVI 只作方向性辅助验证**，渲染视频无温度矩阵、67% 重复帧，不能作 SR 输入或标定源。
- [ ] **锐度指标（Tenengrad 等）与回投残差都不能单独作为 SR 成功证据**——验收看 contour-level 结构一致性 + forward model + 对齐质量三者绑定。

---

## 7. 已知流程断点汇总

1. `rename_data.py --report` 输出路径（`reports/`）与 EP01 cache 读取路径（`output/ep01_data_processing/`）不一致，需手工拷贝（§1.3）。
2. θ 无一键标定脚本：现有 EP02 工具偏诊断/验证，正式标定需人工拟合并手工改 config（§3.1）。
3. 噪声底无自动重算回写脚本，`audit_real_noise.py` 只做噪声族审计（§3.3）。
4. `ep01.py` 硬编码 480×640 矩阵校验，新探测器尺寸需改代码（§2.3）。
5. detector pitch 测量结果需手工写回 `stage_calibration.json`，无自动回写（§3.2）。
