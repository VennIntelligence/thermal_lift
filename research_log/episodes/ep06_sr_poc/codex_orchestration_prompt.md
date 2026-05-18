# EP06 Codex Orchestration Prompt — 2x Contour-Level SR POC

> **本文件是给 Codex 主代理的完整任务提示词。**
> 主代理负责拆解任务、分派子代理、检查质量、迭代修正、最终整合。

---

## 🎯 总体目标

在 thermal_lift 项目的 255 帧主 session LWIR 温度矩阵上，实现 **2x contour-level 超分辨率重建 POC**。使用 5 种经典物理算法（含 baseline），通过双轨输出（highpass 结构图 + raw 温度控制轨）证明多帧微扫描对芯片内部结构/轮廓的增益。

**不是**温度计量 SR，**是**轮廓/结构可见性增强。

---

## 📋 前置条件（已完成，直接使用）

| 产物 | 路径 | 用途 |
|------|------|------|
| 帧审计表 | `output/ep01_data_processing/frame_audit.csv` | 主 session 255 帧筛选 |
| 对齐位移表 | `output/ep05_sr_reassessment/displacement_measurements.csv` | NCC init 位移 |
| 对齐方法对比 | `output/ep05_alignment_sr_capacity/alignment_method_holdout_scores.csv` | 方法选择依据 |
| Contour alignment | `output/ep05_contour_alignment/contour_alignment_results.csv` | 每帧 refined shift |
| Anchor catalog | `output/ep04_global_validation/segment_summary.csv` | 质量门控权重 |
| 物理参数 | `configs/stage_calibration.json`, `configs/noise_floor.json` | PSF σ, noise, θ |
| Core 库 | `core/src/thermal_core/` | IO、plotting、displacement 工具 |

**关键物理参数**:
- TXT 采样 pitch: 10 µm/pixel
- 当前空间分辨率: 20 µm
- 目标: 2x → 10 µm 等效分辨率（5 µm/pixel 输出网格）
- PSF: Gaussian σ ≈ 1.0 px（实测 ESF 等效宽度，含热边缘过渡）
- Noise floor: 0.0724°C
- 主 session: 255 帧
- 旋转角 θ: 47.6°

---

## 🏗️ 项目结构

```
algos/ep06_sr_poc/
├── pyproject.toml              ← 独立 UV 项目
├── .python-version
├── src/
│   ├── common/                 ← 共享基础设施
│   │   ├── __init__.py
│   │   ├── data_loader.py     ← 加载 255 帧 + highpass 预处理
│   │   ├── alignment.py       ← 从 EP05 产物加载对齐位移
│   │   ├── forward_model.py   ← H 矩阵（shift + PSF + downsample）
│   │   ├── metrics.py         ← 评价指标
│   │   └── visualization.py   ← 并排对照图
│   ├── saa/                    ← SAA-uniform + SAA-weighted
│   │   ├── __init__.py
│   │   └── saa.py
│   ├── ibp/                    ← Iterative Back-Projection
│   │   ├── __init__.py
│   │   └── ibp.py
│   └── map_tv/                 ← MAP with TV regularization
│       ├── __init__.py
│       └── map_tv.py
├── scripts/
│   ├── run_saa.py
│   ├── run_ibp.py
│   ├── run_map_tv.py
│   └── run_evaluation.py
└── tests/
    ├── test_forward_model.py   ← 合成数据端到端验证
    ├── test_saa.py
    ├── test_ibp.py
    └── test_map_tv.py
```

---

## 🔧 主代理工作流

### Step 0: 环境搭建

1. 创建 `algos/ep06_sr_poc/` 目录和 `pyproject.toml`
2. 依赖: numpy, scipy, matplotlib, pandas, tqdm, scikit-image
3. `pip install -e ../../core/` 安装 thermal_core
4. 验证能 import thermal_core 并读取数据

### Step 1: 共享基础设施（主代理或单独子代理）

创建 `src/common/` 下的所有模块：

**data_loader.py**:
```python
def load_main_session_frames(data_dir, frame_audit_path):
    """加载 255 帧主 session TXT 温度矩阵，返回 (N, H, W) ndarray。"""
    # 从 frame_audit.csv 筛选 is_main_session == True
    # 按 acquisition_order 排序
    # 返回 frames: (255, 480, 640), metadata: DataFrame

def highpass_preprocess(frames, sigma_bg=5.0):
    """对每帧减去 Gaussian 平滑背景，返回 highpass 结构图。

    sigma_bg: 背景平滑核的 sigma（像素），默认 5.0 px。
    选择依据：远大于 PSF σ≈1 px，保留边缘；远小于芯片尺度，去除漂移。
    """
    # frames_hp[i] = frames[i] - gaussian_filter(frames[i], sigma=sigma_bg)
    # 返回 (255, 480, 640) highpass 图

def offset_correction(frames):
    """Per-frame offset correction for raw-temperature track.

    对每帧减去其均值（或中位数），消除帧间温度偏移。
    """
```

**alignment.py**:
```python
def load_alignment_shifts(method='ncc_init'):
    """从 EP05 产物加载每帧的 (dx, dy) 亚像素位移。

    method: 'ncc_init' | 'filename_affine' | 'contour_refined'
    返回 shifts: (255, 2) ndarray，单位 pixel。
    """

def load_quality_weights():
    """从 EP04/EP05 产物加载每帧的质量权重。

    基于 held-out Chamfer、NCC peak 等指标。
    权重归一化到 [0, 1]，用于 SAA-weighted。
    """
```

**forward_model.py**:
```python
def build_observation_operator(hr_shape, lr_shape, shifts, psf_sigma=1.0):
    """构建 forward model: y_i = D * B * S_i * x + n

    S_i: 按 shifts[i] 平移
    B: Gaussian PSF 模糊 (sigma=psf_sigma)
    D: 2x 下采样

    不需要显式构建大矩阵。用函数式实现：
    forward(x, shift_i) -> y_i_predicted
    adjoint(y_i, shift_i) -> contribution to x
    """

def forward(x_hr, shift, psf_sigma=1.0):
    """HR image -> predicted LR observation for one frame."""
    # 1. shift x_hr by (dx, dy) using interpolation
    # 2. convolve with Gaussian PSF
    # 3. downsample 2x
    return y_predicted

def adjoint(y_residual, shift, psf_sigma=1.0, hr_shape=None):
    """LR residual -> back-projected contribution to HR grid."""
    # 1. upsample 2x (zero-insert or repeat)
    # 2. convolve with PSF transpose (same Gaussian)
    # 3. shift back by (-dx, -dy)
    return contribution
```

**metrics.py**:
```python
def gradient_magnitude(img):
    """计算图像梯度幅值，用于轮廓锐度评价。"""

def contour_chamfer(sr_img, reference_contour_points):
    """SR 图的边缘与 held-out contour 的 Chamfer distance。"""

def split_half_consistency(frames, shifts, sr_method, n_splits=10):
    """将帧随机分两半，各自重建，计算两个 SR 结果的一致性。"""

def artifact_score(sr_img, lr_img):
    """检测振铃、块状伪影等。"""
```

### Step 2: 分派子代理

**同时启动子代理 A、B、C**（它们互相独立）：

---

#### 子代理 A: SAA-uniform + SAA-weighted

**任务描述**:

实现 Shift-and-Add 超分辨率重建，包含两个变体。

**SAA-uniform 算法**:
```
1. 创建 2x HR 网格 (960, 1280)，初始化为 0
2. 创建权重网格，初始化为 0
3. 对每帧 i:
   a. 将帧 i 的每个像素按 shifts[i] 映射到 HR 网格的亚像素位置
   b. 用双线性插值将像素值分配到 HR 网格的 4 个最近邻
   c. 权重网格对应位置 +1（或按插值权重累加）
4. HR = 累加值 / 权重（避免除零）
```

**SAA-weighted 算法**:
```
与 uniform 相同，但步骤 3 中每帧的贡献乘以该帧的质量权重 w_i。
w_i 来自 load_quality_weights()。
```

**验收标准**:
- 合成数据测试：已知 HR 图 → 生成 LR 帧 → SAA 重建 → 与真值 PSNR > 25 dB
- 真实数据：输出 (960, 1280) 的 2x HR 图
- 两个变体的结果都要输出
- 运行时间 < 30 秒（255 帧）

**输出文件**:
- `output/ep06_sr_poc/saa_uniform_highpass.npy`
- `output/ep06_sr_poc/saa_weighted_highpass.npy`
- `output/ep06_sr_poc/saa_uniform_raw.npy`
- `output/ep06_sr_poc/saa_weighted_raw.npy`
- `output/ep06_sr_poc/saa_synthetic_validation.json`

---

#### 子代理 B: IBP (Iterative Back-Projection)

**任务描述**:

实现 Irani & Peleg (1991) 的 Iterative Back-Projection 超分辨率重建。

**IBP 算法**:
```
1. 初始化: x_hr = SAA 结果（或 bicubic 上采样的参考帧）
2. 迭代 k = 1, ..., max_iter:
   a. 对每帧 i:
      - y_pred_i = forward(x_hr, shifts[i], psf_sigma)  # 预测观测
      - residual_i = frames[i] - y_pred_i               # 残差
   b. 累加 back-projection:
      - correction = (1/N) * Σ_i adjoint(residual_i, shifts[i], psf_sigma)
   c. 更新: x_hr = x_hr + beta * correction
   d. 收敛检查: if ||correction|| / ||x_hr|| < tol: break
3. 返回 x_hr
```

**超参数**:
- `max_iter`: 50（默认），用 split-half 选早停
- `beta`: 0.5（步长，太大会振铃）
- `psf_sigma`: 1.0 px
- 收敛 tol: 1e-4

**验收标准**:
- 合成数据测试：已知 HR + PSF → LR 帧 → IBP 重建 → PSNR 比 SAA 高
- 真实数据：输出 (960, 1280) 的 2x HR 图
- 迭代收敛曲线要输出
- 运行时间 < 5 分钟（255 帧，50 iter）
- 不能出现明显振铃（用 artifact_score 检查）

**输出文件**:
- `output/ep06_sr_poc/ibp_highpass.npy`
- `output/ep06_sr_poc/ibp_raw.npy`
- `output/ep06_sr_poc/ibp_convergence.csv`
- `output/ep06_sr_poc/ibp_synthetic_validation.json`

---

#### 子代理 C: MAP-TV

**任务描述**:

实现 Maximum A Posteriori 超分辨率重建，使用 Total Variation 正则化。

**MAP-TV 算法**:

最小化目标函数:
```
J(x) = (1/2N) * Σ_i ||y_i - H_i(x)||² + λ * TV(x)
```

其中:
- `H_i(x) = D * B * S_i * x` 是 forward model
- `TV(x) = Σ_{j,k} sqrt((x[j+1,k]-x[j,k])² + (x[j,k+1]-x[j,k])² + ε²)` 是各向同性 TV
- `ε = 1e-6` 防止梯度为零
- `λ` 是正则化强度

**优化方法**: L-BFGS 或 FISTA (Fast Iterative Shrinkage-Thresholding)

推荐 FISTA 实现:
```
1. 初始化: x = SAA 结果, t = 1, z = x
2. 迭代 k = 1, ..., max_iter:
   a. 计算数据项梯度: grad_data = (1/N) * Σ_i H_i^T(H_i(x) - y_i)
   b. 梯度下降步: x_temp = z - step_size * grad_data
   c. TV proximal 步: x_new = prox_TV(x_temp, lambda * step_size)
      - prox_TV 用 Chambolle 投影算法（20-50 内迭代）
   d. FISTA 加速: t_new = (1 + sqrt(1 + 4*t²)) / 2
      z = x_new + (t-1)/t_new * (x_new - x)
   e. x = x_new, t = t_new
3. 返回 x
```

**超参数**:
- `lambda_tv`: 用 split-half consistency 选择。扫描范围 [1e-4, 1e-1]，对数等间距 10 个点
- `max_iter`: 200
- `psf_sigma`: 1.0 px
- `step_size`: 1 / L（L 是数据项的 Lipschitz 常数，≈ 1.0 对归一化数据）

**⚠️ 关键注意事项**:
- highpass 图是正负双边响应，TV 可能把边缘 lobes 变成块状结构
- λ 必须用 split-half 和 held-out contour 指标选，不能只看视觉锐度
- 如果 FISTA 太慢，可以先用 gradient descent + TV denoising 的交替方向法（ADMM）

**验收标准**:
- 合成数据测试：已知 HR + PSF → LR 帧 → MAP-TV 重建 → PSNR 比 IBP 高
- 真实数据：输出 (960, 1280) 的 2x HR 图
- λ 选择曲线（split-half vs λ）要输出
- 运行时间 < 30 分钟（255 帧，200 iter）
- 不能出现明显块状伪影

**输出文件**:
- `output/ep06_sr_poc/map_tv_highpass.npy`
- `output/ep06_sr_poc/map_tv_raw.npy`
- `output/ep06_sr_poc/map_tv_lambda_selection.csv`
- `output/ep06_sr_poc/map_tv_convergence.csv`
- `output/ep06_sr_poc/map_tv_synthetic_validation.json`

---

### Step 3: 主代理质量检查

子代理完成后，主代理依次检查：

1. **合成数据验证**:
   - SAA PSNR > 25 dB？
   - IBP PSNR > SAA？
   - MAP-TV PSNR > IBP？
   - 如果不满足，分析原因，要求子代理修正

2. **真实数据 sanity check**:
   - 输出尺寸是否正确 (960, 1280)？
   - 是否有 NaN/Inf？
   - 值域是否合理（highpass 应该是零均值、小幅值）？
   - 是否有明显伪影（振铃、块状、条纹）？

3. **对齐一致性**:
   - 用 EP05 的 held-out contour 检查 SR 结果的边缘是否与独立对齐一致
   - 如果 Chamfer > 0.3 px，说明对齐或重建有问题

4. **递进关系**:
   - SAA-weighted 是否比 SAA-uniform 好？（如果是，说明质量门控有效）
   - IBP 是否比 SAA 好？（如果是，说明 forward model 有用）
   - MAP-TV 是否比 IBP 好？（如果是，说明 TV 正则有用）
   - 如果递进关系不成立，需要诊断原因

**如果检查不通过**：
- 定位问题（对齐错误？PSF 参数错？λ 太大/太小？）
- 要求对应子代理修正
- 重新运行评估

### Step 4: 分派子代理 D — 评估与 Notebook 整合

**子代理 D 任务**:

1. **运行统一评估脚本** `scripts/run_evaluation.py`:
   - 加载所有 SR 结果 + LR 参考帧 + bicubic 2x
   - 计算所有指标（gradient magnitude、Chamfer、split-half、artifact score）
   - 输出 `output/ep06_sr_poc/evaluation_summary.csv`

2. **生成并排对照图**:
   - 全图概览：6 列（LR / bicubic / SAA-u / SAA-w / IBP / MAP-TV）
   - 3-4 个 ROI 放大：选 anchor 好的区域 + anchor 差的区域 + 内部结构区域
   - 双轨对照：highpass 主轨 vs raw 控制轨
   - 输出到 `output/ep06_sr_poc/comparison_*.png`

3. **创建 Notebook** `notebooks/ep06_sr_poc/fragments/`:
   - `01_setup.py`: 环境声明、路径、导入
   - `02_algorithm_overview.py`: 算法原理说明（Markdown 为主，含公式）
   - `03_synthetic_validation.py`: 合成数据验证结果
   - `04_main_results.py`: 真实数据并排对照（主轨 highpass）
   - `05_control_track.py`: raw 温度控制轨对照
   - `06_quantitative_evaluation.py`: 指标表格和分布图
   - `07_roi_analysis.py`: ROI 放大对比
   - `08_conclusions.py`: 结论和下一步

4. **创建正式报告** `reports/ep06_sr_poc/sr_poc_report.md`:
   - 算法描述
   - 实验设置
   - 定量结果表
   - 关键可视化
   - 结论和 EP07 建议

---

## 📐 技术细节补充

### Highpass 预处理规范

```python
from scipy.ndimage import gaussian_filter

def highpass_preprocess(frame, sigma_bg=5.0):
    """
    sigma_bg=5.0 的选择依据:
    - 远大于 PSF σ≈1 px → 不会损失边缘信息
    - 远小于芯片结构尺度 (~50-100 px) → 保留内部结构
    - 有效去除帧间温度漂移（低频）

    输出是零均值的结构图，正值=比背景热，负值=比背景冷。
    """
    background = gaussian_filter(frame, sigma=sigma_bg)
    return frame - background
```

### 对齐位移加载

优先使用 `data_driven_ncc_init`（EP05 产物），因为：
- 相位分布连续均匀（4 bins 各 62-65 帧）
- held-out Chamfer 中位 0.16 px
- 不依赖坐标模型假设

备选 `filename_affine_fit`（held-out Chamfer 中位 0.17 px，相位分布也均匀）。

### 质量权重计算

```python
def compute_quality_weights(chamfer_scores, ncc_peaks):
    """
    基于 EP05 contour alignment 的 held-out Chamfer 和 NCC peak。

    w_i = ncc_peak_i * (1 / (chamfer_i + eps))
    归一化到 sum(w) = N
    """
```

### 合成数据验证规范

```python
def generate_synthetic_test():
    """
    1. 创建已知 HR 图 (960, 1280)：含清晰边缘、内部结构
    2. 生成 255 个随机亚像素位移（模拟真实分布）
    3. 对每个位移：shift → PSF blur → downsample → add noise
    4. 得到 255 帧 LR (480, 640)
    5. 用各算法重建，与真值比较

    这是端到端验证，确保代码链路正确。
    """
```

---

## 🔴 硬约束

1. **绝不声称 4x 或 5 µm 分辨率** — 本 EP 只做 2x，输出网格 5 µm/pixel 但实际分辨率需要 MTF 验证
2. **highpass SR 输出是结构图，不是温度 SR 图** — 不能声称恢复了绝对温度
3. **λ 选择必须用 split-half，不能只看视觉** — 防止过拟合
4. **SAA-weighted vs uniform 的对比是必须的** — 回答"增益来自算法还是质量门控"
5. **控制轨必须输出** — 防止 highpass 制造伪结构的质疑
6. **合成验证必须先通过** — 真实数据结果才有意义
7. **遵循项目绘图规范** — CVPR 风格、Times New Roman、300 dpi
8. **遵循 Notebook 管理规范** — fragments/ 模式，不直接编辑 .ipynb

---

## 📊 预期产物清单

```
output/ep06_sr_poc/
├── saa_uniform_highpass.npy
├── saa_uniform_raw.npy
├── saa_weighted_highpass.npy
├── saa_weighted_raw.npy
├── ibp_highpass.npy
├── ibp_raw.npy
├── map_tv_highpass.npy
├── map_tv_raw.npy
├── bicubic_reference.npy
├── lr_reference.npy
├── saa_synthetic_validation.json
├── ibp_synthetic_validation.json
├── ibp_convergence.csv
├── map_tv_synthetic_validation.json
├── map_tv_lambda_selection.csv
├── map_tv_convergence.csv
├── evaluation_summary.csv
├── comparison_fullview.png
├── comparison_roi_1.png
├── comparison_roi_2.png
├── comparison_roi_3.png
├── comparison_control_track.png
├── gradient_magnitude_comparison.png
├── split_half_consistency.png
└── artifact_audit.png
```

---

## ⏱️ 时间预算

| 阶段 | 预计时间 |
|------|----------|
| Step 0-1: 环境 + 共享基础设施 | 15 min |
| Step 2A: SAA | 20 min |
| Step 2B: IBP | 30 min |
| Step 2C: MAP-TV | 45 min |
| Step 3: 质量检查 + 迭代 | 30 min |
| Step 4: 评估 + Notebook | 30 min |
| **总计** | **~3 小时** |

---

## 🔄 迭代策略

如果某个子代理的结果不通过质量检查：

1. **对齐问题** → 检查位移加载是否正确（符号、单位、参考帧）
2. **PSF 问题** → 尝试 σ = 0.5, 0.8, 1.0, 1.2 的网格搜索
3. **IBP 振铃** → 降低 beta、加早停、或加轻微 Gaussian 平滑
4. **MAP-TV 块状** → 降低 λ、或换用各向异性 TV
5. **SAA 模糊** → 检查位移精度是否足够（如果 SAA 和 bicubic 差不多，说明位移信息不够）
6. **递进关系不成立** → 这本身是重要发现，记录并分析原因

**最多迭代 3 轮**。如果 3 轮后仍不通过，记录失败原因和当前最佳结果，作为 EP06 的诚实结论。

---

## 📝 最终交付检查清单

- [ ] 5 种方法的 2x HR 输出（双轨 × 5 = 10 个 .npy）
- [ ] 合成数据验证全部通过
- [ ] 并排对照图（全图 + 3 个 ROI）
- [ ] 定量评估表（gradient mag、Chamfer、split-half、artifact）
- [ ] λ 选择曲线（MAP-TV）
- [ ] 收敛曲线（IBP、MAP-TV）
- [ ] Notebook 构建并执行成功
- [ ] 正式报告完成
- [ ] EP06 README 更新为 ✅
- [ ] research_log/README.md 路线图更新
