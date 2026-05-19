# ThermalChipPhantom / TCForge 计划

> 状态：计划文档
> 基准数据集名称：**ThermalChipPhantom**
> 生成引擎名称：**TCForge**
> 目标：为芯片级 LWIR 多帧微扫描超分辨率提供受控的合成基准数据集和可选训练数据源。

---

## 1. 核心决策

我们将工作拆分为两个命名产物：

| 产物 | 角色 | 位置 |
|---|---|---|
| **TCForge** | 确定性程序化生成引擎 | 源代码位于 `tcforge/src/tcforge/` |
| **ThermalChipPhantom** | 生成的基准数据集和清单文件 | 生成产物位于 `data/synthetic/`，报告汇总位于 `output/` |

代码不能放在 `data/` 下，因为 `data/` 被 Git 忽略。生成的场景、缓存的帧序列、patch 分片和大型数组应放在 `data/synthetic/` 或 `output/` 下，取决于它们是可复用数据还是报告产物。

推荐的仓库布局：

```text
tcforge/
├── pyproject.toml               # 独立 UV 项目配置
├── src/tcforge/
│   ├── __init__.py
│   ├── geometry.py              # 芯片几何图元和掩码
│   ├── physics.py               # 热场、噪声、漂移、高通辅助函数
│   ├── forward.py               # 合成正向模型模式
│   ├── manifest.py              # 元数据 schema、清单写入、哈希
│   ├── evaluate.py              # 基于 GT 的合成指标
│   └── visualization.py         # 紧凑的健全性检查图
├── tests/                       # 单元测试
└── README.md

configs/synthetic/
├── phantom_smoke.json
├── phantom_benchmark.json
├── phantom_patches.json
└── shift_profiles.json

scripts/
├── generate_thermal_chip_phantom.py
├── evaluate_thermal_chip_phantom.py
└── smoke_test_thermal_chip_phantom.py

data/synthetic/
├── thermal_chip_phantom/
│   ├── manifest.csv
│   ├── dataset_metadata.json
│   └── scenes/
└── tcforge_cache/

output/thermal_chip_phantom/
├── smoke_test_report.md
├── baseline_metrics.csv
└── figures/
```

首选 JSON 配置格式。除非我们显式添加 `pyyaml` 到根项目依赖，否则不引入 YAML。

---

## 2. 为什么需要这个数据集

真实主 session 有 255 帧 LWIR 温度矩阵，但没有高分辨率 ground truth。因此，真实数据的验证只能依赖正向一致性、split-half 稳定性、FRC/边缘 MTF 代理指标以及视觉轮廓检查。ThermalChipPhantom 填补的是另一个角色：

1. 在已知物理退化链下提供高分辨率 ground truth，用于方法排名；
2. 通过 PSNR、SSIM、Chamfer 距离、边界 F1、拓扑和 ESF 宽度来量化边界恢复能力；
3. 在花费时间运行真实数据之前，用于调整 WIRE/DIP/Deep Decoder 超参数；
4. 可选地生成 patch 数据，供未来窄域去噪器或扩散先验使用。

该数据集不能单独用于声称真实芯片的分辨率。它是一个受控的诊断环境，而非真实样品的光学 ground truth。

---

## 3. 物理常数和单位

使用已确认的项目常数：

| 参数 | 值 | 备注 |
|---|---:|---|
| LR 探测器网格 | `480 × 640` | 行 × 列 |
| 探测器间距 | `10 µm / LR 像素` | 采样间距，非空间分辨率 |
| 当前空间分辨率 | `20 µm` | 校准分辨率 |
| 主 session 帧数 | `255` | 仅 session 2 |
| 噪声底 | `0.0724°C` | 高斯默认噪声 sigma |
| 电动台-像素旋转角 | `47.6°` | 先验 / 几何方向，非对齐真值 |
| 默认 PSF | `psf_sigma_lr_px = 0.5` | 内部转换为 HR sigma：`sigma_hr = sigma_lr × scale` |

所有几何生成参数应先以物理微米（µm）存储，再转换为 HR 像素。这可以避免混淆 2x HR 像素尺寸与真实光学分辨率。

重要用语：

- `5 µm` 是 2x HR 网格间距，不是分辨率声明。
- `10 µm` 特征低于当前 20 µm 分辨率，应标记为应力测试用例。
- `20 µm` 特征接近当前分辨率极限。
- `30–80 µm` 特征是更安全的轮廓级用例。

---

## 4. 正向模型模式

TCForge 应从第一天起支持两种正向模型模式。

### 4.1 `exact_ep06_point`

该模式尽可能精确地复现当前 EP06 的无矩阵正向约定：

```text
x_hr
  -> 高斯模糊，sigma = psf_sigma_lr_px × scale
  -> 使用 shift=(dx, dy) 采样探测器中心坐标
  -> 添加噪声 / 漂移 / 坏像素
  -> 可选的 LR 高通
```

位移约定必须与 EP05/EP06 匹配：

- `shift=(dx, dy)` 以 LR 像素为单位。
- 它是将观测 LR 帧移动到参考坐标系的位移。
- EP06 的 `forward()` 通过在参考 HR 场景上按探测器位置加上该对齐位移进行采样来预测原始观测。

该模式是与现有 SAA/IBP/MAP-TV 代码进行公平 smoke 测试所必需的。

### 4.2 `physical_block_average`

该模式以更物理的方式建模探测器像素积分：

```text
x_hr
  -> 在参考/观测坐标约定下进行亚像素位移
  -> 高斯模糊
  -> 在 scale × scale 的 HR 像素上进行块平均
  -> 添加噪声 / 漂移 / 坏像素
  -> 可选的 LR 高通
```

该模式可用于测试模型失配以及后续的 PyTorch 正向建模工作。它必须与 `exact_ep06_point` 分开报告；绝不混合指标而不标注模式。

### 4.3 倍率策略

初始实现应支持：

| 倍率 | 状态 | 用途 |
|---:|---|---|
| `2` | P0 | EP06 兼容基准 |
| `4` | P1 | EP08 研究 / 先验辅助可视化 |

EP06 代码目前拒绝 2 以外的倍率。对于 4x，TCForge 可以生成 GT 和 LR 帧序列，但 EP06 经典求解器无法直接复用。

---

## 5. 高通和原始数据轨道

高通必须在与真实处理相同的概念位置应用：

```text
HR 原始温度场
  -> 正向模型
  -> LR 原始帧序列
  -> 添加噪声 / 漂移 / 坏像素
  -> LR 高通帧序列
```

保存原始和高通两个轨道：

| 文件 | 含义 |
|---|---|
| `hr_mask_{scale}x.npy` | 二值材料掩码 GT |
| `hr_temperature_{scale}x.npy` | 原始 HR 热场 GT |
| `hr_edge_map_{scale}x.npy` | 从掩码派生的边缘 GT |
| `lr_burst_raw.npy` | 高通前的模拟 LR 观测 |
| `lr_burst_highpass.npy` | 高通后的模拟 LR 结构图 |
| `shifts.npy` | 使用的精确位移，`(N, 2)` 列为 `[dx, dy]` |

`hr_highpass_{scale}x.npy` 仅可作为辅助可视化目标保存。除非指标明确说明是高通域指标，否则不应将其视为主要物理 GT。

---

## 6. 几何生成器

在芯片坐标系中生成锐利的材料掩码，然后旋转到探测器坐标系。

P0 图元：

| 图元 | 参数 |
|---|---|
| 矩形 | 中心、宽度、高度 |
| 框架 / 边框 | 外矩形减去内矩形 |
| 引脚 / 槽 | 窄矩形 |
| 引脚阵列 | 数量、间距、宽度、长度、方向 |
| L 形 | 两个矩形的并集 |
| T / 十字形 | 多个矩形的并集 |
| 平行沟槽 | 重复的窄切口 |

P1 改进：

| 特性 | 目的 |
|---|---|
| 圆角半径 | 避免不切实际的完美阶梯角 |
| 低于 1 HR 像素的边缘粗糙度 | 受控的真实感应力测试 |
| 缺失 / 断裂引脚 | 拓扑测试 |
| 可变发射率对比度 | 未来物理真实感 |

掩码语义：

- `mask = 1`：芯片 / 材料 / 前景。
- `mask = 0`：背景、空气、基底或切口。
- `temperature = T_bg + delta_T × mask + low_freq_background`。

旋转：

```text
rotation_deg = 47.6 + jitter
默认 jitter = uniform(-1.0, 1.0) deg
```

旋转是几何先验，不是电动台命令为对齐真值的证明。

---

## 7. 难度分级

使用四个分级而非三个，以便明确标注低于分辨率的特征。

| 分级 | 最小特征 | 对比度 | 噪声 | 用途 |
|---|---:|---:|---:|---|
| `easy` | `40–80 µm` | `2.0–3.0°C` | `0.0724°C` | 健全性检查和方法调试 |
| `medium` | `20–40 µm` | `1.0–2.0°C` | `0.0724°C` | 接近项目 POC 目标 |
| `hard` | `15–25 µm` | `0.8–1.5°C` | `0.0724°C` | 接近/低于光学极限的应力测试 |
| `stress` | `10–20 µm` | `0.5–1.0°C` | `0.0724–0.15°C` | 幻觉和不确定性测试 |

不要将 `stress` 级别的恢复结果呈现为真实数据的承诺。它是故意设计为困难的，可能在物理上不可支持。

---

## 8. 漂移和鲁棒性

决策：包含漂移，但将其与主要干净基准分开。

### 8.1 数据集轨道

| 轨道 | 漂移 | 角色 |
|---|---|---|
| `clean` | 关闭 | 匹配正向模型下的核心方法排名 |
| `drift_scalar` | 全局逐帧偏移 | 偏移校正和高通鲁棒性 |
| `drift_lowfreq` | 空间低频帧漂移 | 高通 / 背景去除应力测试 |
| `drift_gain_offset` | 增益加偏移 | 原始控制鲁棒性 |

### 8.2 漂移模型

在正向模型之后、高通之前施加漂移：

```text
lr_raw_k = A_k(x_hr) + noise_k + drift_k
lr_highpass_k = highpass(lr_raw_k)
```

P0 漂移模型：

1. `scalar_offset_random_walk`
   - 每帧一个标量偏移；
   - 按采集顺序的随机游走或低通滤波高斯序列；
   - 默认振幅 `0.02–0.15°C`。

2. `spatial_lowfreq_gaussian`
   - 每帧一个低频场；
   - 生成高斯噪声然后用 `sigma=40–120 LR 像素` 进行模糊；
   - 默认振幅 `0.02–0.20°C`。

3. `gain_plus_offset`
   - `y_k = gain_k × y_k + offset_k`；
   - 默认增益范围 `1 ± 0.002` 到 `1 ± 0.01`。

这直接测试了我们的高通和偏移校正是否行为正确。它也改善了项目叙事，因为我们可以展示干净结果和鲁棒性结果的对比。

---

## 9. 位移配置

P0 位移来源：

| 配置 | 来源 | 用途 |
|---|---|---|
| `real_default_contour_refined` | `output/ep05_contour_alignment/contour_alignment_results.csv` 精化列 | 主要基准 |
| `real_ncc_init` | 同一 CSV 的初始化列 | 相位先验对照 |
| `real_tuned_contour_refined` | `output/ep05_alignment_tuning/full_r360_e93_rad100_s0125/contour_alignment_results.csv` | 灵敏度测试 |

P1 位移配置：

| 配置 | 用途 |
|---|---|
| `stage_prior_affine` | 先验 / 对照，非真值 |
| `jittered_real` | 对齐扰动的鲁棒性 |
| `ideal_phase_grid` | 相位覆盖的理论上界 |

每个生成的场景必须保存 `shifts.npy` 并记录：

- 配置名称；
- 来源路径；
- 来源文件哈希；
- 使用的列；
- 位移单位；
- 位移约定。

---

## 10. 元数据 Schema

每个场景应包含：

```json
{
  "schema_version": "0.1",
  "dataset": "ThermalChipPhantom",
  "engine": "TCForge",
  "scene_id": "tcp_medium_0001",
  "seed": 1001,
  "split": "test",
  "difficulty": "medium",
  "scale": 2,
  "lr_shape": [480, 640],
  "hr_shape": [960, 1280],
  "pixel_size_um": 10.0,
  "spatial_resolution_um": 20.0,
  "geometry": {
    "units": "um",
    "rotation_deg": 47.3,
    "min_feature_um": 20.0,
    "primitives": []
  },
  "physics": {
    "T_bg_c": 21.0,
    "delta_T_c": 1.5,
    "low_freq_background_c": 0.2,
    "psf_sigma_lr_px": 0.5,
    "noise_sigma_c": 0.0724,
    "forward_mode": "exact_ep06_point",
    "highpass_sigma_lr_px": 5.0,
    "drift_model": "none"
  },
  "shifts": {
    "profile": "real_default_contour_refined",
    "source_path": "output/ep05_contour_alignment/contour_alignment_results.csv",
    "source_sha256": "...",
    "columns": ["refined_align_dx_px", "refined_align_dy_px"],
    "units": "LR pixels",
    "convention": "LR-to-reference alignment shift"
  },
  "provenance": {
    "generator_git_sha": "...",
    "created_at_utc": "...",
    "config_path": "configs/synthetic/phantom_benchmark.json"
  }
}
```

数据集级别的 `manifest.csv` 应至少包含：

```text
scene_id, split, difficulty, scale, seed, forward_mode, drift_model,
min_feature_um, delta_T_c, psf_sigma_lr_px, noise_sigma_c,
shift_profile, scene_dir, metadata_sha256
```

---

## 11. 评估指标

仅限合成数据的指标：

| 指标 | 域 | 方向 | 备注 |
|---|---|---|---|
| PSNR | 原始或高通 | 越高越好 | 显式报告数据范围 |
| SSIM | 原始或高通 | 越高越好 | 如依赖可用则可选；否则推迟 |
| NRMSE | 原始或高通 | 越低越好 | 高通域按 GT 标准差归一化 |
| 边界 F1 | 边缘 GT | 越高越好 | 阈值以 HR 像素和 µm 为单位 |
| Chamfer 距离 | 边缘 GT | 越低越好 | 主要轮廓定位指标 |
| Hausdorff 距离 | 边缘 GT | 越低越好 | 敏感的最坏情况指标 |
| ESF 宽度 | 原始 GT 边界 | 无伪影时越低越好 | 必须与稳定性配对使用 |
| 引脚间隙精度 | 几何 GT | 误差越低越好 | 仅用于引脚阵列 |
| 拓扑 F1 | 连通分量 | 越高越好 | 捕获合并/拆分的结构 |

鲁棒性指标：

- 干净 vs 漂移的指标退化；
- 32/64/128/255 帧子集稳定性；
- 噪声种子稳定性；
- PSF 灵敏度，`sigma=0.3/0.5/0.7/1.0`；
- 对齐扰动灵敏度；
- 针对优化合成帧序列的方法，保留帧重投影残差。

不要仅凭梯度/Tenengrad 对方法排名。合成 GT 支持更好的轮廓指标；请使用它们。

---

## 12. 基线方法

P0 基线：

1. LR 单帧参考；
2. 双三次插值显示；
3. SAA 均匀加权；
4. SAA 质量加权；
5. IBP；
6. MAP-TV。

P1 / EP07 方法：

1. WIRE 式隐式神经表示（INR）；
2. DIP；
3. Deep Decoder。

P2 / EP08 方法：

1. 小型单通道去噪器；
2. DiffPIR / DAPS + TCForge 先验；
3. Flower / PnP-Flow，仅在可用域先验之后。

初始 smoke 测试应仅在 5 个场景上运行 P0 基线。

---

## 13. 实现阶段

### S0：锁定规格

- 创建本计划。
- 确认 TCForge 名称和源码/产物拆分。
- 将待定决策转换为 JSON 配置默认值。

### S1：几何和元数据

- 实现矩形、切口、引脚阵列、L/T/十字形、平行沟槽。
- 以 µm 存储几何数据。
- 确定性转换为 HR 网格。
- 保存 `metadata.json` 和 `manifest.csv`。

### S2：物理和轨道

- 构建原始 HR 温度场。
- 添加低频背景。
- 从掩码生成边缘图。
- 实现高通辅助函数，匹配现有 `highpass_preprocess` 的默认语义。

### S3：正向模型

- 实现 `exact_ep06_point`。
- 添加测试，验证在多个随机场景和位移下与当前 EP06 `forward()` 的等价性。
- 将 `physical_block_average` 作为标注的辅助模式实现。
- 添加点源位移符号测试。

### S4：噪声、漂移和坏像素

- 添加固定种子的高斯噪声。
- 添加标量漂移和低频空间漂移。
- 仅在 clean/drift 轨道稳定后添加可选的坏像素掩码。

### S5：生成器 CLI

示例命令：

```bash
uv run python scripts/generate_thermal_chip_phantom.py \
  --config configs/synthetic/phantom_smoke.json \
  --output-root data/synthetic/thermal_chip_phantom_smoke
```

CLI 应支持：

- `--num-scenes`；
- `--difficulty`；
- `--scale`；
- `--forward-mode`；
- `--shift-profile`；
- `--drift-model`；
- `--seed`；
- `--overwrite` 需显式指定。

### S6：评估 CLI

示例命令：

```bash
uv run python scripts/evaluate_thermal_chip_phantom.py \
  --dataset-root data/synthetic/thermal_chip_phantom_smoke \
  --result-root output/thermal_chip_phantom/smoke
```

评估器应接受 `.npy` 格式的方法输出，并写入：

- `baseline_metrics.csv`；
- `boundary_metrics.csv`；
- `robustness_metrics.csv`；
- 紧凑的健全性检查图。

### S7：Smoke 测试

生成 5 个场景：

- 2 个 easy；
- 2 个 medium；
- 1 个 hard；
- 倍率 2；
- `exact_ep06_point`；
- 先无漂移，再复制一份低频漂移版本。

运行 SAA/IBP/MAP-TV 并验证：

- 数组值有限；
- 形状正确；
- 位移约定未反转；
- easy 场景产生合理的指标排序；
- 漂移轨道对原始数据的退化大于高通/偏移校正轨道。

### S8：完整基准

推荐的首个完整基准：

| 分割 | 场景数 | 备注 |
|---|---:|---|
| train | 30 | 仅用于去噪器/先验开发 |
| val | 10 | 超参数调优 |
| test | 20 | 最终报告 |

对于经典方法和测试时方法，不需要 train 分割。对于去噪器/扩散工作，patch 生成应仅从 train 场景中采样。

---

## 14. 必需测试

最低单元测试要求：

1. 确定性生成：相同种子和配置产生相同的哈希值；
2. 几何边界：掩码有限、二值、且在预期面积范围内；
3. 特征尺度：生成的最小特征（像素）与请求的 µm 在容差内匹配；
4. 位移符号：点源测试匹配 EP06 约定；
5. 正向等价性：`exact_ep06_point` 在 2x 倍率下与 EP06 NumPy forward 匹配；
6. PSF 单位：`psf_sigma_lr_px=0.5` 在 2x 下变为 `1.0 HR 像素`；
7. 高通：常数图像的高通约为零；
8. 漂移：漂移在高通之前施加，对原始数据的影响大于高通；
9. 清单完整性：清单中列出的每个场景都存在且哈希匹配。

---

## 15. 报告规则

报告 ThermalChipPhantom 结果时：

- 必须说明 `forward_mode`。
- 必须说明轨道是原始还是高通。
- 必须说明是否启用漂移。
- 必须说明最小特征尺寸（µm）及其与 20 µm 空间分辨率的关系。
- 不要将 4x 输出描述为已验证的 5 µm 分辨率。
- `stress` 级别用例必须明确标注为应力/幻觉测试。
- 区分合成 GT 指标和真实数据正向一致性。

推荐措辞：

> TCForge 生成 ThermalChipPhantom——一个程序化的 LWIR 芯片仿体基准数据集，与项目的微扫描几何和退化链相匹配。它用于在受控的 HR ground truth 下对算法排名，而真实数据的声明仍需通过保留帧重投影、split-half 稳定性和轮廓一致性来把关。

---

## 16. 验收标准

TCForge v0.1 在以下条件满足时视为通过：

1. 源代码位于被忽略的数据目录之外；
2. 可以从配置确定性地重新生成 smoke 数据集，哈希匹配；
3. `exact_ep06_point` 通过与 EP06 正向模型的等价性测试；
4. 生成的场景包含原始 HR、掩码 GT、边缘 GT、LR 原始帧序列、LR 高通帧序列、位移、元数据和清单；
5. 漂移可以独立切换，并在元数据中记录；
6. SAA/IBP/MAP-TV smoke 基线在 5 个场景上端到端运行通过；
7. 至少写入边界 F1、Chamfer、NRMSE 和 PSNR 到 CSV；
8. 默认情况下，生成的 `.npy`、`.csv` 或图表产物不会被暂存到 Git。

---

## 17. 默认初始配置

除非后续实验有意更改，否则使用以下默认值：

```json
{
  "dataset": "ThermalChipPhantom",
  "engine": "TCForge",
  "scale": 2,
  "lr_shape": [480, 640],
  "forward_mode": "exact_ep06_point",
  "psf_sigma_lr_px": 0.5,
  "noise_sigma_c": 0.0724,
  "highpass_sigma_lr_px": 5.0,
  "shift_profile": "real_default_contour_refined",
  "rotation_deg_center": 47.6,
  "rotation_jitter_deg": 1.0,
  "drift_tracks": ["clean", "drift_scalar", "drift_lowfreq"],
  "difficulties": ["easy", "medium", "hard", "stress"]
}
```

---

## 18. 待决决策

当前推荐默认值列于此处；仅在有明确理由时更改。

| 问题 | 默认值 |
|---|---|
| 引擎名称 | TCForge |
| 基准数据集名称 | ThermalChipPhantom |
| 代码位置 | `tcforge/src/tcforge/` |
| 生成数据位置 | `data/synthetic/` |
| 配置格式 | JSON |
| 主要正向模式 | `exact_ep06_point` |
| 辅助正向模式 | `physical_block_average` |
| 主要倍率 | 2x |
| 4x 状态 | P1 研究 / 仅应力测试 |
| 漂移 | 作为鲁棒性轨道包含，非干净基准的主要部分 |
| 低于 20 µm 的特征 | 仅应力测试用例 |
