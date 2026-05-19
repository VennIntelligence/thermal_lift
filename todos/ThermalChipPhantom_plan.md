# TCForge 工程化实施计划（修订版）

> 状态：实施计划 v2.0（修订版）
> 目标：为 TCForge 合成数据生成引擎提供完整、可操作的工程化实施路线图
> 优先级：P0 = 必须完成 / P1 = 应该完成 / P2 = 未来扩展

---

## 1. 决策：独立 UV 包

**TCForge 是一个完全独立的 Python 包**，不与 `algos/ep06_sr_poc/` 或根目录项目共享 venv。

每个算法子目录（`algos/ep06_sr_poc/`、`tcforge/`）各自拥有自己的 `pyproject.toml`、`.python-version` 和独立 venv。这与 AGENTS.md 中规定的「算法隔离规则」完全一致。

**好处**：
- TCForge 的依赖（numpy、scipy）始终受控，不会被根项目或 EP06 的依赖变化影响
- EP06 的 forward model 被作为「锁定的参考实现」复制进 TCForge 的 `tcforge/src/tcforge/_ep06_reference/` 中，作为受测试锁定的版本
- TCForge 可以在任何机器上独立 `uv sync && uv run` 运行，不依赖项目其他部分

**目录结构**：

```
thermal_lift/
├── tcforge/                         # 独立 UV 项目
│   ├── pyproject.toml               # 独立依赖声明
│   ├── .python-version             # Python 版本
│   ├── src/tcforge/
│   │   ├── __init__.py
│   │   ├── _ep06_reference/       # P0: EP06 forward 的锁定副本
│   │   ├── geometry.py             # P0
│   │   ├── physics.py              # P0
│   │   ├── highpass.py             # P0
│   │   ├── shifts.py               # P0
│   │   ├── manifest.py             # P0
│   │   ├── forward.py              # P1: physical_block_average
│   │   ├── evaluate.py             # P1
│   │   └── visualization.py        # P1
│   └── tests/
│       ├── test_geometry.py
│       ├── test_physics.py
│       ├── test_highpass.py
│       ├── test_forward.py
│       ├── test_drift.py
│       └── test_manifest.py
├── configs/synthetic/              # JSON 配置文件（项目级）
│   ├── phantom_smoke.json
│   ├── phantom_benchmark.json
│   └── shift_profiles.json
├── scripts/
│   ├── generate_thermal_chip_phantom.py  # P0
│   ├── evaluate_thermal_chip_phantom.py  # P1
│   └── smoke_test_thermal_chip_phantom.py # P0
├── data/synthetic/                # 合成数据产物（Git ignore）
│   └── thermal_chip_phantom/
└── output/thermal_chip_phantom/  # 报告产物（Git ignore）
```

---

## 2. pyproject.toml（TOML 格式）

**文件：`tcforge/pyproject.toml`** — 使用标准 TOML 格式，不是 JSON。

```toml
[project]
name = "tcforge"
version = "0.1.0"
description = "ThermalChipPhantom synthetic data generator for LWIR microscan SR"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 3. 系统现状

### 3.1 已完成（EP06 侧）

| 组件 | 位置 | 说明 |
|------|------|------|
| EP06 forward model | `algos/ep06_sr_poc/src/common/forward_model.py` | forward/adjoint，transpose 测试通过 |
| EP06 highpass | `algos/ep06_sr_poc/src/common/data_loader.py` 中的 `highpass_preprocess` | mode="nearest"，float32，2D/3D |
| IBP / SAA / MAP-TV | `algos/ep06_sr_poc/src/` | 含并行 |
| 单元测试 | `algos/ep06_sr_poc/tests/` | 11 passed（运行方式见 3.2） |

### 3.2 EP06 测试运行方式

EP06 的 `pyproject.toml` 中**没有声明 `pytest` 作为依赖**，因此直接 `uv run pytest -q` 会失败。

正确的复现命令（从 `algos/ep06_sr_poc/` 目录运行）：

```bash
cd algos/ep06_sr_poc
uv sync
uv run --with pytest pytest -q
```

当前结果：**11 passed**。后续如果修改 EP06 代码，必须用同样命令复现。

### 3.3 TCForge 需要实现的部分

| 组件 | 优先级 |
|------|--------|
| EP06 forward reference copy（锁定版） | P0 |
| 几何生成器 (`geometry.py`) | P0 |
| 温度场 + 噪声注入 (`physics.py`) | P0 |
| 高通预处理（逐字匹配 EP06） | P0 |
| 位移配置加载 (`shifts.py`) | P0 |
| 元数据 + Manifest (`manifest.py`) | P0 |
| 生成器 CLI | P0 |
| physical_block_average forward | P1 |
| 漂移模型（3 种） | P1 |
| 评估器 CLI | P1 |
| 健全性检查可视化 | P1 |
| 扩展指标（Chamfer/Hausdorff/ESF） | P1 |

---

## 4. EP06 Forward Reference 策略

### 4.1 为什么需要复制

`algos/ep06_sr_poc/src/common/forward_model.py` 不是作为包安装的——它的 `pyproject.toml` 中 `package = false`，所以无法通过 `import` 引用。

TCForge 采用 **bootstrap 策略**：将 EP06 forward 的核心实现复制到 `tcforge/src/tcforge/_ep06_reference/forward.py`，作为受测试锁定的参考版本。

### 4.2 实施步骤

1. 从 `algos/ep06_sr_poc/src/common/forward_model.py` 复制 `forward`、`adjoint`、`_sigma_hr`、`_sample_reference_to_lr`、`_scatter_lr_to_reference`、`ObservationOperator`、`build_observation_operator` 到 `tcforge/src/tcforge/_ep06_reference/forward.py`
2. 从 `algos/ep06_sr_poc/tests/test_forward_model.py` 复制全部测试到 `tcforge/tests/test_forward.py`，验证复制后的代码与原版行为完全一致
3. 后续 TCForge 的 `generate_lr_burst()` 内部调用 `_ep06_reference/forward.py` 的 `forward()`

**约束**：EP06 侧 forward 如果被修改，TCForge 侧必须同步更新参考副本并重新通过测试。这是有意为之——确保 EP06 forward 的任何变化都能被 TCForge 感知到。

### 4.3 绝对不能做的事情

```python
# ❌ 错误：algos/ep06_sr_poc 不是一个可 import 的包
from algos.ep06_sr_poc.src.common.forward_model import forward

# ❌ 错误：无法从项目根访问 algos 子目录的 src
from thermal_lift.algos.ep06_sr_poc.src.common.forward_model import forward
```

---

## 5. 高通预处理（精确复刻 EP06）

**文件：`tcforge/src/tcforge/highpass.py`**

必须逐字匹配 `algos/ep06_sr_poc/src/common/data_loader.py` 中的 `highpass_preprocess`：

```python
def highpass_preprocess(
    frames: np.ndarray,
    sigma_bg: float = 5.0,
    *,
    workers: int | None = None,
    n_jobs: int | None = None,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Subtract a Gaussian background from each frame to produce structure maps.

    EP06 约定：
    - 输入 dtype: float32（强制转换）
    - 2D 输入：返回 frame - gaussian_filter(frame, sigma=sigma_bg, mode=mode)
    - 3D 输入：对 axis=0 逐帧处理，sigma tuple=(0, sigma_bg, sigma_bg)
    - 输出 dtype: float32
    - 高斯边界模式: mode="nearest"（不是默认的 "reflect"）
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return arr - gaussian_filter(arr, sigma=sigma_bg, mode=mode)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")

    n_workers = min(_resolve_workers(workers, n_jobs), arr.shape[0])
    if n_workers == 1:
        bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
        return (arr - bg).astype(np.float32, copy=False)

    # 并行实现...
```

**关键参数（全部固定，不开放为可选）**：

| 参数 | 值 | 原因 |
|------|----|------|
| `mode` | `"nearest"` | EP06 默认值 |
| `dtype` | `float32` | EP06 强制转换 |
| 3D sigma | `(0, sigma_bg, sigma_bg)` | 时间轴不变，只在空间上滤波 |
| 返回 dtype | `float32` | EP06 一致 |

**TCForge 使用 `sigma_bg = 5.0` 作为默认值**（对应 `highpass_sigma_lr_px = 5.0`）。

---

## 6. 几何生成器 API（P0）

### 6.1 坐标系统约定

**所有几何操作在统一的 HR canvas 上进行**：

- Canvas 尺寸：`canvas_shape = (960, 1280)`（2x 情况），由 `lr_shape` 和 `scale` 决定
- 坐标原点：左上角为 `(0, 0)`
- 坐标轴：行 = Y（向下），列 = X（向右）
- 单位：输入参数以 µm 为单位，内部转换为 HR 像素
- 转换规则：`n_px = round(size_um / hr_pitch_um)`，`hr_pitch_um = pixel_size_um / scale`

### 6.2 核心接口

```python
def make_rectangle(
    cx_um: float,
    cy_um: float,
    w_um: float,
    h_um: float,
    *,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """
    生成矩形掩码，写入指定 canvas。

    参数
    ----
    cx_um, cy_um: 矩形中心在芯片坐标系下的位置（µm）
    w_um, h_um: 矩形宽和高（µm）
    canvas_shape: 目标 canvas 尺寸（行, 列）
    pixel_size_um: 探测器像素间距（µm）
    scale: 超分辨率倍率

    返回
    ----
    二值掩码，dtype=uint8，shape = canvas_shape
    mask == 1: 芯片材料（高温侧）
    mask == 0: 背景 / 空气 / 切口（低温侧）
    """

def make_frame(
    cx_um: float,
    cy_um: float,
    outer_w_um: float,
    outer_h_um: float,
    inner_w_um: float,
    inner_h_um: float,
    *,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """生成框架掩码（外矩形减去内矩形）。"""

def make_pin_array(
    n_pins: int,
    spacing_um: float,
    pin_w_um: float,
    pin_l_um: float,
    cx_um: float,
    cy_um: float,
    direction: str = "horizontal",
    *,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """生成引脚阵列掩码。direction = 'horizontal'（平行于 X 轴）或 'vertical'。"""

def make_cross(
    cx_um: float,
    cy_um: float,
    arm_w_um: float,
    arm_l_um: float,
    *,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """生成十字形掩码。"""

def make_trenches(
    cx_um: float,
    cy_um: float,
    width_um: float,
    n_trenches: int,
    spacing_um: float,
    direction: str = "horizontal",
    *,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """生成平行沟槽阵列掩码。"""

def make_l_shape(
    cx_um: float,
    cy_um: float,
    w1_um: float,
    h1_um: float,
    w2_um: float,
    h2_um: float,
    *,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """生成 L 形掩码（两个矩形的并集）。"""

def composite(
    *masks: np.ndarray,
    canvas_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    将多个掩码合并为一张图。

    合并规则：任意掩码为 1 的位置即为 1（OR 逻辑）。
    如果所有掩码 shape 不同，padding 到 canvas_shape（缺失区域填 0）。
    如果提供了 canvas_shape，结果 crop/pad 到该尺寸。

    参数
    ----
    masks: 任意数量的二值掩码数组
    canvas_shape: 可选，指定输出 canvas 尺寸

    返回
    ----
    合并后的掩码，dtype=uint8
    """

def rotate_mask(
    mask: np.ndarray,
    angle_deg: float,
    *,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    将掩码旋转指定角度。

    使用 scipy.ndimage.rotate（无 skimage 依赖）。
    旋转后做二值化（threshold = 0.5）。

    参数
    ----
    mask: 二值掩码
    angle_deg: 旋转角度（度），正数为逆时针
    order: 插值阶数，0 = 最近邻（推荐，保留二值性）
    mode: 边界填充模式，默认 constant（空白处填 0）
    cval: mode="constant" 时的填充值
    """
```

### 6.3 旋转的几何先验

```python
def build_scene_mask(
    difficulty: str,
    seed: int,
    *,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.0,
    canvas_shape: tuple[int, int] = (960, 1280),
    pixel_size_um: float = 10.0,
    scale: int = 2,
) -> np.ndarray:
    """
    根据难度级别构建一个完整的场景掩码。

    rotation_deg = rotation_deg_center + uniform(-rotation_jitter_deg, rotation_jitter_deg)
    rotation_jitter 的随机性由 seed 控制。

    旋转是几何先验，不是电动台命令作为对齐真值的证明。
    """
```

### 6.4 禁止事项

- **禁止使用 `skimage.transform.rotate`**：`scipy.ndimage.rotate` 可以完成所有旋转需求
- **禁止不指定 `canvas_shape`**：每个函数必须显式接受 canvas_shape，或在 `build_scene_mask` 中统一处理
- **禁止直接返回 float 掩码**：所有几何函数输出 dtype 必须是 `uint8`，值为 0 或 1

---

## 7. 数据规模

| 场景规模 | lr_burst_raw.npy | lr_burst_highpass.npy | 合计（单 track） |
|----------|-------------------|------------------------|-------------------|
| 1 个全幅场景（255 帧） | ~299 MiB | ~299 MiB | ~598 MiB |
| 5 个 smoke 场景（clean） | ~1.5 GiB | ~1.5 GiB | ~3 GiB |
| 5 个 smoke 场景（clean + 2 drift） | ~1.5 GiB | ~4.5 GiB | ~6 GiB |
| 60 个完整基准（clean） | ~18 GiB | ~18 GiB | ~36 GiB |
| 60 个完整基准（clean + 3 drift） | ~18 GiB | ~54 GiB | ~72 GiB |

**P0 smoke 的 5 个场景至少需要 ~3 GiB 磁盘空间**（clean + 1 drift track）。这还不包括 HR GT 文件（每个场景 ~10 MiB，60 个场景 ~600 MiB）。

### P0 存储策略

- Smoke 场景使用全幅（480×640），因为需要测试 EP06 forward 的边界效应
- 完整基准（60 场景）**必须**增加 patch/crop 选项：从全幅中随机 crop 256×256 或 128×128 的 ROI，大幅减少存储
- 使用 `np.save(..., compress=False)`（默认），不开启压缩（CPU 开销太大）
- 输出目录结构：每个场景一个子目录，LR burst 使用 memory-mapped file 或压缩 .npz 备用方案

### 完整基准的存储选项（未来）

| 方案 | 存储 | CPU | 说明 |
|------|------|-----|------|
| 全幅 .npy | ~72 GiB | 低 | 最简单 |
| Crop ROI .npy | ~2 GiB | 低 | 丢失边缘效应 |
| 按需生成（不存 LR burst） | ~0 GiB | 高 | Generator 作为服务 |
| 压缩 .npz | ~18 GiB | 高 | 折中方案 |

---



### 实施阶段划分

```
阶段 0 + 1
  - TCForge 包骨架（pyproject.toml、目录结构）
  - EP06 forward reference 复制
  - 几何生成器：矩形、框架、引脚阵列、十字、L、沟槽
  - build_scene_mask（难度分级 + 旋转）

阶段 2 + 3
  - 温度场渲染 + 噪声注入
  - 边缘图生成
  - 高通预处理（逐字匹配 EP06）
  - 位移配置加载 + shift profile JSON

阶段 4 + 5
  - Manifest dataclass + JSON/CSV 写入
  - 生成器 CLI 端到端
  - EP06 forward 等价性测试

Smoke 测试 + 修复
  - 生成 5 个 smoke 场景
  - 验证所有文件存在性
  - 验证数组形状
  - 验证无 NaN/Inf
  - 修复所有边界问题

漂移模型（阶段 6，P1）
  - scalar_offset / lowfreq / gain_offset 三种漂移
  - 漂移退化验证

P1 剩余工作
  - physical_block_average forward
  - 评估器 CLI
  - 健全性检查可视化
```

---

## 9. P0/P1 边界（修订）

### P0（必须完成，最小可用版本）

- EP06 forward reference copy + 测试锁定
- 几何生成器（7 个图元 + 旋转 + composite）
- 温度场渲染 + 噪声注入
- 高通预处理（逐字匹配 EP06：`mode="nearest"`, `float32`, 2D/3D）
- 位移配置加载（real_default_contour_refined + ideal_phase_grid）
- Manifest dataclass + JSON/CSV 写入
- 生成器 CLI
- Smoke test（5 个场景，全幅 clean track）
- **无漂移轨道**（drift 放 P1）

### P1（应该完成，完整功能版本）

- physical_block_average forward（与 exact_ep06_point 分开报告）
- 漂移模型（scalar_offset + lowfreq + gain_offset）
- 扩展评估指标（Chamfer、Hausdorff、boundary F1）
- 评估器 CLI
- 健全性检查可视化
- 完整基准（60 场景，含 patch/crop 存储策略）

### P2（未来扩展）

- 几何改进：圆角、亚像素粗糙度、缺失引脚
- 坏像素注入
- ESF 宽度、引脚间隙精度
- WIRE / DIP / Deep Decoder 基线

---

## 10. Smoke 测试验收标准（修订）

P0 smoke 测试的验收标准必须基于**可测量的、无歧义的判据**，而不是方法间的大小比较（后者在合成数据上也可能不稳定）。

### 10.1 结构性验证（无条件通过）

1. **所有 .npy 文件无 NaN/Inf**：`np.isfinite()` 对每个文件返回全 True
2. **形状正确**：
   - `hr_temperature_2x.npy` → shape = (960, 1280)
   - `hr_mask_2x.npy` → shape = (960, 1280)，dtype = uint8，值为 0 或 1
   - `hr_edge_map_2x.npy` → shape = (960, 1280)
   - `lr_burst_raw.npy` → shape = (N, 480, 640)，N ≥ 32
   - `lr_burst_highpass.npy` → shape = (N, 480, 640)
   - `shifts.npy` → shape = (N, 2)
3. **Manifest 完整性**：`manifest.csv` 中每个 scene_id 对应的 `scene_dir` 存在且包含所有必需文件
4. **元数据一致性**：`metadata.json` 中的 `lr_shape`、`hr_shape`、`scale` 与实际文件一致

### 10.2 Forward 一致性验证（无条件通过）

5. **点源符号测试**：在 `hr_temperature_2x.npy` 中放置一个孤立的非零像素，用 EP06 forward 生成 LR，验证采样位置与 EP06 约定一致（`shift=(+0.5, 0)` 使 LR(1,1) 采样 HR 参考坐标 (2,3)）

### 10.3 数值范围验证（无条件通过）

6. **温度场有界**：`T_min ≥ T_bg - low_freq_amplitude - 3σ_noise`，`T_max ≤ T_bg + delta_T + low_freq_amplitude + 3σ_noise`
7. **高通范围合理**：对 easy 场景，`lr_burst_highpass.npy` 的绝对值不超过 `delta_T_c + 3 × noise_sigma_c`
8. **位移单调性**（ideal_phase_grid）：如果使用理论相位网格，检查位移向量的 L2 范数在合理范围内

### 10.4 方法间相对关系（软验收，可警告）

9. **Easy clean PSNR 阈值**：easy 场景中，SAA 的 PSNR（对 hr_temperature_2x.npy）必须 ≥ 某个根据几何和噪声计算的下界（例如 ≥ 15 dB）。如果低于此值，说明 forward 存在问题。**注：不要求 SAA 一定 > bicubic，也不要求 IBP > SAA**——这些相对关系取决于 PSF、噪声、位移覆盖，是不稳定的。
10. **IBP 不明显劣化初值**：IBP 重建结果的 PSNR 不应比初始化图像（SAA）低超过 2 dB

### 10.5 漂移退化验证（P1，drift track）

11. **漂移退化可测量**（仅当生成了 drift track 时）：
    - `drift_scalar` track 的 PSNR 比 clean 低 ≥ `0.5 × drift_amplitude_c`
    - `drift_lowfreq` track 的 PSNR 比 clean 低 ≥ `0.3 × drift_amplitude_c`
    - 退化量必须与配置的 `drift_amplitude_c` 成正比

### 10.6 禁止的验收声明

以下验收标准**不在 P0 范围内**，不在 smoke 阶段强制要求：
- "SAA > bicubic"
- "IBP > SAA"
- "Chamfer distance < X"
- "边界 F1 > Y"

---

## 11. EP06 高通函数签名参考

```python
# 源文件：algos/ep06_sr_poc/src/common/data_loader.py，第 118-137 行
def highpass_preprocess(
    frames: np.ndarray,
    sigma_bg: float = 5.0,
    *,
    workers: int | None = None,
    n_jobs: int | None = None,
    mode: str = "nearest",
) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return arr - gaussian_filter(arr, sigma=sigma_bg, mode=mode)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")

    n_workers = min(_resolve_workers(workers, n_jobs), arr.shape[0])
    if n_workers == 1:
        bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
        return (arr - bg).astype(np.float32, copy=False)
```

TCForge 的 `highpass.py` 必须与上述实现逐字一致。测试用例：

```python
def test_highpass_matches_ep06():
    # 常数帧 → 零高通
    constant = np.full((10, 10), 21.0, dtype=np.float32)
    hp = highpass_preprocess(constant, sigma_bg=5.0)
    assert np.abs(hp).max() < 1e-5

    # 2D 和 3D 行为一致
    frame_2d = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    frame_3d = frame_2d[np.newaxis, ...]
    hp_2d = highpass_preprocess(frame_2d, sigma_bg=5.0)
    hp_3d = highpass_preprocess(frame_3d, sigma_bg=5.0)[0]
    assert np.allclose(hp_2d, hp_3d)

    # dtype = float32（不是 float64）
    result = highpass_preprocess(frame_2d, sigma_bg=5.0)
    assert result.dtype == np.float32
```

---

## 12. 已知风险

1. **Forward 模式必须严格分开报告**：`exact_ep06_point` 和 `physical_block_average` 必须分别报告指标，不能混用。

2. **stress 级别不能外推**：15 µm 以下特征即使 Chamfer 分数好，也必须标注为"应力测试"，不能声称达到对应分辨率。

3. **位移是 prior 不是 GT**：即使使用真实数据位移，它仍然是估计值，不是 ground truth。metadata 中必须显式记录 `"convention": "LR-to-reference alignment shift"`。

4. **Smoke test 先于外部展示**：在 smoke 通过之前，不得向外部展示任何 TCForge 生成的图像或指标。

5. **Seed 管理**：所有随机过程（几何 jitter、噪声、漂移）必须接受 `seed` 参数，记录在 metadata 中，确保可复现。

6. **EP06 forward 变更感知**：如果 EP06 侧的 forward 被修改，TCForge 侧的 `_ep06_reference/forward.py` 必须同步更新并重新通过测试。这是有意设计的耦合。

7. **数据规模规划**：P0 smoke 需要 ~3 GiB，完整基准需要 18-72 GiB。存储策略必须在实施前确认。

---

## 13. 配置参考

### phantom_smoke.json（TOML 格式）

注意：配置文件使用 JSON 格式（不是 TOML），因为项目根已有 JSON 配置约定，且 `json` 是 Python 标准库。

```json
{
  "dataset": "ThermalChipPhantom",
  "engine": "TCForge",
  "version": "0.1.0",
  "num_scenes": 5,
  "scale": 2,
  "lr_shape": [480, 640],
  "hr_shape": [960, 1280],
  "pixel_size_um": 10.0,
  "spatial_resolution_um": 20.0,
  "forward_mode": "exact_ep06_point",
  "psf_sigma_lr_px": 0.5,
  "noise_sigma_c": 0.0724,
  "highpass_sigma_lr_px": 5.0,
  "highpass_mode": "nearest",
  "T_bg_c": 21.0,
  "delta_T_c_by_difficulty": {
    "easy": 2.5,
    "medium": 1.5,
    "hard": 1.0,
    "stress": 0.7
  },
  "low_freq_amplitude_c": 0.2,
  "shift_profile": "real_default_contour_refined",
  "rotation_deg_center": 47.6,
  "rotation_jitter_deg": 1.0,
  "drift_model": "none",
  "difficulties": ["easy", "easy", "medium", "medium", "hard"],
  "seeds": [1001, 1002, 2001, 2002, 3001],
  "split": "test",
  "n_frames_per_scene": 255
}
```

### phantom_benchmark.json

```json
{
  "num_scenes": 60,
  "split": {
    "train": 30,
    "val": 10,
    "test": 20
  },
  "difficulty_distribution": {
    "easy": 15,
    "medium": 25,
    "hard": 15,
    "stress": 5
  },
  "drift_tracks": ["clean", "drift_scalar", "drift_lowfreq"],
  "forward_modes": ["exact_ep06_point"],
  "scales": [2],
  "storage_strategy": "crop_roi",
  "roi_size": [256, 256]
}
```

---s
