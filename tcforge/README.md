# TCForge — ThermalChipPhantom 合成训练池生成器

Synthetic LWIR micro-scan data engine：从参数化芯片几何生成带 HR ground truth
的温度场，经物理前向模型（PSF → 亚像素位移 → 探测器 box 积分 → 噪声/漂移/缺陷）
渲染成 248 帧 LR burst，打包为紧凑训练场景。

## 角色定位

**共享合成数据引擎（独立 UV 包）**。所有神经 SR 主线（EP07 2x UNet、EP12 4x、
EP13/EP14 loss atlas）的训练池、OOD 评测池与合成基准（Stage 2b harness）都由
TCForge 生成；同时提供 drizzle / shift-and-add 等经典重建与 split-half、
boundary-F1 等评估指标，供训练侧与评测侧复用。它不接触真实数据，
真实 248 帧 clean set 的物理常数（20 µm pitch、噪声底、shift 分布）作为
生成参数输入。

## 目录构成

```
tcforge/
├── pyproject.toml        # UV 包定义（hatchling 构建；dev extra = pytest）
├── uv.lock
├── src/tcforge/
│   ├── geometry.py       # 芯片结构二值 mask 生成器（pad/via/trace-bus/trench/多边形等）
│   ├── composer_v7.py    # panel_cluster_v7 场景合成器（v7+ 池的内容几何；ACL-065）
│   ├── physics.py        # 温度渲染、噪声、edge map、漂移模型、PSF 参数采样
│   ├── realism.py        # 真实感增强：探测器缺陷、等温温度、场噪声
│   ├── forward.py        # LR burst 前向模型（physical block-average；FORWARD_MODES）
│   ├── shifts.py         # shift profile 加载/生成（真实 248 帧位移分布、相位格）
│   ├── storage.py        # 紧凑场景存取（COMPACT_SCENE_FILES 契约）
│   ├── fusion.py         # LR burst → 1x 观测特征（obs_features_1x）
│   ├── classical_sr.py   # drizzle / shift-and-add / phase-bin drizzle 经典重建
│   ├── reconstruct.py    # 从紧凑场景参数重建 HR 温度目标
│   ├── evaluate.py       # 指标：PSNR/NRMSE/boundary-F1/split-half 一致性等
│   ├── highpass.py       # 与 EP06 对齐的 highpass 预处理
│   ├── manifest.py       # SceneManifest 数据集清单
│   ├── visualization.py  # 轻量绘图（懒加载 matplotlib）
│   └── _ep06_reference/  # EP06 前向约定参考实现（对照用）
└── tests/                # 14 个模块级 pytest（含 tests/data 固定样例）
```

## 环境安装

作为独立包开发/测试：

```bash
cd tcforge
uv sync --extra dev   # dev extra 含 pytest
```

运行依赖仅 numpy、scipy、matplotlib。两种被消费方式：

- **algo 子项目**：`algos/ep11_dl_benchmark`、`algos/ep12_4x_sr`、
  `algos/ep12_4x_benchmark` 等已在 `[tool.uv.sources]` 中声明
  `tcforge = { path = "../../tcforge", editable = true }`，`uv sync` 自动装入。
- **根目录脚本**：`scripts/generate_training_pool.py` 等直接把 `tcforge/src`
  插入 `sys.path`，在根 UV 环境 `uv run` 即可，无需安装。

## 运行方法

单元测试：

```bash
cd tcforge
uv sync --extra dev && uv run pytest
```

### 生成训练池（根目录执行，主入口）

`scripts/generate_training_pool.py` 读取 `configs/synthetic/` 下的池配置 JSON，
输出目录由配置的 `output_dir` 字段决定（惯例 `data/synthetic/<pool_name>/`）。
CLI 参数（已核实自 argparse 定义）：`--config`（默认
`configs/synthetic/training_pool_4x.json`）、`--pool-size`、`--num-scenes`、
`--output-dir`（覆盖配置值）、`--workers`（场景级进程数，默认 1）、
`--burst-workers`、`--seed`、`--lr-shape ROWS,COLS`。

```bash
# 4x 主训练池（默认配置）
uv run python scripts/generate_training_pool.py --workers 8

# v9 2x 池 pilot
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/pool_2x_v9_pilot.json --workers 8

# 小规模冒烟：只生成前 N 个场景
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/pool_2x_v8_cpu.json --num-scenes 4 --workers 2
```

### 配套根目录脚本（均在根 UV 环境 `uv run` 执行）

| 脚本 | 用途 | 已核实参数 |
|------|------|-----------|
| `scripts/precompute_drizzle_variants.py` | 池侧预计算 drizzle 变体（ACL-018） | `--pool-dir --num-variants --keep-frac-min --keep-frac-max --min-frames --shift-noise-std-px --seed --workers --overwrite` |
| `scripts/evaluate_thermal_chip_phantom.py` | 池级评估（指标 + highpass 抽查） | `--dataset-root --output-dir --highpass-check-frames` |
| `scripts/audit_generated_pool.py` | 抽样审计已生成池 | `--pool --k --out` |
| `scripts/forward_roundtrip_selfcheck.py` | 前向算子往返自检 | `--output-dir --scale --smoke` |
| `scripts/visualize_scene_samples.py`、`scripts/v7_composer_demo.py`、`scripts/v7_content_demo.py`、`scripts/audit_v7_tcforge_gates.py`、`scripts/audit_defect_detectability.py` | v7 内容/缺陷体系演示与门控审计 | 见各脚本头部 |

## 关键输出

每个池输出到 `data/synthetic/<pool_name>/`（Git 忽略）：

- 池根：`manifest.csv`（scene_id / difficulty / seed / scale / PSF / 噪声 / 漂移等字段）
- 每场景目录（紧凑契约 `COMPACT_SCENE_FILES`）：`hr_mask_4x.png`、
  `hr_edge_4x.png`、`obs_features_1x.npz`、`shifts.npy`、`metadata.json`
- 可选伴生文件（按配置生成）：`lr_burst.npy`（EP12 burst augmentation 必需）、
  `classical_sr_{scale}x.npy`、phase-bin drizzle、`defect_instances_2x.npz`、
  HR 温度场

## 相关文档

- 池配置：`configs/synthetic/`
  - 主训练池：`training_pool_2x.json`、`training_pool_2x_burst.json`、`training_pool_4x.json`
  - 代际池：`pool_2x_v3.json` → `pool_2x_v4_defects.json` → `pool_2x_v5_sharp.json` → `pool_2x_v6_*.json` → `pool_2x_v7*.json` → `pool_2x_v8_*.json` → `pool_2x_v9*.json`
  - OOD 评测池：`pool_2x_ood_content*.json`、`pool_2x_ood_noise_*.json`
  - 位移分布：`shift_profiles.json`
- Episode 记录：`research_log/episodes/ep07_thermal_chip_phantom/README.md`
- 正式报告：`paper/reports/ep07_thermal_chip_phantom/`
- 算法变更日志：`research_log/algorithm_changelog.md` — **ACL-008**（抗锯齿覆盖率渲染）、**ACL-018**（池侧 drizzle 变体预计算）、**ACL-023**（探测器 pitch 重标定 20 µm + v3 信息保存管线）、**ACL-030**（v5 GT edge_sigma 1.4→0.8）、**ACL-043**（on-the-fly phase-bin drizzle）、**ACL-045**（TCForge 默认 pitch 修正）、**ACL-065**（composer_v7 + 缺陷/噪声体系升级）、**ACL-068/070**（v7 池缺陷分布归因与 v8 池修复）、**ACL-072/074**（v9 代与 3K 池裁决）
- 主要消费者：`algos/ep07_unet_sr/`、`algos/ep12_4x_sr/`、`algos/ep11_dl_benchmark/`、`algos/ep12_4x_benchmark/`；EP13/EP14 loss atlas 的 tcforge demo 缓存经根脚本 `scripts/build_ep13_tcforge_demo.py` / `scripts/build_ep14_tcforge_demo.py` 生成
