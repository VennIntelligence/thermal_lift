# EP08 — INR-based 2x Contour SR（实验分支）

## 角色定位

实验分支：验证隐式神经表示（SIREN / WIRE）、CNN 深度 prior（Deep Decoder）与 DeepInverse ConvDecoder-DIP 能否在 248 clean 帧上带来可复现的 2x contour-level 增益，并与 EP06 MAP-TV 在同一 forward model / 指标框架下对照。按串行门控推进（P0 forward 等价性 → 单方法训练 → 多方对照 → Stage 3/4 扩量），不是交付主线（主线是 EP07 solver）。

## 目录构成

| 路径 | 职责 |
|---|---|
| `src/ep08/forward.py` | PyTorch forward operator（与 EP06 NumPy 版逐参数等价性验证过，P0 门控）|
| `src/ep08/data.py` / `splits.py` | 248 clean 帧加载与保持位移相位覆盖的 hold-out 切分 |
| `src/ep08/highpass.py` | highpass 预处理（匹配 EP06 `mode="nearest"` 约定）|
| `src/ep08/stage1.py` | Stage 1 训练公共 runner + 公共 CLI（`parse_training_args` / `run_stage1_training`）|
| `src/ep08/trainer.py` | 训练循环（warmup、早停、split-half 一致性）|
| `src/ep08/models/` | `siren.py`、`wire.py`、`deep_decoder.py` |
| `src/ep08/deepinv_wrapper.py` | DeepInverse-DIP 封装 |
| `src/ep08/metrics.py` / `utils.py` | 指标与工具 |
| `configs/` | `siren.yaml` / `wire.yaml` / `deep_decoder.yaml` / `deepinv_dip.yaml` 及 `*_stage3.yaml`；`ep06_baseline_metrics.json` 为 EP06 基线登记 |
| `scripts/` | 训练 / 基线 / 基准 / 编排入口（见运行方法）|
| `tests/` | 单元与 smoke 测试 |

## 环境安装

独立 UV 项目（`[tool.uv] package = false`），依赖 torch、deepinv、`thermal-core`（editable `../../core`）：

```bash
cd algos/ep08_inr_sr
uv sync
uv run pytest -q   # pytest 已在依赖中声明
```

## 运行方法

所有脚本在本目录运行；配置以 YAML 为主，CLI 参数可覆盖（公共 CLI 见 `src/ep08/stage1.py` 的 `parse_training_args`）。

Stage 1 单方法训练（四个薄封装入口，共用同一套参数）：

```bash
# SIREN（synthetic 先行，real 需数据到位）
uv run python scripts/train_siren.py --config configs/siren.yaml \
  --data-mode real --device cuda:0 --max-iter 10000 --n-frames 32 --patch-size 256

# WIRE / Deep Decoder 同参数面
uv run python scripts/train_wire.py --config configs/wire.yaml --data-mode real --device cuda:0

# DeepInverse-DIP（独立入口，参数面相近）
uv run python scripts/train_deepinv_dip.py --config configs/deepinv_dip.yaml \
  --data-mode real --device cuda:0 --in-spatial 30,40
```

Stage 3 统一入口（位置参数选方法，含 `map_tv` 经典对照）：

```bash
uv run python scripts/train_stage3.py wire --config configs/wire_stage3.yaml \
  --device cuda:1 --n-frames 64 --patch-shape full
uv run python scripts/train_stage3.py map_tv --lambda-tv 0.001 --max-iter 100
```

Stage 4 多臂编排控制器（子命令 `status` / `tick`，详见远程 runbook）：

```bash
uv run python scripts/stage4_controller.py status --json
uv run python scripts/stage4_controller.py tick --dry-run
```

TCForge 合成基准（有 HR-GT 的横向对比）：

```bash
uv run python scripts/run_tcforge_benchmark.py --methods siren wire deep_decoder \
  --n-frames 32 --iterations 800 --device cuda:0
```

辅助脚本：`generate_ep06_patch_baseline.py` / `generate_ep06_stage3_baseline.py`（生成 EP06 MAP-TV 对照基线）、`build_stage2_comparison.py`（汇总 Stage 2 对照表）、`validate_p0.py`（P0 门控校验）、`eval_all.py`（收集已保存的 smoke summary）、`run_tcforge_sanity.py`（基准的兼容入口）。

## 关键输出

写到项目根 `output/ep08_inr_sr/`（不入 Git）：

- Stage 1：`siren_stage1/`、`wire_stage1/`、`deep_decoder_stage1/`（`src/ep08/stage1.py` 的 `default_output_dir` 约定）；DeepInverse-DIP 默认 `deepinv_dip_stage2/`（见 `configs/deepinv_dip.yaml`）
- Stage 3：`stage3/<run名>/`（如 `stage3/wire_064_full_preserve`）
- 合成基准：`tcforge_benchmark/`；基准场景缓存在 `data/synthetic/ep08_tcforge_benchmark/`

## 相关文档

- Episode 记录：`research_log/episodes/ep08_inr_sr/README.md`（目标、串行门控、真值边界、当前状态）
- Stage 4 远程编排 runbook：`research_log/episodes/ep08_inr_sr/stage4_remote_operation_guide.md`
- 正式报告：`paper/reports/ep08_inr_sr/initial_report.md`
- Notebook：`notebooks/ep08_inr_sr/`
- `research_log/algorithm_changelog.md`：EP08 无专属 ACL 编号（ACL 主线覆盖 EP07/EP12 等训练管线；EP08 进展记录在 episode README）
