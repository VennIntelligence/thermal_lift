# Windows 4x 合成训练池生成指南

面向在 Windows 主机上生成 **v8 抗锯齿 TCForge 4x 训练池**（仅生成，不含 EP12 训练）。Linux 同样适用。

## 前提

- [Git](https://git-scm.com/download/win)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)（Windows 安装器或 `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`）
- 磁盘：**≥ 250 GB 可用**（1000 scenes 含 `lr_burst.npy` 约 190 GB，再加 drizzle 特征）
- 内存：4x + v8 SSAA 约 **1.3–1.5 GB/worker**；建议 `--workers 1–2` 起步

## 1. 克隆与初始化

```powershell
git clone <repo-url> thermal_lift
cd thermal_lift
uv sync
```

生成脚本通过 `sys.path` 加载 `tcforge/src`，**不需要**单独安装 tcforge venv。

默认位移 CSV 已纳入 Git：

- 配置：`configs/alignment/paths.json`
- 数据：`configs/alignment/contour_alignment_results.csv`

无需再拷贝 `output/ep05_contour_alignment/`。如需临时覆盖，可设环境变量：

```powershell
$env:TCFORGE_REAL_SHIFT_CSV = "D:\path\to\custom.csv"
```

## 2. Smoke 测试（2 场景）

```powershell
uv run python scripts/generate_training_pool.py `
  --config configs/synthetic/training_pool_4x.json `
  --output-dir data/synthetic/training_pool_4x_aa_smoke `
  --num-scenes 2 `
  --workers 1
```

检查 `data/synthetic/training_pool_4x_aa_smoke/scene_0000/metadata.json`：

| 字段 | 期望 |
|------|------|
| `fallback_used` | `false`（真实位移已加载） |
| `mask_semantics` | `"coverage"`（v8 抗锯齿） |
| 文件 | `lr_burst.npy`, `shifts.npy`, `hr_mask_4x.png`, `obs_features_1x.npz` |

## 3. 全量 1000 场景

```powershell
uv run python scripts/generate_training_pool.py `
  --config configs/synthetic/training_pool_4x.json `
  --output-dir data/synthetic/training_pool_4x_aa `
  --pool-size 1000 `
  --workers 2
```

- 支持断点续跑：已有 `metadata.json` 的 scene 会自动跳过
- v8 AA 默认开启（`antialias=True`），无需改 JSON
- 失败 scene 写入 `failed_scenes.log`

## 4. 构建 EP12 drizzle 特征（可选，训练前需要）

若后续在 Linux 上训练 EP12，需先构建 deferred features：

```powershell
uv run python scripts/build_4x_features.py `
  --pool-dir data/synthetic/training_pool_4x_aa `
  --workers 2
```

每 scene 额外产出：`obs_features_4x.npz`、`obs_features_2x_up4x.npz`、`obs_features_1x_up4x.npz`。

## 5. 产物回传

`data/synthetic/training_pool_4x_aa/` 整目录拷贝到训练机（robocopy、rsync、共享盘等）。该目录在 `.gitignore` 中，**不要**尝试 git push 数据。

## 常见问题

**Q: 能否在 Windows 上训练 EP12？**  
本指南只覆盖生成。训练需 CUDA PyTorch，与生成环境独立。

**Q: `fallback_used: true` 是什么意思？**  
未找到位移 CSV 时会降级到 `ideal_phase_grid`。确认已 `git pull` 且存在 `configs/alignment/contour_alignment_results.csv`。

**Q: PowerShell 反引号报错？**  
也可写成单行：

```powershell
uv run python scripts/generate_training_pool.py --config configs/synthetic/training_pool_4x.json --output-dir data/synthetic/training_pool_4x_aa --pool-size 1000 --workers 2
```

**Q: 如何验证 tcforge 单测？**

```powershell
cd tcforge
uv sync
uv run pytest tests/test_shifts.py -q
```
