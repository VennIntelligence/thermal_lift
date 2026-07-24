# EP10 Drizzle CPU SR

CPU 2x micro-scan reconstruction using STScI `drizzle`.

## 角色定位

经典基线：用天文界标准的 STScI drizzle（像素重投影 + 权重累积）做 2x 微扫描重建，是 EP10 方法对比（Drizzle / MAP-TV / TGV）三条经典对照之一，也持续作为 champion 神经模型评测中的独立经典参照（如 cross-FRC 中的 drizzle split）。

## 目录构成

| 路径 | 职责 |
|---|---|
| `src/ep10_drizzle/drizzle_sr.py` | 核心重建逻辑（pixmap 构建、drizzle 调用、split-half / holdout / artifact 指标）|
| `scripts/run_drizzle.py` | 唯一 CLI 入口：加载 248 clean 帧 → highpass → 多 pixfrac drizzle → 指标与图 |
| `tests/test_drizzle_sr.py` | 单元测试 |

## Environment（环境安装）

独立 UV 项目，`thermal-core` 为 editable 路径依赖；pytest 已在依赖中声明：

```bash
cd algos/ep10_drizzle
uv sync
uv run pytest -q
```

The experiment imports EP06 shared modules from `../ep06_sr_poc/src/common`（通过 pytest `pythonpath` 与脚本内 sys.path 注入，无需安装 EP06）。

## API Notes

`drizzle.resample.Drizzle` 2.2.0 exposes `pixfrac` on `Drizzle.add_image`, not on the constructor:

```python
Drizzle(kernel="square", out_shape=(960, 1280), disable_ctx=True)
driz.add_image(data, exptime=1.0, pixmap=pixmap, pixfrac=0.7, in_units="cps")
```

`out_img` is the weighted mean output image. `out_wht` is the accumulated output weight/count image and is used here as the coverage map.

## Run（运行方法）

```bash
cd algos/ep10_drizzle

# 默认：2x，全部 248 clean 帧，contour_refined 对齐
uv run python scripts/run_drizzle.py

# 调试：限制帧数 + 指定 pixfrac 网格
uv run python scripts/run_drizzle.py --limit 32 --pixfracs 0.6 0.7 0.8 --workers 4

# 探索性 4x（仅轮廓过采样可视化，不声明 2.5 um 物理分辨率）
uv run python scripts/run_drizzle.py --scale 4 --output-dir ../../output/ep10_drizzle_4x
```

其余参数：`--alignment-method`（默认 `contour_refined`）、`--highpass-sigma 5.0`、`--psf-sigma 0.5`、`--unsharp-sigma` / `--unsharp-amount`（输出端 unsharp）、`--coverage-threshold`、`--n-splits 10`（split-half 次数）、`--random-state`。

For an auditable detached run with a persistent log（示例中 `/home/ujs/...` 是远程 Linux 计算机路径，本机请替换为实际项目根）:

```bash
cd /home/ujs/mycode/thermal_lift
setsid bash -c 'echo $$ > output/ep10_drizzle/full_run.pid; trap "rm -f output/ep10_drizzle/full_run.pid" EXIT; exec algos/ep10_drizzle/.venv/bin/python -u algos/ep10_drizzle/scripts/run_drizzle.py --output-dir output/ep10_drizzle' \
  > output/ep10_drizzle/full_run.log 2>&1 < /dev/null &
```

## 关键输出

写到项目根 `output/ep10_drizzle/`（4x 探索走 `output/ep10_drizzle_4x/`，不入 Git）：各 pixfrac 的 HR highpass 重建 `.npy`、coverage map、split-half / holdout / artifact 指标 CSV 与对比图。

`artifact_score` is written without the LR overshoot component so it is
comparable with MAP-TV and TGV. The legacy overshoot-inclusive value is kept as
`artifact_score_with_lr_overshoot` for debugging only.

## 相关文档

- Episode 记录：`research_log/episodes/ep10_method_comparison/README.md`（三方法对比范围与质量修正记录）
- 跨方法对比 Notebook：`notebooks/ep10_method_comparison/`
- 兄弟对照项目：`algos/ep10_map_tv_sweep/`、`algos/ep10_tgv_sr/`
- `research_log/algorithm_changelog.md`：无专属 ACL 编号（经典基线，进展记录在 ep10 episode README）
