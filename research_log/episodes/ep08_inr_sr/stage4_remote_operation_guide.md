# EP08 Stage 4 Remote Operation Guide

> 用途：给只能执行终端命令的远程 Codex 使用。远程端不需要理解 notebook 或代码结构，只需要周期性调用 controller，并在 blocked 时把状态和日志带回来。

## 1. Controller 概要

新增脚本：

```bash
/home/ujs/mycode/thermal_lift/algos/ep08_inr_sr/scripts/stage4_controller.py
```

它只做编排和健康检查，不改训练逻辑：

- 计划 Stage 4 progressive 训练：64 -> 128 -> 255 frames。
- 每个阶段默认包含 4 个方法：EP06 MAP-TV、SIREN、WIRE、DeepInv-DIP。
- 默认资源分配：`MAP-TV=cpu`，`SIREN/WIRE=cuda:1` 串行，`DeepInv-DIP=cuda:0`。
- 每次 `tick` 只启动当前阶段中可以安全启动的 pending 任务，然后立即退出。
- 训练在后台进程中运行，日志写入 `output/ep08_inr_sr/stage4_controller/logs/`。
- 状态快照写入 `output/ep08_inr_sr/stage4_controller/status.json`。

Controller 不会从 checkpoint resume；底层 Stage 3 脚本也没有 resume 语义。失败后需要人工归档旧输出目录和旧 launch record，再重跑。

## 2. 远程端每小时调用的命令

先只推进 64 帧健康验证：

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py tick --max-frame 64
```

如果 64 帧通过数值健康检查，并且本机或人工完成视觉检查后，再允许推进完整链路：

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py tick
```

如果你决定让远程端全自动按数值门控推进，可以从一开始就使用第二条命令。这个模式不会做人眼视觉判断，因此最终结论仍要回到 notebook 图像。

## 3. 安全预检

只打印计划，不启动训练：

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py plan --max-frame 64
uv run python scripts/stage4_controller.py tick --max-frame 64 --dry-run
```

查看状态：

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py status --max-frame 64 || true
uv run python scripts/stage4_controller.py status --json --max-frame 64 > /tmp/ep08_stage4_status.json || true
```

`status` 或 `tick` 在 blocked 时会返回非零码，这是预期行为；远程端应把表格、`status.json` 和相关 log 带回来，而不是继续启动新任务。

## 4. 默认启动策略

首个 64 帧 `tick` 会启动：

```text
ep06_map_tv_064_full_preserve  -> cpu
siren_064_full_preserve        -> cuda:1
deepinv_dip_064_full_preserve  -> cuda:0
```

`wire_064_full_preserve` 会等 `siren_064_full_preserve` 完成并通过数值门控后再启动，因为二者默认共享 `cuda:1`。

每个 run 的输出目录：

```text
output/ep08_inr_sr/stage3/{method}_{frames:03d}_full_{preserve|stretch}
```

例如：

```text
output/ep08_inr_sr/stage3/siren_064_full_preserve
output/ep08_inr_sr/stage3/wire_128_full_preserve
output/ep08_inr_sr/stage3/deepinv_dip_255_full_preserve
output/ep08_inr_sr/stage3/ep06_map_tv_255_full_preserve
```

## 5. Controller 数值门控

Controller 用 `metrics.json` 做机器可判定的健康门控：

- `stage_gate` 或 `stage1_gate` 必须是 `complete`。
- `holdout_residual` 必须有限，默认范围为 `[0, 10]`。
- `split_half_nrmse` 必须有限且 `< 1.0`。
- `artifact_score` 必须有限；`> 5.0` 记 warning，`> 20.0` 记 failure。
- `raw_control_agreement` 必须有限。
- SIREN / WIRE / DeepInv-DIP 的 `best_step` 必须 `>= 500`，且不能 `early_stopped_before_500=true`。
- `hr_image.npy`、`hr_raw_control.npy`、`split_half_a.npy`、`split_half_b.npy` 必须存在。

这些只是训练健康门控，不是 SR 成功证明。Highpass 结构、棋盘纹、振铃、hallucination 和 split-half 差异图仍需要 notebook 或人工视觉检查。

## 6. 日志检查

查看某个 run 的最后日志：

```bash
tail -n 120 /home/ujs/mycode/thermal_lift/output/ep08_inr_sr/stage4_controller/logs/siren_064_full_preserve.log
tail -n 120 /home/ujs/mycode/thermal_lift/output/ep08_inr_sr/stage4_controller/logs/deepinv_dip_064_full_preserve.log
```

检查是否 OOM：

```bash
grep -i "out of memory\\|cuda.*memory\\|oom" /home/ujs/mycode/thermal_lift/output/ep08_inr_sr/stage4_controller/logs/*.log
```

查看后台进程：

```bash
ps -ef | grep "scripts/train_stage3.py" | grep -v grep
nvidia-smi
```

## 7. OOM 后的保守重跑

如果某个 run OOM，先归档旧输出和旧 launch record，再用更小 `batch_k` 重新让 controller 启动。

以 `siren_064_full_preserve` 为例：

```bash
cd /home/ujs/mycode/thermal_lift
stamp=$(date +%Y%m%d_%H%M%S)
mv output/ep08_inr_sr/stage3/siren_064_full_preserve output/ep08_inr_sr/stage3/siren_064_full_preserve_oom_$stamp
rm -f output/ep08_inr_sr/stage4_controller/runs/siren_064_full_preserve.launch.json

cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py tick --max-frame 64 --batch-k 4
```

如果 OOM 发生在 128 或 255 帧，把 `--max-frame` 改成对应阶段或直接省略，让 controller 回到当前 blocked phase。

## 8. Notebook 构建

至少 64 帧四方法都完成后，可以构建 EP08 notebook。Notebook 不启动训练，只读取产物：

```bash
cd /home/ujs/mycode/thermal_lift
uv run python scripts/build_notebook.py notebooks/ep08_inr_sr --execute
```

重点输出：

```text
output/ep08_inr_sr/stage3_progressive_metrics.png
output/ep08_inr_sr/stage3_visual_comparison.png
output/ep08_inr_sr/stage3_aspect_ablation.png
```

如果只跑了 `preserve`，aspect ablation 显示 pending 是正常的。

## 9. Stretch ablation

默认主线只跑 `preserve`。如需额外做 64 帧 stretch ablation：

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py tick --frames 64 --max-frame 64 --coord-aspect-mode stretch
```

这会生成 `*_064_full_stretch` 目录。完成后重新执行 notebook，fragment 11 会自动配对 preserve/stretch 指标。

## 10. 远程端 blocked 时应回报的信息

把以下内容带回给本机：

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep08_inr_sr
uv run python scripts/stage4_controller.py status --json > /tmp/ep08_stage4_status.json || true
cp /tmp/ep08_stage4_status.json /home/ujs/mycode/thermal_lift/output/ep08_inr_sr/stage4_controller/status_snapshot.json
```

同时回报：

- `status` 表格中的 failed/warn 行。
- 对应 run 的 log 尾部 120 行。
- `nvidia-smi` 输出。
- 是否存在该 run 的 `metrics.json`。

远程端不应在 blocked 状态下自行修改训练代码；只能做归档旧 run、降低 `--batch-k`、重新 tick 这类简单恢复操作。
