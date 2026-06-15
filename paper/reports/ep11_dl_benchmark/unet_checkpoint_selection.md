# EP11 UNet Checkpoint Selection

> 日期: 2026-06-11  
> 范围: EP07 学习类 SR 四臂 checkpoint 选优整理。  
> 边界: 本报告只做已有 TensorBoard 指标与已有 `eval_real/*.png` 的机械整理；未改算法、未重训、未删除 checkpoint。

## 产物

- 指标 CSV: `output/ep11_dl_benchmark/checkpoint_selection/checkpoint_metrics.csv`
- 候选 CSV: `output/ep11_dl_benchmark/checkpoint_selection/checkpoint_candidates.csv`
- 轨迹图: `output/ep11_dl_benchmark/checkpoint_selection/fig_trajectories.png`
- Pareto 图: `output/ep11_dl_benchmark/checkpoint_selection/fig_pareto.png`
- 视觉面板:
  - `output/ep11_dl_benchmark/checkpoint_selection/panel_v6.png`
  - `output/ep11_dl_benchmark/checkpoint_selection/panel_v8.1a.png`
  - `output/ep11_dl_benchmark/checkpoint_selection/panel_v8.1b.png`
  - `output/ep11_dl_benchmark/checkpoint_selection/panel_v9b.png`
- Manifest: `output/ep11_dl_benchmark/checkpoint_selection/checkpoint_selection_manifest.json`

复现命令:

```bash
algos/ep07_unet_sr/.venv/bin/python3 algos/ep07_unet_sr/scripts/extract_checkpoint_metrics.py
algos/ep07_unet_sr/.venv/bin/python3 algos/ep07_unet_sr/scripts/plot_checkpoint_selection.py
```

## 选择规则

每臂内对 `artifact_score` 与 `raw_control_corr` 做 min-max 归一化，其中 `artifact_score` 越小越好，`raw_control_corr` 越大越好。计算到理想点的欧氏距离后，取距离最小的 3 个 step；同一 5K step 窗口内只保留 1 个。最后固定附加 60K 终点作为漂移对照。

EP10 TGV 参考点在 Pareto 图中标为 `(artifact_score=0.695, raw_control_corr=0.916)`。

## Proxy 警告

1. `artifact_score` / `raw_control_corr` 是 proxy，不是光学 ground truth；跨输入模式不可横比，尤其不能把未来 V9A hybrid 输入与当前 1x-fused 输入四臂直接做数值胜负判断。
2. 两个 proxy 沿“合成先验风格化”轴反向联动，是同一漂移的两个读数：artifact 往往随风格化增强而变差，corr 同时下降。选优只能取折中，而不是期待双优。

## 候选表

| arm | step | artifact | corr | canonical | checkpoint |
|---|---:|---:|---:|---|---|
| v6 | 8000 | 0.330 | 0.774 | yes | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v6_physics/checkpoint_step_008000.pt` |
| v6 | 4000 | 0.337 | 0.774 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v6_physics/checkpoint_step_004000.pt` |
| v6 | 10000 | 0.361 | 0.770 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v6_physics/checkpoint_step_010000.pt` |
| v6 | 60000 | 0.883 | 0.648 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v6_physics/checkpoint_step_060000.pt` |
| v8.1a | 15000 | 0.392 | 0.758 | yes | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1a_loss_cooldown/checkpoint_step_015000.pt` |
| v8.1a | 10000 | 0.390 | 0.756 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1a_loss_cooldown/checkpoint_step_010000.pt` |
| v8.1a | 5000 | 0.379 | 0.750 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1a_loss_cooldown/checkpoint_step_005000.pt` |
| v8.1a | 60000 | 0.643 | 0.689 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1a_loss_cooldown/checkpoint_step_060000.pt` |
| v8.1b | 5000 | 0.370 | 0.739 | yes | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1b_pixelshuffle/checkpoint_step_005000.pt` |
| v8.1b | 10000 | 0.413 | 0.747 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1b_pixelshuffle/checkpoint_step_010000.pt` |
| v8.1b | 15000 | 0.544 | 0.705 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1b_pixelshuffle/checkpoint_step_015000.pt` |
| v8.1b | 60000 | 0.709 | 0.667 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1b_pixelshuffle/checkpoint_step_060000.pt` |
| v9b | 11000 | 0.339 | 0.777 | yes | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v9b_fwd_consistency/checkpoint_step_011000.pt` |
| v9b | 9000 | 0.340 | 0.777 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v9b_fwd_consistency/checkpoint_step_009000.pt` |
| v9b | 3000 | 0.341 | 0.766 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v9b_fwd_consistency/checkpoint_step_003000.pt` |
| v9b | 60000 | 0.655 | 0.688 |  | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v9b_fwd_consistency/checkpoint_step_060000.pt` |

所有候选的 checkpoint 与 `eval_real/unet_step*_center_zoom3x_temperature.png` 均存在；没有缺图需要补算。

## 推荐 Canonical Checkpoint

| arm | canonical checkpoint | 理由 |
|---|---|---|
| v6 | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v6_physics/checkpoint_step_008000.pt` | proxy 折中点最接近理想点，且位于 v6 明显漂移前；最终需人眼在面板中确认。 |
| v8.1a | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1a_loss_cooldown/checkpoint_step_015000.pt` | corr 比 10K/5K 更高，artifact 尚未进入后期单调上爬区，是 conservative loss 臂的早期折中点；最终需人眼在面板中确认。 |
| v8.1b | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v8_1b_pixelshuffle/checkpoint_step_005000.pt` | v8.1b 已判失败臂，仅作为对照；5K 是该臂最不坏的早期 proxy 折中点；最终需人眼在面板中确认。 |
| v9b | `/home/ujs/mycode/thermal_lift/algos/ep07_unet_sr/outputs/ep07_v9b_fwd_consistency/checkpoint_step_011000.pt` | 在四臂中给出最低 artifact 区域内最高 corr 的折中点，且明显早于 40K-60K 漂移段；最终需人眼在面板中确认。 |

当前机械推荐的主线候选是 `v9b` step 11000；若视觉面板确认无异常伪影，可作为 V9B 的 canonical checkpoint。由于 V9B 的 ACL-017 结论已证明 forward consistency 未压平后期漂移，不能因 v9b 早期 proxy 更好而宣称 loss 方案成功。

## 视觉面板读法

每个 `panel_<arm>.png` 横向展示该臂 3 个 proxy 候选与 60K 终点。面板使用已有 real_eval temperature PNG，不重算模型。60K 面板用于直观看后期风格化/漂移，不作为默认选择。

图中普通温度视图用于检查中心区域和内部轮廓是否只是边缘增强；它不是 highpass/residual 诊断图，也不能单独证明真实物理分辨率提升。

## V9A（训练中）完成后补充

V9A hybrid drizzle 输入仍在 GPU 0 训练中，本报告不读取、不改写 `algos/ep07_unet_sr/outputs/ep07_v9a_hybrid_drizzle/`。V9A 完成后应单独补充：

- V9A 同口径 real_eval 曲线与候选表。
- hybrid 输入模式内的 checkpoint 选择结果。
- 与当前四臂的视觉对照，但不把 artifact/corr 数值直接跨输入模式横比。

## 统一横评（任务 C）

已在 GPU 1 上按本报告的 canonical checkpoint 跑完 EP11 248 帧统一横评。运行命令使用 `CUDA_VISIBLE_DEVICES=1 --device cuda:0 --allow-cuda0`；各 `run_manifest.json` 记录的 `device_used` 均为 `cuda:0`，即物理 GPU 1 在受限可见设备列表中的逻辑编号。

注意：下表的 `artifact_score` 来自 `algos/ep11_dl_benchmark/scripts/run_unet_vs_drizzle_2x.py` 的统一横评口径，数值尺度不同于 TensorBoard `eval_real/artifact_score`，不能与前面的候选表直接混算。

| arm | output dir | split-half NRMSE | artifact | corr |
|---|---|---:|---:|---:|
| v6 | `output/ep11_dl_benchmark/checkpoint_selection/v6_step8000/` | 0.049672 | 1.785837 | 0.773807 |
| v8.1a | `output/ep11_dl_benchmark/checkpoint_selection/v8.1a_step15000/` | 0.068707 | 1.987040 | 0.757229 |
| v8.1b | `output/ep11_dl_benchmark/checkpoint_selection/v8.1b_step5000/` | 0.047868 | 1.759053 | 0.741383 |
| v9b | `output/ep11_dl_benchmark/checkpoint_selection/v9b_step11000/` | 0.051657 | 1.732180 | 0.777583 |

统一横评下，v9b step 11000 仍给出最高 `raw_control_corr` 与最低 EP11 artifact；v8.1b step 5000 的 split-half NRMSE 最低但 corr 明显偏低，仍只适合作为失败臂对照。最终 canonical 仍需以视觉面板确认，不由 proxy 单独决定。
