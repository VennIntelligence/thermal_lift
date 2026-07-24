# EP11 — UNet 2x@4000 vs TGV 2x Visual Benchmark

## Goal

Compare the EP07 residual UNet step-4000 checkpoint against the existing EP10
TGV best 2x artifact on the real 248 clean main-session frames.
The comparison is same-domain highpass, same center-third ROI, same 3x display
zoom, and same diverging colormap range.

## Inputs

- Raw input: EP06 clean main session, 248 frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `algos/ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt`.
- TGV highpass: `output/ep10_tgv_sr/best_hr_highpass.npy`.
- Highpass sigma: `5.0`.

## Artifacts

- Script: `algos/ep11_dl_benchmark/scripts/run_unet_vs_drizzle_2x.py`.
- Notebook: `notebooks/ep11_dl_benchmark/`.
- Output: `output/ep11_dl_benchmark/`.

## Boundary

This episode is a quick contour-level visual benchmark. It does not retrain
UNet, does not rerun the full TGV sweep, and does not claim 5 um metrology,
temperature accuracy, or 3x SR. The 3x setting is center-ROI display zoom only;
the reconstruction grid remains EP07 2x. UNet@4000 is a synthetic-pretrained
mid-training checkpoint, so any real-data advantage must be interpreted with
domain-gap risk.

## Progress

- 2026-06-11: Added EP07 four-arm checkpoint-selection整理 for v6 / v8.1a / v8.1b / v9b. Scripts: `algos/ep07_unet_sr/scripts/extract_checkpoint_metrics.py`, `algos/ep07_unet_sr/scripts/plot_checkpoint_selection.py`; report: `paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md`; generated artifacts under `output/ep11_dl_benchmark/checkpoint_selection/`, including GPU 1 unified EP11 reruns for the four recommended canonical checkpoints.

## 最终结论

- 四臂 canonical checkpoint 选优落定：v6@8000、v8.1a@15000、v8.1b@5000、v9b@11000；248 帧统一横评下 v9b step 11000 给出最高 raw_control_corr（0.7776）与最低 EP11 口径 artifact（1.732），v8.1b 只作失败变体对照（split-half NRMSE 最低但 corr 明显偏低）。（出处: `paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md` 候选表与统一横评节）
- 选优是 proxy 折中而非成功声明：artifact_score / raw_control_corr 沿"合成先验风格化"轴反向联动，canonical 最终需人眼面板确认；且 ACL-017 已证 forward consistency 未压平后期漂移，不能因 v9b 早期 proxy 更好宣称该 loss 方案成功。（出处: 同上报告 Proxy 警告节）
- 历史定位：EP11 的 UNet-era proxy 口径属于项目阶段 I（ACL-001–022）；阶段 III 评测仪器修复（ACL-046→049）后，跨方法正式结论一律以 corrected cross-FRC vs drizzle 为准，本 Episode 的数值不再直接用于最终判决。（出处: `research_log/algorithm_changelog.md` 顶部速览 #1/#4）
