# Stage 1a Remote Tasks Report

> **Codex audit correction (2026-07-03)**: 原始报告中的主要产物基本存在，但 Task C 的 final metrics 与 TensorBoard event 不一致，Task E 的 seed 表述容易误读，且 inbox 归档缺少 summary JSON/CSV 副本。本版已按 checkpoint、TensorBoard scalar、FRC summary 和落盘文件重新校正。关键审计文件见 `MANIFEST.md`。

## Task A: CPU 0a 去饱和重跑
- **执行命令**:
  ```bash
  uv run python algos/ep09_psf_calibration/scripts/run_stage0a_mvp.py \
    --use-all-score --sigmas 0.10,0.15,0.20,0.25,0.30,0.40,0.50 \
    --shift-refine-radius 0.8 --shift-refine-step 0.1 \
    --output-dir output/ep09_psf_calibration/stage0a_fullgrid_r08
  ```
- **关键数字**:
  - `val_band_mse_improvement_bootstrap.point_pct`: 20.07%
  - 95% CI: [18.16%, 22.06%]
  - 最优候选参数 (Best candidate): `sigma_x_lr_px=0.1, sigma_y_lr_px=0.1, angle_deg=0.0`
  - Sweep 完整性: 72 行 = train/val 各 35 个 bounded candidate + 各 1 个 baseline no-refine；`n_loaded_frames=248`, `n_train_score_frames=198`, `n_val_score_frames=50`
- **判读**:
  - 改进显著。这印证了此前的结论，即目前的观测结果中去饱和效应占据了一定的主导地位，模型能够通过微调 shift 与极小的 PSF sigma (0.1px) 适配这种现象，达到 20% 左右的 MSE 提升。
  - 注意：`sigma=0.1 px` 是同源 SAA xhat 下的 floor-probe 适配结果，不是物理 PSF 标定结论。
- **异常**: 无
- **归档副本**: `artifacts/task_a/stage0a_summary.json`, `artifacts/task_a/stage0a_model_sweep.csv`, `artifacts/task_a/stage0a_best_shift_refinements.csv`

## Task B: 0d 好例校准 (GPU)
- **执行命令**:
  ```bash
  uv run --directory algos/ep07_unet_sr python scripts/solver_regression_suite.py \
    --checkpoint outputs/solver_v11_k2_p384_nogn_halo96_50k/solver_step_040000.pt \
    --fail-on-skip \
    --output-json output/ep07_solver_regression_suite/goodcase_v11_40k.json

  uv run --directory algos/ep07_unet_sr python scripts/solver_regression_suite.py \
    --checkpoint outputs/solver_v12_promptA_A_nodrizzle_5k/solver_step_005000.pt \
    --fail-on-skip \
    --output-json output/ep07_solver_regression_suite/goodcase_promptA_5k.json
  ```
- **关键数字**:
  - **V11 (Goodcase)**:
    - `flat_roi_artifact`: fail (p95 = 0.1546 > 0.08, std = 0.0061 < 0.04)
    - `tiled_full_halo_extent_consistency`: pass (nrmse = 0.0658 < 0.15, p95_abs = 0.1004 < 0.15)
    - `seam_spectrum`: fail (max_abs_autocorr = 0.8634 > 0.35)
    - `beading_probe`: fail (edge_ratio = 8.59 > 4.0, excess_p95 = 0.663 > 0.04)
  - **PromptA 5k (Badcase)**:
    - `flat_roi_artifact`: fail (p95 = 0.1080 > 0.08, std = 0.0078 < 0.04)
    - `tiled_full_halo_extent_consistency`: pass (nrmse = 0.0528 < 0.15, p95_abs = 0.0802 < 0.15)
    - `seam_spectrum`: fail (max_abs_autocorr = 0.8743 > 0.35)
    - `beading_probe`: fail (edge_ratio = 7.67 > 4.0, excess_p95 = 0.586 > 0.04)
- **判读**:
  - 好例 (V11) 与坏例 (PromptA) 的数值分布极为相近。两者在 `flat_roi_artifact` 的 lowpass_p95_abs_delta_c 均未能满足 < 0.08 的阈值；在 `seam_spectrum` 中均展现出远高于 0.35 的相关性 (0.86 vs 0.87)；在 `beading_probe` 中均出现严重的边缘信号过冲 (edge_ratio 均在 7-8 之间，远超 4.0 阈值)。
  - 这表明现行回归套件无法区分人类主观感知的 “好例”，好例中的 halo / seam / beading 也触碰了极为严苛的物理容限。
- **异常**: 无修改物理配置，严格按指令执行并汇报。
- **归档副本**: `artifacts/task_b/goodcase_v11_40k.json`, `artifacts/task_b/goodcase_promptA_5k.json`

## Task C: v6 no-DR 控制臂训练 (GPU)
- **执行命令**:
  ```bash
  uv run python -m unet_sr.solver_train \
    --training-pool-dir ../../data/synthetic/pool_2x_v6_5k \
    --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
    --solver-no-drizzle --patch-size-hr 384 \
    --total-steps 20000 --batch-size 4 --num-workers 12 \
    --synth-eval-holdout 500 --seed 42 \
    --output-dir outputs/solver_v13_v6_nodr_ctrl
  ```
- **关键数字**:
  - Checkpoint audit: `solver_step_020000.pt` 与 `solver_final.pt` 均存在，checkpoint 内 `step=20000`, `solver_dc_shift_jitter_std_px=0.0`
  - (step=20000) `eval_synth`: `psnr=31.2635 region_rmse=0.1718 boundary_f1=0.8245 out_of_band_ratio=0.04293`
  - (step=20000) `eval_real`: `artifact_score=0.4253 out_of_band_ratio=0.001691 dc_resid_band=1.2409 dc_resid_full=1.5268`
- **判读**:
  - 训练已完成到 20k。原始报告中的 Task C 数字 (`psnr=33.31`, `region_rmse=0.1342`, `boundary_f1=0.8654`, `artifact_score=0.4023`) 与 TensorBoard event file 不一致，不能采信。
- **异常**: 未找到 `outputs/solver_v13_v6_nodr_ctrl.log` stdout/stderr 日志；完成状态由 checkpoint、TensorBoard max step 和 20k eval PNG 共同审计。
- **归档副本**: `artifacts/task_c_d/final_metrics.csv`, `artifacts/task_c_d/checkpoint_audit.json`

## Task D: DR 臂训练 (GPU)
- **执行命令**:
  ```bash
  uv run python -m unet_sr.solver_train \
    --training-pool-dir ../../data/synthetic/pool_2x_v6_5k \
    --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
    --solver-no-drizzle --patch-size-hr 384 \
    --total-steps 20000 --batch-size 4 --num-workers 12 \
    --synth-eval-holdout 500 --seed 42 \
    --solver-dc-shift-jitter-std-px 0.1 \
    --output-dir outputs/solver_v13_v6_dr01
  ```
- **关键数字**:
  - Checkpoint audit: `solver_step_020000.pt` 与 `solver_final.pt` 均存在，checkpoint 内 `step=20000`, `solver_dc_shift_jitter_std_px=0.1`
  - (step=20000) `eval_synth`: `psnr=31.23 region_rmse=0.1767 boundary_f1=0.8246 out_of_band_ratio=0.04191`
  - (step=20000) `eval_real`: `artifact_score=0.4195 out_of_band_ratio=0.001701 dc_resid_band=1.2406 dc_resid_full=1.5281`
- **判读**:
  - 与校正后的 C 控制臂相比，DR 0.1 的 final synth PSNR 只小幅下降 `31.2635 -> 31.2306`，region RMSE 变差 `0.1718 -> 0.1767`，boundary F1 基本持平 `0.8245 -> 0.8246`。
  - real artifact score 不是原始报告所说的上升，而是 final 点小幅下降 `0.4253 -> 0.4195`。这不等于证明 DR 有效，因为两臂在真实全幅重建和 FRC 上仍非常接近，且 0d 回归套件仍无法区分好例/坏例。
- **异常**: 未找到 `outputs/solver_v13_v6_dr01.log` stdout/stderr 日志；完成状态由 checkpoint、TensorBoard max step 和 20k eval PNG 共同审计。
- **归档副本**: `artifacts/task_c_d/final_metrics.csv`, `artifacts/task_c_d/checkpoint_audit.json`

## Task E: 0c 排行榜 (后台空隙计算)
- **执行命令**:
  ```bash
  uv run python scripts/reconstruct_halves.py       # 预计算 V11 / TGV / MAP-TV 的 A/B 输出
  uv run python scripts/reconstruct_c_d_halves.py   # 预计算 C(no-DR) / D(DR 0.1) 的 A/B 输出
  uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
    --output-dir output/stage0c_frc_leaderboard \
    --methods drizzle \
    --artifact-pair v11:output/stage0c_frc_recons/v11_a.npy:output/stage0c_frc_recons/v11_b.npy:2 \
    --artifact-pair tgv:output/stage0c_frc_recons/tgv_a.npy:output/stage0c_frc_recons/tgv_b.npy:2 \
    --artifact-pair maptv:output/stage0c_frc_recons/maptv_a.npy:output/stage0c_frc_recons/maptv_b.npy:2 \
    --artifact-pair c_nodr:output/stage0c_frc_recons/C_nodr_a.npy:output/stage0c_frc_recons/C_nodr_b.npy:2 \
    --artifact-pair d_dr01:output/stage0c_frc_recons/D_dr01_a.npy:output/stage0c_frc_recons/D_dr01_b.npy:2
  ```
- **关键数字**:
  (基于 20 µm spatial resolution 下的 FRC @ 20 µm；run manifest 中只有 `seed=42` 一个 phase-stratified split，不是 42 个 seeds)
  - `drizzle`: FRC @ 20µm = -0.429 (Cutoff @ 29.67 µm)
  - `maptv`: FRC @ 20µm = -0.619 (Cutoff @ 21.23 µm)
  - `tgv`: FRC @ 20µm = -0.403 (Cutoff @ 20.0 µm)
  - `v11`: FRC @ 20µm = 0.9998 (未发生截止，表现出强烈的 spurious 高频相关性)
  - `C_nodr`: FRC @ 20µm = 0.9310 (未发生截止)
  - `D_dr01`: FRC @ 20µm = 0.9975 (未发生截止)
- **判读**:
  - 根据纪律，`aligned_mean` 的异常高 FRC cutoff 不采信。可以看到 V11、C、D 在 20µm 频段的 FRC 均接近于 1，说明深度学习重建出的 A/B 半幅拥有极其一致的高频结构（但从前面的 regression suite 来看，这极大概率是网络将相同的 artifact 或 spurious prior “完美” 地重现在了两个半幅中）。
  - 而经典方法（TGV，MAP-TV）的 cutoff 均在 20-21 µm 附近，这更符合真实的物理信息极限。
- **异常**:
  - `drizzle` 是运行时重建，未保存 A/B `.npy`；5 个 external 方法的 A/B `.npy` 存在于 `output/stage0c_frc_recons/`。
  - `method_summary.csv` 中 external 方法用 `seed=-1` 占位；真实 split 来源是各自预计算脚本复现的 seed-42 A/B frame split。
- **归档副本**: `artifacts/task_e/method_summary.csv`, `artifacts/task_e/run_manifest.json`, `artifacts/task_e/split_balance.csv`

## Overall Audit Conclusion

- Task A/B/E 的关键 summary 数字与落盘文件一致。
- Task C/D 训练产物已完成到 20k；没有仍在运行的训练或 FRC 进程。
- 原始 Task C final metrics 错误，已按 TensorBoard scalar 校正。
- 原始 Task D 判读随 Task C 校正后需要改写：DR 0.1 没有造成原报告所说的明显 PSNR 下降或 artifact 上升，但也没有给出可信正收益。
- Stage 1a 不应被判为“圆满完成”：0d regression suite 对好/坏例区分力不足，Task E 显示 neural methods 的 split-half FRC 存在高度一致的 spurious 高频，C/D 在 real reconstruction 上差异极小。
