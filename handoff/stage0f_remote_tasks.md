# Solver V2 Stage 0f 远端任务包（2026-07-03，写给 5090 上的执行 Agent）

> 背景：Stage 1a 判定为 **inconclusive（仪表失效）**，见 **ACL-046**（先读它，再读 ACL-044/045）。
> 三个仪表问题：0a 打分器半像素约定 bug（已修，`--forward-convention centered`）、split-half FRC 被神经方法的可复现幻觉刷高（已加 `--cross-pair`）、0d 阈值好坏例无判别力（已加分布对比工具）。
> 本任务包 = 用修好的仪表重测。**全部 CPU 任务，GPU 保持空闲**；在 0f 结果出来前不启动任何新训练臂。
> 环境：repo 在 `~/thermal_lift`（WSL，用户 ujs），128 核 CPU。

## 第 0 步 — 同步与自检（必做）

```bash
cd ~/thermal_lift && git pull --ff-only
uv run --with pytest pytest algos/ep09_psf_calibration/tests/ -q   # 应 10 passed
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py --help | grep -q cross-pair && echo FLAGS_OK
```

全绿才继续。任何失败先报告、不要改测试。

## Task 1 — 0a 修正约定重跑 ×2（CPU，每个 ~50–70 min，可并行）

上一轮 r0.8 的精修分布 mean Δ≈(−0.24,−0.24) px 是 forward/SAA 半像素约定差（ACL-046 §1），已修。用 centered 重跑两个变体：

```bash
mkdir -p output/logs
setsid nohup uv run python algos/ep09_psf_calibration/scripts/run_stage0a_mvp.py \
  --use-all-score --sigmas 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50 \
  --anisotropy-ratios 1.0 \
  --shift-refine-radius 0.4 --shift-refine-step 0.05 \
  --forward-convention centered --xhat-source full \
  --output-dir output/ep09_psf_calibration/stage0f_centered_full \
  > output/logs/stage0f_centered_full.log 2>&1 &

setsid nohup uv run python algos/ep09_psf_calibration/scripts/run_stage0a_mvp.py \
  --use-all-score --sigmas 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50 \
  --anisotropy-ratios 1.0 \
  --shift-refine-radius 0.4 --shift-refine-step 0.05 \
  --forward-convention centered --xhat-source split_half \
  --output-dir output/ep09_psf_calibration/stage0f_centered_splithalf \
  > output/logs/stage0f_centered_splithalf.log 2>&1 &
```

（两点降配说明，属设计而非截断：step 0.1→0.05 因为系统偏置消除后要看的是 ≤0.2px 的真实散布结构；同时砍掉 anisotropy 1.5 维（r08 全网格里 iso 恒最优、各向异性行差距在小数点后 4 位），candidates 40→8，抵掉细步长的 3.5× 成本，单个任务 ~50–70 min。radius 0.4 应足够，若 p95 仍顶 0.4√2 如实报告、不要自行扩。两个任务可并行，若互相争 CPU 就串行。）

判读要点（写进汇报，每条都要有数字出处）：
1. `stage0a_best_shift_refinements.csv` 的 **delta_dx/delta_dy 每轴均值**：修正后应≈0；若仍有显著非零系统分量，那是真实的全局对齐偏置，单独报告。
2. delta 范数的 mean/p95 与直方图形状（是否仍在 0/±0.5 处双峰=插值 landscape 伪影嫌疑）。这是**真实逐帧 shift 误差**的第一个干净估计，直接决定 DR 量级与 test-time shift refinement 的优先级。
3. `val_band_mse_improvement_bootstrap`（点估计+CI）与旧 20.07% 对比：修约定后还剩多少"真模型误差可修"。
4. best σ 是否离开网格下边缘；full vs split_half 的 best σ 差异（split_half 消除了帧参与自身 x̂ 的自拟合，σ̂ 仍因 SAA 自带模糊而偏低——解释纪律不变，floor probe 而非物理 σ̂）。
5. split_half 的绝对 band_mse 会比 full 高（每张 x̂ 只有 124 帧）——只比相对改善，不比绝对值。

## Task 2 — cross-method FRC 排行榜（CPU，~15 min）

自相关 FRC 对神经方法失效（ACL-046 §4），改用交叉法：方法 X 的 A 半 vs 方法 Y 的 B 半（对称化取均值）。既有 recon npy 在 `output/stage0c_frc_recons/`（v11/tgv/maptv/C_nodr/D_dr01 的 a/b），drizzle 半幅由本次调用现场生成并保存（native 方法先于 cross-pair 执行，同一条命令内可用）：

```bash
R=output/stage0c_frc_recons
D=output/stage0f_frc_cross/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0f_frc_cross \
  --methods drizzle --save-recons \
  --cross-pair "v11_x_drz:$R/v11_a.npy:$R/v11_b.npy:${D}_a.npy:${D}_b.npy:2" \
  --cross-pair "c_nodr_x_drz:$R/C_nodr_a.npy:$R/C_nodr_b.npy:${D}_a.npy:${D}_b.npy:2" \
  --cross-pair "d_dr01_x_drz:$R/D_dr01_a.npy:$R/D_dr01_b.npy:${D}_a.npy:${D}_b.npy:2" \
  --cross-pair "tgv_x_drz:$R/tgv_a.npy:$R/tgv_b.npy:${D}_a.npy:${D}_b.npy:2" \
  --cross-pair "maptv_x_drz:$R/maptv_a.npy:$R/maptv_b.npy:${D}_a.npy:${D}_b.npy:2" \
  --cross-pair "tgv_x_maptv:$R/tgv_a.npy:$R/tgv_b.npy:$R/maptv_a.npy:$R/maptv_b.npy:2"
```

判读要点：
- 排行榜按 cross 的 1/7 cutoff 与 24–30µm 段 FRC 值列全表；`tgv_x_maptv` 是经典×经典的 sanity 锚（应与两者 self-FRC 大体一致）。
- **C vs D 的复判就看 `c_nodr_x_drz` vs `d_dr01_x_drz`**——这是 1a 第一个可信的真实域相对信号，结论写"相对差"，不写绝对分辨率。
- 神经方法的 cross 若显著低于其 self-FRC，即量化了幻觉占比，如实报告。
- 纪律：两方法仍共享 contour_refined 对齐 → 绝对 µm 依旧偏乐观；below-20µm 一律 audit-only。

## Task 3 — 0d 好/坏例分布表（CPU，~20 min）

先补坏例 JSON（V8/K4 checkpoint 在 `algos/ep07_unet_sr/outputs/` 下自行定位，step_010000 即当时的已知坏例）：

```bash
cd algos/ep07_unet_sr
uv run python scripts/solver_regression_suite.py \
  --checkpoint <v8_k4_ckpt.pt> --fail-on-skip \
  --output-json ../../output/ep07_solver_regression_suite/badcase_v8k4.json
```

（若还有第二个公认坏例 checkpoint，一并生成 badcase_*.json，样本越多分布表越有用。）然后：

```bash
uv run python scripts/compare_regression_distributions.py \
  --case good:v11_40k:../../output/ep07_solver_regression_suite/goodcase_v11_40k.json \
  --case good:promptA_5k:../../output/ep07_solver_regression_suite/goodcase_promptA_5k.json \
  --case bad:v8k4:../../output/ep07_solver_regression_suite/badcase_v8k4.json \
  --output-csv ../../output/ep07_solver_regression_suite/regression_metric_values.csv \
  --output-separability-csv ../../output/ep07_solver_regression_suite/regression_metric_separability.csv
```

注意：promptA 按原 Task B 设计归为 good（上轮 REPORT 曾标 Badcase，归类以 owner 后续确认为准，先照此跑）。**不要改任何阈值**——separability 表只是给 owner 定阈值的数据。

## 汇报要求

- 所有产出集中到 `remote_inbox/20260704_stage0f/`：各 summary JSON/CSV 副本 + `REPORT.md`（每个 Task 一节：命令、关键数字、判读、异常）。
- **每个数字标出处文件**；所有后台任务必须留 stdout log（上轮 Task C/D 无 log、原始数字出错靠事后审计纠正，这次不允许）。
- 任何截断/降配必须显式记录（"截断必 log"）。
- 不改阈值、不动 `configs/`、不 push、不启动任何训练。
- 背景排队（有空再跑，不阻塞）：0e = EP15 20µm 权威频带重跑（roadmap Step 5 欠账）。
