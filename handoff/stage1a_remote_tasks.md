# Solver V2 Stage 1a 远端任务包（2026-07-02，写给 5090 上的执行 Agent）

> 背景：Stage 0 gate review 已通过（**ACL-044**，含全部实测数字），Stage 1a operator DR 代码已合入（**ACL-045**）。
> 先读这两条 changelog 条目再动手；判据与纪律以 `research_log/solver_v2_redesign_proposal.md` §2/§3 为准。
> 环境：repo 在 `~/thermal_lift`（WSL，用户 ujs），128 核 CPU + 单张 RTX 5090（当前空闲）。
> 真实数据：`data/data_raw/infrared_avi`（248 clean 帧，20µm pitch 契约）；合成池：`data/synthetic/pool_2x_v6_5k`。

## 第 0 步 — 同步与自检（必做）

```bash
cd ~/thermal_lift && git pull --ff-only
uv run --with pytest pytest algos/ep09_psf_calibration/tests/ tcforge/tests/test_geometry.py -q
cd algos/ep07_unet_sr && uv run pytest tests/test_dataset.py tests/test_config.py -q && cd ~/thermal_lift
```

全绿才继续。任何失败先报告、不要改测试。

## Task A — 0a 去饱和重跑（CPU，~3h，可与 Task C 并行）

上一轮 full-grid（`output/ep09_psf_calibration/stage0a_fullgrid/`）两个问题：best σ=0.15 在网格边缘；shift 精修 p95=0.5657=0.4√2 顶在边界盒角。放宽两者重跑：

```bash
setsid nohup uv run python algos/ep09_psf_calibration/scripts/run_stage0a_mvp.py \
  --use-all-score --sigmas 0.10,0.15,0.20,0.25,0.30,0.40,0.50 \
  --shift-refine-radius 0.8 --shift-refine-step 0.1 \
  --output-dir output/ep09_psf_calibration/stage0a_fullgrid_r08 \
  > output/logs/stage0a_fullgrid_r08.log 2>&1 &
```

判读要点（写进汇报）：
- 新代码会自动输出 `val_band_mse_improvement_bootstrap`（95% CI + 显著性）——引用它，不要只报点估计。
- p95 是否仍 = 0.8×√2 ≈ 1.131（仍饱和 → 真实误差更大，如实报告，不要再扩半径）。
- best σ 是否仍在网格边缘（0.10）。
- **解释纪律**：x̂ 是同源 SAA（自带一次 PSF），σ 估计系统性偏低；此数字是 floor probe，不是物理 σ̂ 认定，不要写成"PSF 标定结果"。

## Task B — 0d 好例校准（GPU 各 ~分钟级，在 Task C 开始前做完）

对至少两个"已知可接受"的 checkpoint 跑回归套件（V11 主线 + Prompt A/B 之一）：

```bash
cd algos/ep07_unet_sr
uv run python scripts/solver_regression_suite.py \
  --checkpoint <ckpt.pt> --fail-on-skip \
  --output-json ../../output/ep07_solver_regression_suite/goodcase_<name>.json
```

checkpoint 在 `outputs/solver_v11_k2_p384_nogn_halo96_50k/`（取 step_040000）与 `outputs/solver_v12_promptA_*_5k/` 下自行定位。判读：
- 记录 4 probe 各自 pass/fail 与数值。已知坏例（V8/K4）4/4 fail；好例若 flat-ROI/extent/beading 也 fail，或 seam autocorr 好例同样 >0.35，**不要自行改阈值**——把好例/坏例的数值分布并排报告，阈值调整由 owner 定。
- 套件在无 checkpoint 时会静默回退本地数组且 skip 不算失败，因此**必须**带 `--checkpoint` 与 `--fail-on-skip`。

## Task C — v6 no-DR 控制臂（GPU，主件，先跑）

池已从 v5→v6，历史 V11 不能直接当 1a 对照（池+DR 两变量混杂，ACL-044）。控制臂 = ACL-041 E3 主线原命令、只换池、20k steps：

```bash
cd algos/ep07_unet_sr
setsid nohup uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v6_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
  --solver-no-drizzle --patch-size-hr 384 \
  --total-steps 20000 --batch-size 4 --num-workers 12 \
  --synth-eval-holdout 500 --seed 42 \
  --output-dir outputs/solver_v13_v6_nodr_ctrl \
  > outputs/solver_v13_v6_nodr_ctrl.log 2>&1 &
```

（noSE/noGN/full_halo96 已是默认，勿显式恢复 SE/GN。）

## Task D — DR 臂（GPU，Task C 完成后串行，单卡）

与 Task C **逐字相同**，只加一个 flag、换输出目录：

```bash
  --solver-dc-shift-jitter-std-px 0.1 \
  --output-dir outputs/solver_v13_v6_dr01
```

- PSF jitter（`--solver-dc-psf-sigma-jitter-frac` / `--solver-dc-psf-angle-jitter-deg`）**首轮不开**——单变量纪律。若 D 对 C 有效且 Task A 显示 σ 不确定性大，第二轮再议。
- 启动后确认 banner 打印 `operator_DR=shift±0.1px ...`，且 synth_eval 指标与 C 同口径（eval 不扰动，代码已保证）。

## Task E — 0c artifact pairs 排行榜（Task C/D 训练期间的 CPU/GPU 空隙做）

目标：把 TGV / MAP-TV / V11 / (C、D 的 final checkpoint) 全部放进 split-half FRC 同一排行榜，锚点=drizzle（1/7 cutoff mean≈28.2µm，ACL-044）。

1. 分半用 0c 的既有缓存（`output/stage0c_real_split_frc_v2/cache/`，phase_stratified seed42 的 A/B 帧索引可从 `split_balance.csv`/代码 `build_splits` 复现，seed 固定 42）。
2. 对每个方法分别用 A 半、B 半各自重建一张全幅图（模型走 `full_halo96` 推理口径、只喂该半的帧与 shifts；TGV/MAP-TV 用 EP10/EP15 现有实现，参数取各自历史最优）。保存为 `.npy`（960×1280，HR 2x 网格）。
3. 统一评估：

```bash
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0c_frc_leaderboard \
  --artifact-pair v11:<a.npy>:<b.npy>:2 \
  --artifact-pair tgv:<a.npy>:<b.npy>:2 \
  ...（每方法一对）
```

判读纪律（ACL-044）：aligned_mean 的 cutoff 是插值核伪影、不采信；两半共享对齐 → 绝对 µm 偏乐观，**只报方法间相对差**；below-20µm 一律 audit-only。

## 判据（1a 成败判定，proposal §2 Stage 1a）

D（DR 臂）相对 C（控制臂）：
1. **真实 split-half FRC**（Task E 排行榜内相对比较）不降、期望提升;
2. **0d 回归套件**（好例校准后的口径）通过;
3. **synth 不塌**：holdout PSNR 相对 C 的差距在 ~1 dB 内（DR 理论上小幅伤 synth，塌太多=DR 过强）;
4. real-eval 的 artifact/OOB 不升。
dc_resid 仍冻结为非判据（0a 非退化标定完成前）。

## 汇报要求

- 所有产出集中到 `remote_inbox/20260703_stage1a/`：各任务 summary JSON/CSV 副本 + 一份 `REPORT.md`（每个 Task 一节：命令、关键数字、判读、异常）。
- 任何截断/降配（帧数、步数、网格）必须在 REPORT.md 里显式记录（"截断必 log"）。
- 不要改判据阈值、不要动 `configs/`、不要 push（owner 本地拉取审阅后自己合）。
- 背景排队（有空再跑，不阻塞）：0e = EP15 20µm 权威频带重跑（roadmap Step 5 欠账）。
