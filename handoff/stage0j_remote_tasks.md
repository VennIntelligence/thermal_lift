# Solver V2 Stage 0j 远端任务包（2026-07-06，写给 5090 上的执行 Agent）

> 背景：先读 changelog 顶部的"✅ 当前有效结论速览"和 **ACL-050**。v4 裁决（ACL-049）后神经与经典同档，
> 且 V11（50k）cutoff 优于 C（20k）——v6+E3 主线臂没训到成熟度。本轮：**Task 1 = 把最新数据集（v6 池）+
> 最新架构（E3 主线）训到 50k**；**Task 2 = 用中心网格出口渲染 + 排行榜 v5**，回答"成熟度补齐后距 TGV 还差多少"。
> DR 已关案，不开任何 DR 旗标。V11 不重训，只作历史基准。
>
> 🔴 **inbox 纪律（AGENTS.md 新硬规则）**：`remote_inbox/` 只走 rsync/scp/SSH 管道增量同步，**严禁 git add/commit/push**（上轮 8e4467c 已被回滚 ce29f74，不许有第三次）。

## 第 0 步 — 同步与自检（必做）

```bash
cd ~/thermal_lift && git pull --ff-only
uv run --with pytest pytest algos/ep09_psf_calibration/tests/ -q          # 11 passed
uv run --directory algos/ep07_unet_sr pytest tests/test_center_grid.py tests/test_dataset.py -q   # 19 passed
uv run python -c "import sys; sys.path.insert(0,'core/src'); from thermal_core.alignment_paths import default_contour_alignment_csv; assert default_contour_alignment_csv().name=='stage0f_refined_alignment.csv'; print('ALIGN_OK')"
```

全绿才继续。

## Task 1 — v14 成熟度臂：v6+E3 主线 50k（GPU，~2.5–3h）

与 C（`solver_v13_v6_nodr_ctrl`）**完全同参、同 seed，唯一区别 total-steps 20000→50000**（前 20k 顺带复现 C 作 sanity）。按 AGENTS.md 的 tmux 长任务模式跑：

```bash
cd ~/thermal_lift/algos/ep07_unet_sr
# tmux new-window -t 0: -n v14_50k '...'（模式照 AGENTS.md，保留 exit code 与 log）
uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v6_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
  --solver-no-drizzle --patch-size-hr 384 \
  --total-steps 50000 --batch-size 4 --num-workers 12 \
  --synth-eval-holdout 500 --seed 42 \
  --output-dir outputs/solver_v14_v6_nodr_50k \
  2>&1 | tee outputs/solver_v14_v6_nodr_50k.log
```

- 不加任何 `--solver-dc-*` 旗标（DR 关案）。
- **20k 复现性检查**：step 20000 的 eval_synth PSNR 应≈31.26（C 的值，同 seed 同参）；偏差 >0.3dB 先停下报告，不继续。
- banner 应打印 `operator_DR=off`。

## Task 2 — 渲染 + 排行榜 v5（Task 1 完成后，CPU+GPU ~40 min）

1. **渲染**：用你的 scratch 渲染脚本对 `solver_step_020000.pt` 与 `solver_step_050000.pt` 各渲 A/B 半幅，
   **调用 `infer_solver_from_burst_full_halo(..., output_grid="centered")`**（pull 后新参数——这是渲染源头的
   +0.5px 修正，ACL-050）。输出 `output/stage0j_frc_recons/v14_{20k,50k}_{a,b}.npy`。log 打印 alignment CSV 路径
   （应为精修版）与 output_grid 取值。
2. **残余偏移探针**（centered 出口后应只剩 ~0.05px 内容残差；若仍 ~0.5px 说明没走新参数，停下报告）：

```bash
D2=output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/probe_pair_offset.py \
  --output-dir output/stage0j_offset_probe \
  --save-corrected-dir output/stage0j_corrected_recons \
  --pair "v14_50k_a:${D2}_a.npy:output/stage0j_frc_recons/v14_50k_a.npy:2" \
  --pair "v14_50k_b:${D2}_b.npy:output/stage0j_frc_recons/v14_50k_b.npy:2" \
  --pair "v14_20k_a:${D2}_a.npy:output/stage0j_frc_recons/v14_20k_a.npy:2" \
  --pair "v14_20k_b:${D2}_b.npy:output/stage0j_frc_recons/v14_20k_b.npy:2"
```

3. **排行榜 v5**（用探针的校正数组；经典行照 0i 的 tgv/maptv/drizzle 路径同跑对照）：

```bash
C5=output/stage0j_corrected_recons
N=output/stage0h_frc_recons
D2=output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0j_frc_leaderboard_v5 --methods none \
  --cross-pair "v14_50k_x_drz:$C5/v14_50k_a_corrected.npy:$C5/v14_50k_b_corrected.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "v14_20k_x_drz:$C5/v14_20k_a_corrected.npy:$C5/v14_20k_b_corrected.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "tgv_x_drz:$N/tgv_a.npy:$N/tgv_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "maptv_x_drz:$N/maptv_a.npy:$N/maptv_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "v14_50k_x_tgv:$C5/v14_50k_a_corrected.npy:$C5/v14_50k_b_corrected.npy:$N/tgv_a.npy:$N/tgv_b.npy:2"
```

REPORT 必答（24–30µm 段 + 1/7 cutoff，20µm 孔径零点不采信）：
1. **v14_50k vs TGV（0.702@30µm / cutoff 23.03µm）**——主裁决行；
2. v14_20k vs C 的 0i 值（0.649@30µm / 25.45µm）——复现性；
3. v14 50k−20k 提升幅度 + 训练曲线是否已平（判断"再训还有没有肉"）；
4. 残余偏移表（centered 出口的实测残差）。

## 汇报要求

- 产出到 `remote_inbox/20260708_stage0j/`（summary/曲线 CSV 副本 + REPORT.md，每个数字标出处文件）——**只走 rsync/scp，严禁 git**。
- tmux 长任务模式 + stdout log + exit code 照 AGENTS.md；截断/降配显式记录。
- 不改阈值、不动 configs/、不 push、repo 代码不动（scratch 渲染脚本可改）。
