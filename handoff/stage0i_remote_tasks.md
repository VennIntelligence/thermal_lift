# Solver V2 Stage 0i 远端任务包（2026-07-05，写给 5090 上的执行 Agent）

> 背景：Stage 0h 的根因已裁决（**ACL-049**，先读）：神经输出网格带 +0.5 HR px 角点约定
> （`forward_torch.py:19-21` 有文档记载，训练内部自洽），所有神经×经典对比因此系统性低估；
> 校正后同半幅几乎追平经典（v11 vs TGV @24µm=0.808）。**决策：不改训练约定，修在对比层。**
> 本轮就一件事：**用偏移校正后的神经数组重跑排行榜 → v4 = 神经 vs 经典的最终公平裁决**。
> 全 CPU，~15–20 min。训练臂继续冻结。

## 第 0 步 — 同步与自检（必做）

```bash
cd ~/thermal_lift && git pull --ff-only
uv run --with pytest pytest algos/ep09_psf_calibration/tests/ -q   # 应 11 passed
uv run python algos/ep15_info_limit/scripts/probe_pair_offset.py --help | grep -q save-corrected-dir && echo FLAG_OK
```

全绿才继续。

## Task 1 — 神经三方法 a/b 两半分别测偏移 + 存校正数组（~5 min）

对 0h 的新渲染（`output/stage0h_frc_recons/`）逐半测偏移并保存校正版。a/b 两半各测一次是为了
**验证"偏移是渲染常数"**：同方法 a/b 的偏移应一致（差 <0.05px）；若不一致，说明还有别的因素，标红报告。

```bash
N=output/stage0h_frc_recons
D2=output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/probe_pair_offset.py \
  --output-dir output/stage0i_offset_probe \
  --save-corrected-dir output/stage0i_corrected_recons \
  --pair "v11_a:${D2}_a.npy:$N/v11_a.npy:2" \
  --pair "v11_b:${D2}_b.npy:$N/v11_b.npy:2" \
  --pair "c_nodr_a:${D2}_a.npy:$N/C_nodr_a.npy:2" \
  --pair "c_nodr_b:${D2}_b.npy:$N/C_nodr_b.npy:2" \
  --pair "d_dr01_a:${D2}_a.npy:$N/d_dr01_a.npy:2" \
  --pair "d_dr01_b:${D2}_b.npy:$N/d_dr01_b.npy:2"
```

（注意 0h 渲染的文件名大小写以实际为准：`C_nodr_a.npy` / `D_dr01_a.npy`——上面第 5/6 对如与磁盘不符请改成实际路径。）
产出：`output/stage0i_corrected_recons/{v11_a,v11_b,c_nodr_a,...}_corrected.npy`。
汇报：六个偏移值 + 同方法 a/b 偏移一致性 + 校正后残余（对校正数组再测一次应 <0.02px，抽 v11_a 验证即可）。

## Task 2 — 排行榜 v4（校正后最终裁决，~10 min）

用校正数组重跑 cross + 同半幅行；经典行同一条命令重跑作口径对照：

```bash
C=output/stage0i_corrected_recons
N=output/stage0h_frc_recons
D2=output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0i_frc_leaderboard_v4 --methods none \
  --cross-pair "v11_x_drz:$C/v11_a_corrected.npy:$C/v11_b_corrected.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "c_nodr_x_drz:$C/c_nodr_a_corrected.npy:$C/c_nodr_b_corrected.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "d_dr01_x_drz:$C/d_dr01_a_corrected.npy:$C/d_dr01_b_corrected.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "tgv_x_drz:$N/tgv_a.npy:$N/tgv_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "maptv_x_drz:$N/maptv_a.npy:$N/maptv_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "tgv_x_maptv:$N/tgv_a.npy:$N/tgv_b.npy:$N/maptv_a.npy:$N/maptv_b.npy:2" \
  --cross-pair "v11_x_tgv:$C/v11_a_corrected.npy:$C/v11_b_corrected.npy:$N/tgv_a.npy:$N/tgv_b.npy:2" \
  --artifact-pair "v11_samehalf:$C/v11_a_corrected.npy:${D2}_a.npy:2:corrected_same_half" \
  --artifact-pair "c_nodr_samehalf:$C/c_nodr_a_corrected.npy:${D2}_a.npy:2:corrected_same_half" \
  --artifact-pair "d_dr01_samehalf:$C/d_dr01_a_corrected.npy:${D2}_a.npy:2:corrected_same_half"
```

REPORT 必须给出的对照表（24–30µm 段 + 1/7 cutoff，20µm 照旧不采信）：
1. **v4 校正后神经 cross（v11/c/d × drz）vs 经典 cross（tgv 23.03µm / maptv 24.62µm 档）**——这是最终裁决行。
2. v4 vs v3 的神经行提升幅度（量化偏移曾经吃掉多少）。
3. `v11_x_tgv`（校正神经 × 经典，双方都非 drizzle 网格）作旁证。
4. C vs D 校正后是否仍零差。

## 汇报要求

- 产出到 `remote_inbox/20260707_stage0i/`：summary/曲线 CSV 副本 + REPORT.md（命令、关键数字、判读、异常，每个数字标出处）。
- stdout log 照旧保留；不改阈值、不动 configs/、不 push、不训练、repo 代码不动。
