# Solver V2 Stage 0g 远端任务包（2026-07-04，写给 5090 上的执行 Agent）

> 背景：Stage 0f 已核验（**ACL-047**，先读它——里面含对上轮 REPORT 的两处勘误：Task 2 的 FRC 列被看串、Task 3 阈值结论过头）。
> 0f 的结论：真实逐帧 shift 误差 ~0.29px 是头号瓶颈；神经臂 cross-FRC 带内表现差但存在网格约定偏移的替代解释待排除。
> 本任务包两件事：**Task 1 = shift 反馈回路**（0a 精修 shift 反哺全管线，验证是否真信息增益）；**Task 2 = 同半幅跨方法对照**（裁决神经臂带内定罪成立与否）。
> 全 CPU，GPU 保持空闲；训练臂继续冻结。环境：repo 在 `~/thermal_lift`（WSL，用户 ujs），128 核。

## 第 0 步 — 同步与自检（必做）

```bash
cd ~/thermal_lift && git pull --ff-only
uv run --with pytest pytest algos/ep09_psf_calibration/tests/ -q   # 应 11 passed
```

全绿才继续。任何失败先报告、不要改测试。**本轮起新纪律：需要改 repo 代码时先在 REPORT 里报告、经 owner 同意再动（上轮 0fc51c8 直接 push 属越权，内容虽对但流程不再允许）。**

## Task 1 — shift 反馈回路（CPU，总 ~2–2.5h，步骤有依赖需串行）

### 1.1 构建精修对齐资产（秒级）

```bash
uv run python algos/ep09_psf_calibration/scripts/build_refined_alignment.py \
  --refinements-csv output/ep09_psf_calibration/stage0f_centered_full/stage0a_best_shift_refinements.csv \
  --output-csv output/ep09_psf_calibration/stage0f_refined_alignment.csv
```

预期输出（已在本地用同一份数据验证过）：`rows: 248`，`applied delta: mean=0.2878 px, p95=0.4472 px, axis means=(+0.0103, +0.0069)`。若数字不同，停下报告。

### 1.2 — 0a 第二轮迭代收敛检查（~1h）

```bash
setsid nohup uv run python algos/ep09_psf_calibration/scripts/run_stage0a_mvp.py \
  --use-all-score --sigmas 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50 \
  --anisotropy-ratios 1.0 \
  --shift-refine-radius 0.4 --shift-refine-step 0.05 \
  --forward-convention centered --xhat-source full \
  --alignment-csv output/ep09_psf_calibration/stage0f_refined_alignment.csv \
  --output-dir output/ep09_psf_calibration/stage0g_iter2 \
  > output/logs/stage0g_iter2.log 2>&1 &
```

判读要点：
- **收敛信号**：iter2 的 delta norm mean/p95 应显著小于 iter1 的 0.288/0.447（残余 <0.1px 量级=一轮基本收敛；若仍 ~0.29px 环状分布=精修在追逐 x̂ 伪影而非真误差，如实报告，反馈回路判失败）。
- iter2 的 improvement %（baseline 是新对齐下的 σ0.5 no-refine）应大幅缩水——剩下的才是 σ/模型形状项。
- 每轴均值仍应 ≈0。

### 1.3 — drizzle 分半 FRC（精修对齐 vs 旧对齐，~10 min，可与 1.2 并行）

```bash
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0g_frc_refined \
  --alignment-csv output/ep09_psf_calibration/stage0f_refined_alignment.csv \
  --methods drizzle --split-mode both --save-recons --force-inputs
```

（`--force-inputs` 必带：cache 里存的是旧对齐的 shifts。）
判读：drizzle 1/7 cutoff vs 旧值 29.67µm（phase_stratified seed42）/ odd_even 26.9µm。下移=真信息增益；不动或上移=如实报告。**自拟合纪律**：精修 shift 对共享 SAA x̂ 拟合过，两半都被拉向同一参考，cutoff 单独下移不足为证——必须结合 1.4 的负对照读。

### 1.4 — EP15 M2 权威频带重跑（精修对齐，~1h，在 1.3 之后）

```bash
setsid nohup uv run python algos/ep15_info_limit/scripts/run_m2_frc.py \
  --output-dir output/ep15_info_limit/stage0g_20um_refined/m2_frc \
  --alignment-csv output/ep09_psf_calibration/stage0f_refined_alignment.csv \
  --workers 4 --seeds 42 123 456 --crop-lr-px 16 --tukey-alpha 0.25 \
  > output/logs/stage0g_m2_refined.log 2>&1 &
```

判读（对照 0e 基线：cutoff 34.07µm、controls@20µm shuffle 0.867/drift 0.887）：
- first-crossing 1/7 cutoff 是否从 34.07µm 下移，seed std 是否仍 ~1.6µm 量级。
- **负对照是关键**：shift-shuffle / drift control 在 24–40µm 段是否保持低于 main（差距不缩小）。若 main 和 controls 一起抬升=精修在放大对齐伪影，反馈回路判失败。
- aperture dip（20µm）诊断照报。

### Task 1 总判定（写进 REPORT）

三个信号一起给结论：iter2 残余显著缩小 **且** drizzle cutoff 下移 **且** M2 main−control 差距不缩小 → 精修 shift 判定为真信息增益，候选升级为新对齐资产（升级动作 owner 拍板，本轮不动 configs/）。任一信号反向 → 如实报告哪个失败。

## Task 2 — 同半幅跨方法对照（CPU，~5 min，随时可做）

目的：0f cross-FRC 里神经臂带内 ~0.1 有两种解释——(a) 真的丢了带内信息；(b) 神经输出网格与经典网格存在配准/约定偏移导致高频相关被衰减。同半幅（同帧）跨方法 FRC 区分两者：同帧内容一致，忠实方法之间应高相关；若经典×经典高而神经×经典低，用曲线形状区分（刚性偏移=J0 式振荡/规律变号；内容替换=单调衰减）。

```bash
R=output/stage0c_frc_recons
D=output/stage0f_frc_cross/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0g_samehalf_control --methods none \
  --artifact-pair "tgv_a_vs_drz_a:$R/tgv_a.npy:${D}_a.npy:2:same_half_control" \
  --artifact-pair "maptv_a_vs_drz_a:$R/maptv_a.npy:${D}_a.npy:2:same_half_control" \
  --artifact-pair "v11_a_vs_drz_a:$R/v11_a.npy:${D}_a.npy:2:same_half_control" \
  --artifact-pair "c_nodr_a_vs_drz_a:$R/C_nodr_a.npy:${D}_a.npy:2:same_half_control" \
  --artifact-pair "d_dr01_a_vs_drz_a:$R/D_dr01_a.npy:${D}_a.npy:2:same_half_control" \
  --artifact-pair "v11_a_vs_tgv_a:$R/v11_a.npy:$R/tgv_a.npy:2:same_half_control" \
  --artifact-pair "v11_b_vs_drz_b:$R/v11_b.npy:${D}_b.npy:2:same_half_control_bside"
```

判读要点：
- 锚：`tgv_a_vs_drz_a` @24–30µm 应明显高（同帧+都忠实）。
- 若 `v11_a_vs_drz_a` @24–30µm 也高 → 0f 的跨半幅低值主要是真信息缺失以外的因素？不对——同半幅高+跨半幅低恰恰说明神经输出的高频不随帧内容变化（先验主导），**定罪成立**。
- 若 `v11_a_vs_drz_a` 也低 → 看 FRC 曲线形状：规律振荡/变号 → 网格偏移嫌疑，**定罪缓议**，报告曲线；单调衰减 → 神经输出连同帧经典重建都对不上，定罪成立且更重。
- `v11_a_vs_tgv_a`（两个非 drizzle 网格）与 a/b 两侧一致性作旁证。曲线 CSV 全部归档。

## 汇报要求

- 产出集中到 `remote_inbox/20260705_stage0g/`：summary/曲线 CSV 副本 + `REPORT.md`（每个 Task 一节：命令、关键数字、判读、异常；**每个数字标出处文件**）。
- 所有后台任务留 stdout log；任何截断/降配显式记录。
- 不改阈值、不动 `configs/`、不 push、不启动训练、不修 repo 代码（发现代码问题先报告）。
- 背景排队（不阻塞）：无。0d 阈值等 owner 决定，本轮不动。
