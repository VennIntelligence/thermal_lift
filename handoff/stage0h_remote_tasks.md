# Solver V2 Stage 0h 远端任务包（2026-07-05，写给 5090 上的执行 Agent）

> 背景：Stage 0g 判定精修 shift 为**真信息增益**（ACL-048，先读 ACL-047/048）：M2 权威频带 34.07→25.45µm，
> drizzle cutoff 29.67→26.28µm，iter2 收敛 0.012/0.071px。精修对齐已升级为 **repo 默认资产**
> （`configs/alignment/stage0f_refined_alignment.csv`，pull 之后 `load_alignment_shifts` 默认即精修值）。
> 本轮三件事：**Task 1 = 偏移探针**（裁决 V11 同半幅振荡是否网格偏移）；**Task 2 = 用精修对齐重渲染神经臂分半**
> （神经臂此前的 DC 输入还是旧对齐——这是对神经臂的公平重审）；**Task 3 = 全精修口径的 cross-FRC 排行榜 v3**。
> Task 2 用 GPU 推理（非训练，分钟级）；其余 CPU。训练臂继续冻结。

## 第 0 步 — 同步与自检（必做）

```bash
cd ~/thermal_lift && git pull --ff-only
uv run --with pytest pytest algos/ep09_psf_calibration/tests/ -q   # 应 11 passed
uv run python -c "import sys; sys.path.insert(0,'core/src'); from thermal_core.alignment_paths import default_contour_alignment_csv; p=default_contour_alignment_csv(); print(p); assert p.name=='stage0f_refined_alignment.csv'"
```

第三条必须打印 `stage0f_refined_alignment.csv`——这确认默认对齐已切换。全绿才继续。

## Task 1 — 偏移探针（CPU，~5 min）

新工具 `algos/ep15_info_limit/scripts/probe_pair_offset.py`：相位相关估计全局亚像素偏移（两遍迭代精化）
→ Fourier 反移 → 前后 FRC 对比。已合成验证（0.7px 注入恢复 ±0.02px；无偏移对照 ~0.02px）。
对 0g 用过的同半幅数组跑（a=drizzle 锚，b=被测方法；报告的 offset 是 b 相对 a）：

```bash
R=output/stage0c_frc_recons
D=output/stage0f_frc_cross/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/probe_pair_offset.py \
  --output-dir output/stage0h_offset_probe \
  --pair "v11_vs_drz:${D}_a.npy:$R/v11_a.npy:2" \
  --pair "c_nodr_vs_drz:${D}_a.npy:$R/C_nodr_a.npy:2" \
  --pair "d_dr01_vs_drz:${D}_a.npy:$R/D_dr01_a.npy:2" \
  --pair "tgv_vs_drz:${D}_a.npy:$R/tgv_a.npy:2" \
  --pair "maptv_vs_drz:${D}_a.npy:$R/maptv_a.npy:2" \
  --pair "v11_vs_tgv:$R/tgv_a.npy:$R/v11_a.npy:2"
```

判读：
- 预期（ACL-048 的符号翻转分析）：v11 两对显著偏移 + 校正后 sign_changes 塌缩、FRC 回升；tgv/maptv 对 drz 偏移 ≈0（锚）；c/d 偏移 ≈0 且校正后无变化（内容分歧成立）。
- 报告每对：offset (dx,dy) HR px 与 µm、cutoff before/after、sign_changes before/after、frc_at_30/24µm after。
- 若 c/d 也测出 ≥0.2px 偏移：0f/0g 对神经臂的"带内破坏"定罪全部重审，如实报告。

## Task 2 — 神经臂分半重渲染（GPU 推理，~15 min）

用与 0f Task E 相同的分半推理流程（你自己的 `reconstruct_halves.py` / `reconstruct_c_d_halves.py`）重渲染
**V11 / C(v13_nodr) / D(v13_dr01)** 的 A/B 半幅，唯一区别：pull 之后 shifts 默认就是精修对齐——
**在 log 里打印实际加载的 alignment CSV 路径确认**。输出到 `output/stage0h_frc_recons/`（命名沿用 v11_a.npy 等）。

注意：若脚本当时把 shifts 写死为旧 CSV 路径，改成默认加载（这属于你自己的 scratch 脚本，可改；repo 代码仍不许动）。

## Task 3 — 全精修口径排行榜 v3（CPU，~30–60 min）

1. 用精修对齐重建 TGV / MAP-TV 的 A/B 半幅（参数取各自历史最优，与 0f Task E 相同），存 `output/stage0h_frc_recons/`。
2. drizzle 精修半幅已有：`output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42_{a,b}.npy`。
3. 排行榜（self + cross + 同半幅对照一次跑完）：

```bash
N=output/stage0h_frc_recons
D2=output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42
uv run python algos/ep15_info_limit/scripts/run_real_split_frc_v2.py \
  --output-dir output/stage0h_frc_leaderboard --methods none \
  --artifact-pair "v11_self:$N/v11_a.npy:$N/v11_b.npy:2" \
  --artifact-pair "c_nodr_self:$N/C_nodr_a.npy:$N/C_nodr_b.npy:2" \
  --artifact-pair "d_dr01_self:$N/D_dr01_a.npy:$N/D_dr01_b.npy:2" \
  --artifact-pair "tgv_self:$N/tgv_a.npy:$N/tgv_b.npy:2" \
  --artifact-pair "maptv_self:$N/maptv_a.npy:$N/maptv_b.npy:2" \
  --cross-pair "v11_x_drz:$N/v11_a.npy:$N/v11_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "c_nodr_x_drz:$N/C_nodr_a.npy:$N/C_nodr_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "d_dr01_x_drz:$N/D_dr01_a.npy:$N/D_dr01_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "tgv_x_drz:$N/tgv_a.npy:$N/tgv_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "maptv_x_drz:$N/maptv_a.npy:$N/maptv_b.npy:${D2}_a.npy:${D2}_b.npy:2" \
  --cross-pair "tgv_x_maptv:$N/tgv_a.npy:$N/tgv_b.npy:$N/maptv_a.npy:$N/maptv_b.npy:2" \
  --artifact-pair "v11_samehalf:$N/v11_a.npy:${D2}_a.npy:2:same_half_control" \
  --artifact-pair "c_nodr_samehalf:$N/C_nodr_a.npy:${D2}_a.npy:2:same_half_control" \
  --artifact-pair "d_dr01_samehalf:$N/D_dr01_a.npy:${D2}_a.npy:2:same_half_control" \
  --artifact-pair "tgv_samehalf:$N/tgv_a.npy:${D2}_a.npy:2:same_half_control"
```

4. 若 Task 1 对某神经方法测出显著偏移：对该方法的**新渲染**同半幅对再跑一次偏移探针，报告偏移是否随重渲染消失（消失=对齐相关；仍在=渲染约定偏移，单独立项修）。

判读纪律：排行榜按 24–30µm 段 + 1/7 cutoff 读（20µm 是孔径零点不采信）；关键问题依次是——
(a) 换精修对齐后神经臂 cross 是否显著回升（回升=旧 shift 毒害 DC 是主因，翻案）；
(b) C vs D 相对差是否仍为零；(c) 经典臂在精修对齐下的 cutoff 提升幅度（新的要打败的基线）。

## 汇报要求

- 产出到 `remote_inbox/20260706_stage0h/`：各 summary/曲线 CSV 副本 + REPORT.md（命令、关键数字、判读、异常，每个数字标出处文件）。
- **补交上轮欠账**：`output/ep09_psf_calibration/stage0g_iter2/stage0a_summary.json` 一并放入 inbox。
- 后台任务留 stdout log；截断/降配显式记录；不改阈值、不动 configs/、不 push、不训练、repo 代码不动（scratch 脚本除外）。
