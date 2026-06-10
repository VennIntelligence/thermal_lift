# EP15: 4x 信息上限实测 + 经典去卷积锚

**定位**: EP12 4x 网络失败后，用第一性原理测量数据真实信息上限（FRC），并用经典 MAP-TV 去卷积建立后续网络方法必须超越的下限锚。

## M1 结论（相位结构）

**结论**: PASS WITH CAVEATS。`command_prior` 与 `contour_refined` 在 stage-coordinate 5x 相位网格上都覆盖了 25/25 个 cell，说明 248 帧 clean main input 中确实存在 2 µm 等效相位多样性；但 `contour_refined` 不保留 command cell label（match rate 4.4%），且 detector-axis 5x bin 只覆盖 11/25。

关键数值：

| 方法 | stage 5x 占用 | stage 最小/最大计数 | stage entropy | 最近 2 µm 网格 P90 | detector 5x 占用 | command label match |
|---|---:|---:|---:|---:|---:|---:|
| command_prior | 25/25 | 7 / 15 | 0.996 | 0.00 µm | 25/25 | 100.0% |
| ncc_init | 25/25 | 6 / 14 | 0.992 | 1.12 µm | 25/25 | 4.4% |
| contour_refined | 25/25 | 4 / 18 | 0.978 | 1.09 µm | 11/25 | 4.4% |

解释边界：

- stage-coordinate 的 25/25 覆盖支持“采样几何中有 5x 相位多样性”，但这不是 5x 光学可恢复性的证明。
- `contour_refined` 相位点仍接近 2 µm 网格（P90 约 1.09 µm），但和 command prior 的具体格点标签明显不一致；后续不能把 stage command 当作实测对齐真值。
- `contour_refined` 在 detector-axis 5x bin 上塌缩到 11/25，应作为 4x/5x 高倍率风险信号；M2 FRC、M3 σ 仲裁和 M4 去卷积锚仍然必须执行。

产物：

- `output/ep15_info_limit/m1_phase_structure/m1_phase_summary.csv`
- `output/ep15_info_limit/m1_phase_structure/m1_stage_lattice_occupancy_5x.png`
- `output/ep15_info_limit/m1_phase_structure/m1_detector_bin_occupancy_5x.png`
- `output/ep15_info_limit/m1_phase_structure/m1_phase_structure_summary.json`

## M2 结论（FRC 信息截止）

**结论**: 负面结果 / RISK。phase-stratified split-half FRC 的 1/7 截止周期为 **17.03 µm**（3 seeds std **0.50 µm**），half-bit 截止同为 **17.03 µm**；这大于 16 µm，说明实测可相干信息少于 11-14 µm 理论预期。10 µm 孔径零点没有形成可信下陷（10 µm FRC 反而约 0.935，dip margin = -0.390），因此本轮 FRC 不支持把 4x/5x 表达网格解释为 10-14 µm 真实物理信息。按该 cutoff 给后续训练的去卷积目标建议为 `sigma_target_02=0.486 LR px`（MTF=0.2）和 `sigma_target_03=0.421 LR px`（MTF=0.3）。

关键频带读数（主 FRC）：

| 周期 | FRC |
|---:|---:|
| 20 µm | 0.348 |
| 16 µm | 0.138 |
| 14 µm | 0.098 |
| 12 µm | 0.593 |
| 11 µm | 0.877 |
| 10 µm | 0.935 |
| 9 µm | 0.816 |
| 8 µm | 0.545 |

对照组没有完全符合预期，必须视为本次测量的风险信号而不是美化掉：bicubic 阳性对照没有表现为预期的更低频截止（以周期表示反而为 13.58 µm，小于主曲线 17.03 µm）；shift-shuffle 阴性对照在 8-12 µm 的 median FRC 为 0.504，没有崩到 0；按采集顺序前后半的 drift control cutoff 为 26.20 µm，明显差于分层版。结合 M1 中 `contour_refined` detector-axis 5x bin 只覆盖 11/25，以及本次 drizzle 平均 zero coverage 约 27.18%，高频回升更可能混入了 coverage/lattice artifacts 和热漂移，而不是稳健的物理孔径零点证据。

产物：

- `output/ep15_info_limit/m2_frc/frc_curve.png`
- `output/ep15_info_limit/m2_frc/frc_curve.csv`
- `output/ep15_info_limit/m2_frc/frc_controls.png`
- `output/ep15_info_limit/m2_frc/frc_summary.json`
- `output/ep15_info_limit/m2_frc/frc_band_table.csv`

## M3 结论（σ 仲裁）

**结论**: Route B 偏大的机制被支持。多边缘 ESF 显示 die 外边框的 apparent `sigma_total` 明显宽于内部强边缘：外边框 median **1.015 LR px**，内部金属强边缘 median **0.747 LR px**，最陡温度边缘 median **0.888 LR px**；外边框/内部边缘比值 **1.36**，差值 **0.267 LR px**。这说明 EP09 Route B 的 **1.129 LR px** 主要是在测 `系统 PSF ⊗ 热边缘宽度`，不是纯光学 PSF。若把最锐单边缘 `sigma_total=0.546 LR px` 当作光学上界，外边框还需要约 **0.855 LR px** 的热/几何边缘宽度才能解释。

FRC 形状交叉检验也没有支持 `sigma≈1.0 LR px` 的宽 PSF：在 12-80 µm 周期拟合 `Gaussian PSF × 10 µm detector aperture` 的 MTF² 形状时，最佳 grid 点为 **0.2 LR px**（MSE 0.044，corr 0.931），随后 0.3/0.4/0.5 LR px 逐步变差，0.7/1.0 LR px 明显更差。结合 M2 cutoff 给出的 `sigma_target_02=0.486`、`sigma_target_03=0.421`，后续 M4 的可信扫描区间采用 **0.2-0.5 LR px**。注意：ESF 最锐边缘上界 0.546 LR px 略高于该区间，说明“部分解释”仍比“完全定标”更准确；M4 的 σ 扫描已覆盖这一风险。

产物：

- `output/ep15_info_limit/m3_sigma/edge_comparison.png`
- `output/ep15_info_limit/m3_sigma/frc_shape_fit.png`
- `output/ep15_info_limit/m3_sigma/sigma_summary.json`
- `output/ep15_info_limit/m3_sigma/edge_fit_table.csv`
- `output/ep15_info_limit/m3_sigma/edge_summary.csv`
- `output/ep15_info_limit/m3_sigma/frc_shape_fit_scores.csv`

## M4 结论（去卷积锚）

（由下游 agent 回填）
