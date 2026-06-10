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

（由下游 agent 回填）

## M3 结论（σ 仲裁）

（由下游 agent 回填）

## M4 结论（去卷积锚）

（由下游 agent 回填）
