# EP12 4x — UNet@2000 vs Bare Drizzle Benchmark

## Goal

Compare the EP12 drizzle-informed 4x UNet step-2000 checkpoint against bare tcforge
scatter-add drizzle mean on the real 248 clean main-session frames.

## Inputs

- Raw input: EP06 clean main session, 248 frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `algos/ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt`.
- Baseline: bare drizzle mean (`drizzle_features` channel 0).
- Highpass sigma: `5.0`.

## Artifacts

- Script: `algos/ep12_4x_benchmark/scripts/run_ep12_vs_drizzle_4x.py`.
- Notebook: `notebooks/ep12_4x_benchmark/`.
- Output: `output/ep12_4x_benchmark/`.

## Boundary

Contour-level visual benchmark only. Checkpoint is synthetic-pretrained at step 2000;
real-data gains carry domain-gap risk. 3x is display zoom; reconstruction grid is 4x.

After M1-M4 and the EP07x2up vs EP12 four-arm gate, 4x is no longer treated as
evidence for recoverable 10-14 um information. The only acceptable 4x claim is
learning-based regularization on a finer presentation grid: smoother contour
localization with lower artifacts. A 4x checkpoint must simultaneously satisfy:

1. FRC consistency at 20/16/14/12 um bands is not worse than the M4 MAP-TV anchor.
2. Center zigzag FWHM / dip metrics are not worse than the M4 MAP-TV anchor.
3. Artifact score and contour quality are not worse than EP07 2x x2up.

If any gate fails, the 4x branch is not adopted; the delivery fallback is EP07
2x plus MAP-TV post-processed upsampling.

## EP07 2x x2up vs EP12 4x Gate — 2026-06-10

### Inputs

- EP07 2x model: `algos/ep07_unet_sr/outputs/ep07_v6_physics/model_final.pt` (step 60000).
- EP12 4x checkpoint: `algos/ep12_4x_sr/outputs/ep12_hybrid_v1/checkpoint_step_048000.pt`.
- Real data: EP06 clean main session, 248 frames.
- Alignment: `contour_refined` shifts.

### Command

```bash
cd algos/ep12_4x_benchmark
uv run python scripts/run_ep07x2up_vs_ep12_4x.py --device cuda:1
```

### Outputs

- Output directory: `output/ep12_4x_benchmark/ep07x2up_vs_ep12/`.
- Temperature comparison: `ep07x2up_vs_ep12_center_zoom3x_temperature.png`.
- Highpass comparison: `ep07x2up_vs_ep12_center_zoom3x_highpass.png`.
- Center zigzag ROI temperature: `ep07x2up_vs_ep12_zigzag_roi_temperature.png`.
- Center zigzag ROI highpass: `ep07x2up_vs_ep12_zigzag_roi_highpass.png`.
- Metrics CSV: `metrics_summary.csv`.

### Visual Conclusion

EP12 48k 4x does not show a visible gain over EP07 2x x2up in contour clarity or center zigzag-line separability. EP07x2up is sharper but has stronger highpass over/undershoot; EP12 is smoother and closer to bare drizzle, while proxy metrics also do not support EP12 as the better arm.

## 最终结论

- 四臂 gate 判负：EP12 48k 4x 相对 EP07 2x x2up 在轮廓清晰度和中心 zigzag 线分离度上无可见增益，proxy 指标亦不支持 EP12 为更优臂；4x 分支未通过采纳门槛，交付 fallback = EP07 2x + MAP-TV 后处理上采样。（出处: 本 README "EP07 2x x2up vs EP12 4x Gate" 与 Boundary 节）
- EP15 M1–M4 判决后，4x 不再作为可恢复 10–14 µm 信息的证据：M2 phase-stratified split-half FRC 截止 17.03 µm（>16 µm），高频回升被判为 coverage/lattice 伪影与热漂移混入；4x 唯一可接受的解释是"更细呈现网格上的学习型正则化"。（出处: `research_log/episodes/ep15_info_limit/README.md` M2 结论）
- 项目最终倍率口径：2x 是当前数据的合理倍率——实测相位占用 11/25 detector bin，>2x 相位饥饿；此后主线（unrolled solver、champion depb9v6）全部锁定 2x。（出处: `docs/publication_figures/GALLERY.md` 头条成果 #9 / EP15-M1）
