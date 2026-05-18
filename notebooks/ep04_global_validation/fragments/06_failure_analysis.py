# %% [markdown]
# ## Step 5 — Localization Precision 与 Shape Reconstruction 的边界
#
# EP04 可以证明局部热边缘 anchor 的 repeatability 和 CRB consistency；它不能替代 EP06 对内部结构轮廓是否更清楚、更稳定的评价。

# %% [markdown]
# | 评价对象 | Evaluation target | 本 Episode 测量什么 | Measured here | 在 EP06 中怎么用 | Use in EP06 | 不能声称什么 / Not a claim |
# |---|---|---|---|---|---|---|
# | EP04 localization precision | EP04 localization precision | split-half 边缘位置、CRB ratio、NCC phase coverage | split-half edge position, CRB ratio, NCC phase coverage | alignment anchors、frame/segment gates、held-out QC | alignment anchors, frame/segment gates, held-out QC | 不是 dense contour SR，也不是 metrology-grade 5 um 温度恢复 |
# | EP06 shape reconstruction | EP06 shape reconstruction | 这里只测 anchor availability 与 failure modes | only anchor availability and failure modes | 面向内部轮廓，评价 LR/bicubic/SR 的 shape stability | target internal contours, evaluate LR/bicubic/SR shape stability | 不能仅因 anchor rejection 判定 shape reconstruction 失败 |

# %% [markdown]
# > **数据说明**: 表格把 EP04 localization benchmark 与 EP06 shape reconstruction 目标分开，列出各自测量量、用途和不能声称的内容。
# > **读法**: 逐列看“测量对象、用途、不能声称什么”。EP04 关注的是局部边缘点或短 segment 能否被稳定定位；EP06 关注的是重建后内部结构轮廓是否比 LR/bicubic 更清楚、更稳定。
# > **正常/异常理解**: 正常解释是：高质量 localization anchor 可用于配准和 holdout 检查，但不能直接推出整幅图的内部结构已经被正确重建。异常解释是把 pass rate 当作 SR 成功率，或把 fail 段当作不存在的结构；这两种解读都超出 EP04 的证据范围。
# > **对本 Episode 的意义**: 定位精度是配准支撑，不是最终交付；pass/reject rate 只能解释为 anchor 质量门控覆盖率。Chamfer、NCC、curvature proxy 等 proxy 如果在后续分析中出现，也只能作为红外内部一致性或几何近似指标，不能替代未配准的光学真值。
