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
# ### 📝 定位精度与形状重建评价的逻辑边界
#
# 上述边界对照表明确了 EP04 局域边缘定位基准（Localization Benchmark）与 EP06 全局形状超分辨率重建（Shape Reconstruction）之间的物理逻辑分工，划定了指标的外推红线：
# 1. **定位精度指标限定**：以折半估计为代表的局域定位精度，反映的是局部热边缘在亚像素物理尺度下的重复再现不确定度，用于充当对齐计算和测试集（Held-out QC）的安全闸门，不能代表超分辨率后热图图像的绝对空间清晰度或测温精度。
# 2. **重构增益的评判**：超分辨率算法的目标是提高内部芯片结构的轮廓可见性。即使某区域因为没有通过定位门控（Rejection），也只说明它不适合作为对齐控制锚点，该区域仍然可在超分辨率优化迭代中被高保真恢复。
#
# **💡 算法决策**：定位门控只作为几何配准的前置过滤器。后续 2x contour-level 超分辨率重建（EP06）必须基于此逻辑分工，对对齐品质与图像重建质量引入独立评估，不能以偏概全。
