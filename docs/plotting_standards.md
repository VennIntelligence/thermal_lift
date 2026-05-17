# Plotting Standards

本项目所有 Matplotlib/Plotly 图表必须遵守以下统一标准，以满足 CVPR 顶会论文投稿要求和研究 notebook 的展示质量。

## Font Policy

**所有图表使用衬线字体（serif），与 CVPR 论文正文保持一致。**

CVPR 模板正文使用 Times New Roman，图表文字也使用 Times New Roman 可以保证全文视觉统一。衬线字体在学术出版中有深厚传统，在论文级别的字号（≥ 7pt）下可读性完全没有问题。

### 字体优先级

```python
FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
```

### 数学公式字体

数学模式使用 `stix` 或 `cm`（Computer Modern），与 LaTeX 论文正文数学符号风格一致：

```python
"mathtext.fontset": "stix"
```

## Global rcParams

所有可视化模块必须通过统一的 `setup_academic_style()` 函数设置以下参数。**禁止在单�� figure 函数内硬编码字体大小。**

```python
ACADEMIC_RCPARAMS = {
    # ── Font ──
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size":         9,
    "mathtext.fontset":  "stix",

    # ── Axes ──
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.labelweight":  "normal",
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,

    # ── Ticks ──
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,

    # ── Legend ──
    "legend.fontsize":      8,
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.5,
    "legend.borderpad":     0.4,

    # ── Lines ──
    "lines.linewidth":  1.4,
    "lines.markersize": 5,

    # ── Grid ──
    "axes.grid":          False,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "grid.color":         "#cccccc",

    # ── Figure ──
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "figure.constrained_layout.use": True,

    # ── Saving ──
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor":  "white",
}
```

## Figure Sizing

### 单位：英寸，以 CVPR 双栏论文为基准

| Layout         | Width (in) | Height range (in) | 用途                         |
|---------------|------------|-------------------|------------------------------|
| **单栏**       | 3.5        | 2.0 – 3.5         | 单方法对比、小 heatmap         |
| **1.5 栏**     | 5.5        | 2.5 – 4.5         | 分组 bar、scatter、双面板       |
| **双栏**       | 7.2        | 2.5 – 5.5         | dashboard、多面板比较          |
| **notebook 展示** | 7.2 – 8.0 | 按内容调整         | 本地展示用，不直接用于论文       |

CVPR 顶会通用规则：

- **最终论文 figure 宽度不超过 7.1 英寸**（双栏全宽）。
- 本地 notebook 展示可适当放大，但所有比例和字号必须在 3.5 英寸宽度下仍然可辨认。
- 高度以「内容呼吸空间充足、不挤压」为准；若 subplot 行数 > 2，宁可加大高度也不要让 tick label 叠字。

## Colour Policy

### 主色板（≤ 6 种方法对比时使用）

```python
METHOD_PALETTE = {
    "primary":    "#4C72B0",   # steel blue
    "secondary":  "#55A868",   # muted green
    "accent_1":   "#C44E52",   # soft red
    "accent_2":   "#8172B2",   # lavender purple
    "accent_3":   "#DD8452",   # warm orange
    "neutral":    "#937860",   # taupe
}
```

### 热力/残差 colormap

| 场景                      | Colormap     | 说明                                  |
|--------------------------|-------------|---------------------------------------|
| 温度场 / HR image          | `"inferno"`  | 高对比、色盲友好                        |
| 残差 heatmap (单侧 ≥ 0)    | `"YlOrRd"`   | 零值亮，高值暖红                        |
| 残差 diff (双侧 ± )        | `"RdBu_r"`   | 对称发散色，零值白色                     |
| coverage / count           | `"viridis"`  | 感知均匀，打印友好                       |

### 色盲安全

- 禁止仅用红-绿对比来区分数据；必须辅以 marker 形状或 line style 区分。
- 如需 > 6 色，优先使用 `"tab10"` 子集或 Seaborn `"colorblind"` 色板。

## Label & Text Hierarchy

### 字号层级（固定值，不得随意偏离）

| 层级             | ��小 (pt) | 权重     | 用途                   |
|-----------------|----------|---------|------------------------|
| `suptitle`       | 11       | bold    | 仅多面板 figure 的总标题   |
| `ax.set_title`   | 10       | bold    | 单 axes 标题              |
| `ax.set_xlabel/ylabel` | 9  | normal  | 坐标轴标签               |
| tick labels      | 8        | normal  | 坐标轴刻度值              |
| legend text      | 8        | normal  | 图例文字                  |
| annotation text  | 7 – 8    | normal  | heatmap 内数字、bar 顶端值  |
| colorbar label   | 8        | normal  | colorbar 侧标注           |

### 物理量标注规范

- 温度单位始终用 `[$^\circ\mathrm{C}$]` 或 `[°C]`。
- 坐标单位用 `[$\mu$m]`。
- 百分比残差用 `[%]`。
- 物理量 label 始终用「描述 + 单位」格式，例如 `"Mean absolute residual [°C]"`。
- **不要在 label 里放段落式长文**：axis label 不超过 50 个字符；需要补充说明的信息放在 title 或 annotation 中。

## Layout & Readability Rules

### 间距与防重叠

1. **子图间距**：优先使用 `constrained_layout=True` 或 `fig.tight_layout(pad=1.2)`。
2. **tick label 不叠字**：如果 x 轴 tick 过密，使用 `rotation=45, ha="right"` 或减少 tick 数（`MaxNLocator`）。
3. **legend 不遮��数据**：
   - 优先用 `bbox_to_anchor=(0.5, -0.15)` + `loc="upper center"` 放在图下方。
   - 图例 entry ≤ 4 个可尝试放在图内空白区域。
   - **永远不要让 legend 覆盖数据点或曲线。**
4. **colorbar 不挤压主图**：使用 `fraction=0.046, pad=0.04` 或 `ax_divider` 来保持主图面积。
5. **annotation 不相互覆盖**：bar 顶端值标注超过 8 个时考虑只标注极值；heatmap 内标注仅在 grid < 20×20 时使用。

### Spine 和 Grid

- 默认 **关闭** top + right spines（已在 rcParams 中设置）。
- Grid 仅在确实帮助读数的场景启用（如 bar chart 的 y 轴 grid）：
  ```python
  ax.grid(axis="y", alpha=0.3, linewidth=0.5)
  ```
- Heatmap / imshow 类图不加 grid。

### 参考线

- 阈值参考线使用 `ls="--"` 黑色 (`#222222`) 或灰色 (`#666666`)，线宽 `0.8 – 1.0`。
- 参考线必须有对应 legend entry 或 annotation 标注其含义。

## Saving Conventions

```python
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
```

- **dpi=300** 是论文提交最低要求。notebook 内预览可用 `dpi=200`。
- **bbox_inches="tight"** 避免裁掉外置 legend 和 suptitle。
- **facecolor="white"** 避免透明背景导致 PDF 渲染问题。
- 保存格式：notebook 展示用 `.png`；论文���稿用 `.pdf`（矢量）。

## Plotly / Interactive Figures

如果 notebook 使用 Plotly，template 需对齐以下约定：

```python
cvpr_template = go.layout.Template()
cvpr_template.layout.font = dict(
    family="Times New Roman, Times, serif",
    size=13,
    color="black",
)
cvpr_template.layout.plot_bgcolor  = "white"
cvpr_template.layout.paper_bgcolor = "white"
cvpr_template.layout.xaxis = dict(
    showline=True, linewidth=1.5, linecolor="black",
    mirror=True, showgrid=False, ticks="outside",
)
cvpr_template.layout.yaxis = dict(
    showline=True, linewidth=1.5, linecolor="black",
    mirror=True, showgrid=True, gridcolor="#e0e0e0",
    gridwidth=1, ticks="outside",
)
```

## Anti-Patterns（禁止项）

| ❌ 禁止                                 | ✅ 应改为                                     |
|-----------------------------------------|----------------------------------------------|
| `font.family: sans-serif / Arial`       | `font.family: serif / Times New Roman`        |
| 在 figure 函数里写 `fontsize=12`         | 统一由 rcParams 控制                           |
| `fig.tight_layout()` + `constrained_layout=True` 混用 | 选其一，推荐 `constrained_layout`    |
| `savefig(dpi=180)` 或更低               | `savefig(dpi=300)`                            |
| ��轴标签里写中文                         | 图表内容全部英文；中文仅用于 notebook markdown |
| `plt.show()` 在构建脚本里               | 在构建脚本中只 `savefig` + `plt.close(fig)`    |
| legend 覆盖数据区域                      | legend 放图外或空白角落                         |
| 使用 `"jet"` colormap                   | 使用 `"viridis"` / `"inferno"` / `"YlOrRd"`   |
| heatmap 中 annotation 遮挡颜色信息       | grid > 20×20 时移除 text annotation             |
| 图中只用颜色区分类别                      | 同时用 marker 形状或 linestyle 辅助区分          |

## Checklist（每张 figure 提交前必须过的检查点）

- [ ] 字体为 serif (Times New Roman)，衬线字体与正文一致。
- [ ] 所有 axis label 和 title 使用英文。
- [ ] 所有物理量标注包含单位。
- [ ] tick label 无重叠。
- [ ] legend 不覆盖数据。
- [ ] colorbar label 有描述和单位。
- [ ] 无 `"jet"` colormap。
- [ ] dpi ≥ 300。
- [ ] 导出白色背景。
- [ ] 在 3.5 英寸宽度下文字仍可阅读。
