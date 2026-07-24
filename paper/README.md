# Thermal Lift 论文工作区

## 权威来源声明（2026-07 收尾口径）

论文资产的权威来源只有两处：

- **图**：`docs/publication_figures/` — 72 张出版级结果图（GALLERY.md 叙事图册 + `scripts/` 每图一个可复现脚本），覆盖 ACL-023 → ACL-080 时代的全部判决性结果。
- **正式报告**：`paper/reports/` — 各 Episode 的正式分析报告（.md）。

> **`docs/paper/` 已于 2026-07-24 删除。** 该目录是 2026-06 时代的论文 markdown 草稿（曾作为 aaai/zh_conf 的 source of truth），口径已过时，不再引用；如需考古可从 git 历史找回。

## 目录内容

| 路径 | 状态 | 说明 |
|---|---|---|
| `reports/` | ✅ 权威 | 正式分析报告，随项目持续维护 |
| `aaai/` | ⚠️ 2026-06 骨架 | AAAI 风格英文论文骨架（`pdflatex`），使用 AAAI-26 author kit；未跟进 7 月结果，仅供起稿参考 |
| `zh_conf/` | ⚠️ 2026-06 骨架 | 中文双栏会议骨架（`xelatex` + `ctex`）；同上，仅供参考 |
| `slides/` | ⚠️ 2026-06 骨架 | Beamer 报告骨架；同上，仅供参考 |
| `scripts/` `figures/` `tables/` `bib/` `asset_manifest.json` | 工具 | 从 `output/` 导出稳定图表资产的管线 |

真正撰稿时，图直接取自 `docs/publication_figures/figures/`（PNG 预览 + PDF 投稿矢量），数值结论按 ACL 编号溯源到 `research_log/algorithm_changelog.md`。

## 构建

导出 `output/` 中的稳定图表资产：

```bash
uv run python paper/scripts/export_assets.py
```

构建 AAAI 骨架：

```bash
cd paper/aaai
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

构建中文双栏骨架：

```bash
make -C paper zh_conf
```

构建 Beamer 报告：

```bash
cd paper/slides
pdflatex main.tex && pdflatex main.tex
```

`aaai/template/` 仅保留本地 smoke test 所需的 LaTeX style 文件；正式投稿前须替换为目标会议年份的官方 author kit。

## 写作口径（硬边界）

- 当前声明是 **2x 轮廓级（contour-level）热成像 SR POC**。数据口径：原始主扫描 session（session=2）共 **255 帧**，SR 默认输入为剔除 `R != 0` 重复/补采帧后的 **248 帧 clean set**——不要写成"255-frame main session 上的 SR"。
- **不许**写成 4x SR 结果、5 µm 空间分辨率声明、或绝对温度计量结论。
- stage command 与文件名推导的位移只是 prior / 对照，**不是对齐真值**。
- 对神经方法只采信"对独立参照（drizzle）的 cross-FRC + 逐半偏移校正"，自分半 FRC 无效；20 µm 空间周期是探测器孔径零点，该处 FRC 数值一律不采信（依据见 `docs/publication_figures/GALLERY.md` 第二章）。
