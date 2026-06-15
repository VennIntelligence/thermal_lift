# 中文双栏会议模板（zh_conf）

仿 CCF/学会论文排版：**宋体正文 + 宋体加粗标题 + 楷体作者/图题表题 + Times 系英文**，字体来自 CTAN [Fandol](https://ctan.org/pkg/fandol) 开源包。

## 字体方案

| 元素 | 字体 | 说明 |
|---|---|---|
| 主标题 | FandolSong 宋体 | 中文标题；英文术语用 `\eng{LWIR}` 等 → Times |
| 节标题 / 强调 | FandolSong-Bold | 宋体加粗（与正文同族） |
| 摘要标题 | FandolSong-Bold，`\large` 居中 | 跨栏摘要块顶部的「摘  要」 |
| 中文正文 | FandolSong（宋体） | 全文主字体；**段首缩进 2em（两字）** |
| 图题 / 表题 | FandolKai（楷体）`\small` | 标签与说明全文楷体；标签加粗 |
| 作者 | FandolKai（楷体） | 与正文宋体区分层级 |
| 引用 | GB/T 7714 顺序编码 | 正文 `[1]`，文末「参考文献」 |

首次构建会自动下载字体（约 34 MB）：

```bash
make -C paper zh_conf
```

或手动：

```bash
bash paper/zh_conf/scripts/setup_fonts.sh
cd paper/zh_conf && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

## 与英文流的关系

| 流 | 目录 | 编译 | 内容来源 |
|---|---|---|---|
| 英文 | `paper/aaai/` | `pdflatex` | 中文稿译英 |
| 中文 | `paper/zh_conf/` | `xelatex` | 中文稿直接迁 tex |

共享 `paper/bib/refs.bib`、图、表。

## 引用格式

正文 `\cite{key}` → **GB/T 7714 顺序编码** `[1]`、`[2-3]`：参考文献表按正文**首次引用顺序**编号（已 patch `gbt7714-2015-numeric.bst` 的 `presort`，不再按作者字母排序）。

## 英文混排

`\eng{LWIR}` 包裹的英文走 Times；中文保持宋体。

若已安装 `texlive-lang-chinese`（`sudo apt install texlive-lang-chinese`），可将
`\documentclass` 换为 `ctexart` 并启用 `fontset=fandol`，排版会更稳。当前方案**不依赖**系统 ctex，开箱即用。
