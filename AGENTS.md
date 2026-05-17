# Thermal Lift — 智能体工作规范

> 本文件是项目的持久记忆文件，所有参与本项目的智能体必须先读取本文件。

---

## 🔴 输出安全规则

**如果你的单次输出可能超过 16K token，必须采用「分段追加写入」模式：**

1. 预估输出总长度
2. 如果可能超过 15K token，将输出拆分为 ≤12K token 的段落
3. 第一段用 `write_to_file` 创建文件
4. 后续段用 `replace_file_content` 或追加方式写入
5. **绝不允许因为输出截断而丢失内容**

---

## 📁 项目概述

**项目名称**: thermal_lift — 红外热像微扫描超分辨率重建  
**目标**: 从 263 帧 LWIR 亚像素微扫描热像中重建高分辨率温度场  
**前身**: 旧项目经历 19 个 Episode / ~4 天密集工作，已蒸馏结论到 `fresh_start_guide.md`

---

## 🎯 项目目标

### 核心问题

开发一套**红外热像超分辨率算法**，利用亚像素微扫描技术，将 LWIR 红外相机的空间分辨率从 **20 µm 提升到 5 µm**，使之能够分辨微小结构（~5 µm）的红外温度信号。

### 关键定义澄清

| 问题 | 答案 |
|------|------|
| 20 µm 是什么？ | **空间分辨率**（已校准），不是像素物理尺寸 |
| 5 µm 目标是什么？ | **目标空间分辨率**：希望能看清 5 µm 结构的热信号（目前达不到） |
| 波段 | LWIR 长波红外 8–14 µm |
| 数据类型 | 有 raw 温度矩阵数据（TXT），非伪彩色视频 |
| 光学-红外配准 | 光学显微和红外相机**未集成**，无配准 |
| 核心任务 | 开发算法，**测量微小结构的红外温度信号** |
| 验收标准 | 暂无量化验收标准 |
| 现有最优结果 | ~10 µm（使用更高分辨率微距镜头，但效果不佳） |

### 技术路线

通过**亚像素微扫描 + 超分辨率重建**，而非更换昂贵硬件（微距镜头），在软件算法层面突破衍射/采样极限：

1. **数据采集**: 用电动台以已知亚像素步进（2–40 µm）移动样品，获取 263 帧同一视场在不同亚像素偏移下的热像
2. **位移建模**: 将电动台坐标通过旋转角 θ 映射为像素级亚像素位移
3. **超分辨率重建**: 利用多帧亚像素位移信息，在 Nyquist 极限附近恢复高频细节
4. **目标**: 20 µm → 5 µm（4× 分辨率提升）

### 挑战与约束

- **衍射极限**: LWIR 8–14 µm 波长本身限制了光学系统的理论分辨率
- **PSF 衰减**: 光学 PSF 将高频信号衰减到噪声以下，理论上限需要 MTF/SNR 分析
- **噪声底**: 0.0724°C 的探测器噪声决定了可恢复信息的上限
- **位移范围有限**: 总位移仅 ~1.35 pixel，限制了可达到的超分辨倍数

---

## 📐 已确认的物理常数（Ground Truth）

| 参数 | 值 | 备注 |
|---|---|---|
| 探测器输出尺寸 | 640×480 pixels | EP01 实测: 480行×640列 |
| 当前空间分辨率 | 20 μm | 校准得到 |
| 波段 | LWIR 8–14 μm | |
| 电动台-像素旋转角 θ | 47.6° | 0.1° 精度 |
| 光学 PSF | Gaussian σ ≈ 0.5 px | 可能在 0.2–0.5 区间 |
| Noise floor | 0.0724°C | smooth adjacent-coord MAE |
| 坐标集合 (X, Y) | {0,2,4,6,8,10,12,14,16,18,20,24,28,32,36,40} μm | 16 值 |
| 每条扫描线 | 16 个采样点 | |
| 总位移范围 | ~1.35 pixel（沿扫描轴投影后） | |

---

## 📏 坐标模型

```python
import numpy as np
THETA_DEG = 47.6
PIXEL_SIZE_UM = 20.0

def coordinate_to_shift(x_um, y_um):
    theta = np.radians(THETA_DEG)
    dx = (x_um * np.cos(theta) + y_um * np.sin(theta)) / PIXEL_SIZE_UM
    dy = (-x_um * np.sin(theta) + y_um * np.cos(theta)) / PIXEL_SIZE_UM
    return dx, dy
```

---

## 🔴 硬教训（绝对不能重蹈的覆辙）

1. **scale=10 是纯插值，不是超分辨率** — PSF 将 2× Nyquist 以上信号衰减到 <1%
2. **回投残差 ≠ 清晰度** — 不能用 back-projection residual 判断 SR 是否成功
3. **AVI 视频不适合做 SR** — 帧间位移是间歇性大运动 (p95=10.4 px)
4. **跨 session 帧绝不能混合使用** — 跨 session 温度跳变中位 3.55°C（49× noise）
5. **旋转角必须第一天标定** — 旧项目前 6 个 Episode 都用错了位移模型
6. **先用第一性原理（MTF/SNR）算理论极限，再做实验**
7. **重复测量差异很大（1-2°C std）** — 不应跨 repeat 混合
8. **Tenengrad 锐度指标不能单独作为 SR 成功证据**
9. **Session 检测必须使用采集顺序 (`acquisition_order`/mtime)** — 按重命名后文件名字母序会制造 13 个假 session；真实采集顺序为 3 个温度段，主扫描 session=2（255 帧）
10. **图片双重显示的根因与正确模式** — `%matplotlib inline` 下存在**两套独立显示机制**同时生效：① Jupyter 对 cell 最后表达式调 `fig._repr_png_()`（cell output）；② matplotlib `post_execute` hook 对 Gcf 中所有未关闭 figure 调 `flush_figures()`。如果 figure 同时满足「是 cell 返回值」和「留在 Gcf 中」，就会显示两遍。**正确模式**: `save_fig` 使用 `savefig_academic(fig, path)` (默认 `close=True`) 保存并从 Gcf 注销 figure，然后 `return fig`；由 Jupyter cell output 机制提供唯一一次显示（`plt.close()` 只注销 Gcf，Figure 对象的 `_repr_png_()` 仍可用）。**绝不在 `save_fig` 中使用 `close=False`，也不要在 `save_fig` 后再调 `plt.show()`**

---

## 📂 项目结构约定

```
thermal_lift/
├── AGENTS.md                  ← 本文件（持久记忆）       [Git ✅]
├── fresh_start_guide.md       ← 旧项目蒸馏文档           [Git ✅]
├── .gitignore                 ← Git 忽略规则             [Git ✅]
├── pyproject.toml             ← UV 项目配置              [Git ✅]
├── uv.lock                    ← 依赖锁文件               [Git ✅]
├── data/                      ← 原始数据                 [Git ❌ 整体 ignore]
│   ├── data_raw/
│   │   ├── infrared_avi/      ← TXT + BMP + AVI
│   │   ├── optical_fig/       ← 光学参考图
│   │   └── name_rules.txt
│   ├── processed/             ← 预处理中间产物
│   ├── sample/                ← 样本数据
│   └── synthetic/             ← 合成验证数据
├── core/                      ← 共享核心库               [Git ✅]
├── algos/                     ← 算法实现                 [Git ✅ 代码, ❌ .venv]
├── experiments/               ← 实验管理                 [Git ✅]
├── notebooks/                 ← Jupyter 可视化           [Git ✅ fragments/, ❌ .ipynb]
├── output/                    ← 数据产物（CSV、图表）    [Git ❌ 整体 ignore]
├── scripts/                   ← 构建 & 工具脚本          [Git ✅]
├── configs/                   ← 全局物理参数配置         [Git ✅]
├── docs/                      ← 文档                     [Git ✅]
├── reports/                   ← 正式分析报告（.md）      [Git ✅]
├── research_log/              ← 研究日志                 [Git ✅]
└── paper/                     ← 论文                     [Git ✅]
```

### UV 环境管理

**根目录 UV 项目**（`pyproject.toml` 在项目根）:
- 用于全局简单工作: 重命名脚本、可视化、数据审计、Jupyter Notebook
- 依赖: numpy, matplotlib, pandas, scipy, tqdm, jupyter
- 运行方式: `uv run python scripts/xxx.py` 或 `uv run jupyter notebook`

**算法隔离规则**:
- **每个 `algos/xxx/` 是完全独立的项目**，有自己的 `pyproject.toml`、`.python-version`、独立 UV venv
- 不使用 UV workspace — 因为第三方算法可能要求不同 Python 版本
- 共享代码通过 `core/` 包以 `pip install -e ../core` 方式安装到各 venv
- 算法之间**零耦合**，可以独立运行、独立删除

### Git 可复现规则

**原则: Git 只跟踪「源码」，不跟踪「产物」和「数据」。**

| 类别 | Git 策略 | 重建方式 |
|------|----------|----------|
| `data/` | ❌ 整体 ignore | 手动拷贝到新机器 |
| `output/` | ❌ 整体 ignore | notebook 执行时自动生成 |
| `notebooks/**/*.ipynb` | ❌ ignore | `build_all_notebooks.py` 从 fragments/ 重建 |
| `notebooks/**/fragments/` | ✅ track | **这是 notebook 的源码** |
| `.venv/`, `algos/*/.venv/` | ❌ ignore | `uv sync` 重建 |
| `uv.lock` | ��� track | 保证依赖精确可复现 |
| `core/`, `scripts/`, `configs/` | ✅ track | 项目核心代码 |
| `reports/` | ✅ track | 正式 .md 分析报告 |
| `research_log/` | ✅ track | 研究日志和 Episode 进度 |

### 🚀 新机器部署（Git 迁移）

**前提**: 已安装 [uv](https://docs.astral.sh/uv/)，已获取原始数据文件。

```bash
# 1. 克隆仓库
git clone <repo-url> thermal_lift && cd thermal_lift

# 2. 放置数据 — 将原始数据拷贝到 data/ 目录，结构如下:
#    data/
#    ├── data_raw/
#    │   ├── infrared_avi/    ← TXT + BMP + AVI 原始文件
#    │   ├── optical_fig/     ← 光学参考图
#    │   └── name_rules.txt
#    ├── processed/           ← 留空
#    ├── sample/              ← 可选
#    └── synthetic/           ← 留空

# 3. 安装依赖
uv sync

# 4. 安装 core 包（editable 模式）
uv pip install -e core/

# 5. 一键构建所有 Notebook（构建 + 执行，生成 .ipynb 和 output/）
uv run python scripts/build_all_notebooks.py --execute

# 6. 验证 — 打开任意 notebook，应看到完整执行结果（含图片）
```

---

## 📛 数据命名规则

### 原始格式（连写数字）

```
格式: X + Y + R（紧密拼接，X/Y 各 1-2 位，R = 0/1/2）
最右 1 位 = R (重复编号)
左起第 2~3 位 = Y (μm 坐标, 1-2 位)
剩余最左 = X (μm 坐标, 1-2 位)

特殊: 2400 → X=24, Y=0, R=0（不是 X=2, Y=40）
      2，400 → X=2, Y=40, R=0（中文逗号标注歧义）
```

### 目标格式

```
X_Y_R.ext  （例: 2_10_0.txt, 24_0_0.bmp）
```

### 解码消歧策略

X 和 Y 只能取 `{0,2,4,6,8,10,12,14,16,18,20,24,28,32,36,40}`，R 只能取 `{0,1,2}`。  
按前导零约束枚举合法 (X, Y, R) 分割，保留唯一合法解。
实际脚本还必须遵守前导零和明确特判：`0200 → X=0,Y=20,R=0`，`2000 → X=20,Y=0,R=0`，`2400 → X=24,Y=0,R=0`，`2，400 → X=2,Y=40,R=0`。

### 采集顺序规则

`X_Y_R` 文件名是坐标标识，不是采集时序。所有 session 检测、时间线、帧对选择必须使用 `frame_audit.csv` 中的 `acquisition_order`（由文件 mtime 生成），不能按文件名字母序推断采集顺序。

---

## 🎨 绘图规范

**所有 Matplotlib/Plotly 图表必须遵循 CVPR 顶刊风格（Times New Roman 衬线字体、300 dpi、白底）。**

- 完整规范: → `docs/plotting_standards.md`
- Python 模块: → `from thermal_core.plotting import setup_academic_style, savefig_academic`
- **绝不使用** sans-serif 字体、`"jet"` colormap、`dpi < 300`

---

## 📓 Notebook 管理规范

**绝不直接创建或编辑 `.ipynb` 文件。** Notebook 通过「分段片段 + 构建脚本」模式管理。

### 目录结构

**片段脚本必须放在 `fragments/` 子目录内**，与构建产物 `.ipynb` 分层隔离：

```
notebooks/epXX_name/
├── fragments/                 ← 所有 .py 片段在此
│   ├── manifest.txt           ← 片段顺序（可选，缺省按文件名排序）
│   ├── 01_setup.py            ← 片段：导入、路径、配置
│   ├── 02_analysis_step.py    ← 片段：分析步骤
│   └── ...
└── epXX_name.ipynb            ← 构建产物（不要手动编辑）
```

### 环境声明（必须）

**每个 notebook 的 `01_setup.py` 第一个 Markdown cell 必须包含运行环境说明**：

- 使用哪个 UV 环境（根目录 or `algos/xxx/`）
- 安装命令（`uv sync` or `uv pip install -e ...`）
- Kernel 选择提示

这是为了未来不同 notebook 可能使用不同算法环境时，用户能明确知道该如何启动。

### 片段格式

每个 `.py` 文件使用 jupytext percent 格式的 cell 标记：

```python
# %% [markdown]
# ## 标题
# 说明文字

# %%
import numpy as np
# ... code ...
```

### 构建命令

```bash
# 单个 notebook — 仅构建（拼接片段 → .ipynb，不执行）
uv run python scripts/build_notebook.py notebooks/epXX_name

# 单个 notebook — 构建 + 执行（推荐）
uv run python scripts/build_notebook.py notebooks/epXX_name --execute

# 🔥 一键构建所有 notebook（Git 迁移后首选）
uv run python scripts/build_all_notebooks.py --execute
```

**必须使用 `--execute`**：构建后的 notebook 应该是**完全执行好的状态** —
用户打开即可看到全部文字输出和图片，无需自己重新运行。
用户如果想验证，也可以 Restart & Run All 从头独立运行。

### Notebook 内容展示原则

**Notebook 是「报告」，不是「脚本」。** 读者打开 Notebook 应该看到数据洞察和可视化，而不是大段裸处理代码。

1. **以数据和核心发现为中心**
   - 每个分析片段开头用 Markdown cell 说明**目的和结论**
   - Cell 输出应该是图表、关键指标、或简洁的汇总信息
   - 不要在 Notebook 里堆砌数据清洗/IO/解析的裸代码

2. **隐藏实现细节，只露调用层**
   - 数据加载、解析、转换等通用逻辑应提取到 `core/src/thermal_core/` 中
   - Notebook cell 只保留**一行调用 + 结果展示**，例如：
     ```python
     df = load_frame_metadata(DATA_DIR)  # core 封装
     plot_coverage_heatmap(df)            # core 封装
     ```
   - 目标：片段 `.py` 尽量 < 30 行，大部分是 Markdown + 调用 + 绘图

3. **优先使用图片而非表格**
   - 当同一数据既可用表格也可用图片表达时，**优先使用图片**（heatmap / bar chart / histogram）
   - 表格仅用于：精确数值查阅、少量行的汇总统计、需要复制粘贴的数据

4. **图片显示：一图只显示一次**
   - `%matplotlib inline` 下有**两套显示机制**会同时触发：Jupyter cell output（对 cell 最后表达式调 `_repr_png_()`）和 matplotlib `post_execute` hook（对 Gcf 中未关闭 figure 调 `flush_figures()`）
   - **双重显示的根因**: figure 同时「是 cell 返回值」且「留在 Gcf 中」
   - **正确模式 — `save_fig` 标准实现**:
     ```python
     def save_fig(fig, name):
         savefig_academic(fig, OUTPUT_DIR / name)  # 默认 close=True，从 Gcf 注销
         print(f"💾 已保存: output/epXX/{name}")
         return fig  # 已关闭但 _repr_png_() 仍可用，由 Jupyter cell output 显示一次
     ```
   - **三个禁止**: ① 绝不在 `save_fig` 中传 `close=False`；② 绝不在 `save_fig` 后调 `plt.show()`；③ viz 函数内绝不调 `plt.show()`
   - **绝不出现「看不到图片」的情况** — 如果图片只保存不显示，视为 bug

5. **每个数据输出必须附带简要解释和核心发现**
   - 每个图表、表格、关键指标输出后，**紧跟一个 Markdown cell** 说明三件事：
     1. **这张图是什么** — 展示了什么数据、用了什么可视化方式
     2. **数据分布/模式** — 读者应注意的分布特征、趋势、异常点
     3. **核心发现** — 用 1–2 句话总结从图中得出的关键结论
   - 如果输出不是图表，而是表格或关键数值，仍必须用同样结构解释：
     1. **数据说明** — 这些数值来自什么数据、表示什么物理/统计量
     2. **数据分布/模式** — 哪些范围、差异、异常或分组结构最重要
     3. **核心发现** — 这些数值对当前 Episode 的决策意味着什么
   - 格式示例：
     ```markdown
     > **图表说明**: 263 帧逐帧均温直方图，横轴为温度 (°C)，纵轴为帧数。
     > **数据分布**: 均温集中在 20–22°C，但在 23–24°C 出现第二个峰，提示存在不同温度状态的 session。
     > **核心发现**: 温度的双峰分布验证了 session 跳变的存在，跨 session 帧不可混合使用。
     ```
   - **绝不出现「只有图/表/数值没有解读」的情况** — 裸输出无解释视为 bug

6. **提取共用基础设施到 `core/`**
   - 常见可提取模块：`thermal_core.io`（数据读取/解析）、`thermal_core.viz`（可视化工具）
   - 图片保存工具 `save_fig()` 应在 core 中统一实现
   - 所有 Notebook 共享同一套 setup / style 配置

### 操作规则

1. **新增分析** → 在 `fragments/` 内新建 `NN_name.py`，更新 `manifest.txt`
2. **修改分析** → 编辑 `fragments/` 内对应 `.py` 片段，重新构建
3. **绝不手动编辑 `.ipynb`** — 它是构建产物，且被 `.gitignore` 忽略
4. **构建脚本**: `scripts/build_notebook.py`（单个）、`scripts/build_all_notebooks.py`（批量）

---

## 📋 Episode 管理规则

**进度追踪、数据指标、决策记录全部下放到 Episode，不在此文件中维护。**

每个 Episode `epXX_name` 在四个位置有对应目录：

```
research_log/episodes/epXX_name/README.md  ← 进度 + 任务 + 决策记录
notebooks/epXX_name/                       ← Jupyter 可视化 & 交互分析
reports/epXX_name/                         ← 正式分析报告
output/epXX_name/                          ← 数据产物（CSV、图表等）
```

- 新建 Episode 时，在 `research_log/README.md` 路线图中注册
- Episode 路线图: → `research_log/README.md`
- 当前活跃 Episode: → `research_log/episodes/ep02_displacement_calibration/README.md`
