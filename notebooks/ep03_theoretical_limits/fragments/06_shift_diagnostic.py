# %% [markdown]
# ## Step 5 — EP03 SR 设计边界表 / SR Design Boundary Table
#
# 最后把物理边界转成 EP05 的设计约束。这里直接用双语 Markdown 表格区分“EP03 支持的结论”和“EP03 不支持的过度结论”，不再把表格渲染成图片。
#
# 这一步是把前面的教程内容收束成工程决策：哪些可以作为后续实验默认设定，哪些只能作为风险提示，哪些声明目前没有证据。尤其要注意，理论边界的作用是缩小 POC 设计空间，不是跳过真实重建实验。

# %% [markdown]
# | 状态 | Status | 设计规则/声明 | Design rule / claim | EP03 证据边界 | EP03 evidence boundary | EP05/EP06 约束 |
# |---|---|---|---|---|---|---|
# | 支持 | Supported | 使用 2x 输出网格作为 contour-level POC 默认设置 | Use a 2x output grid for the contour-level POC | 10 um detector pitch 意味着 5 um 目标采样间距至少需要 2x 网格 | A 10 um detector pitch means a 5 um target sample spacing requires at least a 2x grid | 2x 只能作为默认重建网格；不能表述为 5 um 温度计量能力 |
# | 支持 | Supported | 用 PSF/MTF 与噪声约束预期 | Constrain expectations with PSF/MTF and noise | Gaussian PSF MTF 会强烈衰减高频，4x-grid Nyquist 附近尤其明显 | Gaussian PSF MTF strongly attenuates high frequencies, especially near 4x-grid Nyquist | 报告 contour/shape evidence 与稳定性；不能只报告显示倍率 |
# | 支持 | Supported | 用局部 ESF/CRB 作为 alignment-anchor confidence | Use local ESF/CRB as alignment-anchor confidence | 高 SNR 局部边缘在受控假设下具备亚像素定位 CRB | High-SNR local edges can have subpixel localization CRB under controlled assumptions | SR fusion 前用 anchor confidence 做帧/patch 质量门控 |
# | 支持 | Supported | stage/file-name shift 只作为 prior | Use stage/file-name shifts only as priors | 物理命令定义期望采样相位，但没有测量图像真实对齐 | Physical commands define expected sampling phases but do not measure image alignment | 用数据驱动 alignment 修正 prior，并拒绝不一致区域 |
# | 不支持 | Not supported | 从单个局部诊断推出全局 SR no-go | Draw a global SR no-go conclusion from one local diagnostic | 局部可观测性诊断依赖 edge/patch，不能代表全局 shape recovery | Local observability diagnostics are edge/patch specific and do not represent global shape recovery | 不能把单个 NCC/ESF 失败外推成全局 SR 否定 |
# | 不支持 | Not supported | 从插值声称 4x 或 5 um 定量分辨率 | Claim 4x or 5 um quantitative resolution from interpolation | MTF/SNR 证据不足以默认保留该声明所需的高频能量 | MTF/SNR evidence does not preserve enough high-frequency energy for that claim by default | 4x 只能作为 visualization/ablation，除非 forward model 与 contour consistency 验证通过 |
# | 不支持 | Not supported | 单独用 residual 或 Tenengrad 证明成功 | Use residual or Tenengrad alone as success evidence | residual 可能因错误原因下降，sharpness 也可能随 artifact 上升 | Residual can fall for wrong reasons and sharpness can rise with artifacts | 必须配合 shape/contour evidence、split checks 与 alignment quality gates |

# %% [markdown]
# > **数据说明**: 这张双语表逐条列出 EP03 可以支持的 SR 设计约束，以及不能从 EP03 推出的过度结论。
# > **怎么读**: 先看每行的“支持/不支持”边界，再看对应证据来自哪个分析模块。支持项可以进入 EP05/EP06 默认方案；不支持项不是说永远不可能，而是说 EP03 没有给出足够证据，不能写进当前结论。
# > **正常/异常理解**: 正常的设计边界表应该同时包含正向约束和禁止过度解释的条目。如果只保留“2x 可行”而删掉限制条件，会放大技术声明；如果只保留风险而忽略局部可观测证据，又会把理论分析误读成全局否定。
# > **核心发现**: EP03 的落点是“先做 2x contour-level SR POC，并用 shape/contour evidence + alignment quality gates 验收”，不是全局否定或“显示倍率就是 SR”。最终结论必须等待真实数据 POC。

# %%
output_index = pd.DataFrame(
    [
        {"artifact": path.name, "relative_path": str(path.relative_to(PROJECT_ROOT))}
        for path in sorted(OUTPUT_DIR.glob("*"))
        if path.is_file()
    ]
)
display(output_index)

# %% [markdown]
# > **数据说明**: 该表列出本次 EP03 notebook 生成的 CSV、JSON 和 PNG 产物。
# > **怎么读**: CSV/JSON 适合复查数值和参数，PNG 适合报告展示。若要追踪某个 EP05 设计选择，应优先从对应 CSV 找到数值来源，再回到图中确认视觉解释是否一致。
# > **正常/异常理解**: 产物覆盖 pixel/resolution 区分、MTF/PSF、noise/SNR、local observability 和 ESF/CRB。若缺少某类产物，说明 Notebook 没有完整执行；若产物存在但结论与本节文字矛盾，应优先检查配置版本和上游 EP01 输出。
# > **核心发现**: 这些产物共同约束 EP05：默认推进 2x contour-level SR POC，所有清晰度指标必须与结构轮廓证据、对齐质量门控和可复现实验记录绑定。EP03 产物是实验设计依据，不是最终客户交付证据。
