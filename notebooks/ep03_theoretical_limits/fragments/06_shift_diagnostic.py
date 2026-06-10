# %% [markdown]
# ## Step 6 — EP03 SR 设计边界表 / SR Design Boundary Table
#
# 最后把物理边界转成 EP05/EP06 的设计约束。这里直接用双语 Markdown 表格区分“EP03 支持的结论”和“EP03 不支持的过度结论”，不再把表格渲染成图片。
#
# 这一步是把前面的教程内容收束成工程决策：哪些可以作为后续实验默认设定，哪些只能作为风险提示，哪些声明目前没有证据。尤其要注意，理论边界的作用是缩小 POC 设计空间，不是跳过真实重建实验。

# %% [markdown]
# | 状态 | Status | 设计规则/声明 | Design rule / claim | EP03 证据边界 | EP03 evidence boundary | EP05/EP06 约束 |
# |---|---|---|---|---|---|---|
# | 支持 | Supported | 使用 2x 输出网格作为 contour-level POC 默认设置 | Use a 2x output grid for the contour-level POC | 10 um detector pitch 意味着 5 um 目标采样间距至少需要 2x 网格 | A 10 um detector pitch means a 5 um target sample spacing requires at least a 2x grid | 2x 只能作为默认重建网格；不能表述为 5 um 温度计量能力 |
# | 支持 | Supported | 用 PSF/MTF 与噪声约束预期 | Constrain expectations with PSF/MTF and noise | Gaussian PSF MTF 会强烈衰减高频，4x-grid Nyquist 附近尤其明显 | Gaussian PSF MTF strongly attenuates high frequencies, especially near 4x-grid Nyquist | 报告 contour/shape evidence 与稳定性；不能只报告显示倍率 |
# | 支持 | Supported | 用 MTF x SNR effective SNR 作为必要条件风险图 | Use MTF x SNR effective SNR as a necessary-condition risk map | `DeltaT * MTF / noise` 可以标记哪些高频声明更容易落入噪声主导区 | `DeltaT * MTF / noise` marks which high-frequency claims are more likely to be noise dominated | effective SNR 高也不是 SR 成功证明；低则需要更强门控或放弃该频率声明 |
# | 支持 | Supported | 用局部 ESF/CRB 作为 alignment-anchor confidence | Use local ESF/CRB as alignment-anchor confidence | 高 SNR 局部边缘在受控假设下具备亚像素定位 CRB | High-SNR local edges can have subpixel localization CRB under controlled assumptions | SR fusion 前用 anchor confidence 做帧/patch 质量门控 |
# | 支持 | Supported | 用 0.05/0.10 px CRB gate 做局部风险标签 | Use 0.05/0.10 px CRB gates as local risk labels | CRB sensitivity 扫描显示温差、PSF、帧数和相位覆盖共同决定 anchor 下界 | CRB sensitivity shows contrast, PSF, frame count, and phase coverage jointly define anchor lower bounds | CRB 是乐观理论下界，不能替代真实 alignment 误差统计 |
# | 支持 | Supported | stage/file-name shift 只作为 prior | Use stage/file-name shifts only as priors | 物理命令定义期望采样相位，但没有测量图像真实对齐 | Physical commands define expected sampling phases but do not measure image alignment | 用数据驱动 alignment 修正 prior，并拒绝不一致区域 |
# | 不支持 | Not supported | 从单个局部诊断推出全局 SR no-go | Draw a global SR no-go conclusion from one local diagnostic | 局部可观测性诊断依赖 edge/patch，不能代表全局 shape recovery | Local observability diagnostics are edge/patch specific and do not represent global shape recovery | 不能把单个 NCC/ESF 失败外推成全局 SR 否定 |
# | 不支持 | Not supported | 从插值声称 4x 或 5 um 定量分辨率 | Claim 4x or 5 um quantitative resolution from interpolation | MTF/SNR 证据不足以默认保留该声明所需的高频能量 | MTF/SNR evidence does not preserve enough high-frequency energy for that claim by default | 4x 只能作为 visualization/ablation，除非 forward model 与 contour consistency 验证通过 |
# | 不支持 | Not supported | 单独用 residual 或 Tenengrad 证明成功 | Use residual or Tenengrad alone as success evidence | residual 可能因错误原因下降，sharpness 也可能随 artifact 上升 | Residual can fall for wrong reasons and sharpness can rise with artifacts | 必须配合 shape/contour evidence、split checks 与 alignment quality gates |

# %% [markdown]
# ### 📝 超分辨率物理边界设计表与算法决策边界
#
# 上述超分辨率物理边界设计表（SR Design Boundary Table）总结了 EP03 理论推导对后续重建流程的直接指导作用。它将衍射极限、MTF衰减和CRB物理定位精度转化为超分辨率重构的具体算法决策，明确了系统能够提供且可被验证的物理增益范畴。
#
# **💡 算法决策**：
# 1. 强制设定 2x 重构网格作为轮廓级（Contour-level）超分辨率重建的技术主线，并在物理上将理论Nyquist周期限定为 10.0 $\mu\text{m}$。
# 2. 任何高频轮廓细节的宣称，必须与其物理信噪比及调制传递函数退化程度进行绑定。
# 3. 后续算法必须构建以局部 CRB 置信度为引导的数据驱动几何对齐闸门（EP04/EP05 Alignment Gate），屏蔽低对比度低响应帧。

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
# ### 📊 本阶段研究产物索引与交叉核对
#
# 汇总了 EP03 研究阶段所生成的全部数字化产物（包括数据表与可视化图表），这些产物被完整缓存于物理路径 `output/ep03_theoretical_limits/` 目录下。
#
# **💡 算法决策**：所有缓存的数字化产物（如 MTF 响应表、CRB 灵敏度扫描表等）共同构成了后续 EP05 配准基线标定和 EP06 重建实验的物理判据。当超分辨率重构中观察到边缘对齐失效或伪影时，算法分析应直接回溯至对应的 MTF x SNR 响应表，核对物理信号是否已在噪声底以下，从而决定是引入更强空域约束还是剔除对应物理帧对。
