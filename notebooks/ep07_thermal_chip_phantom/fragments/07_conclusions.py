# %% [markdown]
# ## 6. 结论与集成交接
#
# EP07 的结论应按工程成熟度分层：
#
# 1. **已固定**: TCForge 独立 UV 包、EP06 forward 副本、geometry/physics/shifts/forward/highpass/manifest/evaluate 模块、P0 生成 CLI、smoke CLI、evaluate CLI 和 EP07 notebook 入口。
# 2. **已展示**: Notebook 现在显式展示 Mermaid 架构图、生成配方、HR 真值层、LR observation 层、forward mode 差异、manifest、artifact catalog、smoke gate、evaluate summary 和 CLI regression 产物摘要。
# 3. **已加保护**: benchmark/P1 config 中尚未物化的功能（例如 drift tracks、split objects、crop ROI storage）会 fail fast，避免静默生成半成品 benchmark。
# 4. **不作声明**: EP07 demo 不证明真实数据 SR 成功，不替代 EP06 主 session 结果，也不把 synthetic shifts 或 stage command 当作真实 alignment ground truth。

# %%
handoff = pd.DataFrame(
    [
        {
            "area": "Package",
            "notebook_evidence": "TCForge import probe + architecture diagram",
            "integration_rule": "run tcforge tests before changing public API",
        },
        {
            "area": "Geometry / Physics",
            "notebook_evidence": "HR mask, HR temperature, edge proxy and profile plots",
            "integration_rule": "keep structure truth, temperature truth and contour proxy separate",
        },
        {
            "area": "Forward",
            "notebook_evidence": "exact_ep06_point output plus physical_block_average preview difference",
            "integration_rule": "report forward modes separately in all metric tables",
        },
        {
            "area": "Highpass",
            "notebook_evidence": "raw/highpass panels and independent highpass allclose smoke row",
            "integration_rule": "preserve EP06 parity: float32, nearest, spatial-only sigma",
        },
        {
            "area": "Manifest / Smoke",
            "notebook_evidence": "artifact catalog, manifest table and detailed smoke_summary.csv",
            "integration_rule": "formal P0 datasets must use CLI manifest and smoke report",
        },
        {
            "area": "Evaluate",
            "notebook_evidence": "scene-level metrics and optional CLI evaluation_summary.csv",
            "integration_rule": "scene health metrics are not SR success metrics unless SR outputs are provided",
        },
        {
            "area": "Scale",
            "notebook_evidence": "small lr_shape=(64,96), n_frames=16 demo",
            "integration_rule": "do not generate full-frame multi-scene data inside notebook",
        },
    ]
)
display(handoff)

# %% [markdown]
# > **数据说明**: 这张交接表把 notebook 中已经展示的证据和后续集成规则绑定起来，避免把展示层和工程验收拆散。
# >
# > **怎么看**: 中列说明本 notebook 现在在哪里展示了后台工作；右列是后续改生成器、forward/highpass 或 benchmark 时必须守住的契约。
# >
# > **异常是否正常**: 小 demo 只用于展示和 smoke，不等价于全幅 P0 benchmark。缺少 SR 输出时，evaluate 只能给 scene/data health 指标。
# >
# > **核心发现**: EP07 的展示层已经从“几张 demo 图”升级为可审计的数据生成报告。

# %%
display(
    Markdown(
        f"""
**EP07 产物摘要**

- Demo 目录: `{relative(DEMO_DIR)}`
- 架构图: 第 1 节 Mermaid flowchart
- 生成配方与产物目录: 第 2 / 第 4 节 Markdown 表格
- 图片:
  - `{relative(OUTPUT_DIR / 'demo_hr_scene.png')}`
  - `{relative(OUTPUT_DIR / 'demo_forward_highpass.png')}`
  - `{relative(OUTPUT_DIR / 'demo_dataset_overview.png')}`
  - `{relative(OUTPUT_DIR / 'demo_profiles_generation_vs_observation.png')}`
- Manifest: `{relative(DEMO_DIR / 'manifest.csv')}`
- Smoke summary: `{relative(DEMO_DIR / 'smoke_summary.csv')}`
- Metadata: `{relative(DEMO_DIR / 'metadata.json')}`
- TCForge import status: `{TCFORGE_VERSION}`
"""
    )
)

# %% [markdown]
# > **数据说明**: 最后一段列出本 notebook 执行后产生的 demo 产物路径，便于复查和后续脚本对齐。
# >
# > **怎么看**: 这些产物都在 `output/` 下，属于可重建数据，不应提交到 Git。应提交的是 notebook `fragments/`、TCForge 源码、配置、research log 和正式 Markdown report 源文件。
# >
# > **异常是否正常**: 如果路径存在但 smoke 未通过，应先修复 generator 或 fallback 路径，再讨论算法指标。若 `TCForge import status` 仍不可导入，说明正式包集成尚未完成。
# >
# > **核心发现**: EP07 notebook 现在满足“生成原理可见、产物用途可见、检测指标可见、图表风格可控”的报告目标。
