# %% [markdown]
# ## Output Inventory

# %%
manifest = read_manifest()
frame_budget = read_csv("frame_budget.csv")
shift_robustness = read_csv("shift_robustness.csv")
alignment_source = read_csv("alignment_source.csv")

inventory = pd.DataFrame(
    [
        {"artifact": "frame_budget.csv", "rows": len(frame_budget), "success": int(frame_budget["status"].eq("success").sum())},
        {"artifact": "shift_robustness.csv", "rows": len(shift_robustness), "success": int(shift_robustness["status"].eq("success").sum())},
        {"artifact": "alignment_source.csv", "rows": len(alignment_source), "success": int(alignment_source["status"].eq("success").sum())},
        {"artifact": "run_manifest.json", "rows": len(manifest.get("runs", [])), "success": sum(1 for row in manifest.get("runs", []) if row.get("status") == "success")},
    ]
)
display(inventory)

# %% [markdown]
# > **Table note**: This inventory checks whether the design matrices were
# > materialized and how many rows succeeded.
# >
# > **Pattern to inspect**: Failed rows should remain in the CSVs and manifest
# > with `status=failed`, so a missing metric is traceable to one run rather than
# > silently disappearing from the aggregate.
# >
# > **Core finding**: Use this table first when reviewing an overnight run; it
# > separates matrix completion from metric interpretation.

# %% [markdown]
# ## Frame Budget

# %%
show_fig(OUTPUT_DIR / "fig_frame_budget.png")
display(
    frame_budget.groupby(["arm", "n_frames"], dropna=False)
    .agg(
        rows=("run_id", "count"),
        success=("status", lambda s: int((s == "success").sum())),
        raw_control_corr_mean=("raw_control_corr", "mean"),
        split_half_nrmse_mean=("split_half_nrmse_median", "mean"),
        artifact_mean=("artifact_score", "mean"),
    )
    .reset_index()
)

# %% [markdown]
# > **Figure note**: The frame-budget curve plots raw-control correlation
# > against the number of clean SR frames used by each classical arm.
# >
# > **How to read it**: Higher raw-control correlation means the reconstructed
# > highpass structure agrees more with a simple raw-temperature mean control.
# > The split-half and artifact columns in the table are proxies; lower
# > split-half NRMSE and lower artifact score are usually preferable, but they
# > are not independent optical resolution evidence.
# >
# > **Core finding**: The expected signal is diminishing returns as N approaches
# > 248 frames. Any non-monotonic point should be checked against its seed and
# > run-level manifest before being interpreted physically.

# %% [markdown]
# ## Shift Robustness

# %%
show_fig(OUTPUT_DIR / "fig_shift_robustness.png")
display(
    shift_robustness.groupby(["arm", "shift_noise_sigma_px"], dropna=False)
    .agg(
        rows=("run_id", "count"),
        success=("status", lambda s: int((s == "success").sum())),
        raw_control_corr_mean=("raw_control_corr", "mean"),
        split_half_nrmse_mean=("split_half_nrmse_median", "mean"),
        frc_10um_mean=("frc_10um", "mean"),
    )
    .reset_index()
)

# %% [markdown]
# > **Figure note**: The robustness curve perturbs contour-refined shifts with
# > isotropic Gaussian noise in LR pixels and tracks proxy degradation.
# >
# > **How to read it**: A stable method should degrade gradually as sigma grows.
# > Sigma is a pressure-test parameter added to measured shifts; it is not a
# > claim about true alignment error.
# >
# > **Core finding**: The useful result is the degradation trend and relative
# > sensitivity, not any single sigma point as a calibrated physical error.

# %% [markdown]
# ## Alignment Source

# %%
show_fig(OUTPUT_DIR / "fig_alignment_source.png")
display(
    alignment_source[
        [
            "arm",
            "shift_source",
            "status",
            "raw_control_corr",
            "split_half_nrmse_median",
            "artifact_score",
            "zigzag_fwhm_median_um",
            "zigzag_dip_depth_median",
        ]
    ]
)

# %% [markdown]
# > **Figure note**: The alignment-source ablation compares command-prior
# > shifts with contour-refined shifts at the full 248-frame budget.
# >
# > **How to read it**: Contour-refined alignment should improve end-to-end
# > stability if the upstream localization contributes useful data constraints.
# > Command prior remains a prior, not a ground-truth alignment target.
# >
# > **Core finding**: This table quantifies the cost of using only stage-command
# > geometry in the final reconstruction pipeline.

# %% [markdown]
# ## Paper Export

# %%
show_fig(PAPER_FIGURE)

# %% [markdown]
# > **Figure note**: `fig07_budget_robustness` combines the frame-budget and
# > shift-robustness panels for the paper figure slot.
# >
# > **How to read it**: The left panel answers how much the classic arms gain
# > from more clean frames; the right panel answers how quickly proxy agreement
# > degrades under synthetic shift perturbations.
# >
# > **Core finding**: The figure supports the Section 6.4-6.5 narrative only as
# > an inference-time stability study. It should not be described as a direct
# > spatial-resolution proof.

