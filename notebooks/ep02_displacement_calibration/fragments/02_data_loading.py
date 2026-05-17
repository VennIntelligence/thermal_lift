# %% [markdown]
# ## 1. 数据加载
#
# 使用 EP01 的 `frame_audit.csv`，其中 `session` 来自真实采集顺序下的均温跳变检测。
# EP02 不重新定义 session，只继承 EP01 的边界，并默认只用主扫描 session 做位移标定。

# %%
df_audit = pd.read_csv(EP01_OUTPUT_DIR / "frame_audit.csv")
required_cols = {"file", "X", "Y", "R", "rows", "cols", "T_mean", "session", "acquisition_order", "is_main_session"}
missing_cols = required_cols - set(df_audit.columns)
if missing_cols:
    raise ValueError(f"Missing required audit columns: {sorted(missing_cols)}")

df_audit = df_audit.sort_values("acquisition_order").reset_index(drop=True)
main_session = int(df_audit.loc[df_audit["is_main_session"].astype(bool), "session"].mode().iat[0])
df_calib = df_audit[df_audit["session"].eq(main_session)].copy()

existing_files = df_audit["file"].map(lambda name: (DATA_DIR / name).exists())
print(f"Audit frames: {len(df_audit)}")
print(f"Existing TXT files: {existing_files.sum()} / {len(existing_files)}")
print(f"Sessions: {df_audit['session'].nunique()}")
print(f"Main calibration session: {main_session} ({len(df_calib)} frames)")
print(f"R=0 frames: {(df_audit['R'] == 0).sum()}")
print(f"R=0 frames in main session: {(df_calib['R'] == 0).sum()}")
print(f"Frame shape: {df_audit['rows'].mode().iat[0]} x {df_audit['cols'].mode().iat[0]}")

session_summary = (
    df_audit.groupby("session")
    .agg(n_frames=("file", "count"), mean_temp=("T_mean", "mean"))
    .reset_index()
)
session_summary

# %% [markdown]
# > **数据说明**: 这张表继承 EP01 的采集顺序 session 划分，统计每个 session 的帧数和平均温度。
# > EP02 后续只在主扫描 session 内构造互相关帧对。
# >
# > **数据分布**: 共有 3 个采集顺序温度段；最大的一段包含 255 帧，均温约 23.3°C。
# > 早期低温/补采帧数量很少，不代表主扫描。
# >
# > **核心发现**: 位移标定应使用主扫描 session。
# > 早期低温/补采帧可以记录为诊断数据，但不应参与主 θ 拟合或 SR 重建。
