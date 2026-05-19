# %% [markdown]
# ## 1. Forward / Highpass Validation Gate
#
# EP08 的第一个门控不是训练效果，而是 PyTorch forward operator 与 EP06 NumPy forward operator 的等价性。Forward operator 使用 constant 边界；highpass preprocessing 是独立 LR 预处理，使用 nearest 边界。

# %%
forward_paths = {
    "forward_validation_json": OUTPUT_DIR / "forward_validation.json",
    "forward_validation_csv": OUTPUT_DIR / "forward_validation.csv",
    "highpass_validation_json": OUTPUT_DIR / "highpass_validation.json",
    "highpass_validation_csv": OUTPUT_DIR / "highpass_validation.csv",
    "split_validation_json": OUTPUT_DIR / "split_validation.json",
}
display(file_status(forward_paths))

forward_validation = read_json_if_exists(forward_paths["forward_validation_json"])
highpass_validation = read_json_if_exists(forward_paths["highpass_validation_json"])
split_validation = read_json_if_exists(forward_paths["split_validation_json"])
gate_rows = [
    {
        "gate": "forward_operator_equivalence",
        "status": forward_validation.get("status", "missing"),
        "max_abs_error": forward_validation.get("max_abs_error"),
        "source": forward_validation.get("source"),
    },
    {
        "gate": "highpass_preprocess_equivalence",
        "status": highpass_validation.get("status", "missing"),
        "max_abs_error": highpass_validation.get("max_abs_error"),
        "source": highpass_validation.get("source"),
    },
    {
        "gate": "phase_stratified_split_reproducibility",
        "status": split_validation.get("status", "missing"),
        "max_abs_error": None,
        "source": split_validation.get("source", "build_train_val_split(seed=42)"),
    },
]
display(pd.DataFrame(gate_rows))

# %% [markdown]
# > **数据说明**: 第一张表检查 validation 产物是否存在；第二张表读取 JSON 中的门控状态、最大绝对误差和来源字段，也记录 train/val split 可复现性。
# >
# > **怎么看**: Forward 与 highpass 是两个独立门控。Forward 的边界模式应匹配 EP06 `mode="constant"`；highpass 背景估计应匹配 EP06 `mode="nearest"`。
# >
# > **正常/异常**: `status=missing` 代表当前尚未运行验证，不是通过。若 forward 与 highpass 只给出一个合并误差，应视为异常，因为两条链路的边界约定不同；若 split 结果不可复现，SIREN/WIRE activation ablation 也不公平。
# >
# > **核心发现**: 在这两个门控通过前，任何 SIREN / WIRE / Deep Decoder 结果都只能作为调试图，不能进入四方对比结论。
