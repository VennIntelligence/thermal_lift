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
# 本小节对 PyTorch 框架内的前向降采样算子（Forward Operator）与 EP06 中基于 NumPy 的经典退化算子之间的等价性进行严格校验。
# 校验包含三个独立门控：
# 1. **前向算子等价性门控**：确保 PyTorch 与 NumPy 在应用相机 PSF 与下采样时其边界模式匹配恒定填充（`constant`），最大绝对误差应收敛至机器精度。
# 2. **高通预处理等价性门控**：确保基于 PyTorch 的空间背景高通滤波在计算局部均值时匹配最近邻填充（`nearest`）。
# 3. **相位分层划分可复现性门控**：锁定随机种子（`seed=42`），保证训练与验证子集的相位空间直方图划分在 SIREN/WIRE 等不同模型消融实验中具有物理一致性。
# 门控状态为 `pass` 是进入深度学习超分辨率训练的前置必要门槛。若等价性校验缺失（`missing`）或误差超限，则指示着物理前向退化模型约定存在偏差，后续的深度网络输出将退化为无物理意义的调试图像。
