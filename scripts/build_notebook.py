#!/usr/bin/env python3
"""
全局 Notebook 构建工具

用法:
    uv run python scripts/build_notebook.py notebooks/ep01_data_processing
    uv run python scripts/build_notebook.py notebooks/ep01_data_processing --execute

功能:
    1. 扫描 fragments/ 子目录（或顶层目录）下 NN_*.py 片段
    2. 按 manifest.txt 或文件名排序，拼接为 .ipynb
    3. --execute: 执行 notebook，将所有输出（含图片）嵌入 .ipynb
       执行后用户打开即可看到完整结果，也可重新从头运行
"""

import json
import re
import sys
from pathlib import Path


def parse_py_to_cells(filepath: Path) -> list[dict]:
    """解析单个 .py 文件为 notebook cells 列表"""
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")

    cells = []
    current_type = "code"
    current_lines = []

    def flush():
        if not current_lines:
            return
        # 去掉末尾空行
        while current_lines and current_lines[-1].strip() == "":
            current_lines.pop()
        if not current_lines:
            return

        if current_type == "markdown":
            # 去掉 markdown 行的 '# ' 前缀
            md_lines = []
            for line in current_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.strip() == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            source = "\n".join(md_lines)
        else:
            source = "\n".join(current_lines)

        cells.append({
            "cell_type": current_type,
            "source": source,
            "metadata": {},
        })

    for line in lines:
        stripped = line.strip()
        if stripped == "# %% [markdown]":
            flush()
            current_lines = []
            current_type = "markdown"
        elif stripped == "# %%":
            flush()
            current_lines = []
            current_type = "code"
        else:
            current_lines.append(line)

    flush()
    return cells


def build_notebook(cells: list[dict], output_path: Path):
    """将 cells 列表写入 .ipynb"""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": [],
    }

    for i, cell in enumerate(cells):
        nb_cell = {
            "id": f"cell-{i:04d}",
            "cell_type": cell["cell_type"],
            "metadata": cell.get("metadata", {}),
            "source": cell["source"].split("\n") if cell["source"] else [],
        }
        # 给每行加 \n（除了最后一行）
        if nb_cell["source"]:
            nb_cell["source"] = [
                line + "\n" for line in nb_cell["source"][:-1]
            ] + [nb_cell["source"][-1]]

        if cell["cell_type"] == "code":
            nb_cell["outputs"] = []
            nb_cell["execution_count"] = None

        nb["cells"].append(nb_cell)

    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 已构建: {output_path} ({len(cells)} cells)")


def execute_notebook(notebook_path: Path):
    """执行 notebook，将输出（含图片）嵌入 .ipynb。

    执行后的 notebook 打开即可看到全部结果，用户也可以重新从头运行。
    """
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        print("❌ 缺少 nbconvert/nbformat，请 uv add nbconvert nbformat")
        sys.exit(1)

    print(f"🔄 执行 notebook: {notebook_path} ...")

    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(
        timeout=600,        # 单 cell 超时 10 分钟
        kernel_name="python3",
    )

    # 以 notebook 所在目录的父级（项目根）为 cwd
    # 使 Path.cwd() 在 notebook 内指向项目根
    project_root = notebook_path.parent
    while not (project_root / "AGENTS.md").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    try:
        ep.preprocess(nb, {"metadata": {"path": str(project_root)}})
        print(f"✅ 执行完成")
    except Exception as e:
        print(f"⚠️ 执行中有错误（已保存部分结果）: {e}")

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"💾 已保存（含执行输出）: {notebook_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/build_notebook.py <notebook_dir> [--execute]")
        print("示例: python scripts/build_notebook.py notebooks/ep01_data_processing --execute")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    do_execute = "--execute" in flags

    target_dir = Path(args[0])
    if not target_dir.is_dir():
        print(f"❌ 目录不存在: {target_dir}")
        sys.exit(1)

    # 新规范: 片段在 fragments/ 子目录; 旧规范: 片段在顶层目录
    fragments_dir = target_dir / "fragments"
    if fragments_dir.is_dir():
        scan_dir = fragments_dir
        print(f"📁 使用 fragments/ 子目录")
    else:
        scan_dir = target_dir
        print(f"📁 使用顶层目录（旧规范，建议迁移到 fragments/）")

    # 确定文件顺序
    manifest = scan_dir / "manifest.txt"
    if manifest.exists():
        file_list = [
            scan_dir / line.strip()
            for line in manifest.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        print(f"📋 使用 manifest.txt ({len(file_list)} 文件)")
    else:
        # 按文件名排序，只取 NN_*.py 格式
        file_list = sorted(
            f for f in scan_dir.glob("*.py")
            if re.match(r"^\d+_", f.name)
        )
        print(f"📂 按文件名排序 ({len(file_list)} 文件)")

    if not file_list:
        print("❌ 没有找到片段文件")
        sys.exit(1)

    # 解析并拼接
    all_cells = []
    for f in file_list:
        if not f.exists():
            print(f"⚠️ 文件不存在，跳过: {f}")
            continue
        cells = parse_py_to_cells(f)
        print(f"  {f.name}: {len(cells)} cells")
        all_cells.extend(cells)

    # 输出
    output_name = target_dir.name + ".ipynb"
    output_path = target_dir / output_name
    build_notebook(all_cells, output_path)

    # 执行（可选）
    if do_execute:
        execute_notebook(output_path)
    else:
        print("💡 提示: 加 --execute 参数可自动执行并嵌入输出（含图片）")


if __name__ == "__main__":
    main()
