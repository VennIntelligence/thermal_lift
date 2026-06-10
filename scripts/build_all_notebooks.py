#!/usr/bin/env python3
"""
一键构建所有 Notebooks

用法:
    uv run python scripts/build_all_notebooks.py              # 仅构建
    uv run python scripts/build_all_notebooks.py --execute     # 构建 + 执行（推荐）

功能:
    扫描 notebooks/ 下所有含 fragments/ 子目录的 episode 目录，
    依次调用 build_notebook.py 构建（并可选执行）每个 notebook。

典型场景:
    - Git 迁移到新机器后，先 `uv run python scripts/build_all_caches.py`，
      再 `uv run python scripts/build_all_notebooks.py --execute`
    - 修改 core/ 库后，批量验证所有 notebook 仍可正常运行

前置建议:
    Notebook 展示层只读缓存 PNG/CSV，重计算在 cache 脚本中完成。
    首次迁移或数据/逻辑变更后请先运行:
        uv run python scripts/build_all_caches.py
"""

import subprocess
import sys
import time
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    notebooks_dir = project_root / "notebooks"
    build_script = project_root / "scripts" / "build_notebook.py"

    if not build_script.exists():
        print(f"❌ 找不到构建脚本: {build_script}")
        sys.exit(1)

    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    # 找所有含 fragments/ 的 notebook 目录，按名称排序
    nb_dirs = sorted(
        d for d in notebooks_dir.iterdir()
        if d.is_dir() and (d / "fragments").is_dir()
    )

    if not nb_dirs:
        print("❌ 没有找到任何含 fragments/ 的 notebook 目录")
        sys.exit(1)

    print(f"📚 发现 {len(nb_dirs)} 个 notebook:")
    for d in nb_dirs:
        n_fragments = len(list((d / "fragments").glob("*.py")))
        print(f"   • {d.name}  ({n_fragments} 片段)")
    print()

    failed = []
    t_total = time.time()

    for i, nb_dir in enumerate(nb_dirs, 1):
        print(f"{'=' * 60}")
        print(f"[{i}/{len(nb_dirs)}] 构建: {nb_dir.name}")
        print(f"{'=' * 60}")

        t_start = time.time()
        cmd = [sys.executable, str(build_script), str(nb_dir)] + flags
        result = subprocess.run(cmd, cwd=str(project_root))
        elapsed = time.time() - t_start

        if result.returncode != 0:
            failed.append(nb_dir.name)
            print(f"⚠️  {nb_dir.name} 构建失败 ({elapsed:.1f}s)")
        else:
            print(f"⏱️  {nb_dir.name} 完成 ({elapsed:.1f}s)")
        print()

    # 汇总
    total_elapsed = time.time() - t_total
    print(f"{'=' * 60}")
    print(f"📊 构建完成: {len(nb_dirs) - len(failed)}/{len(nb_dirs)} 成功  "
          f"(总耗时 {total_elapsed:.1f}s)")
    if failed:
        print(f"❌ 失败: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅ 全部成功！")


if __name__ == "__main__":
    main()
