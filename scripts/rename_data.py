#!/usr/bin/env python3
"""
rename_data.py — 原始热像数据文件重命名工具

将 data/data_raw/infrared_avi/ 下的 TXT/BMP 文件从原始连写格式
重命名为 X_Y_R.ext 标准格式。

命名规则参考: AGENTS.md § 数据命名规则
                data/data_raw/name_rules.txt

用法:
    python scripts/rename_data.py              # dry-run 模式（默认）
    python scripts/rename_data.py --execute    # 实际执行重命名
    python scripts/rename_data.py --report     # 输出 CSV 映射表
    python scripts/rename_data.py --execute --report  # 执行 + 输出 CSV

输入依赖: data/data_raw/infrared_avi/（--data-dir 可覆盖）、
    configs/coordinate_set.json（合法坐标集合）
输出: 就地重命名数据文件；--report 时写 reports/rename_mapping.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── 项目根目录（脚本在 scripts/ 下，根目录上一层） ──────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── 路径常量 ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
CONFIG_PATH = PROJECT_ROOT / "configs" / "coordinate_set.json"
NAME_RULES_PATH = PROJECT_ROOT / "data" / "data_raw" / "name_rules.txt"
REPORTS_DIR = PROJECT_ROOT / "reports"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"

# ── 需要跳过的文件/目录 ──────────────────────────────────────────────
SKIP_FILES = {"name_rules.txt", "visualize_heatmaps.py", ".DS_Store"}
SKIP_DIRS = {"outputs"}
SKIP_EXTENSIONS = {".avi"}


def load_valid_coordinates(config_path: Path) -> set:
    """从 configs/coordinate_set.json 加载合法坐标集合。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if "valid_coordinates" in config:
        return set(config["valid_coordinates"])
    return set(config["x_coords_um"]) | set(config["y_coords_um"])


def verify_agents_md(agents_path: Path) -> None:
    """读取 AGENTS.md 确认命名规则存在。"""
    if not agents_path.exists():
        print(f"⚠ 警告: AGENTS.md 不存在于 {agents_path}")
        return
    text = agents_path.read_text(encoding="utf-8")
    if "数据命名规则" not in text:
        print("⚠ 警告: AGENTS.md 中未找到「数据命名规则」章节")
    else:
        print("✓ AGENTS.md 命名规则章节已确认")


def decode_filename(stem: str, valid_coords: set) -> dict:
    """
    解码原始文件名（不含扩展名）为 (X, Y, R)。

    返回 dict:
        success: True/False
        x, y, r: 解码后的坐标值（仅 success=True 时有效）
        ambiguous: True 表示有多种合法分割
        candidates: 所有合法 (x, y, r) 候选列表
        error: 错误描述（仅 success=False 时有效）
    """
    result = {
        "success": False,
        "x": None, "y": None, "r": None,
        "ambiguous": False,
        "candidates": [],
        "error": None,
    }

    # ── 处理中文逗号分隔的特殊情况 ──────────────────────────────────
    if "，" in stem:
        parts = stem.split("，")
        if len(parts) == 3:
            # "X，Y，R" 格式
            try:
                x_val = int(parts[0])
                y_val = int(parts[1])
                r_val = int(parts[2])
            except ValueError:
                result["error"] = f"中文逗号格式解析失败: '{stem}'"
                return result
        elif len(parts) == 2:
            # "X，YR" 格式，如 "2，400" → X=2, Y=40, R=0
            try:
                x_val = int(parts[0])
            except ValueError:
                result["error"] = f"中文逗号格式 X 解析失败: '{stem}'"
                return result
            yr_str = parts[1]
            # 对 YR 部分用标准纯数字逻辑解码
            if not yr_str.isdigit() or len(yr_str) < 2:
                result["error"] = f"中文逗号格式 YR 解析失败: '{stem}'"
                return result
            r_val = int(yr_str[-1])
            y_str = yr_str[:-1]
            try:
                y_val = int(y_str)
            except ValueError:
                result["error"] = f"中文逗号格式 Y 解析失败: '{stem}'"
                return result
        else:
            result["error"] = f"中文逗号分割段数异常({len(parts)}): '{stem}'"
            return result

        # 校验
        if x_val not in valid_coords:
            result["error"] = f"X={x_val} 不在合法坐标集中"
            return result
        if y_val not in valid_coords:
            result["error"] = f"Y={y_val} 不在合法坐标集中"
            return result
        if r_val not in (0, 1, 2):
            result["error"] = f"R={r_val} 不合法（需为 0/1/2）"
            return result

        result.update({"success": True, "x": x_val, "y": y_val, "r": r_val})
        result["candidates"] = [(x_val, y_val, r_val)]
        return result

    # ── 硬编码特判（name_rules.txt 明确规定） ────────────────────────
    # "2400" → X=24, Y=0, R=0（不是 X=2, Y=40）
    EXPLICIT_OVERRIDES = {
        "2400": (24, 0, 0),
    }

    if stem in EXPLICIT_OVERRIDES:
        x_val, y_val, r_val = EXPLICIT_OVERRIDES[stem]
        result.update({"success": True, "x": x_val, "y": y_val, "r": r_val})
        result["candidates"] = [(x_val, y_val, r_val)]
        return result

    # ── 纯数字处理 ──────────────────────────────────────────────────
    if not stem.isdigit():
        result["error"] = f"文件名不是纯数字: '{stem}'"
        return result

    if len(stem) < 2:
        result["error"] = f"文件名太短: '{stem}'"
        return result

    # 最右 1 位 = R
    r_val = int(stem[-1])
    if r_val not in (0, 1, 2):
        result["error"] = f"R={r_val} 不合法（需为 0/1/2）"
        return result

    xy_str = stem[:-1]  # 去掉 R 后的 X+Y 部分

    if len(xy_str) == 0:
        result["error"] = f"去掉 R 后无 XY 部分: '{stem}'"
        return result

    # 枚举所有可能的 X|Y 分割点
    candidates = []
    for split_pos in range(1, len(xy_str) + 1):
        x_str = xy_str[:split_pos]
        y_str = xy_str[split_pos:]

        # Y 可以是空字符串 → Y=0
        # 例: "00" → xy_str="0", split_pos=1 → x_str="0", y_str=""
        if y_str == "":
            y_val = 0
        else:
            # 排除前导零（除了 "0" 本身）
            if len(y_str) > 1 and y_str[0] == "0":
                continue
            try:
                y_val = int(y_str)
            except ValueError:
                continue

        # 排除 X 的前导零（除了 "0" 本身）
        if len(x_str) > 1 and x_str[0] == "0":
            continue
        try:
            x_val = int(x_str)
        except ValueError:
            continue

        if x_val in valid_coords and y_val in valid_coords:
            candidates.append((x_val, y_val, r_val))

    result["candidates"] = candidates

    if len(candidates) == 0:
        result["error"] = f"无合法 (X,Y) 分割: '{stem}'"
        return result
    elif len(candidates) == 1:
        x, y, r = candidates[0]
        result.update({"success": True, "x": x, "y": y, "r": r})
        return result
    else:
        # ── 消歧规则: 优先选择较小的 X 值（即较短 X 解释）──
        # 原理: 短文件名中，X 用最少位数表示，剩余给 Y。
        # 例: "200" → (2,0,0) 优于 (20,0,0)；"280" → (2,8,0) 优于 (28,0,0)
        # 例: "400" → (4,0,0) 优于 (40,0,0)
        # 注: "2400" 已被硬编码特判为 (24,0,0)
        candidates_sorted = sorted(candidates, key=lambda c: c[0])
        x, y, r = candidates_sorted[0]
        result.update({"success": True, "x": x, "y": y, "r": r})
        result["candidates"] = candidates
        # 记录消歧过程以供审查
        if len(candidates) > 1:
            alt = candidates_sorted[1:]
            print(f"  ℹ 消歧 '{stem}': 选择 (X={x}, Y={y}, R={r})，"
                  f"排除 {alt}")
        return result


def collect_files(data_dir: Path) -> list:
    """收集需要处理的 .txt 和 .bmp 文件。"""
    files = []
    for f in sorted(data_dir.iterdir()):
        if f.is_dir() and f.name in SKIP_DIRS:
            continue
        if f.is_dir():
            continue
        if f.name in SKIP_FILES:
            continue
        if f.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if f.suffix.lower() in (".txt", ".bmp"):
            files.append(f)
    return files


def build_rename_plan(files: list, valid_coords: set) -> tuple:
    """
    构建重命名计划。

    返回: (mappings, errors)
        mappings: list of dict {old_name, new_name, x, y, r, ext}
        errors: list of dict {old_name, error, candidates}
    """
    mappings = []
    errors = []

    for f in files:
        stem = f.stem
        ext = f.suffix  # 保留原始扩展名大小写

        decoded = decode_filename(stem, valid_coords)

        if decoded["success"]:
            new_name = f"{decoded['x']}_{decoded['y']}_{decoded['r']}{ext}"
            mappings.append({
                "old_name": f.name,
                "new_name": new_name,
                "x": decoded["x"],
                "y": decoded["y"],
                "r": decoded["r"],
                "ext": ext,
                "path": f,
            })
        else:
            errors.append({
                "old_name": f.name,
                "error": decoded["error"],
                "ambiguous": decoded["ambiguous"],
                "candidates": decoded["candidates"],
            })

    return mappings, errors


def check_conflicts(mappings: list) -> list:
    """检测目标文件名冲突。"""
    name_counts = Counter(m["new_name"] for m in mappings)
    conflicts = [(name, count) for name, count in name_counts.items() if count > 1]
    return conflicts


def check_txt_bmp_pairs(mappings: list) -> tuple:
    """
    检查 TXT/BMP 配对情况。

    返回: (paired, txt_only, bmp_only)
    """
    txt_coords = set()
    bmp_coords = set()

    for m in mappings:
        coord = (m["x"], m["y"], m["r"])
        if m["ext"].lower() == ".txt":
            txt_coords.add(coord)
        elif m["ext"].lower() == ".bmp":
            bmp_coords.add(coord)

    paired = txt_coords & bmp_coords
    txt_only = txt_coords - bmp_coords
    bmp_only = bmp_coords - txt_coords

    return paired, txt_only, bmp_only


def compute_coordinate_stats(mappings: list) -> tuple:
    """
    计算坐标统计信息。

    返回: (unique_coords, repeat_coords)
        unique_coords: set of (X, Y)
        repeat_coords: dict {(X,Y): [r0, r1, ...]}
    """
    coord_repeats = defaultdict(set)
    for m in mappings:
        if m["ext"].lower() == ".txt":  # 只按 TXT 统计，避免重复
            coord_repeats[(m["x"], m["y"])].add(m["r"])

    unique_coords = set(coord_repeats.keys())
    repeat_coords = {
        k: sorted(v)
        for k, v in coord_repeats.items()
        if len(v) > 1
    }

    return unique_coords, repeat_coords


def print_dry_run_report(mappings, errors, conflicts, paired, txt_only, bmp_only,
                         unique_coords, repeat_coords):
    """打印 dry-run 报告。"""
    print("\n" + "=" * 70)
    print("  RENAME DRY-RUN REPORT")
    print("=" * 70)

    # ── 映射表 ──
    print(f"\n{'─' * 70}")
    print("  FILE MAPPINGS")
    print(f"{'─' * 70}")

    # 按新文件名排序显示
    sorted_mappings = sorted(mappings, key=lambda m: m["new_name"])
    for m in sorted_mappings:
        print(f"  {m['old_name']:20s} → {m['new_name']:20s}  "
              f"(X={m['x']:2d}, Y={m['y']:2d}, R={m['r']})")

    # ── 错误/歧义 ──
    if errors:
        print(f"\n{'─' * 70}")
        print("  ⚠ ERRORS / AMBIGUITIES")
        print(f"{'─' * 70}")
        for e in errors:
            tag = "AMBIGUOUS" if e["ambiguous"] else "ERROR"
            print(f"  [{tag}] {e['old_name']}: {e['error']}")

    # ── 汇总统计 ──
    print(f"\n{'─' * 70}")
    print("  SUMMARY STATISTICS")
    print(f"{'─' * 70}")

    total_files = len(mappings) + len(errors)
    txt_count = sum(1 for m in mappings if m["ext"].lower() == ".txt")
    bmp_count = sum(1 for m in mappings if m["ext"].lower() == ".bmp")

    print(f"  总文件数:        {total_files}")
    print(f"  成功解码:        {len(mappings)}  (TXT={txt_count}, BMP={bmp_count})")
    print(f"  解码失败/歧义:   {len(errors)}")

    # 冲突检测
    if conflicts:
        print(f"\n  🔴 目标文件名冲突: {len(conflicts)} 个")
        for name, count in conflicts:
            sources = [m["old_name"] for m in mappings if m["new_name"] == name]
            print(f"     {name} ← {sources}")
    else:
        print(f"  目标文件名冲突:  无 ✓")

    # TXT/BMP 配对
    print(f"\n  TXT/BMP 配对:")
    print(f"    已配对:        {len(paired)}")
    if txt_only:
        print(f"    仅 TXT (无BMP): {len(txt_only)}")
        for coord in sorted(txt_only):
            print(f"      X={coord[0]}, Y={coord[1]}, R={coord[2]}")
    if bmp_only:
        print(f"    仅 BMP (无TXT): {len(bmp_only)}")
        for coord in sorted(bmp_only):
            print(f"      X={coord[0]}, Y={coord[1]}, R={coord[2]}")
    if not txt_only and not bmp_only:
        print(f"    未配对:        无 ✓")

    # 唯一坐标统计
    print(f"\n  唯一坐标 (X,Y):  {len(unique_coords)}")

    # Repeat 坐标
    if repeat_coords:
        print(f"  多次重复坐标:    {len(repeat_coords)} 个")
        for (x, y), repeats in sorted(repeat_coords.items()):
            print(f"    (X={x:2d}, Y={y:2d}): R={repeats}")
    else:
        print(f"  多次重复坐标:    无")

    print(f"\n{'=' * 70}")


def execute_rename(mappings: list, data_dir: Path) -> tuple:
    """执行实际重命名。返回 (success_count, fail_count, fail_details)。"""
    success = 0
    fails = []

    for m in mappings:
        old_path = m["path"]
        new_path = data_dir / m["new_name"]

        if new_path.exists() and old_path != new_path:
            fails.append((m["old_name"], m["new_name"], "目标文件已存在"))
            continue

        try:
            old_path.rename(new_path)
            success += 1
        except OSError as e:
            fails.append((m["old_name"], m["new_name"], str(e)))

    return success, len(fails), fails


def write_csv_report(mappings: list, errors: list, report_path: Path) -> None:
    """输出 CSV 映射报告。"""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_name", "new_name", "X", "Y", "R", "ext", "status"])

        for m in sorted(mappings, key=lambda m: m["new_name"]):
            writer.writerow([
                m["old_name"], m["new_name"],
                m["x"], m["y"], m["r"], m["ext"],
                "OK"
            ])

        for e in errors:
            writer.writerow([
                e["old_name"], "",
                "", "", "", "",
                f"{'AMBIGUOUS' if e['ambiguous'] else 'ERROR'}: {e['error']}"
            ])

    print(f"\n✓ CSV 报告已写入: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="重命名 LWIR 热像数据文件: 连写格式 → X_Y_R.ext"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="实际执行重命名（默认为 dry-run）"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="输出 CSV 映射表到 reports/rename_mapping.csv"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help=f"数据目录 (默认: {DATA_DIR})"
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    # ── Step 0: 前置检查 ──
    print("=" * 70)
    print("  rename_data.py — LWIR 热像数据文件重命名工具")
    print("=" * 70)

    verify_agents_md(AGENTS_MD)

    if not data_dir.exists():
        print(f"🔴 数据目录不存在: {data_dir}")
        sys.exit(1)

    if not CONFIG_PATH.exists():
        print(f"🔴 配置文件不存在: {CONFIG_PATH}")
        sys.exit(1)

    # ── Step 1: 加载配置 ──
    valid_coords = load_valid_coordinates(CONFIG_PATH)
    print(f"✓ 合法坐标集合已加载: {sorted(valid_coords)}")

    # ── Step 2: 收集文件 ──
    files = collect_files(data_dir)
    print(f"✓ 找到 {len(files)} 个待处理文件 (.txt/.bmp)")

    # ── Step 3: 构建重命名计划 ──
    mappings, errors = build_rename_plan(files, valid_coords)

    # ── Step 4: 检测冲突和配对 ──
    conflicts = check_conflicts(mappings)
    paired, txt_only, bmp_only = check_txt_bmp_pairs(mappings)
    unique_coords, repeat_coords = compute_coordinate_stats(mappings)

    # ── Step 5: 打印报告 ──
    print_dry_run_report(mappings, errors, conflicts, paired, txt_only, bmp_only,
                         unique_coords, repeat_coords)

    # ── Step 6: 输出 CSV ──
    if args.report:
        csv_path = REPORTS_DIR / "rename_mapping.csv"
        write_csv_report(mappings, errors, csv_path)

    # ── Step 7: 执行重命名 ──
    if args.execute:
        if conflicts:
            print("\n🔴 存在文件名冲突，无法执行重命名！请先解决冲突。")
            sys.exit(1)

        print(f"\n{'─' * 70}")
        print("  EXECUTING RENAME...")
        print(f"{'─' * 70}")

        success, fail_count, fails = execute_rename(mappings, data_dir)

        print(f"\n  ✓ 成功重命名: {success}")
        if fail_count:
            print(f"  🔴 失败: {fail_count}")
            for old, new, reason in fails:
                print(f"     {old} → {new}: {reason}")

        print(f"\n{'=' * 70}")
        print("  RENAME COMPLETE")
        print(f"{'=' * 70}")
    else:
        print("\n💡 这是 dry-run 模式。添加 --execute 参数以实际执行重命名。")


if __name__ == "__main__":
    main()
