#!/usr/bin/env bash
# clean.sh — 自动检测并删除 outputs/ 下最近修改的一轮训练产物
# 用法:
#   ./clean.sh          交互确认后删除
#   ./clean.sh -y       跳过确认直接删除
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"

if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "⚠️  outputs/ 目录不存在，无需清理。"
    exit 0
fi

# 找到最近修改的子目录（按 mtime 降序，取第一个）
LATEST=$(find "$OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -rn | head -1 | cut -d' ' -f2-)

if [[ -z "$LATEST" ]]; then
    echo "⚠️  outputs/ 下没有训练目录，无需清理。"
    exit 0
fi

RUN_NAME=$(basename "$LATEST")
RUN_SIZE=$(du -sh "$LATEST" | cut -f1)

# 统计内容
N_CKPT=$(find "$LATEST" -name "*.pt" -o -name "*.pth" 2>/dev/null | wc -l)
N_TB=$(find "$LATEST" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
HAS_CONFIG=$([[ -f "$LATEST/config.json" ]] && echo "yes" || echo "no")

echo "🔍 检测到最近的训练运行:"
echo "   目录:       $LATEST"
echo "   大小:       $RUN_SIZE"
echo "   Checkpoint:  ${N_CKPT} 个"
echo "   TensorBoard: ${N_TB} 个 event 文件"
echo "   config.json: ${HAS_CONFIG}"
echo ""

if [[ "${1:-}" == "-y" ]]; then
    CONFIRM="y"
else
    read -rp "🗑️  确认删除 [${RUN_NAME}]？(y/N) " CONFIRM
fi

if [[ "$CONFIRM" =~ ^[yY]$ ]]; then
    rm -rf "$LATEST"
    echo "✅ 已删除: ${RUN_NAME}"

    # 如果 outputs/ 空了，也删掉空目录
    if [[ -z "$(ls -A "$OUTPUTS_DIR" 2>/dev/null)" ]]; then
        rmdir "$OUTPUTS_DIR"
        echo "   (outputs/ 已空，一并删除)"
    fi
else
    echo "❌ 取消。"
fi
